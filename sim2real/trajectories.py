"""Test motions as joint-space reference trajectories q_ref(t), sampled at the control period.

Every motion starts and ends at rest at its start pose.  Cartesian motions are turned into joint
references with the simulator's IK, so both arms get exactly the same joint reference.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from drawer_rl.kinematics import solve_ik

from .sim_arm import SimArm


@dataclass
class Motion:
    name: str
    q0: np.ndarray                         # start pose (rad)
    t: np.ndarray                          # s
    q: np.ndarray                          # (n, 6) rad
    grip: np.ndarray                       # (n,) 0 closed .. 1 open
    segments: list = field(default_factory=list)   # (t_start, label) for the report

    @property
    def duration(self) -> float:
        return float(self.t[-1])


def _cos_profile(n: int) -> np.ndarray:
    """0 -> 1 with zero velocity at both ends."""
    return 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, n))


def _move(q_a, q_b, peak_speed, dt, min_time=0.6):
    """Smooth joint-space move; the slowest joint sets the time (peak speed = pi/2 * mean speed)."""
    dist = np.abs(np.asarray(q_b) - np.asarray(q_a)).max()
    T = max(min_time, np.pi / 2 * dist / max(peak_speed, 1e-6))
    s = _cos_profile(max(2, int(round(T / dt)) + 1))
    return np.asarray(q_a) + s[:, None] * (np.asarray(q_b) - np.asarray(q_a))


def _hold(q, seconds, dt):
    return np.repeat(np.asarray(q, float)[None], max(1, int(round(seconds / dt))), axis=0)


class Builder:
    def __init__(self, q0, dt, grip=1.0):
        self.q0, self.dt = np.asarray(q0, float), dt
        self.qs, self.gs, self.segs = [self.q0[None]], [np.array([grip])], []
        self.grip = grip

    @property
    def n(self):
        return sum(len(a) for a in self.qs)

    @property
    def last(self):
        return self.qs[-1][-1]

    def label(self, text):
        self.segs.append((self.n * self.dt, text))

    def move(self, q, speed):
        seg = _move(self.last, q, speed, self.dt)[1:]
        self.qs.append(seg)
        self.gs.append(np.full(len(seg), self.grip))

    def hold(self, seconds):
        seg = _hold(self.last, seconds, self.dt)
        self.qs.append(seg)
        self.gs.append(np.full(len(seg), self.grip))

    def gripper(self, g, seconds):
        self.grip = g
        seg = _hold(self.last, seconds, self.dt)
        self.qs.append(seg)
        self.gs.append(np.full(len(seg), g))

    def path(self, qs):
        qs = np.asarray(qs, float)
        self.qs.append(qs)
        self.gs.append(np.full(len(qs), self.grip))

    def build(self, name) -> Motion:
        q = np.vstack(self.qs)
        return Motion(name, self.q0, np.arange(len(q)) * self.dt, q, np.concatenate(self.gs), self.segs)


# ---------------------------------------------------------------------------------- motions
def joint_sweep(arm: SimArm, q0, dt, amp_deg=20.0, speed_deg_s=20.0, joints=range(6)) -> Motion:
    """Each joint on its own: q0 -> +amp -> -amp -> q0.  Reveals per-joint gain, lag, backlash, sign."""
    b, v, a = Builder(q0, dt), np.radians(speed_deg_s), np.radians(amp_deg)
    b.hold(0.5)
    for j in joints:
        b.label(f"joint_{j + 1} sweep")
        for s in (+1, -1, 0):
            q = np.asarray(q0, float).copy()
            q[j] += s * a
            b.move(q, v)
            b.hold(0.4)
    return b.build("joint_sweep")


def steps(arm: SimArm, q0, dt, step_deg=5.0, joints=range(6)) -> Motion:
    """Small fast steps per joint (the reference jumps; each arm's controller limits the speed).
    Reveals latency, rise time, overshoot and steady-state error."""
    b = Builder(q0, dt)
    b.hold(0.5)
    for j in joints:
        b.label(f"joint_{j + 1} step")
        for s in (+1, 0, -1, 0):
            q = np.asarray(q0, float).copy()
            q[j] += s * np.radians(step_deg)
            b.path(_hold(q, 1.2, dt))
    return b.build("steps")


def _ik_path(arm: SimArm, q_seed, p_from, p_to, R, dt, speed):
    """Straight TCP line at constant orientation -> joint path (cosine speed profile)."""
    e, m, d = arm.env, arm.m, arm.d
    dist = np.linalg.norm(np.asarray(p_to) - np.asarray(p_from))
    T = max(0.6, np.pi / 2 * dist / speed)
    s = _cos_profile(max(2, int(round(T / dt)) + 1))[1:]
    out, q = [], np.asarray(q_seed, float).copy()
    for si in s:
        p_w = e.base_pos + e.base_R @ (np.asarray(p_from) + si * (np.asarray(p_to) - np.asarray(p_from)))
        q, err = solve_ik(m, d, e.tcp, e.arm_qadr, e.arm_dadr, p_w, R, arm.q_limits, q_seed=q, iters=60,
                          rot_weight=0.5)
        if err > 2e-3:
            raise ValueError(f"IK failed along the Cartesian path (error {1000 * err:.1f} mm) -- "
                             "use a smaller --amp or another start pose")
        out.append(q.copy())
    return np.array(out)


def cartesian(arm: SimArm, q0, dt, amp_m=0.08, speed_m_s=0.08, axes="xyz") -> Motion:
    """TCP straight lines +-amp along base x, y, z with the gripper orientation held.
    Reveals whether the real arm follows the SIM's kinematics along the paths the policy uses."""
    arm.set_q(q0)
    p0, R = arm.tcp_pos(), arm.env.tcp.mat(arm.d).copy()
    b = Builder(q0, dt)
    b.hold(0.5)
    for ax in axes:
        k = "xyz".index(ax)
        b.label(f"TCP {ax} line")
        p = p0.copy()
        for s in (+1, -1, 0):
            target = p0.copy()
            target[k] += s * amp_m
            b.path(_ik_path(arm, b.last, p, target, R, dt, speed_m_s))
            p = target
            b.hold(0.4)
    return b.build("cartesian")


def pull(arm: SimArm, q0, dt, reach_m=0.12, pull_m=0.15, speed_m_s=0.10) -> Motion:
    """The drawer motion without a drawer: forward along +x (approach), close the gripper, back along
    -x (pull), open.  Start it from the RL start pose."""
    arm.set_q(q0)
    p0, R = arm.tcp_pos(), arm.env.tcp.mat(arm.d).copy()
    b = Builder(q0, dt)
    b.hold(0.5)
    b.label("approach (+x)")
    p1 = p0 + [reach_m, 0, 0]
    b.path(_ik_path(arm, b.last, p0, p1, R, dt, speed_m_s))
    b.label("close gripper")
    b.gripper(0.0, 1.2)
    b.label("pull (-x)")
    p2 = p1 - [pull_m, 0, 0]
    b.path(_ik_path(arm, b.last, p1, p2, R, dt, speed_m_s))
    b.label("open gripper")
    b.gripper(1.0, 1.2)
    b.label("return")
    b.path(_ik_path(arm, b.last, p2, p0, R, dt, speed_m_s))
    b.hold(0.5)
    return b.build("pull")


def gripper_cycles(arm: SimArm, q0, dt, cycles=2) -> Motion:
    """Arm still; gripper closes and opens.  Reveals close/open time and the end positions."""
    b = Builder(q0, dt)
    b.hold(0.5)
    for i in range(cycles):
        b.label(f"close {i + 1}")
        b.gripper(0.0, 2.0)
        b.label(f"open {i + 1}")
        b.gripper(1.0, 2.0)
    return b.build("gripper")


MOTIONS = {"joint_sweep": joint_sweep, "steps": steps, "cartesian": cartesian, "pull": pull,
           "gripper": gripper_cycles}


def check_motion(arm: SimArm, mo: Motion, margin_deg: float, min_tcp_z: float, every: int = 2) -> list[str]:
    """Kinematic pre-check in the simulator (no physics): joint margins, contacts, TCP height."""
    problems = []
    for i in range(0, len(mo.q), every):
        arm.set_q(mo.q[i], grip=float(mo.grip[i]))
        mujoco.mj_collision(arm.m, arm.d)
        t = mo.t[i]
        mg = arm.joint_margin_deg()
        if mg.min() < margin_deg:
            problems.append(f"t={t:.2f}s joint_{int(mg.argmin()) + 1} only {mg.min():.1f} deg from its limit")
        c = arm.contacts()
        if c["table"] or c["self"]:
            problems.append(f"t={t:.2f}s sim contact {c['pairs'][:2]}")
        if arm.tcp_pos()[2] < min_tcp_z:
            problems.append(f"t={t:.2f}s TCP {1000 * arm.tcp_pos()[2]:.0f} mm above the table (< {1000 * min_tcp_z:.0f})")
        if len(problems) > 8:
            break
    v = np.abs(np.diff(mo.q, axis=0)).max(axis=0) / (mo.t[1] - mo.t[0])
    if mo.name != "steps" and arm.joint_max_speed is not None and np.any(v > arm.joint_max_speed):
        problems.append(f"reference joint speed {np.degrees(v.max()):.0f} deg/s exceeds the sim joint limit")
    return problems
