"""The simulated Gen3 Lite, exactly as the RL policy was trained on it (same MJCF, actuator
overrides, table, controller time constants), wrapped for side-by-side comparison with the real arm.

Two ways to use it:
* kinematic: ``set_q(q)`` places the arm at measured joint angles (no physics) -> FK, contacts,
  gravity torques.  Used to shadow the real arm and to check trajectories before running them.
* dynamic:   ``command(dq)`` + ``tick()`` -- the same "move each joint by dq over the next period"
  interface the deploy loop uses (drawer_deploy/backends/sim.py), so mismatches found here are the
  mismatches the policy will meet.
"""
from __future__ import annotations

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from drawer_rl.env import DrawerEnv

# Nominal physics: switch off the per-episode randomisation the env normally applies.
NOMINAL = [
    "task.randomization.validate_reachability=false",
    "task.randomization.actuator_kp_scale=[1.0,1.0]",
    "task.randomization.gripper_force_scale=[1.0,1.0]",
    "task.randomization.action_delay_steps=[0,0]",
    "task.randomization.initial_opening=[0.0,0.0]",
    "robot.start_qpos_noise=0.0",
]


def wrap(q):
    return (np.asarray(q, float) + np.pi) % (2 * np.pi) - np.pi


def pose_from_T(R: np.ndarray, p: np.ndarray) -> np.ndarray:
    """[x, y, z (m), theta_x, theta_y, theta_z (deg)] in the Kortex convention (extrinsic XYZ)."""
    return np.r_[p, Rotation.from_matrix(R).as_euler("xyz", degrees=True)]


def T_from_pose(pose) -> tuple[np.ndarray, np.ndarray]:
    pose = np.asarray(pose, float)
    return Rotation.from_euler("xyz", pose[3:6], degrees=True).as_matrix(), pose[:3].copy()


class SimArm:
    def __init__(self, overrides: list[str] | None = None, cabinet: bool = True):
        self.env = e = DrawerEnv(overrides=NOMINAL + list(overrides or []))
        e.reset(seed=0)
        if not cabinet:                    # move the cabinet well out of the way (real tests without it)
            e.set_drawer_offset(np.array([0.0, 5.0, 0.0]), 0.0)
        self.m, self.d = e.model, e.data
        self.dt = e.dt
        self.q_limits = e.q_limits.copy()                       # what the controller may command
        self.jnt_range = self.m.jnt_range[e.arm_jids].copy()    # the joints' own (MJCF) range
        self.ctrl_range = self.m.actuator_ctrlrange[e.arm_act].copy()
        self.torque_limits = e.arm_torque_limit.copy()
        self.servo_tau = e.controller.servo_tau.copy()
        self.joint_max_speed = e.controller.joint_max_speed
        self.q_start = e.q_start.copy()
        self.q_home = np.asarray(e.rc.get("ik_preferred_qpos"), float)
        self.ee_body = self.m.body("end_effector_link").id
        self.tool_body = self.m.body("tool_frame").id
        self.table_body = self.m.body("table").id
        gb = self.m.geom_bodyid
        robot = e.robot_body_geoms | e.finger_l_geoms | e.finger_r_geoms
        self._robot_geoms = robot
        self._table_geoms = {i for i in range(self.m.ngeom) if gb[i] == self.table_body}
        self._env_geoms = e.env_geoms
        self._body_of = gb
        self.set_q(self.q_home)

    # ------------------------------------------------------------------ state
    def set_q(self, q, grip: float | None = None) -> None:
        """Kinematically place the arm (and hold it there if physics runs afterwards)."""
        e, d = self.env, self.d
        d.qpos[e.arm_qadr] = q
        d.qvel[:] = 0
        d.ctrl[e.arm_act] = q
        if grip is not None:
            g = e.grip_closed + np.clip(grip, 0, 1) * (e.grip_open - e.grip_closed)
            d.ctrl[e.grip_act] = g
            d.qpos[e.grip_qadr] = g
        mujoco.mj_forward(self.m, d)

    def q(self) -> np.ndarray:
        return self.d.qpos[self.env.arm_qadr].copy()

    def dq(self) -> np.ndarray:
        return self.d.qvel[self.env.arm_dadr].copy()

    def gripper(self) -> float:
        e = self.env
        return float((self.d.qpos[e.grip_qadr] - e.grip_closed) / (e.grip_open - e.grip_closed))

    def torque(self) -> np.ndarray:
        return self.d.qfrc_actuator[self.env.arm_dadr].copy()

    # --------------------------------------------------------------- dynamics
    def command(self, dq, grip: float | None = None) -> None:
        """Same semantics as drawer_deploy's SimBackend: target = q + dq (clipped to the limits)."""
        e, d = self.env, self.d
        d.ctrl[e.arm_act] = np.clip(self.q() + np.asarray(dq, float), self.q_limits[:, 0], self.q_limits[:, 1])
        if grip is not None:
            d.ctrl[e.grip_act] = e.grip_closed + np.clip(grip, 0, 1) * (e.grip_open - e.grip_closed)

    def command_target(self, q_target, grip: float | None = None) -> None:
        """Absolute joint target (used to emulate a velocity-commanded arm: target += v * dt)."""
        e, d = self.env, self.d
        d.ctrl[e.arm_act] = np.clip(q_target, self.q_limits[:, 0], self.q_limits[:, 1])
        if grip is not None:
            d.ctrl[e.grip_act] = e.grip_closed + np.clip(grip, 0, 1) * (e.grip_open - e.grip_closed)

    def tick(self, dt: float | None = None) -> None:
        n = self.env.n_substeps if dt is None else max(1, int(round(dt / self.m.opt.timestep)))
        for _ in range(n):
            mujoco.mj_step(self.m, self.d)

    def hold(self) -> None:
        self.d.ctrl[self.env.arm_act] = self.q()

    # ------------------------------------------------------------- kinematics
    def _base(self, p):
        return self.env._w2b(p)

    def flange_pose(self) -> np.ndarray:
        """end_effector_link (tool flange, before any tool transform) in the base frame, Kortex convention."""
        e = self.env
        R = e.base_R.T @ self.d.xmat[self.ee_body].reshape(3, 3)
        return pose_from_T(R, self._base(self.d.xpos[self.ee_body]))

    def tool_pose(self, tool_transform=None) -> np.ndarray:
        """Tool pose like Kortex reports it.  ``tool_transform`` = Kortex tool configuration
        [x, y, z, theta_x, theta_y, theta_z] relative to the flange; None = the MJCF tool_frame."""
        e = self.env
        if tool_transform is None:
            R = e.base_R.T @ self.d.xmat[self.tool_body].reshape(3, 3)
            return pose_from_T(R, self._base(self.d.xpos[self.tool_body]))
        Rf, pf = T_from_pose(self.flange_pose())
        Rt, pt = T_from_pose(tool_transform)
        return pose_from_T(Rf @ Rt, pf + Rf @ pt)

    def tcp_pos(self) -> np.ndarray:
        return self._base(self.env.tcp.pos(self.d))

    def fk(self, q, tool_transform=None) -> np.ndarray:
        self.set_q(q)
        return self.tool_pose(tool_transform)

    def body_pos(self, name: str) -> np.ndarray:
        return self._base(self.d.xpos[self.m.body(name).id])

    def joint_anchor(self, name: str) -> np.ndarray:
        return self._base(self.d.xanchor[self.m.joint(name).id])

    def gravity_torque(self, q) -> np.ndarray:
        """Torque each joint needs to hold the arm still at ``q`` (sim masses)."""
        self.set_q(q)
        return self.d.qfrc_bias[self.env.arm_dadr].copy()

    # --------------------------------------------------------------- checks
    def contacts(self) -> dict:
        """Contacts at the current state: arm vs table, arm vs cabinet, arm vs itself."""
        out = {"table": False, "cabinet": False, "self": False, "pairs": []}
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            if c.dist > 0:
                continue
            g1, g2 = c.geom1, c.geom2
            r1, r2 = g1 in self._robot_geoms, g2 in self._robot_geoms
            if not (r1 or r2):
                continue
            other = g2 if r1 else g1
            if r1 and r2:
                out["self"] = True
            elif other in self._table_geoms:
                out["table"] = True
            elif other in self._env_geoms:
                out["cabinet"] = True
            else:
                continue
            out["pairs"].append((self.m.body(self._body_of[g1]).name, self.m.body(self._body_of[g2]).name))
        return out

    def joint_margin_deg(self, q=None) -> np.ndarray:
        q = self.q() if q is None else np.asarray(q, float)
        return np.degrees(np.minimum(q - self.jnt_range[:, 0], self.jnt_range[:, 1] - q))
