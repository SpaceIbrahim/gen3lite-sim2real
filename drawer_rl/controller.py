"""Cartesian / joint delta controller shared by the simulator AND the real robot.

The policy outputs normalised actions in [-1, 1].  This class turns them into a
joint-position target for the next control period:

    action -> TCP target (accumulated, never further than ``lead`` from the measured
              TCP) -> damped-least-squares IK step -> joint target q + dq

In simulation the joint target goes to MuJoCo's position actuators (drawer_rl/env.py).
On the real arm ``dq / dt`` is sent as a joint-speed command (drawer_deploy/).  Because
both use this one class, the hidden controller state (target lead, target yaw, gripper
command) is identical in sim and real, and can be shown to the policy as observations.

``data`` must hold the MEASURED joint positions with kinematics up to date
(mj_kinematics + mj_comPos, or mj_forward / mj_step).
"""
from __future__ import annotations

import math

import numpy as np

from .kinematics import TCP, dls_step, rotation_error, rotvec_to_mat

_ROT_MODES = {False: None, None: None, True: "full", "full": "full", "yaw": "yaw"}


class DeltaController:
    def __init__(self, model, control_cfg: dict, tcp: TCP, arm_qadr, arm_dadr, q_limits: np.ndarray,
                 base_pos: np.ndarray, base_R: np.ndarray, R_nominal: np.ndarray, dt: float,
                 grip_open: float, grip_closed: float, servo_tau: np.ndarray | None = None):
        self.m, self.cc, self.tcp = model, control_cfg, tcp
        self.arm_qadr, self.arm_dadr, self.q_limits = arm_qadr, arm_dadr, q_limits
        self.base_pos, self.base_R, self.R_nominal, self.dt = base_pos, base_R, R_nominal, dt
        self.grip_open, self.grip_closed = grip_open, grip_closed
        cc = control_cfg
        self.mode = cc["mode"]
        if self.mode == "ee_delta":
            self.rot_mode = _ROT_MODES[cc["ee_rotation_actions"]]
            self.n_act = 3 + {None: 0, "yaw": 1, "full": 3}[self.rot_mode] + 1
        elif self.mode == "joint_delta":
            self.rot_mode = None
            self.n_act = len(arm_qadr) + 1
        else:
            raise ValueError(f"unknown control mode {self.mode}")
        self.ws_low = base_pos + base_R @ np.asarray(cc["workspace"]["low"], float)
        self.ws_high = base_pos + base_R @ np.asarray(cc["workspace"]["high"], float)
        self.max_lead = cc.get("ee_target_lead_steps", 1.5) * cc["ee_max_step"]
        self.max_yaw = math.radians(cc.get("ee_max_yaw_deg", 30.0))
        # Joint speed limit (rad/s, scalar or one per joint); None = unlimited (older configs).
        # A joint step dq is followed at a speed of about dq / servo_tau: in simulation servo_tau is
        # the position servo's time constant (kv / kp); on the real arm the deploy code sends
        # dq / servo_tau as the joint-speed command.  So |dq| <= limit * servo_tau caps both.
        self.servo_tau = np.full(len(arm_qadr), dt) if servo_tau is None else np.asarray(servo_tau, float)
        jms = cc.get("joint_max_speed")
        self.joint_max_speed = None if jms is None else np.broadcast_to(np.asarray(jms, float), (len(arm_qadr),)).copy()
        self.tcp_target = np.zeros(3)
        self.R_target = R_nominal.copy()
        self.tcp_yaw = 0.0
        self.q_target = np.zeros(len(arm_qadr))
        self.grip_cmd = grip_open

    # ------------------------------------------------------------------ state
    def reset(self, data, grip_ctrl: float | None = None) -> None:
        self.tcp_target = self.tcp.pos(data).copy()
        self.R_target = self.R_nominal.copy()
        self.tcp_yaw = 0.0
        self.q_target = data.qpos[self.arm_qadr].copy()
        if grip_ctrl is not None:
            self.grip_cmd = float(grip_ctrl)

    def yaw_rotation(self, yaw: float) -> np.ndarray:
        """World-frame rotation by ``yaw`` (rad) about the robot base z axis."""
        c, s = math.cos(yaw), math.sin(yaw)
        return self.base_R @ np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]) @ self.base_R.T

    def lead(self, data) -> np.ndarray:
        """How far the TCP target is ahead of the measured TCP (robot base frame, m).
        A persistent lead while the hand is blocked means the arm is PUSHING."""
        return self.base_R.T @ (self.tcp_target - self.tcp.pos(data))

    def measured_yaw(self, data) -> float:
        """Yaw of the measured TCP relative to the nominal orientation (rad)."""
        R_rel = self.base_R.T @ self.tcp.mat(data) @ self.R_nominal.T @ self.base_R
        return float(math.atan2(R_rel[1, 0], R_rel[0, 0]))

    def relax(self, data, keep_axis: np.ndarray | None = None) -> None:
        """Safety reflex: pull the target back onto the measured hand so the arm stops pushing.
        ``keep_axis`` (world unit vector) keeps the lead along that direction (e.g. the drawer axis)."""
        cur = self.tcp.pos(data)
        lead = self.tcp_target - cur
        keep = np.zeros(3) if keep_axis is None else keep_axis * float(lead @ keep_axis)
        self.tcp_target = cur + keep
        if self.rot_mode == "yaw":
            self.tcp_yaw = float(np.clip(self.measured_yaw(data), -self.max_yaw, self.max_yaw))
            self.R_target = self.yaw_rotation(self.tcp_yaw) @ self.R_nominal

    # ------------------------------------------------------------------- step
    def _limit_speed(self, dq: np.ndarray) -> np.ndarray:
        """Per-joint limit, like the real actuators.  (Scaling all joints together was tried first:
        near the wrist singularity the wrist joints ask for several rad/s and that throttled the
        whole arm to 0.05 m/s.)"""
        if self.joint_max_speed is None:
            return dq
        lim = self.joint_max_speed * self.servo_tau
        return np.clip(dq, -lim, lim)

    def joint_speed_command(self, dq: np.ndarray) -> np.ndarray:
        """Joint speeds (rad/s) that make a real velocity-controlled arm follow ``dq`` the way the
        simulated position servo does."""
        v = np.asarray(dq, float) / self.servo_tau
        return v if self.joint_max_speed is None else np.clip(v, -self.joint_max_speed, self.joint_max_speed)

    def step(self, a: np.ndarray, data) -> tuple[np.ndarray, float]:
        """Returns (joint position target for this control period, gripper command in ctrl units)."""
        cc = self.cc
        q = data.qpos[self.arm_qadr]
        lo, hi = self.q_limits[:, 0], self.q_limits[:, 1]
        if self.mode == "ee_delta":
            dpos = self.base_R @ (a[:3] * cc["ee_max_step"])
            cur = self.tcp.pos(data)
            # accumulate the target, but never let it run away from the real TCP (anti wind-up).
            # A large lead made the arm keep moving after the policy said "stop" and tore the
            # handle out of the gripper (see docs/training_issues.md).
            self.tcp_target = np.clip(self.tcp_target + dpos, cur - self.max_lead, cur + self.max_lead)
            self.tcp_target = np.clip(self.tcp_target, self.ws_low, self.ws_high)
            if self.rot_mode == "full":
                self.R_target = rotvec_to_mat(self.base_R @ (a[3:6] * cc["ee_max_rot_step"])) @ self.R_target
            elif self.rot_mode == "yaw":
                # yaw about the base z axis, limited to +/- ee_max_yaw_deg from the nominal orientation
                self.tcp_yaw = float(np.clip(self.tcp_yaw + a[3] * cc["ee_max_rot_step"], -self.max_yaw, self.max_yaw))
                self.R_target = self.yaw_rotation(self.tcp_yaw) @ self.R_nominal
            e = np.concatenate([self.tcp_target - cur,
                                cc["ik_rot_weight"] * rotation_error(self.tcp.mat(data), self.R_target)])
            dq = dls_step(self.tcp.jacobian(self.m, data, self.arm_dadr), e, cc["ik_damping"])
            self.q_target = np.clip(q + self._limit_speed(dq), lo, hi)
        else:
            self.q_target = np.clip(self.q_target + a[:len(q)] * cc["joint_max_step"], lo, hi)

        ga = float(a[-1])
        if cc["gripper_mode"] == "binary":
            ga = 1.0 if ga > 0 else -1.0
        target = self.grip_closed + (ga + 1) / 2 * (self.grip_open - self.grip_closed)
        # rate-limit like the real gripper (an instant close made the driven finger hit the bar
        # first and knock the hand off it -- see docs/training_issues.md)
        speed = cc.get("gripper_max_speed", np.inf)        # fraction of the full stroke per second
        if np.isfinite(speed):
            dmax = speed * abs(self.grip_open - self.grip_closed) * self.dt
            target = float(np.clip(target, self.grip_cmd - dmax, self.grip_cmd + dmax))
        self.grip_cmd = target
        return self.q_target, self.grip_cmd

    def gripper_open_fraction(self) -> float:
        """Current gripper command as 0 (closed) .. 1 (open) -- what the real gripper is sent."""
        return float((self.grip_cmd - self.grip_closed) / (self.grip_open - self.grip_closed))
