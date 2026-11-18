"""Cartesian / joint delta controller shared by the simulator AND the real robot.

The policy outputs normalised actions in [-1, 1].  This class turns them into a
joint-position target for the next control period:

    action -> TCP target (accumulated, never further than ``lead`` from the measured
              TCP) -> damped-least-squares IK step -> joint target q + dq

Two actuation models (``control.actuation``):

* ``position`` (v1-v7): the IK step dq becomes a joint target q + dq for MuJoCo's position servos.
* ``velocity`` (v8+): the controller outputs JOINT SPEEDS, like the real Kortex joint-speed
  interface: feedforward (the target's motion this step) + low-gain feedback (``velocity_kp`` x the
  target lead), limited per joint in speed AND acceleration (``joint_max_accel``, the Gen3 Lite's
  Kortex limits).  The sim integrates the speed into its servo target, as Kortex does internally.
  This is the model the mirror tool validated on the real arm (0.1-0.3 deg from sim, lab
  2026-11-17); the old ``dq / servo_tau`` speed mapping oscillated on joints 2-3 there.
  On the real arm ``joint_speed_command()`` returns the speed to send.

Because sim and real use this one class, the hidden controller state (target lead, target yaw,
gripper command, commanded speed) is identical in both and can be shown to the policy.

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
        # actuation model (see module docstring).  Configs without the key (v1-v7 runs) keep "position".
        self.actuation = cc.get("actuation", "position")
        if self.actuation not in ("position", "velocity"):
            raise ValueError(f"control.actuation must be 'position' or 'velocity', not {self.actuation!r}")
        jma = cc.get("joint_max_accel")
        self.joint_max_accel = None if jma is None else np.broadcast_to(np.asarray(jma, float), (len(arm_qadr),)).copy()
        self.velocity_kp = float(cc.get("velocity_kp", 2.0))          # 1/s, feedback on the target lead
        self.servo_max_lag = float(cc.get("servo_max_lag", 0.1))      # rad, sim servo target vs measured q
        self.v_cmd = np.zeros(len(arm_qadr))                          # joint speed sent this step (rad/s)
        self._unwind = np.zeros(len(arm_qadr))                        # relax(): servo-target correction (rad)
        self.q_cmd = np.zeros(len(arm_qadr))                          # integrated speed = sim servo target
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
        self.q_cmd = self.q_target.copy()
        self.v_cmd = np.zeros(len(self.arm_qadr))
        self._unwind = np.zeros(len(self.arm_qadr))
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
        ``keep_axis`` (world unit vector) keeps the lead along that direction (e.g. the drawer axis;
        with velocity actuation only the pulling sense of the servo push is kept)."""
        cur = self.tcp.pos(data)
        lead = self.tcp_target - cur
        keep = np.zeros(3) if keep_axis is None else keep_axis * float(lead @ keep_axis)
        self.tcp_target = cur + keep
        if self.rot_mode == "yaw":
            self.tcp_yaw = float(np.clip(self.measured_yaw(data), -self.max_yaw, self.max_yaw))
            self.R_target = self.yaw_rotation(self.tcp_yaw) @ self.R_nominal
        if self.actuation == "velocity":
            # The arm's own integrated target (Kortex internally, or the sim servo target) also pushes.  It can
            # only be pulled back through the SPEED command, so the next step adds the speed that unwinds it
            # (acceleration-limited like every command), keeping only the part along keep_axis.
            q = data.qpos[self.arm_qadr]
            if keep_axis is None:
                want = q.copy()
            else:
                J = self.tcp.jacobian(self.m, data, self.arm_dadr)
                d_tcp = J[:3] @ (self.q_cmd - q)
                # keep the PULL only: a push into the drawer (fingertips on the front) is released too
                e = np.concatenate([keep_axis * max(0.0, float(d_tcp @ keep_axis)), np.zeros(3)])
                want = q + dls_step(J, e, self.cc["ik_damping"])
            self._unwind = want - self.q_cmd

    # ------------------------------------------------------------------- step
    def _limit_speed(self, dq: np.ndarray) -> np.ndarray:
        """Per-joint limit, like the real actuators.  (Scaling all joints together was tried first:
        near the wrist singularity the wrist joints ask for several rad/s and that throttled the
        whole arm to 0.05 m/s.)"""
        if self.joint_max_speed is None:
            return dq
        lim = self.joint_max_speed * self.servo_tau
        return np.clip(dq, -lim, lim)

    def joint_speed_command(self, dq: np.ndarray | None = None) -> np.ndarray:
        """Joint speeds (rad/s) to send to a real velocity-controlled arm this step.

        ``velocity`` actuation: the speed the controller computed (``dq`` is ignored); speed- and
        acceleration-limited exactly like in training.  ``position`` actuation (v1-v7 policies):
        ``dq / servo_tau``, which mimics the sim servo but OSCILLATED on the real arm (lab 2026-11-17)."""
        if self.actuation == "velocity":
            return self.v_cmd.copy()
        v = np.asarray(dq, float) / self.servo_tau
        return v if self.joint_max_speed is None else np.clip(v, -self.joint_max_speed, self.joint_max_speed)

    def _velocity_command(self, v: np.ndarray) -> np.ndarray:
        """Speed + acceleration limits, like Kortex applies to joint-speed commands.

        The acceleration limit scales the whole speed CHANGE by one factor, so the hand keeps moving
        in the commanded direction.  (Clipping each joint separately bent the TCP path sideways
        whenever joints 2-3, whose limits are 10-20x lower than the wrist's, saturated.)"""
        if self.joint_max_speed is not None:
            v = np.clip(v, -self.joint_max_speed, self.joint_max_speed)
        if self.joint_max_accel is not None:
            dv = v - self.v_cmd
            room = self.joint_max_accel * self.dt
            over = np.max(np.abs(dv) / room)
            if over > 1.0:
                v = self.v_cmd + dv / over
        return v

    def step(self, a: np.ndarray, data) -> tuple[np.ndarray, float]:
        """Returns (joint position target for this control period, gripper command in ctrl units)."""
        cc = self.cc
        q = data.qpos[self.arm_qadr]
        lo, hi = self.q_limits[:, 0], self.q_limits[:, 1]
        if self.mode == "ee_delta":
            dpos = self.base_R @ (a[:3] * cc["ee_max_step"])
            cur = self.tcp.pos(data)
            tcp_target_old, R_target_old = self.tcp_target.copy(), self.R_target.copy()
            # accumulate the target, but never let it run away from the real TCP (anti wind-up).
            # A large lead made the arm keep moving after the policy said "stop" and tore the
            # handle out of the gripper (see docs/training_issues.md).
            # commanded motion this step after the workspace box (the velocity feedforward); the lead clip
            # below only limits how far the target may run ahead, not how fast the policy asks to move
            ff_step = np.clip(self.tcp_target + dpos, self.ws_low, self.ws_high) - self.tcp_target
            self.tcp_target = np.clip(self.tcp_target + dpos, cur - self.max_lead, cur + self.max_lead)
            self.tcp_target = np.clip(self.tcp_target, self.ws_low, self.ws_high)
            if self.rot_mode == "full":
                self.R_target = rotvec_to_mat(self.base_R @ (a[3:6] * cc["ee_max_rot_step"])) @ self.R_target
            elif self.rot_mode == "yaw":
                # yaw about the base z axis, limited to +/- ee_max_yaw_deg from the nominal orientation
                self.tcp_yaw = float(np.clip(self.tcp_yaw + a[3] * cc["ee_max_rot_step"], -self.max_yaw, self.max_yaw))
                self.R_target = self.yaw_rotation(self.tcp_yaw) @ self.R_nominal
            if self.actuation == "velocity":
                # feedforward = the motion the policy commanded this step; feedback = kp x lead before it
                lin = ff_step / self.dt + self.velocity_kp * (tcp_target_old - cur)
                ang = rotation_error(R_target_old, self.R_target) / self.dt + \
                    self.velocity_kp * rotation_error(self.tcp.mat(data), R_target_old)
                e = np.concatenate([lin, cc["ik_rot_weight"] * ang])
                v = dls_step(self.tcp.jacobian(self.m, data, self.arm_dadr), e, cc["ik_damping"])
                v = v + self._unwind / self.dt                   # pull a pushing servo target back (relax)
                self._unwind = np.zeros(len(q))
                self.v_cmd = self._velocity_command(v)
                # the sim servo target integrates the speed (Kortex does the same internally); it may
                # lead the measured joints by at most servo_max_lag (a blocked arm saturates its torque)
                self.q_cmd = np.clip(self.q_cmd + self.v_cmd * self.dt,
                                     q - self.servo_max_lag, q + self.servo_max_lag)
                self.q_cmd = np.clip(self.q_cmd, lo, hi)          # never wind up past a joint limit
                self.q_target = self.q_cmd.copy()
            else:
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
