"""The real Gen3 Lite through the Kinova Kortex API, plus a simulated stand-in for dry runs.

Both classes have the same interface, so every mirror mode can be rehearsed with ``--real fake``
before the real arm is involved.  Joint angles are in rad, wrapped to [-pi, pi] (MJCF convention);
Kortex itself uses degrees, 0..360.

Kortex API: install the wheel from https://github.com/Kinovarobotics/kortex/releases
(kortex_api-2.x). It needs protobuf 3.20, which conflicts with the training venv, so use a
separate venv for the real arm (see sim2real/README.md).
"""
from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass

import numpy as np

from .sim_arm import SimArm, wrap


@dataclass
class RealState:
    t: float                     # s, host clock
    q: np.ndarray                # rad (6)
    dq: np.ndarray               # rad/s
    torque: np.ndarray           # N*m (Gen3 Lite: estimated from motor current)
    current: np.ndarray          # A
    gripper: float               # 0 closed .. 1 open
    tool_pose: np.ndarray        # [x, y, z (m), theta_x, theta_y, theta_z (deg)] as Kortex reports
    fault: str = ""              # non-empty if the arm is in a fault / not ready state


# =============================================================================== real arm
class KortexArm:
    name = "real"

    def __init__(self, ip="192.168.1.10", port=10000, user_env="KINOVA_USERNAME", pass_env="KINOVA_PASSWORD"):
        self.ip, self.port, self.user_env, self.pass_env = ip, port, user_env, pass_env
        self._transport = self._session = None

    # ------------------------------------------------------------ connection
    def connect(self):
        try:
            from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
            from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
            from kortex_api.autogen.client_stubs.ControlConfigClientRpc import ControlConfigClient
            from kortex_api.autogen.messages import Base_pb2, Session_pb2
            from kortex_api.RouterClient import RouterClient
            from kortex_api.SessionManager import SessionManager
            from kortex_api.TCPTransport import TCPTransport
        except ImportError as e:
            raise ImportError("kortex_api is not installed in this Python (see sim2real/README.md); "
                              "use --real fake for a dry run") from e
        self.Base_pb2 = Base_pb2
        user, pwd = os.environ.get(self.user_env), os.environ.get(self.pass_env)
        if not user or not pwd:
            raise RuntimeError(f"set ${self.user_env} and ${self.pass_env} to the arm's login first")
        self._transport = TCPTransport()
        self._router = RouterClient(self._transport, RouterClient.basicErrorCallback)
        self._transport.connect(self.ip, int(self.port))
        info = Session_pb2.CreateSessionInfo()
        info.username, info.password = user, pwd
        info.session_inactivity_timeout = 60000
        info.connection_inactivity_timeout = 2000
        self._session = SessionManager(self._router)
        self._session.CreateSession(info)
        self.base = BaseClient(self._router)
        self.cyclic = BaseCyclicClient(self._router)
        self.config = ControlConfigClient(self._router)
        mode = Base_pb2.ServoingModeInformation()
        mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
        self.base.SetServoingMode(mode)
        self.n_joints = self.base.GetActuatorCount().count
        if self.n_joints != 6:
            raise RuntimeError(f"expected a 6-joint Gen3 Lite, the arm reports {self.n_joints} actuators")

    def close(self):
        if self._session is None:
            return
        try:
            self.stop()
        finally:
            try:
                self._session.CloseSession()
            finally:
                self._transport.disconnect()
                self._session = None

    # ----------------------------------------------------------------- read
    def read(self) -> RealState:
        fb = self.cyclic.RefreshFeedback()
        act = fb.actuators[:6]
        q = wrap(np.radians([a.position for a in act]))
        dq = np.radians([a.velocity for a in act])
        b = fb.base
        try:
            grip = 1.0 - fb.interconnect.gripper_feedback.motor[0].position / 100.0
        except (AttributeError, IndexError):
            grip = np.nan
        fault = ""
        if getattr(b, "fault_bank_a", 0) or getattr(b, "fault_bank_b", 0):
            fault = f"base fault banks a={b.fault_bank_a} b={b.fault_bank_b}"
        return RealState(time.time(), q, dq, np.array([a.torque for a in act]),
                         np.array([a.current_motor for a in act]), float(grip),
                         np.array([b.tool_pose_x, b.tool_pose_y, b.tool_pose_z,
                                   b.tool_pose_theta_x, b.tool_pose_theta_y, b.tool_pose_theta_z]), fault)

    def info(self) -> dict:
        """Limits and tool configuration as the arm itself reports them."""
        out = {}
        try:
            hl = self.config.GetKinematicHardLimits()
            out["hard_limits"] = {"twist_linear_m_s": hl.twist_linear, "twist_angular_deg_s": hl.twist_angular,
                                  "joint_speed_deg_s": list(hl.joint_speed_limits),
                                  "joint_accel_deg_s2": list(hl.joint_acceleration_limits)}
        except Exception as e:  # noqa: BLE001 - report whatever the firmware supports
            out["hard_limits"] = f"unavailable: {e}"
        try:
            sl = self.config.GetAllKinematicSoftLimits()
            out["soft_limits"] = [{"control_mode": int(k.control_mode), "twist_linear_m_s": k.twist_linear,
                                   "twist_angular_deg_s": k.twist_angular,
                                   "joint_speed_deg_s": list(k.joint_speed_limits),
                                   "joint_accel_deg_s2": list(k.joint_acceleration_limits)}
                                  for k in sl.kinematic_limits_list]
        except Exception as e:  # noqa: BLE001
            out["soft_limits"] = f"unavailable: {e}"
        try:
            tc = self.config.GetToolConfiguration()
            t = tc.tool_transform
            out["tool_transform"] = [t.x, t.y, t.z, t.theta_x, t.theta_y, t.theta_z]
            out["tool_mass_kg"] = tc.tool_mass
            out["tool_mass_center"] = [tc.tool_mass_center.x, tc.tool_mass_center.y, tc.tool_mass_center.z]
        except Exception as e:  # noqa: BLE001
            out["tool_transform"] = f"unavailable: {e}"
        try:
            out["arm_state"] = self.Base_pb2.ArmState.Name(self.base.GetArmState().active_state)
        except Exception as e:  # noqa: BLE001
            out["arm_state"] = f"unavailable: {e}"
        return out

    def fk(self, q) -> np.ndarray:
        """The arm's OWN forward kinematics (no motion) -> tool pose, Kortex convention."""
        ja = self.Base_pb2.JointAngles()
        for i, a in enumerate(np.degrees(q) % 360):
            j = ja.joint_angles.add()
            j.joint_identifier, j.value = i, float(a)
        p = self.base.ComputeForwardKinematics(ja)
        return np.array([p.x, p.y, p.z, p.theta_x, p.theta_y, p.theta_z])

    # -------------------------------------------------------------- command
    def send_joint_speeds(self, v_rad_s) -> None:
        js = self.Base_pb2.JointSpeeds()
        for i, v in enumerate(np.degrees(v_rad_s)):
            s = js.joint_speeds.add()
            s.joint_identifier, s.value, s.duration = i, float(v), 0
        self.base.SendJointSpeedsCommand(js)

    def send_gripper(self, g_open: float) -> None:
        """Position command, 0 closed .. 1 open (Kortex: 0 open .. 1 closed)."""
        gc = self.Base_pb2.GripperCommand()
        gc.mode = self.Base_pb2.GRIPPER_POSITION
        f = gc.gripper.finger.add()
        f.finger_identifier, f.value = 1, float(np.clip(1.0 - g_open, 0, 1))
        self.base.SendGripperCommand(gc)

    def move_to(self, q, timeout=30.0) -> bool:
        """Kortex's own joint-space move (its planner and speed limits).  Blocks until done."""
        B = self.Base_pb2
        action = B.Action()
        action.name, action.application_data = "sim2real_move_to", ""
        for i, a in enumerate(np.degrees(q) % 360):
            j = action.reach_joint_angles.joint_angles.joint_angles.add()
            j.joint_identifier, j.value = i, float(a)
        done = threading.Event()
        result = {}

        def cb(n):
            if n.action_event in (B.ACTION_END, B.ACTION_ABORT):
                result["ok"] = n.action_event == B.ACTION_END
                done.set()
        handle = self.base.OnNotificationActionTopic(cb, B.NotificationOptions())
        try:
            self.base.ExecuteAction(action)
            finished = done.wait(timeout)
        finally:
            self.base.Unsubscribe(handle)
        if not finished:
            self.stop()
        return finished and result.get("ok", False)

    def stop(self) -> None:
        try:
            self.send_joint_speeds(np.zeros(6))
        finally:
            self.base.Stop()

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *exc):
        self.close()


# =========================================================================== stand-in arm
class FakeArm:
    """A simulated "real" arm with deliberate differences, to rehearse every mode without hardware.

    It behaves like a velocity-controlled arm (it receives joint SPEEDS, like Kortex) and can be
    given latency, a speed gain error, encoder offsets and a different servo stiffness, so the
    mirror report has something to find.
    """
    name = "fake"

    def __init__(self, latency_steps=1, speed_gain=0.9, q_offset_deg=(0, 0, 0, 0, 0, 0), kp_scale=0.8,
                 tool_transform=(0.0, 0.0, 0.13, 0.0, 0.0, 90.0), start_q=None, overrides=None):
        self.sim = SimArm(overrides=overrides, cabinet=False)
        m, e = self.sim.m, self.sim.env
        m.actuator_gainprm[e.arm_act, 0] *= kp_scale
        m.actuator_biasprm[e.arm_act, 1] *= kp_scale
        self.latency, self.gain = int(latency_steps), float(speed_gain)
        self.q_off = np.radians(q_offset_deg)
        self.tool_transform = list(tool_transform)
        self._queue, self._v, self._g, self._tgt = [], np.zeros(6), 1.0, None
        self.sim.set_q(self.sim.q_home if start_q is None else start_q, grip=1.0)
        self._t0 = time.time()

    def connect(self):
        pass

    def close(self):
        self.stop()

    def read(self) -> RealState:
        s = self.sim
        return RealState(time.time(), wrap(s.q() + self.q_off), s.dq(), s.torque(), np.zeros(6), s.gripper(),
                         s.tool_pose(self.tool_transform))

    def info(self) -> dict:
        return {"hard_limits": {"joint_speed_deg_s": [57.3] * 6, "twist_linear_m_s": 0.25},
                "tool_transform": self.tool_transform, "arm_state": "FAKE"}

    def fk(self, q) -> np.ndarray:
        q_save = self.sim.q()
        p = self.sim.fk(np.asarray(q) - self.q_off, self.tool_transform)
        self.sim.set_q(q_save)
        return p

    def send_joint_speeds(self, v) -> None:
        self._queue.append(np.asarray(v, float) * self.gain)
        if len(self._queue) > self.latency:
            self._v = self._queue.pop(0)

    def send_gripper(self, g_open) -> None:
        self._g = float(g_open)

    def advance(self, dt) -> None:
        """Called by the mirror loop once per period (the real arm advances by itself).
        Like Kortex: the commanded joint speed is integrated into an internal position target."""
        s = self.sim
        if self._tgt is None:
            self._tgt = s.q()
        self._tgt = np.clip(self._tgt + self._v * dt, s.q_limits[:, 0], s.q_limits[:, 1])
        s.command_target(self._tgt, None)
        e = s.env
        cur = (s.d.ctrl[e.grip_act] - e.grip_closed) / (e.grip_open - e.grip_closed)
        step = e.tc["control"].get("gripper_max_speed", 1.5) * dt
        g = cur + np.clip(self._g - cur, -step, step)
        s.d.ctrl[e.grip_act] = e.grip_closed + g * (e.grip_open - e.grip_closed)
        s.tick(dt)

    def move_to(self, q, timeout=30.0) -> bool:
        s = self.sim
        q = np.asarray(q) - self.q_off
        for _ in range(int(timeout / s.dt)):
            err = q - s.q()
            if np.abs(err).max() < np.radians(0.2):
                break
            s.command(np.clip(err, -0.03, 0.03))
            s.tick()
        self.stop()
        return True

    def stop(self) -> None:
        self._queue, self._v, self._tgt = [], np.zeros(6), None
        self.sim.hold()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
