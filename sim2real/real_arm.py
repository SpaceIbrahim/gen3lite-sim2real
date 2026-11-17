"""The real Gen3 Lite through the Kinova Kortex API, plus a simulated stand-in for dry runs.

Both classes have the same interface, so every mirror mode can be rehearsed with ``--real fake``
before the real arm is involved.  Joint angles are in rad, wrapped to [-pi, pi] (MJCF convention);
Kortex reports degrees 0..360 and accepts -180..180 in actions.

Kortex API: install the wheel from Kinova (see README), with protobuf 3.20.
"""
from __future__ import annotations

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
TIMEOUT_DURATION = 20
NOT_IN_CONTROL = (
    "The arm refused the command: another connection is in control of it (Kortex SESSION_NOT_IN_CONTROL).\n"
    "  * close the Kinova Web App tab, or at least leave its control / jog page;\n"
    "  * stop any ROS driver, demo.py or other script connected to the arm;\n"
    "  * let go of the Xbox controller;\n"
    "then run the command again.  Read-only commands (info, fk-check, follow) work without control.")


class ArmNotInControl(RuntimeError):
    pass


def _kortex(fn, *args):
    """Call a Kortex RPC; turn 'session not in control' into a clear ArmNotInControl error."""
    try:
        return fn(*args)
    except Exception as e:  # noqa: BLE001 - KServerException lives in kortex_api, imported lazily
        if "SESSION_NOT_IN_CONTROL" in str(e) or "not in control" in str(e).lower():
            raise ArmNotInControl(NOT_IN_CONTROL) from None
        raise          # s, maximum wait for a Kortex action (as in the lab demo.py)


def check_for_end_or_abort(e, Base_pb2):
    """Closure that sets ``e`` on an END or ABORT action notification (lab demo.py)."""
    def check(notification, e=e):
        print("EVENT : " + Base_pb2.ActionEvent.Name(notification.action_event))
        if notification.action_event in (Base_pb2.ACTION_END, Base_pb2.ACTION_ABORT):
            e.set()
    return check


class KortexArm:
    """The real arm, written the same way as the lab's kinova/manual/demo.py: a DeviceConnection
    router, BaseClient + BaseCyclicClient, joint-angle actions with END/ABORT notifications,
    GRIPPER_POSITION commands (0 open .. 1 closed) and GetMeasuredGripperMovement.

    Added for the mirror: SendJointSpeedsCommand streaming, ComputeForwardKinematics, Stop.
    """
    name = "real"

    def __init__(self, ip="192.168.1.10", username="admin", password="admin"):
        self.ip, self.username, self.password = ip, username, password
        self._conn = None

    # ------------------------------------------------------------ connection
    def connect(self):
        try:
            from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
            from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
            from kortex_api.autogen.client_stubs.ControlConfigClientRpc import ControlConfigClient
            from kortex_api.autogen.messages import Base_pb2

            from .kortex_utilities import DeviceConnection
        except ImportError as e:
            raise ImportError("kortex_api is not installed in this Python (see README); "
                              "use --real fake for a dry run") from e
        self.Base_pb2 = Base_pb2
        self._conn = DeviceConnection.createTcpConnection(self.ip, self.username, self.password)
        router = self._conn.__enter__()
        try:
            self.base = BaseClient(router)
            self.base_cyclic = BaseCyclicClient(router)
            self.config = ControlConfigClient(router)
            # Make sure the arm is in Single Level Servoing mode
            base_servo_mode = Base_pb2.ServoingModeInformation()
            base_servo_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
            self.base.SetServoingMode(base_servo_mode)
            n = self.base.GetActuatorCount().count
            if n != 6:
                raise RuntimeError(f"expected a 6-joint Gen3 Lite, the arm reports {n} actuators")
        except BaseException:
            self._conn.__exit__(None, None, None)
            self._conn = None
            raise

    def close(self):
        if self._conn is None:
            return
        try:
            self.stop(quiet=True)
        finally:
            self._conn.__exit__(None, None, None)
            self._conn = None

    # ----------------------------------------------------------------- read
    def get_gripper(self) -> float:
        """Kortex gripper position, 0 open .. 1 closed (None if no gripper answers)."""
        gripper_request = self.Base_pb2.GripperRequest()
        gripper_request.mode = self.Base_pb2.GRIPPER_POSITION
        gripper_measure = self.base.GetMeasuredGripperMovement(gripper_request)
        if len(gripper_measure.finger):
            return gripper_measure.finger[0].value
        return None

    def read(self, gripper: bool = True) -> RealState:
        """``gripper=False`` skips the extra gripper RPC (the mirror polls it only every few ticks)."""
        feedback = self.base_cyclic.RefreshFeedback()
        act = feedback.actuators[:6]
        q = wrap(np.radians([j.position for j in act]))           # Kortex: degrees, 0..360
        dq = np.radians([j.velocity for j in act])
        b = feedback.base
        g = self.get_gripper() if gripper else None
        fault = ""
        if getattr(b, "fault_bank_a", 0) or getattr(b, "fault_bank_b", 0):
            fault = f"base fault banks a={b.fault_bank_a} b={b.fault_bank_b}"
        return RealState(time.time(), q, dq, np.array([j.torque for j in act]),
                         np.array([j.current_motor for j in act]), np.nan if g is None else 1.0 - g,
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
        except Exception as e:  # noqa: BLE001
            out["tool_transform"] = f"unavailable: {e}"
        try:
            out["arm_state"] = self.Base_pb2.ArmState.Name(self.base.GetArmState().active_state)
        except Exception as e:  # noqa: BLE001
            out["arm_state"] = f"unavailable: {e}"
        try:
            out["home_action_deg"] = self.home_angles_deg()
        except Exception as e:  # noqa: BLE001
            out["home_action_deg"] = f"unavailable: {e}"
        return out

    def fk(self, q) -> np.ndarray:
        """The arm's OWN forward kinematics (no motion) -> tool pose, Kortex convention."""
        ja = self.Base_pb2.JointAngles()
        for joint_id, a in enumerate(np.degrees(wrap(q))):
            j = ja.joint_angles.add()
            j.joint_identifier, j.value = joint_id, float(a)
        p = self.base.ComputeForwardKinematics(ja)
        return np.array([p.x, p.y, p.z, p.theta_x, p.theta_y, p.theta_z])

    # -------------------------------------------------------------- command
    def send_joint_speeds(self, v_rad_s) -> None:
        joint_speeds = self.Base_pb2.JointSpeeds()
        for joint_id, v in enumerate(np.degrees(v_rad_s)):
            s = joint_speeds.joint_speeds.add()
            s.joint_identifier, s.value, s.duration = joint_id, float(v), 0
        _kortex(self.base.SendJointSpeedsCommand, joint_speeds)

    def check_control(self) -> None:
        """Raise ArmNotInControl now (before asking to move) if another connection controls the arm.
        Sends zero joint speeds, which does nothing to an arm at rest."""
        self.send_joint_speeds(np.zeros(6))

    def send_gripper(self, g_open: float) -> None:
        """Our convention 0 closed .. 1 open -> Kortex GRIPPER_POSITION 0 open .. 1 closed."""
        gripper_command = self.Base_pb2.GripperCommand()
        finger = gripper_command.gripper.finger.add()
        gripper_command.mode = self.Base_pb2.GRIPPER_POSITION
        finger.value = float(np.clip(1.0 - g_open, 0, 1))
        _kortex(self.base.SendGripperCommand, gripper_command)

    def _run_action(self, execute) -> bool:
        e = threading.Event()
        notification_handle = self.base.OnNotificationActionTopic(check_for_end_or_abort(e, self.Base_pb2),
                                                                  self.Base_pb2.NotificationOptions())
        try:
            _kortex(execute)
            finished = e.wait(TIMEOUT_DURATION)
        finally:
            self.base.Unsubscribe(notification_handle)
        if not finished:
            print("Timeout on action notification wait")
            self.stop()
        return finished

    def move_to(self, q) -> bool:
        """Angular action movement to joint angles ``q`` (rad), like the lab's
        example_angular_action_movement (angles sent in degrees, -180..180)."""
        action = self.Base_pb2.Action()
        action.name = "sim2real angular action movement"
        action.application_data = ""
        for joint_id, a in enumerate(np.degrees(wrap(q))):
            joint_angle = action.reach_joint_angles.joint_angles.joint_angles.add()
            joint_angle.joint_identifier = joint_id
            joint_angle.value = float(a)
        return self._run_action(lambda: self.base.ExecuteAction(action))

    def _home_action(self):
        action_type = self.Base_pb2.RequestedActionType()
        action_type.action_type = self.Base_pb2.REACH_JOINT_ANGLES
        for action in self.base.ReadAllActions(action_type).action_list:
            if action.name == "Home":
                return action
        return None

    def home_angles_deg(self):
        """Joint angles of the arm's stored "Home" action (deg), to compare with the sim's home."""
        a = self._home_action()
        if a is None:
            return None
        full = self.base.ReadAction(a.handle)
        return [j.value for j in full.reach_joint_angles.joint_angles.joint_angles]

    def move_home(self) -> bool:
        """The arm's own stored "Home" action (lab example_move_to_home_position)."""
        a = self._home_action()
        if a is None:
            print("Can't find the Home action on the arm")
            return False
        return self._run_action(lambda: self.base.ExecuteActionFromReference(a.handle))

    def stop(self, quiet: bool = False) -> None:
        """Zero speeds + Base.Stop().  ``quiet``: on exit, report problems in one line instead of raising
        (an arm we never controlled can't be stopped by us, and that must not hide the real error)."""
        errors = []
        for call in (lambda: self.send_joint_speeds(np.zeros(6)), lambda: _kortex(self.base.Stop)):
            try:
                call()
            except ArmNotInControl:
                errors.append("not in control")
            except Exception as e:  # noqa: BLE001
                errors.append(str(e).splitlines()[0][:120])
        if errors and not quiet:
            raise RuntimeError("stopping the arm failed: " + "; ".join(errors))
        if errors and quiet and errors != ["not in control"] * len(errors):
            print("  ! stop on exit: " + "; ".join(errors))

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
                 tool_transform=(0.0, 0.0, 0.13, 0.0, 0.0, 90.0), start_q=None, overrides=None,
                 accel_limits_deg_s2=(57.3, 28.6, 22.9, 57.3, 573.0, 573.0)):
        self.sim = SimArm(overrides=overrides, cabinet=False)
        m, e = self.sim.m, self.sim.env
        m.actuator_gainprm[e.arm_act, 0] *= kp_scale
        m.actuator_biasprm[e.arm_act, 1] *= kp_scale
        self.latency, self.gain = int(latency_steps), float(speed_gain)
        self.q_off = np.radians(q_offset_deg)
        self.tool_transform = list(tool_transform)
        self._queue, self._v, self._g, self._tgt = [], np.zeros(6), 1.0, None
        self._v_cmd = np.zeros(6)
        self.a_max = np.radians(accel_limits_deg_s2)     # Kortex limits joint acceleration (measured 2026-11-17)
        self.sim.set_q(self.sim.q_home if start_q is None else start_q, grip=1.0)
        self._t0 = time.time()

    def connect(self):
        pass

    def close(self):
        self.stop()

    def read(self, gripper: bool = True) -> RealState:
        s = self.sim
        return RealState(time.time(), wrap(s.q() + self.q_off), s.dq(), s.torque(), np.zeros(6), s.gripper(),
                         s.tool_pose(self.tool_transform))

    def info(self) -> dict:
        return {"hard_limits": {"joint_speed_deg_s": [57.3] * 6, "twist_linear_m_s": 0.25,
                                "joint_accel_deg_s2": np.degrees(self.a_max).tolist()},
                "tool_transform": self.tool_transform, "arm_state": "FAKE",
                "home_action_deg": self.home_angles_deg()}

    def fk(self, q) -> np.ndarray:
        q_save = self.sim.q()
        p = self.sim.fk(np.asarray(q) - self.q_off, self.tool_transform)
        self.sim.set_q(q_save)
        return p

    def send_joint_speeds(self, v) -> None:
        self._queue.append(np.asarray(v, float) * self.gain)
        if len(self._queue) > self.latency:
            self._v_cmd = self._queue.pop(0)

    def send_gripper(self, g_open) -> None:
        self._g = float(g_open)

    def check_control(self) -> None:
        pass

    def advance(self, dt) -> None:
        """Called by the mirror loop once per period (the real arm advances by itself).
        Like Kortex: the commanded joint speed is integrated into an internal position target."""
        s = self.sim
        if self._tgt is None:
            self._tgt = s.q()
        self._v = np.clip(self._v_cmd, self._v - self.a_max * dt, self._v + self.a_max * dt)
        self._tgt = np.clip(self._tgt + self._v * dt, s.q_limits[:, 0], s.q_limits[:, 1])
        s.command_target(self._tgt, None)
        e = s.env
        cur = (s.d.ctrl[e.grip_act] - e.grip_closed) / (e.grip_open - e.grip_closed)
        step = e.tc["control"].get("gripper_max_speed", 1.5) * dt
        g = cur + np.clip(self._g - cur, -step, step)
        s.d.ctrl[e.grip_act] = e.grip_closed + g * (e.grip_open - e.grip_closed)
        s.tick(dt)

    def move_home(self) -> bool:
        return self.move_to(self.sim.q_home + self.q_off)

    def home_angles_deg(self):
        return np.degrees(self.sim.q_home).tolist()

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

    def stop(self, quiet: bool = False) -> None:
        self._queue, self._v, self._v_cmd, self._tgt = [], np.zeros(6), np.zeros(6), None
        self.sim.hold()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
