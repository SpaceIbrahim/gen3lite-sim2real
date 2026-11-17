"""KortexArm builds valid Kortex messages and parses real feedback (no robot needed).

Runs only where kortex_api is installed (the lab laptop). Checks every field name we use against the
installed Kortex message definitions. Tested with kortex_api 2.6.0 + protobuf 3.12.4 / 3.20.3.
"""
import threading
from unittest import mock

import numpy as np
import pytest

pytest.importorskip("kortex_api")
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient  # noqa: E402
from kortex_api.autogen.client_stubs.ControlConfigClientRpc import ControlConfigClient  # noqa: E402
from kortex_api.autogen.messages import Base_pb2, BaseCyclic_pb2, ControlConfig_pb2  # noqa: E402

import sim2real.kortex_utilities  # noqa: E402,F401  (imports transports, router, session manager)
from sim2real.real_arm import KortexArm  # noqa: E402


@pytest.fixture
def arm():
    a = KortexArm()
    a.Base_pb2 = Base_pb2
    a.base = mock.create_autospec(BaseClient, instance=True)        # only real 2.x method names allowed
    a.config = mock.create_autospec(ControlConfigClient, instance=True)
    a.base_cyclic = mock.MagicMock()

    def on_notif(cb, opts):
        n = Base_pb2.ActionNotification()
        n.action_event = Base_pb2.ACTION_END
        threading.Timer(0.05, cb, args=(n,)).start()
        return Base_pb2.NotificationHandle()
    a.base.OnNotificationActionTopic.side_effect = on_notif
    return a


def test_read_feedback(arm):
    fb = BaseCyclic_pb2.Feedback()
    for p in (350.0, 344.0, 75.0, 0.0, 300.0, 10.0):
        x = fb.actuators.add()
        x.position, x.velocity, x.torque, x.current_motor = p, 1.0, 0.5, 0.2
    fb.base.tool_pose_x = 0.438
    arm.base_cyclic.RefreshFeedback.return_value = fb
    g = Base_pb2.Gripper()
    g.finger.add().value = 0.25                                      # Kortex: 0 open .. 1 closed
    arm.base.GetMeasuredGripperMovement.return_value = g
    rs = arm.read()
    assert np.allclose(np.degrees(rs.q), [-10, -16, 75, 0, -60, 10], atol=1e-6)
    assert rs.gripper == pytest.approx(0.75)
    assert rs.tool_pose[0] == pytest.approx(0.438, abs=1e-6)


def test_commands(arm):
    arm.send_joint_speeds(np.radians([1, 2, 3, 4, 5, 6]))
    js = arm.base.SendJointSpeedsCommand.call_args[0][0]
    assert [round(s.value, 3) for s in js.joint_speeds] == [1, 2, 3, 4, 5, 6]
    arm.send_gripper(0.3)
    gc = arm.base.SendGripperCommand.call_args[0][0]
    assert gc.mode == Base_pb2.GRIPPER_POSITION and gc.gripper.finger[0].value == pytest.approx(0.7)
    assert arm.move_to(np.radians([0, -16, 75, 0, -60, 0]))
    act = arm.base.ExecuteAction.call_args[0][0]
    assert [round(j.value) for j in act.reach_joint_angles.joint_angles.joint_angles] == [0, -16, 75, 0, -60, 0]
    arm.stop()
    assert arm.base.Stop.called


def test_home_fk_info(arm):
    home = Base_pb2.Action()
    home.name = "Home"
    lst = Base_pb2.ActionList()
    lst.action_list.extend([home])
    arm.base.ReadAllActions.return_value = lst
    full = Base_pb2.Action()
    for i, v in enumerate([0, 344, 75, 0, 300, 0]):
        j = full.reach_joint_angles.joint_angles.joint_angles.add()
        j.joint_identifier, j.value = i, v
    arm.base.ReadAction.return_value = full
    assert arm.home_angles_deg() == [0, 344, 75, 0, 300, 0]
    assert arm.move_home()
    p = Base_pb2.Pose()
    p.x = 0.44
    arm.base.ComputeForwardKinematics.return_value = p
    assert arm.fk(np.zeros(6))[0] == pytest.approx(0.44)
    kl = ControlConfig_pb2.KinematicLimits()
    kl.twist_linear = 0.25
    kl.joint_speed_limits.extend([50.0] * 6)
    arm.config.GetKinematicHardLimits.return_value = kl
    kll = ControlConfig_pb2.KinematicLimitsList()
    kll.kinematic_limits_list.extend([kl])
    arm.config.GetAllKinematicSoftLimits.return_value = kll
    tc = ControlConfig_pb2.ToolConfiguration()
    tc.tool_transform.z = 0.13
    arm.config.GetToolConfiguration.return_value = tc
    arm.base.GetArmState.return_value = Base_pb2.ArmStateInformation()
    info = arm.info()
    assert info["hard_limits"]["joint_speed_deg_s"] == [50.0] * 6
    assert info["tool_transform"][2] == pytest.approx(0.13)
