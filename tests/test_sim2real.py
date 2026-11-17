"""sim2real: the mirror tool against the simulated stand-in arm (no hardware needed)."""
import numpy as np
import pytest

from sim2real import mirror
from sim2real.real_arm import FakeArm
from sim2real.sim_arm import SimArm
from sim2real.trajectories import check_motion, joint_sweep


@pytest.fixture(scope="module")
def sim():
    return SimArm(cabinet=False)


def test_fk_matches_mjcf_tool_frame(sim):
    """The Kortex-style tool transform reproduces the MJCF tool_frame exactly."""
    for q in (np.zeros(6), sim.q_home, sim.q_start):
        a = sim.fk(q, mirror.MJCF_TOOL)
        b = sim.fk(q, None)
        assert mirror.pose_error(a, b)[0] < 1e-6 and mirror.pose_error(a, b)[1] < 1e-6


def test_fk_check_detects_encoder_offset(sim):
    fake = FakeArm(q_offset_deg=[0, 2, 0, 0, 0, 0])
    q = sim.q_home
    err_mm, _ = mirror.pose_error(fake.fk(q), sim.fk(q, fake.tool_transform))
    assert err_mm > 5          # 2 deg on the shoulder is ~ 2 cm at the tool


def test_precheck_refuses_table_collision(sim):
    mo = joint_sweep(sim, sim.q_home, sim.dt, amp_deg=80, joints=[1])
    assert any("contact" in p or "above the table" in p for p in check_motion(sim, mo, 10, 0.05))
    ok = joint_sweep(sim, sim.q_home, sim.dt, amp_deg=10, joints=[0])
    assert check_motion(sim, ok, 10, 0.05) == []


def test_drive_dry_run_tracks_and_logs(tmp_path, monkeypatch):
    monkeypatch.setattr(mirror, "PROJECT_ROOT", tmp_path)
    p = mirror.main(["--real", "fake", "--set", f"log_dir={tmp_path}", "--set", "fake.latency_steps=0",
                     "drive", "steps", "--param", "joints=[0]"]) or next(tmp_path.glob("*.npz"))
    D = np.load(p)
    assert np.degrees(np.abs(D["q_real"] - D["q_sim"])).max() < 3
    assert len(D["t"]) > 50
