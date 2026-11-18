"""Replay recorded real-arm mirror logs through the simulator to measure actuator fidelity.

    python -m sim2real.replay sim2real/lab_results/2026-11-17_session2/*_drive_*.npz
    python -m sim2real.replay <logs> --set task.control.servo_lead_margin=0.02     # try a different actuator model

For every logged control tick, the joint SPEEDS that were sent to the real Kortex arm (``v_cmd_real``) are applied
to the simulated arm for exactly the time the real tick lasted (``t_wall``), through the v8 actuator model
(speed integrated into the servo target, with the configured lead limits).  The simulated joint trajectory is
compared with the joint angles the real arm reported.  The arm starts at the real arm's first measured pose.

Free-space logs only (the lab motions never touched anything): this checks that an actuator change keeps the
sim matched to the real arm while moving.  Contact behaviour is checked separately (no real contact data yet).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from .sim_arm import NOMINAL, SimArm, wrap


def servo_target(ctrl, q_cmd: np.ndarray, q: np.ndarray, v: np.ndarray, dt: float) -> np.ndarray:
    """Integrate one control period of joint speed into the servo target, with the controller's lead limits
    (the SAME rule as DeltaController in velocity actuation)."""
    q_cmd = q_cmd + v * dt
    return ctrl.limit_servo_target(q_cmd, q, v)


def replay(log: Path, overrides: list[str]) -> dict:
    D = dict(np.load(log))
    meta = json.loads(log.with_suffix(".json").read_text())
    if meta.get("real") != "kortex" or "v_cmd_real" not in D or "t_wall" not in D:
        raise ValueError(f"{log.name}: not a real-arm drive log with commands and wall time")
    sim = SimArm(overrides=list(overrides), cabinet=False)
    ctrl = sim.env.controller
    t = D["t_wall"]
    q_real = D["q_real"]
    v_cmd = D["v_cmd_real"]
    sim.set_q(q_real[0])
    q_cmd = q_real[0].copy()
    q_sim = [sim.q()]
    for k in range(len(t) - 1):
        dt = float(np.clip(t[k + 1] - t[k], 1e-3, 0.5))
        q_cmd = servo_target(ctrl, q_cmd, sim.q(), v_cmd[k], dt)
        sim.command_target(q_cmd)
        for _ in range(max(1, int(round(dt / sim.m.opt.timestep)))):
            mujoco.mj_step(sim.m, sim.d)
        q_sim.append(sim.q())
    q_sim = np.array(q_sim)
    err = np.degrees(wrap(q_sim - q_real))
    moving = np.abs(np.degrees(v_cmd)).max(axis=1) > 1.0
    return {"log": log.name, "motion": meta.get("motion"), "mode": meta.get("command_mode"), "ticks": len(t),
            "rms_deg": np.sqrt((err ** 2).mean(axis=0)).round(3).tolist(),
            "max_deg": np.abs(err).max(axis=0).round(2).tolist(),
            "rms_all_deg": float(np.sqrt((err ** 2).mean())), "max_all_deg": float(np.abs(err).max()),
            "rms_moving_deg": float(np.sqrt((err[moving] ** 2).mean())) if moving.any() else float("nan")}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="env config override (repeatable)")
    args = ap.parse_args(argv)
    rows = []
    for p in args.logs:
        try:
            r = replay(Path(p), NOMINAL + args.set)
        except ValueError as e:
            print(f"skip: {e}")
            continue
        rows.append(r)
        print(f"{r['log'][:15]} {r['motion']:12s} {r['mode']:11s} RMS {r['rms_all_deg']:.2f} deg (moving {r['rms_moving_deg']:.2f}), "
              f"max {r['max_all_deg']:.2f} deg | per joint RMS {r['rms_deg']}")
    if rows:
        print(f"ALL: mean RMS {np.mean([r['rms_all_deg'] for r in rows]):.3f} deg, worst max {max(r['max_all_deg'] for r in rows):.2f} deg")
    return rows


if __name__ == "__main__":
    main()
