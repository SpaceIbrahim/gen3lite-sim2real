"""Run the real Gen3 Lite and the simulated one side by side and record every difference.

    python -m sim2real.mirror info                         # what the arm reports: limits, tool, state
    python -m sim2real.mirror fk-check                     # kinematics: Kortex FK vs sim FK (NO motion)
    python -m sim2real.mirror follow [--viewer]            # you jog the real arm (Xbox / Web App); sim shadows it
    python -m sim2real.mirror goto home|rl_start|zero      # move the real arm to a test pose (Kortex planner)
    python -m sim2real.mirror drive joint_sweep            # SAME commands to both arms, logged
    python -m sim2real.mirror drive steps|cartesian|pull|gripper
    python -m sim2real.report sim2real/logs/<log>.npz      # plots + summary of a drive / follow log

Add ``--real fake`` to any command to rehearse it against a simulated stand-in arm (no hardware).
Safety settings are in sim2real/mirror.yaml (``--set safety.speed_scale=0.3``, repeatable).
Keep a hand on the e-stop whenever the real arm moves.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import yaml
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from drawer_rl.config import PROJECT_ROOT, apply_overrides  # noqa: E402
from sim2real.arm_reference import test_poses  # noqa: E402
from sim2real.real_arm import FakeArm, KortexArm  # noqa: E402
from sim2real.sim_arm import SimArm, T_from_pose, wrap  # noqa: E402
from sim2real.trajectories import MOTIONS, check_motion  # noqa: E402

CFG_PATH = PROJECT_ROOT / "sim2real" / "mirror.yaml"
MJCF_TOOL = [0.0, 0.0, 0.13, 0.0, 0.0, 90.0]      # MJCF tool_frame relative to the flange (Kortex convention)


# ------------------------------------------------------------------------------- helpers
def load_cfg(overrides):
    cfg = yaml.safe_load(CFG_PATH.read_text())
    return apply_overrides(cfg, overrides)


def pose_error(a, b) -> tuple[float, float]:
    """(position error mm, orientation error deg) between two Kortex-style poses."""
    Ra, pa = T_from_pose(a)
    Rb, pb = T_from_pose(b)
    ang = np.degrees(Rotation.from_matrix(Ra.T @ Rb).magnitude())
    return 1000 * float(np.linalg.norm(pa - pb)), float(ang)


def named_pose(name: str, sim: SimArm) -> np.ndarray:
    poses = {"home": sim.q_home, "rl_start": sim.q_start, "zero": np.zeros(6)}
    if name not in poses:
        raise SystemExit(f"unknown pose {name!r}; use one of {list(poses)} or --deg")
    return poses[name].copy()


def make_real(args, cfg, start_q=None):
    if args.real == "fake":
        f = cfg["fake"]
        return FakeArm(f["latency_steps"], f["speed_gain"], f["q_offset_deg"], f["kp_scale"], start_q=start_q,
                       overrides=cfg["sim"]["overrides"])
    r = cfg["robot"]
    return KortexArm(r["ip"], r["port"], r["username_env"], r["password_env"])


def tool_transform_of(real) -> list:
    info = real.info()
    tt = info.get("tool_transform")
    if not isinstance(tt, list):
        print(f"  ! could not read the tool transform ({tt}); assuming the MJCF tool frame {MJCF_TOOL}")
        return MJCF_TOOL
    return tt


def confirm(args, text) -> None:
    if args.real == "fake" or args.yes:
        return
    print(text)
    if input("Type 'go' to move the REAL arm (anything else cancels): ").strip().lower() != "go":
        raise SystemExit("cancelled")


def save_log(cfg, tag, arrays: dict, meta: dict) -> Path:
    d = PROJECT_ROOT / cfg["log_dir"]
    d.mkdir(parents=True, exist_ok=True)
    stem = f"{dt.datetime.now():%Y%m%d_%H%M%S}_{tag}"
    np.savez_compressed(d / f"{stem}.npz", **{k: np.asarray(v) for k, v in arrays.items()})
    (d / f"{stem}.json").write_text(json.dumps(meta, indent=1, default=str))
    return d / f"{stem}.npz"


class Rate:
    def __init__(self, hz):
        self.period, self.next = 1.0 / hz, time.perf_counter()
        self.overruns = 0

    def sleep(self):
        self.next += self.period
        delay = self.next - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
        else:
            self.overruns += 1
            self.next = time.perf_counter()


# ---------------------------------------------------------------------------------- info
def cmd_info(args, cfg):
    sim = SimArm(cfg["sim"]["overrides"], cfg["sim"]["cabinet"])
    with make_real(args, cfg) as real:
        info = real.info()
        rs = real.read()
    print(json.dumps(info, indent=1, default=str))
    print("\n=== compared with the simulation ===")
    hl = info.get("hard_limits", {})
    if isinstance(hl, dict) and hl.get("joint_speed_deg_s"):
        print(f"joint speed limit  real {np.round(hl['joint_speed_deg_s'], 1).tolist()} deg/s | "
              f"sim {np.degrees(sim.joint_max_speed).round(1).tolist() if sim.joint_max_speed is not None else None}")
        print(f"Cartesian speed    real {hl.get('twist_linear_m_s')} m/s | sim "
              f"{sim.env.tc['control']['ee_max_step'] * sim.env.tc['control']['frequency_hz']:.3f} m/s")
    tt = info.get("tool_transform")
    if isinstance(tt, list):
        print(f"tool transform     real {np.round(tt, 4).tolist()} | sim tool_frame {MJCF_TOOL}"
              "   (pos m, angles deg; the RL TCP is 8 mm further along z)")
    print(f"joint angles now   {np.degrees(rs.q).round(2).tolist()} deg")
    print(f"margin to SIM joint limits {sim.joint_margin_deg(rs.q).round(1).tolist()} deg")
    if rs.fault:
        print(f"!! {rs.fault}")


# ------------------------------------------------------------------------------ fk-check
def cmd_fk_check(args, cfg):
    sim = SimArm(cfg["sim"]["overrides"], False)
    with make_real(args, cfg) as real:
        tool = tool_transform_of(real)
        rng = np.random.default_rng(0)
        qs = dict(test_poses(sim))
        lo, hi = sim.jnt_range[:, 0] + np.radians(15), sim.jnt_range[:, 1] - np.radians(15)
        for i in range(args.random):
            qs[f"random {i + 1}"] = rng.uniform(lo, hi)
        rs = real.read()
        qs["measured now"] = rs.q
        rows = []
        print(f"tool transform used: {np.round(tool, 4).tolist()}\n")
        print(f"{'pose':34s} {'pos err mm':>10s} {'rot err deg':>11s}   real xyz (mm) | sim xyz (mm)")
        for name, q in qs.items():
            pr = real.fk(q)
            ps = sim.fk(q, tool)
            ep, er = pose_error(pr, ps)
            rows.append((name, ep, er))
            print(f"{name:34s} {ep:10.2f} {er:11.3f}   {(1000 * pr[:3]).round(1).tolist()} | {(1000 * ps[:3]).round(1).tolist()}")
        # the arm's measured tool pose against the sim FK of its measured joints
        ep, er = pose_error(rs.tool_pose, sim.fk(rs.q, tool))
        print(f"{'feedback tool pose vs sim FK':34s} {ep:10.2f} {er:11.3f}")
    errs = np.array([r[1] for r in rows]), np.array([r[2] for r in rows])
    print(f"\nmax position error {errs[0].max():.2f} mm, max orientation error {errs[1].max():.3f} deg "
          f"over {len(rows)} configurations")
    if errs[0].max() > 5 or errs[1].max() > 1:
        print("=> the simulated kinematics do NOT match the arm. Check the joint zero offsets / directions "
              "(compare the per-pose errors: one joint usually explains them) and the tool transform.")
    else:
        print("=> the simulated kinematics match the arm's own model.")
    save_log(cfg, "fk_check", {"err_mm": errs[0], "err_deg": errs[1]},
             {"names": [r[0] for r in rows], "tool": tool, "real": args.real})


# -------------------------------------------------------------------------------- follow
def cmd_follow(args, cfg):
    """Shadow mode: the real arm is moved by someone else; the simulated arm copies its joints."""
    sim = SimArm(cfg["sim"]["overrides"], cfg["sim"]["cabinet"])
    viewer = mujoco.viewer.launch_passive(sim.m, sim.d) if args.viewer else None
    log = {k: [] for k in ("t", "q_real", "grip_real", "tool_real", "tool_sim", "err_mm", "err_deg", "margin_deg",
                           "torque_real", "torque_gravity_sim", "contact_table", "contact_cabinet", "contact_self")}
    with make_real(args, cfg) as real:
        tool = tool_transform_of(real)
        rate, t0, last_print = Rate(cfg["loop"]["rate_hz"]), time.time(), 0.0
        warned = set()
        print("Shadowing the real arm. Jog it with the Xbox controller / Web App. Ctrl+C to stop.")
        try:
            while args.duration <= 0 or time.time() - t0 < args.duration:
                if args.real == "fake":            # nobody jogs the stand-in: give it a slow wave
                    tt = time.time() - t0
                    real.send_joint_speeds(np.radians(10) * np.sin(0.5 * tt + np.arange(6)) * [1, .5, .5, 1, 1, 1])
                    real.advance(rate.period)
                rs = real.read()
                sim.set_q(rs.q, None if np.isnan(rs.gripper) else rs.gripper)
                mujoco.mj_collision(sim.m, sim.d)
                c = sim.contacts()
                ts = sim.tool_pose(tool)
                ep, er = pose_error(rs.tool_pose, ts)
                mg = sim.joint_margin_deg()
                g = sim.gravity_torque(rs.q)
                sim.set_q(rs.q)
                vals = dict(t=rs.t - t0, q_real=rs.q, grip_real=rs.gripper, tool_real=rs.tool_pose, tool_sim=ts,
                            err_mm=ep, err_deg=er, margin_deg=mg, torque_real=rs.torque, torque_gravity_sim=g,
                            contact_table=c["table"], contact_cabinet=c["cabinet"], contact_self=c["self"])
                for k, v in vals.items():
                    log[k].append(v)
                for key, cond, msg in (("table", c["table"], "SIM: arm touches the table"),
                                       ("cab", c["cabinet"], "SIM: arm touches the cabinet"),
                                       ("self", c["self"], "SIM: self-collision"),
                                       ("lim", mg.min() < 5, f"joint_{mg.argmin() + 1} within 5 deg of its SIM limit"),
                                       ("fault", bool(rs.fault), rs.fault)):
                    if cond and key not in warned:
                        print(f"\n  ! t={rs.t - t0:.1f}s {msg} {c['pairs'][:2] if key in ('table', 'cab', 'self') else ''}")
                    (warned.add if cond else warned.discard)(key)
                if time.time() - last_print > 0.5:
                    last_print = time.time()
                    print(f"\r q={np.degrees(rs.q).round(1).tolist()}  tool err {ep:5.1f} mm {er:5.2f} deg  "
                          f"min margin {mg.min():5.1f} deg   ", end="", flush=True)
                if viewer is not None:
                    if not viewer.is_running():
                        break
                    viewer.sync()
                rate.sleep()
        except KeyboardInterrupt:
            pass
    print()
    if viewer is not None:
        viewer.close()
    p = save_log(cfg, "follow", log, {"mode": "follow", "tool": tool, "real": args.real,
                                      "rate_hz": cfg["loop"]["rate_hz"], "overruns": rate.overruns})
    e = np.array(log["err_mm"]) if log["err_mm"] else np.zeros(1)
    print(f"tool-pose error real vs sim FK: median {np.median(e):.2f} mm, max {e.max():.2f} mm  -> {p}")


# ---------------------------------------------------------------------------------- goto
def _check_joint_path(sim, q_a, q_b, cfg):
    probs = []
    for s in np.linspace(0, 1, 40):
        q = q_a + s * (q_b - q_a)
        sim.set_q(q)
        mujoco.mj_collision(sim.m, sim.d)
        c = sim.contacts()
        if c["table"] or c["self"]:
            probs.append(f"{100 * s:.0f}% of the way: sim contact {c['pairs'][:2]}")
        if sim.tcp_pos()[2] < cfg["safety"]["min_tcp_height_m"]:
            probs.append(f"{100 * s:.0f}% of the way: TCP {1000 * sim.tcp_pos()[2]:.0f} mm above the table")
    if sim.joint_margin_deg(q_b).min() < cfg["safety"]["joint_margin_deg"]:
        probs.append(f"target is {sim.joint_margin_deg(q_b).min():.1f} deg from a joint limit")
    return probs[:6]


def cmd_goto(args, cfg):
    sim = SimArm(cfg["sim"]["overrides"], cfg["sim"]["cabinet"])
    q_t = np.radians(args.deg) if args.deg else named_pose(args.pose, sim)
    with make_real(args, cfg) as real:
        rs = real.read()
        probs = _check_joint_path(sim, rs.q, wrap(rs.q + wrap(q_t - rs.q)), cfg)
        if probs:
            raise SystemExit("refusing: the simulated move is unsafe:\n  " + "\n  ".join(probs))
        confirm(args, f"Move from {np.degrees(rs.q).round(1).tolist()}\n  to   {np.degrees(q_t).round(1).tolist()} deg "
                      "with the arm's own joint-space planner.")
        ok = real.move_to(q_t)
        rs = real.read()
    err = np.degrees(wrap(rs.q - q_t))
    print(f"{'reached' if ok else 'NOT reached'}; joint error {err.round(2).tolist()} deg")
    print("tool pose now (Kortex):", np.r_[1000 * rs.tool_pose[:3], rs.tool_pose[3:]].round(1).tolist(),
          "(mm, deg) -- compare with sim2real/reference/arm_reference.md")


# --------------------------------------------------------------------------------- drive
def cmd_drive(args, cfg):
    S, L = cfg["safety"], cfg["loop"]
    sim = SimArm(cfg["sim"]["overrides"], cfg["sim"]["cabinet"])
    shadow = SimArm(cfg["sim"]["overrides"], cfg["sim"]["cabinet"])     # kinematic copy of the REAL arm
    period = 1.0 / L["rate_hz"]
    q_start = named_pose(args.start, sim) if args.start != "current" else None

    with make_real(args, cfg, start_q=q_start) as real:
        rs = real.read()
        q0 = rs.q if q_start is None else q_start
        params = dict(cfg["motions"].get(args.motion, {}))
        for kv in args.param:
            k, v = kv.split("=", 1)
            params[k] = yaml.safe_load(v)
        mo = MOTIONS[args.motion](sim, q0, period, **params)
        probs = check_motion(sim, mo, S["joint_margin_deg"], S["min_tcp_height_m"])
        if probs:
            raise SystemExit("refusing: the motion is unsafe in simulation:\n  " + "\n  ".join(probs))
        far = np.degrees(np.abs(wrap(rs.q - mo.q0))).max()
        if far > S["start_tol_deg"]:
            raise SystemExit(f"the real arm is {far:.1f} deg away from the motion's start pose; run "
                             f"`python -m sim2real.mirror goto {args.start}` first")
        scale = float(S["speed_scale"])
        T = mo.duration / scale
        exc = np.degrees(np.abs(mo.q - mo.q0).max(axis=0)).round(1)
        confirm(args, f"Motion '{mo.name}' from '{args.start}': {T:.1f} s at speed scale {scale}, "
                      f"max joint excursion {exc.tolist()} deg, joint speed cap {S['max_joint_speed_deg_s']} deg/s, "
                      f"command mode '{L['command_mode']}'.")
        tool = tool_transform_of(real)
        vmax = np.radians(S["max_joint_speed_deg_s"])
        dq_cap = vmax * sim.servo_tau
        sim.set_q(rs.q, None if np.isnan(rs.gripper) else rs.gripper)
        g_sim = sim.gripper()
        if L["command_mode"] not in ("independent", "feedforward"):
            raise SystemExit("loop.command_mode must be 'independent' or 'feedforward'")
        feedforward = L["command_mode"] == "feedforward"
        q_tgt_sim = sim.q()
        g_speed = sim.env.tc["control"].get("gripper_max_speed", 1.5)
        names = ("t", "t_wall", "q_ref", "q_real", "q_sim", "dq_real", "dq_sim", "v_cmd_real", "dq_cmd_real", "dq_cmd_sim",
                 "torque_real", "torque_sim", "current_real", "grip_ref", "grip_real", "grip_sim", "tool_real",
                 "tool_sim_fk_of_real", "tool_sim", "contact_sim", "loop_ms")
        log = {k: [] for k in names}
        abort, rate, last_grip = "", Rate(L["rate_hz"]), None
        n_ticks = int(np.ceil(T / period)) + int(1.0 / period)          # + 1 s to settle at the end
        tq_lim = S["abort_torque_fraction"] * sim.torque_limits
        t0 = time.perf_counter()
        try:
            for k in range(n_ticks):
                tic = time.perf_counter()
                rs = real.read()
                if args.real == "fake":
                    rs.t = k * period
                tr = min((k + 1) * period * scale, mo.duration)
                q_ref = np.array([np.interp(tr, mo.t, mo.q[:, j]) for j in range(6)])
                g_ref = float(np.interp(tr, mo.t, mo.grip))
                q_r, q_s = rs.q, sim.q()
                if feedforward:   # both arms get the reference SPEED, no feedback: pure command response
                    q_ref_now = np.array([np.interp(tr - period * scale, mo.t, mo.q[:, j]) for j in range(6)])
                    v_r = np.clip((q_ref - q_ref_now) / period, -vmax, vmax)
                    dq_r = v_r * sim.servo_tau
                    q_tgt_sim = q_tgt_sim + v_r * period      # the sim integrates it like Kortex does
                    dq_s = q_tgt_sim - q_s
                else:             # each arm tracks the reference from its own state (the deploy loop's way)
                    dq_r = np.clip(wrap(q_ref - q_r), -dq_cap, dq_cap)
                    v_r = np.clip(dq_r / sim.servo_tau, -vmax, vmax)
                    dq_s = np.clip(q_ref - q_s, -dq_cap, dq_cap)
                real.send_joint_speeds(v_r)
                if last_grip is None or abs(g_ref - last_grip) > 1e-3:
                    real.send_gripper(g_ref)
                    last_grip = g_ref
                g_sim += np.clip(g_ref - g_sim, -g_speed * period, g_speed * period)
                if feedforward:
                    sim.command_target(q_tgt_sim, g_sim)
                else:
                    sim.command(dq_s, g_sim)
                # log the state at the START of the period, with the commands sent during it
                shadow.set_q(q_r)
                mujoco.mj_collision(sim.m, sim.d)
                c = sim.contacts()
                vals = dict(t=k * period, t_wall=time.perf_counter() - t0, q_ref=q_ref, q_real=q_r, q_sim=q_s, dq_real=rs.dq, dq_sim=sim.dq(),
                            v_cmd_real=v_r, dq_cmd_real=dq_r, dq_cmd_sim=dq_s, torque_real=rs.torque,
                            torque_sim=sim.torque(), current_real=rs.current, grip_ref=g_ref, grip_real=rs.gripper,
                            grip_sim=sim.gripper(), tool_real=rs.tool_pose, tool_sim_fk_of_real=shadow.tool_pose(tool),
                            tool_sim=sim.tool_pose(tool), contact_sim=c["table"] or c["self"],
                            loop_ms=1000 * (time.perf_counter() - tic))
                for kk, v in vals.items():
                    log[kk].append(v)
                # ---- safety
                track = np.degrees(np.abs(wrap(q_r - q_ref))).max()
                div = np.degrees(np.abs(wrap(q_r - q_s))).max()
                mg = shadow.joint_margin_deg(q_r)
                if rs.fault:
                    abort = f"arm fault: {rs.fault}"
                elif track > S["abort_track_deg"]:
                    abort = f"real arm {track:.1f} deg from the reference"
                elif div > S["abort_diverge_deg"]:
                    abort = f"real and simulated arms {div:.1f} deg apart"
                elif mg.min() < S["abort_margin_deg"]:
                    abort = f"joint_{mg.argmin() + 1} {mg.min():.1f} deg from its sim limit"
                elif np.any(np.abs(rs.torque) > tq_lim) and args.real != "fake":
                    j = int(np.argmax(np.abs(rs.torque) / tq_lim))
                    abort = f"joint_{j + 1} torque {rs.torque[j]:.1f} N*m above {tq_lim[j]:.1f}"
                if abort:
                    print(f"\nABORT at t={k * period:.2f}s: {abort}")
                    break
                sim.tick(period)
                if args.real == "fake":
                    real.advance(period)
                if k % max(1, int(L["rate_hz"] / 2)) == 0:
                    print(f"\r t={k * period:5.1f}/{T:.1f}s  track {track:4.1f} deg  real-sim {div:4.1f} deg  ",
                          end="", flush=True)
                if args.real != "fake":
                    rate.sleep()
        except KeyboardInterrupt:
            abort = "stopped by user (Ctrl+C)"
            print("\n" + abort)
        finally:
            real.stop()
    print()
    meta = {"mode": "drive", "motion": mo.name, "start": args.start, "params": params, "real": args.real,
            "speed_scale": scale, "rate_hz": L["rate_hz"], "command_mode": L["command_mode"], "tool": tool,
            "segments": [(t / scale, s) for t, s in mo.segments], "aborted": abort, "overruns": rate.overruns,
            "servo_tau": sim.servo_tau.tolist(), "sim_torque_limits": sim.torque_limits.tolist(),
            "wall_time_s": time.perf_counter() - t0}
    p = save_log(cfg, f"drive_{mo.name}", log, meta)
    print(f"log: {p}\nreport: python -m sim2real.report {p.relative_to(PROJECT_ROOT)}")
    return p


# ---------------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--real", choices=["kortex", "fake"], default="kortex",
                    help="kortex = the real arm; fake = simulated stand-in (dry run)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override sim2real/mirror.yaml (repeatable), e.g. --set safety.speed_scale=0.3")
    ap.add_argument("--yes", action="store_true", help="skip the 'go' confirmation (not recommended on the real arm)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("info")
    p = sub.add_parser("fk-check")
    p.add_argument("--random", type=int, default=20, help="extra random joint configurations to compare")
    p = sub.add_parser("follow")
    p.add_argument("--viewer", action="store_true")
    p.add_argument("--duration", type=float, default=0, help="seconds (0 = until Ctrl+C)")
    p = sub.add_parser("goto")
    p.add_argument("pose", nargs="?", default="home")
    p.add_argument("--deg", type=float, nargs=6, help="explicit joint angles in degrees (-180..180)")
    p = sub.add_parser("drive")
    p.add_argument("motion", choices=sorted(MOTIONS))
    p.add_argument("--start", default="home", help="home | rl_start | zero | current")
    p.add_argument("--param", nargs="*", default=[], help="motion parameters, e.g. amp_deg=10 joints=[0,1]")
    args = ap.parse_args(argv)
    cfg = load_cfg(args.set)
    return {"info": cmd_info, "fk-check": cmd_fk_check, "follow": cmd_follow, "goto": cmd_goto,
            "drive": cmd_drive}[args.cmd](args, cfg)


if __name__ == "__main__":
    main()
