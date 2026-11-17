"""Summary + plots of a mirror log: where do the real and simulated arms differ?

    python -m sim2real.report sim2real/logs/<stamp>_drive_joint_sweep.npz
    python -m sim2real.report sim2real/logs/<stamp>_follow.npz

Writes <log>.png and <log>_summary.md next to the log and prints the summary.

Per joint (drive logs):
* RMS / max difference real - sim, and real - reference;
* lag: the time shift that best aligns the real joint velocity with the simulated one
  (positive = the real arm is late);
* speed gain: real speed / sim speed while moving (1.0 = same; < 1 = the real joint follows
  commands more slowly -> tune deploy.yaml control.speed_gain or the sim servo);
* hold error: mean |real - reference| while the reference is still;
* torque: real (Kortex estimate) vs sim joint torque while holding still.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from drawer_rl.config import resolve_path  # noqa: E402


def wrapd(a):
    return (np.asarray(a) + 180) % 360 - 180


def lag_ms(a, b, dt, max_lag_s=0.4):
    """Shift (ms) of b relative to a that maximises their correlation; + = b is late."""
    a, b = a - a.mean(), b - b.mean()
    if np.std(a) < 1e-6 or np.std(b) < 1e-6:
        return np.nan
    n = int(max_lag_s / dt)
    lags = np.arange(-n, n + 1)
    c = [np.dot(a[max(0, -k): len(a) - max(0, k)], b[max(0, k): len(b) - max(0, -k)]) for k in lags]
    return 1000 * dt * lags[int(np.argmax(c))]


def first_reach(t, x, start_idx, target, tol=0.1):
    idx = np.where(np.abs(x[start_idx:] - target) < tol)[0]
    return np.nan if len(idx) == 0 else t[start_idx + idx[0]] - t[start_idx]


def torque_sign(meta):
    """Kortex reports joint torque with the opposite sign to MuJoCo (lab run 2026-11-17).  Old logs lack the key."""
    return meta.get("torque_sign_real", -1.0 if meta.get("real") == "kortex" else 1.0)


def time_base(D, meta):
    """Seconds.  Old real-arm logs counted fixed ticks in ``t`` although the loop ran slower: use wall time."""
    if meta.get("real") == "kortex" and "t_wall" in D and "dt" not in D:
        return D["t_wall"]
    return D["t"]


def drive_summary(D, meta):
    t = time_base(D, meta)
    dt = float(np.median(np.diff(t)))
    qr, qs, qref = (np.degrees(D[k]) for k in ("q_real", "q_sim", "q_ref"))
    vr, vs = np.degrees(D["dq_real"]), np.degrees(D["dq_sim"])
    moving_ref = np.r_[np.abs(np.diff(qref, axis=0)).max(axis=1) > 1e-4, False]
    still = ~moving_ref
    for _ in range(int(0.3 / dt)):                     # only well after the reference stopped
        still[1:] &= still[:-1]
    rows = []
    for j in range(6):
        d_rs, d_rr = wrapd(qr[:, j] - qs[:, j]), wrapd(qr[:, j] - qref[:, j])
        mv = np.abs(vs[:, j]) > 2.0
        gain = (np.dot(vr[mv, j], vs[mv, j]) / np.dot(vs[mv, j], vs[mv, j])) if mv.sum() > 5 else np.nan
        tr, ts = torque_sign(meta) * D["torque_real"][still, j], D["torque_sim"][still, j]
        rows.append({
            "joint": j + 1,
            "rms_real_minus_sim_deg": float(np.sqrt(np.mean(d_rs ** 2))),
            "max_real_minus_sim_deg": float(np.abs(d_rs).max()),
            "rms_real_minus_ref_deg": float(np.sqrt(np.mean(d_rr ** 2))),
            "lag_real_vs_sim_ms": float(lag_ms(vs[:, j], vr[:, j], dt)),
            "speed_gain_real_over_sim": float(gain),
            "hold_err_real_deg": float(np.abs(d_rr[still]).mean()) if still.any() else np.nan,
            "hold_err_sim_deg": float(np.abs(wrapd(qs[still, j] - qref[still, j])).mean()) if still.any() else np.nan,
            "hold_torque_real_Nm": float(tr.mean()) if len(tr) else np.nan,
            "hold_torque_sim_Nm": float(ts.mean()) if len(ts) else np.nan,
        })
    out = {"joints": rows}
    # tool pose: the arm's own reported pose vs the sim FK of the same joint angles
    tr_, tf = D["tool_real"], D["tool_sim_fk_of_real"]
    e = 1000 * np.linalg.norm(tr_[:, :3] - tf[:, :3], axis=1)
    out["tool_fk_error_mm"] = {"median": float(np.median(e)), "max": float(e.max())}
    # gripper timing: from each change of the gripper reference to within 10 % of it
    g = {"real": [], "sim": []}
    gref = D["grip_ref"]
    ch = np.where(np.abs(np.diff(gref)) > 0.05)[0] + 1          # (a time-scaled jump spans a few samples)
    starts = [i for n, i in enumerate(ch) if n == 0 or i - ch[n - 1] > 1]
    for i in starts:
        end = i
        while end + 1 < len(gref) and abs(gref[end + 1] - gref[end]) > 0.05:
            end += 1
        for k, key in (("real", "grip_real"), ("sim", "grip_sim")):
            g[k].append(first_reach(t, D[key], i, gref[end]))
    out["gripper_s"] = {k: [round(float(x), 2) for x in v] for k, v in g.items()}
    tw = D["t_wall"] if "t_wall" in D else t
    out["loop_ms"] = {"median": float(np.median(D["loop_ms"])), "max": float(D["loop_ms"].max()),
                      "overruns": meta.get("overruns", 0),
                      "period_median": 1000 * float(np.median(np.diff(tw))) if len(tw) > 1 else float("nan"),
                      "read_median": float(np.median(D["read_ms"])) if "read_ms" in D else float("nan"),
                      "send_median": float(np.median(D["send_ms"])) if "send_ms" in D else float("nan")}
    out["sim_contacts_steps"] = int(D["contact_sim"].sum())
    return out


def to_md(s, meta, log):
    L = [f"# Mirror report: {Path(log).name}", "",
         f"Motion **{meta.get('motion')}** from **{meta.get('start')}**, arm = {meta.get('real')}, "
         f"speed scale {meta.get('speed_scale')}, command mode {meta.get('command_mode')}, {meta.get('rate_hz')} Hz.",
         f"Aborted: **{meta['aborted']}**" if meta.get("aborted") else "Completed without abort.", "",
         "| Joint | RMS real−sim (°) | max real−sim (°) | RMS real−ref (°) | lag real vs sim (ms) | speed gain real/sim | "
         "hold err real / sim (°) | hold torque real / sim (N·m) |", "|---|---|---|---|---|---|---|---|"]
    for r in s["joints"]:
        L.append(f"| {r['joint']} | {r['rms_real_minus_sim_deg']:.2f} | {r['max_real_minus_sim_deg']:.2f} | "
                 f"{r['rms_real_minus_ref_deg']:.2f} | {r['lag_real_vs_sim_ms']:.0f} | {r['speed_gain_real_over_sim']:.2f} | "
                 f"{r['hold_err_real_deg']:.2f} / {r['hold_err_sim_deg']:.2f} | "
                 f"{r['hold_torque_real_Nm']:.2f} / {r['hold_torque_sim_Nm']:.2f} |")
    L += ["", f"* Tool pose reported by the arm vs sim FK of the same joints: median {s['tool_fk_error_mm']['median']:.1f} mm, "
          f"max {s['tool_fk_error_mm']['max']:.1f} mm",
          f"* Gripper time to reach each command (s): real {s['gripper_s']['real']}, sim {s['gripper_s']['sim']}",
          f"* Loop: real period median {s['loop_ms']['period_median']:.0f} ms (target {1000 / meta.get('rate_hz', 20):.0f}), "
          f"work median {s['loop_ms']['median']:.1f} ms (read {s['loop_ms']['read_median']:.1f}, send "
          f"{s['loop_ms']['send_median']:.1f}), max {s['loop_ms']['max']:.1f} ms, {s['loop_ms']['overruns']} overruns",
          f"* Sim arm in contact with table / itself: {s['sim_contacts_steps']} steps", "",
          "How to read it: lag > ~60 ms or speed gain outside 0.8–1.2 means the simulated servo "
          "(configs/robot.yaml actuator gains, task.yaml latency randomisation) does not match the real joint. "
          "A constant hold error on the real arm only is usually gravity sag or a gain the sim does not have. "
          "Tool-pose error > 5 mm means the kinematics or the tool transform differ (run fk-check)."]
    return "\n".join(L)


def plot_drive(D, meta, out):
    t = time_base(D, meta)
    fig, axs = plt.subplots(4, 2, figsize=(14, 13), sharex=True)
    for j in range(6):
        a = axs.flat[j]
        a.plot(t, np.degrees(D["q_ref"][:, j]), "k--", lw=1, label="reference")
        a.plot(t, np.degrees(D["q_real"][:, j]), color="#d62728", lw=1.6, label=f"{meta.get('real')} arm")
        a.plot(t, np.degrees(D["q_sim"][:, j]), color="#1f77b4", lw=1.2, label="sim arm")
        a2 = a.twinx()
        a2.plot(t, wrapd(np.degrees(D["q_real"][:, j] - D["q_sim"][:, j])), color="#7f7f7f", lw=0.8)
        a2.set_ylabel("real − sim (°)", color="#7f7f7f", fontsize=8)
        a.set_ylabel(f"joint_{j + 1} (°)")
        if j == 0:
            a.legend(fontsize=8, loc="upper left")
    a = axs.flat[6]
    a.plot(t, D["grip_ref"], "k--", lw=1, label="reference")
    a.plot(t, D["grip_real"], color="#d62728", label="real")
    a.plot(t, D["grip_sim"], color="#1f77b4", label="sim")
    a.set_ylabel("gripper (1 = open)")
    a.legend(fontsize=8)
    a = axs.flat[7]
    e = 1000 * np.linalg.norm(D["tool_real"][:, :3] - D["tool_sim_fk_of_real"][:, :3], axis=1)
    a.plot(t, e, color="#9467bd", label="tool pose: arm report vs sim FK")
    e2 = 1000 * np.linalg.norm(D["tool_real"][:, :3] - D["tool_sim"][:, :3], axis=1)
    a.plot(t, e2, color="#8c564b", label="tool position: real vs sim arm")
    a.set_ylabel("mm")
    a.legend(fontsize=8)
    for ts, lab in meta.get("segments", []):
        for a in axs.flat:
            a.axvline(ts, color="#bbbbbb", lw=0.6)
        axs.flat[0].text(ts, axs.flat[0].get_ylim()[1], lab, fontsize=7, rotation=90, va="top")
    for a in axs[-1]:
        a.set_xlabel("time (s)")
    fig.suptitle(f"{meta.get('motion')}: real vs simulated Gen3 Lite")
    fig.tight_layout()
    fig.savefig(out, dpi=100)
    plt.close(fig)


def follow_report(D, meta, out_png):
    t = D["t"]
    fig, axs = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    axs[0].plot(t, np.degrees(D["q_real"]))
    axs[0].set_ylabel("joints (°)")
    axs[0].legend([f"j{i}" for i in range(1, 7)], ncol=6, fontsize=8)
    axs[1].plot(t, D["err_mm"], label="position (mm)")
    axs[1].plot(t, D["err_deg"], label="orientation (°)")
    axs[1].set_ylabel("tool pose: arm vs sim FK")
    axs[1].legend(fontsize=8)
    axs[2].plot(t, D["margin_deg"])
    axs[2].set_ylabel("margin to sim limit (°)")
    axs[2].set_xlabel("time (s)")
    for k, c in (("contact_table", "r"), ("contact_cabinet", "m"), ("contact_self", "k")):
        on = D[k].astype(bool)
        if on.any():
            axs[1].plot(t[on], np.zeros(on.sum()), c + "|", ms=12, label=k)
    fig.tight_layout()
    fig.savefig(out_png, dpi=100)
    plt.close(fig)
    md = [f"# Follow report: {Path(out_png).stem}", "",
          f"* Tool pose, arm vs sim FK: median {np.median(D['err_mm']):.2f} mm / {np.median(D['err_deg']):.2f}°, "
          f"max {D['err_mm'].max():.2f} mm / {D['err_deg'].max():.2f}°",
          f"* Closest approach to a SIM joint limit: {D['margin_deg'].min():.1f}° (joint_{int(D['margin_deg'].min(axis=0).argmin()) + 1})",
          f"* Sim contacts: table {int(D['contact_table'].sum())}, cabinet {int(D['contact_cabinet'].sum())}, "
          f"self {int(D['contact_self'].sum())} samples"]
    # static gravity check: real torque estimate vs sim gravity torque (sign and size)
    tr, tg = torque_sign(meta) * D["torque_real"], D["torque_gravity_sim"]
    qd = np.degrees(D["q_real"])
    still = np.r_[True, np.abs(np.diff(qd, axis=0)).max(axis=1) < 0.05]
    fits = []
    for j in range(6):
        x, y = tg[still, j], tr[still, j]
        if len(x) > 20 and np.std(x) > 0.1:
            k, c = np.linalg.lstsq(np.vstack([x, np.ones_like(x)]).T, y, rcond=None)[0]
            fits.append(f"j{j + 1}: real = {k:.2f} x sim {c:+.2f} N·m (corr {np.corrcoef(x, y)[0, 1]:.2f})")
    if fits:
        md.append("* Holding still, real torque vs sim gravity torque (1.0 x = sim masses right): " + "; ".join(fits))
    if np.abs(tr).max() > 0:
        md.append("* Joint torque, real estimate (sign flipped to MuJoCo's) vs sim gravity torque (mean, N·m): " + ", ".join(
            f"j{j + 1} {tr[:, j].mean():.2f}/{tg[:, j].mean():.2f}" for j in range(6)))
    return "\n".join(md)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    args = ap.parse_args(argv)
    p = resolve_path(args.log)
    D = dict(np.load(p))
    meta = json.loads(p.with_suffix(".json").read_text())
    png, mdp = p.with_suffix(".png"), p.with_name(p.stem + "_summary.md")
    if meta.get("mode") == "drive":
        s = drive_summary(D, meta)
        plot_drive(D, meta, png)
        md = to_md(s, meta, p)
        (p.with_name(p.stem + "_summary.json")).write_text(json.dumps(s, indent=1))
    elif meta.get("mode") == "follow":
        md = follow_report(D, meta, png)
    else:
        raise SystemExit("this log has no plots (fk-check prints its result directly)")
    mdp.write_text(md)
    print(md)
    print(f"\nplots: {png}\nsummary: {mdp}")


if __name__ == "__main__":
    main()
