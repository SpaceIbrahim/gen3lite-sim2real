"""Measurement sheet of the SIMULATED Gen3 Lite, to compare with the real arm.

    python -m sim2real.arm_reference            # prints + writes sim2real/reference/arm_reference.{md,json}

Everything is computed from the model the policy was trained on (MJCF + configs/robot.yaml
overrides), so re-run it whenever the model changes.  Positions are in the robot base frame
(origin = centre of the base on the table, +x = robot front, +z up), in millimetres.
"""
from __future__ import annotations

import json

import mujoco
import numpy as np

from drawer_rl.config import PROJECT_ROOT

from .sim_arm import SimArm

OUT = PROJECT_ROOT / "sim2real" / "reference"
JOINTS = [f"joint_{i}" for i in range(1, 7)]
# Gen3 Lite joint ranges (deg) as listed in the Kinova user guide -- verify on the Web App.
KINOVA_SPEC_LIMITS_DEG = [154.1, 150.1, 150.1, 148.98, 144.97, 148.98]


def test_poses(arm: SimArm) -> dict[str, np.ndarray]:
    """Joint configurations to compare (rad).  All are inside the real limits and clear of the table."""
    p = {
        "zero (all joints 0)": np.zeros(6),
        "home (Kinova factory Home)": arm.q_home,
        "rl_start (training start pose)": arm.q_start,
    }
    h = arm.q_home
    for j, deg in [(0, 30), (1, 20), (2, -30), (3, 45), (4, 30), (5, 60)]:
        q = h.copy()
        q[j] += np.radians(deg)
        p[f"home + joint_{j + 1} {deg:+d} deg"] = q
    return p


def _mesh_world_verts(m, d, gid):
    mid = m.geom_dataid[gid]
    v = m.mesh_vert[m.mesh_vertadr[mid]: m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
    return v @ d.geom_xmat[gid].reshape(3, 3).T + d.geom_xpos[gid]


def _body_geoms(m, body_names, collision_only=True):
    ids = {m.body(b).id for b in body_names}
    return [g for g in range(m.ngeom) if m.geom_bodyid[g] in ids and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
            and (not collision_only or m.geom_contype[g] or m.geom_conaffinity[g])]


def gripper_numbers(arm: SimArm) -> dict:
    m, d, e = arm.m, arm.d, arm.env
    g = e.rc["gripper"]
    left, right = _body_geoms(m, g["left_finger_bodies"]), _body_geoms(m, g["right_finger_bodies"])
    out = {}
    for name, frac in (("open", 1.0), ("closed", 0.0)):
        arm.set_q(arm.q_home, grip=frac)
        # finger joints are coupled by equality constraints: let them settle
        for _ in range(300):
            mujoco.mj_step(m, d)
        gap = min(mujoco.mj_geomDistance(m, d, a, b, 0.2, None) for a in left for b in right)
        out[f"finger_gap_{name}_mm"] = round(1000 * max(gap, 0.0), 1)
    # flange -> fingertip along the approach axis (gripper open)
    arm.set_q(arm.q_home, grip=1.0)
    for _ in range(300):
        mujoco.mj_step(m, d)
    R = d.xmat[arm.ee_body].reshape(3, 3)
    z = R[:, 2]
    tip = max(float(((_mesh_world_verts(m, d, gi) - d.xpos[arm.ee_body]) @ z).max()) for gi in left + right)
    out["flange_to_fingertip_mm"] = round(1000 * tip, 1)
    out["flange_to_tool_frame_mm"] = round(1000 * float((d.xpos[arm.tool_body] - d.xpos[arm.ee_body]) @ z), 1)
    out["flange_to_rl_tcp_mm"] = round(1000 * float((e.tcp.pos(d) - d.xpos[arm.ee_body]) @ z), 1)
    out["grip_force_limit_Nm (finger joint, ASSUMED)"] = float(np.abs(m.actuator_forcerange[e.grip_act]).max())
    return out


def base_numbers(arm: SimArm) -> dict:
    m, d = arm.m, arm.d
    arm.set_q(np.zeros(6))
    g = _body_geoms(m, [arm.env.rc["base_body"]])
    v = np.vstack([_mesh_world_verts(m, d, gi) for gi in g])
    vb = (v - arm.env.base_pos) @ arm.env.base_R
    r = np.linalg.norm(vb[:, :2], axis=1)
    return {"base_diameter_mm": round(2000 * float(r.max()), 1),
            "base_mesh_bottom_z_mm": round(1000 * float(vb[:, 2].min()), 1)}


def masses(arm: SimArm) -> dict:
    m = arm.m
    base = m.body(arm.env.rc["base_body"]).id
    grip = m.body("gripper_base_link").id

    def subtree(b):
        return [i for i in range(m.nbody) if _is_desc(m, i, b)]
    return {"arm_total_kg (base_link .. fingers)": round(float(m.body_mass[subtree(base)].sum()), 3),
            "moving_arm_kg (shoulder .. fingers)": round(float(m.body_mass[subtree(m.body('shoulder_link').id)].sum()), 3),
            "gripper_kg": round(float(m.body_mass[subtree(grip)].sum()), 3)}


def _is_desc(m, i, root):
    while i != 0:
        if i == root:
            return True
        i = m.body_parentid[i]
    return root == 0


def kinematic_chain(arm: SimArm) -> list[dict]:
    """Joint-axis anchor points at q = 0 and the distance between consecutive ones."""
    arm.set_q(np.zeros(6))
    pts = [("base (table surface)", np.zeros(3))] + [(j, arm.joint_anchor(j)) for j in JOINTS] + \
          [("flange (end_effector_link)", arm.body_pos("end_effector_link")),
           ("tool_frame (Kortex default tool)", arm.body_pos("tool_frame")),
           ("RL TCP (tool_frame + 8 mm)", arm.tcp_pos())]
    rows, prev = [], None
    for name, p in pts:
        rows.append({"point": name, "pos_mm": (1000 * p).round(1).tolist(),
                     "from_previous_mm": None if prev is None else round(1000 * float(np.linalg.norm(p - prev)), 1)})
        prev = p
    return rows


def build(arm: SimArm) -> dict:
    lim = {}
    for i, j in enumerate(JOINTS):
        lim[j] = {"mjcf_joint_range_deg": np.degrees(arm.jnt_range[i]).round(2).tolist(),
                  "mjcf_actuator_ctrlrange_deg": np.degrees(arm.ctrl_range[i]).round(2).tolist(),
                  "sim_controller_limit_deg": np.degrees(arm.q_limits[i]).round(2).tolist(),
                  "kinova_spec_deg": [-KINOVA_SPEC_LIMITS_DEG[i], KINOVA_SPEC_LIMITS_DEG[i]],
                  "torque_limit_Nm": float(arm.torque_limits[i]),
                  "servo_time_constant_s": round(float(arm.servo_tau[i]), 4)}
    poses = {}
    for name, q in test_poses(arm).items():
        arm.set_q(q)
        c = arm.contacts()
        poses[name] = {
            "q_deg": np.degrees(q).round(2).tolist(),
            "q_deg_kortex_0_360": (np.degrees(q) % 360).round(2).tolist(),
            "tool_pose_kortex": [*(1000 * arm.tool_pose()[:3]).round(1).tolist(), *arm.tool_pose()[3:].round(2).tolist()],
            "flange_mm": (1000 * arm.flange_pose()[:3]).round(1).tolist(),
            "rl_tcp_mm": (1000 * arm.tcp_pos()).round(1).tolist(),
            "wrist_joint_5_centre_mm": (1000 * arm.joint_anchor("joint_5")).round(1).tolist(),
            "elbow_joint_3_centre_mm": (1000 * arm.joint_anchor("joint_3")).round(1).tolist(),
            "gravity_torque_Nm": arm.gravity_torque(q).round(3).tolist(),
            "sim_contact": [p for p in c["pairs"]],
        }
    tc = arm.env.tc["control"]
    speeds = {"cartesian_max_m_s": round(tc["ee_max_step"] * tc["frequency_hz"], 3),
              "joint_max_speed_deg_s": None if arm.joint_max_speed is None else np.degrees(arm.joint_max_speed).round(1).tolist(),
              "control_rate_hz": tc["frequency_hz"],
              "gripper_full_close_s (ASSUMED)": round(1.0 / tc["gripper_max_speed"], 2)
              if tc.get("gripper_max_speed") else "see configs/task.yaml"}
    return {"kinematic_chain_at_zero": kinematic_chain(arm), "base": base_numbers(arm), "masses": masses(arm),
            "gripper": gripper_numbers(arm), "joint_limits": lim, "speeds": speeds, "test_poses": poses}


def to_markdown(r: dict) -> str:
    L = ["# Simulated Gen3 Lite: measurement sheet", "",
         "Generated by `python -m sim2real.arm_reference` from the training model. Base frame: origin at the "
         "centre of the base on the table, +x = robot front, +y = robot left, +z up. Lengths in mm, angles in deg.",
         "Kortex reports joint angles as 0–360°. The `0–360` column is what the Web App shows.", "",
         "## 1. Kinematic chain at q = 0 (joint-axis anchor points)", "",
         "| Point | x | y | z | Distance from previous |", "|---|---|---|---|---|"]
    for row in r["kinematic_chain_at_zero"]:
        x, y, z = row["pos_mm"]
        L.append(f"| {row['point']} | {x} | {y} | {z} | {'' if row['from_previous_mm'] is None else row['from_previous_mm']} |")
    b, ms, g = r["base"], r["masses"], r["gripper"]
    L += ["", "## 2. Base, masses, gripper", "", "| Quantity | Sim | Real (fill in) |", "|---|---|---|",
          f"| Base diameter | {b['base_diameter_mm']} mm | 105 mm (from the table measurements) |",
          f"| Base mesh bottom (z) | {b['base_mesh_bottom_z_mm']} mm | 0 (sits on the table) |"]
    L += [f"| {k} | {v} | |" for k, v in ms.items()]
    L += [f"| {k} | {v} | |" for k, v in g.items()]
    L += ["", "## 3. Joint limits, torque limits, servo", "",
          "| Joint | MJCF joint range | MJCF actuator ctrlrange | Sim controller uses | Kinova user guide (verify) | Torque limit (N·m) | Servo τ (s) |",
          "|---|---|---|---|---|---|---|"]
    for j, v in r["joint_limits"].items():
        f = lambda a: f"{a[0]:.1f} … {a[1]:.1f}"
        L.append(f"| {j} | {f(v['mjcf_joint_range_deg'])} | {f(v['mjcf_actuator_ctrlrange_deg'])} | "
                 f"{f(v['sim_controller_limit_deg'])} | ±{v['kinova_spec_deg'][1]} | {v['torque_limit_Nm']} | {v['servo_time_constant_s']} |")
    s = r["speeds"]
    L += ["", "## 4. Speeds used in simulation", ""] + [f"* {k}: {v}" for k, v in s.items()]
    L += ["", "## 5. Test poses", "",
          "Command each pose on the real arm (Web App > Angular, or `python -m sim2real.mirror goto`), then compare "
          "the **tool pose** the Web App shows and the tape-measured heights. `fk-check` does the tool-pose "
          "comparison automatically without moving the arm.", ""]
    for name, p in r["test_poses"].items():
        tp = p["tool_pose_kortex"]
        L += [f"### {name}", "",
              f"* Joint angles (deg): `{p['q_deg']}`  → Web App (0–360): `{p['q_deg_kortex_0_360']}`",
              f"* Tool pose (Kortex convention): x {tp[0]} mm, y {tp[1]} mm, z {tp[2]} mm, "
              f"θx {tp[3]}°, θy {tp[4]}°, θz {tp[5]}°",
              f"* Flange centre: `{p['flange_mm']}` mm · RL TCP: `{p['rl_tcp_mm']}` mm",
              f"* Elbow (joint 3 axis) centre: `{p['elbow_joint_3_centre_mm']}` mm · wrist (joint 5 axis) centre: "
              f"`{p['wrist_joint_5_centre_mm']}` mm",
              f"* Gravity torque to hold the pose (N·m): `{p['gravity_torque_Nm']}`",
              f"* Sim contacts: {p['sim_contact'] or 'none'}", ""]
    return "\n".join(L)


def main():
    arm = SimArm(cabinet=False)
    r = build(arm)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "arm_reference.json").write_text(json.dumps(r, indent=1))
    md = to_markdown(r)
    (OUT / "arm_reference.md").write_text(md)
    print(md)
    print(f"\nwritten {OUT / 'arm_reference.md'} and .json")


if __name__ == "__main__":
    main()
