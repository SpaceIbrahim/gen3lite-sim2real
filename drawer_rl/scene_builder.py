"""Compose the full training world (floor + table + Gen3 Lite + cabinet).

World frame: origin on the floor directly below the robot base; the robot base
sits on the table top at (0, 0, table_height).  All task quantities are later
expressed in the ROBOT BASE frame, so moving the robot/table/drawer in the
configs never requires code changes.
"""
from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from .config import resolve_path
from .drawer_builder import CABINET_BODY, DRAWER_BODY, add_cabinet

_INTEGRATORS = {
    "euler": mujoco.mjtIntegrator.mjINT_EULER,
    "rk4": mujoco.mjtIntegrator.mjINT_RK4,
    "implicit": mujoco.mjtIntegrator.mjINT_IMPLICIT,
    "implicitfast": mujoco.mjtIntegrator.mjINT_IMPLICITFAST,
}
_CONES = {"pyramidal": mujoco.mjtCone.mjCONE_PYRAMIDAL, "elliptic": mujoco.mjtCone.mjCONE_ELLIPTIC}


def _apply_physics(spec: mujoco.MjSpec, phys: dict) -> None:
    o = spec.option
    o.timestep = phys["timestep"]
    o.integrator = _INTEGRATORS[phys["integrator"]]
    o.cone = _CONES[phys["cone"]]
    o.impratio = phys["impratio"]
    o.noslip_iterations = phys["noslip_iterations"]
    o.tolerance = phys["tolerance"]
    o.iterations = phys["iterations"]
    o.gravity = phys["gravity"]


def _apply_actuator_overrides(spec: mujoco.MjSpec, overrides: dict) -> None:
    for name, ov in (overrides or {}).items():
        act = spec.actuator(name)
        if act is None:
            raise KeyError(f"actuator_overrides: no actuator named '{name}'")
        if "kp" in ov:
            act.gainprm[0] = ov["kp"]
            act.biasprm[1] = -ov["kp"]
        if "kv" in ov:
            act.biasprm[2] = -ov["kv"]
        if "forcerange" in ov:
            act.forcerange = ov["forcerange"]
            act.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE


def build_spec(cfg: dict) -> mujoco.MjSpec:
    rcfg, dcfg, scfg = cfg["robot"], cfg["drawer"], cfg["scene"]
    mjcf = resolve_path(rcfg["mjcf_path"])
    spec = mujoco.MjSpec.from_file(str(mjcf))
    spec.modelname = "gen3_lite_drawer"
    # absolute mesh dir so the generated XML can be saved anywhere
    spec.meshdir = str((mjcf.parent / (spec.meshdir or "")).resolve()) + "/"

    # robot keyframes no longer match nq once the drawer joint is added
    for key in list(spec.keys):
        spec.delete(key)

    _apply_physics(spec, scfg["physics"])
    _apply_actuator_overrides(spec, rcfg.get("actuator_overrides"))
    # optional: stiffer coupling of the under-actuated gripper fingers
    # Optional per-coupling stiffness for the under-actuated gripper.  The MJCF couples the
    # fingertip joints RIGIDLY to the driven joint, so fingertips cannot wrap around an object
    # and a thin bar is squeezed out to the tips.  A soft (spring-like) tip coupling models the
    # real passive fingertips.  Keys are the equality's joint1 name.
    couplings = (rcfg.get("gripper", {}) or {}).get("coupling_solref") or {}
    if couplings:
        for eq in spec.equalities:
            if eq.name1 in couplings:
                eq.solref = couplings[eq.name1]

    tbl = scfg["table"]
    th = tbl["height"]
    base = spec.body(rcfg["base_body"])
    base.pos = [0.0, 0.0, th]

    # ---------- visuals ----------
    spec.visual.global_.offwidth = 1920
    spec.visual.global_.offheight = 1080
    spec.visual.headlight.diffuse = [0.6, 0.6, 0.6]
    spec.visual.headlight.ambient = [0.25, 0.25, 0.25]
    spec.add_texture(name="skybox", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                     rgb1=[0.3, 0.5, 0.7], rgb2=[0, 0, 0], width=512, height=3072)
    spec.add_texture(name="groundplane", type=mujoco.mjtTexture.mjTEXTURE_2D,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER, mark=mujoco.mjtMark.mjMARK_EDGE,
                     rgb1=[0.2, 0.3, 0.4], rgb2=[0.1, 0.2, 0.3], markrgb=[0.8, 0.8, 0.8],
                     width=300, height=300)
    mat = spec.add_material(name="groundplane", texuniform=True, texrepeat=[5, 5], reflectance=0.2)
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "groundplane"

    wb = spec.worldbody
    wb.add_light(name="top_light", pos=[0, 0, 2.5], dir=[0, 0, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)
    wb.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05], material="groundplane")

    # ---------- table ----------
    ox, oy = tbl["robot_offset_on_table"]
    sx, sy = tbl["size"]
    t = tbl["thickness"]
    table = wb.add_body(name="table", pos=[-ox, -oy, 0])
    g = table.add_geom(name="table_top", type=mujoco.mjtGeom.mjGEOM_BOX,
                       pos=[0, 0, th - t / 2], size=[sx / 2, sy / 2, t / 2], rgba=tbl["rgba"])
    g.friction = tbl["friction"]
    leg = 0.03
    for i, (lx, ly) in enumerate([(1, 1), (1, -1), (-1, 1), (-1, -1)]):
        table.add_geom(name=f"table_leg_{i}", type=mujoco.mjtGeom.mjGEOM_BOX,
                       pos=[lx * (sx / 2 - leg), ly * (sy / 2 - leg), (th - t) / 2],
                       size=[leg / 2, leg / 2, (th - t) / 2], rgba=tbl["rgba"],
                       contype=0, conaffinity=0)

    # ---------- cabinet (placed in robot base frame) ----------
    dp = scfg["drawer_placement"]
    add_cabinet(wb, dcfg, pos=[dp["x"], dp["y"], th + dp["z"]], yaw_deg=dp["yaw_deg"])
    # the slide joint keeps the drawer aligned; carcass<->drawer contact only adds jitter
    spec.add_exclude(bodyname1=CABINET_BODY, bodyname2=DRAWER_BODY)
    # robot base is bolted to the table
    spec.add_exclude(bodyname1="table", bodyname2=rcfg["base_body"])

    # ---------- cameras (robot base frame, from scene.yaml) ----------
    for name, cam in (scfg.get("cameras") or {}).items():
        pos = np.array(cam["pos"], float) + [0, 0, th]
        fwd = np.array(cam["lookat"], float) + [0, 0, th] - pos
        fwd /= np.linalg.norm(fwd)
        right = np.cross(fwd, [0, 0, 1.0])
        right /= np.linalg.norm(right)
        up = np.cross(right, fwd)
        wb.add_camera(name=name, pos=pos.tolist(), xyaxes=[*right, *up], fovy=cam.get("fovy", 45))
    return spec


def build_model(cfg: dict, spec_hook=None) -> tuple[mujoco.MjModel, mujoco.MjSpec]:
    spec = build_spec(cfg)
    if spec_hook is not None:
        spec_hook(spec)
    return spec.compile(), spec


def save_scene_xml(cfg: dict, out_path: str | Path) -> Path:
    spec = build_spec(cfg)
    spec.compile()
    out = resolve_path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(spec.to_xml())
    return out
