"""Procedural MJCF model of the two-drawer cabinet.

Every dimension comes from ``configs/drawer.yaml`` -- nothing is hard coded
here, so re-measuring the real cabinet only requires editing the YAML.

Cabinet-local frame (see drawer.yaml): origin at the bottom-centre of the
carcass front edge, +x into the cabinet, +z up.  The top drawer slides along
-x; its joint value is the opening distance in metres.
"""
from __future__ import annotations

import math

import mujoco
import numpy as np

# Names other modules rely on (kept in one place)
CABINET_BODY = "cabinet"
DRAWER_BODY = "top_drawer"
DRAWER_JOINT = "top_drawer_slide"
HANDLE_BAR_GEOM = "handle_bar"
HANDLE_SITE = "handle_site"
DRAWER_FRONT_SITE = "drawer_front_site"

_BOX = mujoco.mjtGeom.mjGEOM_BOX
_CYL = mujoco.mjtGeom.mjGEOM_CYLINDER

# quaternions rotating a cylinder's local z axis onto world y / x
_Q_Z_TO_Y = [math.cos(math.pi / 4), -math.sin(math.pi / 4), 0.0, 0.0]
_Q_Z_TO_X = [math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0]


def drawer_geometry(dcfg: dict) -> dict:
    """Derived quantities (in cabinet-local frame) that other code needs."""
    cab, td, h = dcfg["cabinet"], dcfg["top_drawer"], dcfg["handle"]
    bar_d = h["bar_diameter"] or (h["standoff"] - h["clearance"])
    front_top = cab["height"] - td["top_gap"]
    front_center_z = front_top - td["front_height"] / 2.0
    handle_center = np.array([
        -td["front_thickness"] - h["clearance"] - bar_d / 2.0,
        h["offset_y"],
        front_center_z + h["offset_z"],
    ])
    return {
        "bar_diameter": bar_d,
        "front_top_z": front_top,
        "front_bottom_z": front_top - td["front_height"],
        "front_center_z": front_center_z,
        "handle_center_closed": handle_center,  # handle bar centre when drawer closed
        "open_axis": np.array([-1.0, 0.0, 0.0]),  # direction the drawer opens
    }


def _apply_contact(geom, phys: dict, friction=None):
    geom.friction = friction if friction is not None else phys["geom_friction"]
    geom.solref = phys["solref"]
    solimp = list(phys["solimp"]) + [0.5, 2.0][len(phys["solimp"]) - 3:]  # MuJoCo wants 5 values
    geom.solimp = solimp[:5]


def add_cabinet(parent_body, dcfg: dict, pos, yaw_deg: float = 0.0):
    """Add the cabinet + sliding top drawer as children of ``parent_body``.

    ``pos`` / ``yaw_deg`` place the cabinet-local origin in the parent frame.
    Returns the cabinet body.
    """
    cab, td, h, phys = dcfg["cabinet"], dcfg["top_drawer"], dcfg["handle"], dcfg["physics"]
    geo = drawer_geometry(dcfg)
    W, D, H = cab["width"], cab["depth"], cab["height"]
    st, bt, bk = cab["side_thickness"], cab["bottom_thickness"], cab["back_thickness"]
    tt, oh = cab["top_thickness"], cab["top_overhang"]
    yaw = math.radians(yaw_deg)
    cab_body = parent_body.add_body(
        name=CABINET_BODY, pos=list(pos), quat=[math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
    )

    def box(body, name, center, half, rgba, friction=None):
        g = body.add_geom(name=name, type=_BOX, pos=list(center), size=list(half), rgba=list(rgba))
        _apply_contact(g, phys, friction)
        return g

    # ---------------- carcass (static) ----------------
    side_h = H - tt
    for side, sgn in (("left", 1), ("right", -1)):
        box(cab_body, f"cab_side_{side}", [D / 2, sgn * (W / 2 - st / 2), side_h / 2],
            [D / 2, st / 2, side_h / 2], cab["rgba"])
    box(cab_body, "cab_bottom", [D / 2, 0, bt / 2], [D / 2, W / 2 - st, bt / 2], cab["rgba"])
    box(cab_body, "cab_back", [D - bk / 2, 0, side_h / 2], [bk / 2, W / 2 - st, side_h / 2], cab["rgba"])
    box(cab_body, "cab_top", [(D - oh) / 2, 0, H - tt / 2], [(D + oh) / 2, W / 2, tt / 2], cab["rgba"])
    # shelf the top drawer box rides on
    box_bottom_z = geo["front_bottom_z"] + 0.01
    shelf_t = 0.012
    box(cab_body, "cab_shelf", [D / 2, 0, box_bottom_z - shelf_t / 2 - 0.002],
        [D / 2 - bk, W / 2 - st, shelf_t / 2], cab["rgba"])

    # static bottom drawer front
    bd = dcfg.get("bottom_drawer", {})
    if bd.get("enabled", True):
        top = geo["front_bottom_z"] - bd["gap_below_top_drawer"]
        box(cab_body, "bottom_drawer_front",
            [-td["front_thickness"] / 2, 0, top - bd["front_height"] / 2],
            [td["front_thickness"] / 2, td["front_width"] / 2, bd["front_height"] / 2], td["rgba"])
        plinth_h = top - bd["front_height"] - 0.003
        if plinth_h > 0:  # recessed kick-plate below the bottom drawer
            box(cab_body, "cab_plinth", [0.02, 0, plinth_h / 2], [0.008, W / 2 - st, plinth_h / 2], cab["rgba"])

    # ---------------- top drawer (slides) ----------------
    drw = cab_body.add_body(name=DRAWER_BODY, pos=[0, 0, 0])
    sl = td["slide"]
    j = drw.add_joint(name=DRAWER_JOINT, type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[-1, 0, 0])
    j.range = [0.0, sl["max_travel"]]
    j.limited = mujoco.mjtLimited.mjLIMITED_TRUE
    j.damping = [sl["damping"], 0.0, 0.0]      # (linear, polynomial terms)
    j.frictionloss = sl["frictionloss"]
    j.armature = sl["armature"]
    j.stiffness = [sl["stiffness"], 0.0, 0.0]
    j.springref = sl["stiffness_ref"]

    ft, fw, fh = td["front_thickness"], td["front_width"], td["front_height"]
    box(drw, "drawer_front", [-ft / 2, 0, geo["front_center_z"]], [ft / 2, fw / 2, fh / 2], td["rgba"])
    bw = W / 2 - st - td["box_side_clearance"]          # half width of drawer box
    bd_, bh, wt = td["box_depth"], td["box_height"], td["box_wall"]
    box(drw, "drawer_box_bottom", [bd_ / 2, 0, box_bottom_z + wt / 2], [bd_ / 2, bw, wt / 2], td["rgba"])
    for side, sgn in (("left", 1), ("right", -1)):
        box(drw, f"drawer_box_{side}", [bd_ / 2, sgn * (bw - wt / 2), box_bottom_z + bh / 2],
            [bd_ / 2, wt / 2, bh / 2], td["rgba"])
    box(drw, "drawer_box_back", [bd_ - wt / 2, 0, box_bottom_z + bh / 2], [wt / 2, bw, bh / 2], td["rgba"])

    # handle
    hc = geo["handle_center_closed"]
    bar_r = geo["bar_diameter"] / 2
    post_r = h["post_diameter"] / 2
    bar = drw.add_geom(name=HANDLE_BAR_GEOM, type=_CYL, pos=hc.tolist(), quat=_Q_Z_TO_Y,
                       size=[bar_r, h["length"] / 2], rgba=list(h["rgba"]))
    _apply_contact(bar, phys, h["friction"])
    post_len = h["clearance"] + bar_r            # from drawer face to bar centre
    for side, sgn in (("left", 1), ("right", -1)):
        g = drw.add_geom(name=f"handle_post_{side}", type=_CYL,
                         pos=[-ft - post_len / 2, hc[1] + sgn * (h["length"] / 2 - post_r), hc[2]],
                         quat=_Q_Z_TO_X, size=[post_r, post_len / 2], rgba=list(h["rgba"]))
        _apply_contact(g, phys, h["friction"])
    drw.add_site(name=HANDLE_SITE, pos=hc.tolist(), size=[0.006, 0, 0], rgba=[1, 0, 0, 1])
    drw.add_site(name=DRAWER_FRONT_SITE, pos=[-ft, 0, geo["front_center_z"]], size=[0.004, 0, 0],
                 rgba=[0, 1, 0, 1])

    # explicit drawer inertia (geom densities would make wood far too heavy)
    m = td["mass"]
    drw.explicitinertial = True
    drw.mass = m
    drw.ipos = [bd_ / 2 - ft, 0, box_bottom_z + bh / 2]
    lx, ly, lz = bd_, 2 * bw, bh
    drw.inertia = [m * (ly**2 + lz**2) / 12, m * (lx**2 + lz**2) / 12, m * (lx**2 + ly**2) / 12]
    return cab_body
