"""Feasibility map of drawer placements (distance x yaw) for the current robot / table / drawer.

Built by ``scripts/check_reachability.py --map`` with drawer_rl.reachability, saved as YAML,
and used by DrawerEnv to sample only placements the arm can physically open.

A cell (x offset, yaw) is *feasible* when every (y, z) corner of the small lateral/height
jitter passes the kinematic checks and the centre also passes the full-physics scripted
grasp-and-pull.  The *safe* set is the feasible set eroded by ``erode`` cells, which gives a
safety margin of at least one grid step in distance and angle.
"""
from __future__ import annotations

import datetime as dt
import itertools

import numpy as np
import yaml

from .config import resolve_path


def _fingerprint(cfg: dict) -> dict:
    """The settings a map depends on -- if any change, the map must be rebuilt."""
    return {
        "drawer_placement": cfg["scene"]["drawer_placement"],
        # optional keys only when set, so maps built before they existed still match their own configs
        "table": {k: cfg["scene"]["table"][k] for k in ("height", "size", "robot_offset_on_table",
                                                         "max_cabinet_overhang", "robot_mount_height")
                  if k in cfg["scene"]["table"]},
        "tcp_offset": list(cfg["robot"]["tcp"]["offset"]),
        "start_tcp_pos": cfg["robot"].get("start_tcp_pos"),
        "start_qpos": cfg["robot"].get("start_qpos"),
        "success_opening": cfg["task"]["episode"]["success_opening"],
        **({"max_steps": cfg["task"]["episode"]["max_steps"]} if cfg["task"]["control"].get("actuation") else {}),
        "handle": cfg["drawer"]["handle"]["offset_z"],
        "cabinet_height": cfg["drawer"]["cabinet"]["height"],
        # controller speed limits change what the arm can do in time (audit follow-up: with the
        # joint speed cap the far + clockwise corner was no longer openable)
        "ee_max_step": cfg["task"]["control"]["ee_max_step"],
        "joint_max_speed": cfg["task"]["control"].get("joint_max_speed"),
        # v8 actuation model (velocity commands with acceleration limits, slower gripper)
        **{k: cfg["task"]["control"][k] for k in ("actuation", "joint_max_accel", "velocity_kp", "gripper_max_speed",
                                                   "servo_lead_margin", "cap_tcp_speed")
           if k in cfg["task"]["control"]},
        **({"side_load_friction": cfg["drawer"]["top_drawer"]["slide"]["side_load_friction"]}
           if "side_load_friction" in cfg["drawer"]["top_drawer"]["slide"] else {}),
    }


class PlacementMap:
    def __init__(self, xs, yaws, feasible, margin_deg=None, erode: int = 1, meta=None):
        self.xs = np.asarray(xs, float)
        self.yaws = np.asarray(yaws, float)
        self.feasible = np.asarray(feasible, bool)          # [len(xs), len(yaws)]
        self.margin_deg = None if margin_deg is None else np.asarray(margin_deg, float)
        self.erode = erode
        self.meta = meta or {}
        self.safe = self._eroded(self.feasible, erode)

    @staticmethod
    def _eroded(f, n):
        s = f.copy()
        for _ in range(n):
            p = np.pad(s, 1, constant_values=False)
            s = s & p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:]
        return s

    # -------------------------------------------------------------------- query
    def _idx(self, x, yaw):
        return int(np.abs(self.xs - x).argmin()), int(np.abs(self.yaws - yaw).argmin())

    def is_safe(self, x, yaw) -> bool:
        if not (self.xs[0] <= x <= self.xs[-1] and self.yaws[0] <= yaw <= self.yaws[-1]):
            return False
        i, j = self._idx(x, yaw)
        return bool(self.safe[i, j])

    def sample(self, rng, x_range, yaw_range, max_tries: int = 1000):
        """Uniform (x, yaw) within the ranges, restricted to safe cells (rejection sampling)."""
        for _ in range(max_tries):
            x, yaw = rng.uniform(*x_range), rng.uniform(*yaw_range)
            if self.is_safe(x, yaw):
                return x, yaw
        raise RuntimeError("no safe drawer placement inside the configured ranges -- check the placement map")

    def safe_fraction(self, x_range, yaw_range) -> float:
        xi = (self.xs >= x_range[0]) & (self.xs <= x_range[1])
        yi = (self.yaws >= yaw_range[0]) & (self.yaws <= yaw_range[1])
        sub = self.safe[np.ix_(xi, yi)]
        return float(sub.mean()) if sub.size else 0.0

    def check_matches(self, cfg) -> list[str]:
        fp = _fingerprint(cfg)
        old = self.meta.get("fingerprint", {})
        return [k for k in fp if yaml.safe_load(yaml.safe_dump(fp[k])) != old.get(k)]

    # ---------------------------------------------------------------------- I/O
    def save(self, path, cfg):
        d = {"created": dt.datetime.now().isoformat(timespec="seconds"),
             "fingerprint": _fingerprint(cfg), "erode": self.erode,
             "xs": self.xs.round(4).tolist(), "yaws_deg": self.yaws.round(3).tolist(),
             "feasible": self.feasible.astype(int).tolist(),
             "min_joint_margin_deg": None if self.margin_deg is None else np.round(self.margin_deg, 1).tolist(),
             "note": "rows = x offsets, columns = yaws; 1 = arm can open the drawer. "
                     "Rebuild with scripts/check_reachability.py --map"}
        p = resolve_path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            yaml.safe_dump(d, f, sort_keys=False, width=200)

    @staticmethod
    def load(path, erode: int | None = None) -> "PlacementMap":
        with open(resolve_path(path)) as f:
            d = yaml.safe_load(f)
        return PlacementMap(d["xs"], d["yaws_deg"], d["feasible"], d.get("min_joint_margin_deg"),
                            d.get("erode", 1) if erode is None else erode, d)


def build_map(checker, xs, yaws, ys=(-0.02, 0.02), zs=(-0.01, 0.01), dynamic=True, progress=None):
    feas = np.zeros((len(xs), len(yaws)), bool)
    marg = np.full((len(xs), len(yaws)), np.nan)
    for i, x in enumerate(xs):
        for j, yw in enumerate(yaws):
            corners = [checker.check([x, y, z], dynamic=False, yaw_deg=yw) for y, z in itertools.product(ys, zs)]
            ok = all(c.feasible for c in corners)
            marg[i, j] = min(c.min_joint_margin_deg for c in corners)
            if ok and dynamic:
                ok = checker.check([x, 0.0, 0.0], dynamic=True, yaw_deg=yw).feasible
            feas[i, j] = ok
        if progress:
            progress(i, x, feas[i])
    return feas, marg
