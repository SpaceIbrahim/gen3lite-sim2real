"""Can the arm physically open the drawer at a given placement?

For a drawer offset (robot base frame, from the nominal placement in scene.yaml)
the checker verifies, with the gripper held at the task's fixed orientation:

  1. kinematics  -- an IK solution exists for the TCP at the handle for every
                    opening from closed to success_opening + margin, along one
                    continuous joint path (no flips), with margin to the joint limits
  2. dexterity   -- the arm is not near a singularity anywhere on that path
  3. clearance   -- at every pose on the path the arm body does not touch the
                    cabinet / table and the fingers touch nothing but the handle
  4. layout      -- the cabinet stands fully on the table and the handle path
                    stays inside the controller workspace (task.yaml)
  5. dynamics    -- a scripted grasp-and-pull actually opens the drawer in the
                    full physics simulation (torque limits, friction, contacts)

Used by scripts/check_reachability.py and (optionally) by DrawerEnv at start-up
to refuse randomisation ranges that contain impossible placements.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from .kinematics import rotation_error, solve_ik
from .scripted import run_scripted


@dataclass
class PlacementCheck:
    offset: np.ndarray
    feasible: bool
    yaw_deg: float = 0.0
    reasons: list = field(default_factory=list)
    min_joint_margin_deg: float = np.nan
    min_sigma: float = np.nan              # smallest singular value of the 6x6 Jacobian
    max_ik_err_mm: float = np.nan
    scripted_success: bool | None = None
    scripted_opening: float = np.nan
    limiting_joint: int = 0                 # joint (1-6) closest to its limit along the path
    path_q: np.ndarray | None = None

    def summary(self) -> str:
        s = (f"offset x={self.offset[0]:+.3f} y={self.offset[1]:+.3f} z={self.offset[2]:+.3f} yaw={self.yaw_deg:+5.1f}  "
             f"{'OK  ' if self.feasible else 'FAIL'}  joint margin {self.min_joint_margin_deg:5.1f} deg  "
             f"sigma_min {self.min_sigma:.3f}  ik err {self.max_ik_err_mm:4.1f} mm")
        if self.scripted_success is not None:
            s += f"  scripted {'opened' if self.scripted_success else 'FAILED'} ({self.scripted_opening * 100:.1f} cm)"
        if self.reasons:
            s += "  <- " + "; ".join(self.reasons)
        return s


class ReachabilityChecker:
    def __init__(self, env, opening_margin: float = 0.03, n_path: int = 10, n_approach: int = 12,
                 min_joint_margin_deg: float = 5.0,
                 min_sigma: float = 0.02, max_ik_err_mm: float = 3.0, max_rot_err_deg: float = 3.0):
        self.env = env
        self.m = env.model
        self.d = mujoco.MjData(self.m)
        ep = env.tc["episode"]
        self.openings = np.linspace(0.0, ep["success_opening"] + opening_margin, n_path)
        self.n_approach = n_approach
        self.min_margin = min_joint_margin_deg
        self.min_sigma = min_sigma
        self.max_err = max_ik_err_mm
        self.max_rot = max_rot_err_deg
        sc = env.cfg["scene"]
        self.table = sc["table"]

    # --------------------------------------------------------------- helpers
    def _cabinet_on_table(self, off, yaw_deg: float = 0.0) -> bool:
        """All four corners of the (rotated) cabinet footprint inside the table top (base frame)."""
        dcab = self.env.cfg["drawer"]["cabinet"]
        dp = self.env.cfg["scene"]["drawer_placement"]
        yaw = np.radians(dp["yaw_deg"] + yaw_deg)
        c, s_ = np.cos(yaw), np.sin(yaw)
        corners = np.array([[0, -1], [0, 1], [1, -1], [1, 1]]) * [dcab["depth"], dcab["width"] / 2]
        pts = corners @ np.array([[c, s_], [-s_, c]]) + [dp["x"] + off[0], dp["y"] + off[1]]
        ox, oy = self.table["robot_offset_on_table"]
        sx, sy = self.table["size"]
        tol = float(self.table.get("max_cabinet_overhang", 0.0))
        corners_ok = np.all((pts[:, 0] >= -ox - sx / 2 - tol) & (pts[:, 0] <= -ox + sx / 2 + tol) &
                            (pts[:, 1] >= -oy - sy / 2 - tol) & (pts[:, 1] <= -oy + sy / 2 + tol))
        # the footprint centre must always be on the table (otherwise the real cabinet tips over)
        ctr = pts.mean(axis=0)
        ctr_ok = (-ox - sx / 2 <= ctr[0] <= -ox + sx / 2) and (-oy - sy / 2 <= ctr[1] <= -oy + sy / 2)
        return bool(corners_ok and ctr_ok)

    def _contacts(self):
        env, d = self.env, self.d
        body = finger = False
        for i in range(d.ncon):
            c = d.contact[i]
            for a, b in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
                if b in env.env_geoms:
                    body |= a in env.robot_body_geoms
                    finger |= a in env.finger_l_geoms or a in env.finger_r_geoms
        return body, finger

    # ------------------------------------------------------------- the check
    def check(self, offset, dynamic: bool = True, seed: int = 0, yaw_deg: float = 0.0) -> PlacementCheck:
        env, m, d = self.env, self.m, self.d
        off = np.asarray(offset, float)
        res = PlacementCheck(off, True, yaw_deg)
        if not self._cabinet_on_table(off, yaw_deg):
            res.reasons.append("cabinet not fully on the table")
        env.set_drawer_offset(off, yaw_deg)
        # gripper orientation along the pull: turned with the drawer if the policy can yaw,
        # otherwise the fixed nominal orientation
        if env.mode == "ee_delta" and getattr(env, "rot_mode", None) in ("yaw", "full"):
            max_yaw = env.tc["control"].get("ee_max_yaw_deg", 30.0)
            if abs(yaw_deg) > max_yaw:
                res.reasons.append(f"drawer yaw beyond the gripper yaw limit ({max_yaw} deg)")
            R_goal = env.yaw_rotation(np.radians(np.clip(yaw_deg, -max_yaw, max_yaw))) @ env.R_target_nominal
        else:
            R_goal = env.R_target_nominal
        mujoco.mj_resetData(m, d)
        q = env.q_start.copy()
        margins, sigmas, errs, path = [], [], [], []
        lo, hi = env.q_limits[:, 0], env.q_limits[:, 1]
        ws_lo, ws_hi = env.ws_low, env.ws_high
        # waypoints: (a) approach from the real start pose to the closed handle, turning the gripper
        # gradually (this is how the incremental controller moves -- no jumps to another arm
        # configuration allowed), then (b) the pull along the drawer axis
        d.qpos[env.arm_qadr] = env.q_start
        mujoco.mj_kinematics(m, d)
        start_tcp = env.tcp.pos(d).copy()
        start_R = env.tcp.mat(d).copy()
        d.qpos[env.drawer_qadr] = 0.0
        mujoco.mj_kinematics(m, d)
        h0 = d.site_xpos[env.handle_site].copy()
        waypoints = []
        for k in range(1, self.n_approach + 1):
            f = k / self.n_approach
            yaw_rel = np.arctan2((R_goal @ start_R.T)[1, 0], (R_goal @ start_R.T)[0, 0])
            waypoints.append(("approach", 0.0, start_tcp + f * (h0 - start_tcp), env.yaw_rotation(f * yaw_rel) @ start_R))
        waypoints += [("pull", op, None, R_goal) for op in self.openings]
        for kind, op, pos, R_w in waypoints:
            d.qpos[env.drawer_qadr] = op
            d.qpos[env.grip_qadr] = env.grip_closed if kind == "pull" else env.grip_open
            mujoco.mj_kinematics(m, d)
            handle_w = d.site_xpos[env.handle_site].copy() if pos is None else pos
            where = "on the approach" if kind == "approach" else f"at opening {op:.2f} m"
            if np.any(handle_w < ws_lo - 1e-6) or np.any(handle_w > ws_hi + 1e-6):
                res.reasons.append(f"handle outside control workspace {where}")
                break
            q_new, err = solve_ik(m, d, env.tcp, env.arm_qadr, env.arm_dadr, handle_w, R_w,
                                  env.q_limits, q_seed=q, iters=400)
            rot = np.degrees(np.linalg.norm(rotation_error(env.tcp.mat(d), R_w)))
            errs.append(err * 1000)
            if err * 1000 > self.max_err or rot > self.max_rot:
                res.reasons.append(f"cannot reach the handle {where} (err {err * 1000:.1f} mm, {rot:.1f} deg)")
                break
            if np.max(np.abs(q_new - q)) > 0.6:
                res.reasons.append(f"arm would have to jump to another configuration {where}")
                break
            q = q_new
            path.append(q.copy())
            jm = np.minimum(q - lo, hi - q)
            margins.append(np.degrees(jm.min()))
            res.limiting_joint = int(np.argmin(jm)) + 1 if margins[-1] == min(margins) else res.limiting_joint
            if kind == "approach":
                continue
            mujoco.mj_forward(m, d)
            sigmas.append(np.linalg.svd(env.tcp.jacobian(m, d, env.arm_dadr), compute_uv=False).min())
            body, finger = self._contacts()
            if body:
                res.reasons.append(f"arm body hits cabinet/table at opening {op:.2f} m")
                break
            if finger:
                res.reasons.append(f"fingers hit the drawer front at opening {op:.2f} m")
                break
        if margins:
            res.min_joint_margin_deg = float(min(margins))
            res.min_sigma = float(min(sigmas)) if sigmas else float("nan")
            res.max_ik_err_mm = float(max(errs))
            if res.min_joint_margin_deg < self.min_margin:
                res.reasons.append(f"joint {res.limiting_joint} within {res.min_joint_margin_deg:.1f} deg of its limit")
            if sigmas and res.min_sigma < self.min_sigma:
                res.reasons.append(f"near singular (sigma {res.min_sigma:.3f})")
        res.path_q = np.array(path)
        res.feasible = not res.reasons
        if dynamic and res.feasible:
            _, info, _ = run_scripted(env, seed=seed, options={"drawer_offset": off, "drawer_yaw_deg": yaw_deg})
            res.scripted_success = bool(info["is_success"])
            res.scripted_opening = float(info["opening"])
            if not res.scripted_success:
                res.feasible = False
                res.reasons.append("scripted grasp-and-pull failed in full physics")
        return res

    def sweep(self, xs, ys=(0.0,), z: float = 0.0, dynamic: bool = True, yaw_deg: float = 0.0):
        return [[self.check([x, y, z], dynamic, yaw_deg=yaw_deg) for x in xs] for y in ys]

    def feasible_interval(self, xs, y: float = 0.0, z: float = 0.0, dynamic: bool = True):
        """Largest contiguous x-interval (containing 0 if possible) where every placement is feasible."""
        checks = [self.check([x, y, z], dynamic) for x in xs]
        ok = [c.feasible for c in checks]
        best, cur = None, None
        for i, f in enumerate(ok):
            if f:
                cur = (cur[0], i) if cur else (i, i)
                if best is None or (cur[1] - cur[0]) > (best[1] - best[0]) or \
                        (xs[cur[0]] <= 0 <= xs[cur[1]] and not xs[best[0]] <= 0 <= xs[best[1]]):
                    best = cur
            else:
                cur = None
        return (None if best is None else (float(xs[best[0]]), float(xs[best[1]]))), checks
