"""Gymnasium environment: Kinova Gen3 Lite opens the top drawer of a cabinet.

Everything configurable lives in ``configs/*.yaml``:
  robot.yaml  - joint / actuator / site names, start pose, gripper limits
  drawer.yaml - cabinet, drawer, slide and handle dimensions / physics
  scene.yaml  - table, robot and drawer placement, physics options
  task.yaml   - control mode, observations, reward weights, randomisation

Observations are built from a plain "state dict" in the ROBOT BASE frame by
``build_observation`` so the exact same function can be fed real-robot data
(Kortex feedback + ZED 2 handle pose) at deployment time.
"""
from __future__ import annotations

import math

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from . import drawer_builder as DB
from .config import load_env_config
from .controller import DeltaController
from .kinematics import TCP, orientation_from_axes, rotation_error, solve_ik
from .rewards import DrawerReward
from .scene_builder import build_model

OBS_SIZES = {
    "joint_pos": 6, "joint_vel": 6, "gripper_pos": 1, "tcp_pos": 3, "tcp_rot6d": 6,
    "handle_pos": 3, "handle_to_tcp": 3, "drawer_axis": 3, "drawer_opening": 1,
    # controller state (identical on the real robot, see drawer_rl/controller.py)
    "target_lead": 3, "target_yaw": 1,
}
VISION_KEYS = ("handle_pos", "drawer_axis", "drawer_opening")     # measured by the camera on the real robot


def build_observation(state: dict, include: list[str], last_action: np.ndarray) -> np.ndarray:
    """Flat observation vector from a robot-base-frame state dict.

    Required keys: joint_pos, joint_vel, gripper_pos, tcp_pos, tcp_rot (3x3),
    handle_pos, drawer_axis, drawer_opening.  Re-use this on the real robot.
    """
    parts = []
    for k in include:
        if k == "tcp_rot6d":
            parts.append(state["tcp_rot"][:, :2].T.reshape(-1))
        elif k == "handle_to_tcp":
            parts.append(state["handle_pos"] - state["tcp_pos"])
        elif k == "last_action":
            parts.append(last_action)
        else:
            parts.append(np.atleast_1d(state[k]))
    return np.concatenate(parts).astype(np.float32)


class DrawerEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array", "human"]}

    def __init__(self, cfg: dict | None = None, config_paths: dict | None = None,
                 overrides: list[str] | None = None, render_mode: str | None = None, spec_hook=None):
        """``spec_hook(spec)`` may add things to the scene before it is compiled (e.g. the ArUco
        tags and simulated camera used for vision-in-the-loop tests)."""
        super().__init__()
        self.cfg = cfg if cfg is not None else load_env_config(config_paths, overrides)
        self.render_mode = render_mode
        rc, tc = self.cfg["robot"], self.cfg["task"]
        self.rc, self.tc = rc, tc

        self.model, _ = build_model(self.cfg, spec_hook)
        self.data = mujoco.MjData(self.model)
        self._scratch = mujoco.MjData(self.model)
        m = self.model

        # ---------------- indices ----------------
        self.arm_jids = np.array([m.joint(j).id for j in rc["arm_joints"]])
        self.arm_qadr = m.jnt_qposadr[self.arm_jids]
        self.arm_dadr = m.jnt_dofadr[self.arm_jids]
        self.arm_act = np.array([m.actuator(a).id for a in rc["arm_actuators"]])
        g = rc["gripper"]
        self.grip_act = m.actuator(g["actuator"]).id
        self.grip_qadr = m.jnt_qposadr[m.joint(g["joint"]).id]
        self.grip_open, self.grip_closed = g["open_ctrl"], g["closed_ctrl"]
        self.drawer_jid = m.joint(DB.DRAWER_JOINT).id
        self.drawer_qadr = m.jnt_qposadr[self.drawer_jid]
        self.drawer_dadr = m.jnt_dofadr[self.drawer_jid]
        self.handle_site = m.site(DB.HANDLE_SITE).id
        self.cabinet_body = m.body(DB.CABINET_BODY).id
        self.drawer_body = m.body(DB.DRAWER_BODY).id
        self.base_body = m.body(rc["base_body"]).id
        self.tcp = TCP(m, rc["tcp"]["site"], rc["tcp"]["offset"])
        self.approach_idx = "xyz".index(rc["tcp"]["approach_axis"])

        if rc.get("joint_limits"):
            self.q_limits = np.asarray(rc["joint_limits"], float)
        else:
            self.q_limits = m.jnt_range[self.arm_jids].copy()
        act_lim = m.actuator_ctrlrange[self.arm_act]
        self.q_limits[:, 0] = np.maximum(self.q_limits[:, 0], act_lim[:, 0])
        self.q_limits[:, 1] = np.minimum(self.q_limits[:, 1], act_lim[:, 1])

        # joint torque limits actually enforced by the simulator (joint actuatorfrcrange if set)
        self.arm_torque_limit = np.where(m.jnt_actfrclimited[self.arm_jids],
                                         np.abs(m.jnt_actfrcrange[self.arm_jids]).max(axis=1),
                                         np.abs(m.actuator_forcerange[self.arm_act]).max(axis=1))

        # geom groups for contact checks
        robot_bodies = self._subtree(self.base_body)
        finger_bodies_l = {m.body(b).id for b in g["left_finger_bodies"]}
        finger_bodies_r = {m.body(b).id for b in g["right_finger_bodies"]}
        gb = m.geom_bodyid
        self.handle_geoms = {m.geom(n).id for n in (DB.HANDLE_BAR_GEOM,)}
        self.finger_l_geoms = {i for i in range(m.ngeom) if gb[i] in finger_bodies_l}
        self.finger_r_geoms = {i for i in range(m.ngeom) if gb[i] in finger_bodies_r}
        fingers = self.finger_l_geoms | self.finger_r_geoms
        self.robot_body_geoms = {i for i in range(m.ngeom) if gb[i] in robot_bodies and i not in fingers}
        handle_all = {i for i in range(m.ngeom) if m.geom(i).name.startswith("handle_")}
        self.env_geoms = {i for i in range(m.ngeom)
                          if gb[i] in (self.cabinet_body, self.drawer_body, m.body("table").id)
                          and i not in handle_all}

        # ---------------- base frame ----------------
        mujoco.mj_forward(m, self.data)
        self.base_pos = self.data.xpos[self.base_body].copy()
        self.base_R = self.data.xmat[self.base_body].reshape(3, 3).copy()

        # ---------------- control ----------------
        cc = tc["control"]
        self.n_substeps = max(1, int(round(1.0 / (cc["frequency_hz"] * m.opt.timestep))))
        self.dt = self.n_substeps * m.opt.timestep
        to = cc["target_orientation"]
        R_base = orientation_from_axes(to["approach"], to["finger"], rc["tcp"]["approach_axis"],
                                       rc["tcp"]["finger_axis"])
        self.R_target_nominal = self.base_R @ R_base           # world frame
        # the controller is shared with the real-robot deployment code (drawer_deploy/)
        self.controller = DeltaController(m, cc, self.tcp, self.arm_qadr, self.arm_dadr, self.q_limits,
                                          self.base_pos, self.base_R, self.R_target_nominal, self.dt,
                                          self.grip_open, self.grip_closed,
                                          servo_tau=-m.actuator_biasprm[self.arm_act, 2] / m.actuator_gainprm[self.arm_act, 0])
        self.mode, self.rot_mode = self.controller.mode, self.controller.rot_mode
        self.ws_low, self.ws_high = self.controller.ws_low, self.controller.ws_high
        n_act = self.controller.n_act
        self.action_space = spaces.Box(-1.0, 1.0, shape=(n_act,), dtype=np.float32)

        # ---------------- start pose ----------------
        self.q_start = self._compute_start_q()

        # ---------------- reward / obs ----------------
        ep = tc["episode"]
        self.reward_fn = DrawerReward(tc["reward"], ep["success_opening"])
        self.obs_include = tc["observation"]["include"]
        obs_dim = sum(n_act if k == "last_action" else OBS_SIZES[k] for k in self.obs_include)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)

        # nominal values for domain randomisation
        self._nom = {
            "cab_pos": m.body_pos[self.cabinet_body].copy(),
            "cab_quat": m.body_quat[self.cabinet_body].copy(),
            "slide_damp": m.dof_damping[self.drawer_dadr].copy(),
            "slide_fric": m.dof_frictionloss[self.drawer_dadr].copy(),
            "handle_fric": m.geom_friction[list(self.handle_geoms)].copy(),
            "gainprm": m.actuator_gainprm[self.arm_act].copy(),
            "biasprm": m.actuator_biasprm[self.arm_act].copy(),
            "grip_force": m.actuator_forcerange[self.grip_act].copy(),
        }
        # Rail friction that grows with the sideways/vertical load on the handle (audit recommendation 2):
        # a real drawer jams when it is pulled crooked, so wedging the handle must cost pulling force.
        # 0 = rigid, load-independent slide (v1-v7).
        self.side_load_mu = float(self.cfg["drawer"]["top_drawer"]["slide"].get("side_load_friction", 0.0))
        self._slide_fric_ep = float(m.dof_frictionloss[self.drawer_dadr])
        self._side_force = 0.0
        self._renderer = None
        self._viewer = None
        self.placement_map = None
        if tc["randomization"].get("placement_map"):
            from .placement_map import PlacementMap
            self.placement_map = PlacementMap.load(tc["randomization"]["placement_map"],
                                                   tc["randomization"].get("placement_map_erode"))
        if tc["randomization"].get("validate_reachability", False):
            self._validate_reachability()

    # ------------------------------------------------------------------ utils
    def _subtree(self, root: int) -> set[int]:
        m = self.model
        out = {root}
        for b in range(m.nbody):
            p = b
            while p != 0:
                if p == root:
                    out.add(b)
                    break
                p = m.body_parentid[p]
        return out

    def _b2w(self, p):  # robot-base -> world (points)
        return self.base_pos + self.base_R @ p

    def _w2b(self, p):  # world -> robot-base (points)
        return self.base_R.T @ (p - self.base_pos)

    def _compute_start_q(self) -> np.ndarray:
        rc = self.rc
        if rc.get("start_qpos") is not None:
            return np.asarray(rc["start_qpos"], float)
        target = self._b2w(np.asarray(rc["start_tcp_pos"], float))
        q_pref = np.asarray(rc.get("ik_preferred_qpos") or np.zeros(6), float)
        rng = np.random.default_rng(0)
        seeds = [q_pref] + [rng.uniform(self.q_limits[:, 0], self.q_limits[:, 1]) for _ in range(30)]
        best, best_err = None, np.inf
        for seed in seeds:
            self._scratch.qpos[:] = 0
            q, err = solve_ik(self.model, self._scratch, self.tcp, self.arm_qadr, self.arm_dadr, target,
                              self.R_target_nominal, self.q_limits, q_seed=seed, iters=500)
            rot_err = np.linalg.norm(rotation_error(self.tcp.mat(self._scratch), self.R_target_nominal))
            if err > 0.005 or rot_err > 0.05:
                continue
            score = np.linalg.norm(q - q_pref)          # prefer the solution closest to q_pref
            if score < best_err:
                best, best_err = q, score
        if best is None:
            raise RuntimeError(f"start_tcp_pos {rc['start_tcp_pos']} not reachable; "
                               "set robot.start_qpos explicitly or change the target")
        return best

    def _validate_reachability(self):
        """Kinematic check of the corners of the drawer placement range (fast, no physics)."""
        import itertools

        from .reachability import ReachabilityChecker
        r = self.tc["randomization"]
        dx, dy = r.get("drawer_xy", [0.0, 0.0])
        xs = r.get("drawer_offset_x") or [-dx, dx]
        ys = r.get("drawer_offset_y") or [-dy, dy]
        zs = [-r["drawer_z"], r["drawer_z"]]
        ch = ReachabilityChecker(self)
        if self.placement_map is not None:
            pm = self.placement_map
            stale = pm.check_matches(self.cfg)
            if stale:
                raise ValueError(f"placement map {r['placement_map']} was built for different settings "
                                 f"({', '.join(stale)} changed) -- rebuild it with "
                                 "scripts/check_reachability.py --map")
            frac = pm.safe_fraction(xs, self._yaw_range())
            if frac == 0:
                raise ValueError("no safe placement inside drawer_offset_x x drawer_yaw_range_deg")
            # spot-check random placements the map allows (kinematics, all y/z corners)
            rng = np.random.default_rng(0)
            bad = []
            for _ in range(6):
                x, yw = pm.sample(rng, xs, self._yaw_range())
                bad += [c for c in (ch.check([x, y, z], dynamic=False, yaw_deg=yw)
                                    for y, z in itertools.product(ys, zs)) if not c.feasible]
            self.set_drawer_offset(np.zeros(3), 0.0)
            if bad:
                raise ValueError("placement map allows infeasible placements (map out of date?):\n  " +
                                 "\n  ".join(c.summary() for c in bad[:5]))
            return
        yaws = sorted({*self._yaw_range(), 0.0})
        bad = [c for c in (ch.check([x, y, z], dynamic=False, yaw_deg=yw) for x, y, z, yw in
                           itertools.product(sorted(set(xs)), sorted(set(ys)), sorted(set(zs)), yaws)) if not c.feasible]
        self.set_drawer_offset(np.zeros(3), 0.0)
        if bad:
            raise ValueError("drawer placement range contains placements the arm cannot open:\n  " +
                             "\n  ".join(c.summary() for c in bad) +
                             "\nNarrow task.randomization.drawer_offset_x/_y/drawer_z "
                             "(see scripts/check_reachability.py) or set validate_reachability: false")

    # --------------------------------------------------------------- reset
    def yaw_rotation(self, yaw: float) -> np.ndarray:
        """World-frame rotation by ``yaw`` (rad) about the robot base z axis."""
        return self.controller.yaw_rotation(yaw)

    # controller state, kept as attributes for backwards compatibility
    @property
    def tcp_target(self):
        return self.controller.tcp_target

    @property
    def R_target(self):
        return self.controller.R_target

    @property
    def tcp_yaw(self):
        return self.controller.tcp_yaw

    @property
    def q_target(self):
        return self.controller.q_target

    def sample_drawer_offset(self) -> np.ndarray:
        """Drawer offset (robot base frame, m) from the nominal placement in scene.yaml.

        x/y use ``drawer_offset_x/_y: [min, max]`` if given (asymmetric ranges, e.g. the
        reachable distance band from scripts/check_reachability.py), else +/- ``drawer_xy``.
        """
        r, rng = self.tc["randomization"], self.np_random
        dx, dy = r.get("drawer_xy", [0.0, 0.0])
        ox = rng.uniform(*r["drawer_offset_x"]) if r.get("drawer_offset_x") else rng.uniform(-dx, dx)
        oy = rng.uniform(*r["drawer_offset_y"]) if r.get("drawer_offset_y") else rng.uniform(-dy, dy)
        return np.array([ox, oy, rng.uniform(-r["drawer_z"], r["drawer_z"])])

    def set_drawer_offset(self, off, yaw_deg: float = 0.0):
        """Move the cabinet to nominal + ``off`` (base frame) and rotate it by ``yaw_deg``."""
        m = self.model
        self.drawer_offset = np.asarray(off, float)
        m.body_pos[self.cabinet_body] = self._nom["cab_pos"] + self.base_R @ self.drawer_offset
        dyaw = math.radians(yaw_deg)
        q_yaw = np.array([math.cos(dyaw / 2), 0, 0, math.sin(dyaw / 2)])
        out = np.zeros(4)
        mujoco.mju_mulQuat(out, q_yaw, self._nom["cab_quat"])
        m.body_quat[self.cabinet_body] = out

    def _yaw_range(self):
        r = self.tc["randomization"]
        return r.get("drawer_yaw_range_deg") or [-r.get("drawer_yaw_deg", 0.0), r.get("drawer_yaw_deg", 0.0)]

    def _randomize(self, options=None):
        r, m, rng = self.tc["randomization"], self.model, self.np_random
        options = options or {}
        off = options.get("drawer_offset")
        yaw = options.get("drawer_yaw_deg")
        if off is None and yaw is None and self.placement_map is not None:
            # distance and rotation sampled jointly, only where the arm can physically open the drawer
            off = self.sample_drawer_offset()
            off[0], yaw = self.placement_map.sample(rng, r["drawer_offset_x"], self._yaw_range())
        off = self.sample_drawer_offset() if off is None else np.asarray(off, float)
        yaw = rng.uniform(*self._yaw_range()) if yaw is None else yaw
        self.drawer_yaw_deg = float(yaw)
        self.set_drawer_offset(off, yaw)
        m.dof_damping[self.drawer_dadr] = self._nom["slide_damp"] * rng.uniform(*r["slide_damping_scale"])
        m.dof_frictionloss[self.drawer_dadr] = self._nom["slide_fric"] * rng.uniform(*r["slide_friction_scale"])
        self._slide_fric_ep = float(m.dof_frictionloss[self.drawer_dadr])
        m.geom_friction[list(self.handle_geoms)] = self._nom["handle_fric"] * rng.uniform(*r["handle_friction_scale"])
        s = rng.uniform(*r["actuator_kp_scale"])
        m.actuator_gainprm[self.arm_act, 0] = self._nom["gainprm"][:, 0] * s
        m.actuator_biasprm[self.arm_act, 1] = self._nom["biasprm"][:, 1] * s
        # real grip force is not known precisely: success dropped to 72 % at half force (audit)
        m.actuator_forcerange[self.grip_act] = self._nom["grip_force"] * rng.uniform(*r.get("gripper_force_scale", [1, 1]))
        # constant per-episode error of the observed handle position (vision / calibration bias)
        b = r.get("handle_obs_bias_m", 0.0)
        self.handle_obs_bias = rng.uniform(-b, b, 3) if b else np.zeros(3)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        m, d = self.model, self.data
        self._randomize(options)
        mujoco.mj_resetData(m, d)
        noise = self.rc["start_qpos_noise"]
        q0 = np.clip(self.q_start + self.np_random.uniform(-noise, noise, 6), self.q_limits[:, 0], self.q_limits[:, 1])
        d.qpos[self.arm_qadr] = q0
        d.ctrl[self.arm_act] = q0
        g = self.grip_open if self.rc["start_gripper"] == "open" else self.grip_closed
        d.ctrl[self.grip_act] = g
        d.qpos[self.grip_qadr] = g
        d.qpos[self.drawer_qadr] = self.np_random.uniform(*self.tc["randomization"]["initial_opening"])
        mujoco.mj_forward(m, d)
        # let the gripper / arm settle
        for _ in range(50):
            mujoco.mj_step(m, d)

        self.controller.reset(d, d.ctrl[self.grip_act])
        r = self.tc["randomization"]
        # camera latency / dropouts for the drawer observation (see _obs)
        lo, hi = r.get("vision_delay_steps", [0, 0])
        self._vision_delay = int(self.np_random.integers(lo, hi + 1))
        self._vision_hist, self._vision_last, self._vision_freeze = [], None, 0
        self.last_action = np.zeros(self.action_space.shape, np.float32)
        lo, hi = self.tc["randomization"].get("action_delay_steps", [0, 0])
        self.action_delay = int(self.np_random.integers(lo, hi + 1))
        self._action_queue = [np.zeros(self.action_space.shape, np.float32)] * self.action_delay
        self.prev_opening = self._opening()
        self.prev_tcp = self.tcp.pos(d).copy()
        self._side_force = 0.0
        if self.side_load_mu:
            self.model.dof_frictionloss[self.drawer_dadr] = self._slide_fric_ep
        self.steps = 0
        self.success_count = 0
        self.succeeded = False          # latched once the drawer was held open long enough
        state = self._task_state(self.last_action, self.last_action)
        return self._obs(state), self._info(state, {})

    # ---------------------------------------------------------------- step
    def _apply_action(self, a: np.ndarray):
        d = self.data
        q_target, grip = self.controller.step(a, d)
        d.ctrl[self.arm_act] = q_target
        d.ctrl[self.grip_act] = grip

    def step(self, action):
        a = np.clip(np.asarray(action, np.float32), -1.0, 1.0)
        # simulated control latency (camera + network on the real robot): the action
        # chosen now is executed `action_delay` steps later
        self._action_queue.append(a)
        self._apply_action(self._action_queue.pop(0))
        if self.side_load_mu:      # rail friction from last step's sideways handle load
            self.model.dof_frictionloss[self.drawer_dadr] = self._slide_fric_ep + self.side_load_mu * self._side_force
        try:
            for _ in range(self.n_substeps):
                mujoco.mj_step(self.model, self.data)
        except Exception:  # pragma: no cover
            pass
        self.steps += 1
        unstable = not np.all(np.isfinite(self.data.qpos))

        state = self._task_state(a, self.last_action)
        self._side_force = state["handle_force_perp"]
        reward, terms = self.reward_fn(state)
        ep = self.tc["episode"]
        opened = state["opening"] >= ep["success_opening"]
        if ep.get("success_requires_grasp", False):
            opened = opened and state["grasped"]       # hooking / pushing does not count
        self.success_count = self.success_count + 1 if opened else 0
        success = self.success_count >= ep["success_hold_steps"]
        self.succeeded |= success
        terminated = bool((success and ep["terminate_on_success"]) or unstable)
        truncated = self.steps >= ep["max_steps"]
        if unstable:
            reward = -10.0
            mujoco.mj_resetData(self.model, self.data)

        self.prev_opening = state["opening"]
        self.prev_tcp = state["tcp_pos_w"]
        self.last_action = a
        info = self._info(state, terms)
        info["is_success"] = bool(self.succeeded)
        return self._obs(state), reward, terminated, truncated, info

    # ------------------------------------------------------------ state/obs
    def _opening(self) -> float:
        return float(self.data.qpos[self.drawer_qadr])

    def _contacts(self):
        grasp_l = grasp_r = bad = finger_bad = False
        d = self.data
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = c.geom1, c.geom2
            for a, b in ((g1, g2), (g2, g1)):
                if a in self.handle_geoms:
                    grasp_l |= b in self.finger_l_geoms
                    grasp_r |= b in self.finger_r_geoms
                if b in self.env_geoms:
                    if a in self.robot_body_geoms:
                        bad = True
                    elif a in self.finger_l_geoms or a in self.finger_r_geoms:
                        finger_bad = True       # fingers on the drawer front / cabinet / table
        return grasp_l, grasp_r, bad, finger_bad

    def _handle_force(self) -> np.ndarray:
        """Net contact force (world frame, N) the fingers apply to the handle bar."""
        m, d = self.model, self.data
        F = np.zeros(3)
        f6 = np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = c.geom1, c.geom2
            if g1 in self.handle_geoms or g2 in self.handle_geoms:
                other = g2 if g1 in self.handle_geoms else g1
                if other in self.finger_l_geoms or other in self.finger_r_geoms:
                    mujoco.mj_contactForce(m, d, i, f6)
                    fw = c.frame.reshape(3, 3).T @ f6[:3]          # contact force on geom2 (normal: geom1 -> geom2)
                    F += fw if g2 in self.handle_geoms else -fw
        return F

    def _task_state(self, action, prev_action) -> dict:
        d = self.data
        tcp_w = self.tcp.pos(d).copy()
        R_tcp = self.tcp.mat(d)
        handle_w = d.site_xpos[self.handle_site].copy()
        open_axis_w = -d.xmat[self.drawer_body].reshape(3, 3)[:, 0]  # drawer opens along its -x
        gl, gr, bad, finger_bad = self._contacts()
        rcfg = self.tc["reward"]
        both = rcfg.get("grasp_requires_both_fingers", True)
        gripper_pos = (d.qpos[self.grip_qadr] - self.grip_closed) / (self.grip_open - self.grip_closed)
        touching = (gl and gr) if both else (gl or gr)
        # a real grasp = touching the bar AND the gripper actually closed on it
        grasped = touching and gripper_pos <= rcfg.get("grasp_max_gripper_open", 1.0)
        opening = self._opening()
        F = self._handle_force() if (gl or gr) else np.zeros(3)
        F_par = float(F @ open_axis_w)
        tau_frac = np.abs(d.qfrc_actuator[self.arm_dadr]) / self.arm_torque_limit
        q = d.qpos[self.arm_qadr]
        jr = self.model.jnt_range[self.arm_jids]
        joint_margin = float(np.min(np.minimum(q - jr[:, 0], jr[:, 1] - q)))   # rad, negative = past the limit
        return {
            "joint_margin": joint_margin,
            # controller state: how far the target leads the hand (a lead that persists while the
            # hand is blocked means the arm is pushing) and the commanded gripper yaw
            "target_lead": self.controller.lead(d),
            "target_yaw": self.controller.tcp_yaw,
            # world-frame quantities for the reward
            "handle_force_par": F_par,                                   # pulling force along the drawer axis
            "handle_force_perp": float(np.linalg.norm(F - F_par * open_axis_w)),   # sideways + vertical
            "torque_frac": tau_frac,                                     # |joint torque| / limit, per joint
            "tcp_pos_w": tcp_w,
            "tcp_vel": (tcp_w - self.prev_tcp) / self.dt,
            "tcp_to_handle_dist": float(np.linalg.norm(handle_w - tcp_w)),
            "approach_axis": R_tcp[:, self.approach_idx],
            "open_axis": open_axis_w,
            "grasped": grasped,
            "bad_contact": bad,
            "finger_bad_contact": finger_bad,
            "dt": self.dt,
            "opening": opening,
            "prev_opening": self.prev_opening,
            "action": action,
            "prev_action": prev_action,
            # robot-base-frame quantities for the observation
            "joint_pos": d.qpos[self.arm_qadr].copy(),
            "joint_vel": d.qvel[self.arm_dadr].copy(),
            "gripper_pos": gripper_pos,
            "tcp_pos": self._w2b(tcp_w),
            "tcp_rot": self.base_R.T @ R_tcp,
            "handle_pos": self._w2b(handle_w),
            "drawer_axis": self.base_R.T @ open_axis_w,
            "drawer_opening": opening,
        }

    def _vision_measurement(self, s: dict) -> dict:
        """What the camera pipeline would report this step: noise + calibration bias, then the
        per-episode latency and occasional dropouts (value frozen for a few steps, as when a
        tag is hidden).  Real vision is never instantaneous; training with it keeps the policy
        from relying on perfectly fresh drawer measurements."""
        noise, r, rng = self.tc["observation"]["noise"], self.tc["randomization"], self.np_random
        meas = {
            "handle_pos": s["handle_pos"] + getattr(self, "handle_obs_bias", 0.0)
            + rng.normal(0, noise.get("handle_pos", 0.0) or 0.0, 3),
            "drawer_opening": s["drawer_opening"] + rng.normal(0, noise.get("drawer_opening", 0.0) or 0.0),
        }
        axis = s["drawer_axis"] + rng.normal(0, noise.get("drawer_axis", 0.0) or 0.0, 3)
        meas["drawer_axis"] = axis / max(np.linalg.norm(axis), 1e-9)
        self._vision_hist.append(meas)
        self._vision_hist = self._vision_hist[-(self._vision_delay + 1):]
        if self._vision_freeze > 0 and self._vision_last is not None:
            self._vision_freeze -= 1
            return self._vision_last
        delivered = self._vision_hist[0]                       # the measurement from `delay` steps ago
        if self._vision_last is not None and rng.random() < r.get("vision_dropout_prob", 0.0):
            lo, hi = r.get("vision_dropout_steps", [2, 6])
            self._vision_freeze = int(rng.integers(lo, hi + 1))
        self._vision_last = delivered
        return delivered

    def _obs(self, state: dict) -> np.ndarray:
        noise = self.tc["observation"]["noise"]
        s = dict(state)
        s.update(self._vision_measurement(s))
        for k, std in noise.items():
            if std and k in s and k not in VISION_KEYS:
                s[k] = s[k] + self.np_random.normal(0, std, np.shape(s[k]))
        obs = build_observation(s, self.obs_include, self.last_action)
        # never hand NaN/inf to the learner (would poison VecNormalize and the replay buffer)
        return np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

    def _info(self, state: dict, terms: dict) -> dict:
        info = {"opening": state["opening"], "dist": state["tcp_to_handle_dist"],
                "grasped": bool(state["grasped"]), "bad_contact": bool(state["bad_contact"]),
                "finger_bad_contact": bool(state["finger_bad_contact"]),
                "handle_force_perp": state["handle_force_perp"], "handle_force_par": state["handle_force_par"],
                "max_torque_frac": float(state["torque_frac"].max())}
        info.update({f"r_{k}": float(v) for k, v in terms.items()})
        return info

    # -------------------------------------------------------------- render
    def render(self):
        if self.render_mode == "human":
            if self._viewer is None:
                import mujoco.viewer as mj_viewer
                self._viewer = mj_viewer.launch_passive(self.model, self.data)
            self._viewer.sync()
            return None
        rcfg = self.cfg["scene"]["render"]
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, rcfg["height"], rcfg["width"])
            self._cam = mujoco.MjvCamera()
            c = rcfg["camera"]
            self._cam.lookat[:] = self._b2w(np.asarray(c["lookat"]))
            self._cam.distance, self._cam.azimuth, self._cam.elevation = c["distance"], c["azimuth"], c["elevation"]
        self._renderer.update_scene(self.data, self._cam)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
