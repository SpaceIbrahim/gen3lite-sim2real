"""Reward function for drawer opening.

Kept separate from the environment so it can be edited / swapped without
touching simulation code.  Every term is returned individually (logged to
TensorBoard) and weighted by ``task.yaml -> reward.weights``.

Loopholes found in earlier runs and how this version closes them
(full write-up: docs/training_issues.md):
  * hooking the drawer-front edge instead of the handle -> opening terms are
    only paid while the handle is GRASPED, and finger contact with anything
    but the handle is penalised
  * "grasp" with a wide-open gripper                    -> grasp also requires
    the gripper to be closed past ``grasp_max_gripper_open``
  * yanking the drawer to its end stop                  -> progress reward is
    capped at the success opening; drawer speed above a limit is penalised
  * pushing the drawer shut costs nothing when gated    -> closing is ALWAYS
    penalised (only the positive part of progress is gated)
"""
from __future__ import annotations

import numpy as np


class DrawerReward:
    # term name -> sign (+1 reward, -1 penalty)
    TERMS = {
        "reach": +1, "align": +1, "grasp": +1, "open": +1, "open_progress": +1, "success": +1,
        "off_axis": -1, "collision": -1, "finger_contact": -1, "drawer_speed": -1,
        "handle_side_force": -1, "torque_saturation": -1, "joint_limit": -1,
        "action": -1, "action_rate": -1, "joint_vel": -1,
    }

    def __init__(self, reward_cfg: dict, success_opening: float):
        self.w = {k: float(reward_cfg["weights"].get(k, 0.0)) for k in self.TERMS}
        unknown = set(reward_cfg["weights"]) - set(self.TERMS)
        if unknown:
            raise KeyError(f"Unknown reward terms in config: {unknown}")
        self.reach_scale = reward_cfg["reach_scale"]
        self.open_requires_grasp = reward_cfg.get("open_requires_grasp", True)
        self.max_drawer_speed = reward_cfg.get("max_drawer_speed", np.inf)
        self.align_power = reward_cfg.get("align_power", 1.0)
        self.side_force_free = reward_cfg.get("handle_side_force_free_N", 5.0)
        self.side_force_scale = reward_cfg.get("handle_side_force_scale_N", 10.0)
        self.torque_free = reward_cfg.get("torque_free_fraction", 0.85)
        self.joint_margin_free = np.radians(reward_cfg.get("joint_limit_free_deg", 10.0))
        self.success_opening = success_opening

    def __call__(self, s: dict) -> tuple[float, dict]:
        """``s`` is the state dict produced by ``DrawerEnv._task_state``."""
        target = self.success_opening
        grasp = float(s["grasped"])
        gate = grasp if self.open_requires_grasp else 1.0

        # progress towards the target only (no reward for pulling past it)
        progress = (min(s["opening"], target) - min(s["prev_opening"], target)) / target
        progress = gate * progress if progress > 0 else progress     # closing is always penalised

        perp_vel = s["tcp_vel"] - s["open_axis"] * s["tcp_vel"].dot(s["open_axis"])
        drawer_speed = abs(s["opening"] - s["prev_opening"]) / s["dt"]
        terms = {
            "reach": 1.0 - np.tanh(self.reach_scale * s["tcp_to_handle_dist"]),
            # cos(angle)^p: p = 8 gives 0.97 at 5 deg, 0.76 at 15 deg -> a clear signal to turn the gripper
            "align": max(0.0, float(s["approach_axis"].dot(-s["open_axis"]))) ** self.align_power,
            "grasp": grasp,
            "open": gate * float(np.clip(s["opening"] / target, 0.0, 1.0)),
            "open_progress": progress,
            "success": gate * float(s["opening"] >= target),
            "off_axis": grasp * float(np.linalg.norm(perp_vel)),
            "collision": float(s["bad_contact"]),
            "finger_contact": float(s["finger_bad_contact"]),
            "drawer_speed": max(0.0, drawer_speed - self.max_drawer_speed),
            # load on the handle that is not a pull along the drawer axis (sideways, vertical, or pushing
            # the drawer shut): free in simulation (the slide joint is perfectly
            # rigid) but a real drawer jams or lifts and the real arm hits its torque limits
            "handle_side_force": max(0.0, float(np.hypot(s.get("handle_force_perp", 0.0),
                                                          min(s.get("handle_force_par", 0.0), 0.0)))
                                     - self.side_force_free) / self.side_force_scale,
            # joints driven close to their torque limit (the real Kinova faults there)
            "torque_saturation": float(np.sum(np.clip((np.asarray(s.get("torque_frac", 0.0)) - self.torque_free)
                                                      / (1.0 - self.torque_free), 0.0, None))),
            # closer than joint_limit_free_deg to a joint limit (> 1 = past the limit; MuJoCo limits are soft,
            # the audit found a policy pushing joints up to 7.8 deg past them)
            "joint_limit": max(0.0, (self.joint_margin_free - s.get("joint_margin", np.inf)) / self.joint_margin_free),
            "action": float(np.sum(s["action"] ** 2)),
            "action_rate": float(np.sum((s["action"] - s["prev_action"]) ** 2)),
            "joint_vel": float(np.sum(s["joint_vel"] ** 2)),
        }
        total = sum(self.TERMS[k] * self.w[k] * v for k, v in terms.items())
        return float(total), terms
