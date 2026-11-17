"""Hand-written reach -> grasp -> pull controller.

Not a learned policy: used as a physics sanity check (can the drawer be opened
at all in a given setup?) by the tests, scripts/scripted_demo.py and the
reachability checker.  It reads ground-truth state from the env.
"""
from __future__ import annotations

import numpy as np


def run_scripted(env, seed=0, options=None, pull_lead=0.02, close_gripper=True, steps=None,
                 hold_extra=0.02, on_step=None):
    """Run one episode.  Returns (episode return, final info, per-step log).

    pull_lead   : how far ahead (m) of the TCP the pull target is set each step
    hold_extra  : stop pulling once the drawer is this far past the success opening
    on_step     : optional callback(env, info) for rendering / logging
    """
    env.reset(seed=seed, options=options)
    target = env.tc["episode"]["success_opening"]
    step = env.tc["control"]["ee_max_step"]
    steps = steps or env.tc["episode"]["max_steps"]
    phase, tp, ret, info, log = "approach", 0, 0.0, {}, []
    for t in range(steps):
        s = env._task_state(env.last_action, env.last_action)
        tcp, h, ax = s["tcp_pos"], s["handle_pos"], s["drawer_axis"]
        g = 1.0
        if phase == "approach":                      # line up 6 cm in front of the handle
            goal = h + ax * 0.06
            if np.linalg.norm(goal - tcp) < 0.01 or t - tp > 80:
                phase, tp = "insert", t
        elif phase == "insert":                      # move straight in
            goal = h
            if np.linalg.norm(goal - tcp) < 0.006 or t - tp > 40:
                phase, tp = "grasp", t
        elif phase == "grasp":                       # close and wait for the rate-limited gripper
            goal, g = h, (-1.0 if close_gripper else 1.0)
            if t - tp > 25:
                phase = "pull"
        else:                                        # pull along the drawer axis, then hold
            g = -1.0 if close_gripper else 1.0
            goal = tcp + ax * pull_lead if s["opening"] < target + hold_extra else tcp
        a = np.zeros(env.action_space.shape)
        a[:3] = np.clip((goal - tcp) / step, -1, 1)      # base-frame deltas, like the policy
        if getattr(env, "rot_mode", None) == "yaw":      # turn the gripper to face the drawer
            want = np.arctan2(-ax[1], -ax[0])            # approach direction = into the drawer
            a[3] = np.clip((want - env.tcp_yaw) / env.tc["control"]["ee_max_rot_step"], -1, 1)
        a[-1] = g
        _, r, term, trunc, info = env.step(a)
        ret += r
        log.append({"phase": phase, "opening": info["opening"], "grasped": info["grasped"],
                    "bad_contact": info["bad_contact"]})
        if on_step:
            on_step(env, info)
        if term or trunc:
            break
    return ret, info, log
