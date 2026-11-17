"""Hand-written reach -> grasp -> pull controller.

Not a learned policy: used as a physics sanity check (can the drawer be opened
at all in a given setup?) by the tests, scripts/scripted_demo.py and the
reachability checker.  It reads ground-truth state from the env.
"""
from __future__ import annotations

import numpy as np


def run_scripted(env, seed=0, options=None, pull_lead=0.02, close_gripper=True, steps=None,
                 hold_extra=0.02, on_step=None, speed_gain=3.0, pull_speed=0.10, compliant=None):
    """Run one episode.  Returns (episode return, final info, per-step log).

    pull_lead   : how far ahead (m) of the TCP the pull target is set each step
    hold_extra  : stop pulling once the drawer is this far past the success opening
    on_step     : optional callback(env, info) for rendering / logging
    speed_gain  : (velocity actuation) commanded TCP speed per metre of distance to the goal, 1/s
    pull_speed  : (velocity actuation) drawer pulling speed, m/s
    compliant   : while holding / pulling the handle, pull the controller target back onto the hand
                  except along the drawer axis (audit: a stiff hold wedged the handle with 20-40 N of side
                  load; compliant ~6 N).  Default: on with velocity actuation (v8), off otherwise.
    """
    env.reset(seed=seed, options=options)
    target = env.tc["episode"]["success_opening"]
    step = env.tc["control"]["ee_max_step"]
    steps = steps or env.tc["episode"]["max_steps"]
    phase, tp, ret, info, log = "approach", 0, 0.0, {}, []
    pulling = False
    # wait for the rate-limited gripper to close (+0.25 s); 25 steps was enough for the old 0.67 s gripper
    gms = env.tc["control"].get("gripper_max_speed")
    grasp_wait = max(25, int(np.ceil((1.0 / gms + 0.25) / env.dt))) if gms else 25
    # Velocity actuation (v8) turns each action into a SPEED command, so "move to the goal in one step"
    # (gain 1/dt = 20 /s) overshoots under the real arm's acceleration limits and servo lag.  Use a speed
    # proportional to the distance, gain ``speed_gain`` (1/s), instead.
    velocity = env.tc["control"].get("actuation", "position") == "velocity"
    k = speed_gain * env.dt if velocity else 1.0
    compliant = velocity if compliant is None else compliant
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
            # stay stiff on the bar until it is gripped (a compliant hand gets pushed off it by the fingers
            # brushing the drawer front), then hold where the hand is
            goal, g = (tcp if compliant and s["grasped"] else h), (-1.0 if close_gripper else 1.0)
            if t - tp > grasp_wait:
                phase = "pull"
        else:                                        # pull along the drawer axis, then hold
            g = -1.0 if close_gripper else 1.0
            pulling = s["opening"] < target + hold_extra
            goal = tcp + ax * pull_lead if pulling else tcp
        a = np.zeros(env.action_space.shape)
        a[:3] = np.clip(k * (goal - tcp) / step, -1, 1)  # base-frame deltas, like the policy
        if velocity and phase == "pull" and pulling:     # pull at a steady speed (drawer limit 0.25 m/s)
            a[:3] = np.clip(ax * pull_speed * env.dt / step, -1, 1)
        holding = compliant and (phase == "pull" or (phase == "grasp" and s["grasped"]))
        if getattr(env, "rot_mode", None) == "yaw" and not holding:   # turn the gripper to face the drawer
            want = np.arctan2(-ax[1], -ax[0])            # approach direction = into the drawer
            a[3] = np.clip(k * (want - env.tcp_yaw) / env.tc["control"]["ee_max_rot_step"], -1, 1)
        if holding:                                      # stop pushing except along the drawer axis
            env.controller.relax(env.data, keep_axis=env.base_R @ ax)
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
