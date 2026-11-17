# Real-arm checklist: what to check while we have the arm

**Already confirmed (2026-11-17, sessions 1–2):**
* kinematics (0.00 mm), tool frame, Home pose, speed limits;
* masses (gravity torque 1.0–1.1× sim) and the joint 3 limit.

**Already found:**
* joint 2–3 acceleration limits;
* 75–150 ms response delay;
* gripper 3× slower than assumed;
* the deploy speed mapping oscillates;
* Kortex torque sign is reversed.

Below is what's still open, most useful first. Each item says what it confirms in the sim. All commands run from
the repo folder on the lab laptop. Run `git pull` first: the stable `tracking` mode is now the default.

Bring: tape measure, ruler, calipers, a spring scale or luggage scale (0–50 N), a kitchen scale, masking tape,
a pen, and the cabinet.

---

## A. Finish the motion tests (~20 min, no setup): **DONE in session 2 (2026-11-17)**

All passed in tracking mode, within 0.1–0.3° of the sim. Pull at full speed is refused by the acceleration
check; 0.75 is the fastest allowed. Below for reference / re-runs after sim changes.

Start at Home each time: `python3 -m sim2real.mirror goto home`.

| Command | Confirms | Look for in the report |
|---|---|---|
| `drive gripper` | Gripper open/close time in both directions, end readings | Sim assumes 0.67 s; the lab run showed ~2.2 to close |
| `--set safety.speed_scale=0.3 drive steps --param "joints=[0]"`, then `drive steps` | Delay, rise time, overshoot, steady error **per joint** | `lag` column, hold error |
| `drive joint_sweep` | Speed gain per joint and gravity sag while moving | speed gain ≈ 1, RMS real−sim |
| `--set loop.command_mode=feedforward drive joint_sweep` | Raw response with no feedback: the cleanest delay and gain numbers | lag (ms), gain |
| `drive cartesian` | Straight TCP lines: whole-arm behaviour like the policy's motion | tool position real vs sim |
| `goto rl_start`, then `drive pull --start rl_start` | The drawer motion, now in tracking mode | Should finish without abort |
| The same with `--set safety.speed_scale=1.0` | Full speed, once 0.5 is clean | Does it still track? |

Every report now also prints **read** and **send** times per tick. That shows which Kortex call makes the loop
75 ms instead of 50 ms (needed for change #5 in `docs/pending_sim_changes.md`).

## B. Confirm the table and base placement with the arm as a probe (~15 min)

The sim assumes the base sits on the table top, +x points to the 83.5 cm edge and +y to the 71 cm side. Use the
arm itself as a measuring device. Jog with the Xbox controller or Web App and read the **tool pose** in the Web App.

1. **Base frame direction:** +x toward the 83.5 cm edge is **confirmed** (session 2). Still to check: jog +y,
   which must move toward the 71 cm side.
2. **Table height relative to the base:** point the gripper straight down, close it, and lower it until the
   fingertips just touch the table.
   * The robot sits on a mount of **about 5 cm**, so expect tool z ≈ **−24 mm**: +26 mm fingertip offset
     (sheet section 2) minus the mount height.
   * **Mount height = 26 mm − (the tool z you read).** That gives the exact value for pending change #9.
   * Repeat at 2–3 places on the table (front, left, right) to check the base is level.
3. **Table position:** put two tape marks on the table measured from the base centre, e.g. 40 cm straight ahead
   and 30 cm to the robot's left. Touch each with the fingertips.
   * Web App x and y should read about **(0.40, 0.00)** and **(0.00, 0.30)**.
   * This confirms the frame direction and that `configs/scene.yaml` puts the table in the right place.
4. Optional: touch near the front edge (x ≈ 0.75 m; the arm reaches ~0.76 m) to confirm the edge distance.

## C. Confirm the drawer placement (~15 min, needs the cabinet)

Place the cabinet where it will be for the real tests: front edge 0.60 m in front of the base centre, 0.18 m to
the robot's right, facing the robot, overhanging the table edge by ~11 cm.

1. Run `goto rl_start`. The open gripper should point straight at the drawer, about 20 cm in front of the handle.
2. Jog the gripper (open, level, pointing +x) until the handle bar sits centred between the finger pads, as if
   grasping. Read the tool pose.
   * Without the mount it would be **x 0.552, y −0.180, z 0.383 m**: the handle centre (0.5595, −0.180, 0.3825),
     minus the 8 mm between the tool frame and the RL TCP. The v8 sim includes the 5 cm mount, so it now expects
     **x 0.552, y −0.180, z 0.333 m** (handle centre 0.5595, −0.180, 0.3325), because the cabinet stands on the
     table, 5 cm below the base.
   * Differences tell you how far the real cabinet is from the sim's, and give the handle height.
3. Close the gripper on the bar. Note the **gripper reading** in the Web App. The deploy code treats < 0.4 as
   "closed on something", and the sim's fingers stop at the 1.1 cm bar.
4. **By hand** (arm released and the gripper opened again, or with the arm out of the way), pull the drawer
   with the spring scale:
   * the force to start it moving (sim: 3 N friction) and to keep it moving slowly (damping);
   * its full travel (sim: 30 cm);
   * whether the overhanging cabinet slides or tips when the drawer is 15 cm open.
5. Measure the top drawer front height (13.5 or 14.5 cm? the PDF disagrees) and the handle (10.5 cm long,
   3 cm standoff, 1.1 cm bar).

## D. Gripper numbers (~10 min)

* **Finger gap** fully open and fully closed, with calipers. Sim: 49 mm and 1 mm.
* **Grip force:** close the gripper on the kitchen scale (or the spring scale through a loop) and read the force.
  The sim assumes ~50 N and randomises ×0.5–1.2.
* **Speed modes:** the lab run used `GRIPPER_POSITION` (default speed). Note whether the Web App gripper speed
  setting changes the 2.2 s close time. If there's a faster setting, we can use it instead of slowing the sim.

## E. Workspace and obstacles (~5 min): only the laptop, which can be moved (session 2)

Run `python3 -m sim2real.mirror follow --viewer` and slowly sweep the gripper through the drawer region:
from `rl_start` to the handle, then 15 cm back toward the robot.
* The sim warns if the arm would hit the table or itself. **Look at the real room:** the laptops, the plexiglass
  divider and the second arm aren't in the sim. Move anything within ~0.8 m of the base out of the way.
* Note the closest joint-limit margin printed for this region (joint 4 limits clockwise drawer rotation).

## F. Repeatability and settings (~5 min)

* `goto home` three times, from different start poses. Note the tool pose each time; the spread should be < 1 mm.
* In the Web App, read (don't change):
  * protection zones;
  * collision or torque thresholds;
  * the gripper speed setting;
  * the control mode used for joint speeds.

  These decide whether a real drawer pull could trip a fault.

## G. Vision (needs the ZED 2 and printed tags; can be a separate session)

The camera-to-robot calibration needs the real arm, so it's worth doing while you have it. See
`docs/vision_pipeline.md` in the drawerRL project:

1. Mount the camera where `plan_camera.py` suggests.
2. Run `calibrate_extrinsics.py`: hand-eye with a tag on the gripper, or touch-points with the TCP.
3. Run `setup_scene.py` with the cabinet in place. It checks the measured drawer pose against the placement map.

## What to send back

* the `sim2real/logs/` folder (push a branch like last time);
* the terminal output of any command that printed numbers (`goto`, `info`);
* the numbers from B–D written down (a photo of the notes is fine).
