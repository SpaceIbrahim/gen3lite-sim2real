# sim2real: run the real and simulated Gen3 Lite side by side

This folder checks how closely the simulated arm the policy was trained on matches the real arm. It uses
the same MJCF, actuator overrides, table and controller time constants as training. Three ways to compare:

| Command | Real arm moves? | What it shows |
|---|---|---|
| `info` | no | Limits, tool configuration and arm state as the arm reports them, compared with the sim |
| `fk-check` | **no** | Kortex's own forward kinematics vs the sim's, at ~30 joint configurations. Finds joint zero/direction mismatches and tool-frame differences |
| `follow` | you jog it (Xbox / Web App) | The sim copies the measured joints live. It warns when the sim arm would hit the table, the cabinet or itself, or get near a joint limit, and logs the tool-pose error plus real vs sim gravity torque |
| `goto <pose>` | yes (Kortex planner) | Moves to `home` (the arm's stored Home action, like the lab demo), `rl_start` or `zero` after checking the path in sim. Prints the tool pose to compare with the sheet, and for Home also how far it is from the sim's home |
| `drive <motion>` | yes (joint speeds) | Sends the **same** commands to both arms and logs both. Shows lag, speed gain, tracking, sag, gripper timing and torque differences |

`python -m sim2real.report <log>.npz` turns a `drive` or `follow` log into plots and a summary table.

`python -m sim2real.arm_reference` writes [reference/arm_reference.md](reference/arm_reference.md). That's the
measurement sheet of the simulated arm (dimensions, tool pose at test poses, limits, masses, gripper).

## Install (on the machine connected to the arm)

That laptop gets its own copy: `scripts/export_sim2real_repo.sh <folder>` exports this folder plus the
simulation it needs (`drawer_rl/`, `configs/`, the MJCF) as a standalone repo. Install with `pip3 --user`,
no sudo or virtualenv needed. Full steps are in the exported README ([STANDALONE_README.md](STANDALONE_README.md)):

```bash
pip3 install --user -r requirements.txt          # leaves kortex_api / protobuf alone; numpy kept < 2
```

Tested with Python 3.10, kortex_api 2.6.0 and protobuf 3.12.4 / 3.20.3. Protobuf 3.5.1, the version Kortex 2.6
pins, can't be imported on Python 3.10; the exported README says what to do.

Connecting works like the lab's `kinova/manual/demo.py`: same `DeviceConnection` helper
([kortex_utilities.py](kortex_utilities.py)), with `--ip`, `-u` and `-p` arguments that default to `admin` / `admin`.
Put them before the command: `python3 -m sim2real.mirror -u admin -p <password> info`.

The arm's address is `robot.ip` in [mirror.yaml](mirror.yaml): `192.168.1.10` over USB, otherwise the Ethernet IP.
Run everything from the repo root: `python3 -m sim2real.mirror ...`. (On this training machine, the training venv
has protobuf 7, so `--real fake` works here but the Kortex API doesn't.)

## Recommended order (first session)

1. **Dry run without hardware:** add `--real fake` to any command. A simulated stand-in arm with deliberate
   differences (1-step latency, 0.9 speed gain, 0.5° encoder offset on joint 5, softer servo) takes the
   real arm's place, so every report has something to find.
   ```bash
   python -m sim2real.mirror --real fake drive joint_sweep
   ```
2. `python -m sim2real.mirror info`: check the reported tool transform and speed limits.
3. `python -m sim2real.mirror fk-check`: no motion. Expect < 2 mm and < 0.5°. If one joint's poses are far off,
   that joint's zero or direction differs between Kortex and the MJCF.
4. `python -m sim2real.mirror follow --viewer`: jog the arm slowly with the Xbox controller.
   * Lower the gripper toward the table and note the real height when the sim first reports a table contact.
     That checks the table/base model.
   * Hold still in a few poses to log real vs sim gravity torque.
5. `python -m sim2real.mirror goto home`, then the motions, slowest first. Keep a hand on the e-stop:
   ```bash
   python -m sim2real.mirror --set safety.speed_scale=0.3 drive gripper
   python -m sim2real.mirror --set safety.speed_scale=0.3 drive steps --param "joints=[0]"
   python -m sim2real.mirror drive steps
   python -m sim2real.mirror drive joint_sweep
   python -m sim2real.mirror --set loop.command_mode=feedforward drive joint_sweep
   python -m sim2real.mirror drive cartesian
   python -m sim2real.mirror goto rl_start && python -m sim2real.mirror drive pull --start rl_start
   ```
6. `python -m sim2real.report sim2real/logs/<stamp>_drive_<motion>.npz` for each log.

## Motions (`--param name=value` overrides the defaults in mirror.yaml)

| Motion | What it does | Look at |
|---|---|---|
| `joint_sweep` | Each joint alone: ±20° at 20°/s (× speed_scale) | Sign/direction, speed gain, lag, sag |
| `steps` | ±5° reference jumps per joint, held 1.2 s | Latency, rise time, overshoot, steady-state error |
| `cartesian` | TCP straight lines ±8 cm in x, y, z, orientation held | Whole-arm behaviour along policy-like paths |
| `pull` | From `rl_start`: forward 12 cm, close gripper, back 15 cm, open | The drawer motion without a drawer |
| `gripper` | Close/open twice, arm still | Close time (sim assumes ~0.67 s) and end positions |

## Command modes (`loop.command_mode`)

* `tracking` (default): both arms get joint **speeds** `v_ref + kp·(q_ref − q_own)` (kp = 2/s), limited in speed
  **and acceleration** to the arm's own limits (`info` reads them). The sim integrates its speed into its servo
  target like Kortex does. Remaining real−sim differences are true response differences.
* `feedforward`: both arms get only the reference speed, with no feedback. Raw speed gain, delay and drift show
  up clearly.
* `deploy`: the deploy loop's mapping (step `dq`, speed = `dq / servo_tau`, no acceleration limit). On the real arm
  this **oscillated on joints 2–3** in the first lab session, see
  `docs/lab_results_2026-11-17.md` in the drawerRL project (logs on branch `lab-results-2026-11-17-1`).

The loop runs on wall time: the sim advances by the time that really passed (the real loop ran at ~13 Hz,
not 20). A motion whose reference needs more joint acceleration than the arm allows is refused. Joint 2 and 3
allow only 28.6 and 22.9 °/s²; lower `safety.speed_scale` if that happens.

## Safety built in

* Every motion is first checked kinematically in sim: joint margin ≥ 10°, no table or self contact, TCP ≥ 5 cm
  above the table. Unsafe motions are refused.
* The real arm must already be at the motion's start pose (`goto` first).
* A typed `go` is required before the real arm moves. `--yes` skips it; don't use that on the real arm.
* The run aborts when:
  * the real arm is > 12° from the reference or from the sim arm;
  * a joint gets within 4° of its sim limit;
  * a torque exceeds 1.2× the sim torque limit;
  * the arm reports a fault;
  * you press Ctrl+C.

  Every exit path sends zero speeds and `Base.Stop()`.
* This is **not** a replacement for the e-stop. Kortex keeps its own limits and protection zones.

## Files

* `sim_arm.py`: the simulated arm (kinematic and dynamic use)
* `real_arm.py`: `KortexArm` (Kortex API, written like the lab demo.py) and `FakeArm` (dry-run stand-in)
* `kortex_utilities.py`: the lab's `DeviceConnection` helper
* `trajectories.py`: test motions and the sim pre-check
* `mirror.py`: the CLI
* `report.py`: plots and summary
* `arm_reference.py`: the measurement sheet
* `mirror.yaml`: settings
* `logs/`: `<stamp>_<mode>.npz` (+ `.json` metadata, `.png`, `_summary.md`)
* `tests/test_sim2real.py` (in the repo's `tests/`): dry-run tests
