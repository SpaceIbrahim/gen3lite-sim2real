# Gen3 Lite sim-to-real mirror

Runs a real Kinova **Gen3 Lite** and its MuJoCo simulation side by side to find where they differ: kinematics,
joint limits, lag, speed gain, sag, gripper timing and torques. The simulated arm is the exact model a
drawer-opening RL policy was trained on (MJCF, actuator settings, table, controller time constants).

This repo is exported from the main drawer-RL project (`scripts/export_sim2real_repo.sh` there). It contains only
what the mirror needs:
* `sim2real/`: the tools
* `drawer_rl/`: simulation code
* `configs/`: world and robot configs
* `model/kinova_gen3_lite/`: MJCF from MuJoCo Menagerie, Kinova BSD license

## Install (no sudo, no virtualenv)

Python 3.10 is the tested version. Install everything into your user site-packages:

```bash
git clone <this repo> && cd <repo>
pip3 install --user -r requirements.txt
```

Then install the **Kinova Kortex API** wheel, which is not on PyPI. Download `kortex_api-2.x...-py3-none-any.whl`
(the link is in the README of github.com/Kinovarobotics/kortex), then:

```bash
pip3 install --user kortex_api-*.whl
pip3 install --user "protobuf==3.20.3"      # Kortex needs protobuf 3.x; run this AFTER the wheel
```

Notes:
* `protobuf==3.20.3` in your user site-packages overrides any newer system protobuf for **your** Python only.
  If another tool of yours needs protobuf 4+, it will see 3.20.3 too.
* On Python 3.11+, if Kortex complains about protobuf descriptors, run with
  `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`.
* If pip refuses with "externally-managed-environment" (Ubuntu 23.04+), add `--break-system-packages`.
  With `--user` this still only touches your home directory.
* Without the Kortex wheel everything still works with `--real fake`, a simulated stand-in arm.

Check the install without the robot:

```bash
python3 -m pytest tests -q                        # 4 dry-run tests, ~10 s
python3 -m sim2real.mirror --real fake drive joint_sweep
python3 -m sim2real.report sim2real/logs/<newest>_drive_joint_sweep.npz
```

## Connect to the arm

Connection works like the lab's `kinova/manual/demo.py` and `utilities.py` (same `DeviceConnection` helper,
`--ip`, `-u` and `-p` arguments, default `admin` / `admin`):

```bash
python3 -m sim2real.mirror info                          # 192.168.1.10, admin/admin
python3 -m sim2real.mirror --ip 192.168.1.10 -u admin -p <password> info
```

The default IP is `robot.ip` in `sim2real/mirror.yaml`: `192.168.1.10` over USB, otherwise the arm's Ethernet IP.
`goto home` uses the arm's stored **Home** action, like `example_move_to_home_position` in the lab demo.
Close the Kinova Web App's Cartesian/joystick control (and any ROS driver) while the mirror sends commands.

## Use

See [sim2real/README.md](sim2real/README.md) for every command, the test order, the motions and the safety checks.
In short:

```bash
python3 -m sim2real.mirror info                     # the arm's limits / tool vs the sim
python3 -m sim2real.mirror fk-check                 # kinematics, NO motion
python3 -m sim2real.mirror follow --viewer          # jog with the Xbox controller; the sim copies it
python3 -m sim2real.mirror goto home
python3 -m sim2real.mirror --set safety.speed_scale=0.3 drive steps --param "joints=[0]"
python3 -m sim2real.mirror drive joint_sweep
python3 -m sim2real.report sim2real/logs/<log>.npz
python3 -m sim2real.arm_reference                   # regenerate the sim measurement sheet
```

Bring the `sim2real/logs/` folder back to the training machine to analyse it there too.
Keep a hand on the e-stop whenever the real arm moves.
