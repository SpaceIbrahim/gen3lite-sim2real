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

Tested with **Python 3.10** and **kortex_api 2.6.0** (the lab laptop's versions):

```bash
git clone https://github.com/SpaceIbrahim/gen3lite-sim2real.git && cd gen3lite-sim2real
pip3 install --user -r requirements.txt
```

`requirements.txt` doesn't touch the Kortex API or protobuf. It keeps numpy below 2 so ROS 2 Python modules on the
same machine keep working.

Check that Kortex imports with this `python3`:

```bash
python3 -c "import google.protobuf as p, kortex_api; from kortex_api.autogen.messages import Base_pb2; print('protobuf', p.__version__, 'OK')"
```

* If that prints `OK`, you're done. The lab `demo.py` already works with this setup, so the mirror will too.
* If it fails with `module 'collections' has no attribute 'MutableMapping'`, the protobuf it loads really is
  **3.5.1**, which is too old for Python 3.10. Kortex 2.6 asks for exactly 3.5.1 in its metadata, but it works
  with newer 3.x. Tested with **3.12.4 and 3.20.3**:
  ```bash
  pip3 install --user "protobuf==3.20.3"     # pip warns that kortex-api wants 3.5.1: ignore that
  ```
* No Kortex at all? Get `kortex_api-2.6.0.post3-py3-none-any.whl` from Kinova (link in the README of
  github.com/Kinovarobotics/kortex), then:
  ```bash
  pip3 install --user kortex_api-2.6.0.post3-py3-none-any.whl deprecated "protobuf==3.20.3"
  ```
* If pip refuses with "externally-managed-environment" (Ubuntu 23.04+), add `--break-system-packages`.
  With `--user` this still only touches your home directory.
* Without Kortex everything still works with `--real fake`, a simulated stand-in arm.

Check the install without the robot:

```bash
python3 -m pytest tests -q                        # dry-run + Kortex message tests, ~10 s
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
