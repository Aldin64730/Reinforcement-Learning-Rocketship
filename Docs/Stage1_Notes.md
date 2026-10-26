# Rocket Lander: Stage 1 notes (setup, design, thesis material)

Unity 6000.6.6f1, ML-Agents Unity package `com.unity.ml-agents` 4.1.0, Python `mlagents` 1.1.0 (PPO).

---

## 1. Install the Python side (once)

ML-Agents' Python trainer needs **Python 3.10** (3.10.1 to 3.10.12). Your system Python is 3.14 and will not work, so install 3.10 next to it:

```powershell
winget install --id Python.Python.3.10 -e
cd "C:\Users\Matrix\Reinforcement Learning Rocketship"
py -3.10 -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install torch~=2.2.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python -m pip install mlagents==1.1.0
.\.venv\Scripts\python -m pip install "setuptools==69.5.1" "torch~=2.2.1" --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
.\.venv\Scripts\python -m mlagents.trainers.learn --help
```

- **setuptools must be < 81.** mlagents 1.1.0 imports `pkg_resources`, which newer setuptools removed. The error is `ModuleNotFoundError: No module named 'pkg_resources'`.
- **torch must stay at 2.2.x.** Installing mlagents can pull in the newest torch. From torch 2.6 on, `torch.load` rejects ML-Agents checkpoints, which breaks `--resume` and `--initialize-from`. Check with `.\.venv\Scripts\python -m pip list` (tested: torch 2.2.2+cpu, numpy 1.23.5, setuptools 69.5.1).

- **CPU PyTorch on purpose.** A 2x256 MLP trains as fast on the CPU, and the torch 2.2 CUDA wheels predate RTX 50-series (Blackwell) support.
- If `pip install mlagents` fails building `grpcio`, run `.\.venv\Scripts\python -m pip install grpcio==1.48.2` first, then retry.
- You can activate the venv with `.\.venv\Scripts\Activate.ps1`. If PowerShell blocks scripts, call the venv Python directly as shown above. Windows Smart App Control blocks the generated `mlagents-learn.exe` / `tensorboard.exe` launchers, so always use `python -m mlagents.trainers.learn` and `python -m tensorboard.main`.

## 2. Scene setup (already done in the Editor; repeat these steps to rebuild it by hand)

Scene: `Assets/RocketLander/Scenes/RocketTraining.unity`. Your original `SampleScene`, `Rocketship.prefab` and `LandingPlatform.prefab` are untouched.

```
TrainingArea            RocketEnvironment, WindField          (prefab: Assets/RocketLander/Prefabs/TrainingArea.prefab)
├─ Ground               Cube 100 x 1 x 100 m, top surface at y = 0
├─ LandingPad           your pad, flattened to 0.5 m (scale z 100 -> 25), LandingPad component, BoxCollider
└─ Rocket               Rigidbody, RocketController, BehaviorParameters, RocketAgent, DecisionRequester
   ├─ Model             all Blender meshes, rotated 22 deg about Y so the pods lie on local +X/-X/+Z/-Z (meshes have no colliders)
   ├─ Colliders         Hull (capsule r 1.0, 7.2 m), MainNozzle (box), PodFront/Back/Left/Right (capsules r 0.25),
   │                    FootFront/Back/Left/Right (spheres r 0.15, PhysicsMaterial RocketFoot: friction 0.9, bounce 0)
   └─ Thrusters         Main (0, -3.19, 0), Front (0, -0.6, 1.31), Back (0, -0.6, -1.40), Left (-1.38, -0.6, 0), Right (1.40, -0.6, 0)
                        each with a Flame child (scaled with thrust; visual only)
```

- 16 copies of the prefab sit on a 4 x 4 grid 100 m apart. Out-of-bounds is 30 m, so rockets never reach a neighbouring area.
- No tags are needed: the pad is recognised by its `LandingPad` component.
- Project Settings > Player > **Run In Background = on**. Without it the Editor pauses Play mode (and training) when it loses focus.

### Component values (they must match the code)

| Component | Field | Value |
|---|---|---|
| Behavior Parameters | Behavior Name | `RocketLander` (must match the YAML) |
| | Vector Observation > Space Size | **22** (`RocketAgent.ObservationSize`) |
| | Stacked Vectors | 1 |
| | Actions > Continuous Actions | **3** (`RocketAgent.ContinuousActionCount`) |
| | Actions > Discrete Branches | **0** |
| | Behavior Type | Default (trains when Python is connected, else runs the Heuristic) |
| | Model | none (later: the trained `.onnx`) |
| Rocket Agent | Max Step | **0**: time limit is `max_episode_seconds` |
| Decision Requester | Decision Period | **5** (0.1 s; 10 decisions per second) |
| | Take Actions Between Decisions | **on** (forces applied every physics step) |
| Time | Fixed Timestep | 0.02 s (50 Hz physics) |

`RocketAgent.Initialize()` logs an error if any of these drift out of sync.

### Testing without training
Press Play with no trainer running. Behavior Type Default with no model falls back to the **Heuristic**:
- `Autopilot` (default) flies the PD baseline.
- `Keyboard`: W/S throttle, Space full, X cut, arrow keys tilt.

The first area shows a HUD. Tick `Log Episodes` on a RocketAgent to print every outcome.

## 3. Design (with the reasoning for the thesis)

### Physics model (`RocketController`)
- Mass 1000 kg; centre of mass 0.5 m below the pivot (`centerOfMass`). The inertia tensor is set explicitly as a solid cylinder (r = 1 m, h = 7.6 m), not left to Unity's collider-based automatic estimate, so it is reproducible and easy to report.
- **Main engine** at the bottom on the axis: thrust up to TWR 2.0 (hover throttle 0.5). The throttle follows the command at most 4 units/s (0 to 100% in 0.25 s) to model engine spool-up.
- **Side pods** fire downward, each up to 0.15 of the weight, at about 1.35 m from the axis. Differential thrust gives about 2 kN·m of pitch/roll torque (about 22 deg/s²). All thrust uses `Rigidbody.AddForceAtPosition` at the nozzle positions, so torque comes from the real lever arms rather than a scripted `AddTorque`.
- **No yaw actuator.** The rocket and its legs are rotationally symmetric, so yaw does not affect landing. Spawn yaw is fully random and all observations are in the body frame, which makes the policy yaw-invariant.
- **Fuel:** one shared tank (60 units). Main engine burns 4 units/s at full throttle, each pod 0.4 units/s. When the tank is empty, all thrust stops. Mass stays constant (simplification).
- **Aerodynamics:** quadratic drag on the airspeed (velocity minus wind), split into axial (k = 1.5) and lateral (k = 9) parts. It acts at a centre of pressure 0.5 m above the centre of mass. With no wind (Stages 1-2) this is ordinary air drag. In Stage 3 the same code turns wind into a push plus a tilting torque, with no new physics code.

### Actions: 3 continuous values in [-1, 1]
| # | Meaning | Mapping |
|---|---|---|
| 0 | main throttle | throttle = (a+1)/2 |
| 1 | pitch | a > 0 fires the Back pod (tilts towards +Z), a < 0 the Front pod |
| 2 | roll | a > 0 fires the Right pod (tilts towards -X), a < 0 the Left pod |

Why 3 mixed actions rather than 5 raw throttles: each pod can only push, so for every axis exactly one pod of the pair should fire. Exposing "desired torque" removes redundant combinations (e.g. front and back both firing) that the policy would otherwise have to learn to avoid. This shrinks the search space while keeping physically real forces.

### Observations: 22 floats, mostly in the rocket's body frame, fixed scales, clipped to [-1, 1]
| Obs | Size | Normalisation | Why |
|---|---|---|---|
| world up in body frame | 3 | unit vector | tilt without Euler angles: no 359 deg / 0 deg wrap and no gimbal lock, smooth and yaw-invariant |
| linear velocity (body) | 3 | / 25 m/s | aligned with the actuators |
| angular velocity (body) | 3 | / 5 rad/s | needed for damping (the D in PD) |
| altitude of feet above pad | 1 | / 60 m | the central landing variable |
| vertical speed (world) | 1 | / 25 m/s | redundant but very informative; saves the network a rotation |
| vector feet to pad centre (body) | 3 | / 60 m | direction to fly, in actuator coordinates |
| horizontal distance to pad | 1 | / 60 m | precision |
| fuel fraction | 1 | 0..1 | resource awareness |
| actual main throttle | 1 | 0..1 | differs from the command (spool lag) |
| previous actions | 3 | -1..1 | current control state; reduces jitter |
| feet on pad / 4 | 1 | 0..1 | touchdown sensor: tells the agent to cut the engine |
| elapsed time / limit | 1 | 0..1 | makes the time limit observable (Markov), needed because the timeout is penalised |

- **Scales are fixed and sized for Stage 3.** A policy carries over between stages only if its inputs keep their meaning. For the same reason `normalize: false` in the YAML: running normalisation statistics would drift when the curriculum changes the spawn distribution.
- **Not observed:** wind (Stage 3 trade-off: observing it makes the task easier but less realistic; leaving it unobserved forces a reactive, robust policy). There are also no ray sensors: the pad position is known exactly, and rays would only add cost and noise.

### Reward (`RewardSettings`, all coefficients in the Inspector)
Per physics step:
1. **Potential-based shaping** r = Φ(s′) − Φ(s), with
   Φ = −0.3·dist − 0.1·alt − 0.3·speed − 2.0·tilt[rad] + 1.0·(feet on pad / 4).
   Over an episode the sum telescopes to Φ(s_T) − Φ(s_0). Loops such as touching the pad and lifting off again therefore earn nothing (Ng, Harada & Russell 1999). Gymnasium's LunarLander uses the same construction.
2. **Dense penalties** (per second, scaled by dt):
   - time −0.05
   - pitch/roll rate −0.05 per rad/s
   - **descent envelope** −0.5 per m/s of descent speed above v_safe(h) = 1.0 + 0.5·h, i.e. "fast is fine when high, slow down near the ground"
   - fuel −0.02 per unit burned
3. **Terminal rewards:**
   - landed: +10, plus 5·(1 − dist / 5 m) precision bonus
   - crash or tipped over: −10
   - off-pad touchdown: −6
   - out of bounds: −10
   - timeout: −5

**Landed** means: at least 3 feet on the pad, speed < 0.5 m/s, tilt < 10°, pitch/roll rate < 0.3 rad/s and main throttle ≤ 0.4, all held for 1 s.
**Crash** means: any non-foot collider touches anything, or a foot impact > 3 m/s.
**Tipped over** means tilt > 80°.

Reward-hacking risks and what blocks them:
- **Hovering forever:** time penalty + finite fuel + timeout penalty, with the time fraction observed. Shaping gives no reward for staying put.
- **Farming shaping:** impossible, because potential-based shaping telescopes.
- **Suicide** (crashing early to stop accumulating penalties): crash −10 is far larger than any penalty that could be avoided (about −0.1/s), and a landing earns about +15.
- **Landing at the edge / bouncing:** the precision bonus and the 1 s stable-hold requirement.
- **Hovering on the pad with the engine on:** the throttle ≤ 0.4 condition. The legs must carry the weight.
- **Nuance to mention:** Φ(s_T) is not forced to 0, so shaping also acts as a small terminal bonus for ending near, slow and upright. The speed in Φ(s_T) is measured after the collision, so a crash "refunds" the speed term. This is harmless here because −10 dominates, but it is worth reporting (see ablation A2).

## 4. Training

```powershell
.\.venv\Scripts\python -m mlagents.trainers.learn config/rocket_lander_stage1.yaml --run-id=stage1_v1 --torch-device=cpu
```
When it says "Start training by pressing the Play button", press **Play** in Unity. Ctrl+C stops training and saves `results/stage1_v1/RocketLander.onnx`. Add `--resume` to continue a run, or `--force` to overwrite one.

**Faster: build an executable.** File > Build Profiles > Windows. Untick Development Build. Build to `Builds/RocketLander/RocketLander.exe`. Then run:
```powershell
.\.venv\Scripts\python -m mlagents.trainers.learn config/rocket_lander_stage1.yaml --run-id=stage1_v1 --torch-device=cpu --env=Builds/RocketLander/RocketLander.exe --num-envs=4 --no-graphics
```
16 areas x 4 processes = 64 rockets in parallel. Rebuild the exe after every C# or scene change.

**Key hyperparameters** (`config/rocket_lander_stage1.yaml`):
- `batch_size` 1024 / `buffer_size` 20480: large batches stabilise continuous-control gradients. The buffer is about 20 batches of fresh on-policy data per update.
- `learning_rate` 3e-4, linear decay.
- `beta` 5e-3: entropy bonus. Too low and it gets stuck hovering; too high and it stays twitchy.
- `epsilon` 0.2: PPO clip range.
- `lambd` 0.95: GAE.
- `num_epoch` 3.
- `gamma` 0.99 per decision (0.1 s), so the effective horizon is about 10 s, matching episode length.
- `time_horizon` 128.
- Network: 2 x 256.

**Curriculum** (3 lessons, all spawn parameters advance together when the smoothed reward reaches 10, i.e. about 80% landings):

| Lesson | spawn height (m above pad) | horizontal radius | tilt | initial speed |
|---|---|---|---|---|
| Hop | 3-6 | 0.5 m | 0° | ≤ 0.5 m/s |
| Low | 5-10 | 1.0 m | 3° | ≤ 1.0 m/s |
| Full | 8-15 | 2.0 m | 5° | ≤ 1.5 m/s |

## 5. Monitoring

```powershell
.\.venv\Scripts\python -m tensorboard.main --logdir results
```
Open http://localhost:6006.

| Metric | Healthy Stage 1 run |
|---|---|
| `Environment/Cumulative Reward` | starts around −10 (crashes), climbs past 0 and settles at +13..+16 |
| `Rocket/SuccessRate` | 0 at first, then towards > 0.9 per lesson (it drops when a lesson advances, then recovers) |
| `Environment/Lesson Number/spawn_height_min` | 0 → 1 → 2 |
| `Outcome/*` | Crashed falls first; Timeout should stay near 0 (if it rises, the agent is hovering) |
| `Landing/TouchdownSpeed`, `Landing/Distance` | falling over time |
| `Reward/Shaping`, `Reward/Dense`, `Reward/Fuel`, `Reward/Terminal` | Terminal should dominate. If Dense is very negative, the agent is getting punished more than it is learning |
| `Policy/Entropy` | slowly decreasing; a crash to near zero early means premature convergence |
| `Losses/Value Loss` | spikes at lesson changes, then decreases |
| `Environment/Episode Length` | short (crashes) → longer (learning to brake) → shorter again (efficient landings) |

**Reference (non-learning) result:** in the Editor test (Stage 1 "Full" spawn ranges), the PD autopilot landed 374 of 374 episodes after a mode switch, with about 0.7-1.1 m/s touchdowns, 0.1-1.5 m from centre, and 6-20 s per episode. It is a good baseline for the thesis.

## 6. Thesis material

### Methods draft (Stage 1)
> We model a vertical-landing rocket in Unity 6 with PhysX rigid-body dynamics at 50 Hz. The 1000 kg vehicle has a throttleable main engine (thrust-to-weight 2.0, rate-limited to model spool-up) and four downward-firing attitude thrusters. Their differential thrust produces pitch and roll torques through their physical lever arms. A single fuel tank supplies all engines. Aerodynamic drag is quadratic in airspeed and acts at a centre of pressure above the centre of mass, which later allows wind disturbances without changing the model. The agent is trained with Proximal Policy Optimization (Unity ML-Agents 4.1, PyTorch) and acts at 10 Hz with three continuous actions: main throttle and pitch/roll torque commands, mapped to the thrusters by a fixed mixer. Observations (22 dimensions) are expressed in the vehicle's body frame with fixed normalisation: attitude as the gravity direction, linear and angular velocity, altitude, relative pad position, fuel, engine state, previous action, leg contact and elapsed time. The reward combines potential-based shaping (distance, altitude, speed, tilt, leg contact), dense penalties for time, fuel, angular rate and exceeding an altitude-dependent safe descent speed, and terminal rewards for landing, scaled by precision, or failure. A landing requires at least three legs on the pad, speed below 0.5 m/s, tilt below 10° and an idle engine, held for one second. Training uses 16 parallel environments and a three-lesson curriculum over spawn height, horizontal offset, tilt and initial velocity, advanced automatically when the mean return exceeds a threshold.

### What to record (Stage 1)
- Learning curves (mean ± std over **3-5 seeds**, `--seed=N`; different seeds use different `--run-id`s): cumulative reward and success rate vs. steps, with lesson changes marked.
- Final evaluation, about 200 episodes per policy with no training. Set the `.onnx` as the Model, Behavior Type = Inference Only, and record the TensorBoard/console stats:
  - success rate and outcome breakdown
  - touchdown speed
  - landing distance (histogram or scatter plot of landing points on the pad)
  - landing tilt, fuel used, time to land
- A trajectory plot of altitude vs. vertical speed for a few episodes against the safe-descent envelope line. This is very explanatory.
- Comparison table: RL policy vs. PD autopilot on the same metrics.

### Custom stats from C#
`Academy.Instance.StatsRecorder.Add("Group/Name", value)` (default aggregation: average per summary period). `RocketAgent.RecordStats()` already logs:
- `Rocket/SuccessRate`, `Rocket/EpisodeSeconds`, `Rocket/FinalDistance`, `Rocket/FuelUsed*`, `Rocket/TouchdownSpeed`
- `Outcome/*` rates
- `Landing/Distance|TouchdownSpeed|TiltDeg|Seconds` (successful landings only)
- `Reward/Shaping|Dense|Fuel|Terminal` (per-episode reward breakdown)

Export the curves from TensorBoard (download CSV) and plot them with matplotlib.

### Ablation ideas
- **A1, reward:** sparse (terminal only) vs. + shaping vs. + descent envelope. This shows why shaping matters.
- **A2, shaping form:** Φ(s_T) = 0 (strictly policy-invariant) vs. the current version.
- **A3, observations:** remove previous actions / throttle state, contact sensor, or time fraction; Euler angles instead of the up vector.
- **A4, action space:** 3 mixed actions vs. 5 raw thruster throttles (needs Continuous Actions = 5 and a different mapping).
- **A5, curriculum:** with vs. without (train directly on "Full").
- **A6, normalisation:** manual fixed scales vs. `normalize: true`.
- **A7, decision period:** 2 / 5 / 10, a control-frequency trade-off.
- **A8, baseline:** RL vs. PD autopilot, including robustness when spawn ranges go beyond the training ranges.

## 7. Pitfalls checklist
- **Physics only in FixedUpdate.** Here, forces are applied in `OnActionReceived`, which ML-Agents calls from FixedUpdate. Never in `Update`.
- **Read physics state from the Rigidbody** (`rb.position/rotation`), not the Transform. With interpolation on, Transforms hold the interpolated render pose.
- **Collision callbacks run after the physics step.** The code records contacts there and decides termination at the start of the next step, never by calling `EndEpisode()` inside `OnCollisionEnter`.
- **Rigidbody sleeping** stops `OnCollisionStay`. `sleepThreshold = 0` prevents that.
- **Centre of mass / inertia:** set explicitly. Unity's automatic values change whenever you edit colliders.
- **Behavior Parameters ↔ code:** 22 observations, 3 continuous actions, and Behavior Name = YAML key.
- **Decision period vs. physics:** with "Take Actions Between Decisions", the last action repeats for 5 physics steps. Rewards are summed over those steps.
- **Normalisation scales:** never change them between stages. Retrain if you must.
- **Time scale:** training runs at time scale 20, but the physics step stays 0.02 s, so the dynamics are identical. Don't raise Fixed Timestep to speed things up.
- **Run In Background** must be on, or keep the Editor in the foreground while training.
- **After changing C#**, stop training, let Unity recompile, then restart with `--resume` (same observation/action sizes) or a new run-id.
