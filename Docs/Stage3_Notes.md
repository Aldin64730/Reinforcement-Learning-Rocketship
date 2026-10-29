# Rocket Lander: Stage 3 notes (wind: steady wind + gusts + turbulence)

Builds on [Stage2_Notes.md](Stage2_Notes.md). Observations (22) and actions (3) are unchanged, and **the wind is not observed**. The policy is warm-started from Stage 2.

## 1. Wind model (`WindField` + `RocketController.ApplyAerodynamics`)
Every episode draws a new wind condition:
- **Steady wind:** speed uniform in [0, wind_speed_max], from a uniformly random horizontal direction.
- **Gusts:** random events, about 6 per minute, each lasting 1-3 s. Each has a smooth ½(1 − cos) profile and a peak of 0.3-1.0 × gust_speed_max, in the mean direction ± 45°.
- **Turbulence:** Perlin noise per axis with amplitude turbulence_intensity. The vertical component is 30% of that.

The wind acts through quadratic drag on the airspeed v_air = v_rocket − v_wind, split into an axial part (k = 1.5 N/(m/s)²) and a lateral part (k = 9 N/(m/s)²). The force is applied at the centre of pressure, 0.5 m above the centre of mass, so a crosswind **pushes and tilts** the rocket.

**Why the observations are unchanged:**
- A real lander does not measure the wind field exactly.
- The policy can still infer the wind indirectly from drift (velocity) and from the attitude it needs to hold.
- Keeping 22 observations allows transfer from Stage 2. Adding wind observations or memory would change the network input and force training from scratch, so those are planned as ablations.

## 2. Curriculum and calibration
Spawn conditions are fixed at the Stage 2 final lesson for the whole stage: 10-35 m high, ≤ 20 m offset, ≤ 30° tilt, ≤ 5 m/s, ≤ 0.4 rad/s. Only the wind grows.

| Lesson | Steady wind (max) | Gust peak (max) | Turbulence | Threshold |
|---|---|---|---|---|
| W0 | 0 | 0 | 0 | 25.0 |
| W1 | 2.5 m/s | 1.5 m/s | 0.4 m/s | 24.5 |
| W2 | 5 m/s | 2.5 m/s | 0.75 m/s | 24.0 |
| W3 | 7.5 m/s | 4 m/s | 1.1 m/s | 25.5 (≈ 95% landings) |
| W4 (final) | 10 m/s | 5 m/s | 1.5 m/s | - |

**Why W4 is not stronger.** The first proposal (12 + 8 + 2 m/s, peaks about 20 m/s) was rejected after testing:
- The peak wind torque, 9·20²·0.5 ≈ 1.8 kN·m, equals the side-pod control torque (about 2 kN·m).
- The tipping moment on a landed rocket, 3.6 kN × 3.6 m ≈ 13 kN·m, exceeds the gravity restoring moment, m·g·r_leg = 9810 × 1.14 ≈ 11.2 kN·m.
- Neither controller succeeded with a mean wind above 12 m/s.

The final W4 peaks at about 15-17 m/s, which uses about 55% of the control authority. A landed rocket with the engine off stays put even at 12 m/s steady plus 5 m/s gusts (17.4 m/s peak): 880/880 episodes.

**Calibration** (16 rockets, Stage 2 final spawn conditions; success rate):

| Lesson | PID autopilot | Stage 2 policy (zero-shot) | Stage 2 policy, success by mean wind band |
|---|---|---|---|
| W0 | 97.1% | 99.0% | – |
| W1 | 97.1% | 99.3% | – |
| W2 | 97.3% | 99.0% | 0-4 m/s 99%, 4-8 m/s 100% |
| W3 | 97.7% | 94.3% | 0-4 m/s 98%, 4-8 m/s 90% |
| W4 | 76.6% (8+ m/s: 15%) | **72.8%** | 0-4 m/s 97%, 4-8 m/s 88%, **8+ m/s 5%** |

- The Stage 2 policy's main failure in strong wind is `Timeout`: it hovers near the pad and cannot settle.
- Mean returns of the Stage 2 policy: 30.8 / 30.2 / 28.1 / 25.2 / 18.7 for W0-W4.

**Baseline change.** The PD autopilot was extended to a **PID** with a horizontal-position integral term (gain 0.1, active only within 3 m of the pad, limit 3 m/s²). The pure PD settles about F/(m·k_p) ≈ 3-4 m downwind and never lands in steady wind. Tuning results:

| Autopilot | No wind | W4 |
|---|---|---|
| PD | 98.7% | 65.6% |
| PID, integral active within 3 m | 99.1% | 84.1% |
| PID, integral active within 8 m | 87% | – (wind-up: overshoot and hovering) |
| PID, gain 0.2 | 51% | – |

## 3. Training

```powershell
.\.venv\Scripts\python -m mlagents.trainers.learn config/rocket_lander_stage3.yaml --run-id=stage3_v1 --initialize-from=stage2_v1 --torch-device=cpu
```
Same hyperparameters as Stage 2 (γ = 0.995, time_horizon 256, 20 M steps). Reward v2 is unchanged.

**New TensorBoard stats:**
- `Wind/MeanSpeed`, `Wind/MaxSpeed`
- `WindSuccess/0-4 m_s`, `WindSuccess/4-8 m_s`, `WindSuccess/8-12 m_s`, `WindSuccess/12+ m_s`: the landing rate in each mean-wind band

Showcase CSVs gain the columns `wind_mean`, `wind_max` and `wind_dir_deg`. The showcase HUD shows the wind, and a blue arrow shows its direction and strength during the replay.

**What to expect:**
- W0-W2 pass quickly, since the Stage 2 policy already handles them.
- W3 requires about 95% landings.
- The main learning happens on W4, in the 8+ m/s band. Watch `WindSuccess/8-12 m_s` and `Outcome/Timeout`.

## 4. Thesis material

**Methods draft (Stage 3).** Stage 3 adds stochastic wind as an unobserved disturbance: a steady component of random direction and speed, gust events, and Perlin-noise turbulence. Its effect enters through quadratic aerodynamic drag on the airspeed, applied at a centre of pressure above the centre of mass, so crosswind both translates and rotates the vehicle. The policy observes no wind information and must infer the disturbance from its own drift and attitude. This keeps the observation space identical to Stage 2 and allows warm-starting from the Stage 2 policy. Wind strength is increased over five curriculum lessons up to a 10 m/s steady wind, 5 m/s gusts and 1.5 m/s turbulence. This limit was chosen from a control-authority analysis (peak wind torque ≈ 55% of the attitude-thruster torque) and a static-stability check of the landed vehicle. Recoverability and baselines were measured beforehand with a PID controller (an integral term was required to counter steady wind) and with the Stage 2 policy without wind training.

**What to record:**
- Learning curves: `Rocket/SuccessRate`, the `WindSuccess/*` bands, `Landing/Distance`, `Outcome/Timeout`, and the lesson number.
- **Robustness curve:** success rate and landing distance against mean wind speed, for PID vs. the Stage 2 policy vs. the Stage 3 policy. Use the showcase CSVs, e.g. 500 episodes per policy at W4.
- Landing-error direction relative to the wind direction. Do landings drift downwind? Use the CSV `wind_dir_deg`.

**Ablation ideas:**
- **C1, privileged wind observation:** add the wind vector as 3 extra observations (from scratch). This gives an upper bound on what wind knowledge is worth.
- **C2, memory:** stacked observations (3-5) or LSTM (from scratch). Can the policy estimate the wind itself?
- **C3, transfer vs. scratch, and curriculum vs. direct W4.**
- **C4, domain randomisation for robustness:** train at W4 and test at stronger wind (12 + 8 m/s) for out-of-distribution behaviour.

## 5. Results: run `stage3_v1` (20 M steps, about 163k episodes)

**Curriculum progress:**
- W0 → W3 within the first 0.1 M steps, since the Stage 2 policy already handled them.
- **Stuck on W3 from 0.1 M to 11.3 M steps.** The W3 threshold of 25.5 was calibrated on the *deterministic* Stage 2 policy (mean return 25.2). During training the policy is *stochastic*, so its returns are lower (20-24.5). The threshold was therefore barely reachable, and W4 received only 8.7 M of the 20 M steps.
- *Lesson for the thesis:* calibrate curriculum thresholds on the stochastic training policy, or use a lower margin.

**Training curve on W4** (stochastic policy):
- Overall success 76-87%.
- Success in the 8-12 m/s wind band rose from about 0.24 to 0.56 over the last 8.7 M steps and was still rising, so the run had not converged.

**Evaluation at W4** (showcase scene, deterministic policy; conditions as the final training lesson):

| Policy | n | Success | 0-4 m/s | 4-8 m/s | 8-12 m/s | Landing distance (mean) | Touchdown speed | Time | Crashes |
|---|---|---|---|---|---|---|---|---|---|
| **Stage 3 PPO** | 1500 | **83.4%** | 95.3% | 92.8% | **45.6%** | 1.74 m | 0.93 m/s | 10.0 s | 3.9% |
| Stage 2 PPO (zero-shot) | 600 | 71.0% | 100% | 77.9% | 7.0% | 1.55 m | 0.57 m/s | 9.9 s | 0.5% |
| PID autopilot | 500 | 77.4% | 98.3% | 96.9% | 16.8% | 0.79 m | 0.87 m/s | 23.1 s | 0% |

Wind bands are by mean wind speed in the episode. Files: `Evaluation/*stage3_v1_20261029_*`, `Evaluation/*stage2_v1_20261029_*` and `Evaluation/*heuristic_Autopilot_20261029_*`.

**Interpretation:**
- Wind training improved strong-wind success 6.5× (7% → 46%) and overall success by 12 points. It beats the PID baseline overall and especially in strong wind, while landing about 2× faster.
- **Remaining failures:** `Timeout` (8.9%, hovering in strong wind), `Crashed` (3.9%) and `OffPad` (3.7%).
- **Trade-offs:**
  - Slightly worse in calm air than Stage 2 (95% vs. 100%).
  - Firmer touchdowns in calm air (1.1 vs. 0.6 m/s).
  - More crashes: the policy became more aggressive.
- **Precision drops with wind speed:** 0.85 → 2.06 → 3.65 m from the centre. The memoryless policy cannot estimate the steady wind, so it lands downwind of the centre. The PID's integral term avoids this, holding 0.7-0.8 m at all wind speeds. This directly motivates ablation C2 (memory or stacked observations) and C1 (privileged wind observation).
- **Next step taken:** training continued on W4 (`--resume`, max_steps 2.0e7 → 3.5e7). See section 6.

## 6. Results after continued training (`stage3_v1`, resumed to 35 M steps, about 292k episodes in total)

**Training on W4, 20 M → 35 M steps** (stochastic policy):
- The 8-12 m/s band rose from about 0.56 to 0.65-0.76.
- `Outcome/Timeout` fell from about 8% to about 0-2%. The policy stopped hovering and commits to the landing.
- `Outcome/Crashed` rose to about 3-7%.

The final model is saved as `Assets/RocketLander/Models/RocketLander_stage3_v1_35M.onnx`. The 20 M model is kept as `RocketLander_stage3_v1.onnx`.

**Evaluation at W4** (deterministic; file `Evaluation/RocketShowcase_RocketLander_stage3_v1_35M_20261029_193435.csv`):

| Policy | n | Success | 0-4 m/s | 4-8 m/s | 8-12 m/s | Landing distance | Touchdown speed | Main failures |
|---|---|---|---|---|---|---|---|---|
| **Stage 3 PPO @ 35 M** | 1000 | **90.0%** | 95.9% | 94.5% | **72.8%** | 1.71 m | 0.83 m/s | Crashed 5.5%, OffPad 3.6% |
| Stage 3 PPO @ 20 M | 1700 | 83.1% | 95.3% | 92.5% | 44.5% | 1.72 m | 0.93 m/s | Timeout 9.1%, Crashed 4.1% |
| Stage 2 PPO (zero-shot) | 600 | 71.0% | 100% | 77.9% | 7.0% | 1.55 m | 0.57 m/s | Timeout 22% |
| PID autopilot | 500 | 77.4% | 98.3% | 96.9% | 16.8% | 0.79 m | 0.87 m/s | Timeout 21% |

**Failure analysis @ 35 M:**
- **Crashed (55/1000).** These are spread over all wind bands (15 / 20 / 20), so they are *not* primarily caused by wind. They come from higher spawn tilt (mean 19.8° vs. 15° overall), reach a 44° peak tilt, and hit at about 8 m/s after about 8 s. This is the aggressive high-tilt recovery already seen in Stage 2 (0.4% there), now more frequent. Wind training made the policy more aggressive.
  - Possible fix: a larger crash penalty or a stronger descent-envelope penalty (a reward v3 ablation).
- **OffPad (36/1000).** All of these are in the 8-12 m/s band. They are soft (0.4 m/s) but 4.6 m from the centre, i.e. blown past the pad edge. This is the same downwind-drift limitation as the precision numbers (0.81 → 1.76 → 3.47 m by wind band).
  - Possible fix: memory (stacked observations / LSTM) or a privileged wind observation (ablations C1/C2).

**Conclusion for Stage 3:**
- Training in wind raised success at the hardest conditions from 71% (Stage 2 policy) to 90%.
- In strong wind it went from 7% to 73%, which is 4.3× better than the PID baseline in that band.
- The remaining gap is precision and safety in strong wind, which a memoryless, wind-blind policy can only partly solve.
