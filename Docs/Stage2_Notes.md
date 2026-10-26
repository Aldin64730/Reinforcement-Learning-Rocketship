# Rocket Lander: Stage 2 notes (random position + bounded tilt)

Builds on [Stage1_Notes.md](Stage1_Notes.md). Observations (22) and actions (3) are unchanged, so the Stage 1 policy can be used as the starting point.

## 1. What changed and why

| Change | Stage 1 | Stage 2 | Reason |
|---|---|---|---|
| Spawn difficulty | 8-15 m, ≤ 2 m offset, ≤ 5° | up to 10-35 m, ≤ 20 m offset, ≤ 30° tilt, ≤ 5 m/s, ≤ 0.4 rad/s | Stage 2 task |
| Precision bonus | 5·(1 − d/5 m) | **8·(1 − d/3 m)** | Stage 1 landed about 1.9 m off-centre: the old bonus was too flat |
| Soft-landing bonus | none | **5·(1 − v_touchdown / 3 m/s)** | Stage 1 touched down at about 1.9 m/s: nothing rewarded going below the 3 m/s crash limit |
| Near-pad distance shaping | 0.3/m | 0.3/m + **0.7/m within 3 m** (potential-based) | stronger pull towards the exact centre |
| γ (discount) | 0.99 | **0.995** | flights are now 10-25 s (100-250 decisions) |
| time_horizon | 128 | 256 | longer episodes |
| Fuel / time limit / bounds / ceiling | 60 / 30 s / 30 m / 40 m | 100 / 45 s / **45 m** / 60 m | longer flights; bounds chosen after the recoverability test |
| Initialisation | random | **Stage 1 policy** (`--initialize-from=stage1_v1`) | transfer: it can already hover and land |

New TensorBoard stats:
- `Spawn/Height`, `Spawn/Distance`, `Spawn/TiltDeg` (the starting conditions)
- `Rocket/MaxTiltDeg` (the largest tilt during the flight)

The showcase scene now also writes every evaluated episode to `Evaluation/*.csv` for per-episode plots.

## 2. Curriculum (calibrated in Unity before training)

| Lesson | Height | Offset | Tilt | Speed | Spin | Threshold (smoothed mean reward) |
|---|---|---|---|---|---|---|
| L0 (= Stage 1 final) | 8-15 m | 2 m | 5° | 1.5 m/s | 0 | 13.0 |
| L1 | 10-20 m | 5 m | 10° | 2 m/s | 0.1 rad/s | 13.5 |
| L2 | 10-25 m | 10 m | 15° | 3 m/s | 0.2 rad/s | 14.0 |
| L3 | 10-30 m | 15 m | 22° | 4 m/s | 0.3 rad/s | 15.0 |
| L4 (final) | 10-35 m | 20 m | 30° | 5 m/s | 0.4 rad/s | - |

**Calibration** (reward v2, 16 rockets, about 90 s of simulated time per lesson):

| Lesson | PD autopilot: success / landing return | Stage 1 policy, no retraining: success / landing return / main failure |
|---|---|---|
| L0 | 100% / 21.8 | 100% / 15.7 |
| L1 | 100% / 23.3 | 79% / 15.8 / OffPad |
| L2 | 100% / 24.1 | 31% / 16.9 / OffPad |
| L3 | 100% / 25.7 | 16% / 17.4 / OffPad |
| L4 | 98.9% (n = 530, bounds 45 m) / 27.4 | 8% / 19.1 / OffPad + Crashed |

How the thresholds were chosen:
- A landing returns about 16-19 when it is Stage-1 quality and about 22-27 when it is precise and soft. Failures return about −4 to −8.
- The thresholds therefore correspond to roughly 66-90% landings, depending on landing quality.

**Recoverability:** at L4 the autopilot's largest tilt during flight was 37°, well below the 80° tip-over limit. Its success by spawn tilt:

| Spawn tilt | Autopilot success |
|---|---|
| 0-10° | 100% |
| 10-20° | 99.4% |
| 20-30° | 97.0% |

The remaining failures are the hardest combination (≈30° tilt plus 5 m/s of drift in the same direction), and they happen because the PD baseline keeps the engine pushing while it is still tilted. With 40 m bounds it was 96%, which is why the bounds were widened to 45 m. The ground extends 50 m around each pad.

**Zero-shot result (thesis-worthy):** the Stage 1 policy does not generalise to larger offsets. It lands vertically next to the pad (`OffPad`), because Stage 1 never required horizontal travel. This motivates the curriculum.

## 3. Training

```powershell
.\.venv\Scripts\python -m mlagents.trainers.learn config/rocket_lander_stage2.yaml --run-id=stage2_v1 --initialize-from=stage1_v1 --torch-device=cpu
```
Press **Play** when asked. 20 M steps take roughly 3-3.5 h at about 100k steps/min. Ctrl+C stops it early and saves the model.

**What to expect:**
- **Steps 0-1 M (L0):** reward jumps quickly to about 15+, since the policy already knows this task. It then adapts to the new precision and soft-landing bonuses, so `Landing/Distance` and `Landing/TouchdownSpeed` should start falling.
- **Each lesson change:** `Environment/Lesson Number` steps up. Success drops temporarily and `OffPad` rises while it learns to fly sideways, then recovers.
- **Warning signs:**
  - `Outcome/Timeout` rising means it is hovering.
  - `Outcome/TippedOver` or `Rocket/MaxTiltDeg` near 80 means it can't recover the attitude.
  - Stuck on one lesson for more than 3-4 M steps: send me the curves.

## 4. Thesis material

**Methods draft (Stage 2).** Stage 2 extends the task to recovery and guidance. The rocket spawns at a random position up to 20 m from the pad and 10-35 m above it, with an initial tilt up to 30° about a random horizontal axis, a random initial velocity (≤ 5 m/s) and angular velocity (≤ 0.4 rad/s). The difficulty is raised over five curriculum lessons that advance automatically on mean return. Before training, recoverability of every lesson was verified with a hand-tuned cascaded PD controller, which landed ≥ 98.9% of episodes at the hardest setting. The reward was revised (v2) to address two shortcomings observed in Stage 1, off-centre (≈1.9 m) and firm (≈1.9 m/s) landings: a precision bonus concentrated within 3 m of the centre, a soft-landing bonus decreasing linearly with touchdown speed, and a steeper potential-based distance term near the pad. The discount factor was raised to 0.995 to match the longer flights. Training was initialised from the Stage 1 policy (transfer learning), which is possible because the observation and action spaces are unchanged.

**What to record:**
- Learning curves with lesson changes marked:
  - `Environment/Cumulative Reward`, `Rocket/SuccessRate`, the `Outcome/*` rates
  - `Landing/Distance`, `Landing/TouchdownSpeed`
- **Final evaluation** with the showcase scene (CSV export) at L4 conditions, about 500 episodes:
  - success rate vs. spawn tilt (bins of 5°), spawn distance and spawn height
  - landing-point scatter on the pad
  - touchdown-speed histogram
  - max tilt during the flight vs. spawn tilt
- **Comparison table:** Stage 1 policy (zero-shot) vs. Stage 2 policy vs. PD autopilot, on the same L4 conditions.

**Ablation ideas:**
- **B1, transfer vs. scratch:** the same config without `--initialize-from`. Compare steps to reach L4 and final success.
- **B2, curriculum vs. none:** train directly on L4.
- **B3, reward v1 vs. v2:** shows what the precision and soft-landing terms do to landing distance and touchdown speed.
- **B4, γ = 0.99 vs. 0.995.**
- **B5, observation ablation:** remove the previous actions or the angular velocity. Recovery from tilt should suffer.

## 5. Results (run `stage2_v1`, 20 M steps, 2 h 57 min, about 206k episodes)

**Curriculum progress:**
- L0 → L1 immediately (the Stage 1 policy already solved L0).
- L2 at about 0.24 M steps, L3 at about 1.0 M, L4 (final) at about 1.8 M.
- The remaining steps refined landing quality on L4.

**Training curve on L4:**

| Steps | Success | Landing distance | Touchdown speed |
|---|---|---|---|
| 2 M | 59% | 1.9 m | 1.8 m/s |
| 4 M | 90% | 1.2 m | 0.63 m/s |
| 10 M | 97% | 0.65 m | 0.77 m/s |
| 20 M | 99.5% | 0.29 m | 0.59 m/s |

Final mean reward 30.7 (std 3.1).

**Evaluation:** showcase scene, L4 conditions, deterministic policy, 500 episodes. File: `Evaluation/RocketShowcase_RocketLander_stage2_v1_20261026_164722.csv`.

| Policy (L4 conditions) | Success | Landing distance (mean) | Touchdown speed (mean) | Flight time |
|---|---|---|---|---|
| **Stage 2 PPO policy** | **99.6%** (498/500) | **0.26 m** (91.6% < 0.5 m, 98.2% < 1 m) | **0.65 m/s** (p95 0.97) | 9.5 s |
| PD autopilot (calibration, n = 530) | 98.9% | 0.91 m | 0.90 m/s | 18.3 s |
| Stage 1 policy, zero-shot (n = 362) | 8% | 2.96 m | 2.14 m/s | 10.7 s |

**Success by spawn tilt:**

| Spawn tilt | Success |
|---|---|
| 0-10° | 100% (n = 169) |
| 10-20° | 100% (n = 168) |
| 20-30° | 98.8% (n = 163) |

**Success by spawn distance:** 100% / 100% / 99.3% / 99.5% for 0-5 / 5-10 / 10-15 / 15-20 m.

**Other observations:**
- The policy flies aggressively: mean max tilt during flight is 22° (p95 32°), and it lands about twice as fast as the PD baseline.
- **Failure mode** (2/500): both started at 23-29° tilt, peaked at about 47° tilt and hit the pad at about 9 m/s. These are aggressive late corrections from high-tilt starts.
