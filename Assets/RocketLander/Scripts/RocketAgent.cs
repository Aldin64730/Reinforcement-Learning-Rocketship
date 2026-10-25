using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Policies;
using Unity.MLAgents.Sensors;
using UnityEngine;
#if ENABLE_INPUT_SYSTEM
using UnityEngine.InputSystem;
#endif

namespace RocketLander
{
    public enum EpisodeOutcome { None, Landed, Crashed, TippedOver, OffPad, OutOfBounds, Timeout }

    /// <summary>All reward coefficients in one place (tunable in the Inspector).</summary>
    [System.Serializable]
    public class RewardSettings
    {
        [Header("Potential-based shaping: r = Phi(s') - Phi(s)")]
        [Tooltip("Phi -= this * horizontal distance to pad centre (per m).")]
        public float distanceWeight = 0.3f;
        [Tooltip("Phi -= this * altitude above pad (per m).")]
        public float altitudeWeight = 0.1f;
        [Tooltip("Phi -= this * speed (per m/s).")]
        public float speedWeight = 0.3f;
        [Tooltip("Phi -= this * tilt from vertical (per radian).")]
        public float tiltWeight = 2.0f;
        [Tooltip("Phi += this * fraction of feet touching the pad.")]
        public float feetContactWeight = 1.0f;

        [Header("Dense penalties (per second, except fuel)")]
        [Tooltip("Constant penalty per second (discourages hovering).")]
        public float timePenalty = 0.05f;
        [Tooltip("Penalty per fuel unit burned.")]
        public float fuelPenalty = 0.02f;
        [Tooltip("Penalty per (rad/s) of pitch/roll rate, per second (smooth flight).")]
        public float angularVelocityPenalty = 0.05f;
        [Tooltip("Penalty per (m/s) of descent speed above the safe envelope, per second.")]
        public float descentEnvelopePenalty = 0.5f;
        [Tooltip("Safe descent speed at altitude 0 (m/s).")]
        public float safeTouchdownSpeed = 1.0f;
        [Tooltip("Extra safe descent speed per metre of altitude ((m/s)/m).")]
        public float safeDescentSlope = 0.5f;

        [Header("Terminal rewards")]
        public float landingReward = 10f;
        [Tooltip("Extra reward for landing near the centre: precisionBonus * (1 - distance / padHalfSize).")]
        public float precisionBonus = 5f;
        public float crashPenalty = -10f;
        [Tooltip("Touching the ground (off the pad) without crashing.")]
        public float offPadPenalty = -6f;
        public float outOfBoundsPenalty = -10f;
        public float timeoutPenalty = -5f;
    }

    /// <summary>When an episode counts as a landing or a failure.</summary>
    [System.Serializable]
    public class LandingCriteria
    {
        [Tooltip("A foot impact faster than this (m/s) is a crash (hard landing).")]
        public float maxImpactSpeed = 3f;
        [Tooltip("Landed only if speed is below this (m/s)...")]
        public float maxLandedSpeed = 0.5f;
        [Tooltip("...tilt is below this (deg)...")]
        public float maxLandedTiltDeg = 10f;
        [Tooltip("...pitch/roll rate is below this (rad/s)...")]
        public float maxLandedAngularSpeed = 0.3f;
        [Tooltip("...main throttle is at or below this (0..1), i.e. the legs carry the weight...")]
        public float maxLandedThrottle = 0.4f;
        [Tooltip("...at least this many feet are on the pad...")]
        public int minFeetOnPad = 3;
        [Tooltip("...continuously for this long (s).")]
        public float holdSeconds = 1f;
        [Tooltip("Tilt beyond this (deg) at any time ends the episode as TippedOver.")]
        public float maxFlightTiltDeg = 80f;
    }

    /// <summary>
    /// Normalisation scales for observations. Keep them FIXED across all curriculum stages,
    /// otherwise a policy trained in Stage 1 sees differently scaled inputs in Stage 2.
    /// Chosen for the largest expected Stage 3 ranges; values are then clipped to [-1, 1].
    /// </summary>
    [System.Serializable]
    public class ObservationScales
    {
        public float speed = 25f;         // m/s
        public float angularSpeed = 5f;   // rad/s
        public float altitude = 60f;      // m
        public float distance = 60f;      // m
    }

    /// <summary>
    /// ML-Agents agent that learns to land the rocket.
    ///
    /// Actions (3 continuous, each in [-1, 1]):
    ///   0: main-engine throttle command (-1 = off, +1 = full)
    ///   1: pitch command (differential front/back pod thrust)
    ///   2: roll command (differential left/right pod thrust)
    ///
    /// Observations (22 floats): see CollectObservations.
    ///
    /// Per physics step (OnActionReceived, called every FixedUpdate because the DecisionRequester has
    /// "Take Actions Between Decisions" enabled):
    ///   latch contacts -> measure state -> shaping reward -> termination check -> dense penalties -> apply forces.
    /// </summary>
    [RequireComponent(typeof(RocketController))]
    public class RocketAgent : Agent
    {
        /// <summary>Must equal Behavior Parameters > Vector Observation > Space Size.</summary>
        public const int ObservationSize = 22;
        /// <summary>Must equal Behavior Parameters > Actions > Continuous Actions.</summary>
        public const int ContinuousActionCount = 3;

        public enum HeuristicMode { Autopilot, Keyboard }

        [Header("References")]
        public RocketEnvironment environment;

        [Header("Rewards (all coefficients in one place)")]
        public RewardSettings rewards = new RewardSettings();

        [Header("Landing / failure criteria")]
        public LandingCriteria landing = new LandingCriteria();

        [Header("Observation normalisation (keep fixed across stages!)")]
        public ObservationScales scales = new ObservationScales();

        [Header("Heuristic (Behavior Type = Heuristic Only)")]
        public HeuristicMode heuristicMode = HeuristicMode.Autopilot;
        public AutopilotGains autopilot = new AutopilotGains();

        [Header("Debug")]
        [Tooltip("Show an on-screen readout for this rocket (enable on ONE agent only).")]
        public bool showHud;
        [Tooltip("Log each episode's outcome to the Console.")]
        public bool logEpisodes;

        RocketController rocket;

        // Episode state
        float episodeTime;
        float landedTimer;
        float previousPotential;
        Vector3 lastActions;          // (main, pitch, roll) as sent by the policy, [-1, 1]
        bool touchedDown;
        float spawnAltitude;
        float touchdownSpeed;
        float manualThrottle;         // keyboard heuristic

        // Reward breakdown for diagnostics (logged to TensorBoard)
        float rShaping, rDense, rFuel, rTerminal;
        EpisodeOutcome lastOutcome;
        float lastEpisodeReward;

        struct FlightState
        {
            public Vector3 feetPosition;      // point between the feet (world)
            public float altitude;            // feet height above pad top (m)
            public Vector3 toPad;             // pad target - feet position (world, m)
            public float horizontalDistance;  // m
            public Vector3 velocity;          // world, m/s
            public float speed;
            public Vector3 angularVelocity;   // world, rad/s
            public Vector3 localAngularVelocity;
            public float tiltDeg;             // angle between rocket up and world up
        }

        // ------------------------------------------------------------------ setup

        public override void Initialize()
        {
            rocket = GetComponent<RocketController>();
            if (!environment) environment = GetComponentInParent<RocketEnvironment>();

            // Catch Inspector/code mismatches early (a very common ML-Agents mistake).
            var bp = GetComponent<BehaviorParameters>();
            if (bp.BrainParameters.VectorObservationSize != ObservationSize)
                Debug.LogError($"[RocketAgent] Behavior Parameters Vector Observation Space Size is {bp.BrainParameters.VectorObservationSize}, code expects {ObservationSize}.", this);
            if (bp.BrainParameters.ActionSpec.NumContinuousActions != ContinuousActionCount || bp.BrainParameters.ActionSpec.NumDiscreteActions != 0)
                Debug.LogError($"[RocketAgent] Behavior Parameters must have {ContinuousActionCount} continuous actions and 0 discrete branches.", this);
            if (MaxStep != 0)
                Debug.LogWarning("[RocketAgent] Set Max Step to 0: the time limit is handled by max_episode_seconds (with a timeout penalty).", this);
            var dr = GetComponent<DecisionRequester>();
            if (!dr || !dr.TakeActionsBetweenDecisions)
                Debug.LogError("[RocketAgent] Needs a DecisionRequester with 'Take Actions Between Decisions' enabled (physics is applied every step).", this);
        }

        public override void OnEpisodeBegin()
        {
            environment.ReadParameters();
            EpisodeSettings settings = environment.Settings;

            rocket.ResetState(settings.fuelCapacity);
            environment.SampleSpawn(rocket.FootOffset, out Vector3 pos, out Quaternion rot, out Vector3 vel, out Vector3 angVel);
            rocket.Teleport(pos, rot, vel, angVel);

            episodeTime = 0f;
            landedTimer = 0f;
            touchedDown = false;
            touchdownSpeed = 0f;
            lastActions = new Vector3(-1f, 0f, 0f); // engine off, no torque
            manualThrottle = 0f;
            rShaping = rDense = rFuel = rTerminal = 0f;
            FlightState start = Measure();
            spawnAltitude = start.altitude;
            previousPotential = Potential(start);
        }

        // ------------------------------------------------------------ observations

        public override void CollectObservations(VectorSensor sensor)
        {
            rocket.LatchContacts(Academy.Instance.TotalStepCount);
            FlightState s = Measure();
            Quaternion toLocal = Quaternion.Inverse(rocket.Body.rotation);

            // 1. Attitude (3): world up expressed in the rocket frame. Encodes tilt without the
            //    wrap-around (359 deg -> 0 deg) and gimbal-lock problems of Euler angles, and is yaw-invariant.
            sensor.AddObservation(toLocal * Vector3.up);
            // 2. Linear velocity in the rocket frame (3): aligned with the actuators.
            sensor.AddObservation(Clip(toLocal * s.velocity / scales.speed));
            // 3. Angular velocity in the rocket frame (3).
            sensor.AddObservation(Clip(s.localAngularVelocity / scales.angularSpeed));
            // 4. Altitude of the feet above the pad (1).
            sensor.AddObservation(Clip(s.altitude / scales.altitude));
            // 5. Vertical speed in world frame (1): the key landing quantity, given explicitly.
            sensor.AddObservation(Clip(s.velocity.y / scales.speed));
            // 6. Vector from feet to pad centre in the rocket frame (3).
            sensor.AddObservation(Clip(toLocal * s.toPad / scales.distance));
            // 7. Horizontal distance to pad centre (1).
            sensor.AddObservation(Clip(s.horizontalDistance / scales.distance));
            // 8. Fuel remaining (1), 0..1.
            sensor.AddObservation(rocket.FuelFraction);
            // 9. Actual main-engine throttle (1), 0..1 (differs from the command because of spool-up lag).
            sensor.AddObservation(rocket.MainThrottle);
            // 10. Previous actions (3): the current control state.
            sensor.AddObservation(lastActions);
            // 11. Leg contact sensor (1): fraction of feet touching the pad.
            sensor.AddObservation(rocket.FeetOnPad / 4f);
            // 12. Elapsed episode time fraction (1): makes the time limit observable (Markov).
            sensor.AddObservation(Mathf.Clamp01(episodeTime / environment.Settings.maxEpisodeSeconds));
            // Total: 3+3+3+1+1+3+1+1+1+3+1+1 = 22 = ObservationSize
        }

        // ---------------------------------------------------------------- actions

        public override void OnActionReceived(ActionBuffers actions)
        {
            float dt = Time.fixedDeltaTime;
            rocket.LatchContacts(Academy.Instance.TotalStepCount);
            episodeTime += dt;
            FlightState s = Measure();

            if (!touchedDown && rocket.FeetOnPad + rocket.FeetOnGround > 0)
            {
                touchedDown = true;
                touchdownSpeed = rocket.MaxFootImpactSpeed > 0f ? rocket.MaxFootImpactSpeed : s.speed;
            }

            // 1) Potential-based shaping. Sums to Phi(end) - Phi(start) over an episode,
            //    so it cannot be farmed by oscillating (e.g. touching the pad and lifting off again).
            float phi = Potential(s);
            Give(ref rShaping, phi - previousPotential);
            previousPotential = phi;

            // 2) Termination
            EpisodeOutcome outcome = CheckTermination(s, dt);
            if (outcome != EpisodeOutcome.None)
            {
                Finish(outcome, s);
                return;
            }

            // 3) Dense penalties (scaled by dt so the coefficients are per second,
            //    independent of the physics rate and decision period).
            float descentSpeed = -s.velocity.y;
            float safeSpeed = rewards.safeTouchdownSpeed + rewards.safeDescentSlope * Mathf.Max(0f, s.altitude);
            float pitchRollRate = new Vector2(s.localAngularVelocity.x, s.localAngularVelocity.z).magnitude;
            float dense = -rewards.timePenalty
                          - rewards.angularVelocityPenalty * pitchRollRate
                          - rewards.descentEnvelopePenalty * Mathf.Max(0f, descentSpeed - safeSpeed);
            Give(ref rDense, dense * dt);

            // 4) Act
            var a = actions.ContinuousActions;
            float main = Mathf.Clamp(a[0], -1f, 1f);
            float pitch = Mathf.Clamp(a[1], -1f, 1f);
            float roll = Mathf.Clamp(a[2], -1f, 1f);
            float fuelBurned = rocket.ApplyControls(0.5f * (main + 1f), pitch, roll, dt);
            Give(ref rFuel, -rewards.fuelPenalty * fuelBurned);
            lastActions = new Vector3(main, pitch, roll);
        }

        public override void Heuristic(in ActionBuffers actionsOut)
        {
            var ca = actionsOut.ContinuousActions;
            Vector3 a;
            if (heuristicMode == HeuristicMode.Autopilot)
            {
                rocket.LatchContacts(Academy.Instance.TotalStepCount);
                FlightState s = Measure();
                a = RocketAutopilot.Compute(autopilot, rocket.Body, s.feetPosition, environment.PadTop,
                                            s.altitude, rocket.FeetOnPad, rocket.mainThrustToWeight);
            }
            else
            {
                a = KeyboardActions();
            }
            ca[0] = a.x;
            ca[1] = a.y;
            ca[2] = a.z;
        }

        Vector3 KeyboardActions()
        {
            float pitch = 0f, roll = 0f;
#if ENABLE_INPUT_SYSTEM
            Keyboard kb = Keyboard.current;
            if (kb != null)
            {
                // W/S: throttle up/down (held), Space: full throttle, X: cut engine.
                if (kb.wKey.isPressed) manualThrottle += 0.05f;
                if (kb.sKey.isPressed) manualThrottle -= 0.05f;
                if (kb.xKey.isPressed) manualThrottle = 0f;
                manualThrottle = Mathf.Clamp01(manualThrottle);
                if (kb.spaceKey.isPressed) manualThrottle = 1f;
                // Arrows: tilt forward/back (pitch) and left/right (roll).
                if (kb.upArrowKey.isPressed) pitch += 1f;
                if (kb.downArrowKey.isPressed) pitch -= 1f;
                if (kb.leftArrowKey.isPressed) roll += 1f;
                if (kb.rightArrowKey.isPressed) roll -= 1f;
            }
#endif
            return new Vector3(manualThrottle * 2f - 1f, pitch, roll);
        }

        // ---------------------------------------------------------------- rewards

        /// <summary>Shaping potential: higher (closer to 0) = closer to a good landing state.</summary>
        float Potential(in FlightState s)
        {
            return -rewards.distanceWeight * s.horizontalDistance
                   - rewards.altitudeWeight * Mathf.Max(0f, s.altitude)
                   - rewards.speedWeight * s.speed
                   - rewards.tiltWeight * s.tiltDeg * Mathf.Deg2Rad
                   + rewards.feetContactWeight * (rocket.FeetOnPad / 4f);
        }

        EpisodeOutcome CheckTermination(in FlightState s, float dt)
        {
            if (rocket.BodyContact) return EpisodeOutcome.Crashed;                         // hull/pod/nozzle hit something
            if (rocket.MaxFootImpactSpeed > landing.maxImpactSpeed) return EpisodeOutcome.Crashed; // hard landing
            if (rocket.FeetOnGround > 0) return EpisodeOutcome.OffPad;                     // touched the ground, not the pad
            if (s.tiltDeg > landing.maxFlightTiltDeg) return EpisodeOutcome.TippedOver;
            if (environment.IsOutOfBounds(s.horizontalDistance, s.altitude)) return EpisodeOutcome.OutOfBounds;

            bool resting = rocket.FeetOnPad >= landing.minFeetOnPad
                           && s.speed < landing.maxLandedSpeed
                           && s.tiltDeg < landing.maxLandedTiltDeg
                           && new Vector2(s.localAngularVelocity.x, s.localAngularVelocity.z).magnitude < landing.maxLandedAngularSpeed
                           && rocket.MainThrottle <= landing.maxLandedThrottle;
            landedTimer = resting ? landedTimer + dt : 0f;
            if (landedTimer >= landing.holdSeconds) return EpisodeOutcome.Landed;

            // The time limit is a real terminal state with a penalty. The elapsed time fraction is
            // observed, so the value function can anticipate it.
            if (episodeTime >= environment.Settings.maxEpisodeSeconds) return EpisodeOutcome.Timeout;
            return EpisodeOutcome.None;
        }

        void Finish(EpisodeOutcome outcome, in FlightState s)
        {
            float terminal;
            switch (outcome)
            {
                case EpisodeOutcome.Landed:
                    terminal = rewards.landingReward
                               + rewards.precisionBonus * Mathf.Clamp01(1f - s.horizontalDistance / environment.pad.HalfSize);
                    break;
                case EpisodeOutcome.Crashed:
                case EpisodeOutcome.TippedOver: terminal = rewards.crashPenalty; break;
                case EpisodeOutcome.OffPad: terminal = rewards.offPadPenalty; break;
                case EpisodeOutcome.OutOfBounds: terminal = rewards.outOfBoundsPenalty; break;
                default: terminal = rewards.timeoutPenalty; break;
            }
            Give(ref rTerminal, terminal);
            RecordStats(outcome, s);

            lastOutcome = outcome;
            lastEpisodeReward = GetCumulativeReward();
            if (logEpisodes)
                Debug.Log($"[RocketAgent] {name}: {outcome} after {episodeTime:F1}s, reward {lastEpisodeReward:F2} " +
                          $"(shaping {rShaping:F2}, dense {rDense:F2}, fuel {rFuel:F2}, terminal {rTerminal:F2}), " +
                          $"dist {s.horizontalDistance:F2} m, touchdown {touchdownSpeed:F2} m/s, tilt {s.tiltDeg:F1} deg", this);

            EpisodeEnded?.Invoke(this, new EpisodeSummary
            {
                outcome = outcome,
                reward = lastEpisodeReward,
                seconds = episodeTime,
                distance = s.horizontalDistance,
                touchdownSpeed = touchdownSpeed,
                tiltDeg = s.tiltDeg,
                fuelUsedFraction = 1f - rocket.FuelFraction,
                spawnAltitude = spawnAltitude,
            });
            EndEpisode();
        }

        /// <summary>Result of one finished episode (used by the showcase and evaluation tools).</summary>
        public struct EpisodeSummary
        {
            public EpisodeOutcome outcome;
            public float reward, seconds, distance, touchdownSpeed, tiltDeg, fuelUsedFraction, spawnAltitude;
        }

        /// <summary>Raised at the end of every episode, just before the rocket is reset.</summary>
        public event System.Action<RocketAgent, EpisodeSummary> EpisodeEnded;

        /// <summary>Custom TensorBoard metrics. "Average" aggregation of 0/1 values gives rates.</summary>
        void RecordStats(EpisodeOutcome outcome, in FlightState s)
        {
            StatsRecorder stats = Academy.Instance.StatsRecorder;
            stats.Add("Rocket/SuccessRate", outcome == EpisodeOutcome.Landed ? 1f : 0f);
            stats.Add("Outcome/Crashed", outcome == EpisodeOutcome.Crashed ? 1f : 0f);
            stats.Add("Outcome/TippedOver", outcome == EpisodeOutcome.TippedOver ? 1f : 0f);
            stats.Add("Outcome/OffPad", outcome == EpisodeOutcome.OffPad ? 1f : 0f);
            stats.Add("Outcome/OutOfBounds", outcome == EpisodeOutcome.OutOfBounds ? 1f : 0f);
            stats.Add("Outcome/Timeout", outcome == EpisodeOutcome.Timeout ? 1f : 0f);

            stats.Add("Rocket/EpisodeSeconds", episodeTime);
            stats.Add("Rocket/FinalDistance", s.horizontalDistance);
            stats.Add("Rocket/FuelUsedFraction", 1f - rocket.FuelFraction);
            stats.Add("Rocket/FuelUsedMain", rocket.FuelUsedMain);
            stats.Add("Rocket/FuelUsedSide", rocket.FuelUsedSide);
            if (touchedDown) stats.Add("Rocket/TouchdownSpeed", touchdownSpeed);
            if (outcome == EpisodeOutcome.Landed)
            {
                stats.Add("Landing/Distance", s.horizontalDistance);
                stats.Add("Landing/TouchdownSpeed", touchdownSpeed);
                stats.Add("Landing/TiltDeg", s.tiltDeg);
                stats.Add("Landing/Seconds", episodeTime);
            }

            stats.Add("Reward/Shaping", rShaping);
            stats.Add("Reward/Dense", rDense);
            stats.Add("Reward/Fuel", rFuel);
            stats.Add("Reward/Terminal", rTerminal);
        }

        void Give(ref float bucket, float reward)
        {
            bucket += reward;
            AddReward(reward);
        }

        // ---------------------------------------------------------------- helpers

        /// <summary>Reads the physics state from the Rigidbody (not the Transform, which may be interpolated).</summary>
        FlightState Measure()
        {
            Rigidbody rb = rocket.Body;
            Quaternion rot = rb.rotation;
            Vector3 up = rot * Vector3.up;
            Vector3 padTop = environment.PadTop;

            FlightState s;
            s.feetPosition = rb.position - up * rocket.FootOffset;
            s.altitude = s.feetPosition.y - padTop.y;
            s.toPad = padTop - s.feetPosition;
            s.horizontalDistance = new Vector2(s.toPad.x, s.toPad.z).magnitude;
            s.velocity = rb.linearVelocity;
            s.speed = s.velocity.magnitude;
            s.angularVelocity = rb.angularVelocity;
            s.localAngularVelocity = Quaternion.Inverse(rot) * s.angularVelocity;
            s.tiltDeg = Vector3.Angle(up, Vector3.up);
            return s;
        }

        static float Clip(float v) => Mathf.Clamp(v, -1f, 1f);
        static Vector3 Clip(Vector3 v) => new Vector3(Clip(v.x), Clip(v.y), Clip(v.z));

        void OnGUI()
        {
            if (!showHud || rocket == null || rocket.Body == null || environment == null) return;
            FlightState s = Measure();
            GUI.Box(new Rect(10, 10, 290, 205), GUIContent.none);
            GUI.Label(new Rect(20, 15, 280, 200),
                $"Altitude      {s.altitude,7:F2} m\n" +
                $"Vertical spd  {s.velocity.y,7:F2} m/s\n" +
                $"Speed         {s.speed,7:F2} m/s\n" +
                $"Distance      {s.horizontalDistance,7:F2} m\n" +
                $"Tilt          {s.tiltDeg,7:F1} deg\n" +
                $"Throttle      {rocket.MainThrottle,7:F2} (cmd {rocket.MainThrottleCommand:F2})\n" +
                $"Pitch / Roll  {rocket.PitchCommand,5:F2} / {rocket.RollCommand:F2}\n" +
                $"Fuel          {rocket.FuelFraction * 100f,6:F1} %\n" +
                $"Feet on pad   {rocket.FeetOnPad}\n" +
                $"Reward        {GetCumulativeReward(),7:F2}\n" +
                $"Last episode  {lastOutcome} ({lastEpisodeReward:F2})");
        }
    }
}
