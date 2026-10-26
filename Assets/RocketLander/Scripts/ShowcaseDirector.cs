using System.Collections.Generic;
using Unity.MLAgents;
using UnityEngine;
#if ENABLE_INPUT_SYSTEM
using UnityEngine.InputSystem;
#endif

namespace RocketLander
{
    /// <summary>
    /// Showcase for one rocket:
    ///  1. EVALUATE: flies the trained policy for N episodes as fast as possible
    ///     (the Academy and physics are stepped manually, many steps per frame) and records every flight.
    ///  2. REPLAY: replays the best flight in real time with a follow camera.
    ///
    /// ML-Agents does not record flights during training, so the "best episode" is found by evaluating
    /// the final trained model here. Three "best" criteria are kept at the same time and can be switched
    /// while replaying.
    ///
    /// Controls: B = switch criterion, R = restart replay, Space = pause, N = evaluate a new batch,
    ///           +/- = replay speed.
    /// </summary>
    public class ShowcaseDirector : MonoBehaviour
    {
        public enum BestBy { HighestReward, MostCentred, SoftestTouchdown }

        [Header("References")]
        public RocketAgent agent;
        public Camera showcaseCamera;

        [Header("Evaluation")]
        [Tooltip("How many episodes to fly before picking the best one.")]
        public int episodesToEvaluate = 100;
        [Tooltip("Physics steps simulated per rendered frame while evaluating (higher = faster).")]
        public int physicsStepsPerFrame = 1000;
        [Tooltip("Write every evaluated episode to <project>/Evaluation/<scene>_<time>.csv (for thesis plots).")]
        public bool exportCsv = true;

        [Header("Replay")]
        public BestBy bestBy = BestBy.HighestReward;
        public float replaySpeed = 1f;
        [Tooltip("Seconds to hold the landed rocket before the replay loops.")]
        public float pauseAfterReplay = 2f;
        [Tooltip("Camera position relative to the rocket (world axes).")]
        public Vector3 cameraOffset = new Vector3(0f, 2.5f, -22f);
        public float cameraSmoothing = 3f;

        struct Frame
        {
            public Vector3 position;
            public Quaternion rotation;
            public float main, front, back, left, right;
        }

        class Flight
        {
            public readonly List<Frame> frames = new List<Frame>(512);
            public RocketAgent.EpisodeSummary summary;
        }

        enum Phase { Evaluating, Replaying }

        Phase phase;
        RocketController rocket;
        Flight current;
        readonly Flight[] best = new Flight[3];
        int evaluated, landed;
        readonly List<RocketAgent.EpisodeSummary> allSummaries = new List<RocketAgent.EpisodeSummary>();
        string lastCsv;
        float replayTime;
        bool paused;
        float dt;

        Flight Shown => best[(int)bestBy];

        void Start()
        {
            if (!agent) agent = FindAnyObjectByType<RocketAgent>();
            if (!showcaseCamera) showcaseCamera = Camera.main;
            rocket = agent.GetComponent<RocketController>();
            agent.showHud = false;
            agent.logEpisodes = false;
            agent.EpisodeEnded += OnEpisodeEnded;
            dt = Time.fixedDeltaTime;
            BeginEvaluation();
        }

        void OnDestroy()
        {
            if (agent) agent.EpisodeEnded -= OnEpisodeEnded;
            // Restore global settings (they can otherwise leak into other scenes / the project).
            Physics.simulationMode = SimulationMode.FixedUpdate;
            if (Academy.IsInitialized) Academy.Instance.AutomaticSteppingEnabled = true;
        }

        // ------------------------------------------------------------ evaluation

        void BeginEvaluation()
        {
            phase = Phase.Evaluating;
            evaluated = landed = 0;
            allSummaries.Clear();
            for (int i = 0; i < best.Length; i++) best[i] = null;
            current = new Flight();

            // Take manual control of time: no automatic Academy steps, no automatic physics.
            Academy.Instance.AutomaticSteppingEnabled = false;
            Physics.simulationMode = SimulationMode.Script;
            rocket.Body.isKinematic = false;
            rocket.Body.interpolation = RigidbodyInterpolation.None;
        }

        void RunEvaluationSteps()
        {
            Academy academy = Academy.Instance;
            for (int i = 0; i < physicsStepsPerFrame && evaluated < episodesToEvaluate; i++)
            {
                academy.EnvironmentStep();   // observe, decide (trained model), apply forces
                Physics.Simulate(dt);        // advance physics by one fixed step
                Record();
            }
            if (evaluated >= episodesToEvaluate) BeginReplay();
        }

        void Record()
        {
            Rigidbody rb = rocket.Body;
            current.frames.Add(new Frame
            {
                position = rb.position,
                rotation = rb.rotation,
                main = rocket.MainThrottle,
                front = rocket.SideThrottle(RocketController.Front),
                back = rocket.SideThrottle(RocketController.Back),
                left = rocket.SideThrottle(RocketController.Left),
                right = rocket.SideThrottle(RocketController.Right),
            });
        }

        void OnEpisodeEnded(RocketAgent a, RocketAgent.EpisodeSummary summary)
        {
            if (phase != Phase.Evaluating) return;
            current.summary = summary;
            allSummaries.Add(summary);
            evaluated++;
            if (summary.outcome == EpisodeOutcome.Landed) landed++;
            for (int i = 0; i < best.Length; i++)
                if (best[i] == null || IsBetter(current.summary, best[i].summary, (BestBy)i))
                    best[i] = current;
            current = new Flight();
        }

        static bool IsBetter(RocketAgent.EpisodeSummary a, RocketAgent.EpisodeSummary b, BestBy criterion)
        {
            bool aLanded = a.outcome == EpisodeOutcome.Landed, bLanded = b.outcome == EpisodeOutcome.Landed;
            if (aLanded != bLanded) return aLanded; // a landing always beats a failure
            switch (criterion)
            {
                case BestBy.MostCentred: return a.distance < b.distance;
                case BestBy.SoftestTouchdown: return a.touchdownSpeed < b.touchdownSpeed;
                default: return a.reward > b.reward;
            }
        }

        // ---------------------------------------------------------------- replay

        void WriteCsv()
        {
            var inv = System.Globalization.CultureInfo.InvariantCulture;
            string dir = System.IO.Path.Combine(Application.dataPath, "..", "Evaluation");
            System.IO.Directory.CreateDirectory(dir);
            var modelAsset = agent.GetComponent<Unity.MLAgents.Policies.BehaviorParameters>().Model;
            string model = modelAsset ? modelAsset.name : "heuristic";
            string file = System.IO.Path.Combine(dir,
                $"{UnityEngine.SceneManagement.SceneManager.GetActiveScene().name}_{model}_{System.DateTime.Now:yyyyMMdd_HHmmss}.csv");
            var sb = new System.Text.StringBuilder();
            sb.AppendLine("episode,outcome,reward,seconds,spawn_height,spawn_distance,spawn_tilt_deg,max_tilt_deg,landing_distance,touchdown_speed,final_tilt_deg,fuel_used");
            for (int i = 0; i < allSummaries.Count; i++)
            {
                var s = allSummaries[i];
                sb.AppendLine(string.Join(",", i.ToString(inv), s.outcome.ToString(), s.reward.ToString("F3", inv), s.seconds.ToString("F2", inv),
                    s.spawnAltitude.ToString("F2", inv), s.spawnDistance.ToString("F2", inv), s.spawnTiltDeg.ToString("F2", inv),
                    s.maxTiltDeg.ToString("F2", inv), s.distance.ToString("F3", inv), s.touchdownSpeed.ToString("F3", inv),
                    s.tiltDeg.ToString("F2", inv), s.fuelUsedFraction.ToString("F3", inv)));
            }
            System.IO.File.WriteAllText(file, sb.ToString());
            lastCsv = System.IO.Path.GetFullPath(file);
            Debug.Log("[Showcase] Wrote " + lastCsv);
        }

        void BeginReplay()
        {
            if (exportCsv && allSummaries.Count > 0) WriteCsv();
            phase = Phase.Replaying;
            // Freeze the simulation: the rocket is now just posed from the recording.
            rocket.Body.isKinematic = true;
            replayTime = 0f;
            paused = false;
            SnapCamera();
        }

        void Update()
        {
            HandleInput();
            if (phase == Phase.Evaluating)
            {
                RunEvaluationSteps();
                return;
            }

            Flight f = Shown;
            if (f == null || f.frames.Count == 0) return;
            float duration = (f.frames.Count - 1) * dt;
            if (!paused) replayTime += Time.deltaTime * replaySpeed;
            if (replayTime > duration + pauseAfterReplay)
            {
                replayTime = 0f;
                SnapCamera();
            }
            ApplyFrame(f, Mathf.Min(replayTime, duration));
        }

        void ApplyFrame(Flight f, float time)
        {
            float x = time / dt;
            int i = Mathf.Clamp(Mathf.FloorToInt(x), 0, f.frames.Count - 1);
            int j = Mathf.Min(i + 1, f.frames.Count - 1);
            float t = x - i;
            Frame a = f.frames[i], b = f.frames[j];
            rocket.transform.SetPositionAndRotation(Vector3.Lerp(a.position, b.position, t),
                                                    Quaternion.Slerp(a.rotation, b.rotation, t));
            rocket.SetReplayVisuals(Mathf.Lerp(a.main, b.main, t), Mathf.Lerp(a.front, b.front, t),
                                    Mathf.Lerp(a.back, b.back, t), Mathf.Lerp(a.left, b.left, t),
                                    Mathf.Lerp(a.right, b.right, t));
        }

        // ---------------------------------------------------------------- camera

        Vector3 LookTarget()
        {
            Vector3 pad = agent.environment.PadTop;
            return Vector3.Lerp(pad, rocket.transform.position, 0.6f);
        }

        void SnapCamera()
        {
            if (!showcaseCamera || phase != Phase.Replaying || Shown == null) return;
            ApplyFrame(Shown, 0f);
            showcaseCamera.transform.position = rocket.transform.position + cameraOffset;
            showcaseCamera.transform.LookAt(LookTarget());
        }

        void LateUpdate()
        {
            if (!showcaseCamera || phase != Phase.Replaying) return;
            Vector3 desired = rocket.transform.position + cameraOffset;
            float k = 1f - Mathf.Exp(-cameraSmoothing * Time.deltaTime);
            showcaseCamera.transform.position = Vector3.Lerp(showcaseCamera.transform.position, desired, k);
            Quaternion look = Quaternion.LookRotation(LookTarget() - showcaseCamera.transform.position);
            showcaseCamera.transform.rotation = Quaternion.Slerp(showcaseCamera.transform.rotation, look, k);
        }

        // ---------------------------------------------------------------- input & UI

        void HandleInput()
        {
#if ENABLE_INPUT_SYSTEM
            Keyboard kb = Keyboard.current;
            if (kb == null) return;
            if (kb.nKey.wasPressedThisFrame) { BeginEvaluation(); agent.EndEpisode(); } // un-freeze first, then respawn
            if (phase != Phase.Replaying) return;
            if (kb.bKey.wasPressedThisFrame) { bestBy = (BestBy)(((int)bestBy + 1) % 3); replayTime = 0f; SnapCamera(); }
            if (kb.rKey.wasPressedThisFrame) { replayTime = 0f; SnapCamera(); }
            if (kb.spaceKey.wasPressedThisFrame) paused = !paused;
            if (kb.equalsKey.wasPressedThisFrame || kb.numpadPlusKey.wasPressedThisFrame) replaySpeed = Mathf.Min(4f, replaySpeed * 2f);
            if (kb.minusKey.wasPressedThisFrame || kb.numpadMinusKey.wasPressedThisFrame) replaySpeed = Mathf.Max(0.125f, replaySpeed * 0.5f);
#endif
        }

        void OnGUI()
        {
            var style = new GUIStyle(GUI.skin.label) { fontSize = 16 };
            GUI.Box(new Rect(10, 10, 430, 250), GUIContent.none);
            var r = new Rect(22, 18, 410, 240);

            if (phase == Phase.Evaluating)
            {
                GUI.Label(r, $"Evaluating trained policy...\n\nEpisode {evaluated} / {episodesToEvaluate}\nLanded {landed}", style);
                return;
            }

            Flight f = Shown;
            if (f == null) return;
            RocketAgent.EpisodeSummary s = f.summary;
            float duration = (f.frames.Count - 1) * dt;
            Frame now = f.frames[Mathf.Clamp(Mathf.FloorToInt(Mathf.Min(replayTime, duration) / dt), 0, f.frames.Count - 1)];
            float altitude = now.position.y - rocket.FootOffset - agent.environment.PadTop.y;

            GUI.Label(r,
                $"Best of {evaluated} episodes ({landed} landed)\n" +
                $"Criterion: {Nice(bestBy)}   [B to switch]\n\n" +
                $"Outcome        {s.outcome}\n" +
                $"Reward         {s.reward:F2}\n" +
                $"Spawn height   {s.spawnAltitude:F1} m\n" +
                $"Landing offset {s.distance:F2} m from centre\n" +
                $"Touchdown      {s.touchdownSpeed:F2} m/s\n" +
                $"Flight time    {s.seconds:F1} s\n\n" +
                $"Replay {Mathf.Min(replayTime, duration):F1} / {duration:F1} s   x{replaySpeed:0.###}{(paused ? "  PAUSED" : "")}   alt {Mathf.Max(0f, altitude):F1} m",
                style);
            GUI.Label(new Rect(12, Screen.height - 30, 700, 24),
                "B criterion   R restart   Space pause   +/- speed   N new evaluation", style);
        }

        static string Nice(BestBy b) => b == BestBy.HighestReward ? "highest reward"
                                      : b == BestBy.MostCentred ? "most centred landing"
                                      : "softest touchdown";
    }
}
