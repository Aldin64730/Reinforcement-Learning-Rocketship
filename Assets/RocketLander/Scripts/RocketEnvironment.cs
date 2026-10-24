using Unity.MLAgents;
using UnityEngine;

namespace RocketLander
{
    /// <summary>
    /// Per-episode settings, read from ML-Agents environment parameters (curriculum) with the
    /// Inspector values as defaults.
    /// </summary>
    public struct EpisodeSettings
    {
        public float spawnHeightMin, spawnHeightMax;
        public float spawnHorizontalRadius;
        public float spawnTiltMaxDeg;
        public float spawnSpeedMax;
        public float spawnAngularSpeedMax;
        public float fuelCapacity;
        public float maxEpisodeSeconds;
        public float boundsRadius;
        public float ceilingHeight;
        public float windSpeedMin, windSpeedMax, gustSpeedMax, turbulenceIntensity;
    }

    /// <summary>
    /// Manages one training area (pad + ground + wind): reads curriculum parameters, samples spawn
    /// states and defines the flight volume. Every difficulty knob for Stages 1-3 lives here, so
    /// later stages only need different parameter values (in the trainer YAML), not new code.
    ///
    /// All positions are relative to the pad, so this area can be copied anywhere in the scene
    /// for parallel training.
    /// </summary>
    public class RocketEnvironment : MonoBehaviour
    {
        [Header("References")]
        public LandingPad pad;
        public WindField wind;

        [Header("Spawn (defaults = end of Stage 1; YAML environment parameters override these)")]
        [Tooltip("Minimum height of the feet above the pad top (m). env param: spawn_height_min")]
        public float spawnHeightMin = 8f;
        [Tooltip("Maximum height of the feet above the pad top (m). env param: spawn_height_max")]
        public float spawnHeightMax = 15f;
        [Tooltip("Maximum horizontal offset from the pad centre (m), uniform over a disc. env param: spawn_horizontal_radius")]
        public float spawnHorizontalRadius = 2f;
        [Tooltip("Maximum initial tilt from vertical (degrees). env param: spawn_tilt_max_deg")]
        public float spawnTiltMaxDeg = 5f;
        [Tooltip("Maximum initial linear speed (m/s), random direction. env param: spawn_speed_max")]
        public float spawnSpeedMax = 1.5f;
        [Tooltip("Maximum initial angular speed (rad/s), random axis. env param: spawn_angular_speed_max")]
        public float spawnAngularSpeedMax = 0f;

        [Header("Episode limits")]
        [Tooltip("Fuel units in the shared tank. env param: fuel_capacity")]
        public float fuelCapacity = 60f;
        [Tooltip("Episode time limit (s). env param: max_episode_seconds")]
        public float maxEpisodeSeconds = 30f;
        [Tooltip("Out of bounds when the horizontal distance from the pad centre exceeds this (m). env param: bounds_radius")]
        public float boundsRadius = 30f;
        [Tooltip("Out of bounds above this height over the pad (m). env param: ceiling_height")]
        public float ceilingHeight = 40f;

        [Header("Wind (Stage 3; 0 = no wind)")]
        [Tooltip("env param: wind_speed_min (m/s)")] public float windSpeedMin = 0f;
        [Tooltip("env param: wind_speed_max (m/s)")] public float windSpeedMax = 0f;
        [Tooltip("env param: gust_speed_max (m/s)")] public float gustSpeedMax = 0f;
        [Tooltip("env param: turbulence_intensity (m/s)")] public float turbulenceIntensity = 0f;

        public EpisodeSettings Settings { get; private set; }
        /// <summary>Centre of the pad top surface, cached at the start of each episode.</summary>
        public Vector3 PadTop { get; private set; }

        void Awake()
        {
            if (!pad) pad = GetComponentInChildren<LandingPad>();
            if (!wind) wind = GetComponentInChildren<WindField>();
            ReadParameters();
        }

        /// <summary>Reads the current curriculum values. Called at the start of every episode.</summary>
        public void ReadParameters()
        {
            var p = Academy.Instance.EnvironmentParameters;
            var s = new EpisodeSettings
            {
                spawnHeightMin = p.GetWithDefault("spawn_height_min", spawnHeightMin),
                spawnHeightMax = p.GetWithDefault("spawn_height_max", spawnHeightMax),
                spawnHorizontalRadius = p.GetWithDefault("spawn_horizontal_radius", spawnHorizontalRadius),
                spawnTiltMaxDeg = p.GetWithDefault("spawn_tilt_max_deg", spawnTiltMaxDeg),
                spawnSpeedMax = p.GetWithDefault("spawn_speed_max", spawnSpeedMax),
                spawnAngularSpeedMax = p.GetWithDefault("spawn_angular_speed_max", spawnAngularSpeedMax),
                fuelCapacity = p.GetWithDefault("fuel_capacity", fuelCapacity),
                maxEpisodeSeconds = p.GetWithDefault("max_episode_seconds", maxEpisodeSeconds),
                boundsRadius = p.GetWithDefault("bounds_radius", boundsRadius),
                ceilingHeight = p.GetWithDefault("ceiling_height", ceilingHeight),
                windSpeedMin = p.GetWithDefault("wind_speed_min", windSpeedMin),
                windSpeedMax = p.GetWithDefault("wind_speed_max", windSpeedMax),
                gustSpeedMax = p.GetWithDefault("gust_speed_max", gustSpeedMax),
                turbulenceIntensity = p.GetWithDefault("turbulence_intensity", turbulenceIntensity),
            };
            s.spawnHeightMax = Mathf.Max(s.spawnHeightMin, s.spawnHeightMax);
            Settings = s;
            PadTop = pad.TopCenter;

            if (wind) wind.Configure(s.windSpeedMin, s.windSpeedMax, s.gustSpeedMax, s.turbulenceIntensity);
        }

        /// <summary>
        /// Samples a spawn state. Yaw is always fully random: the rocket is yaw-symmetric and has no
        /// yaw actuator, so the policy should be yaw-invariant from the start.
        /// </summary>
        /// <param name="footOffset">Distance from the rocket pivot to the bottom of its feet (m).</param>
        public void SampleSpawn(float footOffset, out Vector3 position, out Quaternion rotation,
                                out Vector3 velocity, out Vector3 angularVelocity)
        {
            EpisodeSettings s = Settings;

            // Uniform over a disc (sqrt makes the density uniform in area, not clustered at the centre).
            float radius = s.spawnHorizontalRadius * Mathf.Sqrt(Random.value);
            float angle = Random.Range(0f, 2f * Mathf.PI);
            float height = Random.Range(s.spawnHeightMin, s.spawnHeightMax);
            position = PadTop + new Vector3(radius * Mathf.Cos(angle), height + footOffset, radius * Mathf.Sin(angle));

            // Random yaw, then a tilt of uniform magnitude about a random horizontal axis.
            Quaternion yaw = Quaternion.Euler(0f, Random.Range(0f, 360f), 0f);
            Vector3 tiltAxis = Quaternion.Euler(0f, Random.Range(0f, 360f), 0f) * Vector3.right;
            Quaternion tilt = Quaternion.AngleAxis(Random.Range(0f, s.spawnTiltMaxDeg), tiltAxis);
            rotation = tilt * yaw;

            velocity = Random.insideUnitSphere * s.spawnSpeedMax;
            angularVelocity = Random.insideUnitSphere * s.spawnAngularSpeedMax;
        }

        public bool IsOutOfBounds(float horizontalDistance, float altitude)
        {
            EpisodeSettings s = Settings;
            return horizontalDistance > s.boundsRadius || altitude > s.ceilingHeight || altitude < -3f;
        }

        void OnDrawGizmosSelected()
        {
            if (!pad) return;
            Vector3 c = pad.TopCenter;
            Gizmos.color = new Color(1f, 0.5f, 0f);
            Gizmos.DrawWireCube(c + Vector3.up * (spawnHeightMin + spawnHeightMax) * 0.5f,
                new Vector3(spawnHorizontalRadius * 2f, spawnHeightMax - spawnHeightMin, spawnHorizontalRadius * 2f));
            Gizmos.color = Color.red;
            Gizmos.DrawWireCube(c + Vector3.up * ceilingHeight * 0.5f, new Vector3(boundsRadius * 2f, ceilingHeight, boundsRadius * 2f));
        }
    }
}
