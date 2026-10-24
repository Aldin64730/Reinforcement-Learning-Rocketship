using UnityEngine;

namespace RocketLander
{
    /// <summary>
    /// Wind model for one training area (Stage 3). Produces a wind velocity (m/s) per physics step:
    ///   wind = mean wind (random direction and speed per episode)
    ///        + turbulence (smooth Perlin noise)
    ///        + gusts (random events with a smooth 1 - cos profile).
    /// The rocket turns the wind into forces through aerodynamic drag on its airspeed
    /// (see RocketController.ApplyAerodynamics), so wind also creates torque when the
    /// centre of pressure is offset from the centre of mass.
    ///
    /// Stage 1 / 2: all strengths are 0, so Step() returns Vector3.zero.
    /// Values are set per episode by RocketEnvironment from ML-Agents environment parameters.
    /// </summary>
    public class WindField : MonoBehaviour
    {
        [Header("Mean wind (randomised per episode)")]
        [Tooltip("Minimum mean horizontal wind speed (m/s). env param: wind_speed_min")]
        public float meanSpeedMin = 0f;
        [Tooltip("Maximum mean horizontal wind speed (m/s). env param: wind_speed_max")]
        public float meanSpeedMax = 0f;

        [Header("Gusts")]
        [Tooltip("Peak extra wind speed of a gust (m/s). env param: gust_speed_max")]
        public float gustSpeedMax = 0f;
        [Tooltip("Average number of gusts per minute.")]
        public float gustsPerMinute = 6f;
        [Tooltip("Gust duration range in seconds (x = min, y = max).")]
        public Vector2 gustDurationRange = new Vector2(1f, 3f);
        [Tooltip("Gust direction is the mean direction +/- this many degrees.")]
        public float gustDirectionSpreadDeg = 45f;

        [Header("Turbulence")]
        [Tooltip("Amplitude of turbulent wind fluctuations (m/s). env param: turbulence_intensity")]
        public float turbulenceIntensity = 0f;
        [Tooltip("How fast the turbulence changes (noise samples per second).")]
        public float turbulenceFrequency = 0.5f;
        [Tooltip("Vertical turbulence relative to horizontal (real turbulence is mostly horizontal near the ground).")]
        public float verticalTurbulenceFactor = 0.3f;

        public Vector3 MeanWind { get; private set; }
        public Vector3 CurrentWind { get; private set; }

        float time;
        Vector3 noiseOffset;
        bool gustActive;
        Vector3 gustDirection;
        float gustPeak, gustStart, gustDuration;

        /// <summary>Called by RocketEnvironment with the current curriculum values.</summary>
        public void Configure(float speedMin, float speedMax, float gustMax, float turbulence)
        {
            meanSpeedMin = speedMin;
            meanSpeedMax = Mathf.Max(speedMin, speedMax);
            gustSpeedMax = gustMax;
            turbulenceIntensity = turbulence;
        }

        /// <summary>Draws a new wind condition for the next episode.</summary>
        public void ResetEpisode()
        {
            time = 0f;
            gustActive = false;
            Vector3 direction = Quaternion.Euler(0f, Random.Range(0f, 360f), 0f) * Vector3.forward;
            MeanWind = direction * Random.Range(meanSpeedMin, meanSpeedMax);
            noiseOffset = new Vector3(Random.Range(0f, 1000f), Random.Range(0f, 1000f), Random.Range(0f, 1000f));
            CurrentWind = MeanWind;
        }

        /// <summary>Advances the wind by one physics step and returns the wind velocity (m/s).</summary>
        public Vector3 Step(float dt)
        {
            time += dt;
            Vector3 wind = MeanWind;

            if (turbulenceIntensity > 0f)
            {
                float t = time * turbulenceFrequency;
                wind += new Vector3(
                    Noise(noiseOffset.x + t),
                    Noise(noiseOffset.y + t) * verticalTurbulenceFactor,
                    Noise(noiseOffset.z + t)) * turbulenceIntensity;
            }

            if (gustSpeedMax > 0f)
            {
                if (!gustActive && Random.value < gustsPerMinute / 60f * dt)
                    StartGust();

                if (gustActive)
                {
                    float p = (time - gustStart) / gustDuration;
                    if (p >= 1f)
                        gustActive = false;
                    else
                        wind += gustDirection * (gustPeak * 0.5f * (1f - Mathf.Cos(2f * Mathf.PI * p)));
                }
            }

            CurrentWind = wind;
            return wind;
        }

        void StartGust()
        {
            gustActive = true;
            gustStart = time;
            gustDuration = Random.Range(gustDurationRange.x, gustDurationRange.y);
            gustPeak = Random.Range(0.3f, 1f) * gustSpeedMax;
            Vector3 baseDirection = MeanWind.sqrMagnitude > 0.01f
                ? MeanWind.normalized
                : Quaternion.Euler(0f, Random.Range(0f, 360f), 0f) * Vector3.forward;
            gustDirection = Quaternion.Euler(0f, Random.Range(-gustDirectionSpreadDeg, gustDirectionSpreadDeg), 0f) * baseDirection;
        }

        // Perlin noise remapped to roughly [-1, 1].
        static float Noise(float x) => (Mathf.PerlinNoise(x, 0.37f) - 0.5f) * 2f;
    }
}
