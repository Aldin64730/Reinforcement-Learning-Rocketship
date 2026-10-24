using UnityEngine;

namespace RocketLander
{
    [System.Serializable]
    public class AutopilotGains
    {
        [Header("Horizontal guidance (PD on position -> desired tilt)")]
        public float positionGain = 0.25f;     // (m/s^2) per m
        public float velocityGain = 1.0f;      // (m/s^2) per m/s
        public float maxLateralAccel = 2f;     // m/s^2
        public float maxTiltDeg = 15f;

        [Header("Vertical guidance (descent-speed schedule)")]
        public float descentGain = 0.4f;       // desired descent speed per metre of altitude
        public float maxDescentSpeed = 5f;     // m/s
        public float touchdownSpeed = 0.6f;    // m/s
        public float verticalGain = 1.5f;      // (m/s^2) per m/s of vertical speed error
        [Tooltip("Do not descend below this altitude while further than approachRadius from the pad centre.")]
        public float approachAltitude = 3f;
        public float approachRadius = 1.5f;

        [Header("Attitude control (PD on tilt error)")]
        public float attitudeGain = 5f;
        public float rateGain = 6f;
    }

    /// <summary>
    /// Classical (non-learning) cascaded PD controller. Two uses:
    ///  1. A sanity check that the environment is solvable with the given thrust and torque
    ///     (run with Behavior Type = Heuristic Only before training).
    ///  2. A baseline to compare the RL policy against in the thesis.
    /// It outputs the same 3 actions as the policy, in the same [-1, 1] ranges.
    /// </summary>
    public static class RocketAutopilot
    {
        /// <returns>(main, pitch, roll) actions in [-1, 1].</returns>
        public static Vector3 Compute(AutopilotGains g, Rigidbody rb, Vector3 feetPosition, Vector3 padTop,
                                      float altitude, int feetOnPad, float mainThrustToWeight)
        {
            float gravity = -Physics.gravity.y;
            Quaternion rot = rb.rotation;
            Quaternion inv = Quaternion.Inverse(rot);
            Vector3 up = rot * Vector3.up;
            Vector3 velocity = rb.linearVelocity;

            // 1) Horizontal: desired horizontal acceleration -> desired thrust direction.
            Vector3 toPad = padTop - feetPosition;
            Vector3 horizontalError = new Vector3(toPad.x, 0f, toPad.z);
            Vector3 horizontalVelocity = new Vector3(velocity.x, 0f, velocity.z);
            Vector3 desiredAccel = Vector3.ClampMagnitude(
                g.positionGain * horizontalError - g.velocityGain * horizontalVelocity, g.maxLateralAccel);
            Vector3 desiredUp = (desiredAccel + Vector3.up * gravity).normalized;
            if (Vector3.Angle(desiredUp, Vector3.up) > g.maxTiltDeg)
                desiredUp = Vector3.RotateTowards(Vector3.up, desiredUp, g.maxTiltDeg * Mathf.Deg2Rad, 0f);

            // 2) Vertical: descend faster when high, slow to touchdownSpeed near the pad.
            float desiredVy = -Mathf.Clamp(g.descentGain * altitude + g.touchdownSpeed, g.touchdownSpeed, g.maxDescentSpeed);
            if (horizontalError.magnitude > g.approachRadius && altitude < g.approachAltitude)
                desiredVy = 0.5f * (g.approachAltitude - altitude); // climb back / hold until centred
            float cosTilt = Mathf.Max(0.5f, Vector3.Dot(up, Vector3.up));
            float accelUp = gravity + g.verticalGain * (desiredVy - velocity.y);
            float throttle = accelUp / (cosTilt * mainThrustToWeight * gravity);

            // 3) Attitude: rotate 'up' towards 'desiredUp'. cross(up, desiredUp) is the rotation axis;
            //    its local X/Z parts map directly onto the pitch/roll commands (see RocketController).
            Vector3 errorLocal = inv * Vector3.Cross(up, desiredUp);
            Vector3 rateLocal = inv * rb.angularVelocity;
            float pitch = g.attitudeGain * errorLocal.x - g.rateGain * rateLocal.x;
            float roll = g.attitudeGain * errorLocal.z - g.rateGain * rateLocal.z;

            // 4) Touchdown sensor: engine off as soon as any foot touches the pad
            //    (otherwise it can end up balancing on one leg at hover throttle).
            if (feetOnPad >= 1) throttle = 0f;

            return new Vector3(Mathf.Clamp01(throttle) * 2f - 1f, Mathf.Clamp(pitch, -1f, 1f), Mathf.Clamp(roll, -1f, 1f));
        }
    }
}
