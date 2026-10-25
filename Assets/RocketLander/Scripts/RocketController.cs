using UnityEngine;

namespace RocketLander
{
    /// <summary>
    /// Low-level rocket physics: 5 thrusters, a shared fuel tank, main-engine spool-up,
    /// aerodynamic drag (including wind) and landing-leg contact sensing.
    ///
    /// This class contains NO learning logic, so the RL agent, the PD autopilot baseline and
    /// a human (keyboard) all fly exactly the same physics.
    ///
    /// Forces are applied only from ApplyControls(), which the agent calls from
    /// OnActionReceived(). ML-Agents calls that from FixedUpdate, so forces are applied
    /// once per physics step (never apply physics forces in Update()).
    ///
    /// Thruster layout (rocket-local axes): main engine on the axis at the bottom;
    /// side pods fire DOWNWARD (along +up) at Front (+Z), Back (-Z), Right (+X), Left (-X).
    /// Because the pods are offset sideways from the centre of mass, differential pod thrust
    /// creates pitch (about X) and roll (about Z) torque. Nothing can create yaw torque, but
    /// yaw is irrelevant for landing this symmetric rocket.
    /// </summary>
    [RequireComponent(typeof(Rigidbody))]
    public class RocketController : MonoBehaviour
    {
        public const int SideCount = 4;
        public const int Front = 0, Back = 1, Left = 2, Right = 3;

        [Header("Mass properties (set explicitly for reproducibility)")]
        [Tooltip("Rigidbody mass (kg).")]
        public float mass = 1000f;
        [Tooltip("Centre of mass in rocket-local coordinates (m). Slightly below the geometric centre, like a fuelled rocket.")]
        public Vector3 centerOfMass = new Vector3(0f, -0.5f, 0f);
        [Tooltip("The inertia tensor is computed as a solid cylinder with this radius (m)...")]
        public float inertiaRadius = 1.0f;
        [Tooltip("...and this height (m).")]
        public float inertiaHeight = 7.6f;
        [Tooltip("Rigidbody angular damping: a simple stand-in for aerodynamic rotational damping.")]
        public float angularDamping = 0.3f;

        [Header("Main engine")]
        public Transform mainNozzle;
        [Tooltip("Maximum main-engine thrust as a multiple of the rocket's weight (thrust-to-weight ratio). Hover throttle = 1 / TWR.")]
        public float mainThrustToWeight = 2.0f;
        [Tooltip("How fast the main-engine throttle can change, in full-range units per second (engine spool-up lag).")]
        public float throttleRate = 4f;

        [Header("Side attitude thrusters (pods fire downward)")]
        public Transform frontThruster;
        public Transform backThruster;
        public Transform leftThruster;
        public Transform rightThruster;
        [Tooltip("Maximum thrust of EACH side pod as a multiple of the rocket's weight.")]
        public float sideThrustToWeight = 0.15f;

        [Header("Fuel (one shared tank; capacity is set per episode by RocketEnvironment)")]
        [Tooltip("Fuel units burned per second by the main engine at full throttle.")]
        public float mainBurnRate = 4f;
        [Tooltip("Fuel units burned per second by EACH side pod at full throttle.")]
        public float sideBurnRate = 0.4f;

        [Header("Aerodynamics and wind")]
        [Tooltip("Axial drag coefficient k (N per (m/s)^2) = 0.5*rho*Cd*A with A = pi*1^2 m^2 (top/bottom area).")]
        public float axialDrag = 1.5f;
        [Tooltip("Lateral drag coefficient k (N per (m/s)^2) = 0.5*rho*Cd*A with A = 2 m x 7.2 m side area.")]
        public float lateralDrag = 9f;
        [Tooltip("Centre of pressure distance above the centre of mass (m). > 0 means side wind tilts the rocket.")]
        public float centerOfPressureAboveCoM = 0.5f;
        [Tooltip("Optional wind source (Stage 3). Without one, the air is still.")]
        public WindField wind;

        [Header("Landing legs (foot colliders = touchdown sensors)")]
        public Collider[] footColliders = new Collider[4];

        [Header("Visuals (optional)")]
        [Tooltip("Flame pivots (scaled along local Y with thrust): main, front, back, left, right.")]
        public Transform[] flames = new Transform[5];

        // ---------- Public state (read by the agent / autopilot / HUD) ----------
        public Rigidbody Body { get; private set; }
        /// <summary>Distance from the rocket pivot down to the bottom of the feet (m).</summary>
        public float FootOffset { get; private set; }
        /// <summary>Actual main-engine throttle 0..1 (lags behind the command).</summary>
        public float MainThrottle { get; private set; }
        public float MainThrottleCommand { get; private set; }
        public float PitchCommand { get; private set; }
        public float RollCommand { get; private set; }
        public float Fuel { get; private set; }
        public float FuelCapacity { get; private set; } = 1f;
        public float FuelFraction => FuelCapacity > 0f ? Fuel / FuelCapacity : 0f;
        public float FuelUsedMain { get; private set; }
        public float FuelUsedSide { get; private set; }
        public float HoverThrottle => 1f / Mathf.Max(0.01f, mainThrustToWeight);
        public Vector3 CurrentWind { get; private set; }
        public float SideThrottle(int index) => sideThrottle[index];

        // Contact sensing, latched once per physics step (see LatchContacts).
        public int FeetOnPad { get; private set; }
        public int FeetOnGround { get; private set; }
        public bool BodyContact { get; private set; }
        /// <summary>Largest foot impact speed (m/s) of contacts that STARTED in the last physics step.</summary>
        public float MaxFootImpactSpeed { get; private set; }

        readonly float[] sideThrottle = new float[SideCount];
        Transform[] sideThrusters;
        Vector3 mainNozzleLocal;
        readonly Vector3[] sideThrusterLocal = new Vector3[SideCount];
        Vector3[] flameBaseScales;
        float effectiveFuelScale = 1f;

        // Accumulators filled by collision callbacks during the physics step.
        readonly bool[] footOnPadAcc = new bool[4];
        readonly bool[] footOnGroundAcc = new bool[4];
        bool bodyContactAcc;
        float footImpactAcc;
        int latchedStep = -1;

        float Weight => mass * -Physics.gravity.y;

        void Awake()
        {
            Body = GetComponent<Rigidbody>();
            ApplyPhysicsSettings();

            // Store thruster positions in rocket-local space once. In FixedUpdate we rebuild
            // world positions from the Rigidbody pose instead of reading child Transforms,
            // because with interpolation enabled the Transforms hold interpolated
            // (render) poses, not the physics pose.
            sideThrusters = new[] { frontThruster, backThruster, leftThruster, rightThruster };
            mainNozzleLocal = mainNozzle ? transform.InverseTransformPoint(mainNozzle.position) : Vector3.zero;
            for (int i = 0; i < SideCount; i++)
                sideThrusterLocal[i] = sideThrusters[i] ? transform.InverseTransformPoint(sideThrusters[i].position) : Vector3.zero;

            FootOffset = ComputeFootOffset();

            flameBaseScales = new Vector3[flames.Length];
            for (int i = 0; i < flames.Length; i++)
                if (flames[i]) flameBaseScales[i] = flames[i].localScale;
        }

        /// <summary>Applies the Inspector physics values to the Rigidbody (explicit values instead of Unity's automatic ones).</summary>
        public void ApplyPhysicsSettings()
        {
            Body.mass = mass;
            Body.centerOfMass = centerOfMass; // disables automatic centre of mass
            float r2 = inertiaRadius * inertiaRadius;
            float h2 = inertiaHeight * inertiaHeight;
            float ixx = mass * (3f * r2 + h2) / 12f;   // solid cylinder, about a transverse axis
            float iyy = 0.5f * mass * r2;               // solid cylinder, about its long axis
            Body.inertiaTensor = new Vector3(ixx, iyy, ixx);
            Body.inertiaTensorRotation = Quaternion.identity;
            Body.linearDamping = 0f;                    // drag is modelled explicitly in ApplyAerodynamics
            Body.angularDamping = angularDamping;
            Body.useGravity = true;
            Body.isKinematic = false;
            Body.interpolation = RigidbodyInterpolation.Interpolate;      // smooth visuals only
            Body.collisionDetectionMode = CollisionDetectionMode.Continuous; // no tunnelling into the pad
            Body.sleepThreshold = 0f;                   // a sleeping body stops sending OnCollisionStay
            Body.maxAngularVelocity = 10f;
        }

        /// <summary>Resets fuel, engines and contact state at the start of an episode.</summary>
        public void ResetState(float fuelCapacity)
        {
            FuelCapacity = Mathf.Max(0.01f, fuelCapacity);
            Fuel = FuelCapacity;
            FuelUsedMain = FuelUsedSide = 0f;
            MainThrottle = MainThrottleCommand = PitchCommand = RollCommand = 0f;
            for (int i = 0; i < SideCount; i++) sideThrottle[i] = 0f;
            effectiveFuelScale = 1f;
            ClearContacts();
            if (wind) wind.ResetEpisode();
            CurrentWind = wind ? wind.CurrentWind : Vector3.zero;
            UpdateFlames();
        }

        /// <summary>Teleports the rocket (used for spawning).</summary>
        public void Teleport(Vector3 position, Quaternion rotation, Vector3 velocity, Vector3 angularVelocity)
        {
            Body.position = position;
            Body.rotation = rotation;
            transform.SetPositionAndRotation(position, rotation);
            Body.linearVelocity = velocity;
            Body.angularVelocity = angularVelocity;
            ClearContacts();
        }

        /// <summary>
        /// Applies one physics step of control. Call exactly once per FixedUpdate.
        /// </summary>
        /// <param name="mainThrottleCommand">0..1 main-engine throttle command.</param>
        /// <param name="pitch">-1..1, positive = torque about +X (tilts the nose towards +Z). Fires the Back pod.</param>
        /// <param name="roll">-1..1, positive = torque about +Z (tilts the nose towards -X). Fires the Right pod.</param>
        /// <returns>Fuel units burned this step.</returns>
        public float ApplyControls(float mainThrottleCommand, float pitch, float roll, float dt)
        {
            MainThrottleCommand = Mathf.Clamp01(mainThrottleCommand);
            PitchCommand = Mathf.Clamp(pitch, -1f, 1f);
            RollCommand = Mathf.Clamp(roll, -1f, 1f);

            // Main engine: first-order-ish spool-up lag (rate limit).
            MainThrottle = Mathf.MoveTowards(MainThrottle, MainThrottleCommand, throttleRate * dt);

            // Mixer: each pod only pushes, so each torque direction is produced by one pod.
            // Pod at local (0, y, -r) pushing +up gives torque r*F about +X, so the Back pod gives +pitch.
            sideThrottle[Front] = Mathf.Max(0f, -PitchCommand);
            sideThrottle[Back] = Mathf.Max(0f, PitchCommand);
            sideThrottle[Right] = Mathf.Max(0f, RollCommand);
            sideThrottle[Left] = Mathf.Max(0f, -RollCommand);

            // Fuel: one shared tank. If it would run dry this step, scale all thrust down.
            float mainBurn = MainThrottle * mainBurnRate * dt;
            float sideBurn = (sideThrottle[0] + sideThrottle[1] + sideThrottle[2] + sideThrottle[3]) * sideBurnRate * dt;
            float totalBurn = mainBurn + sideBurn;
            effectiveFuelScale = totalBurn <= Fuel ? 1f : (totalBurn > 0f ? Fuel / totalBurn : 0f);
            Fuel = Mathf.Max(0f, Fuel - totalBurn * effectiveFuelScale);
            FuelUsedMain += mainBurn * effectiveFuelScale;
            FuelUsedSide += sideBurn * effectiveFuelScale;

            // Thrust forces act along the rocket's up axis at the nozzle positions.
            // AddForceAtPosition produces the correct torque about the centre of mass automatically.
            Quaternion rot = Body.rotation;
            Vector3 up = rot * Vector3.up;
            Vector3 pos = Body.position;

            float mainForce = MainThrottle * effectiveFuelScale * mainThrustToWeight * Weight;
            if (mainForce > 0f)
                Body.AddForceAtPosition(up * mainForce, pos + rot * mainNozzleLocal, ForceMode.Force);

            float sideMax = sideThrustToWeight * Weight * effectiveFuelScale;
            for (int i = 0; i < SideCount; i++)
                if (sideThrottle[i] > 0f)
                    Body.AddForceAtPosition(up * (sideThrottle[i] * sideMax), pos + rot * sideThrusterLocal[i], ForceMode.Force);

            ApplyAerodynamics(dt);
            UpdateFlames();
            return totalBurn * effectiveFuelScale;
        }

        /// <summary>
        /// Quadratic drag on the airspeed (rocket velocity minus wind), split into axial and lateral parts
        /// and applied at the centre of pressure. With no wind this is ordinary air drag; with wind it
        /// creates both a push and (because CoP is not at CoM) a tilting torque.
        /// </summary>
        void ApplyAerodynamics(float dt)
        {
            Quaternion rot = Body.rotation;
            Vector3 centerOfPressure = Body.worldCenterOfMass + rot * Vector3.up * centerOfPressureAboveCoM;
            CurrentWind = wind ? wind.Step(dt) : Vector3.zero;

            Vector3 airspeed = Body.GetPointVelocity(centerOfPressure) - CurrentWind;
            Vector3 local = Quaternion.Inverse(rot) * airspeed;
            Vector3 lateral = new Vector3(local.x, 0f, local.z);
            Vector3 dragLocal = -lateralDrag * lateral.magnitude * lateral
                                + Vector3.down * (axialDrag * local.y * Mathf.Abs(local.y));
            Body.AddForceAtPosition(rot * dragLocal, centerOfPressure, ForceMode.Force);
        }

        // ---------------- Contact sensing ----------------

        /// <summary>
        /// Copies the contacts collected during the last physics step into the public properties.
        /// Safe to call several times per step: only the first call with a given stepId latches
        /// (the agent passes Academy.TotalStepCount, which is shared by CollectObservations and
        /// OnActionReceived within one step).
        /// Collision callbacks run AFTER the physics step, so this has to happen at the start
        /// of the next step, and episode termination is decided there, not inside OnCollisionEnter.
        /// </summary>
        public void LatchContacts(int stepId)
        {
            if (stepId == latchedStep) return;
            latchedStep = stepId;

            int onPad = 0, onGround = 0;
            for (int i = 0; i < 4; i++)
            {
                if (footOnPadAcc[i]) onPad++;
                else if (footOnGroundAcc[i]) onGround++;
                footOnPadAcc[i] = footOnGroundAcc[i] = false;
            }
            FeetOnPad = onPad;
            FeetOnGround = onGround;
            BodyContact = bodyContactAcc;
            MaxFootImpactSpeed = footImpactAcc;
            bodyContactAcc = false;
            footImpactAcc = 0f;
        }

        void ClearContacts()
        {
            for (int i = 0; i < 4; i++) footOnPadAcc[i] = footOnGroundAcc[i] = false;
            bodyContactAcc = false;
            footImpactAcc = 0f;
            FeetOnPad = FeetOnGround = 0;
            BodyContact = false;
            MaxFootImpactSpeed = 0f;
        }

        void OnCollisionEnter(Collision collision) => RecordContact(collision, true);
        void OnCollisionStay(Collision collision) => RecordContact(collision, false);

        void RecordContact(Collision collision, bool isNewContact)
        {
            bool otherIsPad = collision.collider.TryGetComponent<LandingPad>(out _);
            int count = collision.contactCount;
            for (int c = 0; c < count; c++)
            {
                Collider mine = collision.GetContact(c).thisCollider;
                int foot = FootIndex(mine);
                if (foot >= 0)
                {
                    if (otherIsPad) footOnPadAcc[foot] = true;
                    else footOnGroundAcc[foot] = true;
                    if (isNewContact)
                        footImpactAcc = Mathf.Max(footImpactAcc, collision.relativeVelocity.magnitude);
                }
                else
                {
                    bodyContactAcc = true; // anything other than a foot touching something = crash
                }
            }
        }

        int FootIndex(Collider c)
        {
            for (int i = 0; i < footColliders.Length; i++)
                if (footColliders[i] == c) return i;
            return -1;
        }

        /// <summary>Distance from the rocket pivot to the lowest point of the feet, measured along the rocket axis.</summary>
        float ComputeFootOffset()
        {
            float minY = 0f;
            foreach (Collider c in footColliders)
            {
                if (!c) continue;
                if (c is SphereCollider s)
                {
                    Vector3 centerLocal = transform.InverseTransformPoint(s.transform.TransformPoint(s.center));
                    float radius = s.radius * MaxAbs(s.transform.lossyScale) / MaxAbs(transform.lossyScale);
                    minY = Mathf.Min(minY, centerLocal.y - radius);
                }
                else
                {
                    minY = Mathf.Min(minY, transform.InverseTransformPoint(c.bounds.min).y);
                }
            }
            return -minY;
        }

        static float MaxAbs(Vector3 v) => Mathf.Max(Mathf.Abs(v.x), Mathf.Max(Mathf.Abs(v.y), Mathf.Abs(v.z)));

        // ---------------- Visuals ----------------

        /// <summary>Drives only the flame visuals (used when replaying a recorded flight; no forces).</summary>
        public void SetReplayVisuals(float main, float front, float back, float left, float right)
        {
            MainThrottle = main;
            sideThrottle[Front] = front;
            sideThrottle[Back] = back;
            sideThrottle[Left] = left;
            sideThrottle[Right] = right;
            effectiveFuelScale = 1f;
            UpdateFlames();
        }

        void UpdateFlames()
        {
            if (flameBaseScales == null) return;
            for (int i = 0; i < flames.Length; i++)
            {
                Transform f = flames[i];
                if (!f) continue;
                float t = (i == 0 ? MainThrottle : sideThrottle[i - 1]) * effectiveFuelScale;
                bool on = t > 0.02f;
                if (f.gameObject.activeSelf != on) f.gameObject.SetActive(on);
                Vector3 s = flameBaseScales[i];
                f.localScale = new Vector3(s.x, s.y * t, s.z);
            }
        }

        void OnDrawGizmosSelected()
        {
            Rigidbody rb = Body ? Body : GetComponent<Rigidbody>();
            if (!rb) return;
            Gizmos.color = Color.yellow;
            Vector3 com = Application.isPlaying ? rb.worldCenterOfMass : transform.TransformPoint(centerOfMass);
            Gizmos.DrawWireSphere(com, 0.25f);
            Gizmos.color = Color.cyan;
            Gizmos.DrawWireSphere(com + transform.up * centerOfPressureAboveCoM, 0.2f);
        }
    }
}
