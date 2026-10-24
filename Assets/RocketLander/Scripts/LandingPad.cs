using UnityEngine;

namespace RocketLander
{
    /// <summary>
    /// Marks a collider as the landing pad and exposes its geometry.
    /// Contact code recognises the pad by this component instead of by tag,
    /// so no Tag Manager setup is needed and copies of the training area work out of the box.
    /// Assumes a square, axis-aligned pad (rotation about X only, like the imported mesh).
    /// </summary>
    [RequireComponent(typeof(BoxCollider))]
    public class LandingPad : MonoBehaviour
    {
        BoxCollider box;

        BoxCollider Box => box ? box : (box = GetComponent<BoxCollider>());

        /// <summary>World-space centre of the pad's top surface: the landing target.</summary>
        public Vector3 TopCenter
        {
            get
            {
                Bounds b = Box.bounds;
                return new Vector3(b.center.x, b.max.y, b.center.z);
            }
        }

        /// <summary>Half the pad width in metres (5 m for the 10 x 10 m pad).</summary>
        public float HalfSize
        {
            get
            {
                Vector3 e = Box.bounds.extents;
                return Mathf.Min(e.x, e.z);
            }
        }

        void OnDrawGizmos()
        {
            Gizmos.color = Color.green;
            Gizmos.DrawWireSphere(TopCenter, 0.3f);
        }
    }
}
