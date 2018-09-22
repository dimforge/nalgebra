use na::{Real, Unit, UnitQuaternion, U3};

use aliases::{Vec, Qua};

/// Rotates a quaternion from a vector of 3 components normalized axis and an angle.
///
/// # Parameters
///     * `q` - Source orientation
///     * `angle` - Angle expressed in radians.
///     * `axis` - Normalized axis of the rotation, must be normalized.
pub fn rotate_normalized_axis<N: Real>(q: &Qua<N>, angle: N, axis: &Vec<N, U3>) -> Qua<N> {
    q * UnitQuaternion::from_axis_angle(&Unit::new_unchecked(*axis), angle).unwrap()
}