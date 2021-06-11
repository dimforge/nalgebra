use na::{RealField, Rotation3, Unit, UnitQuaternion};

use crate::aliases::{Qua, TMat4, TVec3};

/// Builds a rotation 4 * 4 matrix created from a normalized axis and an angle.
///
/// # Parameters:
///
/// * `m` - Input matrix multiplied by this rotation matrix.
/// * `angle` - Rotation angle expressed in radians.
/// * `axis` - Rotation axis, must be normalized.
pub fn rotate_normalized_axis<T: RealField>(m: &TMat4<T>, angle: T, axis: &TVec3<T>) -> TMat4<T> {
    m * Rotation3::from_axis_angle(&Unit::new_unchecked(*axis), angle).to_homogeneous()
}

/// Rotates a quaternion from a vector of 3 components normalized axis and an angle.
///
/// # Parameters:
///
/// * `q` - Source orientation.
/// * `angle` - Angle expressed in radians.
/// * `axis` - Normalized axis of the rotation, must be normalized.
pub fn quat_rotate_normalized_axis<T: RealField>(q: &Qua<T>, angle: T, axis: &TVec3<T>) -> Qua<T> {
    q * UnitQuaternion::from_axis_angle(&Unit::new_unchecked(*axis), angle).into_inner()
}
