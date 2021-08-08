use na::{Unit, UnitQuaternion};

use crate::aliases::{Qua, TVec3};
use crate::RealNumber;

/// The rotation angle of this quaternion assumed to be normalized.
pub fn quat_angle<T: RealNumber>(x: &Qua<T>) -> T {
    UnitQuaternion::from_quaternion(*x).angle()
}

/// Creates a quaternion from an axis and an angle.
pub fn quat_angle_axis<T: RealNumber>(angle: T, axis: &TVec3<T>) -> Qua<T> {
    UnitQuaternion::from_axis_angle(&Unit::new_normalize(*axis), angle).into_inner()
}

/// The rotation axis of a quaternion assumed to be normalized.
pub fn quat_axis<T: RealNumber>(x: &Qua<T>) -> TVec3<T> {
    if let Some(a) = UnitQuaternion::from_quaternion(*x).axis() {
        a.into_inner()
    } else {
        TVec3::zeros()
    }
}
