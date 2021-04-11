use na::{RealField, Unit, UnitQuaternion};

use crate::aliases::{Qua, TVec3};

/// The rotation angle of this quaternion assumed to be normalized.
pub fn quat_angle<T: RealField>(x: &Qua<T>) -> T {
    UnitQuaternion::from_quaternion(*x).angle()
}

/// Creates a quaternion from an axis and an angle.
pub fn quat_angle_axis<T: RealField>(angle: T, axis: &TVec3<T>) -> Qua<T> {
    UnitQuaternion::from_axis_angle(&Unit::new_normalize(*axis), angle).into_inner()
}

/// The rotation axis of a quaternion assumed to be normalized.
pub fn quat_axis<T: RealField>(x: &Qua<T>) -> TVec3<T> {
    if let Some(a) = UnitQuaternion::from_quaternion(*x).axis() {
        a.into_inner()
    } else {
        TVec3::zeros()
    }
}
