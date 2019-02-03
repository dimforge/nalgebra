use na::{Real, Unit, UnitQuaternion};

use aliases::{Qua, TVec3};

/// The rotation angle of this quaternion assumed to be normalized.
pub fn quat_angle<N: Real>(x: &Qua<N>) -> N {
    UnitQuaternion::from_quaternion(*x).angle()
}

/// Creates a quaternion from an axis and an angle.
pub fn quat_angle_axis<N: Real>(angle: N, axis: &TVec3<N>) -> Qua<N> {
    UnitQuaternion::from_axis_angle(&Unit::new_normalize(*axis), angle).into_inner()
}

/// The rotation axis of a quaternion assumed to be normalized.
pub fn quat_axis<N: Real>(x: &Qua<N>) -> TVec3<N> {
    if let Some(a) = UnitQuaternion::from_quaternion(*x).axis() {
        a.into_inner()
    } else {
        TVec3::zeros()
    }
}
