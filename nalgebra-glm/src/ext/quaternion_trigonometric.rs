use na::{Real, U3, Unit, UnitQuaternion, Vector3};

use aliases::{Vec, Qua};

/// The rotation angle of this quaternion assumed to be normalized.
pub fn quat_angle<N: Real>(x: &Qua<N>) -> N {
    UnitQuaternion::from_quaternion(*x).angle()
}

/// Creates a quaternion from an axis and an angle.
pub fn quat_angle_axis<N: Real>(angle: N, axis: &Vec<N, U3>) -> Qua<N> {
    UnitQuaternion::from_axis_angle(&Unit::new_normalize(*axis), angle).unwrap()
}

/// The rotation axis of a quaternion assumed to be normalized.
pub fn quat_axis<N: Real>(x: &Qua<N>) -> Vec<N, U3> {
    if let Some(a) = UnitQuaternion::from_quaternion(*x).axis() {
        a.unwrap()
    } else {
        Vector3::zeros()
    }
}