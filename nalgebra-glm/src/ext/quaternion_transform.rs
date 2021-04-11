use na::{RealField, Unit, UnitQuaternion};

use crate::aliases::{Qua, TVec3};

/// Computes the quaternion exponential.
pub fn quat_exp<T: RealField>(q: &Qua<T>) -> Qua<T> {
    q.exp()
}

/// Computes the quaternion logarithm.
pub fn quat_log<T: RealField>(q: &Qua<T>) -> Qua<T> {
    q.ln()
}

/// Raises the quaternion `q` to the power `y`.
pub fn quat_pow<T: RealField>(q: &Qua<T>, y: T) -> Qua<T> {
    q.powf(y)
}

/// Builds a quaternion from an axis and an angle, and right-multiply it to the quaternion `q`.
pub fn quat_rotate<T: RealField>(q: &Qua<T>, angle: T, axis: &TVec3<T>) -> Qua<T> {
    q * UnitQuaternion::from_axis_angle(&Unit::new_normalize(*axis), angle).into_inner()
}

//pub fn quat_sqrt<T: RealField>(q: &Qua<T>) -> Qua<T> {
//    unimplemented!()
//}
