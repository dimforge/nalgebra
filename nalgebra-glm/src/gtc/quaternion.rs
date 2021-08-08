use na::UnitQuaternion;

use crate::aliases::{Qua, TMat4, TVec, TVec3};
use crate::RealNumber;

/// Euler angles of the quaternion `q` as (pitch, yaw, roll).
pub fn quat_euler_angles<T: RealNumber>(x: &Qua<T>) -> TVec3<T> {
    let q = UnitQuaternion::new_unchecked(*x);
    let a = q.euler_angles();
    TVec3::new(a.2, a.1, a.0)
}

/// Component-wise `>` comparison between two quaternions.
pub fn quat_greater_than<T: RealNumber>(x: &Qua<T>, y: &Qua<T>) -> TVec<bool, 4> {
    crate::greater_than(&x.coords, &y.coords)
}

/// Component-wise `>=` comparison between two quaternions.
pub fn quat_greater_than_equal<T: RealNumber>(x: &Qua<T>, y: &Qua<T>) -> TVec<bool, 4> {
    crate::greater_than_equal(&x.coords, &y.coords)
}

/// Component-wise `<` comparison between two quaternions.
pub fn quat_less_than<T: RealNumber>(x: &Qua<T>, y: &Qua<T>) -> TVec<bool, 4> {
    crate::less_than(&x.coords, &y.coords)
}

/// Component-wise `<=` comparison between two quaternions.
pub fn quat_less_than_equal<T: RealNumber>(x: &Qua<T>, y: &Qua<T>) -> TVec<bool, 4> {
    crate::less_than_equal(&x.coords, &y.coords)
}

/// Convert a quaternion to a rotation matrix in homogeneous coordinates.
pub fn quat_cast<T: RealNumber>(x: &Qua<T>) -> TMat4<T> {
    crate::quat_to_mat4(x)
}

/// Computes a right hand look-at quaternion
///
/// # Parameters
///
/// * `direction` - Direction vector point at where to look
/// * `up` - Object up vector
///
pub fn quat_look_at<T: RealNumber>(direction: &TVec3<T>, up: &TVec3<T>) -> Qua<T> {
    quat_look_at_rh(direction, up)
}

/// Computes a left-handed look-at quaternion (equivalent to a left-handed look-at matrix).
pub fn quat_look_at_lh<T: RealNumber>(direction: &TVec3<T>, up: &TVec3<T>) -> Qua<T> {
    UnitQuaternion::look_at_lh(direction, up).into_inner()
}

/// Computes a right-handed look-at quaternion (equivalent to a right-handed look-at matrix).
pub fn quat_look_at_rh<T: RealNumber>(direction: &TVec3<T>, up: &TVec3<T>) -> Qua<T> {
    UnitQuaternion::look_at_rh(direction, up).into_inner()
}

/// The "roll" Euler angle of the quaternion `x` assumed to be normalized.
pub fn quat_roll<T: RealNumber>(x: &Qua<T>) -> T {
    // TODO: optimize this.
    quat_euler_angles(x).z
}

/// The "yaw" Euler angle of the quaternion `x` assumed to be normalized.
pub fn quat_yaw<T: RealNumber>(x: &Qua<T>) -> T {
    // TODO: optimize this.
    quat_euler_angles(x).y
}

/// The "pitch" Euler angle of the quaternion `x` assumed to be normalized.
pub fn quat_pitch<T: RealNumber>(x: &Qua<T>) -> T {
    // TODO: optimize this.
    quat_euler_angles(x).x
}
