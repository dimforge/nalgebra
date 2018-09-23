use na::{Real, U4, UnitQuaternion};

use aliases::{Qua, TVec, TVec3, TMat4};


/// Euler angles of the quaternion `q` as (pitch, yaw, roll).
pub fn quat_euler_angles<N: Real>(x: &Qua<N>) -> TVec3<N> {
    let q = UnitQuaternion::new_unchecked(*x);
    let a = q.to_euler_angles();
    TVec3::new(a.2, a.1, a.0)
}

/// Component-wise `>` comparison between two quaternions.
pub fn quat_greater_than<N: Real>(x: &Qua<N>, y: &Qua<N>) -> TVec<bool, U4> {
    ::greater_than(&x.coords, &y.coords)
}

/// Component-wise `>=` comparison between two quaternions.
pub fn quat_greater_than_equal<N: Real>(x: &Qua<N>, y: &Qua<N>) -> TVec<bool, U4> {
    ::greater_than_equal(&x.coords, &y.coords)
}

/// Component-wise `<` comparison between two quaternions.
pub fn quat_less_than<N: Real>(x: &Qua<N>, y: &Qua<N>) -> TVec<bool, U4> {
    ::less_than(&x.coords, &y.coords)
}

/// Component-wise `<=` comparison between two quaternions.
pub fn quat_less_than_equal<N: Real>(x: &Qua<N>, y: &Qua<N>) -> TVec<bool, U4> {
    ::less_than_equal(&x.coords, &y.coords)
}

/// Convert a quaternion to a rotation matrix in homogeneous coordinates.
pub fn quat_cast<N: Real>(x: &Qua<N>) -> TMat4<N> {
    ::quat_to_mat4(x)
}
/// Computes a right-handed look-at quaternion (equivalent to a right-handed look-at matrix).
pub fn quat_look_at<N: Real>(direction: &TVec3<N>, up: &TVec3<N>) -> Qua<N> {
    quat_look_at_rh(direction, up)
}

/// Computes a left-handed look-at quaternion (equivalent to a left-handed look-at matrix).
pub fn quat_look_at_lh<N: Real>(direction: &TVec3<N>, up: &TVec3<N>) -> Qua<N> {
    UnitQuaternion::look_at_lh(direction, up).unwrap()
}

/// Computes a right-handed look-at quaternion (equivalent to a right-handed look-at matrix).
pub fn quat_look_at_rh<N: Real>(direction: &TVec3<N>, up: &TVec3<N>) -> Qua<N> {
    UnitQuaternion::look_at_rh(direction, up).unwrap()
}

/// The "roll" euler angle of the quaternion `x` assumed to be normalized.
pub fn quat_roll<N: Real>(x: &Qua<N>) -> N {
    // FIXME: optimize this.
    quat_euler_angles(x).z
}

/// The "yaw" euler angle of the quaternion `x` assumed to be normalized.
pub fn quat_yaw<N: Real>(x: &Qua<N>) -> N {
    // FIXME: optimize this.
    quat_euler_angles(x).y
}

/// The "pitch" euler angle of the quaternion `x` assumed to be normalized.
pub fn quat_pitch<N: Real>(x: &Qua<N>) -> N {
    // FIXME: optimize this.
    quat_euler_angles(x).x
}
