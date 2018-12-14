use na::{Real, UnitQuaternion, U4};

use aliases::{Qua, TMat4, TVec, TVec3};

/// Euler angles of the quaternion `q` as (pitch, yaw, roll).
pub fn quat_euler_angles<N: Real>(x: &Qua<N>) -> TVec3<N> {
    let q = UnitQuaternion::new_unchecked(*x);
    let a = q.euler_angles();
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

/// Computes a look-at quaternion based on the defaults configured for the library at build time
///
/// # Parameters
///
/// * `direction` - Direction vector point at where to look
/// * `up` - Object up vector
///
/// # Compile Options
///
/// There is 1 compile option that changes the behaviour of the function:
/// 1. left_hand_default/right_hand_default
///
/// ##### left_hand_default/right_hand_default
/// Depending on which option is set the function will return either a left hand or a right
/// hand look at quaternion.
///
pub fn quat_look_at<N: Real>(direction: &TVec3<N>, up: &TVec3<N>) -> Qua<N> {
    if cfg!(feature="right_hand_default") {
        quat_look_at_rh(direction, up)
    } else if cfg!(feature="left_hand_default") {
        quat_look_at_lh(direction, up)
    } else {
        unimplemented!()
    }
}

/// Computes a left-handed look-at quaternion (equivalent to a left-handed look-at matrix).
pub fn quat_look_at_lh<N: Real>(direction: &TVec3<N>, up: &TVec3<N>) -> Qua<N> {
    UnitQuaternion::look_at_lh(direction, up).unwrap()
}

/// Computes a right-handed look-at quaternion (equivalent to a right-handed look-at matrix).
pub fn quat_look_at_rh<N: Real>(direction: &TVec3<N>, up: &TVec3<N>) -> Qua<N> {
    UnitQuaternion::look_at_rh(direction, up).unwrap()
}

/// The "roll" Euler angle of the quaternion `x` assumed to be normalized.
pub fn quat_roll<N: Real>(x: &Qua<N>) -> N {
    // FIXME: optimize this.
    quat_euler_angles(x).z
}

/// The "yaw" Euler angle of the quaternion `x` assumed to be normalized.
pub fn quat_yaw<N: Real>(x: &Qua<N>) -> N {
    // FIXME: optimize this.
    quat_euler_angles(x).y
}

/// The "pitch" Euler angle of the quaternion `x` assumed to be normalized.
pub fn quat_pitch<N: Real>(x: &Qua<N>) -> N {
    // FIXME: optimize this.
    quat_euler_angles(x).x
}
