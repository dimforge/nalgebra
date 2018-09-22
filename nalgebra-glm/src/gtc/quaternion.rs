use na::{Real, U3, U4, UnitQuaternion, Vector3, Rotation3};

use aliases::{Qua, Vec, Mat};


/// Euler angles of the quaternion `q` as (pitch, yaw, roll).
pub fn quat_euler_angles<N: Real>(x: &Qua<N>) -> Vec<N, U3> {
    let q = UnitQuaternion::new_unchecked(*x);
    let a = q.to_euler_angles();
    Vector3::new(a.2, a.1, a.0)
}

/// Component-wise `>` comparison between two quaternions.
pub fn quat_greater_than<N: Real>(x: &Qua<N>, y: &Qua<N>) -> Vec<bool, U4> {
    ::greater_than(&x.coords, &y.coords)
}

/// Component-wise `>=` comparison between two quaternions.
pub fn quat_greater_than_equal<N: Real>(x: &Qua<N>, y: &Qua<N>) -> Vec<bool, U4> {
    ::greater_than_equal(&x.coords, &y.coords)
}

/// Component-wise `<` comparison between two quaternions.
pub fn quat_less_than<N: Real>(x: &Qua<N>, y: &Qua<N>) -> Vec<bool, U4> {
    ::less_than(&x.coords, &y.coords)
}

/// Component-wise `<=` comparison between two quaternions.
pub fn quat_less_than_equal<N: Real>(x: &Qua<N>, y: &Qua<N>) -> Vec<bool, U4> {
    ::less_than_equal(&x.coords, &y.coords)
}

/// Convert a quaternion to a rotation matrix in homogeneous coordinates.
pub fn quat_cast<N: Real>(x: &Qua<N>) -> Mat<N, U4, U4> {
    ::quat_to_mat4(x)
}

/// Convert a rotation matrix to a quaternion.
pub fn quat_to_mat3<N: Real>(x: &Mat<N, U3, U3>) -> Qua<N> {
    let rot = Rotation3::from_matrix_unchecked(x);
    UnitQuaternion::from_rotation_matrix(&rot).unwrap()
}

/// Computes a right-handed look-at quaternion (equivalent to a right-handed look-at matrix).
pub fn quat_look_at<N: Real>(direction: &Vec<N, U3>, up: &Vec<N, U3>) -> Qua<N> {
    quat_look_at_rh(direction, up)
}

/// Computes a left-handed look-at quaternion (equivalent to a left-handed look-at matrix).
pub fn quat_look_at_lh<N: Real>(direction: &Vec<N, U3>, up: &Vec<N, U3>) -> Qua<N> {
    UnitQuaternion::look_at_lh(direction, up).unwrap()
}

/// Computes a right-handed look-at quaternion (equivalent to a right-handed look-at matrix).
pub fn quat_look_at_rh<N: Real>(direction: &Vec<N, U3>, up: &Vec<N, U3>) -> Qua<N> {
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
