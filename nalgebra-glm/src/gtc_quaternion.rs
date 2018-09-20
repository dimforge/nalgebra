use na::{self, Real, U3, U4, UnitQuaternion, Vector3, Rotation3};

use aliases::{Qua, Vec, Mat};


/// Euler angles of the quaternion as (pitch, yaw, roll).
pub fn euler_angles<N: Real>(x: &Qua<N>) -> Vec<N, U3> {
    let q = UnitQuaternion::new_unchecked(*x);
    let a = q.to_euler_angles();
    Vector3::new(a.2, a.1, a.0)
}

pub fn greater_than<N: Real>(x: &Qua<N>, y: &Qua<N>) -> Vec<bool, U4> {
    ::greater_than(&x.coords, &y.coords)
}

pub fn greater_than_equal<N: Real>(x: &Qua<N>, y: &Qua<N>) -> Vec<bool, U4> {
    ::greater_than_equal(&x.coords, &y.coords)
}

pub fn less_than<N: Real>(x: &Qua<N>, y: &Qua<N>) -> Vec<bool, U4> {
    ::less_than(&x.coords, &y.coords)
}

pub fn less_than_equal<N: Real>(x: &Qua<N>, y: &Qua<N>) -> Vec<bool, U4> {
    ::less_than_equal(&x.coords, &y.coords)
}

pub fn mat3_cast<N: Real>(x: Qua<N>) -> Mat<N, U3, U3> {
    let q = UnitQuaternion::new_unchecked(x);
    q.to_rotation_matrix().unwrap()
}

pub fn mat4_cast<N: Real>(x: Qua<N>) -> Mat<N, U4, U4> {
    let q = UnitQuaternion::new_unchecked(x);
    q.to_homogeneous()
}

pub fn quat_cast<N: Real>(x: Mat<N, U3, U3>) -> Qua<N> {
    let rot = Rotation3::from_matrix_unchecked(x);
    UnitQuaternion::from_rotation_matrix(&rot).unwrap()
}

pub fn quat_cast2<N: Real>(x: Mat<N, U4, U4>) -> Qua<N> {
    quat_cast(x.fixed_slice::<U3, U3>(0, 0).into_owned())
}

pub fn quat_look_at<N: Real>(direction: &Vec<N, U3>, up: &Vec<N, U3>) -> Qua<N> {
    quat_look_at_rh(direction, up)
}

pub fn quat_look_at_lh<N: Real>(direction: &Vec<N, U3>, up: &Vec<N, U3>) -> Qua<N> {
    UnitQuaternion::look_at_lh(direction, up).unwrap()
}

pub fn quat_look_at_rh<N: Real>(direction: &Vec<N, U3>, up: &Vec<N, U3>) -> Qua<N> {
    UnitQuaternion::look_at_rh(direction, up).unwrap()
}

pub fn roll<N: Real>(x: &Qua<N>) -> N {
    // FIXME: optimize this.
    euler_angles(x).z
}

pub fn yaw<N: Real>(x: &Qua<N>) -> N {
    // FIXME: optimize this.
    euler_angles(x).y
}

pub fn pitch<N: Real>(x: &Qua<N>) -> N {
    // FIXME: optimize this.
    euler_angles(x).x
}
