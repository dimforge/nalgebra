use na::{Real, U3, Unit, UnitQuaternion, Vector3};

use aliases::{Vec, Qua};

pub fn angle<N: Real>(x: &Qua<N>) -> N {
    UnitQuaternion::from_quaternion(*x).angle()
}

pub fn angleAxis<N: Real>(angle: N, axis: &Vec<N, U3>) -> Qua<N> {
    UnitQuaternion::from_axis_angle(&Unit::new_normalize(*axis), angle).unwrap()
}

pub fn axis<N: Real>(x: &Qua<N>) -> Vec<N, U3> {
    if let Some(a) = UnitQuaternion::from_quaternion(*x).axis() {
        a.unwrap()
    } else {
        Vector3::zeros()
    }
}