use na::{Real, U2, U3, U4, Rotation3, Vector3, Unit, UnitComplex};

use aliases::{Vec, Mat};

pub fn orientation<N: Real>(normal: &Vec<N, U3>, up: &Vec<N, U3>) -> Mat<N, U4, U4> {
    if let Some(r) = Rotation3::rotation_between(normal, up) {
        r.to_homogeneous()
    } else {
        Mat::<N, U4, U4>::identity()
    }
}

pub fn rotate2<N: Real>(v: &Vec<N, U2>, angle: N) -> Vec<N, U2> {
    UnitComplex::new(angle) * v
}

pub fn rotate<N: Real>(v: &Vec<N, U3>, angle: N, normal: &Vec<N, U3>) -> Vec<N, U3> {
    Rotation3::from_axis_angle(&Unit::new_normalize(*normal), angle) * v
}

pub fn rotate4<N: Real>(v: &Vec<N, U4>, angle: N, normal: &Vec<N, U3>) -> Vec<N, U4> {
    Rotation3::from_axis_angle(&Unit::new_normalize(*normal), angle).to_homogeneous() * v
}

pub fn rotateX<N: Real>(v: &Vec<N, U3>, angle: N) -> Vec<N, U3> {
    Rotation3::from_axis_angle(&Vector3::x_axis(), angle) * v
}

pub fn rotateX4<N: Real>(v: &Vec<N, U4>, angle: N) -> Vec<N, U4> {
    Rotation3::from_axis_angle(&Vector3::x_axis(), angle).to_homogeneous() * v
}

pub fn rotateY<N: Real>(v: &Vec<N, U3>, angle: N) -> Vec<N, U3> {
    Rotation3::from_axis_angle(&Vector3::y_axis(), angle) * v
}

pub fn rotateY4<N: Real>(v: &Vec<N, U4>, angle: N) -> Vec<N, U4> {
    Rotation3::from_axis_angle(&Vector3::y_axis(), angle).to_homogeneous() * v
}

pub fn rotateZ<N: Real>(v: &Vec<N, U3>, angle: N) -> Vec<N, U3> {
    Rotation3::from_axis_angle(&Vector3::z_axis(), angle) * v
}

pub fn rotateZ4<N: Real>(v: &Vec<N, U4>, angle: N) -> Vec<N, U4> {
    Rotation3::from_axis_angle(&Vector3::z_axis(), angle).to_homogeneous() * v
}

pub fn slerp<N: Real>(x: &Vec<N, U3>, y: &Vec<N, U3>, a: N) -> Vec<N, U3> {
    unimplemented!()
}
