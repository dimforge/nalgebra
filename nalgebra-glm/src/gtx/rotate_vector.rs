use na::{Real, U2, U3, U4, Rotation3, Vector3, Unit, UnitComplex};

use aliases::{Vec, Mat};

/// Build the rotation matrix needed to align `normal` and `up`.
pub fn orientation<N: Real>(normal: &Vec<N, U3>, up: &Vec<N, U3>) -> Mat<N, U4, U4> {
    if let Some(r) = Rotation3::rotation_between(normal, up) {
        r.to_homogeneous()
    } else {
        Mat::<N, U4, U4>::identity()
    }
}

/// Rotate a two dimensional vector.
pub fn rotate_vec2<N: Real>(v: &Vec<N, U2>, angle: N) -> Vec<N, U2> {
    UnitComplex::new(angle) * v
}

/// Rotate a three dimensional vector around an axis.
pub fn rotate_vec3<N: Real>(v: &Vec<N, U3>, angle: N, normal: &Vec<N, U3>) -> Vec<N, U3> {
    Rotation3::from_axis_angle(&Unit::new_normalize(*normal), angle) * v
}

/// Rotate a thee dimensional vector in homogeneous coordinates around an axis.
pub fn rotate_vec4<N: Real>(v: &Vec<N, U4>, angle: N, normal: &Vec<N, U3>) -> Vec<N, U4> {
    Rotation3::from_axis_angle(&Unit::new_normalize(*normal), angle).to_homogeneous() * v
}

/// Rotate a three dimensional vector around the `X` axis.
pub fn rotate_x_vec3<N: Real>(v: &Vec<N, U3>, angle: N) -> Vec<N, U3> {
    Rotation3::from_axis_angle(&Vector3::x_axis(), angle) * v
}

/// Rotate a three dimensional vector in homogeneous coordinates around the `X` axis.
pub fn rotate_x<N: Real>(v: &Vec<N, U4>, angle: N) -> Vec<N, U4> {
    Rotation3::from_axis_angle(&Vector3::x_axis(), angle).to_homogeneous() * v
}

/// Rotate a three dimensional vector around the `Y` axis.
pub fn rotate_y_vec3<N: Real>(v: &Vec<N, U3>, angle: N) -> Vec<N, U3> {
    Rotation3::from_axis_angle(&Vector3::y_axis(), angle) * v
}

/// Rotate a three dimensional vector in homogeneous coordinates around the `Y` axis.
pub fn rotate_y<N: Real>(v: &Vec<N, U4>, angle: N) -> Vec<N, U4> {
    Rotation3::from_axis_angle(&Vector3::y_axis(), angle).to_homogeneous() * v
}

/// Rotate a three dimensional vector around the `Z` axis.
pub fn rotate_z_vec3<N: Real>(v: &Vec<N, U3>, angle: N) -> Vec<N, U3> {
    Rotation3::from_axis_angle(&Vector3::z_axis(), angle) * v
}

/// Rotate a three dimensional vector in homogeneous coordinates around the `Z` axis.
pub fn rotate_z<N: Real>(v: &Vec<N, U4>, angle: N) -> Vec<N, U4> {
    Rotation3::from_axis_angle(&Vector3::z_axis(), angle).to_homogeneous() * v
}

/// Computes a spehical linear interpolation between the vectors `x` and `y` assumed to be normalized.
pub fn slerp<N: Real>(x: &Vec<N, U3>, y: &Vec<N, U3>, a: N) -> Vec<N, U3> {
    Unit::new_unchecked(*x).slerp(&Unit::new_unchecked(*y), a).unwrap()
}
