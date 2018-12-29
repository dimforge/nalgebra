use na::{Real, Rotation3, Unit, UnitComplex};

use aliases::{TMat4, TVec2, TVec3, TVec4};

/// Build the rotation matrix needed to align `normal` and `up`.
pub fn orientation<N: Real>(normal: &TVec3<N>, up: &TVec3<N>) -> TMat4<N> {
    if let Some(r) = Rotation3::rotation_between(normal, up) {
        r.to_homogeneous()
    } else {
        TMat4::identity()
    }
}

/// Rotate a two dimensional vector.
pub fn rotate_vec2<N: Real>(v: &TVec2<N>, angle: N) -> TVec2<N> {
    UnitComplex::new(angle) * v
}

/// Rotate a three dimensional vector around an axis.
pub fn rotate_vec3<N: Real>(v: &TVec3<N>, angle: N, normal: &TVec3<N>) -> TVec3<N> {
    Rotation3::from_axis_angle(&Unit::new_normalize(*normal), angle) * v
}

/// Rotate a thee dimensional vector in homogeneous coordinates around an axis.
pub fn rotate_vec4<N: Real>(v: &TVec4<N>, angle: N, normal: &TVec3<N>) -> TVec4<N> {
    Rotation3::from_axis_angle(&Unit::new_normalize(*normal), angle).to_homogeneous() * v
}

/// Rotate a three dimensional vector around the `X` axis.
pub fn rotate_x_vec3<N: Real>(v: &TVec3<N>, angle: N) -> TVec3<N> {
    Rotation3::from_axis_angle(&TVec3::x_axis(), angle) * v
}

/// Rotate a three dimensional vector in homogeneous coordinates around the `X` axis.
pub fn rotate_x_vec4<N: Real>(v: &TVec4<N>, angle: N) -> TVec4<N> {
    Rotation3::from_axis_angle(&TVec3::x_axis(), angle).to_homogeneous() * v
}

/// Rotate a three dimensional vector around the `Y` axis.
pub fn rotate_y_vec3<N: Real>(v: &TVec3<N>, angle: N) -> TVec3<N> {
    Rotation3::from_axis_angle(&TVec3::y_axis(), angle) * v
}

/// Rotate a three dimensional vector in homogeneous coordinates around the `Y` axis.
pub fn rotate_y_vec4<N: Real>(v: &TVec4<N>, angle: N) -> TVec4<N> {
    Rotation3::from_axis_angle(&TVec3::y_axis(), angle).to_homogeneous() * v
}

/// Rotate a three dimensional vector around the `Z` axis.
pub fn rotate_z_vec3<N: Real>(v: &TVec3<N>, angle: N) -> TVec3<N> {
    Rotation3::from_axis_angle(&TVec3::z_axis(), angle) * v
}

/// Rotate a three dimensional vector in homogeneous coordinates around the `Z` axis.
pub fn rotate_z_vec4<N: Real>(v: &TVec4<N>, angle: N) -> TVec4<N> {
    Rotation3::from_axis_angle(&TVec3::z_axis(), angle).to_homogeneous() * v
}

/// Computes a spherical linear interpolation between the vectors `x` and `y` assumed to be normalized.
pub fn slerp<N: Real>(x: &TVec3<N>, y: &TVec3<N>, a: N) -> TVec3<N> {
    Unit::new_unchecked(*x)
        .slerp(&Unit::new_unchecked(*y), a)
        .into_inner()
}
