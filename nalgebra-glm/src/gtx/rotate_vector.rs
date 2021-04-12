use na::{RealField, Rotation3, Unit, UnitComplex};

use crate::aliases::{TMat4, TVec2, TVec3, TVec4};

/// Build the rotation matrix needed to align `normal` and `up`.
pub fn orientation<T: RealField>(normal: &TVec3<T>, up: &TVec3<T>) -> TMat4<T> {
    if let Some(r) = Rotation3::rotation_between(normal, up) {
        r.to_homogeneous()
    } else {
        TMat4::identity()
    }
}

/// Rotate a two dimensional vector.
pub fn rotate_vec2<T: RealField>(v: &TVec2<T>, angle: T) -> TVec2<T> {
    UnitComplex::new(angle) * v
}

/// Rotate a three dimensional vector around an axis.
pub fn rotate_vec3<T: RealField>(v: &TVec3<T>, angle: T, normal: &TVec3<T>) -> TVec3<T> {
    Rotation3::from_axis_angle(&Unit::new_normalize(*normal), angle) * v
}

/// Rotate a thee dimensional vector in homogeneous coordinates around an axis.
pub fn rotate_vec4<T: RealField>(v: &TVec4<T>, angle: T, normal: &TVec3<T>) -> TVec4<T> {
    Rotation3::from_axis_angle(&Unit::new_normalize(*normal), angle).to_homogeneous() * v
}

/// Rotate a three dimensional vector around the `X` axis.
pub fn rotate_x_vec3<T: RealField>(v: &TVec3<T>, angle: T) -> TVec3<T> {
    Rotation3::from_axis_angle(&TVec3::x_axis(), angle) * v
}

/// Rotate a three dimensional vector in homogeneous coordinates around the `X` axis.
pub fn rotate_x_vec4<T: RealField>(v: &TVec4<T>, angle: T) -> TVec4<T> {
    Rotation3::from_axis_angle(&TVec3::x_axis(), angle).to_homogeneous() * v
}

/// Rotate a three dimensional vector around the `Y` axis.
pub fn rotate_y_vec3<T: RealField>(v: &TVec3<T>, angle: T) -> TVec3<T> {
    Rotation3::from_axis_angle(&TVec3::y_axis(), angle) * v
}

/// Rotate a three dimensional vector in homogeneous coordinates around the `Y` axis.
pub fn rotate_y_vec4<T: RealField>(v: &TVec4<T>, angle: T) -> TVec4<T> {
    Rotation3::from_axis_angle(&TVec3::y_axis(), angle).to_homogeneous() * v
}

/// Rotate a three dimensional vector around the `Z` axis.
pub fn rotate_z_vec3<T: RealField>(v: &TVec3<T>, angle: T) -> TVec3<T> {
    Rotation3::from_axis_angle(&TVec3::z_axis(), angle) * v
}

/// Rotate a three dimensional vector in homogeneous coordinates around the `Z` axis.
pub fn rotate_z_vec4<T: RealField>(v: &TVec4<T>, angle: T) -> TVec4<T> {
    Rotation3::from_axis_angle(&TVec3::z_axis(), angle).to_homogeneous() * v
}

/// Computes a spherical linear interpolation between the vectors `x` and `y` assumed to be normalized.
pub fn slerp<T: RealField>(x: &TVec3<T>, y: &TVec3<T>, a: T) -> TVec3<T> {
    Unit::new_unchecked(*x)
        .slerp(&Unit::new_unchecked(*y), a)
        .into_inner()
}
