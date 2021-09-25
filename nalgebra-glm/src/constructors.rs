use crate::aliases::{
    Qua, TMat, TMat2, TMat2x3, TMat2x4, TMat3, TMat3x2, TMat3x4, TMat4, TMat4x2, TMat4x3, TVec1,
    TVec2, TVec3, TVec4,
};
use crate::RealNumber;
use na::Scalar;

/// Creates a new 1D vector.
///
/// # Examples:
///
/// Construct a vector of `bool`s:
///
/// ```
/// # use nalgebra_glm as glm;
/// let v = glm::vec1(true);
/// ```
pub fn vec1<T: Scalar>(x: T) -> TVec1<T> {
    TVec1::new(x)
}

/// Creates a new 2D vector.
pub fn vec2<T: Scalar>(x: T, y: T) -> TVec2<T> {
    TVec2::new(x, y)
}

/// Creates a new 3D vector.
pub fn vec3<T: Scalar>(x: T, y: T, z: T) -> TVec3<T> {
    TVec3::new(x, y, z)
}

/// Creates a new 4D vector.
pub fn vec4<T: Scalar>(x: T, y: T, z: T, w: T) -> TVec4<T> {
    TVec4::new(x, y, z, w)
}

/// Create a new 2x2 matrix.
#[rustfmt::skip]
pub fn mat2<T: Scalar>(m11: T, m12: T,
                       m21: T, m22: T) -> TMat2<T> {
    TMat::<T, 2, 2>::new(
        m11, m12,
        m21, m22,
    )
}

/// Create a new 2x2 matrix.
#[rustfmt::skip]
pub fn mat2x2<T: Scalar>(m11: T, m12: T,
                         m21: T, m22: T) -> TMat2<T> {
    TMat::<T, 2, 2>::new(
        m11, m12,
        m21, m22,
    )
}

/// Create a new 2x3 matrix.
#[rustfmt::skip]
pub fn mat2x3<T: Scalar>(m11: T, m12: T, m13: T,
                         m21: T, m22: T, m23: T) -> TMat2x3<T> {
    TMat::<T, 2, 3>::new(
        m11, m12, m13,
        m21, m22, m23,
    )
}

/// Create a new 2x4 matrix.
#[rustfmt::skip]
pub fn mat2x4<T: Scalar>(m11: T, m12: T, m13: T, m14: T,
                         m21: T, m22: T, m23: T, m24: T) -> TMat2x4<T> {
    TMat::<T, 2, 4>::new(
        m11, m12, m13, m14,
        m21, m22, m23, m24,
    )
}

/// Create a new 3x3 matrix.
///
/// # Example
/// ```
/// # use nalgebra_glm::mat3;
/// let m = mat3(
///     1.0, 2.0, 3.0,
///     4.0, 5.0, 6.0,
///     7.0, 8.0, 9.0
/// );
/// assert!(
///     m.m11 == 1.0 && m.m12 == 2.0 && m.m13 == 3.0 &&
///     m.m21 == 4.0 && m.m22 == 5.0 && m.m23 == 6.0 &&
///     m.m31 == 7.0 && m.m32 == 8.0 && m.m33 == 9.0
/// );
/// ```
#[rustfmt::skip]
pub fn mat3<T: Scalar>(m11: T, m12: T, m13: T,
                       m21: T, m22: T, m23: T,
                       m31: T, m32: T, m33: T) -> TMat3<T> {
    TMat::<T, 3, 3>::new(
        m11, m12, m13,
        m21, m22, m23,
        m31, m32, m33,
    )
}

/// Create a new 3x2 matrix.
#[rustfmt::skip]
pub fn mat3x2<T: Scalar>(m11: T, m12: T,
                         m21: T, m22: T,
                         m31: T, m32: T) -> TMat3x2<T> {
    TMat::<T, 3, 2>::new(
        m11, m12,
        m21, m22,
        m31, m32,
    )
}

/// Create a new 3x3 matrix.
#[rustfmt::skip]
pub fn mat3x3<T: Scalar>(m11: T, m12: T, m13: T,
                         m21: T, m22: T, m23: T,
                         m31: T, m32: T, m33: T) -> TMat3<T> {
    TMat::<T, 3, 3>::new(
        m11, m12, m13,
        m21, m22, m23,
        m31, m32, m33,
    )
}

/// Create a new 3x4 matrix.
#[rustfmt::skip]
pub fn mat3x4<T: Scalar>(m11: T, m12: T, m13: T, m14: T,
                         m21: T, m22: T, m23: T, m24: T,
                         m31: T, m32: T, m33: T, m34: T) -> TMat3x4<T> {
    TMat::<T, 3, 4>::new(
        m11, m12, m13, m14,
        m21, m22, m23, m24,
        m31, m32, m33, m34,
    )
}

/// Create a new 4x2 matrix.
#[rustfmt::skip]
pub fn mat4x2<T: Scalar>(m11: T, m12: T,
                         m21: T, m22: T,
                         m31: T, m32: T,
                         m41: T, m42: T) -> TMat4x2<T> {
    TMat::<T, 4, 2>::new(
        m11, m12,
        m21, m22,
        m31, m32,
        m41, m42,
    )
}

/// Create a new 4x3 matrix.
#[rustfmt::skip]
pub fn mat4x3<T: Scalar>(m11: T, m12: T, m13: T,
                         m21: T, m22: T, m23: T,
                         m31: T, m32: T, m33: T,
                         m41: T, m42: T, m43: T) -> TMat4x3<T> {
    TMat::<T, 4, 3>::new(
        m11, m12, m13,
        m21, m22, m23,
        m31, m32, m33,
        m41, m42, m43,
    )
}

/// Create a new 4x4 matrix.
#[rustfmt::skip]
pub fn mat4x4<T: Scalar>(m11: T, m12: T, m13: T, m14: T,
                         m21: T, m22: T, m23: T, m24: T,
                         m31: T, m32: T, m33: T, m34: T,
                         m41: T, m42: T, m43: T, m44: T) -> TMat4<T> {
    TMat::<T, 4, 4>::new(
        m11, m12, m13, m14,
        m21, m22, m23, m24,
        m31, m32, m33, m34,
        m41, m42, m43, m44,
    )
}

/// Create a new 4x4 matrix.
#[rustfmt::skip]
pub fn mat4<T: Scalar>(m11: T, m12: T, m13: T, m14: T,
                       m21: T, m22: T, m23: T, m24: T,
                       m31: T, m32: T, m33: T, m34: T,
                       m41: T, m42: T, m43: T, m44: T) -> TMat4<T> {
    TMat::<T, 4, 4>::new(
        m11, m12, m13, m14,
        m21, m22, m23, m24,
        m31, m32, m33, m34,
        m41, m42, m43, m44,
    )
}

/// Creates a new quaternion.
pub fn quat<T: RealNumber>(x: T, y: T, z: T, w: T) -> Qua<T> {
    Qua::new(w, x, y, z)
}
