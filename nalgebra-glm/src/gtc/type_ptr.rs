use na::{Quaternion, RealField, Scalar};

use crate::aliases::{
    Qua, TMat, TMat2, TMat2x3, TMat2x4, TMat3, TMat3x2, TMat3x4, TMat4, TMat4x2, TMat4x3, TVec1,
    TVec2, TVec3, TVec4,
};
use crate::traits::Number;

/// Creates a 2x2 matrix from a slice arranged in column-major order.
pub fn make_mat2<T: Scalar>(ptr: &[T]) -> TMat2<T> {
    TMat2::from_column_slice(ptr)
}

/// Creates a 2x2 matrix from a slice arranged in column-major order.
pub fn make_mat2x2<T: Scalar>(ptr: &[T]) -> TMat2<T> {
    TMat2::from_column_slice(ptr)
}

/// Creates a 2x3 matrix from a slice arranged in column-major order.
pub fn make_mat2x3<T: Scalar>(ptr: &[T]) -> TMat2x3<T> {
    TMat2x3::from_column_slice(ptr)
}

/// Creates a 2x4 matrix from a slice arranged in column-major order.
pub fn make_mat2x4<T: Scalar>(ptr: &[T]) -> TMat2x4<T> {
    TMat2x4::from_column_slice(ptr)
}

/// Creates a 3 matrix from a slice arranged in column-major order.
pub fn make_mat3<T: Scalar>(ptr: &[T]) -> TMat3<T> {
    TMat3::from_column_slice(ptr)
}

/// Creates a 3x2 matrix from a slice arranged in column-major order.
pub fn make_mat3x2<T: Scalar>(ptr: &[T]) -> TMat3x2<T> {
    TMat3x2::from_column_slice(ptr)
}

/// Creates a 3x3 matrix from a slice arranged in column-major order.
pub fn make_mat3x3<T: Scalar>(ptr: &[T]) -> TMat3<T> {
    TMat3::from_column_slice(ptr)
}

/// Creates a 3x4 matrix from a slice arranged in column-major order.
pub fn make_mat3x4<T: Scalar>(ptr: &[T]) -> TMat3x4<T> {
    TMat3x4::from_column_slice(ptr)
}

/// Creates a 4x4 matrix from a slice arranged in column-major order.
pub fn make_mat4<T: Scalar>(ptr: &[T]) -> TMat4<T> {
    TMat4::from_column_slice(ptr)
}

/// Creates a 4x2 matrix from a slice arranged in column-major order.
pub fn make_mat4x2<T: Scalar>(ptr: &[T]) -> TMat4x2<T> {
    TMat4x2::from_column_slice(ptr)
}

/// Creates a 4x3 matrix from a slice arranged in column-major order.
pub fn make_mat4x3<T: Scalar>(ptr: &[T]) -> TMat4x3<T> {
    TMat4x3::from_column_slice(ptr)
}

/// Creates a 4x4 matrix from a slice arranged in column-major order.
pub fn make_mat4x4<T: Scalar>(ptr: &[T]) -> TMat4<T> {
    TMat4::from_column_slice(ptr)
}

/// Converts a 2x2 matrix to a 3x3 matrix.
pub fn mat2_to_mat3<T: Number>(m: &TMat2<T>) -> TMat3<T> {
    let _0 = T::zero();
    let _1 = T::one();

    TMat3::new(m.m11, m.m12, _0, m.m21, m.m22, _0, _0, _0, _1)
}

/// Converts a 3x3 matrix to a 2x2 matrix.
pub fn mat3_to_mat2<T: Scalar>(m: &TMat3<T>) -> TMat2<T> {
    TMat2::new(m.m11.clone(), m.m12.clone(), m.m21.clone(), m.m22.clone())
}

/// Converts a 3x3 matrix to a 4x4 matrix.
pub fn mat3_to_mat4<T: Number>(m: &TMat3<T>) -> TMat4<T> {
    let _0 = T::zero();
    let _1 = T::one();

    TMat4::new(
        m.m11, m.m12, m.m13, _0, m.m21, m.m22, m.m23, _0, m.m31, m.m32, m.m33, _0, _0, _0, _0, _1,
    )
}

/// Converts a 4x4 matrix to a 3x3 matrix.
pub fn mat4_to_mat3<T: Scalar>(m: &TMat4<T>) -> TMat3<T> {
    TMat3::new(
        m.m11.clone(),
        m.m12.clone(),
        m.m13.clone(),
        m.m21.clone(),
        m.m22.clone(),
        m.m23.clone(),
        m.m31.clone(),
        m.m32.clone(),
        m.m33.clone(),
    )
}

/// Converts a 2x2 matrix to a 4x4 matrix.
pub fn mat2_to_mat4<T: Number>(m: &TMat2<T>) -> TMat4<T> {
    let _0 = T::zero();
    let _1 = T::one();

    TMat4::new(
        m.m11, m.m12, _0, _0, m.m21, m.m22, _0, _0, _0, _0, _1, _0, _0, _0, _0, _1,
    )
}

/// Converts a 4x4 matrix to a 2x2 matrix.
pub fn mat4_to_mat2<T: Scalar>(m: &TMat4<T>) -> TMat2<T> {
    TMat2::new(m.m11.clone(), m.m12.clone(), m.m21.clone(), m.m22.clone())
}

/// Creates a quaternion from a slice arranged as `[x, y, z, w]`.
pub fn make_quat<T: RealField>(ptr: &[T]) -> Qua<T> {
    Quaternion::from(TVec4::from_column_slice(ptr))
}

/// Creates a 1D vector from a slice.
///
/// # See also:
///
/// * [`make_vec2`](fn.make_vec2.html)
/// * [`make_vec3`](fn.make_vec3.html)
/// * [`make_vec4`](fn.make_vec4.html)
pub fn make_vec1<T: Scalar>(v: &TVec1<T>) -> TVec1<T> {
    v.clone()
}

/// Creates a 1D vector from another vector.
///
/// # See also:
///
/// * [`vec1_to_vec1`](fn.vec1_to_vec1.html)
/// * [`vec3_to_vec1`](fn.vec3_to_vec1.html)
/// * [`vec4_to_vec1`](fn.vec4_to_vec1.html)
/// * [`vec1_to_vec2`](fn.vec1_to_vec2.html)
/// * [`vec1_to_vec3`](fn.vec1_to_vec3.html)
/// * [`vec1_to_vec4`](fn.vec1_to_vec4.html)
pub fn vec2_to_vec1<T: Scalar>(v: &TVec2<T>) -> TVec1<T> {
    TVec1::new(v.x.clone())
}

/// Creates a 1D vector from another vector.
///
/// # See also:
///
/// * [`vec1_to_vec1`](fn.vec1_to_vec1.html)
/// * [`vec2_to_vec1`](fn.vec2_to_vec1.html)
/// * [`vec4_to_vec1`](fn.vec4_to_vec1.html)
/// * [`vec1_to_vec2`](fn.vec1_to_vec2.html)
/// * [`vec1_to_vec3`](fn.vec1_to_vec3.html)
/// * [`vec1_to_vec4`](fn.vec1_to_vec4.html)
pub fn vec3_to_vec1<T: Scalar>(v: &TVec3<T>) -> TVec1<T> {
    TVec1::new(v.x.clone())
}

/// Creates a 1D vector from another vector.
///
/// # See also:
///
/// * [`vec1_to_vec1`](fn.vec1_to_vec1.html)
/// * [`vec2_to_vec1`](fn.vec2_to_vec1.html)
/// * [`vec3_to_vec1`](fn.vec3_to_vec1.html)
/// * [`vec1_to_vec2`](fn.vec1_to_vec2.html)
/// * [`vec1_to_vec3`](fn.vec1_to_vec3.html)
/// * [`vec1_to_vec4`](fn.vec1_to_vec4.html)
pub fn vec4_to_vec1<T: Scalar>(v: &TVec4<T>) -> TVec1<T> {
    TVec1::new(v.x.clone())
}

/// Creates a 2D vector from another vector.
///
/// Missing components, if any, are set to 0.
///
/// # See also:
///
/// * [`vec3_to_vec2`](fn.vec3_to_vec2.html)
/// * [`vec4_to_vec2`](fn.vec4_to_vec2.html)
/// * [`vec2_to_vec1`](fn.vec2_to_vec1.html)
/// * [`vec2_to_vec2`](fn.vec2_to_vec2.html)
/// * [`vec2_to_vec3`](fn.vec2_to_vec3.html)
/// * [`vec2_to_vec4`](fn.vec2_to_vec4.html)
pub fn vec1_to_vec2<T: Number>(v: &TVec1<T>) -> TVec2<T> {
    TVec2::new(v.x.clone(), T::zero())
}

/// Creates a 2D vector from another vector.
///
/// # See also:
///
/// * [`vec1_to_vec2`](fn.vec1_to_vec2.html)
/// * [`vec3_to_vec2`](fn.vec3_to_vec2.html)
/// * [`vec4_to_vec2`](fn.vec4_to_vec2.html)
/// * [`vec2_to_vec1`](fn.vec2_to_vec1.html)
/// * [`vec2_to_vec2`](fn.vec2_to_vec2.html)
/// * [`vec2_to_vec3`](fn.vec2_to_vec3.html)
/// * [`vec2_to_vec4`](fn.vec2_to_vec4.html)
pub fn vec2_to_vec2<T: Scalar>(v: &TVec2<T>) -> TVec2<T> {
    v.clone()
}

/// Creates a 2D vector from another vector.
///
/// # See also:
///
/// * [`vec1_to_vec2`](fn.vec1_to_vec2.html)
/// * [`vec4_to_vec2`](fn.vec4_to_vec2.html)
/// * [`vec2_to_vec1`](fn.vec2_to_vec1.html)
/// * [`vec2_to_vec2`](fn.vec2_to_vec2.html)
/// * [`vec2_to_vec3`](fn.vec2_to_vec3.html)
/// * [`vec2_to_vec4`](fn.vec2_to_vec4.html)
pub fn vec3_to_vec2<T: Scalar>(v: &TVec3<T>) -> TVec2<T> {
    TVec2::new(v.x.clone(), v.y.clone())
}

/// Creates a 2D vector from another vector.
///
/// # See also:
///
/// * [`vec1_to_vec2`](fn.vec1_to_vec2.html)
/// * [`vec3_to_vec2`](fn.vec4_to_vec2.html)
/// * [`vec2_to_vec1`](fn.vec2_to_vec1.html)
/// * [`vec2_to_vec2`](fn.vec2_to_vec2.html)
/// * [`vec2_to_vec3`](fn.vec2_to_vec3.html)
/// * [`vec2_to_vec4`](fn.vec2_to_vec4.html)
pub fn vec4_to_vec2<T: Scalar>(v: &TVec4<T>) -> TVec2<T> {
    TVec2::new(v.x.clone(), v.y.clone())
}

/// Creates a 2D vector from a slice.
///
/// # See also:
///
/// * [`make_vec1`](fn.make_vec1.html)
/// * [`make_vec3`](fn.make_vec3.html)
/// * [`make_vec4`](fn.make_vec4.html)
pub fn make_vec2<T: Scalar>(ptr: &[T]) -> TVec2<T> {
    TVec2::from_column_slice(ptr)
}

/// Creates a 3D vector from another vector.
///
/// Missing components, if any, are set to 0.
///
/// # See also:
///
/// * [`vec2_to_vec3`](fn.vec2_to_vec3.html)
/// * [`vec3_to_vec3`](fn.vec3_to_vec3.html)
/// * [`vec4_to_vec3`](fn.vec4_to_vec3.html)
/// * [`vec1_to_vec2`](fn.vec1_to_vec2.html)
/// * [`vec1_to_vec4`](fn.vec1_to_vec4.html)
pub fn vec1_to_vec3<T: Number>(v: &TVec1<T>) -> TVec3<T> {
    TVec3::new(v.x.clone(), T::zero(), T::zero())
}

/// Creates a 3D vector from another vector.
///
/// Missing components, if any, are set to 0.
///
/// # See also:
///
/// * [`vec1_to_vec3`](fn.vec1_to_vec3.html)
/// * [`vec3_to_vec3`](fn.vec3_to_vec3.html)
/// * [`vec4_to_vec3`](fn.vec4_to_vec3.html)
/// * [`vec3_to_vec1`](fn.vec3_to_vec1.html)
/// * [`vec3_to_vec2`](fn.vec3_to_vec2.html)
/// * [`vec3_to_vec4`](fn.vec3_to_vec4.html)
pub fn vec2_to_vec3<T: Number>(v: &TVec2<T>) -> TVec3<T> {
    TVec3::new(v.x.clone(), v.y.clone(), T::zero())
}

/// Creates a 3D vector from another vector.
///
/// # See also:
///
/// * [`vec1_to_vec3`](fn.vec1_to_vec3.html)
/// * [`vec2_to_vec3`](fn.vec2_to_vec3.html)
/// * [`vec4_to_vec3`](fn.vec4_to_vec3.html)
/// * [`vec3_to_vec1`](fn.vec3_to_vec1.html)
/// * [`vec3_to_vec2`](fn.vec3_to_vec2.html)
/// * [`vec3_to_vec4`](fn.vec3_to_vec4.html)
pub fn vec3_to_vec3<T: Scalar>(v: &TVec3<T>) -> TVec3<T> {
    v.clone()
}

/// Creates a 3D vector from another vector.
///
/// # See also:
///
/// * [`vec1_to_vec3`](fn.vec1_to_vec3.html)
/// * [`vec2_to_vec3`](fn.vec2_to_vec3.html)
/// * [`vec3_to_vec3`](fn.vec3_to_vec3.html)
/// * [`vec3_to_vec1`](fn.vec3_to_vec1.html)
/// * [`vec3_to_vec2`](fn.vec3_to_vec2.html)
/// * [`vec3_to_vec4`](fn.vec3_to_vec4.html)
pub fn vec4_to_vec3<T: Scalar>(v: &TVec4<T>) -> TVec3<T> {
    TVec3::new(v.x.clone(), v.y.clone(), v.z.clone())
}

/// Creates a 3D vector from another vector.
///
/// # See also:
///
/// * [`make_vec1`](fn.make_vec1.html)
/// * [`make_vec2`](fn.make_vec2.html)
/// * [`make_vec4`](fn.make_vec4.html)
pub fn make_vec3<T: Scalar>(ptr: &[T]) -> TVec3<T> {
    TVec3::from_column_slice(ptr)
}

/// Creates a 4D vector from another vector.
///
/// Missing components, if any, are set to 0.
///
/// # See also:
///
/// * [`vec2_to_vec4`](fn.vec2_to_vec4.html)
/// * [`vec3_to_vec4`](fn.vec3_to_vec4.html)
/// * [`vec4_to_vec4`](fn.vec4_to_vec4.html)
/// * [`vec1_to_vec2`](fn.vec1_to_vec2.html)
/// * [`vec1_to_vec3`](fn.vec1_to_vec3.html)
/// * [`vec1_to_vec4`](fn.vec1_to_vec4.html)
pub fn vec1_to_vec4<T: Number>(v: &TVec1<T>) -> TVec4<T> {
    TVec4::new(v.x, T::zero(), T::zero(), T::zero())
}

/// Creates a 4D vector from another vector.
///
/// Missing components, if any, are set to 0.
///
/// # See also:
///
/// * [`vec1_to_vec4`](fn.vec1_to_vec4.html)
/// * [`vec3_to_vec4`](fn.vec3_to_vec4.html)
/// * [`vec4_to_vec4`](fn.vec4_to_vec4.html)
/// * [`vec2_to_vec1`](fn.vec2_to_vec1.html)
/// * [`vec2_to_vec2`](fn.vec2_to_vec2.html)
/// * [`vec2_to_vec3`](fn.vec2_to_vec3.html)
pub fn vec2_to_vec4<T: Number>(v: &TVec2<T>) -> TVec4<T> {
    TVec4::new(v.x, v.y, T::zero(), T::zero())
}

/// Creates a 4D vector from another vector.
///
/// Missing components, if any, are set to 0.
///
/// # See also:
///
/// * [`vec1_to_vec4`](fn.vec1_to_vec4.html)
/// * [`vec2_to_vec4`](fn.vec2_to_vec4.html)
/// * [`vec4_to_vec4`](fn.vec4_to_vec4.html)
/// * [`vec3_to_vec1`](fn.vec3_to_vec1.html)
/// * [`vec3_to_vec2`](fn.vec3_to_vec2.html)
/// * [`vec3_to_vec3`](fn.vec3_to_vec3.html)
pub fn vec3_to_vec4<T: Number>(v: &TVec3<T>) -> TVec4<T> {
    TVec4::new(v.x, v.y, v.z, T::zero())
}

/// Creates a 4D vector from another vector.
///
/// # See also:
///
/// * [`vec1_to_vec4`](fn.vec1_to_vec4.html)
/// * [`vec2_to_vec4`](fn.vec2_to_vec4.html)
/// * [`vec3_to_vec4`](fn.vec3_to_vec4.html)
/// * [`vec4_to_vec1`](fn.vec4_to_vec1.html)
/// * [`vec4_to_vec2`](fn.vec4_to_vec2.html)
/// * [`vec4_to_vec3`](fn.vec4_to_vec3.html)
pub fn vec4_to_vec4<T: Scalar>(v: &TVec4<T>) -> TVec4<T> {
    v.clone()
}

/// Creates a 4D vector from another vector.
///
/// # See also:
///
/// * [`make_vec1`](fn.make_vec1.html)
/// * [`make_vec2`](fn.make_vec2.html)
/// * [`make_vec3`](fn.make_vec3.html)
pub fn make_vec4<T: Scalar>(ptr: &[T]) -> TVec4<T> {
    TVec4::from_column_slice(ptr)
}

/// Converts a matrix or vector to a slice arranged in column-major order.
pub fn value_ptr<T: Scalar, const R: usize, const C: usize>(x: &TMat<T, R, C>) -> &[T] {
    x.as_slice()
}

/// Converts a matrix or vector to a mutable slice arranged in column-major order.
pub fn value_ptr_mut<T: Scalar, const R: usize, const C: usize>(x: &mut TMat<T, R, C>) -> &mut [T] {
    x.as_mut_slice()
}
