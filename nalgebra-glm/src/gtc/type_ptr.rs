use na::{DefaultAllocator, Quaternion, Real, Scalar};

use aliases::{
    Qua, TMat, TMat2, TMat2x3, TMat2x4, TMat3, TMat3x2, TMat3x4, TMat4, TMat4x2, TMat4x3, TVec1,
    TVec2, TVec3, TVec4,
};
use traits::{Alloc, Dimension, Number};

/// Creates a 2x2 matrix from a slice arranged in column-major order.
pub fn make_mat2<N: Scalar>(ptr: &[N]) -> TMat2<N> {
    TMat2::from_column_slice(ptr)
}

/// Creates a 2x2 matrix from a slice arranged in column-major order.
pub fn make_mat2x2<N: Scalar>(ptr: &[N]) -> TMat2<N> {
    TMat2::from_column_slice(ptr)
}

/// Creates a 2x3 matrix from a slice arranged in column-major order.
pub fn make_mat2x3<N: Scalar>(ptr: &[N]) -> TMat2x3<N> {
    TMat2x3::from_column_slice(ptr)
}

/// Creates a 2x4 matrix from a slice arranged in column-major order.
pub fn make_mat2x4<N: Scalar>(ptr: &[N]) -> TMat2x4<N> {
    TMat2x4::from_column_slice(ptr)
}

/// Creates a 3 matrix from a slice arranged in column-major order.
pub fn make_mat3<N: Scalar>(ptr: &[N]) -> TMat3<N> {
    TMat3::from_column_slice(ptr)
}

/// Creates a 3x2 matrix from a slice arranged in column-major order.
pub fn make_mat3x2<N: Scalar>(ptr: &[N]) -> TMat3x2<N> {
    TMat3x2::from_column_slice(ptr)
}

/// Creates a 3x3 matrix from a slice arranged in column-major order.
pub fn make_mat3x3<N: Scalar>(ptr: &[N]) -> TMat3<N> {
    TMat3::from_column_slice(ptr)
}

/// Creates a 3x4 matrix from a slice arranged in column-major order.
pub fn make_mat3x4<N: Scalar>(ptr: &[N]) -> TMat3x4<N> {
    TMat3x4::from_column_slice(ptr)
}

/// Creates a 4x4 matrix from a slice arranged in column-major order.
pub fn make_mat4<N: Scalar>(ptr: &[N]) -> TMat4<N> {
    TMat4::from_column_slice(ptr)
}

/// Creates a 4x2 matrix from a slice arranged in column-major order.
pub fn make_mat4x2<N: Scalar>(ptr: &[N]) -> TMat4x2<N> {
    TMat4x2::from_column_slice(ptr)
}

/// Creates a 4x3 matrix from a slice arranged in column-major order.
pub fn make_mat4x3<N: Scalar>(ptr: &[N]) -> TMat4x3<N> {
    TMat4x3::from_column_slice(ptr)
}

/// Creates a 4x4 matrix from a slice arranged in column-major order.
pub fn make_mat4x4<N: Scalar>(ptr: &[N]) -> TMat4<N> {
    TMat4::from_column_slice(ptr)
}

/// Converts a 2x2 matrix to a 3x3 matrix.
pub fn mat2_to_mat3<N: Number>(m: &TMat2<N>) -> TMat3<N> {
    let _0 = N::zero();
    let _1 = N::one();

    TMat3::new(m.m11, m.m12, _0, m.m21, m.m22, _0, _0, _0, _1)
}

/// Converts a 3x3 matrix to a 2x2 matrix.
pub fn mat3_to_mat2<N: Scalar>(m: &TMat3<N>) -> TMat2<N> {
    TMat2::new(m.m11, m.m12, m.m21, m.m22)
}

/// Converts a 3x3 matrix to a 4x4 matrix.
pub fn mat3_to_mat4<N: Number>(m: &TMat3<N>) -> TMat4<N> {
    let _0 = N::zero();
    let _1 = N::one();

    TMat4::new(
        m.m11, m.m12, m.m13, _0, m.m21, m.m22, m.m23, _0, m.m31, m.m32, m.m33, _0, _0, _0, _0, _1,
    )
}

/// Converts a 4x4 matrix to a 3x3 matrix.
pub fn mat4_to_mat3<N: Scalar>(m: &TMat4<N>) -> TMat3<N> {
    TMat3::new(
        m.m11, m.m12, m.m13, m.m21, m.m22, m.m23, m.m31, m.m32, m.m33,
    )
}

/// Converts a 2x2 matrix to a 4x4 matrix.
pub fn mat2_to_mat4<N: Number>(m: &TMat2<N>) -> TMat4<N> {
    let _0 = N::zero();
    let _1 = N::one();

    TMat4::new(
        m.m11, m.m12, _0, _0, m.m21, m.m22, _0, _0, _0, _0, _1, _0, _0, _0, _0, _1,
    )
}

/// Converts a 4x4 matrix to a 2x2 matrix.
pub fn mat4_to_mat2<N: Scalar>(m: &TMat4<N>) -> TMat2<N> {
    TMat2::new(m.m11, m.m12, m.m21, m.m22)
}

/// Creates a quaternion from a slice arranged as `[x, y, z, w]`.
pub fn make_quat<N: Real>(ptr: &[N]) -> Qua<N> {
    Quaternion::from(TVec4::from_column_slice(ptr))
}

/// Creates a 1D vector from a slice.
///
/// # See also:
///
/// * [`make_vec2`](fn.make_vec2.html)
/// * [`make_vec3`](fn.make_vec3.html)
/// * [`make_vec4`](fn.make_vec4.html)
pub fn make_vec1<N: Scalar>(v: &TVec1<N>) -> TVec1<N> {
    *v
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
pub fn vec2_to_vec1<N: Scalar>(v: &TVec2<N>) -> TVec1<N> {
    TVec1::new(v.x)
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
pub fn vec3_to_vec1<N: Scalar>(v: &TVec3<N>) -> TVec1<N> {
    TVec1::new(v.x)
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
pub fn vec4_to_vec1<N: Scalar>(v: &TVec4<N>) -> TVec1<N> {
    TVec1::new(v.x)
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
pub fn vec1_to_vec2<N: Number>(v: &TVec1<N>) -> TVec2<N> {
    TVec2::new(v.x, N::zero())
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
pub fn vec2_to_vec2<N: Scalar>(v: &TVec2<N>) -> TVec2<N> {
    *v
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
pub fn vec3_to_vec2<N: Scalar>(v: &TVec3<N>) -> TVec2<N> {
    TVec2::new(v.x, v.y)
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
pub fn vec4_to_vec2<N: Scalar>(v: &TVec4<N>) -> TVec2<N> {
    TVec2::new(v.x, v.y)
}

/// Creates a 2D vector from a slice.
///
/// # See also:
///
/// * [`make_vec1`](fn.make_vec1.html)
/// * [`make_vec3`](fn.make_vec3.html)
/// * [`make_vec4`](fn.make_vec4.html)
pub fn make_vec2<N: Scalar>(ptr: &[N]) -> TVec2<N> {
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
pub fn vec1_to_vec3<N: Number>(v: &TVec1<N>) -> TVec3<N> {
    TVec3::new(v.x, N::zero(), N::zero())
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
pub fn vec2_to_vec3<N: Number>(v: &TVec2<N>) -> TVec3<N> {
    TVec3::new(v.x, v.y, N::zero())
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
pub fn vec3_to_vec3<N: Scalar>(v: &TVec3<N>) -> TVec3<N> {
    *v
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
pub fn vec4_to_vec3<N: Scalar>(v: &TVec4<N>) -> TVec3<N> {
    TVec3::new(v.x, v.y, v.z)
}

/// Creates a 3D vector from another vector.
///
/// # See also:
///
/// * [`make_vec1`](fn.make_vec1.html)
/// * [`make_vec2`](fn.make_vec2.html)
/// * [`make_vec4`](fn.make_vec4.html)
pub fn make_vec3<N: Scalar>(ptr: &[N]) -> TVec3<N> {
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
pub fn vec1_to_vec4<N: Number>(v: &TVec1<N>) -> TVec4<N> {
    TVec4::new(v.x, N::zero(), N::zero(), N::zero())
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
pub fn vec2_to_vec4<N: Number>(v: &TVec2<N>) -> TVec4<N> {
    TVec4::new(v.x, v.y, N::zero(), N::zero())
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
pub fn vec3_to_vec4<N: Number>(v: &TVec3<N>) -> TVec4<N> {
    TVec4::new(v.x, v.y, v.z, N::zero())
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
pub fn vec4_to_vec4<N: Scalar>(v: &TVec4<N>) -> TVec4<N> {
    *v
}

/// Creates a 4D vector from another vector.
///
/// # See also:
///
/// * [`make_vec1`](fn.make_vec1.html)
/// * [`make_vec2`](fn.make_vec2.html)
/// * [`make_vec3`](fn.make_vec3.html)
pub fn make_vec4<N: Scalar>(ptr: &[N]) -> TVec4<N> {
    TVec4::from_column_slice(ptr)
}

/// Converts a matrix or vector to a slice arranged in column-major order.
pub fn value_ptr<N: Scalar, R: Dimension, C: Dimension>(x: &TMat<N, R, C>) -> &[N]
where DefaultAllocator: Alloc<N, R, C> {
    x.as_slice()
}

/// Converts a matrix or vector to a mutable slice arranged in column-major order.
pub fn value_ptr_mut<N: Scalar, R: Dimension, C: Dimension>(x: &mut TMat<N, R, C>) -> &mut [N]
where DefaultAllocator: Alloc<N, R, C> {
    x.as_mut_slice()
}
