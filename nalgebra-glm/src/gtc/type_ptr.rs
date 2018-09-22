use na::{Scalar, Real, U1, U2, U3, U4, DefaultAllocator,
         Quaternion, Matrix2, Matrix3, Matrix4, Vector1, Vector2, Vector3, Vector4,
         Matrix2x3, Matrix2x4, Matrix3x2, Matrix3x4, Matrix4x2, Matrix4x3};

use traits::{Number, Alloc, Dimension};
use aliases::{Qua, Vec, Mat};

/// Creates a 2x2 matrix from a slice arranged in column-major order.
pub fn make_mat2<N: Scalar>(ptr: &[N]) -> Mat<N, U2, U2> {
    Matrix2::from_column_slice(ptr)
}

/// Creates a 2x2 matrix from a slice arranged in column-major order.
pub fn make_mat2x2<N: Scalar>(ptr: &[N]) -> Mat<N, U2, U2> {
    Matrix2::from_column_slice(ptr)
}

/// Creates a 2x3 matrix from a slice arranged in column-major order.
pub fn make_mat2x3<N: Scalar>(ptr: &[N]) -> Mat<N, U2, U3> {
    Matrix2x3::from_column_slice(ptr)
}

/// Creates a 2x4 matrix from a slice arranged in column-major order.
pub fn make_mat2x4<N: Scalar>(ptr: &[N]) -> Mat<N, U2, U4> {
    Matrix2x4::from_column_slice(ptr)
}

/// Creates a 3 matrix from a slice arranged in column-major order.
pub fn make_mat3<N: Scalar>(ptr: &[N]) -> Mat<N, U3, U3> {
    Matrix3::from_column_slice(ptr)
}

/// Creates a 3x2 matrix from a slice arranged in column-major order.
pub fn make_mat3x2<N: Scalar>(ptr: &[N]) -> Mat<N, U3, U2> {
    Matrix3x2::from_column_slice(ptr)
}

/// Creates a 3x3 matrix from a slice arranged in column-major order.
pub fn make_mat3x3<N: Scalar>(ptr: &[N]) -> Mat<N, U3, U3> {
    Matrix3::from_column_slice(ptr)
}

/// Creates a 3x4 matrix from a slice arranged in column-major order.
pub fn make_mat3x4<N: Scalar>(ptr: &[N]) -> Mat<N, U3, U4> {
    Matrix3x4::from_column_slice(ptr)
}

/// Creates a 4x4 matrix from a slice arranged in column-major order.
pub fn make_mat4<N: Scalar>(ptr: &[N]) -> Mat<N, U4, U4> {
    Matrix4::from_column_slice(ptr)
}

/// Creates a 4x2 matrix from a slice arranged in column-major order.
pub fn make_mat4x2<N: Scalar>(ptr: &[N]) -> Mat<N, U4, U2> {
    Matrix4x2::from_column_slice(ptr)
}

/// Creates a 4x3 matrix from a slice arranged in column-major order.
pub fn make_mat4x3<N: Scalar>(ptr: &[N]) -> Mat<N, U4, U3> {
    Matrix4x3::from_column_slice(ptr)
}

/// Creates a 4x4 matrix from a slice arranged in column-major order.
pub fn make_mat4x4<N: Scalar>(ptr: &[N]) -> Mat<N, U4, U4> {
    Matrix4::from_column_slice(ptr)
}

/// Converts a 2x2 matrix to a 3x3 matrix.
pub fn mat2_to_mat3<N: Number>(m: &Mat<N, U2, U2>) -> Mat<N, U3, U3> {
    let _0 = N::zero();
    let _1 = N::one();

    Matrix3::new(
        m.m11, m.m12, _0,
        m.m21, m.m22, _0,
        _0, _0, _1
    )
}

/// Converts a 3x3 matrix to a 2x2 matrix.
pub fn mat3_to_mat2<N: Scalar>(m: &Mat<N, U3, U3>) -> Mat<N, U2, U2> {
    Matrix2::new(
        m.m11, m.m12,
        m.m21, m.m22
    )
}

/// Converts a 3x3 matrix to a 4x4 matrix.
pub fn mat3_to_mat4<N: Number>(m: &Mat<N, U3, U3>) -> Mat<N, U4, U4> {
    let _0 = N::zero();
    let _1 = N::one();

    Matrix4::new(
        m.m11, m.m12, m.m13, _0,
        m.m21, m.m22, m.m23, _0,
        m.m31, m.m32, m.m33, _0,
        _0, _0, _0, _1,
    )
}

/// Converts a 4x4 matrix to a 3x3 matrix.
pub fn mat4_to_mat3<N: Scalar>(m: &Mat<N, U4, U4>) -> Mat<N, U3, U3> {
    Matrix3::new(
        m.m11, m.m12, m.m13,
        m.m21, m.m22, m.m23,
        m.m31, m.m32, m.m33,
    )
}

/// Converts a 2x2 matrix to a 4x4 matrix.
pub fn mat2_to_mat4<N: Number>(m: &Mat<N, U2, U2>) -> Mat<N, U4, U4> {
    let _0 = N::zero();
    let _1 = N::one();

    Matrix4::new(
        m.m11, m.m12, _0, _0,
        m.m21, m.m22, _0, _0,
        _0, _0, _1, _0,
        _0, _0, _0, _1,
    )
}

/// Converts a 4x4 matrix to a 2x2 matrix.
pub fn mat4_to_mat2<N: Scalar>(m: &Mat<N, U4, U4>) -> Mat<N, U2, U2> {
    Matrix2::new(
        m.m11, m.m12,
        m.m21, m.m22,
    )
}

/// Creates a quaternion from a slice arranged as `[x, y, z, w]`.
pub fn make_quat<N: Real>(ptr: &[N]) -> Qua<N> {
    Quaternion::from_vector(Vector4::from_column_slice(ptr))
}

/// Creates a 1D vector from a slice.
pub fn make_vec1<N: Scalar>(v: &Vec<N, U1>) -> Vec<N, U1> {
    *v
}

/// Creates a 1D vector from another vector.
pub fn vec2_to_vec1<N: Scalar>(v: &Vec<N, U2>) -> Vec<N, U1> {
    Vector1::new(v.x)
}

/// Creates a 1D vector from another vector.
pub fn vec3_to_vec1<N: Scalar>(v: &Vec<N, U3>) -> Vec<N, U1> {
    Vector1::new(v.x)
}

/// Creates a 1D vector from another vector.
pub fn vec4_to_vec1<N: Scalar>(v: &Vec<N, U4>) -> Vec<N, U1> {
    Vector1::new(v.x)
}

/// Creates a 2D vector from another vector.
///
/// Missing components, if any, are set to 0.
pub fn vec1_to_vec2<N: Number>(v: &Vec<N, U1>) -> Vec<N, U2> {
    Vector2::new(v.x, N::zero())
}

/// Creates a 2D vector from another vector.
pub fn vec2_to_vec2<N: Scalar>(v: &Vec<N, U2>) -> Vec<N, U2> {
    *v
}

/// Creates a 2D vector from another vector.
pub fn vec3_to_vec2<N: Scalar>(v: &Vec<N, U3>) -> Vec<N, U2> {
    Vector2::new(v.x, v.y)
}

/// Creates a 2D vector from another vector.
pub fn vec4_to_vec2<N: Scalar>(v: &Vec<N, U4>) -> Vec<N, U2> {
    Vector2::new(v.x, v.y)
}

/// Creates a 2D vector from a slice.
pub fn make_vec2<N: Scalar>(ptr: &[N]) -> Vec<N, U2> {
    Vector2::from_column_slice(ptr)
}

/// Creates a 3D vector from another vector.
///
/// Missing components, if any, are set to 0.
pub fn vec1_to_vec3<N: Number>(v: &Vec<N, U1>) -> Vec<N, U3> {
    Vector3::new(v.x, N::zero(), N::zero())
}

/// Creates a 3D vector from another vector.
///
/// Missing components, if any, are set to 0.
pub fn vec2_to_vec3<N: Number>(v: &Vec<N, U2>) -> Vec<N, U3> {
    Vector3::new(v.x, v.y, N::zero())
}

/// Creates a 3D vector from another vector.
pub fn vec3_to_vec3<N: Scalar>(v: &Vec<N, U3>) -> Vec<N, U3> {
    *v
}

/// Creates a 3D vector from another vector.
pub fn vec4_to_vec3<N: Scalar>(v: &Vec<N, U4>) -> Vec<N, U3> {
    Vector3::new(v.x, v.y, v.z)
}

/// Creates a 3D vector from another vector.
pub fn make_vec3<N: Scalar>(ptr: &[N]) -> Vec<N, U3> {
    Vector3::from_column_slice(ptr)
}

/// Creates a 4D vector from another vector.
///
/// Missing components, if any, are set to 0.
pub fn vec1_to_vec4<N: Number>(v: &Vec<N, U1>) -> Vec<N, U4> {
    Vector4::new(v.x, N::zero(), N::zero(), N::zero())
}

/// Creates a 4D vector from another vector.
///
/// Missing components, if any, are set to 0.
pub fn vec2_to_vec4<N: Number>(v: &Vec<N, U2>) -> Vec<N, U4> {
    Vector4::new(v.x, v.y, N::zero(), N::zero())
}

/// Creates a 4D vector from another vector.
///
/// Missing components, if any, are set to 0.
pub fn vec3_to_vec4<N: Number>(v: &Vec<N, U3>) -> Vec<N, U4> {
    Vector4::new(v.x, v.y, v.z, N::zero())
}

/// Creates a 4D vector from another vector.
pub fn vec4_to_vec4<N: Scalar>(v: &Vec<N, U4>) -> Vec<N, U4> {
    *v
}

/// Creates a 4D vector from another vector.
pub fn make_vec4<N: Scalar>(ptr: &[N]) -> Vec<N, U4> {
    Vector4::from_column_slice(ptr)
}

/// Converts a matrix or vector to a slice arranged in column-major order.
pub fn value_ptr<N: Scalar, R: Dimension, C: Dimension>(x: &Mat<N, R, C>) -> &[N]
    where DefaultAllocator: Alloc<N, R, C> {
    x.as_slice()
}

/// Converts a matrix or vector to a mutable slice arranged in column-major order.
pub fn value_ptr_mut<N: Scalar, R: Dimension, C: Dimension>(x: &mut Mat<N, R, C>) -> &mut [N]
    where DefaultAllocator: Alloc<N, R, C> {
    x.as_mut_slice()
}

