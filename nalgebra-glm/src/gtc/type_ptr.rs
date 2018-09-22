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

/// Creates a quaternion from a slice arranged as `[x, y, z, w]`.
pub fn make_quat<N: Real>(ptr: &[N]) -> Qua<N> {
    Quaternion::from_vector(Vector4::from_column_slice(ptr))
}

/// Creates a 1D vector from another vector.
///
/// Missing components, if any, are set to 0.
pub fn make_vec1<N: Scalar>(v: &Vec<N, U1>) -> Vec<N, U1> {
    *v
}

/// Creates a 1D vector from another vector.
///
/// Missing components, if any, are set to 0.
pub fn make_vec1_2<N: Scalar>(v: &Vec<N, U2>) -> Vec<N, U1> {
    Vector1::new(v.x)
}

/// Creates a 1D vector from another vector.
///
/// Missing components, if any, are set to 0.
pub fn make_vec1_3<N: Scalar>(v: &Vec<N, U3>) -> Vec<N, U1> {
    Vector1::new(v.x)
}

/// Creates a 1D vector from another vector.
///
/// Missing components, if any, are set to 0.
pub fn make_vec1_4<N: Scalar>(v: &Vec<N, U4>) -> Vec<N, U1> {
    Vector1::new(v.x)
}

/// Creates a 2D vector from another vector.
///
/// Missing components, if any, are set to 0.
pub fn make_vec2_1<N: Number>(v: &Vec<N, U1>) -> Vec<N, U2> {
    Vector2::new(v.x, N::zero())
}

/// Creates a 2D vector from another vector.
///
/// Missing components, if any, are set to 0.
pub fn make_vec2_2<N: Scalar>(v: &Vec<N, U2>) -> Vec<N, U2> {
    *v
}

/// Creates a 2D vector from another vector.
///
/// Missing components, if any, are set to 0.
pub fn make_vec2_3<N: Scalar>(v: &Vec<N, U3>) -> Vec<N, U2> {
    Vector2::new(v.x, v.y)
}

/// Creates a 2D vector from another vector.
///
/// Missing components, if any, are set to 0.
pub fn make_vec2_4<N: Scalar>(v: &Vec<N, U4>) -> Vec<N, U2> {
    Vector2::new(v.x, v.y)
}

/// Creates a 2D vector from another vector.
///
/// Missing components, if any, are set to 0.
pub fn make_vec2<N: Scalar>(ptr: &[N]) -> Vec<N, U2> {
    Vector2::from_column_slice(ptr)
}

/// Creates a 3D vector from another vector.
///
/// Missing components, if any, are set to 0.
pub fn make_vec3_1<N: Number>(v: &Vec<N, U1>) -> Vec<N, U3> {
    Vector3::new(v.x, N::zero(), N::zero())
}

/// Creates a 3D vector from another vector.
///
/// Missing components, if any, are set to 0.
pub fn make_vec3_2<N: Number>(v: &Vec<N, U2>) -> Vec<N, U3> {
    Vector3::new(v.x, v.y, N::zero())
}

/// Creates a 3D vector from another vector.
///
/// Missing components, if any, are set to 0.
pub fn make_vec3_3<N: Scalar>(v: &Vec<N, U3>) -> Vec<N, U3> {
    *v
}

/// Creates a 3D vector from another vector.
///
/// Missing components, if any, are set to 0.
pub fn make_vec3_4<N: Scalar>(v: &Vec<N, U4>) -> Vec<N, U3> {
    Vector3::new(v.x, v.y, v.z)
}

/// Creates a 3D vector from another vector.
///
/// Missing components, if any, are set to 0.
pub fn make_vec3<N: Scalar>(ptr: &[N]) -> Vec<N, U3> {
    Vector3::from_column_slice(ptr)
}

/// Creates a 4D vector from another vector.
///
/// Missing components, if any, are set to 0.
pub fn make_vec4_1<N: Number>(v: &Vec<N, U1>) -> Vec<N, U4> {
    Vector4::new(v.x, N::zero(), N::zero(), N::zero())
}

/// Creates a 4D vector from another vector.
///
/// Missing components, if any, are set to 0.
pub fn make_vec4_2<N: Number>(v: &Vec<N, U2>) -> Vec<N, U4> {
    Vector4::new(v.x, v.y, N::zero(), N::zero())
}

/// Creates a 4D vector from another vector.
///
/// Missing components, if any, are set to 0.
pub fn make_vec4_3<N: Number>(v: &Vec<N, U3>) -> Vec<N, U4> {
    Vector4::new(v.x, v.y, v.z, N::zero())
}

/// Creates a 4D vector from another vector.
///
/// Missing components, if any, are set to 0.
pub fn make_vec4_4<N: Scalar>(v: &Vec<N, U4>) -> Vec<N, U4> {
    *v
}

/// Creates a 4D vector from another vector.
///
/// Missing components, if any, are set to 0.
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

