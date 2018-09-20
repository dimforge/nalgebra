use na::{Scalar, Real, U1, U2, U3, U4, DefaultAllocator,
         Quaternion, Matrix2, Matrix3, Matrix4, Vector1, Vector2, Vector3, Vector4,
         Matrix2x3, Matrix2x4, Matrix3x2, Matrix3x4, Matrix4x2, Matrix4x3};

use traits::{Number, Alloc, Dimension};
use aliases::{Qua, Vec, Mat};


pub fn make_mat2<N: Scalar>(ptr: &[N]) -> Mat<N, U2, U2> {
    Matrix2::from_column_slice(ptr)
}

pub fn make_mat2x2<N: Scalar>(ptr: &[N]) -> Mat<N, U2, U2> {
    Matrix2::from_column_slice(ptr)
}

pub fn make_mat2x3<N: Scalar>(ptr: &[N]) -> Mat<N, U2, U3> {
    Matrix2x3::from_column_slice(ptr)
}

pub fn make_mat2x4<N: Scalar>(ptr: &[N]) -> Mat<N, U2, U4> {
    Matrix2x4::from_column_slice(ptr)
}

pub fn make_mat3<N: Scalar>(ptr: &[N]) -> Mat<N, U3, U3> {
    Matrix3::from_column_slice(ptr)
}

pub fn make_mat3x2<N: Scalar>(ptr: &[N]) -> Mat<N, U3, U2> {
    Matrix3x2::from_column_slice(ptr)
}

pub fn make_mat3x3<N: Scalar>(ptr: &[N]) -> Mat<N, U3, U3> {
    Matrix3::from_column_slice(ptr)
}

pub fn make_mat3x4<N: Scalar>(ptr: &[N]) -> Mat<N, U3, U4> {
    Matrix3x4::from_column_slice(ptr)
}

pub fn make_mat4<N: Scalar>(ptr: &[N]) -> Mat<N, U4, U4> {
    Matrix4::from_column_slice(ptr)
}

pub fn make_mat4x2<N: Scalar>(ptr: &[N]) -> Mat<N, U4, U2> {
    Matrix4x2::from_column_slice(ptr)
}

pub fn make_mat4x3<N: Scalar>(ptr: &[N]) -> Mat<N, U4, U3> {
    Matrix4x3::from_column_slice(ptr)
}

pub fn make_mat4x4<N: Scalar>(ptr: &[N]) -> Mat<N, U4, U4> {
    Matrix4::from_column_slice(ptr)
}

pub fn make_quat<N: Real>(ptr: &[N]) -> Qua<N> {
    Quaternion::from_vector(Vector4::from_column_slice(ptr))
}

pub fn make_vec1<N: Scalar>(v: &Vec<N, U1>) -> Vec<N, U1> {
    *v
}

pub fn make_vec1_2<N: Scalar>(v: &Vec<N, U2>) -> Vec<N, U1> {
    Vector1::new(v.x)
}

pub fn make_vec1_3<N: Scalar>(v: &Vec<N, U3>) -> Vec<N, U1> {
    Vector1::new(v.x)
}

pub fn make_vec1_4<N: Scalar>(v: &Vec<N, U4>) -> Vec<N, U1> {
    Vector1::new(v.x)
}

pub fn make_vec2_1<N: Number>(v: &Vec<N, U1>) -> Vec<N, U2> {
    Vector2::new(v.x, N::zero())
}

pub fn make_vec2_2<N: Scalar>(v: &Vec<N, U2>) -> Vec<N, U2> {
    *v
}

pub fn make_vec2_3<N: Scalar>(v: &Vec<N, U3>) -> Vec<N, U2> {
    Vector2::new(v.x, v.y)
}

pub fn make_vec2_4<N: Scalar>(v: &Vec<N, U4>) -> Vec<N, U2> {
    Vector2::new(v.x, v.y)
}

pub fn make_vec2<N: Scalar>(ptr: &[N]) -> Vec<N, U2> {
    Vector2::from_column_slice(ptr)
}

pub fn make_vec3_1<N: Number>(v: &Vec<N, U1>) -> Vec<N, U3> {
    Vector3::new(v.x, N::zero(), N::zero())
}

pub fn make_vec3_2<N: Number>(v: &Vec<N, U2>) -> Vec<N, U3> {
    Vector3::new(v.x, v.y, N::zero())
}

pub fn make_vec3_3<N: Scalar>(v: &Vec<N, U3>) -> Vec<N, U3> {
    *v
}

pub fn make_vec3_4<N: Scalar>(v: &Vec<N, U4>) -> Vec<N, U3> {
    Vector3::new(v.x, v.y, v.z)
}

pub fn make_vec3<N: Scalar>(ptr: &[N]) -> Vec<N, U3> {
    Vector3::from_column_slice(ptr)
}

pub fn make_vec4_1<N: Number>(v: &Vec<N, U1>) -> Vec<N, U4> {
    Vector4::new(v.x, N::zero(), N::zero(), N::zero())
}

pub fn make_vec4_2<N: Number>(v: &Vec<N, U2>) -> Vec<N, U4> {
    Vector4::new(v.x, v.y, N::zero(), N::zero())
}

pub fn make_vec4_3<N: Number>(v: &Vec<N, U3>) -> Vec<N, U4> {
    Vector4::new(v.x, v.y, v.z, N::zero())
}

pub fn make_vec4_4<N: Scalar>(v: &Vec<N, U4>) -> Vec<N, U4> {
    *v
}

pub fn make_vec4<N: Scalar>(ptr: &[N]) -> Vec<N, U4> {
    Vector4::from_column_slice(ptr)
}

pub fn value_ptr<N: Scalar, R: Dimension, C: Dimension>(x: &Mat<N, R, C>) -> &[N]
    where DefaultAllocator: Alloc<N, R, C> {
    x.as_slice()
}

pub fn value_ptr_mut<N: Scalar, R: Dimension, C: Dimension>(x: &mut Mat<N, R, C>) -> &mut [N]
    where DefaultAllocator: Alloc<N, R, C> {
    x.as_mut_slice()
}

