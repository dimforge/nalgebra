use na::{Scalar, Real, U1, U2, U3, U4};
use aliases::{Vec, Mat, Qua};


/// Creates a new 1D vector.
pub fn vec1<N: Scalar>(x: N) -> Vec<N, U1> {
    Vec::<N, U1>::new(x)
}

/// Creates a new 2D vector.
pub fn vec2<N: Scalar>(x: N, y: N) -> Vec<N, U2> {
    Vec::<N, U2>::new(x, y)
}

/// Creates a new 3D vector.
pub fn vec3<N: Scalar>(x: N, y: N, z: N) -> Vec<N, U3> {
    Vec::<N, U3>::new(x, y, z)
}

/// Creates a new 4D vector.
pub fn vec4<N: Scalar>(x: N, y: N, z: N, w: N) -> Vec<N, U4> {
    Vec::<N, U4>::new(x, y, z, w)
}


/// Create a new 2x2 matrix.
pub fn mat2<N: Scalar>(m11: N, m12: N, m21: N, m22: N) -> Mat<N, U2, U2> {
    Mat::<N, U2, U2>::new(
        m11, m12,
        m21, m22,
    )
}

/// Create a new 2x2 matrix.
pub fn mat2x2<N: Scalar>(m11: N, m12: N, m21: N, m22: N) -> Mat<N, U2, U2> {
    Mat::<N, U2, U2>::new(
        m11, m12,
        m21, m22,
    )
}

/// Create a new 2x3 matrix.
pub fn mat2x3<N: Scalar>(m11: N, m12: N, m13: N, m21: N, m22: N, m23: N) -> Mat<N, U2, U3> {
    Mat::<N, U2, U3>::new(
        m11, m12, m13,
        m21, m22, m23,
    )
}

/// Create a new 2x4 matrix.
pub fn mat2x4<N: Scalar>(m11: N, m12: N, m13: N, m14: N, m21: N, m22: N, m23: N, m24: N) -> Mat<N, U2, U4> {
    Mat::<N, U2, U4>::new(
        m11, m12, m13, m14,
        m21, m22, m23, m24,
    )
}

/// Create a new 3x3 matrix.
pub fn mat3<N: Scalar>(m11: N, m12: N, m13: N, m21: N, m22: N, m23: N, m31: N, m32: N, m33: N) -> Mat<N, U3, U3> {
    Mat::<N, U3, U3>::new(
        m11, m12, m13,
        m21, m22, m23,
        m31, m32, m33,
    )
}

/// Create a new 3x2 matrix.
pub fn mat3x2<N: Scalar>(m11: N, m12: N, m21: N, m22: N, m31: N, m32: N) -> Mat<N, U3, U2> {
    Mat::<N, U3, U2>::new(
        m11, m12,
        m21, m22,
        m31, m32,
    )
}

/// Create a new 3x3 matrix.
pub fn mat3x3<N: Scalar>(m11: N, m12: N, m13: N, m21: N, m22: N, m23: N, m31: N, m32: N, m33: N) -> Mat<N, U3, U3> {
    Mat::<N, U3, U3>::new(
        m11, m12, m13,
        m31, m32, m33,
        m21, m22, m23,
    )
}

/// Create a new 3x4 matrix.
pub fn mat3x4<N: Scalar>(m11: N, m12: N, m13: N, m14: N, m21: N, m22: N, m23: N, m24: N, m31: N, m32: N, m33: N, m34: N) -> Mat<N, U3, U4> {
    Mat::<N, U3, U4>::new(
        m11, m12, m13, m14,
        m21, m22, m23, m24,
        m31, m32, m33, m34,
    )
}

/// Create a new 4x2 matrix.
pub fn mat4x2<N: Scalar>(m11: N, m12: N, m21: N, m22: N, m31: N, m32: N, m41: N, m42: N) -> Mat<N, U4, U2> {
    Mat::<N, U4, U2>::new(
        m11, m12,
        m21, m22,
        m31, m32,
        m41, m42,
    )
}

/// Create a new 4x3 matrix.
pub fn mat4x3<N: Scalar>(m11: N, m12: N, m13: N, m21: N, m22: N, m23: N, m31: N, m32: N, m33: N, m41: N, m42: N, m43: N) -> Mat<N, U4, U3> {
    Mat::<N, U4, U3>::new(
        m11, m12, m13,
        m21, m22, m23,
        m31, m32, m33,
        m41, m42, m43,
    )
}

/// Create a new 4x4 matrix.
pub fn mat4x4<N: Scalar>(m11: N, m12: N, m13: N, m14: N, m21: N, m22: N, m23: N, m24: N, m31: N, m32: N, m33: N, m34: N, m41: N, m42: N, m43: N, m44: N) -> Mat<N, U4, U4> {
    Mat::<N, U4, U4>::new(
        m11, m12, m13, m14,
        m21, m22, m23, m24,
        m31, m32, m33, m34,
        m41, m42, m43, m44,
    )
}

/// Create a new 4x4 matrix.
pub fn mat4<N: Scalar>(m11: N, m12: N, m13: N, m14: N, m21: N, m22: N, m23: N, m24: N, m31: N, m32: N, m33: N, m34: N, m41: N, m42: N, m43: N, m44: N) -> Mat<N, U4, U4> {
    Mat::<N, U4, U4>::new(
        m11, m12, m13, m14,
        m21, m22, m23, m24,
        m31, m32, m33, m34,
        m41, m42, m43, m44,
    )
}

/// Creates a new quaternion.
pub fn quat<N: Real>(x: N, y: N, z: N, w: N) -> Qua<N> {
    Qua::new(x, y, z, w)
}
