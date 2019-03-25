use na::RealField;

use crate::aliases::{TMat3, TMat4, TVec3};

/// Builds a 3x3 matrix `m` such that for any `v`: `m * v == cross(x, v)`.
///
/// # See also:
///
/// * [`matrix_cross`](fn.matrix_cross.html)
pub fn matrix_cross3<N: RealField>(x: &TVec3<N>) -> TMat3<N> {
    x.cross_matrix()
}

/// Builds a 4x4 matrix `m` such that for any `v`: `m * v == cross(x, v)`.
///
/// # See also:
///
/// * [`matrix_cross3`](fn.matrix_cross3.html)
pub fn matrix_cross<N: RealField>(x: &TVec3<N>) -> TMat4<N> {
    crate::mat3_to_mat4(&x.cross_matrix())
}
