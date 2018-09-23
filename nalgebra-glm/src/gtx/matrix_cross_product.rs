use na::Real;

use aliases::{TMat3, TMat4, TVec3};

/// Builds a 3x3 matrix `m` such that for any `v`: `m * v == cross(x, v)`.
pub fn matrix_cross3<N: Real>(x: &TVec3<N>) -> TMat3<N> {
    x.cross_matrix()
}

/// Builds a 4x4 matrix `m` such that for any `v`: `m * v == cross(x, v)`.
pub fn matrix_cross<N: Real>(x: &TVec3<N>) -> TMat4<N> {
    let m = x.cross_matrix();

    // FIXME: use a dedicated constructor from Matrix3 to Matrix4.
    TMat4::new(
        m.m11, m.m12, m.m13, N::zero(),
        m.m21, m.m22, m.m23, N::zero(),
        m.m31, m.m32, m.m33, N::zero(),
        N::zero(), N::zero(), N::zero(), N::one(),
    )
}
