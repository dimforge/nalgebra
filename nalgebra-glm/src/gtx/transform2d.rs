use na::{Real, UnitComplex};

use aliases::{TMat3, TVec2};
use traits::Number;

/// Builds a 2D rotation matrix from an angle and right-multiply it to `m`.
///
/// # See also:
///
/// * [`rotation2d`](fn.rotation2d.html)
/// * [`scale2d`](fn.scale2d.html)
/// * [`scaling2d`](fn.scaling2d.html)
/// * [`translate2d`](fn.translate2d.html)
/// * [`translation2d`](fn.translation2d.html)
pub fn rotate2d<N: Real>(m: &TMat3<N>, angle: N) -> TMat3<N> {
    m * UnitComplex::new(angle).to_homogeneous()
}

/// Builds a 2D scaling matrix and right-multiply it to `m`.
///
/// # See also:
///
/// * [`rotate2d`](fn.rotate2d.html)
/// * [`rotation2d`](fn.rotation2d.html)
/// * [`scaling2d`](fn.scaling2d.html)
/// * [`translate2d`](fn.translate2d.html)
/// * [`translation2d`](fn.translation2d.html)
pub fn scale2d<N: Number>(m: &TMat3<N>, v: &TVec2<N>) -> TMat3<N> {
    m.prepend_nonuniform_scaling(v)
}

/// Builds a translation matrix and right-multiply it to `m`.
///
/// # See also:
///
/// * [`rotate2d`](fn.rotate2d.html)
/// * [`rotation2d`](fn.rotation2d.html)
/// * [`scale2d`](fn.scale2d.html)
/// * [`scaling2d`](fn.scaling2d.html)
/// * [`translation2d`](fn.translation2d.html)
pub fn translate2d<N: Number>(m: &TMat3<N>, v: &TVec2<N>) -> TMat3<N> {
    m.prepend_translation(v)
}
