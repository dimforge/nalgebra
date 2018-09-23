use na::{Real, UnitComplex};

use traits::Number;
use aliases::{TMat3, TVec2};

/// Builds a 2D rotation matrix from an angle and right-multiply it to `m`.
pub fn rotate2d<N: Real>(m: &TMat3<N>, angle: N) -> TMat3<N> {
    m * UnitComplex::new(angle).to_homogeneous()
}

/// Builds a 2D scaling matrix and right-multiply it to `m`.
pub fn scale2d<N: Number>(m: &TMat3<N>, v: &TVec2<N>) -> TMat3<N> {
    m.prepend_nonuniform_scaling(v)
}

/// Builds a translation matrix and right-multiply it to `m`.
pub fn translate2d<N: Number>(m: &TMat3<N>, v: &TVec2<N>) -> TMat3<N> {
    m.prepend_translation(v)
}