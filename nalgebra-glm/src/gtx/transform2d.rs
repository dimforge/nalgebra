use na::{Real, U2, U3, UnitComplex};

use traits::Number;
use aliases::{Mat, Vec};

/// Builds a 2D rotation matrix from an angle and right-multiply it to `m`.
pub fn rotate2d<N: Real>(m: &Mat<N, U3, U3>, angle: N) -> Mat<N, U3, U3> {
    m * UnitComplex::new(angle).to_homogeneous()
}

/// Builds a 2D scaling matrix and right-multiply it to `m`.
pub fn scale2d<N: Number>(m: &Mat<N, U3, U3>, v: &Vec<N, U2>) -> Mat<N, U3, U3> {
    m.prepend_nonuniform_scaling(v)
}

/// Builds a translation matrix and right-multiply it to `m`.
pub fn translate2d<N: Number>(m: &Mat<N, U3, U3>, v: &Vec<N, U2>) -> Mat<N, U3, U3> {
    m.prepend_translation(v)
}
