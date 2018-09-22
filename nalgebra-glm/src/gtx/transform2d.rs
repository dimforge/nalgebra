use na::{Real, U2, U3, UnitComplex, Matrix3};

use traits::Number;
use aliases::{Mat, Vec};

/// Builds a 2D rotation matrix from an angle and right-multiply it to `m`.
pub fn rotate<N: Real>(m: &Mat<N, U3, U3>, angle: N) -> Mat<N, U3, U3> {
    m * UnitComplex::new(angle).to_homogeneous()
}

/// Builds a 2D scaling matrix and right-multiply it to `m`.
pub fn scale<N: Number>(m: &Mat<N, U3, U3>, v: &Vec<N, U2>) -> Mat<N, U3, U3> {
    m.prepend_nonuniform_scaling(v)
}

/// Builds a 2D shearing matrix on the `X` axis and right-multiply it to `m`.
pub fn shear_x<N: Number>(m: &Mat<N, U3, U3>, y: N) -> Mat<N, U3, U3> {
    let _0 = N::zero();
    let _1 = N::one();

    let shear = Matrix3::new(
        _1,  y, _0,
        _0, _1, _0,
        _0, _0, _1
    );
    m * shear
}

/// Builds a 2D shearing matrix on the `Y` axis and right-multiply it to `m`.
pub fn shear_y<N: Number>(m: &Mat<N, U3, U3>, x: N) -> Mat<N, U3, U3> {
    let _0 = N::zero();
    let _1 = N::one();

    let shear = Matrix3::new(
        _1, _0, _0,
         x, _1, _0,
        _0, _0, _1
    );
    m * shear
}

/// Builds a translation matrix and right-multiply it to `m`.
pub fn translate<N: Number>(m: &Mat<N, U3, U3>, v: &Vec<N, U2>) -> Mat<N, U3, U3> {
    m.prepend_translation(v)
}
