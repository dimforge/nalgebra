use na::{Real, U2, U3, UnitComplex, Matrix3};

use traits::Number;
use aliases::{Mat, Vec};

pub fn rotate<N: Real>(m: &Mat<N, U3, U3>, angle: N) -> Mat<N, U3, U3> {
    m * UnitComplex::new(angle).to_homogeneous()
}

pub fn scale<N: Number>(m: &Mat<N, U3, U3>, v: &Vec<N, U2>) -> Mat<N, U3, U3> {
    m.prepend_nonuniform_scaling(v)
}

pub fn shearX<N: Number>(m: &Mat<N, U3, U3>, y: N) -> Mat<N, U3, U3> {
    let _0 = N::zero();
    let _1 = N::one();

    let shear = Matrix3::new(
        _1,  y, _0,
        _0, _1, _0,
        _0, _0, _1
    );
    m * shear
}

pub fn shearY<N: Number>(m: &Mat<N, U3, U3>, x: N) -> Mat<N, U3, U3> {
    let _0 = N::zero();
    let _1 = N::one();

    let shear = Matrix3::new(
        _1, _0, _0,
         x, _1, _0,
        _0, _0, _1
    );
    m * shear
}

pub fn translate<N: Number>(m: &Mat<N, U3, U3>, v: &Vec<N, U2>) -> Mat<N, U3, U3> {
    m.prepend_translation(v)
}
