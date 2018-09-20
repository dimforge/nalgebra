use na::{self, U2, U3, U4, Matrix3, Matrix4};

use traits::Number;
use aliases::{Mat, Vec};

pub fn proj2D<N: Number>(m: &Mat<N, U3, U3>, normal: &Vec<N, U2>) -> Mat<N, U3, U3> {
    let mut res = Matrix3::identity();

    {
        let mut part = res.fixed_slice_mut::<U2, U2>(0, 0);
        part -= normal * normal.transpose();
    }

    res
}

pub fn proj3D<N: Number>(m: &Mat<N, U4, U4>, normal: &Vec<N, U3>) -> Mat<N, U4, U4> {
    let mut res = Matrix4::identity();

    {
        let mut part = res.fixed_slice_mut::<U3, U3>(0, 0);
        part -= normal * normal.transpose();
    }

    res
}

pub fn reflect2D<N: Number>(m: &Mat<N, U3, U3>, normal: &Vec<N, U2>) -> Mat<N, U3, U3> {
    let mut res = Matrix3::identity();

    {
        let mut part = res.fixed_slice_mut::<U2, U2>(0, 0);
        part -= (normal * N::from_f64(2.0).unwrap()) * normal.transpose();
    }

    res
}

pub fn reflect3D<N: Number>(m: &Mat<N, U4, U4>, normal: &Vec<N, U3>) -> Mat<N, U4, U4> {
    let mut res = Matrix4::identity();

    {
        let mut part = res.fixed_slice_mut::<U3, U3>(0, 0);
        part -= (normal * N::from_f64(2.0).unwrap()) * normal.transpose();
    }

    res
}

pub fn scaleBias<N: Number>(scale: N, bias: N) -> Mat<N, U4, U4> {
    let _0 = N::zero();
    let _1 = N::one();

    Matrix4::new(
        scale, _0, _0, bias,
        _0, scale, _0, bias,
        _0, _0, scale, bias,
        _0, _0, _0, _1,
    )
}

pub fn scaleBias2<N: Number>(m: &Mat<N, U4, U4>, scale: N, bias: N) -> Mat<N, U4, U4> {
    m * scaleBias(scale, bias)
}

pub fn shearX2D<N: Number>(m: &Mat<N, U3, U3>, y: N) -> Mat<N, U3, U3> {
    let _0 = N::zero();
    let _1 = N::one();

    let shear = Matrix3::new(
        _1,  y, _0,
        _0, _1, _0,
        _0, _0, _1
    );
    m * shear
}

pub fn shearX3D<N: Number>(m: &Mat<N, U4, U4>, y: N, z: N) -> Mat<N, U4, U4> {
    let _0 = N::zero();
    let _1 = N::one();
    let shear = Matrix4::new(
        _1, _0, _0, _0,
         y, _1, _0, _0,
         z, _0, _1, _0,
        _0, _0, _0, _1,
    );

    m * shear
}

pub fn shearY2D<N: Number>(m: &Mat<N, U3, U3>, x: N) -> Mat<N, U3, U3> {
    let _0 = N::zero();
    let _1 = N::one();

    let shear = Matrix3::new(
        _1, _0, _0,
        x, _1, _0,
        _0, _0, _1
    );
    m * shear
}

pub fn shearY3D<N: Number>(m: &Mat<N, U4, U4>, x: N, z: N) -> Mat<N, U4, U4> {
    let _0 = N::zero();
    let _1 = N::one();
    let shear = Matrix4::new(
        _1,  x, _0, _0,
        _0, _1, _0, _0,
        _0,  z, _1, _0,
        _0, _0, _0, _1,
    );

    m * shear
}

pub fn shearZ3D<N: Number>(m: &Mat<N, U4, U4>, x: N, y: N) -> Mat<N, U4, U4> {
    let _0 = N::zero();
    let _1 = N::one();
    let shear = Matrix4::new(
        _1, _0,  x, _0,
        _0, _1,  y, _0,
        _0, _0, _1, _0,
        _0, _0, _0, _1,
    );

    m * shear
}
