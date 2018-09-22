use na::{U2, U3, U4, Matrix3, Matrix4};

use traits::Number;
use aliases::{Mat, Vec};

/// Build planar projection matrix along normal axis and right-multiply it to `m`.
pub fn proj2d<N: Number>(m: &Mat<N, U3, U3>, normal: &Vec<N, U2>) -> Mat<N, U3, U3> {
    let mut res = Matrix3::identity();

    {
        let mut part = res.fixed_slice_mut::<U2, U2>(0, 0);
        part -= normal * normal.transpose();
    }

    m * res
}

/// Build planar projection matrix along normal axis, and right-multiply it to `m`.
pub fn proj<N: Number>(m: &Mat<N, U4, U4>, normal: &Vec<N, U3>) -> Mat<N, U4, U4> {
    let mut res = Matrix4::identity();

    {
        let mut part = res.fixed_slice_mut::<U3, U3>(0, 0);
        part -= normal * normal.transpose();
    }

    m * res
}

/// Builds a reflection matrix and right-multiply it to `m`.
pub fn reflect2d<N: Number>(m: &Mat<N, U3, U3>, normal: &Vec<N, U2>) -> Mat<N, U3, U3> {
    let mut res = Matrix3::identity();

    {
        let mut part = res.fixed_slice_mut::<U2, U2>(0, 0);
        part -= (normal * N::from_f64(2.0).unwrap()) * normal.transpose();
    }

    m * res
}

/// Builds a reflection matrix, and right-multiply it to `m`.
pub fn reflect<N: Number>(m: &Mat<N, U4, U4>, normal: &Vec<N, U3>) -> Mat<N, U4, U4> {
    let mut res = Matrix4::identity();

    {
        let mut part = res.fixed_slice_mut::<U3, U3>(0, 0);
        part -= (normal * N::from_f64(2.0).unwrap()) * normal.transpose();
    }

    m * res
}

/// Builds a scale-bias matrix.
pub fn scale_bias_matrix<N: Number>(scale: N, bias: N) -> Mat<N, U4, U4> {
    let _0 = N::zero();
    let _1 = N::one();

    Matrix4::new(
        scale, _0, _0, bias,
        _0, scale, _0, bias,
        _0, _0, scale, bias,
        _0, _0, _0, _1,
    )
}

/// Builds a scale-bias matrix, and right-multiply it to `m`.
pub fn scale_bias<N: Number>(m: &Mat<N, U4, U4>, scale: N, bias: N) -> Mat<N, U4, U4> {
    m * scale_bias_matrix(scale, bias)
}

/// Transforms a matrix with a shearing on X axis.
pub fn shear2d_x<N: Number>(m: &Mat<N, U3, U3>, y: N) -> Mat<N, U3, U3> {
    let _0 = N::zero();
    let _1 = N::one();

    let shear = Matrix3::new(
        _1,  y, _0,
        _0, _1, _0,
        _0, _0, _1
    );
    m * shear
}

/// Transforms a matrix with a shearing on Y axis.
pub fn shear_x<N: Number>(m: &Mat<N, U4, U4>, y: N, z: N) -> Mat<N, U4, U4> {
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

/// Transforms a matrix with a shearing on Y axis.
pub fn shear_y_mat3<N: Number>(m: &Mat<N, U3, U3>, x: N) -> Mat<N, U3, U3> {
    let _0 = N::zero();
    let _1 = N::one();

    let shear = Matrix3::new(
        _1, _0, _0,
        x, _1, _0,
        _0, _0, _1
    );
    m * shear
}

/// Transforms a matrix with a shearing on Y axis.
pub fn shear_y<N: Number>(m: &Mat<N, U4, U4>, x: N, z: N) -> Mat<N, U4, U4> {
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

/// Transforms a matrix with a shearing on Z axis.
pub fn shear_z<N: Number>(m: &Mat<N, U4, U4>, x: N, y: N) -> Mat<N, U4, U4> {
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
