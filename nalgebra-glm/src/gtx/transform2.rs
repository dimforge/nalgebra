use na::{U2, U3};

use aliases::{TMat3, TMat4, TVec2, TVec3};
use traits::Number;

/// Build planar projection matrix along normal axis and right-multiply it to `m`.
pub fn proj2d<N: Number>(m: &TMat3<N>, normal: &TVec2<N>) -> TMat3<N> {
    let mut res = TMat3::identity();

    {
        let mut part = res.fixed_slice_mut::<U2, U2>(0, 0);
        part -= normal * normal.transpose();
    }

    m * res
}

/// Build planar projection matrix along normal axis, and right-multiply it to `m`.
pub fn proj<N: Number>(m: &TMat4<N>, normal: &TVec3<N>) -> TMat4<N> {
    let mut res = TMat4::identity();

    {
        let mut part = res.fixed_slice_mut::<U3, U3>(0, 0);
        part -= normal * normal.transpose();
    }

    m * res
}

/// Builds a reflection matrix and right-multiply it to `m`.
pub fn reflect2d<N: Number>(m: &TMat3<N>, normal: &TVec2<N>) -> TMat3<N> {
    let mut res = TMat3::identity();

    {
        let mut part = res.fixed_slice_mut::<U2, U2>(0, 0);
        part -= (normal * N::from_f64(2.0).unwrap()) * normal.transpose();
    }

    m * res
}

/// Builds a reflection matrix, and right-multiply it to `m`.
pub fn reflect<N: Number>(m: &TMat4<N>, normal: &TVec3<N>) -> TMat4<N> {
    let mut res = TMat4::identity();

    {
        let mut part = res.fixed_slice_mut::<U3, U3>(0, 0);
        part -= (normal * N::from_f64(2.0).unwrap()) * normal.transpose();
    }

    m * res
}

/// Builds a scale-bias matrix.
pub fn scale_bias_matrix<N: Number>(scale: N, bias: N) -> TMat4<N> {
    let _0 = N::zero();
    let _1 = N::one();

    TMat4::new(
        scale, _0, _0, bias, _0, scale, _0, bias, _0, _0, scale, bias, _0, _0, _0, _1,
    )
}

/// Builds a scale-bias matrix, and right-multiply it to `m`.
pub fn scale_bias<N: Number>(m: &TMat4<N>, scale: N, bias: N) -> TMat4<N> {
    m * scale_bias_matrix(scale, bias)
}

/// Transforms a matrix with a shearing on X axis.
pub fn shear2d_x<N: Number>(m: &TMat3<N>, y: N) -> TMat3<N> {
    let _0 = N::zero();
    let _1 = N::one();

    let shear = TMat3::new(_1, y, _0, _0, _1, _0, _0, _0, _1);
    m * shear
}

/// Transforms a matrix with a shearing on Y axis.
pub fn shear_x<N: Number>(m: &TMat4<N>, y: N, z: N) -> TMat4<N> {
    let _0 = N::zero();
    let _1 = N::one();
    let shear = TMat4::new(_1, _0, _0, _0, y, _1, _0, _0, z, _0, _1, _0, _0, _0, _0, _1);

    m * shear
}

/// Transforms a matrix with a shearing on Y axis.
pub fn shear2d_y<N: Number>(m: &TMat3<N>, x: N) -> TMat3<N> {
    let _0 = N::zero();
    let _1 = N::one();

    let shear = TMat3::new(_1, _0, _0, x, _1, _0, _0, _0, _1);
    m * shear
}

/// Transforms a matrix with a shearing on Y axis.
pub fn shear_y<N: Number>(m: &TMat4<N>, x: N, z: N) -> TMat4<N> {
    let _0 = N::zero();
    let _1 = N::one();
    let shear = TMat4::new(_1, x, _0, _0, _0, _1, _0, _0, _0, z, _1, _0, _0, _0, _0, _1);

    m * shear
}

/// Transforms a matrix with a shearing on Z axis.
pub fn shear_z<N: Number>(m: &TMat4<N>, x: N, y: N) -> TMat4<N> {
    let _0 = N::zero();
    let _1 = N::one();
    let shear = TMat4::new(_1, _0, x, _0, _0, _1, y, _0, _0, _0, _1, _0, _0, _0, _0, _1);

    m * shear
}
