use crate::aliases::{TMat3, TMat4, TVec2, TVec3};
use crate::traits::Number;

/// Build planar projection matrix along normal axis and right-multiply it to `m`.
pub fn proj2d<T: Number>(m: &TMat3<T>, normal: &TVec2<T>) -> TMat3<T> {
    let mut res = TMat3::identity();

    {
        let mut part = res.fixed_slice_mut::<2, 2>(0, 0);
        part -= normal * normal.transpose();
    }

    m * res
}

/// Build planar projection matrix along normal axis, and right-multiply it to `m`.
pub fn proj<T: Number>(m: &TMat4<T>, normal: &TVec3<T>) -> TMat4<T> {
    let mut res = TMat4::identity();

    {
        let mut part = res.fixed_slice_mut::<3, 3>(0, 0);
        part -= normal * normal.transpose();
    }

    m * res
}

/// Builds a reflection matrix and right-multiply it to `m`.
pub fn reflect2d<T: Number>(m: &TMat3<T>, normal: &TVec2<T>) -> TMat3<T> {
    let mut res = TMat3::identity();

    {
        let mut part = res.fixed_slice_mut::<2, 2>(0, 0);
        part -= (normal * T::from_subset(&2.0)) * normal.transpose();
    }

    m * res
}

/// Builds a reflection matrix, and right-multiply it to `m`.
pub fn reflect<T: Number>(m: &TMat4<T>, normal: &TVec3<T>) -> TMat4<T> {
    let mut res = TMat4::identity();

    {
        let mut part = res.fixed_slice_mut::<3, 3>(0, 0);
        part -= (normal * T::from_subset(&2.0)) * normal.transpose();
    }

    m * res
}

/// Builds a scale-bias matrix.
pub fn scale_bias_matrix<T: Number>(scale: T, bias: T) -> TMat4<T> {
    let _0 = T::zero();
    let _1 = T::one();

    TMat4::new(
        scale, _0, _0, bias, _0, scale, _0, bias, _0, _0, scale, bias, _0, _0, _0, _1,
    )
}

/// Builds a scale-bias matrix, and right-multiply it to `m`.
pub fn scale_bias<T: Number>(m: &TMat4<T>, scale: T, bias: T) -> TMat4<T> {
    m * scale_bias_matrix(scale, bias)
}

/// Transforms a matrix with a shearing on X axis.
pub fn shear2d_x<T: Number>(m: &TMat3<T>, y: T) -> TMat3<T> {
    let _0 = T::zero();
    let _1 = T::one();

    let shear = TMat3::new(_1, y, _0, _0, _1, _0, _0, _0, _1);
    m * shear
}

/// Transforms a matrix with a shearing on Y axis.
pub fn shear_x<T: Number>(m: &TMat4<T>, y: T, z: T) -> TMat4<T> {
    let _0 = T::zero();
    let _1 = T::one();
    let shear = TMat4::new(_1, _0, _0, _0, y, _1, _0, _0, z, _0, _1, _0, _0, _0, _0, _1);

    m * shear
}

/// Transforms a matrix with a shearing on Y axis.
pub fn shear2d_y<T: Number>(m: &TMat3<T>, x: T) -> TMat3<T> {
    let _0 = T::zero();
    let _1 = T::one();

    let shear = TMat3::new(_1, _0, _0, x, _1, _0, _0, _0, _1);
    m * shear
}

/// Transforms a matrix with a shearing on Y axis.
pub fn shear_y<T: Number>(m: &TMat4<T>, x: T, z: T) -> TMat4<T> {
    let _0 = T::zero();
    let _1 = T::one();
    let shear = TMat4::new(_1, x, _0, _0, _0, _1, _0, _0, _0, z, _1, _0, _0, _0, _0, _1);

    m * shear
}

/// Transforms a matrix with a shearing on Z axis.
pub fn shear_z<T: Number>(m: &TMat4<T>, x: T, y: T) -> TMat4<T> {
    let _0 = T::zero();
    let _1 = T::one();
    let shear = TMat4::new(_1, _0, x, _0, _0, _1, y, _0, _0, _0, _1, _0, _0, _0, _0, _1);

    m * shear
}
