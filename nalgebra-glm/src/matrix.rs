use na::{Const, DimMin, RealField, Scalar};

use crate::aliases::{TMat, TVec};
use crate::traits::Number;

/// The determinant of the matrix `m`.
pub fn determinant<T: RealField, const D: usize>(m: &TMat<T, D, D>) -> T
where
    Const<D>: DimMin<Const<D>, Output = Const<D>>,
{
    m.determinant()
}

/// The inverse of the matrix `m`.
pub fn inverse<T: RealField, const D: usize>(m: &TMat<T, D, D>) -> TMat<T, D, D> {
    m.clone()
        .try_inverse()
        .unwrap_or_else(TMat::<T, D, D>::zeros)
}

/// Component-wise multiplication of two matrices.
pub fn matrix_comp_mult<T: Number, const R: usize, const C: usize>(
    x: &TMat<T, R, C>,
    y: &TMat<T, R, C>,
) -> TMat<T, R, C> {
    x.component_mul(y)
}

/// Treats the first parameter `c` as a column vector and the second parameter `r` as a row vector and does a linear algebraic matrix multiply `c * r`.
pub fn outer_product<T: Number, const R: usize, const C: usize>(
    c: &TVec<T, R>,
    r: &TVec<T, C>,
) -> TMat<T, R, C> {
    c * r.transpose()
}

/// The transpose of the matrix `m`.
pub fn transpose<T: Scalar, const R: usize, const C: usize>(x: &TMat<T, R, C>) -> TMat<T, C, R> {
    x.transpose()
}
