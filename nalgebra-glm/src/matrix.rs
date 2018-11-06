use na::{DefaultAllocator, Real, Scalar};

use aliases::{TMat, TVec};
use traits::{Alloc, Dimension, Number};

/// The determinant of the matrix `m`.
pub fn determinant<N: Real, D: Dimension>(m: &TMat<N, D, D>) -> N
where DefaultAllocator: Alloc<N, D, D> {
    m.determinant()
}

/// The inverse of the matrix `m`.
pub fn inverse<N: Real, D: Dimension>(m: &TMat<N, D, D>) -> TMat<N, D, D>
where DefaultAllocator: Alloc<N, D, D> {
    m.clone()
        .try_inverse()
        .unwrap_or_else(TMat::<N, D, D>::zeros)
}

/// Component-wise multiplication of two matrices.
pub fn matrix_comp_mult<N: Number, R: Dimension, C: Dimension>(
    x: &TMat<N, R, C>,
    y: &TMat<N, R, C>,
) -> TMat<N, R, C>
where
    DefaultAllocator: Alloc<N, R, C>,
{
    x.component_mul(y)
}

/// Treats the first parameter `c` as a column vector and the second parameter `r` as a row vector and does a linear algebraic matrix multiply `c * r`.
pub fn outer_product<N: Number, R: Dimension, C: Dimension>(
    c: &TVec<N, R>,
    r: &TVec<N, C>,
) -> TMat<N, R, C>
where
    DefaultAllocator: Alloc<N, R, C>,
{
    c * r.transpose()
}

/// The transpose of the matrix `m`.
pub fn transpose<N: Scalar, R: Dimension, C: Dimension>(x: &TMat<N, R, C>) -> TMat<N, C, R>
where DefaultAllocator: Alloc<N, R, C> {
    x.transpose()
}
