use na::{DefaultAllocator, Real};

use aliases::TMat;
use traits::{Alloc, Dimension};

/// Fast matrix inverse for affine matrix.
pub fn affine_inverse<N: Real, D: Dimension>(m: TMat<N, D, D>) -> TMat<N, D, D>
where DefaultAllocator: Alloc<N, D, D> {
    // FIXME: this should be optimized.
    m.try_inverse().unwrap_or_else(TMat::<_, D, D>::zeros)
}

/// Compute the transpose of the inverse of a matrix.
pub fn inverse_transpose<N: Real, D: Dimension>(m: TMat<N, D, D>) -> TMat<N, D, D>
where DefaultAllocator: Alloc<N, D, D> {
    m.try_inverse()
        .unwrap_or_else(TMat::<_, D, D>::zeros)
        .transpose()
}
