use na::{Real, DefaultAllocator, Transform, TAffine};

use traits::{Alloc, Dimension};
use aliases::Mat;

pub fn affine_inverse<N: Real, D: Dimension>(m: Mat<N, D, D>) -> Mat<N, D, D>
    where DefaultAllocator: Alloc<N, D, D> {
    // FIXME: this should be optimized.
    m.try_inverse().unwrap_or(Mat::<_, D, D>::zeros())
}

pub fn inverse_transpose<N: Real, D: Dimension>(m: Mat<N, D, D>) -> Mat<N, D, D>
    where DefaultAllocator: Alloc<N, D, D> {
    m.try_inverse().unwrap_or(Mat::<_, D, D>::zeros()).transpose()
}