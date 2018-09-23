use na::{Real, DefaultAllocator};

use traits::{Dimension, Alloc};
use aliases::TVec;

/// The dot product of the normalized version of `x` and `y`.
pub fn fast_normalize_dot<N: Real, D: Dimension>(x: &TVec<N, D>, y: &TVec<N, D>) -> N
    where DefaultAllocator: Alloc<N, D> {
    // XXX: improve those.
    x.normalize().dot(&y.normalize())
}

/// The dot product of the normalized version of `x` and `y`.
pub fn normalize_dot<N: Real, D: Dimension>(x: &TVec<N, D>, y: &TVec<N, D>) -> N
    where DefaultAllocator: Alloc<N, D> {
    // XXX: improve those.
    x.normalize().dot(&y.normalize())
}