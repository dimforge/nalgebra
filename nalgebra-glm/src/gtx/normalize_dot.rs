use na::{Real, DefaultAllocator};

use traits::{Dimension, Alloc};
use aliases::Vec;

/// The dot product of the normalized version of `x` and `y`.
pub fn fast_normalize_dot<N: Real, D: Dimension>(x: &Vec<N, D>, y: &Vec<N, D>) -> N
    where DefaultAllocator: Alloc<N, D> {
    // XXX: improve those.
    x.normalize().dot(&y.normalize())
}

/// The dot product of the normalized version of `x` and `y`.
pub fn normalize_dot<N: Real, D: Dimension>(x: &Vec<N, D>, y: &Vec<N, D>) -> N
    where DefaultAllocator: Alloc<N, D> {
    // XXX: improve those.
    x.normalize().dot(&y.normalize())
}