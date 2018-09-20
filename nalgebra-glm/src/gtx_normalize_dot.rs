use na::{Real, DefaultAllocator};

use traits::{Dimension, Alloc};
use aliases::Vec;


pub fn fastNormalizeDot<N: Real, D: Dimension>(x: &Vec<N, D>, y: &Vec<N, D>) -> N
    where DefaultAllocator: Alloc<N, D> {
    // XXX: improve those.
    x.normalize().dot(&y.normalize())
}

pub fn normalizeDot<N: Real, D: Dimension>(x: &Vec<N, D>, y: &Vec<N, D>) -> N
    where DefaultAllocator: Alloc<N, D> {
    // XXX: improve those.
    x.normalize().dot(&y.normalize())
}