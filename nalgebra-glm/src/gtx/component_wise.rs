use na::{self, DefaultAllocator};

use traits::{Number, Alloc, Dimension};
use aliases::TMat;

/// The sum of every components of the given matrix or vector.
pub fn comp_add<N: Number, R: Dimension, C: Dimension>(m: &TMat<N, R, C>) -> N
    where DefaultAllocator: Alloc<N, R, C> {
    m.iter().fold(N::zero(), |x, y| x + *y)
}

/// The maximum of every components of the given matrix or vector.
pub fn comp_max<N: Number, R: Dimension, C: Dimension>(m: &TMat<N, R, C>) -> N
    where DefaultAllocator: Alloc<N, R, C> {
    m.iter().fold(N::min_value(), |x, y| na::sup(&x, y))
}

/// The minimum of every components of the given matrix or vector.
pub fn comp_min<N: Number, R: Dimension, C: Dimension>(m: &TMat<N, R, C>) -> N
    where DefaultAllocator: Alloc<N, R, C> {
    m.iter().fold(N::max_value(), |x, y| na::inf(&x, y))
}

/// The product of every components of the given matrix or vector.
pub fn comp_mul<N: Number, R: Dimension, C: Dimension>(m: &TMat<N, R, C>) -> N
    where DefaultAllocator: Alloc<N, R, C> {
    m.iter().fold(N::one(), |x, y| x * *y)
}

//pub fn vec< L, floatType, Q > compNormalize (vec< L, T, Q > const &v)
//pub fn vec< L, T, Q > compScale (vec< L, floatType, Q > const &v)