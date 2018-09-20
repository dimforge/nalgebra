use na::{self, Scalar, Real, DefaultAllocator};

use traits::{Number, Alloc, Dimension};
use aliases::Mat;


pub fn comp_add<N: Number, R: Dimension, C: Dimension>(m: &Mat<N, R, C>) -> N
    where DefaultAllocator: Alloc<N, R, C> {
    m.iter().fold(N::zero(), |x, y| x + *y)
}

pub fn comp_max<N: Number, R: Dimension, C: Dimension>(m: &Mat<N, R, C>) -> N
    where DefaultAllocator: Alloc<N, R, C> {
    m.iter().fold(N::min_value(), |x, y| na::sup(&x, y))
}

pub fn comp_min<N: Number, R: Dimension, C: Dimension>(m: &Mat<N, R, C>) -> N
    where DefaultAllocator: Alloc<N, R, C> {
    m.iter().fold(N::max_value(), |x, y| na::inf(&x, y))
}

pub fn comp_mul<N: Number, R: Dimension, C: Dimension>(m: &Mat<N, R, C>) -> N
    where DefaultAllocator: Alloc<N, R, C> {
    m.iter().fold(N::one(), |x, y| x * *y)
}

//pub fn vec< L, floatType, Q > compNormalize (vec< L, T, Q > const &v)
//pub fn vec< L, T, Q > compScale (vec< L, floatType, Q > const &v)