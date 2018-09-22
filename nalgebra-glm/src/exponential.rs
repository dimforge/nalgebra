use na::{Real, DefaultAllocator};
use aliases::Vec;
use traits::{Alloc, Dimension};

/// Componentwise exponential.
pub fn exp<N: Real, D: Dimension>(v: &Vec<N, D>) -> Vec<N, D>
    where DefaultAllocator: Alloc<N, D> {
    v.map(|x| x.exp())
}

/// Componentwise base-2 exponential.
pub fn exp2<N: Real, D: Dimension>(v: &Vec<N, D>) -> Vec<N, D>
    where DefaultAllocator: Alloc<N, D> {
    v.map(|x| x.exp2())
}

/// Compute the inverse of the square root of each component of `v`.
pub fn inversesqrt<N: Real, D: Dimension>(v: &Vec<N, D>) -> Vec<N, D>
    where DefaultAllocator: Alloc<N, D> {
    v.map(|x| N::one() / x.sqrt())

}

/// Componentwise logarithm.
pub fn log<N: Real, D: Dimension>(v: &Vec<N, D>) -> Vec<N, D>
    where DefaultAllocator: Alloc<N, D> {
    v.map(|x| x.ln())
}

/// Componentwise base-2 logarithm.
pub fn log2<N: Real, D: Dimension>(v: &Vec<N, D>) -> Vec<N, D>
    where DefaultAllocator: Alloc<N, D> {
    v.map(|x| x.log2())
}

/// Componentwise power.
pub fn pow<N: Real, D: Dimension>(base: &Vec<N, D>, exponent: &Vec<N, D>) -> Vec<N, D>
    where DefaultAllocator: Alloc<N, D> {
    base.zip_map(exponent, |b, e| b.powf(e))
}

/// Componentwise square root.
pub fn sqrt<N: Real, D: Dimension>(v: &Vec<N, D>) -> Vec<N, D>
    where DefaultAllocator: Alloc<N, D> {
    v.map(|x| x.sqrt())
}
