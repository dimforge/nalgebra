use na::{Real, DefaultAllocator};
use aliases::TVec;
use traits::{Alloc, Dimension};

/// Component-wise exponential.
pub fn exp<N: Real, D: Dimension>(v: &TVec<N, D>) -> TVec<N, D>
    where DefaultAllocator: Alloc<N, D> {
    v.map(|x| x.exp())
}

/// Component-wise base-2 exponential.
pub fn exp2<N: Real, D: Dimension>(v: &TVec<N, D>) -> TVec<N, D>
    where DefaultAllocator: Alloc<N, D> {
    v.map(|x| x.exp2())
}

/// Compute the inverse of the square root of each component of `v`.
pub fn inversesqrt<N: Real, D: Dimension>(v: &TVec<N, D>) -> TVec<N, D>
    where DefaultAllocator: Alloc<N, D> {
    v.map(|x| N::one() / x.sqrt())

}

/// Component-wise logarithm.
pub fn log<N: Real, D: Dimension>(v: &TVec<N, D>) -> TVec<N, D>
    where DefaultAllocator: Alloc<N, D> {
    v.map(|x| x.ln())
}

/// Component-wise base-2 logarithm.
pub fn log2<N: Real, D: Dimension>(v: &TVec<N, D>) -> TVec<N, D>
    where DefaultAllocator: Alloc<N, D> {
    v.map(|x| x.log2())
}

/// Component-wise power.
pub fn pow<N: Real, D: Dimension>(base: &TVec<N, D>, exponent: &TVec<N, D>) -> TVec<N, D>
    where DefaultAllocator: Alloc<N, D> {
    base.zip_map(exponent, |b, e| b.powf(e))
}

/// Component-wise square root.
pub fn sqrt<N: Real, D: Dimension>(v: &TVec<N, D>) -> TVec<N, D>
    where DefaultAllocator: Alloc<N, D> {
    v.map(|x| x.sqrt())
}
