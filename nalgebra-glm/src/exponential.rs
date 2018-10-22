use aliases::TVec;
use na::{DefaultAllocator, Real};
use traits::{Alloc, Dimension};

/// Component-wise exponential.
///
/// # See also:
///
/// * [`exp2`](fn.exp2.html)
pub fn exp<N: Real, D: Dimension>(v: &TVec<N, D>) -> TVec<N, D>
where DefaultAllocator: Alloc<N, D> {
    v.map(|x| x.exp())
}

/// Component-wise base-2 exponential.
///
/// # See also:
///
/// * [`exp`](fn.exp.html)
pub fn exp2<N: Real, D: Dimension>(v: &TVec<N, D>) -> TVec<N, D>
where DefaultAllocator: Alloc<N, D> {
    v.map(|x| x.exp2())
}

/// Compute the inverse of the square root of each component of `v`.
///
/// # See also:
///
/// * [`sqrt`](fn.sqrt.html)
pub fn inversesqrt<N: Real, D: Dimension>(v: &TVec<N, D>) -> TVec<N, D>
where DefaultAllocator: Alloc<N, D> {
    v.map(|x| N::one() / x.sqrt())
}

/// Component-wise logarithm.
///
/// # See also:
///
/// * [`log2`](fn.log2.html)
pub fn log<N: Real, D: Dimension>(v: &TVec<N, D>) -> TVec<N, D>
where DefaultAllocator: Alloc<N, D> {
    v.map(|x| x.ln())
}

/// Component-wise base-2 logarithm.
///
/// # See also:
///
/// * [`log`](fn.log.html)
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
///
/// # See also:
///
/// * [`exp`](fn.exp.html)
/// * [`exp2`](fn.exp2.html)
/// * [`inversesqrt`](fn.inversesqrt.html)
/// * [`pow`](fn.pow.html)
pub fn sqrt<N: Real, D: Dimension>(v: &TVec<N, D>) -> TVec<N, D>
where DefaultAllocator: Alloc<N, D> {
    v.map(|x| x.sqrt())
}
