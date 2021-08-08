use crate::aliases::TVec;
use crate::RealNumber;

/// Component-wise exponential.
///
/// # See also:
///
/// * [`exp2`](fn.exp2.html)
pub fn exp<T: RealNumber, const D: usize>(v: &TVec<T, D>) -> TVec<T, D> {
    v.map(|x| x.exp())
}

/// Component-wise base-2 exponential.
///
/// # See also:
///
/// * [`exp`](fn.exp.html)
pub fn exp2<T: RealNumber, const D: usize>(v: &TVec<T, D>) -> TVec<T, D> {
    v.map(|x| x.exp2())
}

/// Compute the inverse of the square root of each component of `v`.
///
/// # See also:
///
/// * [`sqrt`](fn.sqrt.html)
pub fn inversesqrt<T: RealNumber, const D: usize>(v: &TVec<T, D>) -> TVec<T, D> {
    v.map(|x| T::one() / x.sqrt())
}

/// Component-wise logarithm.
///
/// # See also:
///
/// * [`log2`](fn.log2.html)
pub fn log<T: RealNumber, const D: usize>(v: &TVec<T, D>) -> TVec<T, D> {
    v.map(|x| x.ln())
}

/// Component-wise base-2 logarithm.
///
/// # See also:
///
/// * [`log`](fn.log.html)
pub fn log2<T: RealNumber, const D: usize>(v: &TVec<T, D>) -> TVec<T, D> {
    v.map(|x| x.log2())
}

/// Component-wise power.
pub fn pow<T: RealNumber, const D: usize>(base: &TVec<T, D>, exponent: &TVec<T, D>) -> TVec<T, D> {
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
pub fn sqrt<T: RealNumber, const D: usize>(v: &TVec<T, D>) -> TVec<T, D> {
    v.map(|x| x.sqrt())
}
