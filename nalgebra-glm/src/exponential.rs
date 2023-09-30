use crate::aliases::TVec;
use crate::RealNumber;

/// Component-wise exponential.
///
/// # See also:
///
/// * [`exp2()`]
pub fn exp<T: RealNumber, const D: usize>(v: &TVec<T, D>) -> TVec<T, D> {
    v.map(|x| x.exp())
}

/// Component-wise base-2 exponential.
///
/// # See also:
///
/// * [`exp()`]
pub fn exp2<T: RealNumber, const D: usize>(v: &TVec<T, D>) -> TVec<T, D> {
    v.map(|x| x.exp2())
}

/// Compute the inverse of the square root of each component of `v`.
///
/// # See also:
///
/// * [`sqrt()`]
pub fn inversesqrt<T: RealNumber, const D: usize>(v: &TVec<T, D>) -> TVec<T, D> {
    v.map(|x| T::one() / x.sqrt())
}

/// Component-wise logarithm.
///
/// # See also:
///
/// * [`log2()`]
pub fn log<T: RealNumber, const D: usize>(v: &TVec<T, D>) -> TVec<T, D> {
    v.map(|x| x.ln())
}

/// Component-wise base-2 logarithm.
///
/// # See also:
///
/// * [`log()`]
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
/// * [`exp()`]
/// * [`exp2()`]
/// * [`inversesqrt()`]
/// * [`pow`]
pub fn sqrt<T: RealNumber, const D: usize>(v: &TVec<T, D>) -> TVec<T, D> {
    v.map(|x| x.sqrt())
}
