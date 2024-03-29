use crate::RealNumber;

use crate::aliases::TVec;

/// The dot product of the normalized version of `x` and `y`.
///
/// This is currently the same as [`normalize_dot()`]
///
/// # See also:
///
/// * [`normalize_dot()`]
pub fn fast_normalize_dot<T: RealNumber, const D: usize>(x: &TVec<T, D>, y: &TVec<T, D>) -> T {
    // XXX: improve those.
    x.normalize().dot(&y.normalize())
}

/// The dot product of the normalized version of `x` and `y`.
///
/// # See also:
///
/// * [`fast_normalize_dot()`]
pub fn normalize_dot<T: RealNumber, const D: usize>(x: &TVec<T, D>, y: &TVec<T, D>) -> T {
    // XXX: improve those.
    x.normalize().dot(&y.normalize())
}
