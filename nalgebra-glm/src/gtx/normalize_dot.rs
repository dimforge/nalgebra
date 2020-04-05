use na::{DefaultAllocator, RealField};

use crate::aliases::TVec;
use crate::traits::{Alloc, Dimension};

/// The dot product of the normalized version of `x` and `y`.
///
/// This is currently the same as [`normalize_dot`](fn.normalize_dot.html).
///
/// # See also:
///
/// * [`normalize_dot`](fn.normalize_dot.html`)
pub fn fast_normalize_dot<N: RealField, D: Dimension>(x: &TVec<N, D>, y: &TVec<N, D>) -> N
where
    DefaultAllocator: Alloc<N, D>,
{
    // XXX: improve those.
    x.normalize().dot(&y.normalize())
}

/// The dot product of the normalized version of `x` and `y`.
///
/// # See also:
///
/// * [`fast_normalize_dot`](fn.fast_normalize_dot.html`)
pub fn normalize_dot<N: RealField, D: Dimension>(x: &TVec<N, D>, y: &TVec<N, D>) -> N
where
    DefaultAllocator: Alloc<N, D>,
{
    // XXX: improve those.
    x.normalize().dot(&y.normalize())
}
