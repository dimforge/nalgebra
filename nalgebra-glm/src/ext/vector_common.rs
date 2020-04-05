use na::{self, DefaultAllocator};

use crate::aliases::TVec;
use crate::traits::{Alloc, Dimension, Number};

/// Component-wise maximum between a vector and a scalar.
///
/// # See also:
///
/// * [`comp_max`](fn.comp_max.html)
/// * [`comp_min`](fn.comp_min.html)
/// * [`max2`](fn.max2.html)
/// * [`max3`](fn.max3.html)
/// * [`max4`](fn.max4.html)
/// * [`min`](fn.min.html)
/// * [`min2`](fn.min2.html)
/// * [`min3`](fn.min3.html)
/// * [`min4`](fn.min4.html)
pub fn max<N: Number, D: Dimension>(a: &TVec<N, D>, b: N) -> TVec<N, D>
where
    DefaultAllocator: Alloc<N, D>,
{
    a.map(|a| crate::max2_scalar(a, b))
}

/// Component-wise maximum between two vectors.
///
/// # See also:
///
/// * [`comp_max`](fn.comp_max.html)
/// * [`comp_min`](fn.comp_min.html)
/// * [`max`](fn.max.html)
/// * [`max3`](fn.max3.html)
/// * [`max4`](fn.max4.html)
/// * [`min`](fn.min.html)
/// * [`min2`](fn.min2.html)
/// * [`min3`](fn.min3.html)
/// * [`min4`](fn.min4.html)
pub fn max2<N: Number, D: Dimension>(a: &TVec<N, D>, b: &TVec<N, D>) -> TVec<N, D>
where
    DefaultAllocator: Alloc<N, D>,
{
    a.zip_map(b, |a, b| crate::max2_scalar(a, b))
}

/// Component-wise maximum between three vectors.
///
/// # See also:
///
/// * [`comp_max`](fn.comp_max.html)
/// * [`comp_min`](fn.comp_min.html)
/// * [`max`](fn.max.html)
/// * [`max2`](fn.max2.html)
/// * [`max4`](fn.max4.html)
/// * [`min`](fn.min.html)
/// * [`min2`](fn.min2.html)
/// * [`min3`](fn.min3.html)
/// * [`min4`](fn.min4.html)
pub fn max3<N: Number, D: Dimension>(a: &TVec<N, D>, b: &TVec<N, D>, c: &TVec<N, D>) -> TVec<N, D>
where
    DefaultAllocator: Alloc<N, D>,
{
    max2(&max2(a, b), c)
}

/// Component-wise maximum between four vectors.
///
/// # See also:
///
/// * [`comp_max`](fn.comp_max.html)
/// * [`comp_min`](fn.comp_min.html)
/// * [`max`](fn.max.html)
/// * [`max2`](fn.max2.html)
/// * [`max3`](fn.max3.html)
/// * [`min`](fn.min.html)
/// * [`min2`](fn.min2.html)
/// * [`min3`](fn.min3.html)
/// * [`min4`](fn.min4.html)
pub fn max4<N: Number, D: Dimension>(
    a: &TVec<N, D>,
    b: &TVec<N, D>,
    c: &TVec<N, D>,
    d: &TVec<N, D>,
) -> TVec<N, D>
where
    DefaultAllocator: Alloc<N, D>,
{
    max2(&max2(a, b), &max2(c, d))
}

/// Component-wise minimum between a vector and a scalar.
///
/// # See also:
///
/// * [`comp_max`](fn.comp_max.html)
/// * [`comp_min`](fn.comp_min.html)
/// * [`max`](fn.max.html)
/// * [`max2`](fn.max2.html)
/// * [`max3`](fn.max3.html)
/// * [`max4`](fn.max4.html)
/// * [`min2`](fn.min2.html)
/// * [`min3`](fn.min3.html)
/// * [`min4`](fn.min4.html)
pub fn min<N: Number, D: Dimension>(x: &TVec<N, D>, y: N) -> TVec<N, D>
where
    DefaultAllocator: Alloc<N, D>,
{
    x.map(|x| crate::min2_scalar(x, y))
}

/// Component-wise minimum between two vectors.
///
/// # See also:
///
/// * [`comp_max`](fn.comp_max.html)
/// * [`comp_min`](fn.comp_min.html)
/// * [`max`](fn.max.html)
/// * [`max2`](fn.max2.html)
/// * [`max3`](fn.max3.html)
/// * [`max4`](fn.max4.html)
/// * [`min`](fn.min.html)
/// * [`min3`](fn.min3.html)
/// * [`min4`](fn.min4.html)
pub fn min2<N: Number, D: Dimension>(x: &TVec<N, D>, y: &TVec<N, D>) -> TVec<N, D>
where
    DefaultAllocator: Alloc<N, D>,
{
    x.zip_map(y, |a, b| crate::min2_scalar(a, b))
}

/// Component-wise minimum between three vectors.
///
/// # See also:
///
/// * [`comp_max`](fn.comp_max.html)
/// * [`comp_min`](fn.comp_min.html)
/// * [`max`](fn.max.html)
/// * [`max2`](fn.max2.html)
/// * [`max3`](fn.max3.html)
/// * [`max4`](fn.max4.html)
/// * [`min`](fn.min.html)
/// * [`min2`](fn.min2.html)
/// * [`min4`](fn.min4.html)
pub fn min3<N: Number, D: Dimension>(a: &TVec<N, D>, b: &TVec<N, D>, c: &TVec<N, D>) -> TVec<N, D>
where
    DefaultAllocator: Alloc<N, D>,
{
    min2(&min2(a, b), c)
}

/// Component-wise minimum between four vectors.
///
/// # See also:
///
/// * [`comp_max`](fn.comp_max.html)
/// * [`comp_min`](fn.comp_min.html)
/// * [`max`](fn.max.html)
/// * [`max2`](fn.max2.html)
/// * [`max3`](fn.max3.html)
/// * [`max4`](fn.max4.html)
/// * [`min`](fn.min.html)
/// * [`min2`](fn.min2.html)
/// * [`min3`](fn.min3.html)
pub fn min4<N: Number, D: Dimension>(
    a: &TVec<N, D>,
    b: &TVec<N, D>,
    c: &TVec<N, D>,
    d: &TVec<N, D>,
) -> TVec<N, D>
where
    DefaultAllocator: Alloc<N, D>,
{
    min2(&min2(a, b), &min2(c, d))
}
