use crate::aliases::TVec;
use crate::traits::Number;

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
pub fn max<T: Number, const D: usize>(a: &TVec<T, D>, b: T) -> TVec<T, D> {
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
pub fn max2<T: Number, const D: usize>(a: &TVec<T, D>, b: &TVec<T, D>) -> TVec<T, D> {
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
pub fn max3<T: Number, const D: usize>(
    a: &TVec<T, D>,
    b: &TVec<T, D>,
    c: &TVec<T, D>,
) -> TVec<T, D> {
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
pub fn max4<T: Number, const D: usize>(
    a: &TVec<T, D>,
    b: &TVec<T, D>,
    c: &TVec<T, D>,
    d: &TVec<T, D>,
) -> TVec<T, D> {
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
pub fn min<T: Number, const D: usize>(x: &TVec<T, D>, y: T) -> TVec<T, D> {
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
pub fn min2<T: Number, const D: usize>(x: &TVec<T, D>, y: &TVec<T, D>) -> TVec<T, D> {
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
pub fn min3<T: Number, const D: usize>(
    a: &TVec<T, D>,
    b: &TVec<T, D>,
    c: &TVec<T, D>,
) -> TVec<T, D> {
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
pub fn min4<T: Number, const D: usize>(
    a: &TVec<T, D>,
    b: &TVec<T, D>,
    c: &TVec<T, D>,
    d: &TVec<T, D>,
) -> TVec<T, D> {
    min2(&min2(a, b), &min2(c, d))
}
