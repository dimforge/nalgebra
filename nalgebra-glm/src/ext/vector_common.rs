use crate::aliases::TVec;
use crate::traits::Number;

/// Component-wise maximum between a vector and a scalar.
///
/// # See also:
///
/// * [`comp_max()`](crate::comp_max)
/// * [`comp_min()`](crate::comp_min)
/// * [`max2()`]
/// * [`max3()`]
/// * [`max4()`]
/// * [`min()`]
/// * [`min2()`]
/// * [`min3()`]
/// * [`min4()`]
pub fn max<T: Number, const D: usize>(a: &TVec<T, D>, b: T) -> TVec<T, D> {
    a.map(|a| crate::max2_scalar(a, b))
}

/// Component-wise maximum between two vectors.
///
/// # See also:
///
/// * [`comp_max()`](crate::comp_max)
/// * [`comp_min()`](crate::comp_min)
/// * [`max()`]
/// * [`max3()`]
/// * [`max4()`]
/// * [`min()`]
/// * [`min2()`]
/// * [`min3()`]
/// * [`min4()`]
pub fn max2<T: Number, const D: usize>(a: &TVec<T, D>, b: &TVec<T, D>) -> TVec<T, D> {
    a.zip_map(b, |a, b| crate::max2_scalar(a, b))
}

/// Component-wise maximum between three vectors.
///
/// # See also:
///
/// * [`comp_max()`](crate::comp_max)
/// * [`comp_min()`](crate::comp_min)
/// * [`max()`]
/// * [`max2()`]
/// * [`max4()`]
/// * [`min()`]
/// * [`min2()`]
/// * [`min3()`]
/// * [`min4()`]
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
/// * [`comp_max()`](crate::comp_max)
/// * [`comp_min()`](crate::comp_min)
/// * [`max()`]
/// * [`max2()`]
/// * [`max3()`]
/// * [`min()`]
/// * [`min2()`]
/// * [`min3()`]
/// * [`min4()`]
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
/// * [`comp_max()`](crate::comp_max)
/// * [`comp_min()`](crate::comp_min)
/// * [`max()`]
/// * [`max2()`]
/// * [`max3()`]
/// * [`max4()`]
/// * [`min2()`]
/// * [`min3()`]
/// * [`min4()`]
pub fn min<T: Number, const D: usize>(x: &TVec<T, D>, y: T) -> TVec<T, D> {
    x.map(|x| crate::min2_scalar(x, y))
}

/// Component-wise minimum between two vectors.
///
/// # See also:
///
/// * [`comp_max()`](crate::comp_max)
/// * [`comp_min()`](crate::comp_min)
/// * [`max()`]
/// * [`max2()`]
/// * [`max3()`]
/// * [`max4()`]
/// * [`min()`]
/// * [`min3()`]
/// * [`min4()`]
pub fn min2<T: Number, const D: usize>(x: &TVec<T, D>, y: &TVec<T, D>) -> TVec<T, D> {
    x.zip_map(y, |a, b| crate::min2_scalar(a, b))
}

/// Component-wise minimum between three vectors.
///
/// # See also:
///
/// * [`comp_max()`](crate::comp_max)
/// * [`comp_min()`](crate::comp_min)
/// * [`max()`]
/// * [`max2()`]
/// * [`max3()`]
/// * [`max4()`]
/// * [`min()`]
/// * [`min2()`]
/// * [`min4()`]
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
/// * [`comp_max()`](crate::comp_max)
/// * [`comp_min()`](crate::comp_min)
/// * [`max()`]
/// * [`max2()`]
/// * [`max3()`]
/// * [`max4()`]
/// * [`min()`]
/// * [`min2()`]
/// * [`min3()`]
pub fn min4<T: Number, const D: usize>(
    a: &TVec<T, D>,
    b: &TVec<T, D>,
    c: &TVec<T, D>,
    d: &TVec<T, D>,
) -> TVec<T, D> {
    min2(&min2(a, b), &min2(c, d))
}
