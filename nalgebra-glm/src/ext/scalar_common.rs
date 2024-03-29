use crate::traits::Number;

/// Returns the maximum among two values.
///
/// # Examples:
///
/// ```
/// # use nalgebra_glm as glm;
/// assert_eq!(2.0, glm::max2_scalar(1.0, 2.0));
/// assert_eq!(1, glm::max2_scalar(0, 1));
/// ```
///
/// # See also:
///
/// * [`max4_scalar()`]
/// * [`min3_scalar()`]
/// * [`min4_scalar()`]
pub fn max2_scalar<T: Number>(a: T, b: T) -> T {
    if a >= b {
        a
    } else {
        b
    }
}

/// Returns the maximum among two values.
///
/// # Examples:
///
/// ```
/// # use nalgebra_glm as glm;
/// assert_eq!(1.0, glm::min2_scalar(1.0, 2.0));
/// assert_eq!(0, glm::min2_scalar(0, 1));
/// ```
///
/// # See also:
///
/// * [`max4_scalar()`]
/// * [`min3_scalar()`]
/// * [`min4_scalar()`]
pub fn min2_scalar<T: Number>(a: T, b: T) -> T {
    if a <= b {
        a
    } else {
        b
    }
}

/// Returns the maximum among three values.
///
/// # Examples:
///
/// ```
/// # use nalgebra_glm as glm;
/// assert_eq!(3.0, glm::max3_scalar(1.0, 2.0, 3.0));
/// assert_eq!(2, glm::max3_scalar(0, 1, 2));
/// ```
///
/// # See also:
///
/// * [`max4_scalar()`]
/// * [`min3_scalar()`]
/// * [`min4_scalar()`]
pub fn max3_scalar<T: Number>(a: T, b: T, c: T) -> T {
    max2_scalar(max2_scalar(a, b), c)
}

/// Returns the maximum among four values.
///
/// # Examples:
///
/// ```
/// # use nalgebra_glm as glm;
/// assert_eq!(4.0, glm::max4_scalar(2.0, 4.0, 1.0, 3.0));
/// assert_eq!(7, glm::max4_scalar(1, 0, 7, 2));
/// ```
///
/// # See also:
///
/// * [`max3_scalar()`]
/// * [`min3_scalar()`]
/// * [`min4_scalar()`]
pub fn max4_scalar<T: Number>(a: T, b: T, c: T, d: T) -> T {
    max2_scalar(max2_scalar(a, b), max2_scalar(c, d))
}

/// Returns the minimum among three values.
///
/// # Examples:
///
/// ```
/// # use nalgebra_glm as glm;
/// assert_eq!(1.0, glm::min3_scalar(1.0, 2.0, 3.0));
/// assert_eq!(0, glm::min3_scalar(0, 1, 2));
/// ```
///
/// # See also:
///
/// * [`max3_scalar()`]
/// * [`max4_scalar()`]
/// * [`min4_scalar()`]
pub fn min3_scalar<T: Number>(a: T, b: T, c: T) -> T {
    min2_scalar(min2_scalar(a, b), c)
}

/// Returns the minimum among four values.
///
/// # Examples:
///
/// ```
/// # use nalgebra_glm as glm;
/// assert_eq!(1.0, glm::min4_scalar(2.0, 4.0, 1.0, 3.0));
/// assert_eq!(0, glm::min4_scalar(1, 0, 7, 2));
/// ```
///
/// # See also:
///
/// * [`max3_scalar()`]
/// * [`max4_scalar()`]
/// * [`min3_scalar()`]
pub fn min4_scalar<T: Number>(a: T, b: T, c: T, d: T) -> T {
    min2_scalar(min2_scalar(a, b), min2_scalar(c, d))
}
