use crate::aliases::TMat;
use crate::traits::Number;

/// The sum of every component of the given matrix or vector.
///
/// # Examples:
///
/// ```
/// # use nalgebra_glm as glm;
/// let vec = glm::vec2(3.0, 4.0);
/// assert_eq!(7.0, glm::comp_add(&vec));
///
/// let mat = glm::mat2(0.0, 1.0, -3.0, 3.0);
/// assert_eq!(1.0, glm::comp_add(&mat));
/// ```
///
/// # See also:
///
/// * [`comp_max()`]
/// * [`comp_min()`]
/// * [`comp_mul()`]
pub fn comp_add<T: Number, const R: usize, const C: usize>(m: &TMat<T, R, C>) -> T {
    m.iter().fold(T::zero(), |x, y| x + *y)
}

/// The maximum of every component of the given matrix or vector.
///
/// # Examples:
///
/// ```
/// # use nalgebra_glm as glm;
/// let vec = glm::vec2(3.0, 4.0);
/// assert_eq!(4.0, glm::comp_max(&vec));
///
/// let mat = glm::mat2(0.0, 1.0, -3.0, 3.0);
/// assert_eq!(3.0, glm::comp_max(&mat));
/// ```
///
/// # See also:
///
/// * [`comp_add()`]
/// * [`comp_max()`]
/// * [`comp_min()`]
/// * [`max()`](crate::max)
/// * [`max2()`](crate::max2)
/// * [`max3()`](crate::max3)
/// * [`max4()`](crate::max4)
pub fn comp_max<T: Number, const R: usize, const C: usize>(m: &TMat<T, R, C>) -> T {
    m.iter()
        .fold(T::min_value(), |x, y| crate::max2_scalar(x, *y))
}

/// The minimum of every component of the given matrix or vector.
///
/// # Examples:
///
/// ```
/// # use nalgebra_glm as glm;
/// let vec = glm::vec2(3.0, 4.0);
/// assert_eq!(3.0, glm::comp_min(&vec));
///
/// let mat = glm::mat2(0.0, 1.0, -3.0, 3.0);
/// assert_eq!(-3.0, glm::comp_min(&mat));
/// ```
///
/// # See also:
///
/// * [`comp_add()`]
/// * [`comp_max()`]
/// * [`comp_mul()`]
/// * [`min()`](crate::min)
/// * [`min2()`](crate::min2)
/// * [`min3()`](crate::min3)
/// * [`min4()`](crate::min4)
pub fn comp_min<T: Number, const R: usize, const C: usize>(m: &TMat<T, R, C>) -> T {
    m.iter()
        .fold(T::max_value(), |x, y| crate::min2_scalar(x, *y))
}

/// The product of every component of the given matrix or vector.
///
/// # Examples:
///
/// ```
/// # use nalgebra_glm as glm;
/// let vec = glm::vec2(3.0, 4.0);
/// assert_eq!(12.0, glm::comp_mul(&vec));
///
/// let mat = glm::mat2(1.0, 1.0, -3.0, 3.0);
/// assert_eq!(-9.0, glm::comp_mul(&mat));
/// ```
///
/// # See also:
///
/// * [`comp_add()`]
/// * [`comp_max()`]
/// * [`comp_min()`]
pub fn comp_mul<T: Number, const R: usize, const C: usize>(m: &TMat<T, R, C>) -> T {
    m.iter().fold(T::one(), |x, y| x * *y)
}

//pub fn vec< L, floatType, Q > compNormalize (vec< L, T, Q > const &v)
//pub fn vec< L, T, Q > compScale (vec< L, floatType, Q > const &v)
