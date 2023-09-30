use crate::aliases::TVec;
use crate::traits::Number;

/// Component-wise approximate equality of two vectors, using a scalar epsilon.
///
/// # See also:
///
/// * [`equal_eps_vec()`]
/// * [`not_equal_eps()`]
/// * [`not_equal_eps_vec()`]
pub fn equal_eps<T: Number, const D: usize>(
    x: &TVec<T, D>,
    y: &TVec<T, D>,
    epsilon: T,
) -> TVec<bool, D> {
    x.zip_map(y, |x, y| abs_diff_eq!(x, y, epsilon = epsilon))
}

/// Component-wise approximate equality of two vectors, using a per-component epsilon.
///
/// # See also:
///
/// * [`equal_eps()`]
/// * [`not_equal_eps()`]
/// * [`not_equal_eps_vec()`]
pub fn equal_eps_vec<T: Number, const D: usize>(
    x: &TVec<T, D>,
    y: &TVec<T, D>,
    epsilon: &TVec<T, D>,
) -> TVec<bool, D> {
    x.zip_zip_map(y, epsilon, |x, y, eps| abs_diff_eq!(x, y, epsilon = eps))
}

/// Component-wise approximate non-equality of two vectors, using a scalar epsilon.
///
/// # See also:
///
/// * [`equal_eps()`]
/// * [`equal_eps_vec()`]
/// * [`not_equal_eps_vec()`]
pub fn not_equal_eps<T: Number, const D: usize>(
    x: &TVec<T, D>,
    y: &TVec<T, D>,
    epsilon: T,
) -> TVec<bool, D> {
    x.zip_map(y, |x, y| abs_diff_ne!(x, y, epsilon = epsilon))
}

/// Component-wise approximate non-equality of two vectors, using a per-component epsilon.
///
/// # See also:
///
/// * [`equal_eps()`]
/// * [`equal_eps_vec()`]
/// * [`not_equal_eps()`]
pub fn not_equal_eps_vec<T: Number, const D: usize>(
    x: &TVec<T, D>,
    y: &TVec<T, D>,
    epsilon: &TVec<T, D>,
) -> TVec<bool, D> {
    x.zip_zip_map(y, epsilon, |x, y, eps| abs_diff_ne!(x, y, epsilon = eps))
}
