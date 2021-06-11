use crate::aliases::TVec;
use crate::traits::Number;

/// Component-wise approximate equality of two vectors, using a scalar epsilon.
///
/// # See also:
///
/// * [`equal_eps_vec`](fn.equal_eps_vec.html)
/// * [`not_equal_eps`](fn.not_equal_eps.html)
/// * [`not_equal_eps_vec`](fn.not_equal_eps_vec.html)
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
/// * [`equal_eps`](fn.equal_eps.html)
/// * [`not_equal_eps`](fn.not_equal_eps.html)
/// * [`not_equal_eps_vec`](fn.not_equal_eps_vec.html)
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
/// * [`equal_eps`](fn.equal_eps.html)
/// * [`equal_eps_vec`](fn.equal_eps_vec.html)
/// * [`not_equal_eps_vec`](fn.not_equal_eps_vec.html)
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
/// * [`equal_eps`](fn.equal_eps.html)
/// * [`equal_eps_vec`](fn.equal_eps_vec.html)
/// * [`not_equal_eps`](fn.not_equal_eps.html)
pub fn not_equal_eps_vec<T: Number, const D: usize>(
    x: &TVec<T, D>,
    y: &TVec<T, D>,
    epsilon: &TVec<T, D>,
) -> TVec<bool, D> {
    x.zip_zip_map(y, epsilon, |x, y, eps| abs_diff_ne!(x, y, epsilon = eps))
}
