use na::DefaultAllocator;

use aliases::TVec;
use traits::{Alloc, Dimension, Number};

/// Component-wise approximate equality of two vectors, using a scalar epsilon.
///
/// # See also:
///
/// * [`equal_eps_vec`](fn.equal_eps_vec.html)
/// * [`not_equal_eps`](fn.not_equal_eps.html)
/// * [`not_equal_eps_vec`](fn.not_equal_eps_vec.html)
pub fn equal_eps<N: Number, D: Dimension>(
    x: &TVec<N, D>,
    y: &TVec<N, D>,
    epsilon: N,
) -> TVec<bool, D>
where
    DefaultAllocator: Alloc<N, D>,
{
    x.zip_map(y, |x, y| abs_diff_eq!(x, y, epsilon = epsilon))
}

/// Component-wise approximate equality of two vectors, using a per-component epsilon.
///
/// # See also:
///
/// * [`equal_eps`](fn.equal_eps.html)
/// * [`not_equal_eps`](fn.not_equal_eps.html)
/// * [`not_equal_eps_vec`](fn.not_equal_eps_vec.html)
pub fn equal_eps_vec<N: Number, D: Dimension>(
    x: &TVec<N, D>,
    y: &TVec<N, D>,
    epsilon: &TVec<N, D>,
) -> TVec<bool, D>
where
    DefaultAllocator: Alloc<N, D>,
{
    x.zip_zip_map(y, epsilon, |x, y, eps| abs_diff_eq!(x, y, epsilon = eps))
}

/// Component-wise approximate non-equality of two vectors, using a scalar epsilon.
///
/// # See also:
///
/// * [`equal_eps`](fn.equal_eps.html)
/// * [`equal_eps_vec`](fn.equal_eps_vec.html)
/// * [`not_equal_eps_vec`](fn.not_equal_eps_vec.html)
pub fn not_equal_eps<N: Number, D: Dimension>(
    x: &TVec<N, D>,
    y: &TVec<N, D>,
    epsilon: N,
) -> TVec<bool, D>
where
    DefaultAllocator: Alloc<N, D>,
{
    x.zip_map(y, |x, y| abs_diff_ne!(x, y, epsilon = epsilon))
}

/// Component-wise approximate non-equality of two vectors, using a per-component epsilon.
///
/// # See also:
///
/// * [`equal_eps`](fn.equal_eps.html)
/// * [`equal_eps_vec`](fn.equal_eps_vec.html)
/// * [`not_equal_eps`](fn.not_equal_eps.html)
pub fn not_equal_eps_vec<N: Number, D: Dimension>(
    x: &TVec<N, D>,
    y: &TVec<N, D>,
    epsilon: &TVec<N, D>,
) -> TVec<bool, D>
where
    DefaultAllocator: Alloc<N, D>,
{
    x.zip_zip_map(y, epsilon, |x, y, eps| abs_diff_ne!(x, y, epsilon = eps))
}
