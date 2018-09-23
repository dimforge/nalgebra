use na::{DefaultAllocator};

use traits::{Alloc, Number, Dimension};
use aliases::TVec;

/// Component-wise approximate equality of two vectors, using a scalar epsilon.
pub fn equal_eps<N: Number, D: Dimension>(x: &TVec<N, D>, y: &TVec<N, D>, epsilon: N) -> TVec<bool, D>
    where DefaultAllocator: Alloc<N, D> {
    x.zip_map(y, |x, y| abs_diff_eq!(x, y, epsilon = epsilon))
}

/// Component-wise approximate equality of two vectors, using a per-component epsilon.
pub fn equal_eps_vec<N: Number, D: Dimension>(x: &TVec<N, D>, y: &TVec<N, D>, epsilon: &TVec<N, D>) -> TVec<bool, D>
    where DefaultAllocator: Alloc<N, D> {
    x.zip_zip_map(y, epsilon, |x, y, eps| abs_diff_eq!(x, y, epsilon = eps))
}

/// Component-wise approximate non-equality of two vectors, using a scalar epsilon.
pub fn not_equal_eps<N: Number, D: Dimension>(x: &TVec<N, D>, y: &TVec<N, D>, epsilon: N) -> TVec<bool, D>
    where DefaultAllocator: Alloc<N, D> {
    x.zip_map(y, |x, y| abs_diff_ne!(x, y, epsilon = epsilon))
}

/// Component-wise approximate non-equality of two vectors, using a per-component epsilon.
pub fn not_equal_eps_vec<N: Number, D: Dimension>(x: &TVec<N, D>, y: &TVec<N, D>, epsilon: &TVec<N, D>) -> TVec<bool, D>
    where DefaultAllocator: Alloc<N, D> {
    x.zip_zip_map(y, epsilon, |x, y, eps| abs_diff_ne!(x, y, epsilon = eps))
}
