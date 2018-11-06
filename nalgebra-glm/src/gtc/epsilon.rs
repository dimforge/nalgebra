// NOTE those are actually duplicates of vector_relational.rs

/*
use approx::AbsDiffEq;
use na::DefaultAllocator;

use traits::{Alloc, Number, Dimension};
use aliases::TVec;

/// Component-wise approximate equality beween two vectors.
pub fn epsilon_equal<N: Number, D: Dimension>(x: &TVec<N, D>, y: &TVec<N, D>, epsilon: N) -> TVec<bool, D>
    where DefaultAllocator: Alloc<N, D> {
    x.zip_map(y, |x, y| abs_diff_eq!(x, y, epsilon = epsilon))
}

/// Component-wise approximate equality beween two scalars.
pub fn epsilon_equal2<N: AbsDiffEq<Epsilon = N>>(x: N, y: N, epsilon: N) -> bool {
    abs_diff_eq!(x, y, epsilon = epsilon)
}

/// Component-wise approximate non-equality beween two vectors.
pub fn epsilon_not_equal<N: Number, D: Dimension>(x: &TVec<N, D>, y: &TVec<N, D>, epsilon: N) -> TVec<bool, D>
    where DefaultAllocator: Alloc<N, D> {
    x.zip_map(y, |x, y| abs_diff_ne!(x, y, epsilon = epsilon))
}

/// Component-wise approximate non-equality beween two scalars.
pub fn epsilon_not_equal2<N: AbsDiffEq<Epsilon = N>>(x: N, y: N, epsilon: N) -> bool {
    abs_diff_ne!(x, y, epsilon = epsilon)
}
*/
