// NOTE those are actually duplicates of vector_relational.rs

/*
use approx::AbsDiffEq;
use na::DefaultAllocator;

use crate::traits::{Alloc, Number, Dimension};
use crate::aliases::TVec;

/// Component-wise approximate equality beween two vectors.
pub fn epsilon_equal<T: Number, const D: usize>(x: &TVec<T, D>, y: &TVec<T, D>, epsilon: T) -> TVec<bool, D>
    where DefaultAllocator: Alloc<T, D> {
    x.zip_map(y, |x, y| abs_diff_eq!(x, y, epsilon = epsilon))
}

/// Component-wise approximate equality beween two scalars.
pub fn epsilon_equal2<T: AbsDiffEq<Epsilon = T>>(x: T, y: T, epsilon: T) -> bool {
    abs_diff_eq!(x, y, epsilon = epsilon)
}

/// Component-wise approximate non-equality beween two vectors.
pub fn epsilon_not_equal<T: Number, const D: usize>(x: &TVec<T, D>, y: &TVec<T, D>, epsilon: T) -> TVec<bool, D>
    where DefaultAllocator: Alloc<T, D> {
    x.zip_map(y, |x, y| abs_diff_ne!(x, y, epsilon = epsilon))
}

/// Component-wise approximate non-equality beween two scalars.
pub fn epsilon_not_equal2<T: AbsDiffEq<Epsilon = T>>(x: T, y: T, epsilon: T) -> bool {
    abs_diff_ne!(x, y, epsilon = epsilon)
}
*/
