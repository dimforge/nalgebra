use approx::AbsDiffEq;
use na::DefaultAllocator;

use traits::{Alloc, Number, Dimension};
use aliases::Vec;

/// Component-wise approximate equality beween two vectors.
pub fn epsilon_equal<N: Number, D: Dimension>(x: &Vec<N, D>, y: &Vec<N, D>, epsilon: N) -> Vec<bool, D>
    where DefaultAllocator: Alloc<N, D> {
    x.zip_map(y, |x, y| abs_diff_eq!(x, y, epsilon = epsilon))
}

/// Component-wise approximate equality beween two scalars.
pub fn epsilon_equal2<N: AbsDiffEq<Epsilon = N>>(x: N, y: N, epsilon: N) -> bool {
    abs_diff_eq!(x, y, epsilon = epsilon)
}

/// Component-wise approximate non-equality beween two vectors.
pub fn epsilon_not_equal<N: Number, D: Dimension>(x: &Vec<N, D>, y: &Vec<N, D>, epsilon: N) -> Vec<bool, D>
    where DefaultAllocator: Alloc<N, D> {
    x.zip_map(y, |x, y| abs_diff_ne!(x, y, epsilon = epsilon))
}

/// Component-wise approximate non-equality beween two scalars.
pub fn epsilon_not_equal2<N: AbsDiffEq<Epsilon = N>>(x: N, y: N, epsilon: N) -> bool {
    abs_diff_ne!(x, y, epsilon = epsilon)
}
