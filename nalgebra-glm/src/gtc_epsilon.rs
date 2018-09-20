use approx::AbsDiffEq;
use na::DefaultAllocator;

use traits::{Alloc, Number, Dimension};
use aliases::Vec;

pub fn epsilonEqual<N: Number, D: Dimension>(x: &Vec<N, D>, y: &Vec<N, D>, epsilon: N) -> Vec<bool, D>
    where DefaultAllocator: Alloc<N, D> {
    x.zip_map(y, |x, y| abs_diff_eq!(x, y, epsilon = epsilon))
}

pub fn epsilonEqual2<N: AbsDiffEq<Epsilon = N>>(x: N, y: N, epsilon: N) -> bool {
    abs_diff_eq!(x, y, epsilon = epsilon)
}

pub fn epsilonNotEqual<N: Number, D: Dimension>(x: &Vec<N, D>, y: &Vec<N, D>, epsilon: N) -> Vec<bool, D>
    where DefaultAllocator: Alloc<N, D> {
    x.zip_map(y, |x, y| abs_diff_ne!(x, y, epsilon = epsilon))
}

pub fn epsilonNotEqual2<N: AbsDiffEq<Epsilon = N>>(x: N, y: N, epsilon: N) -> bool {
    abs_diff_ne!(x, y, epsilon = epsilon)
}
