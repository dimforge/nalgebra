use na::{DefaultAllocator};

use traits::{Alloc, Number, Dimension};
use aliases::Vec;

pub fn equal_eps<N: Number, D: Dimension>(x: &Vec<N, D>, y: &Vec<N, D>, epsilon: N) -> Vec<bool, D>
    where DefaultAllocator: Alloc<N, D> {
    x.zip_map(y, |x, y| abs_diff_eq!(x, y))
}

pub fn equal_eps_vec<N: Number, D: Dimension>(x: &Vec<N, D>, y: &Vec<N, D>, epsilon: &Vec<N, D>) -> Vec<bool, D>
    where DefaultAllocator: Alloc<N, D> {
    x.zip_zip_map(y, epsilon, |x, y, eps| abs_diff_eq!(x, y, epsilon = eps))
}

pub fn not_equal_eps<N: Number, D: Dimension>(x: &Vec<N, D>, y: &Vec<N, D>, epsilon: N) -> Vec<bool, D>
    where DefaultAllocator: Alloc<N, D> {
    x.zip_map(y, |x, y| abs_diff_ne!(x, y))
}

pub fn not_equal_eps_vec<N: Number, D: Dimension>(x: &Vec<N, D>, y: &Vec<N, D>, epsilon: &Vec<N, D>) -> Vec<bool, D>
    where DefaultAllocator: Alloc<N, D> {
    x.zip_zip_map(y, epsilon, |x, y, eps| abs_diff_ne!(x, y, epsilon = eps))
}
