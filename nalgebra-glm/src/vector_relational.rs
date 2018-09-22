use na::{DefaultAllocator};

use aliases::Vec;
use traits::{Number, Alloc, Dimension};

/// Checks that all the vector components are `true`.
pub fn all<D: Dimension>(v: &Vec<bool, D>) -> bool
    where DefaultAllocator: Alloc<bool, D> {
    v.iter().all(|x| *x)
}

/// Checks that at least one of the vector components is `true`.
pub fn any<D: Dimension>(v: &Vec<bool, D>) -> bool
    where DefaultAllocator: Alloc<bool, D> {
    v.iter().any(|x| *x)
}

/// Componentwise equality comparison.
pub fn equal<N: Number, D: Dimension>(x: &Vec<N, D>, y: &Vec<N, D>) -> Vec<bool, D>
    where DefaultAllocator: Alloc<N, D> {
    x.zip_map(y, |x, y| x == y)
}

/// Componentwise `>` comparison.
pub fn greater_than<N: Number, D: Dimension>(x: &Vec<N, D>, y: &Vec<N, D>) -> Vec<bool, D>
    where DefaultAllocator: Alloc<N, D> {
    x.zip_map(y, |x, y| x > y)
}

/// Componentwise `>=` comparison.
pub fn greater_than_equal<N: Number, D: Dimension>(x: &Vec<N, D>, y: &Vec<N, D>) -> Vec<bool, D>
    where DefaultAllocator: Alloc<N, D> {
    x.zip_map(y, |x, y| x >= y)
}

/// Componentwise `<` comparison.
pub fn less_than<N: Number, D: Dimension>(x: &Vec<N, D>, y: &Vec<N, D>) -> Vec<bool, D>
    where DefaultAllocator: Alloc<N, D> {
    x.zip_map(y, |x, y| x < y)
}

/// Componentwise `>=` comparison.
pub fn less_than_equal<N: Number, D: Dimension>(x: &Vec<N, D>, y: &Vec<N, D>) -> Vec<bool, D>
    where DefaultAllocator: Alloc<N, D> {
    x.zip_map(y, |x, y| x <= y)
}

/// Componentwise not `!`.
pub fn not<D: Dimension>(v: &Vec<bool, D>) -> Vec<bool, D>
    where DefaultAllocator: Alloc<bool, D> {
    v.map(|x| !x)
}

/// Componentwise not-equality `!=`.
pub fn not_equal<N: Number, D: Dimension>(x: &Vec<N, D>, y: &Vec<N, D>) -> Vec<bool, D>
    where DefaultAllocator: Alloc<N, D> {
    x.zip_map(y, |x, y| x != y)
}
