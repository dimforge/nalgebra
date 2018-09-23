use na::{DefaultAllocator};

use aliases::TVec;
use traits::{Number, Alloc, Dimension};

/// Checks that all the vector components are `true`.
pub fn all<D: Dimension>(v: &TVec<bool, D>) -> bool
    where DefaultAllocator: Alloc<bool, D> {
    v.iter().all(|x| *x)
}

/// Checks that at least one of the vector components is `true`.
pub fn any<D: Dimension>(v: &TVec<bool, D>) -> bool
    where DefaultAllocator: Alloc<bool, D> {
    v.iter().any(|x| *x)
}

/// Component-wise equality comparison.
pub fn equal<N: Number, D: Dimension>(x: &TVec<N, D>, y: &TVec<N, D>) -> TVec<bool, D>
    where DefaultAllocator: Alloc<N, D> {
    x.zip_map(y, |x, y| x == y)
}

/// Component-wise `>` comparison.
pub fn greater_than<N: Number, D: Dimension>(x: &TVec<N, D>, y: &TVec<N, D>) -> TVec<bool, D>
    where DefaultAllocator: Alloc<N, D> {
    x.zip_map(y, |x, y| x > y)
}

/// Component-wise `>=` comparison.
pub fn greater_than_equal<N: Number, D: Dimension>(x: &TVec<N, D>, y: &TVec<N, D>) -> TVec<bool, D>
    where DefaultAllocator: Alloc<N, D> {
    x.zip_map(y, |x, y| x >= y)
}

/// Component-wise `<` comparison.
pub fn less_than<N: Number, D: Dimension>(x: &TVec<N, D>, y: &TVec<N, D>) -> TVec<bool, D>
    where DefaultAllocator: Alloc<N, D> {
    x.zip_map(y, |x, y| x < y)
}

/// Component-wise `>=` comparison.
pub fn less_than_equal<N: Number, D: Dimension>(x: &TVec<N, D>, y: &TVec<N, D>) -> TVec<bool, D>
    where DefaultAllocator: Alloc<N, D> {
    x.zip_map(y, |x, y| x <= y)
}

/// Component-wise not `!`.
pub fn not<D: Dimension>(v: &TVec<bool, D>) -> TVec<bool, D>
    where DefaultAllocator: Alloc<bool, D> {
    v.map(|x| !x)
}

/// Component-wise not-equality `!=`.
pub fn not_equal<N: Number, D: Dimension>(x: &TVec<N, D>, y: &TVec<N, D>) -> TVec<bool, D>
    where DefaultAllocator: Alloc<N, D> {
    x.zip_map(y, |x, y| x != y)
}
