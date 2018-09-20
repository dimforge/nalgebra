use na::{self, Real, DimName, DefaultAllocator};

use aliases::Vec;
use traits::{Number, Alloc};


pub fn all<D: DimName>(v: &Vec<bool, D>) -> bool
    where DefaultAllocator: Alloc<bool, D> {
    v.iter().all(|x| *x)
}

pub fn any<D: DimName>(v: &Vec<bool, D>) -> bool
    where DefaultAllocator: Alloc<bool, D> {
    v.iter().any(|x| *x)
}

pub fn equal<N: Number, D: DimName>(x: &Vec<N, D>, y: &Vec<N, D>) -> Vec<bool, D>
    where DefaultAllocator: Alloc<N, D> {
    x.zip_map(y, |x, y| x == y)
}

pub fn greaterThan<N: Number, D: DimName>(x: &Vec<N, D>, y: &Vec<N, D>) -> Vec<bool, D>
    where DefaultAllocator: Alloc<N, D> {
    x.zip_map(y, |x, y| x > y)
}

pub fn greaterThanEqual<N: Number, D: DimName>(x: &Vec<N, D>, y: &Vec<N, D>) -> Vec<bool, D>
    where DefaultAllocator: Alloc<N, D> {
    x.zip_map(y, |x, y| x >= y)
}

pub fn lessThan<N: Number, D: DimName>(x: &Vec<N, D>, y: &Vec<N, D>) -> Vec<bool, D>
    where DefaultAllocator: Alloc<N, D> {
    x.zip_map(y, |x, y| x < y)
}

pub fn lessThanEqual<N: Number, D: DimName>(x: &Vec<N, D>, y: &Vec<N, D>) -> Vec<bool, D>
    where DefaultAllocator: Alloc<N, D> {
    x.zip_map(y, |x, y| x <= y)
}

pub fn not<D: DimName>(v: &Vec<bool, D>) -> Vec<bool, D>
    where DefaultAllocator: Alloc<bool, D> {
    v.map(|x| !x)
}

pub fn notEqual<N: Number, D: DimName>(x: &Vec<N, D>, y: &Vec<N, D>) -> Vec<bool, D>
    where DefaultAllocator: Alloc<N, D> {
    x.zip_map(y, |x, y| x != y)
}
