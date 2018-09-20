use na::{self, Real, DefaultAllocator};

use aliases::Vec;
use traits::{Number, Alloc, Dimension};


pub fn all<D: Dimension>(v: &Vec<bool, D>) -> bool
    where DefaultAllocator: Alloc<bool, D> {
    v.iter().all(|x| *x)
}

pub fn any<D: Dimension>(v: &Vec<bool, D>) -> bool
    where DefaultAllocator: Alloc<bool, D> {
    v.iter().any(|x| *x)
}

pub fn equal<N: Number, D: Dimension>(x: &Vec<N, D>, y: &Vec<N, D>) -> Vec<bool, D>
    where DefaultAllocator: Alloc<N, D> {
    x.zip_map(y, |x, y| x == y)
}

pub fn greater_than<N: Number, D: Dimension>(x: &Vec<N, D>, y: &Vec<N, D>) -> Vec<bool, D>
    where DefaultAllocator: Alloc<N, D> {
    x.zip_map(y, |x, y| x > y)
}

pub fn greater_than_equal<N: Number, D: Dimension>(x: &Vec<N, D>, y: &Vec<N, D>) -> Vec<bool, D>
    where DefaultAllocator: Alloc<N, D> {
    x.zip_map(y, |x, y| x >= y)
}

pub fn less_than<N: Number, D: Dimension>(x: &Vec<N, D>, y: &Vec<N, D>) -> Vec<bool, D>
    where DefaultAllocator: Alloc<N, D> {
    x.zip_map(y, |x, y| x < y)
}

pub fn less_than_equal<N: Number, D: Dimension>(x: &Vec<N, D>, y: &Vec<N, D>) -> Vec<bool, D>
    where DefaultAllocator: Alloc<N, D> {
    x.zip_map(y, |x, y| x <= y)
}

pub fn not<D: Dimension>(v: &Vec<bool, D>) -> Vec<bool, D>
    where DefaultAllocator: Alloc<bool, D> {
    v.map(|x| !x)
}

pub fn not_equal<N: Number, D: Dimension>(x: &Vec<N, D>, y: &Vec<N, D>) -> Vec<bool, D>
    where DefaultAllocator: Alloc<N, D> {
    x.zip_map(y, |x, y| x != y)
}
