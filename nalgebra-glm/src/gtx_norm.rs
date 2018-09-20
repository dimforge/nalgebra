use na::{Real, DefaultAllocator};

use traits::{Alloc, Dimension};
use aliases::Vec;

pub fn distance2<N: Real, D: Dimension>(p0: &Vec<N, D>, p1: &Vec<N, D>) -> N
    where DefaultAllocator: Alloc<N, D> {
    (p1 - p0).norm_squared()
}

pub fn l1_distance<N: Real, D: Dimension>(x: &Vec<N, D>, y: &Vec<N, D>) -> N
    where DefaultAllocator: Alloc<N, D> {
    l1_norm(&(x - y))
}

pub fn l1_norm<N: Real, D: Dimension>(v: &Vec<N, D>) -> N
    where DefaultAllocator: Alloc<N, D> {
    ::comp_add(&v.abs())
}

pub fn l2_distance<N: Real, D: Dimension>(x: &Vec<N, D>, y: &Vec<N, D>) -> N
    where DefaultAllocator: Alloc<N, D> {
    l2_norm(&(y - x))
}

pub fn l2_norm<N: Real, D: Dimension>(x: &Vec<N, D>) -> N
    where DefaultAllocator: Alloc<N, D> {
    x.norm()
}

pub fn length2<N: Real, D: Dimension>(x: &Vec<N, D>) -> N
    where DefaultAllocator: Alloc<N, D> {
    x.norm_squared()
}

//pub fn lxNorm<N: Real, D: Dimension>(x: &Vec<N, D>, y: &Vec<N, D>, unsigned int Depth) -> N
//    where DefaultAllocator: Alloc<N, D> {
//    unimplemented!()
//}
//
//pub fn lxNorm<N: Real, D: Dimension>(x: &Vec<N, D>, unsigned int Depth) -> N
//    where DefaultAllocator: Alloc<N, D> {
//    unimplemented!()
//}
