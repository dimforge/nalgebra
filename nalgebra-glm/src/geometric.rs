use na::{Scalar, Real, DimName, U3, DefaultAllocator};

use traits::{Number, Alloc};
use aliases::Vec;

pub fn cross<N: Number, D: DimName>(x: &Vec<N, U3>, y: &Vec<N, U3>) -> Vec<N, U3> {
    x.cross(y)
}

pub fn distance<N: Real, D: DimName>(p0: &Vec<N, D>, p1: &Vec<N, D>) -> N
    where DefaultAllocator: Alloc<N, D> {
   (p1 - p0).norm()
}

pub fn dot<N: Number, D: DimName>(x: &Vec<N, D>, y: &Vec<N, D>) -> N
    where DefaultAllocator: Alloc<N, D> {
    x.dot(y)
}

pub fn faceforward<N: Scalar, D: DimName>(N: &Vec<N, D>, I: &Vec<N, D>, Nref: &Vec<N, D>) -> Vec<N, D>
    where DefaultAllocator: Alloc<N, D> {
    unimplemented!()
}

pub fn length<N: Real, D: DimName>(x: &Vec<N, D>) -> N
    where DefaultAllocator: Alloc<N, D> {
    x.norm()
}

pub fn normalize<N: Real, D: DimName>(x: &Vec<N, D>) -> Vec<N, D>
    where DefaultAllocator: Alloc<N, D> {
    x.normalize()
}

pub fn reflect<N: Scalar, D: DimName>(I: &Vec<N, D>, N: &Vec<N, D>) -> Vec<N, D>
    where DefaultAllocator: Alloc<N, D> {
    unimplemented!()
}

pub fn  refract<N: Scalar, D: DimName>(I: &Vec<N, D>, N: &Vec<N, D>, eta: N) -> Vec<N, D>
    where DefaultAllocator: Alloc<N, D> {
    unimplemented!()
}