use na::{self, Scalar, Real, U3, DefaultAllocator};

use traits::{Number, Alloc, Dimension};
use aliases::Vec;

pub fn cross<N: Number, D: Dimension>(x: &Vec<N, U3>, y: &Vec<N, U3>) -> Vec<N, U3> {
    x.cross(y)
}

pub fn distance<N: Real, D: Dimension>(p0: &Vec<N, D>, p1: &Vec<N, D>) -> N
    where DefaultAllocator: Alloc<N, D> {
   (p1 - p0).norm()
}

pub fn dot<N: Number, D: Dimension>(x: &Vec<N, D>, y: &Vec<N, D>) -> N
    where DefaultAllocator: Alloc<N, D> {
    x.dot(y)
}

pub fn faceforward<N: Number, D: Dimension>(n: &Vec<N, D>, i: &Vec<N, D>, nref: &Vec<N, D>) -> Vec<N, D>
    where DefaultAllocator: Alloc<N, D> {
    if nref.dot(i) < N::zero() {
        n.clone()
    } else {
        -n.clone()
    }
}

pub fn length<N: Real, D: Dimension>(x: &Vec<N, D>) -> N
    where DefaultAllocator: Alloc<N, D> {
    x.norm()
}

pub fn normalize<N: Real, D: Dimension>(x: &Vec<N, D>) -> Vec<N, D>
    where DefaultAllocator: Alloc<N, D> {
    x.normalize()
}

pub fn reflect<N: Number, D: Dimension>(i: &Vec<N, D>, n: &Vec<N, D>) -> Vec<N, D>
    where DefaultAllocator: Alloc<N, D> {
    let _2 = N::one() + N::one();
    i - n * (n.dot(i) * _2)
}

pub fn refract<N: Real, D: Dimension>(i: &Vec<N, D>, n: &Vec<N, D>, eta: N) -> Vec<N, D>
    where DefaultAllocator: Alloc<N, D> {

    let ni = n.dot(i);
    let k = N::one() - eta * eta * (N::one() - ni * ni);

    if k < N::zero() {
        Vec::<_, D>::zeros()
    }
    else {
        i * eta - n * (eta * dot(n, i) + k.sqrt())
    }
}