use na::{Real, DefaultAllocator};

use traits::{Number, Alloc, Dimension};
use aliases::{Vec, TVec3};

/// The cross product of two vectors.
pub fn cross<N: Number, D: Dimension>(x: &TVec3<N>, y: &TVec3<N>) -> TVec3<N> {
    x.cross(y)
}

/// The distance between two points.
pub fn distance<N: Real, D: Dimension>(p0: &Vec<N, D>, p1: &Vec<N, D>) -> N
    where DefaultAllocator: Alloc<N, D> {
   (p1 - p0).norm()
}

/// The dot product of two vectors.
pub fn dot<N: Number, D: Dimension>(x: &Vec<N, D>, y: &Vec<N, D>) -> N
    where DefaultAllocator: Alloc<N, D> {
    x.dot(y)
}

/// If `dot(nref, i) < 0.0`, return `n`, otherwise, return `-n`.
pub fn faceforward<N: Number, D: Dimension>(n: &Vec<N, D>, i: &Vec<N, D>, nref: &Vec<N, D>) -> Vec<N, D>
    where DefaultAllocator: Alloc<N, D> {
    if nref.dot(i) < N::zero() {
        n.clone()
    } else {
        -n.clone()
    }
}

/// The magnitude of a vector.
pub fn length<N: Real, D: Dimension>(x: &Vec<N, D>) -> N
    where DefaultAllocator: Alloc<N, D> {
    x.norm()
}

/// The magnitude of a vector.
pub fn magnitude<N: Real, D: Dimension>(x: &Vec<N, D>) -> N
    where DefaultAllocator: Alloc<N, D> {
    x.norm()
}

/// Normalizes a vector.
pub fn normalize<N: Real, D: Dimension>(x: &Vec<N, D>) -> Vec<N, D>
    where DefaultAllocator: Alloc<N, D> {
    x.normalize()
}

/// For the incident vector `i` and surface orientation `n`, returns the reflection direction : `result = i - 2.0 * dot(n, i) * n`.
pub fn reflect_vec<N: Number, D: Dimension>(i: &Vec<N, D>, n: &Vec<N, D>) -> Vec<N, D>
    where DefaultAllocator: Alloc<N, D> {
    let _2 = N::one() + N::one();
    i - n * (n.dot(i) * _2)
}

/// For the incident vector `i` and surface normal `n`, and the ratio of indices of refraction `eta`, return the refraction vector.
pub fn refract_vec<N: Real, D: Dimension>(i: &Vec<N, D>, n: &Vec<N, D>, eta: N) -> Vec<N, D>
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