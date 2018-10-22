use na::{DefaultAllocator, Real};

use aliases::{TVec, TVec3};
use traits::{Alloc, Dimension, Number};

/// The cross product of two vectors.
pub fn cross<N: Number, D: Dimension>(x: &TVec3<N>, y: &TVec3<N>) -> TVec3<N> {
    x.cross(y)
}

/// The distance between two points.
///
/// # See also:
///
/// * [`distance2`](fn.distance2.html)
pub fn distance<N: Real, D: Dimension>(p0: &TVec<N, D>, p1: &TVec<N, D>) -> N
where DefaultAllocator: Alloc<N, D> {
    (p1 - p0).norm()
}

/// The dot product of two vectors.
pub fn dot<N: Number, D: Dimension>(x: &TVec<N, D>, y: &TVec<N, D>) -> N
where DefaultAllocator: Alloc<N, D> {
    x.dot(y)
}

/// If `dot(nref, i) < 0.0`, return `n`, otherwise, return `-n`.
pub fn faceforward<N: Number, D: Dimension>(
    n: &TVec<N, D>,
    i: &TVec<N, D>,
    nref: &TVec<N, D>,
) -> TVec<N, D>
where
    DefaultAllocator: Alloc<N, D>,
{
    if nref.dot(i) < N::zero() {
        n.clone()
    } else {
        -n.clone()
    }
}

/// The magnitude of a vector.
///
/// A synonym for [`magnitude`](fn.magnitude.html).
///
/// # See also:
///
/// * [`length2`](fn.length2.html)
/// * [`magnitude`](fn.magnitude.html)
/// * [`magnitude2`](fn.magnitude2.html)
pub fn length<N: Real, D: Dimension>(x: &TVec<N, D>) -> N
where DefaultAllocator: Alloc<N, D> {
    x.norm()
}

/// The magnitude of a vector.
///
/// A wrapper around [`nalgebra::norm`](../nalgebra/fn.norm.html).
///
/// # See also:
///
/// * [`length`](fn.length.html)
/// * [`magnitude2`](fn.magnitude2.html)
/// * [`nalgebra::norm`](../nalgebra/fn.norm.html)
pub fn magnitude<N: Real, D: Dimension>(x: &TVec<N, D>) -> N
where DefaultAllocator: Alloc<N, D> {
    x.norm()
}

/// Normalizes a vector.
pub fn normalize<N: Real, D: Dimension>(x: &TVec<N, D>) -> TVec<N, D>
where DefaultAllocator: Alloc<N, D> {
    x.normalize()
}

/// For the incident vector `i` and surface orientation `n`, returns the reflection direction : `result = i - 2.0 * dot(n, i) * n`.
pub fn reflect_vec<N: Number, D: Dimension>(i: &TVec<N, D>, n: &TVec<N, D>) -> TVec<N, D>
where DefaultAllocator: Alloc<N, D> {
    let _2 = N::one() + N::one();
    i - n * (n.dot(i) * _2)
}

/// For the incident vector `i` and surface normal `n`, and the ratio of indices of refraction `eta`, return the refraction vector.
pub fn refract_vec<N: Real, D: Dimension>(i: &TVec<N, D>, n: &TVec<N, D>, eta: N) -> TVec<N, D>
where DefaultAllocator: Alloc<N, D> {
    let ni = n.dot(i);
    let k = N::one() - eta * eta * (N::one() - ni * ni);

    if k < N::zero() {
        TVec::<_, D>::zeros()
    } else {
        i * eta - n * (eta * dot(n, i) + k.sqrt())
    }
}
