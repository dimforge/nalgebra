use na::{Real, DefaultAllocator};

use traits::{Alloc, Dimension};
use aliases::TVec;

/// The squared distance between two points.
///
/// # See also:
///
/// * [`distance`](fn.distance.html)
pub fn distance2<N: Real, D: Dimension>(p0: &TVec<N, D>, p1: &TVec<N, D>) -> N
    where DefaultAllocator: Alloc<N, D> {
    (p1 - p0).norm_squared()
}

/// The l1-norm of `x - y`.
pub fn l1_distance<N: Real, D: Dimension>(x: &TVec<N, D>, y: &TVec<N, D>) -> N
    where DefaultAllocator: Alloc<N, D> {
    l1_norm(&(y - x))
}

/// The l1-norm of `v`.
pub fn l1_norm<N: Real, D: Dimension>(v: &TVec<N, D>) -> N
    where DefaultAllocator: Alloc<N, D> {
    ::comp_add(&v.abs())
}

/// The l2-norm of `x - y`.
pub fn l2_distance<N: Real, D: Dimension>(x: &TVec<N, D>, y: &TVec<N, D>) -> N
    where DefaultAllocator: Alloc<N, D> {
    l2_norm(&(y - x))
}

/// The l2-norm of `v`.
pub fn l2_norm<N: Real, D: Dimension>(x: &TVec<N, D>) -> N
    where DefaultAllocator: Alloc<N, D> {
    x.norm()
}

/// The squared magnitude of `x`.
///
/// A synonym for [`magnitude2`](fn.magnitude2.html).
///
/// # See also:
///
/// * [`distance`](fn.distance.html)
/// * [`distance2`](fn.distance2.html)
/// * [`length`](fn.length.html)
/// * [`magnitude`](fn.magnitude.html)
/// * [`magnitude2`](fn.magnitude2.html)
pub fn length2<N: Real, D: Dimension>(x: &TVec<N, D>) -> N
    where DefaultAllocator: Alloc<N, D> {
    x.norm_squared()
}

/// The squared magnitude of `x`.
///
/// A wrapper around [`nalgebra::norm_squared`](../nalgebra/fn.norm_squared.html).
///
/// # See also:
///
/// * [`distance`](fn.distance.html)
/// * [`distance2`](fn.distance2.html)
/// * [`length2`](fn.length2.html)
/// * [`magnitude`](fn.magnitude.html)
/// * [`nalgebra::norm_squared`](../nalgebra/fn.norm_squared.html)
pub fn magnitude2<N: Real, D: Dimension>(x: &TVec<N, D>) -> N
    where DefaultAllocator: Alloc<N, D> {
    x.norm_squared()
}

//pub fn lxNorm<N: Real, D: Dimension>(x: &TVec<N, D>, y: &TVec<N, D>, unsigned int Depth) -> N
//    where DefaultAllocator: Alloc<N, D> {
//    unimplemented!()
//}
//
//pub fn lxNorm<N: Real, D: Dimension>(x: &TVec<N, D>, unsigned int Depth) -> N
//    where DefaultAllocator: Alloc<N, D> {
//    unimplemented!()
//}
