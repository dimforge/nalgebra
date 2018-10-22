use na::{DefaultAllocator, Real};

use aliases::TVec;
use traits::{Alloc, Dimension};

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
///
/// # See also:
///
/// * [`l1_norm`](fn.l1_norm.html)
/// * [`l2_distance`](fn.l2_distance.html)
/// * [`l2_norm`](fn.l2_norm.html)
pub fn l1_distance<N: Real, D: Dimension>(x: &TVec<N, D>, y: &TVec<N, D>) -> N
where DefaultAllocator: Alloc<N, D> {
    l1_norm(&(y - x))
}

/// The l1-norm of `v`.
///
/// This is also known as the "Manhattan distance" or "taxicab distance" and
/// corresponds to the sum of the absolute values of the components of `v`.
///
/// # See also:
///
/// * [`l1_distance`](fn.l1_distance.html)
/// * [`l2_distance`](fn.l2_distance.html)
/// * [`l2_norm`](fn.l2_norm.html)
pub fn l1_norm<N: Real, D: Dimension>(v: &TVec<N, D>) -> N
where DefaultAllocator: Alloc<N, D> {
    ::comp_add(&v.abs())
}

/// The l2-norm of `x - y`.
///
/// This is the same value as returned by [`length2`](fn.length2.html) and
/// [`magnitude2`](fn.magnitude2.html).
///
/// # See also:
///
/// * [`l1_distance`](fn.l1_distance.html)
/// * [`l1_norm`](fn.l1_norm.html)
/// * [`l2_norm`](fn.l2_norm.html)
/// * [`length`](fn.length.html)
/// * [`length2`](fn.length2.html)
/// * [`magnitude`](fn.magnitude.html)
/// * [`magnitude2`](fn.magnitude2.html)
pub fn l2_distance<N: Real, D: Dimension>(x: &TVec<N, D>, y: &TVec<N, D>) -> N
where DefaultAllocator: Alloc<N, D> {
    l2_norm(&(y - x))
}

/// The l2-norm of `v`.
///
/// This is also known as the Euclidean norm.
///
/// This is the same value as returned by [`length`](fn.length.html) and
/// [`magnitude`](fn.magnitude.html).
///
/// # See also:
///
/// * [`l1_distance`](fn.l1_distance.html)
/// * [`l1_norm`](fn.l1_norm.html)
/// * [`l2_distance`](fn.l2_distance.html)
/// * [`length`](fn.length.html)
/// * [`length2`](fn.length2.html)
/// * [`magnitude`](fn.magnitude.html)
/// * [`magnitude2`](fn.magnitude2.html)
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
