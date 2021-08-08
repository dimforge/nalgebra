use crate::aliases::TVec;
use crate::RealNumber;

/// The squared distance between two points.
///
/// # See also:
///
/// * [`distance`](fn.distance.html)
pub fn distance2<T: RealNumber, const D: usize>(p0: &TVec<T, D>, p1: &TVec<T, D>) -> T {
    (p1 - p0).norm_squared()
}

/// The l1-norm of `x - y`.
///
/// # See also:
///
/// * [`l1_norm`](fn.l1_norm.html)
/// * [`l2_distance`](fn.l2_distance.html)
/// * [`l2_norm`](fn.l2_norm.html)
pub fn l1_distance<T: RealNumber, const D: usize>(x: &TVec<T, D>, y: &TVec<T, D>) -> T {
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
pub fn l1_norm<T: RealNumber, const D: usize>(v: &TVec<T, D>) -> T {
    crate::comp_add(&v.abs())
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
pub fn l2_distance<T: RealNumber, const D: usize>(x: &TVec<T, D>, y: &TVec<T, D>) -> T {
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
pub fn l2_norm<T: RealNumber, const D: usize>(x: &TVec<T, D>) -> T {
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
pub fn length2<T: RealNumber, const D: usize>(x: &TVec<T, D>) -> T {
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
pub fn magnitude2<T: RealNumber, const D: usize>(x: &TVec<T, D>) -> T {
    x.norm_squared()
}

//pub fn lxNorm<T: RealNumber, const D: usize>(x: &TVec<T, D>, y: &TVec<T, D>, unsigned int Depth) -> T {
//    unimplemented!()
//}
//
//pub fn lxNorm<T: RealNumber, const D: usize>(x: &TVec<T, D>, unsigned int Depth) -> T  {
//    unimplemented!()
//}
