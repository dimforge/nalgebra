use crate::RealNumber;
use crate::aliases::TVec;

/// The squared distance between two points.
///
/// # See also:
///
/// * [`distance()`](crate::distance)
pub fn distance2<T: RealNumber, const D: usize>(p0: &TVec<T, D>, p1: &TVec<T, D>) -> T {
    (p1 - p0).norm_squared()
}

/// The l1-norm of `x - y`.
///
/// # See also:
///
/// * [`l1_norm()`]
/// * [`l2_distance()`]
/// * [`l2_norm()`]
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
/// * [`l1_distance()`]
/// * [`l2_distance()`]
/// * [`l2_norm()`]
pub fn l1_norm<T: RealNumber, const D: usize>(v: &TVec<T, D>) -> T {
    crate::comp_add(&v.abs())
}

/// The l2-norm of `x - y`.
///
/// This is the same value as returned by [`length2()`] and
/// [`magnitude2()`].
///
/// # See also:
///
/// * [`l1_distance()`]
/// * [`l1_norm()`]
/// * [`l2_norm()`]
/// * [`length()`](crate::length)
/// * [`length2()`]
/// * [`magnitude()`](crate::magnitude)
/// * [`magnitude2()`]
pub fn l2_distance<T: RealNumber, const D: usize>(x: &TVec<T, D>, y: &TVec<T, D>) -> T {
    l2_norm(&(y - x))
}

/// The l2-norm of `v`.
///
/// This is also known as the Euclidean norm.
///
/// This is the same value as returned by [`length()`](crate::length) and
/// [`magnitude()`](crate::magnitude).
///
/// # See also:
///
/// * [`l1_distance()`]
/// * [`l1_norm()`]
/// * [`l2_distance()`]
/// * [`length()`](crate::length)
/// * [`length2()`]
/// * [`magnitude()`](crate::magnitude)
/// * [`magnitude2()`]
pub fn l2_norm<T: RealNumber, const D: usize>(x: &TVec<T, D>) -> T {
    x.norm()
}

/// The squared magnitude of `x`.
///
/// A synonym for [`magnitude2()`].
///
/// # See also:
///
/// * [`distance()`](crate::distance)
/// * [`distance2()`]
/// * [`length()`](crate::length)
/// * [`magnitude()`](crate::magnitude)
/// * [`magnitude2()`]
pub fn length2<T: RealNumber, const D: usize>(x: &TVec<T, D>) -> T {
    x.norm_squared()
}

/// The squared magnitude of `x`.
///
/// A wrapper around [`nalgebra::norm_squared`](../nalgebra/fn.norm_squared.html).
///
/// # See also:
///
/// * [`distance()`](crate::distance)
/// * [`distance2()`]
/// * [`length2()`]
/// * [`magnitude()`](crate::magnitude)
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
