use crate::RealNumber;

use crate::aliases::{TVec, TVec3};
use crate::traits::Number;

/// The cross product of two vectors.
pub fn cross<T: Number>(x: &TVec3<T>, y: &TVec3<T>) -> TVec3<T> {
    x.cross(y)
}

/// The distance between two points.
///
/// # See also:
///
/// * [`distance2()`](crate::distance2)
pub fn distance<T: RealNumber, const D: usize>(p0: &TVec<T, D>, p1: &TVec<T, D>) -> T {
    (p1 - p0).norm()
}

/// The dot product of two vectors.
pub fn dot<T: Number, const D: usize>(x: &TVec<T, D>, y: &TVec<T, D>) -> T {
    x.dot(y)
}

/// If `dot(nref, i) < 0.0`, return `n`, otherwise, return `-n`.
pub fn faceforward<T: Number, const D: usize>(
    n: &TVec<T, D>,
    i: &TVec<T, D>,
    nref: &TVec<T, D>,
) -> TVec<T, D> {
    if nref.dot(i) < T::zero() {
        n.clone()
    } else {
        -n.clone()
    }
}

/// The magnitude of a vector.
///
/// A synonym for [`magnitude()`].
///
/// # See also:
///
/// * [`length2()`](crate::length2)
/// * [`magnitude()`]
/// * [`magnitude2()`](crate::magnitude2)
pub fn length<T: RealNumber, const D: usize>(x: &TVec<T, D>) -> T {
    x.norm()
}

/// The magnitude of a vector.
///
/// A wrapper around [`nalgebra::norm`](../nalgebra/fn.norm.html).
///
/// # See also:
///
/// * [`length()`]
/// * [`magnitude2()`](crate::magnitude2)
/// * [`nalgebra::norm`](../nalgebra/fn.norm.html)
pub fn magnitude<T: RealNumber, const D: usize>(x: &TVec<T, D>) -> T {
    x.norm()
}

/// Normalizes a vector.
pub fn normalize<T: RealNumber, const D: usize>(x: &TVec<T, D>) -> TVec<T, D> {
    x.normalize()
}

/// For the incident vector `i` and surface orientation `n`, returns the reflection direction : `result = i - 2.0 * dot(n, i) * n`.
pub fn reflect_vec<T: Number, const D: usize>(i: &TVec<T, D>, n: &TVec<T, D>) -> TVec<T, D> {
    let _2 = T::one() + T::one();
    i - n * (n.dot(i) * _2)
}

/// For the incident vector `i` and surface normal `n`, and the ratio of indices of refraction `eta`, return the refraction vector.
pub fn refract_vec<T: RealNumber, const D: usize>(
    i: &TVec<T, D>,
    n: &TVec<T, D>,
    eta: T,
) -> TVec<T, D> {
    let ni = n.dot(i);
    let k = T::one() - eta * eta * (T::one() - ni * ni);

    if k < T::zero() {
        TVec::<_, D>::zeros()
    } else {
        i * eta - n * (eta * dot(n, i) + k.sqrt())
    }
}
