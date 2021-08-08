use crate::RealNumber;

use crate::aliases::{TVec, TVec2, TVec3};
use crate::traits::Number;

/// Returns `true` if two vectors are collinear (up to an epsilon).
///
/// # See also:
///
/// * [`are_collinear2d`](fn.are_collinear2d.html)
pub fn are_collinear<T: Number>(v0: &TVec3<T>, v1: &TVec3<T>, epsilon: T) -> bool {
    is_null(&v0.cross(v1), epsilon)
}

/// Returns `true` if two 2D vectors are collinear (up to an epsilon).
///
/// # See also:
///
/// * [`are_collinear`](fn.are_collinear.html)
pub fn are_collinear2d<T: Number>(v0: &TVec2<T>, v1: &TVec2<T>, epsilon: T) -> bool {
    abs_diff_eq!(v0.perp(v1), T::zero(), epsilon = epsilon)
}

/// Returns `true` if two vectors are orthogonal (up to an epsilon).
pub fn are_orthogonal<T: Number, const D: usize>(
    v0: &TVec<T, D>,
    v1: &TVec<T, D>,
    epsilon: T,
) -> bool {
    abs_diff_eq!(v0.dot(v1), T::zero(), epsilon = epsilon)
}

//pub fn are_orthonormal<T: Number, const D: usize>(v0: &TVec<T, D>, v1: &TVec<T, D>, epsilon: T) -> bool {
//    unimplemented!()
//}

/// Returns `true` if all the components of `v` are zero (up to an epsilon).
pub fn is_comp_null<T: Number, const D: usize>(v: &TVec<T, D>, epsilon: T) -> TVec<bool, D> {
    v.map(|x| abs_diff_eq!(x, T::zero(), epsilon = epsilon))
}

/// Returns `true` if `v` has a magnitude of 1 (up to an epsilon).
pub fn is_normalized<T: RealNumber, const D: usize>(v: &TVec<T, D>, epsilon: T) -> bool {
    abs_diff_eq!(v.norm_squared(), T::one(), epsilon = epsilon * epsilon)
}

/// Returns `true` if `v` is zero (up to an epsilon).
pub fn is_null<T: Number, const D: usize>(v: &TVec<T, D>, epsilon: T) -> bool {
    abs_diff_eq!(*v, TVec::<T, D>::zeros(), epsilon = epsilon)
}
