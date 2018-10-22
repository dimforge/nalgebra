use na::{DefaultAllocator, Real};

use aliases::{TVec, TVec2, TVec3};
use traits::{Alloc, Dimension, Number};

/// Returns `true` if two vectors are collinear (up to an epsilon).
///
/// # See also:
///
/// * [`are_collinear2d`](fn.are_collinear2d.html)
pub fn are_collinear<N: Number>(v0: &TVec3<N>, v1: &TVec3<N>, epsilon: N) -> bool {
    is_null(&v0.cross(v1), epsilon)
}

/// Returns `true` if two 2D vectors are collinear (up to an epsilon).
///
/// # See also:
///
/// * [`are_collinear`](fn.are_collinear.html)
pub fn are_collinear2d<N: Number>(v0: &TVec2<N>, v1: &TVec2<N>, epsilon: N) -> bool {
    abs_diff_eq!(v0.perp(v1), N::zero(), epsilon = epsilon)
}

/// Returns `true` if two vectors are orthogonal (up to an epsilon).
pub fn are_orthogonal<N: Number, D: Dimension>(
    v0: &TVec<N, D>,
    v1: &TVec<N, D>,
    epsilon: N,
) -> bool
where
    DefaultAllocator: Alloc<N, D>,
{
    abs_diff_eq!(v0.dot(v1), N::zero(), epsilon = epsilon)
}

//pub fn are_orthonormal<N: Number, D: Dimension>(v0: &TVec<N, D>, v1: &TVec<N, D>, epsilon: N) -> bool
//    where DefaultAllocator: Alloc<N, D> {
//    unimplemented!()
//}

/// Returns `true` if all the components of `v` are zero (up to an epsilon).
pub fn is_comp_null<N: Number, D: Dimension>(v: &TVec<N, D>, epsilon: N) -> TVec<bool, D>
where DefaultAllocator: Alloc<N, D> {
    v.map(|x| abs_diff_eq!(x, N::zero(), epsilon = epsilon))
}

/// Returns `true` if `v` has a magnitude of 1 (up to an epsilon).
pub fn is_normalized<N: Real, D: Dimension>(v: &TVec<N, D>, epsilon: N) -> bool
where DefaultAllocator: Alloc<N, D> {
    abs_diff_eq!(v.norm_squared(), N::one(), epsilon = epsilon * epsilon)
}

/// Returns `true` if `v` is zero (up to an epsilon).
pub fn is_null<N: Number, D: Dimension>(v: &TVec<N, D>, epsilon: N) -> bool
where DefaultAllocator: Alloc<N, D> {
    abs_diff_eq!(*v, TVec::<N, D>::zeros(), epsilon = epsilon)
}
