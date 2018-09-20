use na::{Real, DefaultAllocator, U2, U3};

use traits::{Number, Dimension, Alloc};
use aliases::Vec;

pub fn are_collinear<N: Number>(v0: &Vec<N, U3>, v1: &Vec<N, U3>, epsilon: N) -> bool {
    is_null(&v0.cross(v1), epsilon)
}

pub fn are_collinear2<N: Number>(v0: &Vec<N, U2>, v1: &Vec<N, U2>, epsilon: N) -> bool {
    abs_diff_eq!(v0.perp(v1), N::zero(), epsilon = epsilon)
}

pub fn are_orthogonal<N: Number, D: Dimension>(v0: &Vec<N, D>, v1: &Vec<N, D>, epsilon: N) -> bool
    where DefaultAllocator: Alloc<N, D> {
    abs_diff_eq!(v0.dot(v1), N::zero(), epsilon = epsilon)
}

pub fn are_orthonormal<N: Number, D: Dimension>(v0: &Vec<N, D>, v1: &Vec<N, D>, epsilon: N) -> bool
    where DefaultAllocator: Alloc<N, D> {
    unimplemented!()
}

pub fn is_comp_null<N: Number, D: Dimension>(v: &Vec<N, D>, epsilon: N) -> Vec<bool, D>
    where DefaultAllocator: Alloc<N, D> {
    v.map(|x| abs_diff_eq!(x, N::zero(), epsilon = epsilon))
}

pub fn is_normalized<N: Real, D: Dimension>(v: &Vec<N, D>, epsilon: N) -> bool
    where DefaultAllocator: Alloc<N, D> {
    abs_diff_eq!(v.norm_squared(), N::one(), epsilon = epsilon * epsilon)
}

pub fn is_null<N: Number, D: Dimension>(v: &Vec<N, D>, epsilon: N) -> bool
    where DefaultAllocator: Alloc<N, D> {
    abs_diff_eq!(*v, Vec::<N, D>::zeros(), epsilon = epsilon)
}
