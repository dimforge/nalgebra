use std::cast;
use std::num::{Zero, One, Float, Bounded};
use std::slice::{Items, MutItems};
use std::iter::{Iterator, FromIterator};
use traits::operations::ApproxEq;
use traits::structure::{Iterable, IterableMut, Indexable, Basis, Dim};
use traits::geometry::{Translation, Dot, Norm};
use structs::vec;

impl<N> vec::Vec0<N> {
    /// Creates a new vector.
    #[inline]
    pub fn new() -> vec::Vec0<N> {
        vec::Vec0
    }
}

impl<N> Indexable<uint, N> for vec::Vec0<N> {
    #[inline]
    fn at(&self, _: uint) -> N {
        fail!("Cannot index a Vec0.")
    }

    #[inline]
    fn set(&mut self, _: uint, _: N) {
    }

    #[inline]
    fn swap(&mut self, _: uint, _: uint) {
    }

    #[inline]
    unsafe fn unsafe_at(&self, _: uint) -> N {
        fail!("Cannot index a Vec0.")
    }

    #[inline]
    unsafe fn unsafe_set(&mut self, _: uint, _: N) {
    }
}

impl<N> vec::Vec0<N> {
    /// Creates a new vector. The parameter is not taken in account.
    #[inline]
    pub fn new_repeat(_: N) -> vec::Vec0<N> {
        vec::Vec0
    }
}

impl<N> Iterable<N> for vec::Vec0<N> {
    #[inline]
    fn iter<'l>(&'l self) -> Items<'l, N> {
        unsafe { cast::transmute::<&'l vec::Vec0<N>, &'l [N, ..0]>(self).iter() }
    }
}

impl<N> IterableMut<N> for vec::Vec0<N> {
    #[inline]
    fn mut_iter<'l>(&'l mut self) -> MutItems<'l, N> {
        unsafe { cast::transmute::<&'l mut vec::Vec0<N>, &'l mut [N, ..0]>(self).mut_iter() }
    }
}

impl<N> Dim for vec::Vec0<N> {
    #[inline]
    fn dim(_: Option<vec::Vec0<N>>) -> uint {
        0
    }
}

impl<N> Basis for vec::Vec0<N> {
    #[inline(always)]
    fn canonical_basis(_: |vec::Vec0<N>| -> bool) { }

    #[inline(always)]
    fn orthonormal_subspace_basis(_: &vec::Vec0<N>, _: |vec::Vec0<N>| -> bool) { }
}

impl<N, T> Add<T, vec::Vec0<N>> for vec::Vec0<N> {
    #[inline]
    fn add(&self, _: &T) -> vec::Vec0<N> {
        vec::Vec0
    }
}

impl<N, T> Sub<T, vec::Vec0<N>> for vec::Vec0<N> {
    #[inline]
    fn sub(&self, _: &T) -> vec::Vec0<N> { 
        vec::Vec0
    }
}

impl<N: Neg<N>> Neg<vec::Vec0<N>> for vec::Vec0<N> {
    #[inline]
    fn neg(&self) -> vec::Vec0<N> {
        vec::Vec0
    }
}

impl<N: Num> Dot<N> for vec::Vec0<N> {
    #[inline]
    fn dot(_: &vec::Vec0<N>, _: &vec::Vec0<N>) -> N {
        Zero::zero()
    }

    #[inline]
    fn sub_dot(_: &vec::Vec0<N>, _: &vec::Vec0<N>, _: &vec::Vec0<N>) -> N {
        Zero::zero()
    } 
}

impl<N, T> Mul<T, vec::Vec0<N>> for vec::Vec0<N> {
    #[inline]
    fn mul(&self, _: &T) -> vec::Vec0<N> {
        vec::Vec0
    }
}

impl<N, T> Div<T, vec::Vec0<N>> for vec::Vec0<N> {
    #[inline]
    fn div(&self, _: &T) -> vec::Vec0<N> {
        vec::Vec0
    }
}

impl<N: Clone + Add<N, N> + Neg<N>> Translation<vec::Vec0<N>> for vec::Vec0<N> {
    #[inline]
    fn translation(&self) -> vec::Vec0<N> {
        self.clone()
    }

    #[inline]
    fn inv_translation(&self) -> vec::Vec0<N> {
        -self
    }

    #[inline]
    fn append_translation(&mut self, t: &vec::Vec0<N>) {
        *self = *t + *self;
    }

    #[inline]
    fn append_translation_cpy(vec: &vec::Vec0<N>, t: &vec::Vec0<N>) -> vec::Vec0<N> {
        *t + vec
    }

    #[inline]
    fn prepend_translation(&mut self, t: &vec::Vec0<N>) {
        *self = *self + *t;
    }

    #[inline]
    fn prepend_translation_cpy(vec: &vec::Vec0<N>, t: &vec::Vec0<N>) -> vec::Vec0<N> {
        vec + *t
    }

    #[inline]
    fn set_translation(&mut self, _: vec::Vec0<N>) {
    }
}

impl<N: Num + Float> Norm<N> for vec::Vec0<N> {
    #[inline]
    fn sqnorm(_: &vec::Vec0<N>) -> N {
        Zero::zero()
    }

    #[inline]
    fn norm(_: &vec::Vec0<N>) -> N {
        Zero::zero()
    }

    #[inline]
    fn normalize_cpy(_: &vec::Vec0<N>) -> vec::Vec0<N> {
        Zero::zero()
    }

    #[inline]
    fn normalize(&mut self) -> N {
        Zero::zero()
    }
}

impl<N: ApproxEq<N>> ApproxEq<N> for vec::Vec0<N> {
    #[inline]
    fn approx_epsilon(_: Option<vec::Vec0<N>>) -> N {
        ApproxEq::approx_epsilon(None::<N>)
    }

    #[inline]
    fn approx_eq(_: &vec::Vec0<N>, _: &vec::Vec0<N>) -> bool {
        true
    }

    #[inline]
    fn approx_eq_eps(_: &vec::Vec0<N>, _: &vec::Vec0<N>, _: &N) -> bool {
        true
    }
}

impl<N: One> One for vec::Vec0<N> {
    #[inline]
    fn one() -> vec::Vec0<N> {
        vec::Vec0
    }
}

impl<N> FromIterator<N> for vec::Vec0<N> {
    #[inline]
    fn from_iterator<I: Iterator<N>>(_: I) -> vec::Vec0<N> {
        vec::Vec0
    }
}

impl<N: Bounded> Bounded for vec::Vec0<N> {
    #[inline]
    fn max_value() -> vec::Vec0<N> {
        vec::Vec0
    }

    #[inline]
    fn min_value() -> vec::Vec0<N> {
        vec::Vec0
    }
}
