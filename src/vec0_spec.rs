use std::cast;
use std::num::{Zero, One, Algebraic, Bounded};
use std::vec::{VecIterator, VecMutIterator};
use std::iterator::{Iterator, FromIterator};
use std::cmp::ApproxEq;
use traits::iterable::{Iterable, IterableMut};
use traits::basis::Basis;
use traits::dim::Dim;
use traits::dot::Dot;
use traits::sub_dot::SubDot;
use traits::norm::Norm;
use traits::translation::{Translation, Translatable};
use traits::scalar_op::{ScalarMul, ScalarDiv, ScalarAdd, ScalarSub};
use traits::ring::{Ring, DivisionRing};
use traits::indexable::Indexable;
use vec;

impl<N> vec::Vec0<N> {
    /// Creates a new vector.
    #[inline]
    pub fn new() -> vec::Vec0<N> {
        vec::Vec0
    }
}

impl<N: Clone> Indexable<uint, N> for vec::Vec0<N> {
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
}

impl<N: Clone> vec::Vec0<N> {
    /// Creates a new vector. The parameter is not taken in account.
    #[inline]
    pub fn new_repeat(_: N) -> vec::Vec0<N> {
        vec::Vec0
    }
}

impl<N> Iterable<N> for vec::Vec0<N> {
    #[inline]
    fn iter<'l>(&'l self) -> VecIterator<'l, N> {
        unsafe { cast::transmute::<&'l vec::Vec0<N>, &'l [N, ..0]>(self).iter() }
    }
}

impl<N> IterableMut<N> for vec::Vec0<N> {
    #[inline]
    fn mut_iter<'l>(&'l mut self) -> VecMutIterator<'l, N> {
        unsafe { cast::transmute::<&'l mut vec::Vec0<N>, &'l mut [N, ..0]>(self).mut_iter() }
    }
}

impl<N> Dim for vec::Vec0<N> {
    #[inline]
    fn dim() -> uint {
        0
    }
}

impl<N: Clone + DivisionRing + Algebraic + ApproxEq<N>> Basis for vec::Vec0<N> {
    #[inline(always)]
    fn canonical_basis(_: &fn(vec::Vec0<N>) -> bool) { }

    #[inline(always)]
    fn orthonormal_subspace_basis(&self, _: &fn(vec::Vec0<N>) -> bool) { }
}

impl<N: Clone + Add<N,N>> Add<vec::Vec0<N>, vec::Vec0<N>> for vec::Vec0<N> {
    #[inline]
    fn add(&self, _: &vec::Vec0<N>) -> vec::Vec0<N> {
        vec::Vec0
    }
}

impl<N: Clone + Sub<N,N>> Sub<vec::Vec0<N>, vec::Vec0<N>> for vec::Vec0<N> {
    #[inline]
    fn sub(&self, _: &vec::Vec0<N>) -> vec::Vec0<N> { 
        vec::Vec0
    }
}

impl<N: Neg<N>> Neg<vec::Vec0<N>> for vec::Vec0<N> {
    #[inline]
    fn neg(&self) -> vec::Vec0<N> {
        vec::Vec0
    }
}

impl<N: Ring> Dot<N> for vec::Vec0<N> {
    #[inline]
    fn dot(&self, _: &vec::Vec0<N>) -> N {
        Zero::zero()
    } 
}

impl<N: Clone + Ring> SubDot<N> for vec::Vec0<N> {
    #[inline]
    fn sub_dot(&self, _: &vec::Vec0<N>, _: &vec::Vec0<N>) -> N {
        Zero::zero()
    } 
}

impl<N: Mul<N, N>> ScalarMul<N> for vec::Vec0<N> {
    #[inline]
    fn scalar_mul(&self, _: &N) -> vec::Vec0<N> {
        vec::Vec0
    }

    #[inline]
    fn scalar_mul_inplace(&mut self, _: &N) { }
}

impl<N: Div<N, N>> ScalarDiv<N> for vec::Vec0<N> {
    #[inline]
    fn scalar_div(&self, _: &N) -> vec::Vec0<N> {
        vec::Vec0
    }

    #[inline]
    fn scalar_div_inplace(&mut self, _: &N) { }
}

impl<N: Add<N, N>> ScalarAdd<N> for vec::Vec0<N> {
    #[inline]
    fn scalar_add(&self, _: &N) -> vec::Vec0<N> {
        vec::Vec0
    }

    #[inline]
    fn scalar_add_inplace(&mut self, _: &N) { }
}

impl<N: Sub<N, N>> ScalarSub<N> for vec::Vec0<N> {
    #[inline]
    fn scalar_sub(&self, _: &N) -> vec::Vec0<N> {
        vec::Vec0
    }

    #[inline]
    fn scalar_sub_inplace(&mut self, _: &N) { }
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
    fn translate_by(&mut self, t: &vec::Vec0<N>) {
        *self = *self + *t;
    }
}

impl<N: Add<N, N> + Neg<N> + Clone> Translatable<vec::Vec0<N>, vec::Vec0<N>> for vec::Vec0<N> {
    #[inline]
    fn translated(&self, t: &vec::Vec0<N>) -> vec::Vec0<N> {
        self + *t
    }
}

impl<N: Clone + DivisionRing + Algebraic> Norm<N> for vec::Vec0<N> {
    #[inline]
    fn sqnorm(&self) -> N {
        self.dot(self)
    }

    #[inline]
    fn norm(&self) -> N {
        self.sqnorm().sqrt()
    }

    #[inline]
    fn normalized(&self) -> vec::Vec0<N> {
        let mut res : vec::Vec0<N> = self.clone();

        res.normalize();

        res
    }

    #[inline]
    fn normalize(&mut self) -> N {
        let l = self.norm();

        self.scalar_div_inplace(&l);

        l
    }
}

impl<N: ApproxEq<N>> ApproxEq<N> for vec::Vec0<N> {
    #[inline]
    fn approx_epsilon() -> N {
        ApproxEq::approx_epsilon::<N, N>()
    }

    #[inline]
    fn approx_eq(&self, _: &vec::Vec0<N>) -> bool {
        true
    }

    #[inline]
    fn approx_eq_eps(&self, _: &vec::Vec0<N>, _: &N) -> bool {
        true
    }
}

impl<N: Clone + One> One for vec::Vec0<N> {
    #[inline]
    fn one() -> vec::Vec0<N> {
        vec::Vec0
    }
}

impl<N> FromIterator<N> for vec::Vec0<N> {
    #[inline]
    fn from_iterator<I: Iterator<N>>(_: &mut I) -> vec::Vec0<N> {
        vec::Vec0
    }
}

impl<N: Bounded + Clone> Bounded for vec::Vec0<N> {
    #[inline]
    fn max_value() -> vec::Vec0<N> {
        vec::Vec0
    }

    #[inline]
    fn min_value() -> vec::Vec0<N> {
        vec::Vec0
    }
}
