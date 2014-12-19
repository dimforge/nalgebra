//! Vector with dimensions unknown at compile-time.

#![allow(missing_docs)] // we hide doc to not have to document the $trhs double dispatch trait.

use std::rand::Rand;
use std::rand;
use std::slice::{Items, MutItems};
use std::iter::FromIterator;
use traits::operations::{ApproxEq, Axpy};
use traits::geometry::{Dot, Norm};
use traits::structure::{Iterable, IterableMut, Indexable, Shape, BaseFloat, BaseNum, Zero, One};

/// Heap allocated, dynamically sized vector.
#[deriving(Eq, PartialEq, Show, Clone)]
pub struct DVec<N> {
    /// Components of the vector. Contains as much elements as the vector dimension.
    pub at: Vec<N>
}

impl<N> DVec<N> {
    /// Creates an uninitialized vec.
    #[inline]
    pub unsafe fn new_uninitialized(dim: uint) -> DVec<N> {
        let mut vec = Vec::with_capacity(dim);
        vec.set_len(dim);

        DVec {
            at: vec
        }
    }
}

impl<N: Clone> DVec<N> {
    /// Builds a vector filled with a constant.
    #[inline]
    pub fn from_elem(dim: uint, elem: N) -> DVec<N> {
        DVec { at: Vec::from_elem(dim, elem) }
    }

    /// Builds a vector filled with the components provided by a vector.
    ///
    /// The vector must have at least `dim` elements.
    #[inline]
    pub fn from_slice(dim: uint, vec: &[N]) -> DVec<N> {
        assert!(dim <= vec.len());

        DVec {
            at: vec.slice_to(dim).to_vec()
        }
    }
}

impl<N> DVec<N> {
    /// Builds a vector filled with the result of a function.
    #[inline(always)]
    pub fn from_fn(dim: uint, f: |uint| -> N) -> DVec<N> {
        DVec { at: Vec::from_fn(dim, |i| f(i)) }
    }

    #[inline]
    pub fn len(&self) -> uint {
        self.at.len()
    }
}

impl<N> FromIterator<N> for DVec<N> {
    #[inline]
    fn from_iter<I: Iterator<N>>(mut param: I) -> DVec<N> {
        let mut res = DVec { at: Vec::new() };

        for e in param {
            res.at.push(e)
        }

        res
    }
}


dvec_impl!(DVec);

/// Stack-allocated, dynamically sized vector with a maximum size of 1.
pub struct DVec1<N> {
    at:  [N, ..1],
    dim: uint
}

small_dvec_impl!(DVec1, 1, 0);
small_dvec_from_impl!(DVec1, 1, ::zero());


/// Stack-allocated, dynamically sized vector with a maximum size of 2.
pub struct DVec2<N> {
    at:  [N, ..2],
    dim: uint
}

small_dvec_impl!(DVec2, 2, 0, 1);
small_dvec_from_impl!(DVec2, 2, ::zero(), ::zero());


/// Stack-allocated, dynamically sized vector with a maximum size of 3.
pub struct DVec3<N> {
    at:  [N, ..3],
    dim: uint
}

small_dvec_impl!(DVec3, 3, 0, 1, 2);
small_dvec_from_impl!(DVec3, 3, ::zero(), ::zero(), ::zero());


/// Stack-allocated, dynamically sized vector with a maximum size of 4.
pub struct DVec4<N> {
    at:  [N, ..4],
    dim: uint
}

small_dvec_impl!(DVec4, 4, 0, 1, 2, 3);
small_dvec_from_impl!(DVec4, 4, ::zero(), ::zero(), ::zero(), ::zero());


/// Stack-allocated, dynamically sized vector with a maximum size of 5.
pub struct DVec5<N> {
    at:  [N, ..5],
    dim: uint
}

small_dvec_impl!(DVec5, 5, 0, 1, 2, 3, 4);
small_dvec_from_impl!(DVec5, 5, ::zero(), ::zero(), ::zero(), ::zero(), ::zero());


/// Stack-allocated, dynamically sized vector with a maximum size of 6.
pub struct DVec6<N> {
    at:  [N, ..6],
    dim: uint
}

small_dvec_impl!(DVec6, 6, 0, 1, 2, 3, 4, 5);
small_dvec_from_impl!(DVec6, 6, ::zero(), ::zero(), ::zero(), ::zero(), ::zero(), ::zero());
