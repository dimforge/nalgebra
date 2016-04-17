//! Vector with dimensions unknown at compile-time.

use std::slice::{Iter, IterMut};
use std::iter::{FromIterator, IntoIterator};
use std::iter::repeat;
use std::ops::{Add, Sub, Mul, Div, Neg, AddAssign, SubAssign, MulAssign, DivAssign, Index, IndexMut};
use std::mem;
use rand::{self, Rand};
use num::{Zero, One};
use structs::DMatrix;
use traits::operations::{ApproxEq, Axpy, Mean, Outer};
use traits::geometry::{Dot, Norm};
use traits::structure::{Iterable, IterableMut, Indexable, Shape, BaseFloat, BaseNum, Cast};
#[cfg(feature="arbitrary")]
use quickcheck::{Arbitrary, Gen};

/// Heap allocated, dynamically sized vector.
#[derive(Eq, PartialEq, Debug, Clone)]
pub struct DVector<N> {
    /// Components of the vector. Contains as much elements as the vector dimension.
    pub at: Vec<N>
}

impl<N> DVector<N> {
    /// Creates an uninitialized vector.
    #[inline]
    pub unsafe fn new_uninitialized(dimension: usize) -> DVector<N> {
        let mut vector = Vec::with_capacity(dimension);
        vector.set_len(dimension);

        DVector {
            at: vector
        }
    }
}

impl<N: Clone> DVector<N> {
    /// Builds a vector filled with a constant.
    #[inline]
    pub fn from_elem(dimension: usize, elem: N) -> DVector<N> {
        DVector { at: repeat(elem).take(dimension).collect() }
    }

    /// Builds a vector filled with the components provided by a vector.
    ///
    /// The vector must have at least `dimension` elements.
    #[inline]
    pub fn from_slice(dimension: usize, vector: &[N]) -> DVector<N> {
        assert!(dimension <= vector.len());

        DVector {
            at: vector[.. dimension].to_vec()
        }
    }
}

impl<N> DVector<N> {
    /// Builds a vector filled with the results of a function applied to each of its component coordinates.
    #[inline(always)]
    pub fn from_fn<F: FnMut(usize) -> N>(dimension: usize, mut f: F) -> DVector<N> {
        DVector { at: (0 .. dimension).map(|i| f(i)).collect() }
    }

    /// The vector length.
    #[inline]
    pub fn len(&self) -> usize {
        self.at.len()
    }
}

impl<N> FromIterator<N> for DVector<N> {
    #[inline]
    fn from_iter<I: IntoIterator<Item = N>>(param: I) -> DVector<N> {
        DVector { at: param.into_iter().collect() }
    }
}

impl<N: Copy + BaseNum> Outer for DVector<N> {
    type OuterProductType = DMatrix<N>;

    #[inline]
    fn outer(&self, other: &DVector<N>) -> DMatrix<N> {
        let mut res = unsafe { DMatrix::new_uninitialized(self.at.len(), other.at.len()) };

        for i in 0 .. self.at.len() {
            for j in 0 .. other.at.len() {
                unsafe {
                    res.unsafe_set((i, j), self.unsafe_at(i) * other.unsafe_at(j));
                }
            }
        }

        res
    }
}

#[cfg(feature="arbitrary")]
impl<N: Arbitrary> Arbitrary for DVector<N> {
    fn arbitrary<G: Gen>(g: &mut G) -> DVector<N> {
        DVector { at: Arbitrary::arbitrary(g) }
    }
}


dvec_impl!(DVector);

/// Stack-allocated, dynamically sized vector with a maximum size of 1.
pub struct DVector1<N> {
    at:  [N; 1],
    dimension: usize
}

small_dvec_impl!(DVector1, 1, 0);
small_dvec_from_impl!(DVector1, 1, ::zero());


/// Stack-allocated, dynamically sized vector with a maximum size of 2.
pub struct DVector2<N> {
    at:  [N; 2],
    dimension: usize
}

small_dvec_impl!(DVector2, 2, 0, 1);
small_dvec_from_impl!(DVector2, 2, ::zero(), ::zero());


/// Stack-allocated, dynamically sized vector with a maximum size of 3.
pub struct DVector3<N> {
    at:  [N; 3],
    dimension: usize
}

small_dvec_impl!(DVector3, 3, 0, 1, 2);
small_dvec_from_impl!(DVector3, 3, ::zero(), ::zero(), ::zero());


/// Stack-allocated, dynamically sized vector with a maximum size of 4.
pub struct DVector4<N> {
    at:  [N; 4],
    dimension: usize
}

small_dvec_impl!(DVector4, 4, 0, 1, 2, 3);
small_dvec_from_impl!(DVector4, 4, ::zero(), ::zero(), ::zero(), ::zero());


/// Stack-allocated, dynamically sized vector with a maximum size of 5.
pub struct DVector5<N> {
    at:  [N; 5],
    dimension: usize
}

small_dvec_impl!(DVector5, 5, 0, 1, 2, 3, 4);
small_dvec_from_impl!(DVector5, 5, ::zero(), ::zero(), ::zero(), ::zero(), ::zero());


/// Stack-allocated, dynamically sized vector with a maximum size of 6.
pub struct DVector6<N> {
    at:  [N; 6],
    dimension: usize
}

small_dvec_impl!(DVector6, 6, 0, 1, 2, 3, 4, 5);
small_dvec_from_impl!(DVector6, 6, ::zero(), ::zero(), ::zero(), ::zero(), ::zero(), ::zero());
