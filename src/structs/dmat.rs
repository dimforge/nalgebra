//! Matrix with dimensions unknown at compile-time.

#![allow(missing_docs)] // we hide doc to not have to document the $trhs double dispatch trait.

use std::cmp;
use std::mem;
use std::iter::repeat;
use std::ops::{Add, Sub, Mul, Div, Index, IndexMut};
use std::fmt::{Debug, Formatter, Result};
use rand::{self, Rand};
use num::{Zero, One};
use structs::dvec::DVec;
use traits::operations::{ApproxEq, Inv, Transpose, Mean, Cov};
use traits::structure::{Cast, Col, ColSlice, Row, RowSlice, Diag, DiagMut, Eye, Indexable, Shape, BaseNum};
#[cfg(feature="arbitrary")]
use quickcheck::{Arbitrary, Gen};


/// Matrix with dimensions unknown at compile-time.
#[derive(Eq, PartialEq, Clone)]
pub struct DMat<N> {
    nrows: usize,
    ncols: usize,
    mij:   Vec<N>
}

impl<N> DMat<N> {
    /// Creates an uninitialized matrix.
    #[inline]
    pub unsafe fn new_uninitialized(nrows: usize, ncols: usize) -> DMat<N> {
        let mut vec = Vec::with_capacity(nrows * ncols);
        vec.set_len(nrows * ncols);

        DMat {
            nrows: nrows,
            ncols: ncols,
            mij:   vec
        }
    }
}

impl<N: Clone + Copy> DMat<N> {
    /// Builds a matrix filled with a given constant.
    #[inline]
    pub fn from_elem(nrows: usize, ncols: usize, val: N) -> DMat<N> {
        DMat {
            nrows: nrows,
            ncols: ncols,
            mij:   repeat(val).take(nrows * ncols).collect()
        }
    }

    /// Builds a matrix filled with the components provided by a vector.
    /// The vector contains the matrix data in row-major order.
    /// Note that `from_col_vec` is much faster than `from_row_vec` since a `DMat` stores its data
    /// in column-major order.
    ///
    /// The vector must have at least `nrows * ncols` elements.
    #[inline]
    pub fn from_row_vec(nrows: usize, ncols: usize, vec: &[N]) -> DMat<N> {
        let mut res = DMat::from_col_vec(ncols, nrows, vec);

        // we transpose because the buffer is row_major
        res.transpose_mut();

        res
    }

    /// Builds a matrix filled with the components provided by a vector.
    /// The vector contains the matrix data in column-major order.
    /// Note that `from_col_vec` is much faster than `from_row_vec` since a `DMat` stores its data
    /// in column-major order.
    ///
    /// The vector must have at least `nrows * ncols` elements.
    #[inline]
    pub fn from_col_vec(nrows: usize, ncols: usize, vec: &[N]) -> DMat<N> {
        assert!(nrows * ncols == vec.len());

        DMat {
            nrows: nrows,
            ncols: ncols,
            mij:   vec.to_vec()
        }
    }
}

impl<N> DMat<N> {
    /// Builds a matrix using an initialization function.
    #[inline(always)]
    pub fn from_fn<F: FnMut(usize, usize) -> N>(nrows: usize, ncols: usize, mut f: F) -> DMat<N> {
        DMat {
            nrows: nrows,
            ncols: ncols,
            mij:   (0 .. nrows * ncols).map(|i| { let m = i / nrows; f(i - m * nrows, m) }).collect()
        }
    }

    /// Transforms this matrix into an array. This consumes the matrix and is O(1).
    /// The returned vector contains the matrix data in column-major order.
    #[inline]
    pub fn into_vec(self) -> Vec<N> {
        self.mij
    }
}

dmat_impl!(DMat);


pub struct DMat1<N> {
    nrows: usize,
    ncols: usize,
    mij:   [N; 1 * 1],
}

small_dmat_impl!(DMat1, 1, 0);
small_dmat_from_impl!(DMat1, 1, ::zero());


pub struct DMat2<N> {
    nrows: usize,
    ncols: usize,
    mij:   [N; 2 * 2],
}

small_dmat_impl!(DMat2, 2, 0, 1,
                           2, 3);
small_dmat_from_impl!(DMat2, 2, ::zero(), ::zero(),
                                ::zero(), ::zero());


pub struct DMat3<N> {
    nrows: usize,
    ncols: usize,
    mij:   [N; 3 * 3],
}

small_dmat_impl!(DMat3, 3, 0, 1, 2,
                           3, 4, 5,
                           6, 7, 8);
small_dmat_from_impl!(DMat3, 3, ::zero(), ::zero(), ::zero(),
                                ::zero(), ::zero(), ::zero(),
                                ::zero(), ::zero(), ::zero());


pub struct DMat4<N> {
    nrows: usize,
    ncols: usize,
    mij:   [N; 4 * 4],
}

small_dmat_impl!(DMat4, 4,  0,  1,  2,  3,
                            4,  5,  6,  7,
                            8,  9,  10, 11,
                            12, 13, 14, 15);
small_dmat_from_impl!(DMat4, 4, ::zero(), ::zero(), ::zero(), ::zero(),
                                ::zero(), ::zero(), ::zero(), ::zero(),
                                ::zero(), ::zero(), ::zero(), ::zero(),
                                ::zero(), ::zero(), ::zero(), ::zero());


pub struct DMat5<N> {
    nrows: usize,
    ncols: usize,
    mij:   [N; 5 * 5],
}

small_dmat_impl!(DMat5, 5, 0,  1,  2,  3,  4,
                           5,  6,  7,  8,  9,
                           10, 11, 12, 13, 14,
                           15, 16, 17, 18, 19,
                           20, 21, 22, 23, 24);
small_dmat_from_impl!(DMat5, 5, ::zero(), ::zero(), ::zero(), ::zero(), ::zero(),
                                ::zero(), ::zero(), ::zero(), ::zero(), ::zero(),
                                ::zero(), ::zero(), ::zero(), ::zero(), ::zero(),
                                ::zero(), ::zero(), ::zero(), ::zero(), ::zero(),
                                ::zero(), ::zero(), ::zero(), ::zero(), ::zero());


pub struct DMat6<N> {
    nrows: usize,
    ncols: usize,
    mij:   [N; 6 * 6],
}

small_dmat_impl!(DMat6, 6, 0,  1,  2,  3,  4,  5,
                           6,  7,  8,  9,  10, 11,
                           12, 13, 14, 15, 16, 17,
                           18, 19, 20, 21, 22, 23,
                           24, 25, 26, 27, 28, 29,
                           30, 31, 32, 33, 34, 35);
small_dmat_from_impl!(DMat6, 6, ::zero(), ::zero(), ::zero(), ::zero(), ::zero(), ::zero(),
                                ::zero(), ::zero(), ::zero(), ::zero(), ::zero(), ::zero(),
                                ::zero(), ::zero(), ::zero(), ::zero(), ::zero(), ::zero(),
                                ::zero(), ::zero(), ::zero(), ::zero(), ::zero(), ::zero(),
                                ::zero(), ::zero(), ::zero(), ::zero(), ::zero(), ::zero(),
                                ::zero(), ::zero(), ::zero(), ::zero(), ::zero(), ::zero());
