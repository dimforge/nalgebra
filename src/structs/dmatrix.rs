//! Matrix with dimensions unknown at compile-time.

use std::cmp;
use std::mem;
use std::iter::repeat;
use std::ops::{Add, Sub, Mul, Div, AddAssign, SubAssign, MulAssign, Index, IndexMut};
use std::fmt::{Debug, Formatter, Result};
use rand::{self, Rand};
use num::{Zero, One};
use structs::dvector::{DVector, DVector1, DVector2, DVector3, DVector4, DVector5, DVector6};
use traits::operations::{ApproxEq, Inverse, Transpose, Mean, Covariance};
use traits::structure::{Cast, Column, ColumnSlice, Row, RowSlice, Diagonal, DiagMut, Eye, Indexable, Shape, BaseNum};
#[cfg(feature="arbitrary")]
use quickcheck::{Arbitrary, Gen};


/// Matrix with dimensions unknown at compile-time.
#[derive(Eq, PartialEq, Clone)]
pub struct DMatrix<N> {
    nrows: usize,
    ncols: usize,
    mij:   Vec<N>
}

impl<N> DMatrix<N> {
    /// Creates an uninitialized matrix.
    #[inline]
    pub unsafe fn new_uninitialized(nrows: usize, ncols: usize) -> DMatrix<N> {
        let mut vector = Vec::with_capacity(nrows * ncols);
        vector.set_len(nrows * ncols);

        DMatrix {
            nrows: nrows,
            ncols: ncols,
            mij:   vector
        }
    }
}

impl<N: Clone + Copy> DMatrix<N> {
    /// Builds a matrix filled with a given constant.
    #[inline]
    pub fn from_element(nrows: usize, ncols: usize, val: N) -> DMatrix<N> {
        DMatrix {
            nrows: nrows,
            ncols: ncols,
            mij:   repeat(val).take(nrows * ncols).collect()
        }
    }

    /// Builds a matrix filled with the components provided by a vector.
    /// The vector contains the matrix data in row-major order.
    /// Note that `from_column_vector` is much faster than `from_row_vector` since a `DMatrix` stores its data
    /// in column-major order.
    ///
    /// The vector must have exactly `nrows * ncols` elements.
    #[inline]
    pub fn from_row_vector(nrows: usize, ncols: usize, vector: &[N]) -> DMatrix<N> {
        DMatrix::from_row_iter(nrows, ncols, vector.to_vec())
    }

    /// Builds a matrix filled with the components provided by a vector.
    /// The vector contains the matrix data in column-major order.
    /// Note that `from_column_vector` is much faster than `from_row_vector` since a `DMatrix` stores its data
    /// in column-major order.
    ///
    /// The vector must have exactly `nrows * ncols` elements.
    #[inline]
    pub fn from_column_vector(nrows: usize, ncols: usize, vector: &[N]) -> DMatrix<N> {
        DMatrix::from_column_iter(nrows, ncols, vector.to_vec())
    }

    /// Builds a matrix filled with the components provided by a source that may be moved into an iterator.
    /// The source contains the matrix data in row-major order.
    /// Note that `from_column_iter` is much faster than `from_row_iter` since a `DMatrix` stores its data
    /// in column-major order.
    ///
    /// The source must have exactly `nrows * ncols` elements.
    #[inline]
    pub fn from_row_iter<I: IntoIterator<Item = N>>(nrows: usize, ncols: usize, param: I) -> DMatrix<N> {
        let mut res = DMatrix::from_column_iter(ncols, nrows, param);

        // we transpose because the buffer is row_major
        res.transpose_mut();

        res
    }


    /// Builds a matrix filled with the components provided by a source that may be moved into an iterator.
    /// The source contains the matrix data in column-major order.
    /// Note that `from_column_iter` is much faster than `from_row_iter` since a `DMatrix` stores its data
    /// in column-major order.
    ///
    /// The source must have exactly `nrows * ncols` elements.
    #[inline]
    pub fn from_column_iter<I: IntoIterator<Item = N>>(nrows: usize, ncols: usize, param: I) -> DMatrix<N> {
        let mij: Vec<N> = param.into_iter().collect();

        assert!(nrows * ncols == mij.len(), "The ammount of data provided does not matches the matrix size.");

        DMatrix {
            nrows: nrows,
            ncols: ncols,
            mij:   mij
        }
    }
}

impl<N> DMatrix<N> {
    /// Builds a matrix filled with the results of a function applied to each of its component coordinates.
    #[inline(always)]
    pub fn from_fn<F: FnMut(usize, usize) -> N>(nrows: usize, ncols: usize, mut f: F) -> DMatrix<N> {
        DMatrix {
            nrows: nrows,
            ncols: ncols,
            mij:   (0 .. nrows * ncols).map(|i| { let m = i / nrows; f(i - m * nrows, m) }).collect()
        }
    }

    /// Transforms this matrix into an array. This consumes the matrix and is O(1).
    /// The returned vector contains the matrix data in column-major order.
    #[inline]
    pub fn into_vector(self) -> Vec<N> {
        self.mij
    }
}

dmat_impl!(DMatrix, DVector);


/// A stack-allocated dynamically sized matrix with at most one row and column.
pub struct DMatrix1<N> {
    nrows: usize,
    ncols: usize,
    mij:   [N; 1 * 1],
}

small_dmat_impl!(DMatrix1, DVector1, 1, 0);
small_dmat_from_impl!(DMatrix1, 1, ::zero());


/// A stack-allocated dynamically sized square or rectangular matrix with at most 2 rows and columns.
pub struct DMatrix2<N> {
    nrows: usize,
    ncols: usize,
    mij:   [N; 2 * 2],
}

small_dmat_impl!(DMatrix2, DVector2, 2, 0, 1,
                                  2, 3);
small_dmat_from_impl!(DMatrix2, 2, ::zero(), ::zero(),
                                ::zero(), ::zero());


/// A stack-allocated dynamically sized square or rectangular matrix with at most 3 rows and columns.
pub struct DMatrix3<N> {
    nrows: usize,
    ncols: usize,
    mij:   [N; 3 * 3],
}

small_dmat_impl!(DMatrix3, DVector3, 3, 0, 1, 2,
                                  3, 4, 5,
                                  6, 7, 8);
small_dmat_from_impl!(DMatrix3, 3, ::zero(), ::zero(), ::zero(),
                                ::zero(), ::zero(), ::zero(),
                                ::zero(), ::zero(), ::zero());


/// A stack-allocated dynamically sized square or rectangular matrix with at most 4 rows and columns.
pub struct DMatrix4<N> {
    nrows: usize,
    ncols: usize,
    mij:   [N; 4 * 4],
}

small_dmat_impl!(DMatrix4, DVector4, 4,  0,  1,  2,  3,
                                   4,  5,  6,  7,
                                   8,  9,  10, 11,
                                   12, 13, 14, 15);
small_dmat_from_impl!(DMatrix4, 4, ::zero(), ::zero(), ::zero(), ::zero(),
                                ::zero(), ::zero(), ::zero(), ::zero(),
                                ::zero(), ::zero(), ::zero(), ::zero(),
                                ::zero(), ::zero(), ::zero(), ::zero());


/// A stack-allocated dynamically sized square or rectangular matrix with at most 5 rows and columns.
pub struct DMatrix5<N> {
    nrows: usize,
    ncols: usize,
    mij:   [N; 5 * 5],
}

small_dmat_impl!(DMatrix5, DVector5, 5, 0,  1,  2,  3,  4,
                                  5,  6,  7,  8,  9,
                                  10, 11, 12, 13, 14,
                                  15, 16, 17, 18, 19,
                                  20, 21, 22, 23, 24);
small_dmat_from_impl!(DMatrix5, 5, ::zero(), ::zero(), ::zero(), ::zero(), ::zero(),
                                ::zero(), ::zero(), ::zero(), ::zero(), ::zero(),
                                ::zero(), ::zero(), ::zero(), ::zero(), ::zero(),
                                ::zero(), ::zero(), ::zero(), ::zero(), ::zero(),
                                ::zero(), ::zero(), ::zero(), ::zero(), ::zero());


/// A stack-allocated dynamically sized square or rectangular matrix with at most 6 rows and columns.
pub struct DMatrix6<N> {
    nrows: usize,
    ncols: usize,
    mij:   [N; 6 * 6],
}

small_dmat_impl!(DMatrix6, DVector6, 6, 0,  1,  2,  3,  4,  5,
                                  6,  7,  8,  9,  10, 11,
                                  12, 13, 14, 15, 16, 17,
                                  18, 19, 20, 21, 22, 23,
                                  24, 25, 26, 27, 28, 29,
                                  30, 31, 32, 33, 34, 35);
small_dmat_from_impl!(DMatrix6, 6, ::zero(), ::zero(), ::zero(), ::zero(), ::zero(), ::zero(),
                                ::zero(), ::zero(), ::zero(), ::zero(), ::zero(), ::zero(),
                                ::zero(), ::zero(), ::zero(), ::zero(), ::zero(), ::zero(),
                                ::zero(), ::zero(), ::zero(), ::zero(), ::zero(), ::zero(),
                                ::zero(), ::zero(), ::zero(), ::zero(), ::zero(), ::zero(),
                                ::zero(), ::zero(), ::zero(), ::zero(), ::zero(), ::zero());
