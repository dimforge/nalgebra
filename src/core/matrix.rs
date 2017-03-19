use num::Zero;

use std::cmp::Ordering;
use std::marker::PhantomData;
use std::fmt;
use std::any::TypeId;
use std::mem;
use approx::ApproxEq;

use alga::general::{Ring, Real};

use core::{Scalar, Unit};
use core::dimension::{Dim, DimAdd, DimSum, U1, U2};
use core::constraint::{ShapeConstraint, SameNumberOfRows, SameNumberOfColumns};
use core::iter::{MatrixIter, MatrixIterMut};
use core::allocator::{Allocator, OwnedAllocator, SameShapeAllocator, SameShapeR, SameShapeC};
use core::storage::{Storage, StorageMut, Owned, OwnedStorage, MulStorage, TrMulStorage, SumStorage};

/// The type of the result of a matrix allocation by the allocator `A`.
pub type OwnedMatrix<N, R, C, A> = Matrix<N, R, C, <A as Allocator<N, R, C>>::Buffer>;

/// A square matrix.
pub type SquareMatrix<N, D, S> = Matrix<N, D, D, S>;

/// The type of the result of a square matrix allocation by the allocator `A`.
pub type OwnedSquareMatrix<N, D, A> = OwnedMatrix<N, D, D, A>;

/// A matrix with one column and `D` rows.
pub type ColumnVector<N, D, S> = Matrix<N, D, U1, S>;

/// An owned matrix with one column and `D` rows.
pub type OwnedColumnVector<N, D, A> = OwnedMatrix<N, D, U1, A>;

/// An owned matrix with one row and `D` columns.
pub type OwnedRowVector<N, D, A> = OwnedMatrix<N, U1, D, A>;

/// The type of the result of a matrix sum.
pub type MatrixSum<N, R1, C1, R2, C2, SA> =
    Matrix<N, SameShapeR<R1, R2>, SameShapeC<C1, C2>, SumStorage<N, R1, C1, R2, C2, SA>>;

/// The type of the result of a matrix sum.
pub type ColumnVectorSum<N, R1, R2, SA> =
    Matrix<N, SameShapeR<R1, R2>, U1, SumStorage<N, R1, U1, R2, U1, SA>>;

/// The type of the result of a matrix cross product.
pub type MatrixCross<N, R1, C1, R2, C2, SA> = MatrixSum<N, R1, C1, R2, C2, SA>;

/// The type of the result of a matrix multiplication.
pub type MatrixMul<N, R1, C1, C2, SA> = Matrix<N, R1, C2, MulStorage<N, R1, C1, C2, SA>>;

/// The type of the result of a matrix transpose-multiplication.
pub type MatrixTrMul<N, R1, C1, C2, SA> = Matrix<N, C1, C2, TrMulStorage<N, R1, C1, C2, SA>>;

/// The matrix with storage `S` and scalar type changed from `NOld` to `NNew`.
pub type MatrixWithScalar<NOld, NNew, R, C, S> =
Matrix<NNew, R, C, <<S as Storage<NOld, R, C>>::Alloc as Allocator<NNew, R, C>>::Buffer>;

/// The most generic column-major matrix (and vector) type.
///
/// It combines four type parameters:
/// - `N`: for the matrix components scalar type.
/// - `R`: for the matrix number of rows.
/// - `C`: for the matrix number of columns.
/// - `S`: for the matrix data storage, i.e., the buffer that actually contains the matrix
/// components.
///
/// The matrix dimensions parameters `R` and `C` can either be:
/// - type-level unsigned integer contants (e.g. `U1`, `U124`) from the `nalgebra::` root module.
/// All numbers from 0 to 127 are defined that way.
/// - type-level unsigned integer constants (e.g. `U1024`, `U10000`) from the `typenum::` crate.
/// Using those, you will not get error messages as nice as for numbers smaller than 128 defined on
/// the `nalgebra::` module.
/// - the special value `Dynamic` from the `nalgebra::` root module. This indicates that the
/// specified dimension is not known at compile-time. Note that this will generally imply that the
/// matrix data storage `S` performs a dynamic allocation and contains extra metadata for the
/// matrix shape.
///
/// Note that mixing `Dynamic` with type-level unsigned integers is allowed. Actually, a
/// dynamically-sized column vector should be represented as a `Matrix<N, Dynamic, U1, S>` (given
/// some concrete types for `N` and a compatible data storage type `S`).
#[repr(C)]
#[derive(Hash, Debug, Clone, Copy)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
pub struct Matrix<N: Scalar, R: Dim, C: Dim, S> {
    /// The data storage that contains all the matrix components and informations about its number
    /// of rows and column (if needed).
    pub data:  S,

    #[cfg_attr(feature = "serde-serialize", serde(skip_serializing, skip_deserializing))]
    _phantoms: PhantomData<(N, R, C)>
}

impl<N: Scalar, R: Dim, C: Dim, S> Matrix<N, R, C, S> {
    /// Creates a new matrix with the given data without statically checking that the matrix
    /// dimension matches the storage dimension.
    #[inline]
    pub unsafe fn from_data_statically_unchecked(data: S) -> Matrix<N, R, C, S> {
        Matrix {
            data:      data,
            _phantoms: PhantomData
        }
    }
}

impl<N: Scalar, R: Dim, C: Dim, S: Storage<N, R, C>> Matrix<N, R, C, S> {
    /// Creates a new matrix with the given data.
    #[inline]
    pub fn from_data(data: S) -> Matrix<N, R, C, S> {
        unsafe {
            Self::from_data_statically_unchecked(data)
        }
    }

    /// Moves this matrix into one that owns its data.
    #[inline]
    pub fn into_owned(self) -> OwnedMatrix<N, R, C, S::Alloc> {
        Matrix::from_data(self.data.into_owned())
    }

    // FIXME: this could probably benefit from specialization.
    // XXX: bad name.
    /// Moves this matrix into one that owns its data. The actual type of the result depends on
    /// matrix storage combination rules for addition.
    #[inline]
    pub fn into_owned_sum<R2, C2>(self) -> MatrixSum<N, R, C, R2, C2, S>
        where R2: Dim, C2: Dim,
              S::Alloc: SameShapeAllocator<N, R, C, R2, C2, S>,
              ShapeConstraint: SameNumberOfRows<R, R2> + SameNumberOfColumns<C, C2> {
        if TypeId::of::<SumStorage<N, R, C, R2, C2, S>>() == TypeId::of::<Owned<N, R, C, S::Alloc>>() {
            // We can just return `self.into_owned()`.

            unsafe {
                // FIXME: check that those copies are optimized away by the compiler.
                let owned = self.into_owned();
                let res = mem::transmute_copy(&owned);
                mem::forget(owned);
                res
            }
        }
        else {
            self.clone_owned_sum()
        }
    }

    /// Clones this matrix into one that owns its data.
    #[inline]
    pub fn clone_owned(&self) -> OwnedMatrix<N, R, C, S::Alloc> {
        Matrix::from_data(self.data.clone_owned())
    }

    /// Clones this matrix into one that owns its data. The actual type of the result depends on
    /// matrix storage combination rules for addition.
    #[inline]
    pub fn clone_owned_sum<R2, C2>(&self) -> MatrixSum<N, R, C, R2, C2, S>
        where R2: Dim, C2: Dim,
              S::Alloc: SameShapeAllocator<N, R, C, R2, C2, S>,
              ShapeConstraint: SameNumberOfRows<R, R2> + SameNumberOfColumns<C, C2> {
        let (nrows, ncols) = self.shape();
        let nrows: SameShapeR<R, R2> = Dim::from_usize(nrows);
        let ncols: SameShapeC<C, C2> = Dim::from_usize(ncols);

        let mut res: MatrixSum<N, R, C, R2, C2, S> = unsafe {
            Matrix::new_uninitialized_generic(nrows, ncols)
        };

        for (r, s) in res.iter_mut().zip(self.iter()) {
            *r = *s
        }

        res
    }

    /// The total number of elements of this matrix.
    #[inline]
    pub fn len(&self) -> usize {
        let (nrows, ncols) = self.shape();
        nrows * ncols
    }

    /// The shape of this matrix returned as the tuple (number of rows, number of columns).
    #[inline]
    pub fn shape(&self) -> (usize, usize) {
        let (nrows, ncols) = self.data.shape();
        (nrows.value(), ncols.value())
    }

    /// The number of rows of this matrix.
    #[inline]
    pub fn nrows(&self) -> usize {
        self.shape().0
    }

    /// The number of columns of this matrix.
    #[inline]
    pub fn ncols(&self) -> usize {
        self.shape().1
    }

    /// The strides (row stride, column stride) of this matrix.
    #[inline]
    pub fn strides(&self) -> (usize, usize) {
        let (srows, scols) = self.data.strides();
        (srows.value(), scols.value())
    }

    /// Iterates through this matrix coordinates.
    #[inline]
    pub fn iter(&self) -> MatrixIter<N, R, C, S> {
        MatrixIter::new(&self.data)
    }

    /// Computes the row and column coordinates of the i-th element of this matrix seen as a
    /// vector.
    #[inline]
    pub fn vector_to_matrix_index(&self, i: usize) -> (usize, usize) {
        let (nrows, ncols) = self.shape();

        // Two most common uses that should be optimized by the compiler for statically-sized
        // matrices.
        if nrows == 1 {
            (0, i)
        }
        else if ncols == 1 {
            (i, 0)
        }
        else {
            (i % nrows, i / nrows)
        }
    }

    /// Gets a reference to the element of this matrix at row `irow` and column `icol` without
    /// bound-checking.
    #[inline]
    pub unsafe fn get_unchecked(&self, irow: usize, icol: usize) -> &N {
        self.data.get_unchecked(irow, icol)
    }

    /// Tests whether `self` and `rhs` are equal up to a given epsilon.
    ///
    /// See `relative_eq` from the `ApproxEq` trait for more details.
    #[inline]
    pub fn relative_eq<R2, C2, SB>(&self, other: &Matrix<N, R2, C2, SB>,
                                   eps: N::Epsilon, max_relative: N::Epsilon)
                                   -> bool
        where N: ApproxEq,
              R2: Dim, C2: Dim,
              SB: Storage<N, R2, C2>,
              N::Epsilon: Copy,
              ShapeConstraint: SameNumberOfRows<R, R2> + SameNumberOfColumns<C, C2> {

       assert!(self.shape() == other.shape());
       self.iter().zip(other.iter()).all(|(a, b)| a.relative_eq(b, eps, max_relative))
    }
}

impl<N: Scalar, R: Dim, C: Dim, S: StorageMut<N, R, C>> Matrix<N, R, C, S> {
    /// Mutably iterates through this matrix coordinates.
    #[inline]
    pub fn iter_mut(&mut self) -> MatrixIterMut<N, R, C, S> {
        MatrixIterMut::new(&mut self.data)
    }

    /// Gets a mutable reference to the i-th element of this matrix.
    #[inline]
    pub unsafe fn get_unchecked_mut(&mut self, irow: usize, icol: usize) -> &mut N {
        self.data.get_unchecked_mut(irow, icol)
    }

    /// Swaps two entries without bound-checking.
    #[inline]
    pub unsafe fn swap_unchecked(&mut self, row_cols1: (usize, usize), row_cols2: (usize, usize)) {
        self.data.swap_unchecked(row_cols1, row_cols2)
    }

    /// Swaps two entries.
    #[inline]
    pub fn swap(&mut self, row_cols1: (usize, usize), row_cols2: (usize, usize)) {
        let (nrows, ncols) = self.shape();
        assert!(row_cols1.0 < nrows && row_cols1.1 < ncols, "Matrix elements swap index out of bounds.");
        assert!(row_cols2.0 < nrows && row_cols2.1 < ncols, "Matrix elements swap index out of bounds.");
        unsafe { self.swap_unchecked(row_cols1, row_cols2) }
    }

    /// Fills this matrix with the content of another one. Both must have the same shape.
    #[inline]
    pub fn copy_from<R2, C2, SB>(&mut self, other: &Matrix<N, R2, C2, SB>)
        where R2: Dim, C2: Dim,
              SB: Storage<N, R2, C2>,
              ShapeConstraint: SameNumberOfRows<R, R2> + SameNumberOfColumns<C, C2> {
        assert!(self.shape() == other.shape(), "Unable to copy from a matrix with a different shape.");

        for (out, other) in self.iter_mut().zip(other.iter()) {
            *out = *other
        }
    }

    /// Sets all the entries of this matrix to `value`.
    #[inline]
    pub fn fill(&mut self, value: N) {
        for e in self.iter_mut() {
            *e = value
        }
    }
}

impl<N: Scalar, R: Dim, C: Dim, S: OwnedStorage<N, R, C>> Matrix<N, R, C, S>
    // XXX: see the rust issue #26026
    where S::Alloc: OwnedAllocator<N, R, C, S> {

    /// Extracts a slice containing the entire matrix entries orderd column-by-columns.
    #[inline]
    pub fn as_slice(&self) -> &[N] {
        self.data.as_slice()
    }

    /// Extracts a mutable slice containing the entire matrix entries orderd column-by-columns.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [N] {
        self.data.as_mut_slice()
    }

    /// Returns a matrix containing the result of `f` applied to each of its entries.
    #[inline]
    pub fn map<F: FnMut(N) -> N>(&self, mut f: F) -> Matrix<N, R, C, S> {
        let shape = self.data.shape();

        let mut res: Matrix<N, R, C, S>;
        res = unsafe { Self::new_uninitialized_generic(shape.0, shape.1) };

        for i in 0 .. shape.0.value() * shape.1.value() {
            unsafe {
                let a = *self.data.get_unchecked_linear(i);
                *res.data.get_unchecked_linear_mut(i) = f(a)
            }
        }

        res
    }

    /// Returns a matrix containing the result of `f` applied to each entries of `self` and
    /// `rhs`.
    #[inline]
    pub fn zip_map<F: FnMut(N, N) -> N>(&self, rhs: &Matrix<N, R, C, S>, mut f: F) -> Matrix<N, R, C, S> {
        let shape_generic = self.data.shape();
        let shape = self.shape();

        let mut res: Matrix<N, R, C, S>;
        res = unsafe { Self::new_uninitialized_generic(shape_generic.0, shape_generic.1) };

        assert!(shape == rhs.shape(), "Matrix simultaneous traversal error: dimension mismatch.");

        for i in 0 .. shape.0 * shape.1 {
            unsafe {
                let a = *self.data.get_unchecked_linear(i);
                let b = *rhs.data.get_unchecked_linear(i);
                *res.data.get_unchecked_linear_mut(i) = f(a, b)
            }
        }

        res
    }
}

impl<N: Scalar, R: Dim, C: Dim, S: Storage<N, R, C>> Matrix<N, R, C, S>
    where S::Alloc: Allocator<N, C, R> {
    /// Transposes `self`.
    #[inline]
    pub fn transpose(&self) -> OwnedMatrix<N, C, R, S::Alloc> {
        let (nrows, ncols) = self.data.shape();

        unsafe {
            let mut res: OwnedMatrix<N, C, R, S::Alloc> = Matrix::new_uninitialized_generic(ncols, nrows);
            for i in 0 .. nrows.value() {
                for j in 0 .. ncols.value() {
                    *res.get_unchecked_mut(j, i) = *self.get_unchecked(i, j);
                }
            }

            res
        }
    }
}

impl<N: Scalar, D: Dim, S: StorageMut<N, D, D>> Matrix<N, D, D, S> {
    /// Transposes the square matrix `self` in-place.
    pub fn transpose_mut(&mut self) {
        assert!(self.is_square(), "Unable to transpose a non-square matrix in-place.");

        let dim = self.shape().0;

        for i in 1 .. dim {
            for j in 0 .. i {
                unsafe { self.swap_unchecked((i, j), (j, i)) }
            }
        }
    }
}

impl<N, D: Dim, S> SquareMatrix<N, D, S>
    where N: Scalar,
          S: Storage<N, D, D>,
          S::Alloc: Allocator<N, D, U1> {
    /// Creates a square matrix with its diagonal set to `diag` and all other entries set to 0.
    #[inline]
    pub fn diagonal(&self) -> OwnedColumnVector<N, D, S::Alloc> {
        assert!(self.is_square(), "Unable to get the diagonal of a non-square.");

        let dim = self.data.shape().0;
        let mut res = unsafe { OwnedColumnVector::<N, D, S::Alloc>::new_uninitialized_generic(dim, U1) };

        for i in 0 .. dim.value() {
            unsafe { *res.get_unchecked_mut(i, 0) = *self.get_unchecked(i, i); }
        }

        res
    }

    /// Computes a trace of a square matrix, i.e., the sum of its diagonal elements.
    #[inline]
    pub fn trace(&self) -> N
        where N: Ring {
        assert!(self.is_square(), "Cannot compute the trace of non-square matrix.");

        let dim = self.data.shape().0;
        let mut res = N::zero();

        for i in 0 .. dim.value() {
            res += unsafe { *self.get_unchecked(i, i) };
        }

        res

    }
}

impl<N, D, S> ColumnVector<N, D, S>
    where N: Scalar + Zero,
          D: DimAdd<U1>,
          S: Storage<N, D, U1>,
          S::Alloc: Allocator<N, DimSum<D, U1>, U1> {
    /// Computes the coordinates in projective space of this vector, i.e., appends a `0` to its
    /// coordinates.
    #[inline]
    pub fn to_homogeneous(&self) -> OwnedColumnVector<N, DimSum<D, U1>, S::Alloc> {
        let len = self.len();
        let hnrows = DimSum::<D, U1>::from_usize(len + 1);
        let mut res = unsafe { OwnedColumnVector::<N, _, S::Alloc>::new_uninitialized_generic(hnrows, U1) };
        res.generic_slice_mut((0, 0), self.data.shape()).copy_from(self);
        res[(len, 0)] = N::zero();

        res
    }

    /// Constructs a vector from coordinates in projective space, i.e., removes a `0` at the end of
    /// `self`. Returns `None` if this last component is not zero.
    #[inline]
    pub fn from_homogeneous<SB>(v: ColumnVector<N, DimSum<D, U1>, SB>) -> Option<OwnedColumnVector<N, D, S::Alloc>>
        where SB: Storage<N, DimSum<D, U1>, U1, Alloc = S::Alloc> {
        if v[v.len() - 1].is_zero() {
            let nrows = D::from_usize(v.len() - 1);
            Some(v.generic_slice((0, 0), (nrows, U1)).into_owned())
        }
        else {
            None
        }
    }
}


// // /*
// //  *
// //  * Conversions (AsRef, AsMut, From)
// //  *
// //  */
// // impl<N, R, C, A> FromIterator for Matrix<N, R, C, A>
// //     where N: Scalar + Rand,
// //           R: Dim,
// //           C: Dim,
// //           A: Allocator<N, R, C> {
// //     #[inline]
// //     fn from_iter<I: IntoIterator<Item = N>>(iter: I) -> Matrix<N, R, C, A> {
// //         let mut iter = iter.into_iter();
// //     }
// // }
// //
// //         impl<N> AsRef<[[N; $dimension]; $dimension]> for $t<N> {
// //             #[inline]
// //             fn as_ref(&self) -> &[[N; $dimension]; $dimension] {
// //                 unsafe {
// //                     mem::transmute(self)
// //                 }
// //             }
// //         }
//
// //         impl<N> AsMut<[[N; $dimension]; $dimension]> for $t<N> {
// //             #[inline]
// //             fn as_mut(&mut self) -> &mut [[N; $dimension]; $dimension] {
// //                 unsafe {
// //                     mem::transmute(self)
// //                 }
// //             }
// //         }
//
// //         impl<'a, N> From<&'a [[N; $dimension]; $dimension]> for &'a $t<N> {
// //             #[inline]
// //             fn from(arr: &'a [[N; $dimension]; $dimension]) -> &'a $t<N> {
// //                 unsafe {
// //                     mem::transmute(arr)
// //                 }
// //             }
// //         }
//
// //         impl<'a, N> From<&'a mut [[N; $dimension]; $dimension]> for &'a mut $t<N> {
// //             #[inline]
// //             fn from(arr: &'a mut [[N; $dimension]; $dimension]) -> &'a mut $t<N> {
// //                 unsafe {
// //                     mem::transmute(arr)
// //                 }
// //             }
// //         }
//
// //         impl<'a, N: Clone> From<&'a [[N; $dimension]; $dimension]> for $t<N> {
// //             #[inline]
// //             fn from(arr: &'a [[N; $dimension]; $dimension]) -> $t<N> {
// //                 let tref: &$t<N> = From::from(arr);
// //                 tref.clone()
// //             }
// //         }
//
// //         impl<N: Field + Copy> MatrixEdit for $t<N> {
// //             type RowSlice    = $dvector<N>;
// //             type ColumnSlice = $dvector<N>;
// //             type MinorMatrix = $tsmaller<N>;
// //
// //             #[inline]
// //             fn column_slice(&self, cid: usize, rstart: usize, rend: usize) -> Self::ColumnSlice {
// //                 let column = self.column(cid);
// //
// //                 $dvector::from_slice(rend - rstart, &column.as_ref()[rstart .. rend])
// //             }
//
// //             #[inline]
// //             fn row_slice(&self, rid: usize, cstart: usize, cend: usize) -> Self::RowSlice {
// //                 let row = self.row(rid);
// //
// //                 $dvector::from_slice(cend - cstart, &row.as_ref()[cstart .. cend])
// //             }
//
// //             // FIXME: optimize that (+ this is a Copy/paste from dmatrix).
// //             #[inline]
// //             fn delete_row_column(&self, row_id: usize, column_id: usize) -> Self::MinorMatrix {
// //                 assert!(row_id < $dimension && column_id < $dimension);
// //
// //                 unsafe {
// //                     let mut res = $tsmaller::new_uninitialized_generic($dimension - 1, $dimension - 1);
// //
// //                     for irow in 0 .. row_id {
// //                         for icol in 0 .. column_id {
// //                             res.unsafe_set((irow, icol), self.unsafe_at((irow, icol)))
// //                         }
// //
// //                         for icol in column_id + 1 .. $dimension {
// //                             res.unsafe_set((irow, icol - 1), self.unsafe_at((irow, icol)))
// //                         }
// //                     }
// //
// //                     for irow in row_id + 1 .. $dimension {
// //                         for icol in 0 .. column_id {
// //                             res.unsafe_set((irow - 1, icol), self.unsafe_at((irow, icol)))
// //                         }
// //
// //                         for icol in column_id + 1 .. $dimension {
// //                             res.unsafe_set((irow - 1, icol - 1), self.unsafe_at((irow, icol)))
// //                         }
// //                     }
// //
// //                     res
// //                 }
// //             }
//
// //             // FIXME: optimize that (+ this is a Copy/paste from dmatrix).
// //             #[inline]
// //             fn swap_rows(&mut self, row_id1: usize, row_id2: usize) {
// //                 if row_id1 != row_id2 {
// //                     assert!(row_id1 < $dimension && row_id2 < $dimension);
// //
// //                     for icol in 0 .. $dimension {
// //                         self.swap((row_id1, icol), (row_id2, icol))
// //                     }
// //                 }
// //             }
// //
// //             // FIXME: optimize that (+ this is a Copy/paste from dmatrix).
// //             #[inline]
// //             fn swap_columns(&mut self, column_id1: usize, column_id2: usize) {
// //                 if column_id1 != column_id2 {
// //                     assert!(column_id1 < $dimension && column_id2 < $dimension);
// //
// //                     for irow in 0 .. $dimension {
// //                         self.swap((irow, column_id1), (irow, column_id2))
// //                     }
// //                 }
// //             }
// //         }
// //
// //         /*
// //          *
// //          * Mean
// //          *
// //          */
// //         impl<N: Field  + Copy+ SupersetOf<f64>> Mean<$vector<N>> for $t<N> {
// //             fn mean(&self) -> $vector<N> {
// //                 let mut res: $vector<N> = ::zero();
// //                 let normalizer: N  = ::convert(1.0f64 / $dimension as f64);
// //
// //                 for i in 0 .. $dimension {
// //                     for j in 0 .. $dimension {
// //                         unsafe {
// //                             let acc = res.unsafe_at(j) + self.unsafe_at((i, j)) * normalizer;
// //                             res.unsafe_set(j, acc);
// //                         }
// //                     }
// //                 }
// //
// //                 res
// //             }
// //         }
// //
// //         /*
// //          *
// //          * Componentwise unary operations.
// //          *
// //          */
// //         componentwise_absolute!($t, $($compN),+);
// //     )
// // );
//
//
// // FIXME: specialize for row-major/column major
// //
//
// // macro_rules! to_homogeneous_impl(
// //   ($t: ident, $t2: ident, $dimension: expr, $dim2: expr) => (
// //     impl<N: Field + Copy> ToHomogeneous<$t2<N>> for $t<N> {
// //         #[inline]
// //         fn to_homogeneous(&self) -> $t2<N> {
// //             let mut res: $t2<N> = ::one();
// //
// //             for i in 0 .. $dimension {
// //                 for j in 0 .. $dimension {
// //                     res[(i, j)] = self[(i, j)]
// //                 }
// //             }
// //
// //             res
// //         }
// //     }
// //   )
// // );
//
// // macro_rules! from_homogeneous_impl(
// //   ($t: ident, $t2: ident, $dimension: expr, $dim2: expr) => (
// //     impl<N: Field + Copy> FromHomogeneous<$t2<N>> for $t<N> {
// //         #[inline]
// //         fn from(m: &$t2<N>) -> $t<N> {
// //             let mut res: $t<N> = ::one();
// //
// //             for i in 0 .. $dimension {
// //                 for j in 0 .. $dimension {
// //                     res[(i, j)] = m[(i, j)]
// //                 }
// //             }
// //
// //             // FIXME: do we have to deal the lost components
// //             // (like if the 1 is not a 1… do we have to divide?)
// //
// //             res
// //         }
// //     }
// //   )
// // );
//
//
// // macro_rules! eigen_qr_impl(
// //     ($t: ident, $v: ident) => (
// //         impl<N: Real> EigenQR for $t<N> {
// //             fn eigen_qr(&self, eps: N, niter: usize) -> ($t<N>, $v<N>) {
// //                 linalg::eigen_qr(self, eps, niter)
// //             }
// //         }
// //     )
// // );


impl<N, R: Dim, C: Dim, S> ApproxEq for Matrix<N, R, C, S>
    where N: Scalar + ApproxEq,
          S: Storage<N, R, C>,
          N::Epsilon: Copy {
    type Epsilon = N::Epsilon;

    #[inline]
    fn default_epsilon() -> Self::Epsilon {
        N::default_epsilon()
    }

    #[inline]
    fn default_max_relative() -> Self::Epsilon {
        N::default_max_relative()
    }

    #[inline]
    fn default_max_ulps() -> u32 {
        N::default_max_ulps()
    }

    #[inline]
    fn relative_eq(&self, other: &Self, epsilon: Self::Epsilon, max_relative: Self::Epsilon) -> bool {
        self.relative_eq(other, epsilon, max_relative)
    }

    #[inline]
    fn ulps_eq(&self, other: &Self, epsilon: Self::Epsilon, max_ulps: u32) -> bool {
       assert!(self.shape() == other.shape());
        self.iter().zip(other.iter()).all(|(a, b)| a.ulps_eq(b, epsilon, max_ulps))
    }
}

impl<N, R: Dim, C: Dim, S> PartialOrd for Matrix<N, R, C, S>
    where N: Scalar + PartialOrd,
          S: Storage<N, R, C,> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        assert!(self.shape() == other.shape(), "Matrix comparison error: dimensions mismatch.");

        let first_ord = unsafe { self.data.get_unchecked_linear(0).partial_cmp(other.data.get_unchecked_linear(0)) };

        if let Some(mut first_ord) = first_ord {
            let mut it = self.iter().zip(other.iter());
            let _ = it.next(); // Drop the first elements (we already tested it).

            for (left, right) in it {
                if let Some(ord) = left.partial_cmp(right) {
                    match ord {
                        Ordering::Equal => { /* Does not change anything. */},
                        Ordering::Less  => {
                            if first_ord == Ordering::Greater {
                                return None;
                            }
                            first_ord = ord
                        },
                        Ordering::Greater => {
                            if first_ord == Ordering::Less {
                                return None;
                            }
                            first_ord = ord
                        },
                    }
                }
                else {
                    return None
                }
            }
        }

        None
    }

    #[inline]
    fn lt(&self, right: &Self) -> bool {
        assert!(self.shape() == right.shape(), "Matrix comparison error: dimensions mismatch.");
        self.iter().zip(right.iter()).all(|(a, b)| a.lt(b))
    }

    #[inline]
    fn le(&self, right: &Self) -> bool {
        assert!(self.shape() == right.shape(), "Matrix comparison error: dimensions mismatch.");
        self.iter().zip(right.iter()).all(|(a, b)| a.le(b))
    }

    #[inline]
    fn gt(&self, right: &Self) -> bool {
        assert!(self.shape() == right.shape(), "Matrix comparison error: dimensions mismatch.");
        self.iter().zip(right.iter()).all(|(a, b)| a.gt(b))
    }

    #[inline]
    fn ge(&self, right: &Self) -> bool {
        assert!(self.shape() == right.shape(), "Matrix comparison error: dimensions mismatch.");
        self.iter().zip(right.iter()).all(|(a, b)| a.ge(b))
    }
}

impl<N, R: Dim, C: Dim, S> Eq for Matrix<N, R, C, S>
    where N: Scalar + Eq,
          S: Storage<N, R, C> { }

impl<N, R: Dim, C: Dim, S> PartialEq for Matrix<N, R, C, S>
    where N: Scalar,
          S: Storage<N, R, C> {
    #[inline]
    fn eq(&self, right: &Matrix<N, R, C, S>) -> bool {
        assert!(self.shape() == right.shape(), "Matrix equality test dimension mismatch.");
        self.iter().zip(right.iter()).all(|(l, r)| l == r)
    }
}


// FIXME: the bounds are much too restrictive here! This won't even work for, e.g.,
// integer-valued matrices...
impl<N, R: Dim, C: Dim, S> fmt::Display for Matrix<N, R, C, S>
    where N: Real + fmt::Display,
          S: Storage<N, R, C>,
          S::Alloc: Allocator<usize, R, C> {
    // XXX: will not always work correctly due to rounding errors.
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fn integral_length<N: Real>(val: &N) -> usize {
            let mut res = 1;
            let mut curr: N = ::convert(10.0f64);

            while curr <= *val {
                curr = curr * ::convert(10.0f64);
                res += 1;
            }

            if val.is_sign_negative() {
                res + 1
            }
            else {
                res
            }
        }

        let (nrows, ncols) = self.data.shape();
        let mut max_decimal_length = 0;
        let mut decimal_lengths: MatrixWithScalar<N, usize, R, C, S> =
            Matrix::from_element_generic(nrows, ncols, 0);
        let (nrows, ncols) = self.shape();

        for i in 0 .. nrows {
            for j in 0 .. ncols {
                decimal_lengths[(i, j)] = integral_length(&self[(i, j)]);
                max_decimal_length = ::max(max_decimal_length, decimal_lengths[(i, j)]);
            }
        }

        let precision = f.precision().unwrap_or(3);
        let max_number_length = max_decimal_length + precision + 1;

        try!(writeln!(f, "  ┌ {:>width$} ┐", "", width = max_number_length * ncols + ncols - 1));

        for i in 0 .. nrows {
            try!(write!(f, "  │"));
            for j in 0 .. ncols {
                let number_length = decimal_lengths[(i, j)] + precision + 1;
                let pad = max_number_length - number_length;
                try!(write!(f, " {:>thepad$}", "", thepad = pad));
                try!(write!(f, "{:.*}", precision, (*self)[(i, j)]));
            }
            try!(writeln!(f, " │"));
        }

        writeln!(f, "  └ {:>width$} ┘", "", width = max_number_length * ncols + ncols - 1)
    }
}


impl<N, R: Dim, C: Dim, S> Matrix<N, R, C, S>
    where N: Scalar + Ring,
          S: Storage<N, R, C> {
    /// The dot product between two matrices (seen as vectors).
    #[inline]
    pub fn dot<R2: Dim, C2: Dim, SB>(&self, other: &Matrix<N, R2, C2, SB>) -> N
        where SB: Storage<N, R2, C2>,
              ShapeConstraint: SameNumberOfRows<R, R2> + SameNumberOfColumns<C, C2> {
        assert!(self.shape() == other.shape(), "Dot product dimension mismatch.");
        self.iter().zip(other.iter()).fold(N::zero(), |acc, (a, b)| acc + *a * *b)
    }

    // FIXME: we could specialize this for when we only have vectors in which case we can just use
    // `iter().zip(iter())` as for the regular `.dot` method.
    /// The dot product between the transpose of `self` and `other`.
    #[inline]
    pub fn tr_dot<R2: Dim, C2: Dim, SB>(&self, other: &Matrix<N, R2, C2, SB>) -> N
        where SB: Storage<N, R2, C2>,
              ShapeConstraint: SameNumberOfRows<C, R2> + SameNumberOfColumns<R, C2> {
        let (nrows, ncols) = self.shape();
        assert!((ncols, nrows) == other.shape(), "Dot product dimension mismatch.");

        let mut res = N::zero();

        for i in 0 .. nrows {
            for j in 0 .. ncols {
                unsafe {
                    res += *self.get_unchecked(i, j) * *other.get_unchecked(j, i);
                }
            }
        }

        res
    }


    /// The squared L2 norm of this matrix.
    #[inline]
    pub fn norm_squared(&self) -> N {
        self.dot(self)
    }

    /// The perpendicular product between two 2D column vectors, i.e. `a.x * b.y - a.y * b.x`.
    #[inline]
    pub fn perp<R2, C2, SB>(&self, b: &Matrix<N, R2, C2, SB>) -> N
        where R2: Dim, C2: Dim,
              SB: Storage<N, R2, C2>,
              ShapeConstraint: SameNumberOfRows<R, U2>    +
                               SameNumberOfColumns<C, U1> +
                               SameNumberOfRows<R2, U2>   +
                               SameNumberOfColumns<C2, U1> {
        assert!(self.shape() == (2, 1), "2D perpendicular product ");

        unsafe {
            *self.get_unchecked(0, 0) * *b.get_unchecked(1, 0) -
            *self.get_unchecked(1, 0) * *b.get_unchecked(0, 0)
        }
    }

    // FIXME: use specialization instead of an assertion.
    /// The 3D cross product between two vectors.
    ///
    /// Panics if the shape is not 3D vector. In the future, this will be implemented only for
    /// dynamically-sized matrices and statically-sized 3D matrices.
    #[inline]
    pub fn cross<R2, C2, SB>(&self, b: &Matrix<N, R2, C2, SB>) -> MatrixCross<N, R, C, R2, C2, S>
        where R2: Dim, C2: Dim,
              SB: Storage<N, R2, C2>,
              S::Alloc: SameShapeAllocator<N, R, C, R2, C2, S>,
              ShapeConstraint: SameNumberOfRows<R, R2> + SameNumberOfColumns<C, C2> {
        let shape = self.shape();
        assert!(shape == b.shape(), "Vector cross product dimension mismatch.");
        assert!((shape.0 == 3 && shape.1 == 1) || (shape.0 == 1 && shape.1 == 3),
                "Vector cross product dimension mismatch.");

        if shape.0 == 3 {
            unsafe {
                // FIXME: soooo ugly!
                let nrows = SameShapeR::<R, R2>::from_usize(3);
                let ncols = SameShapeC::<C, C2>::from_usize(1);
                let mut res = Matrix::new_uninitialized_generic(nrows, ncols);

                let ax = *self.get_unchecked(0, 0);
                let ay = *self.get_unchecked(1, 0);
                let az = *self.get_unchecked(2, 0);

                let bx = *b.get_unchecked(0, 0);
                let by = *b.get_unchecked(1, 0);
                let bz = *b.get_unchecked(2, 0);

                *res.get_unchecked_mut(0, 0) = ay * bz - az * by;
                *res.get_unchecked_mut(1, 0) = az * bx - ax * bz;
                *res.get_unchecked_mut(2, 0) = ax * by - ay * bx;

                res
            }
        }
        else {
            unsafe {
                // FIXME: soooo ugly!
                let nrows = SameShapeR::<R, R2>::from_usize(1);
                let ncols = SameShapeC::<C, C2>::from_usize(3);
                let mut res = Matrix::new_uninitialized_generic(nrows, ncols);

                let ax = *self.get_unchecked(0, 0);
                let ay = *self.get_unchecked(0, 1);
                let az = *self.get_unchecked(0, 2);

                let bx = *b.get_unchecked(0, 0);
                let by = *b.get_unchecked(0, 1);
                let bz = *b.get_unchecked(0, 2);

                *res.get_unchecked_mut(0, 0) = ay * bz - az * by;
                *res.get_unchecked_mut(0, 1) = az * bx - ax * bz;
                *res.get_unchecked_mut(0, 2) = az * bx - ax * bz;

                res
            }
        }
    }
}

impl<N, R: Dim, C: Dim, S> Matrix<N, R, C, S>
    where N: Real,
          S: Storage<N, R, C> {
    /// The smallest angle between two matrices seen as vectors.
    #[inline]
    pub fn angle<R2: Dim, C2: Dim, SB>(&self, other: &Matrix<N, R2, C2, SB>) -> N
        where SB: Storage<N, R2, C2>,
              ShapeConstraint: SameNumberOfRows<R, R2> + SameNumberOfColumns<C, C2> {
        let prod = self.dot(other);
        let n1   = self.norm();
        let n2   = other.norm();

        if n1.is_zero() || n2.is_zero() {
            N::zero()
        }
        else {
            let cang = prod / (n1 * n2);

            if cang > N::one() {
                N::zero()
            }
            else if cang < -N::one() {
                N::pi()
            }
            else {
                cang.acos()
            }
        }
    }

    /// The L2 norm of this matrix.
    #[inline]
    pub fn norm(&self) -> N {
        self.norm_squared().sqrt()
    }

    /// Returns a normalized version of this matrix.
    #[inline]
    pub fn normalize(&self) -> OwnedMatrix<N, R, C, S::Alloc> {
        self / self.norm()
    }

    /// Returns a normalized version of this matrix unless its norm as smaller or equal to `eps`.
    #[inline]
    pub fn try_normalize(&self, min_norm: N) -> Option<OwnedMatrix<N, R, C, S::Alloc>> {
        let n = self.norm();

        if n <= min_norm {
            None
        }
        else {
            Some(self / n)
        }
    }
}

impl<N, R: Dim, C: Dim, S> Matrix<N, R, C, S>
    where N: Real,
          S: StorageMut<N, R, C> {
    /// Normalizes this matrix in-place and returns its norm.
    #[inline]
    pub fn normalize_mut(&mut self) -> N {
        let n = self.norm();
        *self /= n;

        n
    }

    /// Normalizes this matrix in-place or does nothing if its norm is smaller or equal to `eps`.
    ///
    /// If the normalization succeded, returns the old normal of this matrix.
    #[inline]
    pub fn try_normalize_mut(&mut self, min_norm: N) -> Option<N> {
        let n = self.norm();

        if n <= min_norm {
            None
        }
        else {
            *self /= n;
            Some(n)
        }
    }
}

impl<N, R: Dim, C: Dim, S> ApproxEq for Unit<Matrix<N, R, C, S>>
    where N: Scalar + ApproxEq,
          S: Storage<N, R, C>,
          N::Epsilon: Copy {
    type Epsilon = N::Epsilon;

    #[inline]
    fn default_epsilon() -> Self::Epsilon {
        N::default_epsilon()
    }

    #[inline]
    fn default_max_relative() -> Self::Epsilon {
        N::default_max_relative()
    }

    #[inline]
    fn default_max_ulps() -> u32 {
        N::default_max_ulps()
    }

    #[inline]
    fn relative_eq(&self, other: &Self, epsilon: Self::Epsilon, max_relative: Self::Epsilon) -> bool {
        self.as_ref().relative_eq(other.as_ref(), epsilon, max_relative)
    }

    #[inline]
    fn ulps_eq(&self, other: &Self, epsilon: Self::Epsilon, max_ulps: u32) -> bool {
        self.as_ref().ulps_eq(other.as_ref(), epsilon, max_ulps)
    }
}
