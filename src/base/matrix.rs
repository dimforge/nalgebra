use num::Zero;
use num_complex::Complex;
#[cfg(feature = "abomonation-serialize")]
use std::io::{Result as IOResult, Write};

use approx::{AbsDiffEq, RelativeEq, UlpsEq};
use std::any::TypeId;
use std::cmp::Ordering;
use std::fmt;
use std::marker::PhantomData;
use std::mem;

#[cfg(feature = "serde-serialize")]
use serde::{Deserialize, Deserializer, Serialize, Serializer};

#[cfg(feature = "abomonation-serialize")]
use abomonation::Abomonation;

use alga::general::{Real, Ring};

use base::allocator::{Allocator, SameShapeAllocator, SameShapeC, SameShapeR};
use base::constraint::{DimEq, SameNumberOfColumns, SameNumberOfRows, ShapeConstraint};
use base::dimension::{Dim, DimAdd, DimSum, U1, U2, U3};
use base::iter::{MatrixIter, MatrixIterMut};
use base::storage::{
    ContiguousStorage, ContiguousStorageMut, Owned, SameShapeStorage, Storage, StorageMut,
};
use base::{DefaultAllocator, MatrixMN, MatrixN, Scalar, Unit, VectorN};

/// A square matrix.
pub type SquareMatrix<N, D, S> = Matrix<N, D, D, S>;

/// A matrix with one column and `D` rows.
pub type Vector<N, D, S> = Matrix<N, D, U1, S>;

/// A matrix with one row and `D` columns .
pub type RowVector<N, D, S> = Matrix<N, U1, D, S>;

/// The type of the result of a matrix sum.
pub type MatrixSum<N, R1, C1, R2, C2> =
    Matrix<N, SameShapeR<R1, R2>, SameShapeC<C1, C2>, SameShapeStorage<N, R1, C1, R2, C2>>;

/// The type of the result of a matrix sum.
pub type VectorSum<N, R1, R2> =
    Matrix<N, SameShapeR<R1, R2>, U1, SameShapeStorage<N, R1, U1, R2, U1>>;

/// The type of the result of a matrix cross product.
pub type MatrixCross<N, R1, C1, R2, C2> =
    Matrix<N, SameShapeR<R1, R2>, SameShapeC<C1, C2>, SameShapeStorage<N, R1, C1, R2, C2>>;

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
#[derive(Hash, Clone, Copy)]
pub struct Matrix<N: Scalar, R: Dim, C: Dim, S> {
    /// The data storage that contains all the matrix components and informations about its number
    /// of rows and column (if needed).
    pub data: S,

    _phantoms: PhantomData<(N, R, C)>,
}

impl<N: Scalar, R: Dim, C: Dim, S: fmt::Debug> fmt::Debug for Matrix<N, R, C, S> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        formatter
            .debug_struct("Matrix")
            .field("data", &self.data)
            .finish()
    }
}

#[cfg(feature = "serde-serialize")]
impl<N, R, C, S> Serialize for Matrix<N, R, C, S>
where
    N: Scalar,
    R: Dim,
    C: Dim,
    S: Serialize,
{
    fn serialize<T>(&self, serializer: T) -> Result<T::Ok, T::Error>
    where
        T: Serializer,
    {
        self.data.serialize(serializer)
    }
}

#[cfg(feature = "serde-serialize")]
impl<'de, N, R, C, S> Deserialize<'de> for Matrix<N, R, C, S>
where
    N: Scalar,
    R: Dim,
    C: Dim,
    S: Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        S::deserialize(deserializer).map(|x| Matrix {
            data: x,
            _phantoms: PhantomData,
        })
    }
}

#[cfg(feature = "abomonation-serialize")]
impl<N: Scalar, R: Dim, C: Dim, S: Abomonation> Abomonation for Matrix<N, R, C, S> {
    unsafe fn entomb<W: Write>(&self, writer: &mut W) -> IOResult<()> {
        self.data.entomb(writer)
    }

    unsafe fn exhume<'a, 'b>(&'a mut self, bytes: &'b mut [u8]) -> Option<&'b mut [u8]> {
        self.data.exhume(bytes)
    }

    fn extent(&self) -> usize {
        self.data.extent()
    }
}

impl<N: Scalar, R: Dim, C: Dim, S> Matrix<N, R, C, S> {
    /// Creates a new matrix with the given data without statically checking that the matrix
    /// dimension matches the storage dimension.
    #[inline]
    pub unsafe fn from_data_statically_unchecked(data: S) -> Matrix<N, R, C, S> {
        Matrix {
            data: data,
            _phantoms: PhantomData,
        }
    }
}

impl<N: Scalar, R: Dim, C: Dim, S: Storage<N, R, C>> Matrix<N, R, C, S> {
    /// Creates a new matrix with the given data.
    #[inline]
    pub fn from_data(data: S) -> Matrix<N, R, C, S> {
        unsafe { Self::from_data_statically_unchecked(data) }
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
        } else if ncols == 1 {
            (i, 0)
        } else {
            (i % nrows, i / nrows)
        }
    }

    /// Gets a reference to the element of this matrix at row `irow` and column `icol` without
    /// bound-checking.
    #[inline]
    pub unsafe fn get_unchecked(&self, irow: usize, icol: usize) -> &N {
        debug_assert!(
            irow < self.nrows() && icol < self.ncols(),
            "Matrix index out of bounds."
        );
        self.data.get_unchecked(irow, icol)
    }

    /// Tests whether `self` and `rhs` are equal up to a given epsilon.
    ///
    /// See `relative_eq` from the `RelativeEq` trait for more details.
    #[inline]
    pub fn relative_eq<R2, C2, SB>(
        &self,
        other: &Matrix<N, R2, C2, SB>,
        eps: N::Epsilon,
        max_relative: N::Epsilon,
    ) -> bool
    where
        N: RelativeEq,
        R2: Dim,
        C2: Dim,
        SB: Storage<N, R2, C2>,
        N::Epsilon: Copy,
        ShapeConstraint: SameNumberOfRows<R, R2> + SameNumberOfColumns<C, C2>,
    {
        assert!(self.shape() == other.shape());
        self.iter()
            .zip(other.iter())
            .all(|(a, b)| a.relative_eq(b, eps, max_relative))
    }

    /// Tests whether `self` and `rhs` are exactly equal.
    #[inline]
    pub fn eq<R2, C2, SB>(&self, other: &Matrix<N, R2, C2, SB>) -> bool
    where
        N: PartialEq,
        R2: Dim,
        C2: Dim,
        SB: Storage<N, R2, C2>,
        ShapeConstraint: SameNumberOfRows<R, R2> + SameNumberOfColumns<C, C2>,
    {
        assert!(self.shape() == other.shape());
        self.iter().zip(other.iter()).all(|(a, b)| *a == *b)
    }

    /// Moves this matrix into one that owns its data.
    #[inline]
    pub fn into_owned(self) -> MatrixMN<N, R, C>
    where
        DefaultAllocator: Allocator<N, R, C>,
    {
        Matrix::from_data(self.data.into_owned())
    }

    // FIXME: this could probably benefit from specialization.
    // XXX: bad name.
    /// Moves this matrix into one that owns its data. The actual type of the result depends on
    /// matrix storage combination rules for addition.
    #[inline]
    pub fn into_owned_sum<R2, C2>(self) -> MatrixSum<N, R, C, R2, C2>
    where
        R2: Dim,
        C2: Dim,
        DefaultAllocator: SameShapeAllocator<N, R, C, R2, C2>,
        ShapeConstraint: SameNumberOfRows<R, R2> + SameNumberOfColumns<C, C2>,
    {
        if TypeId::of::<SameShapeStorage<N, R, C, R2, C2>>() == TypeId::of::<Owned<N, R, C>>() {
            // We can just return `self.into_owned()`.

            unsafe {
                // FIXME: check that those copies are optimized away by the compiler.
                let owned = self.into_owned();
                let res = mem::transmute_copy(&owned);
                mem::forget(owned);
                res
            }
        } else {
            self.clone_owned_sum()
        }
    }

    /// Clones this matrix to one that owns its data.
    #[inline]
    pub fn clone_owned(&self) -> MatrixMN<N, R, C>
    where
        DefaultAllocator: Allocator<N, R, C>,
    {
        Matrix::from_data(self.data.clone_owned())
    }

    /// Clones this matrix into one that owns its data. The actual type of the result depends on
    /// matrix storage combination rules for addition.
    #[inline]
    pub fn clone_owned_sum<R2, C2>(&self) -> MatrixSum<N, R, C, R2, C2>
    where
        R2: Dim,
        C2: Dim,
        DefaultAllocator: SameShapeAllocator<N, R, C, R2, C2>,
        ShapeConstraint: SameNumberOfRows<R, R2> + SameNumberOfColumns<C, C2>,
    {
        let (nrows, ncols) = self.shape();
        let nrows: SameShapeR<R, R2> = Dim::from_usize(nrows);
        let ncols: SameShapeC<C, C2> = Dim::from_usize(ncols);

        let mut res: MatrixSum<N, R, C, R2, C2> =
            unsafe { Matrix::new_uninitialized_generic(nrows, ncols) };

        // FIXME: use copy_from
        for j in 0..res.ncols() {
            for i in 0..res.nrows() {
                unsafe {
                    *res.get_unchecked_mut(i, j) = *self.get_unchecked(i, j);
                }
            }
        }

        res
    }

    /// Returns a matrix containing the result of `f` applied to each of its entries.
    #[inline]
    pub fn map<N2: Scalar, F: FnMut(N) -> N2>(&self, mut f: F) -> MatrixMN<N2, R, C>
    where
        DefaultAllocator: Allocator<N2, R, C>,
    {
        let (nrows, ncols) = self.data.shape();

        let mut res = unsafe { MatrixMN::new_uninitialized_generic(nrows, ncols) };

        for j in 0..ncols.value() {
            for i in 0..nrows.value() {
                unsafe {
                    let a = *self.data.get_unchecked(i, j);
                    *res.data.get_unchecked_mut(i, j) = f(a)
                }
            }
        }

        res
    }

    /// Returns a matrix containing the result of `f` applied to each entries of `self` and
    /// `rhs`.
    #[inline]
    pub fn zip_map<N2, N3, S2, F>(&self, rhs: &Matrix<N2, R, C, S2>, mut f: F) -> MatrixMN<N3, R, C>
    where
        N2: Scalar,
        N3: Scalar,
        S2: Storage<N2, R, C>,
        F: FnMut(N, N2) -> N3,
        DefaultAllocator: Allocator<N3, R, C>,
    {
        let (nrows, ncols) = self.data.shape();

        let mut res = unsafe { MatrixMN::new_uninitialized_generic(nrows, ncols) };

        assert!(
            (nrows.value(), ncols.value()) == rhs.shape(),
            "Matrix simultaneous traversal error: dimension mismatch."
        );

        for j in 0..ncols.value() {
            for i in 0..nrows.value() {
                unsafe {
                    let a = *self.data.get_unchecked(i, j);
                    let b = *rhs.data.get_unchecked(i, j);
                    *res.data.get_unchecked_mut(i, j) = f(a, b)
                }
            }
        }

        res
    }

    /// Transposes `self` and store the result into `out`.
    #[inline]
    pub fn transpose_to<R2, C2, SB>(&self, out: &mut Matrix<N, R2, C2, SB>)
    where
        R2: Dim,
        C2: Dim,
        SB: StorageMut<N, R2, C2>,
        ShapeConstraint: SameNumberOfRows<R, C2> + SameNumberOfColumns<C, R2>,
    {
        let (nrows, ncols) = self.shape();
        assert!(
            (ncols, nrows) == out.shape(),
            "Incompatible shape for transpose-copy."
        );

        // FIXME: optimize that.
        for i in 0..nrows {
            for j in 0..ncols {
                unsafe {
                    *out.get_unchecked_mut(j, i) = *self.get_unchecked(i, j);
                }
            }
        }
    }

    /// Transposes `self`.
    #[inline]
    pub fn transpose(&self) -> MatrixMN<N, C, R>
    where
        DefaultAllocator: Allocator<N, C, R>,
    {
        let (nrows, ncols) = self.data.shape();

        unsafe {
            let mut res = Matrix::new_uninitialized_generic(ncols, nrows);
            self.transpose_to(&mut res);

            res
        }
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
        debug_assert!(
            irow < self.nrows() && icol < self.ncols(),
            "Matrix index out of bounds."
        );
        self.data.get_unchecked_mut(irow, icol)
    }

    /// Swaps two entries without bound-checking.
    #[inline]
    pub unsafe fn swap_unchecked(&mut self, row_cols1: (usize, usize), row_cols2: (usize, usize)) {
        debug_assert!(row_cols1.0 < self.nrows() && row_cols1.1 < self.ncols());
        debug_assert!(row_cols2.0 < self.nrows() && row_cols2.1 < self.ncols());
        self.data.swap_unchecked(row_cols1, row_cols2)
    }

    /// Swaps two entries.
    #[inline]
    pub fn swap(&mut self, row_cols1: (usize, usize), row_cols2: (usize, usize)) {
        let (nrows, ncols) = self.shape();
        assert!(
            row_cols1.0 < nrows && row_cols1.1 < ncols,
            "Matrix elements swap index out of bounds."
        );
        assert!(
            row_cols2.0 < nrows && row_cols2.1 < ncols,
            "Matrix elements swap index out of bounds."
        );
        unsafe { self.swap_unchecked(row_cols1, row_cols2) }
    }

    /// Fills this matrix with the content of a slice. Both must hold the same number of elements.
    ///
    /// The components of the slice are assumed to be ordered in column-major order.
    #[inline]
    pub fn copy_from_slice(&mut self, slice: &[N]) {
        let (nrows, ncols) = self.shape();

        assert!(
            nrows * ncols == slice.len(),
            "The slice must contain the same number of elements as the matrix."
        );

        for j in 0..ncols {
            for i in 0..nrows {
                unsafe {
                    *self.get_unchecked_mut(i, j) = *slice.get_unchecked(i + j * nrows);
                }
            }
        }
    }

    /// Fills this matrix with the content of another one. Both must have the same shape.
    #[inline]
    pub fn copy_from<R2, C2, SB>(&mut self, other: &Matrix<N, R2, C2, SB>)
    where
        R2: Dim,
        C2: Dim,
        SB: Storage<N, R2, C2>,
        ShapeConstraint: SameNumberOfRows<R, R2> + SameNumberOfColumns<C, C2>,
    {
        assert!(
            self.shape() == other.shape(),
            "Unable to copy from a matrix with a different shape."
        );

        for j in 0..self.ncols() {
            for i in 0..self.nrows() {
                unsafe {
                    *self.get_unchecked_mut(i, j) = *other.get_unchecked(i, j);
                }
            }
        }
    }

    /// Fills this matrix with the content of the transpose another one.
    #[inline]
    pub fn tr_copy_from<R2, C2, SB>(&mut self, other: &Matrix<N, R2, C2, SB>)
    where
        R2: Dim,
        C2: Dim,
        SB: Storage<N, R2, C2>,
        ShapeConstraint: DimEq<R, C2> + SameNumberOfColumns<C, R2>,
    {
        let (nrows, ncols) = self.shape();
        assert!(
            (ncols, nrows) == other.shape(),
            "Unable to copy from a matrix with incompatible shape."
        );

        for j in 0..ncols {
            for i in 0..nrows {
                unsafe {
                    *self.get_unchecked_mut(i, j) = *other.get_unchecked(j, i);
                }
            }
        }
    }

    /// Replaces each component of `self` by the result of a closure `f` applied on it.
    #[inline]
    pub fn apply<F: FnMut(N) -> N>(&mut self, mut f: F)
    where
        DefaultAllocator: Allocator<N, R, C>,
    {
        let (nrows, ncols) = self.shape();

        for j in 0..ncols {
            for i in 0..nrows {
                unsafe {
                    let e = self.data.get_unchecked_mut(i, j);
                    *e = f(*e)
                }
            }
        }
    }
}

impl<N: Scalar, D: Dim, S: Storage<N, D>> Vector<N, D, S> {
    /// Gets a reference to the i-th element of this column vector without bound checking.
    #[inline]
    pub unsafe fn vget_unchecked(&self, i: usize) -> &N {
        debug_assert!(i < self.nrows(), "Vector index out of bounds.");
        let i = i * self.strides().0;
        self.data.get_unchecked_linear(i)
    }
}

impl<N: Scalar, D: Dim, S: StorageMut<N, D>> Vector<N, D, S> {
    /// Gets a mutable reference to the i-th element of this column vector without bound checking.
    #[inline]
    pub unsafe fn vget_unchecked_mut(&mut self, i: usize) -> &mut N {
        debug_assert!(i < self.nrows(), "Vector index out of bounds.");
        let i = i * self.strides().0;
        self.data.get_unchecked_linear_mut(i)
    }
}

impl<N: Scalar, R: Dim, C: Dim, S: ContiguousStorage<N, R, C>> Matrix<N, R, C, S> {
    /// Extracts a slice containing the entire matrix entries ordered column-by-columns.
    #[inline]
    pub fn as_slice(&self) -> &[N] {
        self.data.as_slice()
    }
}

impl<N: Scalar, R: Dim, C: Dim, S: ContiguousStorageMut<N, R, C>> Matrix<N, R, C, S> {
    /// Extracts a mutable slice containing the entire matrix entries ordered column-by-columns.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [N] {
        self.data.as_mut_slice()
    }
}

impl<N: Scalar, D: Dim, S: StorageMut<N, D, D>> Matrix<N, D, D, S> {
    /// Transposes the square matrix `self` in-place.
    pub fn transpose_mut(&mut self) {
        assert!(
            self.is_square(),
            "Unable to transpose a non-square matrix in-place."
        );

        let dim = self.shape().0;

        for i in 1..dim {
            for j in 0..i {
                unsafe { self.swap_unchecked((i, j), (j, i)) }
            }
        }
    }
}

impl<N: Real, R: Dim, C: Dim, S: Storage<Complex<N>, R, C>> Matrix<Complex<N>, R, C, S> {
    /// Takes the conjugate and transposes `self` and store the result into `out`.
    #[inline]
    pub fn conjugate_transpose_to<R2, C2, SB>(&self, out: &mut Matrix<Complex<N>, R2, C2, SB>)
    where
        R2: Dim,
        C2: Dim,
        SB: StorageMut<Complex<N>, R2, C2>,
        ShapeConstraint: SameNumberOfRows<R, C2> + SameNumberOfColumns<C, R2>,
    {
        let (nrows, ncols) = self.shape();
        assert!(
            (ncols, nrows) == out.shape(),
            "Incompatible shape for transpose-copy."
        );

        // FIXME: optimize that.
        for i in 0..nrows {
            for j in 0..ncols {
                unsafe {
                    *out.get_unchecked_mut(j, i) = self.get_unchecked(i, j).conj();
                }
            }
        }
    }

    /// The conjugate transposition of `self`.
    #[inline]
    pub fn conjugate_transpose(&self) -> MatrixMN<Complex<N>, C, R>
    where
        DefaultAllocator: Allocator<Complex<N>, C, R>,
    {
        let (nrows, ncols) = self.data.shape();

        unsafe {
            let mut res: MatrixMN<_, C, R> = Matrix::new_uninitialized_generic(ncols, nrows);
            self.conjugate_transpose_to(&mut res);

            res
        }
    }
}

impl<N: Real, D: Dim, S: StorageMut<Complex<N>, D, D>> Matrix<Complex<N>, D, D, S> {
    /// Sets `self` to its conjugate transpose.
    pub fn conjugate_transpose_mut(&mut self) {
        assert!(
            self.is_square(),
            "Unable to transpose a non-square matrix in-place."
        );

        let dim = self.shape().0;

        for i in 1..dim {
            for j in 0..i {
                unsafe {
                    let ref_ij = self.get_unchecked_mut(i, j) as *mut Complex<N>;
                    let ref_ji = self.get_unchecked_mut(j, i) as *mut Complex<N>;
                    let conj_ij = (*ref_ij).conj();
                    let conj_ji = (*ref_ji).conj();
                    *ref_ij = conj_ji;
                    *ref_ji = conj_ij;
                }
            }
        }
    }
}

impl<N: Scalar, D: Dim, S: Storage<N, D, D>> SquareMatrix<N, D, S> {
    /// Creates a square matrix with its diagonal set to `diag` and all other entries set to 0.
    #[inline]
    pub fn diagonal(&self) -> VectorN<N, D>
    where
        DefaultAllocator: Allocator<N, D>,
    {
        assert!(
            self.is_square(),
            "Unable to get the diagonal of a non-square matrix."
        );

        let dim = self.data.shape().0;
        let mut res = unsafe { VectorN::new_uninitialized_generic(dim, U1) };

        for i in 0..dim.value() {
            unsafe {
                *res.vget_unchecked_mut(i) = *self.get_unchecked(i, i);
            }
        }

        res
    }

    /// Computes a trace of a square matrix, i.e., the sum of its diagonal elements.
    #[inline]
    pub fn trace(&self) -> N
    where
        N: Ring,
    {
        assert!(
            self.is_square(),
            "Cannot compute the trace of non-square matrix."
        );

        let dim = self.data.shape().0;
        let mut res = N::zero();

        for i in 0..dim.value() {
            res += unsafe { *self.get_unchecked(i, i) };
        }

        res
    }
}

impl<N: Scalar + Zero, D: DimAdd<U1>, S: Storage<N, D>> Vector<N, D, S> {
    /// Computes the coordinates in projective space of this vector, i.e., appends a `0` to its
    /// coordinates.
    #[inline]
    pub fn to_homogeneous(&self) -> VectorN<N, DimSum<D, U1>>
    where
        DefaultAllocator: Allocator<N, DimSum<D, U1>>,
    {
        let len = self.len();
        let hnrows = DimSum::<D, U1>::from_usize(len + 1);
        let mut res = unsafe { VectorN::<N, _>::new_uninitialized_generic(hnrows, U1) };
        res.generic_slice_mut((0, 0), self.data.shape())
            .copy_from(self);
        res[(len, 0)] = N::zero();

        res
    }

    /// Constructs a vector from coordinates in projective space, i.e., removes a `0` at the end of
    /// `self`. Returns `None` if this last component is not zero.
    #[inline]
    pub fn from_homogeneous<SB>(v: Vector<N, DimSum<D, U1>, SB>) -> Option<VectorN<N, D>>
    where
        SB: Storage<N, DimSum<D, U1>>,
        DefaultAllocator: Allocator<N, D>,
    {
        if v[v.len() - 1].is_zero() {
            let nrows = D::from_usize(v.len() - 1);
            Some(v.generic_slice((0, 0), (nrows, U1)).into_owned())
        } else {
            None
        }
    }
}

impl<N, R: Dim, C: Dim, S> AbsDiffEq for Matrix<N, R, C, S>
where
    N: Scalar + AbsDiffEq,
    S: Storage<N, R, C>,
    N::Epsilon: Copy,
{
    type Epsilon = N::Epsilon;

    #[inline]
    fn default_epsilon() -> Self::Epsilon {
        N::default_epsilon()
    }

    #[inline]
    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        self.iter()
            .zip(other.iter())
            .all(|(a, b)| a.abs_diff_eq(b, epsilon))
    }
}

impl<N, R: Dim, C: Dim, S> RelativeEq for Matrix<N, R, C, S>
where
    N: Scalar + RelativeEq,
    S: Storage<N, R, C>,
    N::Epsilon: Copy,
{
    #[inline]
    fn default_max_relative() -> Self::Epsilon {
        N::default_max_relative()
    }

    #[inline]
    fn relative_eq(
        &self,
        other: &Self,
        epsilon: Self::Epsilon,
        max_relative: Self::Epsilon,
    ) -> bool {
        self.relative_eq(other, epsilon, max_relative)
    }
}

impl<N, R: Dim, C: Dim, S> UlpsEq for Matrix<N, R, C, S>
where
    N: Scalar + UlpsEq,
    S: Storage<N, R, C>,
    N::Epsilon: Copy,
{
    #[inline]
    fn default_max_ulps() -> u32 {
        N::default_max_ulps()
    }

    #[inline]
    fn ulps_eq(&self, other: &Self, epsilon: Self::Epsilon, max_ulps: u32) -> bool {
        assert!(self.shape() == other.shape());
        self.iter()
            .zip(other.iter())
            .all(|(a, b)| a.ulps_eq(b, epsilon, max_ulps))
    }
}

impl<N, R: Dim, C: Dim, S> PartialOrd for Matrix<N, R, C, S>
where
    N: Scalar + PartialOrd,
    S: Storage<N, R, C>,
{
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        assert!(
            self.shape() == other.shape(),
            "Matrix comparison error: dimensions mismatch."
        );

        let first_ord = unsafe {
            self.data
                .get_unchecked_linear(0)
                .partial_cmp(other.data.get_unchecked_linear(0))
        };

        if let Some(mut first_ord) = first_ord {
            let mut it = self.iter().zip(other.iter());
            let _ = it.next(); // Drop the first elements (we already tested it).

            for (left, right) in it {
                if let Some(ord) = left.partial_cmp(right) {
                    match ord {
                        Ordering::Equal => { /* Does not change anything. */ }
                        Ordering::Less => {
                            if first_ord == Ordering::Greater {
                                return None;
                            }
                            first_ord = ord
                        }
                        Ordering::Greater => {
                            if first_ord == Ordering::Less {
                                return None;
                            }
                            first_ord = ord
                        }
                    }
                } else {
                    return None;
                }
            }
        }

        None
    }

    #[inline]
    fn lt(&self, right: &Self) -> bool {
        assert!(
            self.shape() == right.shape(),
            "Matrix comparison error: dimensions mismatch."
        );
        self.iter().zip(right.iter()).all(|(a, b)| a.lt(b))
    }

    #[inline]
    fn le(&self, right: &Self) -> bool {
        assert!(
            self.shape() == right.shape(),
            "Matrix comparison error: dimensions mismatch."
        );
        self.iter().zip(right.iter()).all(|(a, b)| a.le(b))
    }

    #[inline]
    fn gt(&self, right: &Self) -> bool {
        assert!(
            self.shape() == right.shape(),
            "Matrix comparison error: dimensions mismatch."
        );
        self.iter().zip(right.iter()).all(|(a, b)| a.gt(b))
    }

    #[inline]
    fn ge(&self, right: &Self) -> bool {
        assert!(
            self.shape() == right.shape(),
            "Matrix comparison error: dimensions mismatch."
        );
        self.iter().zip(right.iter()).all(|(a, b)| a.ge(b))
    }
}

impl<N, R: Dim, C: Dim, S> Eq for Matrix<N, R, C, S>
where
    N: Scalar + Eq,
    S: Storage<N, R, C>,
{
}

impl<N, R: Dim, C: Dim, S> PartialEq for Matrix<N, R, C, S>
where
    N: Scalar,
    S: Storage<N, R, C>,
{
    #[inline]
    fn eq(&self, right: &Matrix<N, R, C, S>) -> bool {
        assert!(
            self.shape() == right.shape(),
            "Matrix equality test dimension mismatch."
        );
        self.iter().zip(right.iter()).all(|(l, r)| l == r)
    }
}

impl<N, R: Dim, C: Dim, S> fmt::Display for Matrix<N, R, C, S>
where
    N: Scalar + fmt::Display,
    S: Storage<N, R, C>,
    DefaultAllocator: Allocator<usize, R, C>,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        #[cfg(feature = "std")]
        fn val_width<N: Scalar + fmt::Display>(val: N, f: &mut fmt::Formatter) -> usize {
            match f.precision() {
                Some(precision) => format!("{:.1$}", val, precision).chars().count(),
                None => format!("{}", val).chars().count(),
            }
        }

        #[cfg(not(feature = "std"))]
        fn val_width<N: Scalar + fmt::Display>(_: N, _: &mut fmt::Formatter) -> usize {
            4
        }

        let (nrows, ncols) = self.data.shape();

        if nrows.value() == 0 || ncols.value() == 0 {
            return write!(f, "[ ]");
        }

        let mut max_length = 0;
        let mut lengths: MatrixMN<usize, R, C> = Matrix::zeros_generic(nrows, ncols);
        let (nrows, ncols) = self.shape();

        for i in 0..nrows {
            for j in 0..ncols {
                lengths[(i, j)] = val_width(self[(i, j)], f);
                max_length = ::max(max_length, lengths[(i, j)]);
            }
        }

        let max_length_with_space = max_length + 1;

        try!(writeln!(f, ""));
        try!(writeln!(
            f,
            "  ┌ {:>width$} ┐",
            "",
            width = max_length_with_space * ncols - 1
        ));

        for i in 0..nrows {
            try!(write!(f, "  │"));
            for j in 0..ncols {
                let number_length = lengths[(i, j)] + 1;
                let pad = max_length_with_space - number_length;
                try!(write!(f, " {:>thepad$}", "", thepad = pad));
                match f.precision() {
                    Some(precision) => try!(write!(f, "{:.1$}", (*self)[(i, j)], precision)),
                    None => try!(write!(f, "{}", (*self)[(i, j)])),
                }
            }
            try!(writeln!(f, " │"));
        }

        try!(writeln!(
            f,
            "  └ {:>width$} ┘",
            "",
            width = max_length_with_space * ncols - 1
        ));
        writeln!(f, "")
    }
}

impl<N: Scalar + Ring, R: Dim, C: Dim, S: Storage<N, R, C>> Matrix<N, R, C, S> {
    /// The perpendicular product between two 2D column vectors, i.e. `a.x * b.y - a.y * b.x`.
    #[inline]
    pub fn perp<R2, C2, SB>(&self, b: &Matrix<N, R2, C2, SB>) -> N
    where
        R2: Dim,
        C2: Dim,
        SB: Storage<N, R2, C2>,
        ShapeConstraint: SameNumberOfRows<R, U2>
            + SameNumberOfColumns<C, U1>
            + SameNumberOfRows<R2, U2>
            + SameNumberOfColumns<C2, U1>,
    {
        assert!(self.shape() == (2, 1), "2D perpendicular product ");

        unsafe {
            *self.get_unchecked(0, 0) * *b.get_unchecked(1, 0)
                - *self.get_unchecked(1, 0) * *b.get_unchecked(0, 0)
        }
    }

    // FIXME: use specialization instead of an assertion.
    /// The 3D cross product between two vectors.
    ///
    /// Panics if the shape is not 3D vector. In the future, this will be implemented only for
    /// dynamically-sized matrices and statically-sized 3D matrices.
    #[inline]
    pub fn cross<R2, C2, SB>(&self, b: &Matrix<N, R2, C2, SB>) -> MatrixCross<N, R, C, R2, C2>
    where
        R2: Dim,
        C2: Dim,
        SB: Storage<N, R2, C2>,
        DefaultAllocator: SameShapeAllocator<N, R, C, R2, C2>,
        ShapeConstraint: SameNumberOfRows<R, R2> + SameNumberOfColumns<C, C2>,
    {
        let shape = self.shape();
        assert!(
            shape == b.shape(),
            "Vector cross product dimension mismatch."
        );
        assert!(
            (shape.0 == 3 && shape.1 == 1) || (shape.0 == 1 && shape.1 == 3),
            "Vector cross product dimension mismatch."
        );

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
        } else {
            unsafe {
                // FIXME: ugly!
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
                *res.get_unchecked_mut(0, 2) = ax * by - ay * bx;

                res
            }
        }
    }
}

impl<N: Real, S: Storage<N, U3>> Vector<N, U3, S>
where
    DefaultAllocator: Allocator<N, U3>,
{
    /// Computes the matrix `M` such that for all vector `v` we have `M * v == self.cross(&v)`.
    #[inline]
    pub fn cross_matrix(&self) -> MatrixN<N, U3> {
        MatrixN::<N, U3>::new(
            N::zero(),
            -self[2],
            self[1],
            self[2],
            N::zero(),
            -self[0],
            -self[1],
            self[0],
            N::zero(),
        )
    }
}

impl<N: Real, R: Dim, C: Dim, S: Storage<N, R, C>> Matrix<N, R, C, S> {
    /// The smallest angle between two vectors.
    #[inline]
    pub fn angle<R2: Dim, C2: Dim, SB>(&self, other: &Matrix<N, R2, C2, SB>) -> N
    where
        SB: Storage<N, R2, C2>,
        ShapeConstraint: DimEq<R, R2> + DimEq<C, C2>,
    {
        let prod = self.dot(other);
        let n1 = self.norm();
        let n2 = other.norm();

        if n1.is_zero() || n2.is_zero() {
            N::zero()
        } else {
            let cang = prod / (n1 * n2);

            if cang > N::one() {
                N::zero()
            } else if cang < -N::one() {
                N::pi()
            } else {
                cang.acos()
            }
        }
    }
}

impl<N: Real, R: Dim, C: Dim, S: Storage<N, R, C>> Matrix<N, R, C, S> {
    /// The squared L2 norm of this vector.
    #[inline]
    pub fn norm_squared(&self) -> N {
        let mut res = N::zero();

        for i in 0..self.ncols() {
            let col = self.column(i);
            res += col.dot(&col)
        }

        res
    }

    /// The L2 norm of this matrix.
    #[inline]
    pub fn norm(&self) -> N {
        self.norm_squared().sqrt()
    }

    /// Returns a normalized version of this matrix.
    #[inline]
    pub fn normalize(&self) -> MatrixMN<N, R, C>
    where
        DefaultAllocator: Allocator<N, R, C>,
    {
        self / self.norm()
    }

    /// Returns a normalized version of this matrix unless its norm as smaller or equal to `eps`.
    #[inline]
    pub fn try_normalize(&self, min_norm: N) -> Option<MatrixMN<N, R, C>>
    where
        DefaultAllocator: Allocator<N, R, C>,
    {
        let n = self.norm();

        if n <= min_norm {
            None
        } else {
            Some(self / n)
        }
    }
}

impl<N: Real, R: Dim, C: Dim, S: StorageMut<N, R, C>> Matrix<N, R, C, S> {
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
        } else {
            *self /= n;
            Some(n)
        }
    }
}

impl<N, R: Dim, C: Dim, S> AbsDiffEq for Unit<Matrix<N, R, C, S>>
where
    N: Scalar + AbsDiffEq,
    S: Storage<N, R, C>,
    N::Epsilon: Copy,
{
    type Epsilon = N::Epsilon;

    #[inline]
    fn default_epsilon() -> Self::Epsilon {
        N::default_epsilon()
    }

    #[inline]
    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        self.as_ref().abs_diff_eq(other.as_ref(), epsilon)
    }
}

impl<N, R: Dim, C: Dim, S> RelativeEq for Unit<Matrix<N, R, C, S>>
where
    N: Scalar + RelativeEq,
    S: Storage<N, R, C>,
    N::Epsilon: Copy,
{
    #[inline]
    fn default_max_relative() -> Self::Epsilon {
        N::default_max_relative()
    }

    #[inline]
    fn relative_eq(
        &self,
        other: &Self,
        epsilon: Self::Epsilon,
        max_relative: Self::Epsilon,
    ) -> bool {
        self.as_ref()
            .relative_eq(other.as_ref(), epsilon, max_relative)
    }
}

impl<N, R: Dim, C: Dim, S> UlpsEq for Unit<Matrix<N, R, C, S>>
where
    N: Scalar + UlpsEq,
    S: Storage<N, R, C>,
    N::Epsilon: Copy,
{
    #[inline]
    fn default_max_ulps() -> u32 {
        N::default_max_ulps()
    }

    #[inline]
    fn ulps_eq(&self, other: &Self, epsilon: Self::Epsilon, max_ulps: u32) -> bool {
        self.as_ref().ulps_eq(other.as_ref(), epsilon, max_ulps)
    }
}
