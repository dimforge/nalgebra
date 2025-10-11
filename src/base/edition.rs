use num::{One, Zero};
use std::cmp;
#[cfg(any(feature = "std", feature = "alloc"))]
use std::iter::ExactSizeIterator;
use std::ptr;

use crate::base::allocator::{Allocator, Reallocator};
use crate::base::constraint::{DimEq, SameNumberOfColumns, SameNumberOfRows, ShapeConstraint};
#[cfg(any(feature = "std", feature = "alloc"))]
use crate::base::dimension::Dyn;
use crate::base::dimension::{Const, Dim, DimAdd, DimDiff, DimMin, DimMinimum, DimSub, DimSum, U1};
use crate::base::storage::{RawStorage, RawStorageMut, ReshapableStorage};
use crate::base::{DefaultAllocator, Matrix, OMatrix, RowVector, Scalar, Vector};
use crate::{Storage, UninitMatrix};
use std::mem::MaybeUninit;

/// # Triangular matrix extraction
impl<T: Scalar + Zero, R: Dim, C: Dim, S: Storage<T, R, C>> Matrix<T, R, C, S> {
    /// Extracts the upper triangular part of this matrix (including the diagonal).
    ///
    /// This function creates a new matrix where all elements below the main diagonal are set to zero,
    /// while elements on and above the diagonal are preserved from the original matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// # use nalgebra::Matrix3;
    /// let m = Matrix3::new(
    ///     1.0, 2.0, 3.0,
    ///     4.0, 5.0, 6.0,
    ///     7.0, 8.0, 9.0,
    /// );
    ///
    /// let upper = m.upper_triangle();
    /// assert_eq!(upper, Matrix3::new(
    ///     1.0, 2.0, 3.0,
    ///     0.0, 5.0, 6.0,
    ///     0.0, 0.0, 9.0,
    /// ));
    /// ```
    ///
    /// Works with non-square matrices as well:
    ///
    /// ```
    /// # use nalgebra::Matrix2x3;
    /// let m = Matrix2x3::new(
    ///     1.0, 2.0, 3.0,
    ///     4.0, 5.0, 6.0,
    /// );
    ///
    /// let upper = m.upper_triangle();
    /// assert_eq!(upper, Matrix2x3::new(
    ///     1.0, 2.0, 3.0,
    ///     0.0, 5.0, 6.0,
    /// ));
    /// ```
    ///
    /// # See Also
    /// - [`lower_triangle`](Self::lower_triangle) - extracts the lower triangular part
    /// - [`fill_upper_triangle`](Self::fill_upper_triangle) - fills the upper triangle in-place
    /// - [`fill_lower_triangle`](Self::fill_lower_triangle) - fills the lower triangle in-place
    #[inline]
    #[must_use]
    pub fn upper_triangle(&self) -> OMatrix<T, R, C>
    where
        DefaultAllocator: Allocator<R, C>,
    {
        let mut res = self.clone_owned();
        res.fill_lower_triangle(T::zero(), 1);

        res
    }

    /// Extracts the lower triangular part of this matrix (including the diagonal).
    ///
    /// This function creates a new matrix where all elements above the main diagonal are set to zero,
    /// while elements on and below the diagonal are preserved from the original matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// # use nalgebra::Matrix3;
    /// let m = Matrix3::new(
    ///     1.0, 2.0, 3.0,
    ///     4.0, 5.0, 6.0,
    ///     7.0, 8.0, 9.0,
    /// );
    ///
    /// let lower = m.lower_triangle();
    /// assert_eq!(lower, Matrix3::new(
    ///     1.0, 0.0, 0.0,
    ///     4.0, 5.0, 0.0,
    ///     7.0, 8.0, 9.0,
    /// ));
    /// ```
    ///
    /// Works with non-square matrices as well:
    ///
    /// ```
    /// # use nalgebra::Matrix3x2;
    /// let m = Matrix3x2::new(
    ///     1.0, 2.0,
    ///     3.0, 4.0,
    ///     5.0, 6.0,
    /// );
    ///
    /// let lower = m.lower_triangle();
    /// assert_eq!(lower, Matrix3x2::new(
    ///     1.0, 0.0,
    ///     3.0, 4.0,
    ///     5.0, 6.0,
    /// ));
    /// ```
    ///
    /// # See Also
    /// - [`upper_triangle`](Self::upper_triangle) - extracts the upper triangular part
    /// - [`fill_upper_triangle`](Self::fill_upper_triangle) - fills the upper triangle in-place
    /// - [`fill_lower_triangle`](Self::fill_lower_triangle) - fills the lower triangle in-place
    #[inline]
    #[must_use]
    pub fn lower_triangle(&self) -> OMatrix<T, R, C>
    where
        DefaultAllocator: Allocator<R, C>,
    {
        let mut res = self.clone_owned();
        res.fill_upper_triangle(T::zero(), 1);

        res
    }
}

/// # Rows and columns extraction
impl<T: Scalar, R: Dim, C: Dim, S: Storage<T, R, C>> Matrix<T, R, C, S> {
    /// Creates a new matrix by extracting the given set of rows from `self`.
    ///
    /// This function allows you to select multiple rows by their indices and combine them into a new matrix.
    /// The rows in the resulting matrix will be in the same order as the indices provided.
    /// You can select the same row multiple times or reorder rows as needed.
    ///
    /// # Arguments
    /// * `irows` - An iterator of row indices to extract. The indices must be valid (less than the number of rows).
    ///
    /// # Panics
    /// Panics if any row index is out of bounds.
    ///
    /// # Examples
    ///
    /// Basic row selection:
    ///
    /// ```
    /// # use nalgebra::Matrix3x4;
    /// let m = Matrix3x4::new(
    ///     1, 2, 3, 4,
    ///     5, 6, 7, 8,
    ///     9, 10, 11, 12,
    /// );
    ///
    /// // Select rows 0 and 2
    /// let result = m.select_rows(&[0, 2]);
    /// assert_eq!(result.nrows(), 2);
    /// assert_eq!(result.ncols(), 4);
    /// assert_eq!(result[(0, 0)], 1);
    /// assert_eq!(result[(1, 0)], 9);
    /// ```
    ///
    /// Reordering rows:
    ///
    /// ```
    /// # use nalgebra::Matrix3;
    /// let m = Matrix3::new(
    ///     1.0, 2.0, 3.0,
    ///     4.0, 5.0, 6.0,
    ///     7.0, 8.0, 9.0,
    /// );
    ///
    /// // Reverse the row order
    /// let reversed = m.select_rows(&[2, 1, 0]);
    /// assert_eq!(reversed[(0, 0)], 7.0);
    /// assert_eq!(reversed[(1, 0)], 4.0);
    /// assert_eq!(reversed[(2, 0)], 1.0);
    /// ```
    ///
    /// Duplicating rows:
    ///
    /// ```
    /// # use nalgebra::Matrix2x3;
    /// let m = Matrix2x3::new(
    ///     1, 2, 3,
    ///     4, 5, 6,
    /// );
    ///
    /// // Duplicate the first row
    /// let result = m.select_rows(&[0, 0, 1]);
    /// assert_eq!(result.nrows(), 3);
    /// assert_eq!(result[(0, 0)], 1);
    /// assert_eq!(result[(1, 0)], 1);
    /// assert_eq!(result[(2, 0)], 4);
    /// ```
    ///
    /// # See Also
    /// - [`select_columns`](Self::select_columns) - selects specific columns
    /// - [`remove_row`](Self::remove_row) - removes a single row
    /// - [`remove_rows_at`](Self::remove_rows_at) - removes multiple rows
    #[cfg(any(feature = "std", feature = "alloc"))]
    #[must_use]
    pub fn select_rows<'a, I>(&self, irows: I) -> OMatrix<T, Dyn, C>
    where
        I: IntoIterator<Item = &'a usize>,
        I::IntoIter: ExactSizeIterator + Clone,
        DefaultAllocator: Allocator<Dyn, C>,
    {
        let irows = irows.into_iter();
        let ncols = self.shape_generic().1;
        let mut res = Matrix::uninit(Dyn(irows.len()), ncols);

        // First, check that all the indices from irows are valid.
        // This will allow us to use unchecked access in the inner loop.
        for i in irows.clone() {
            assert!(*i < self.nrows(), "Row index out of bounds.")
        }

        for j in 0..ncols.value() {
            // TODO: use unchecked column indexing
            let mut res = res.column_mut(j);
            let src = self.column(j);

            for (destination, source) in irows.clone().enumerate() {
                // Safety: all indices are in range.
                unsafe {
                    *res.vget_unchecked_mut(destination) =
                        MaybeUninit::new(src.vget_unchecked(*source).clone());
                }
            }
        }

        // Safety: res is now fully initialized.
        unsafe { res.assume_init() }
    }

    /// Creates a new matrix by extracting the given set of columns from `self`.
    ///
    /// This function allows you to select multiple columns by their indices and combine them into a new matrix.
    /// The columns in the resulting matrix will be in the same order as the indices provided.
    /// You can select the same column multiple times or reorder columns as needed.
    ///
    /// # Arguments
    /// * `icols` - An iterator of column indices to extract. The indices must be valid (less than the number of columns).
    ///
    /// # Panics
    /// Panics if any column index is out of bounds.
    ///
    /// # Examples
    ///
    /// Basic column selection:
    ///
    /// ```
    /// # use nalgebra::Matrix3x4;
    /// let m = Matrix3x4::new(
    ///     1, 2, 3, 4,
    ///     5, 6, 7, 8,
    ///     9, 10, 11, 12,
    /// );
    ///
    /// // Select columns 0 and 3
    /// let result = m.select_columns(&[0, 3]);
    /// assert_eq!(result.nrows(), 3);
    /// assert_eq!(result.ncols(), 2);
    /// assert_eq!(result[(0, 0)], 1);
    /// assert_eq!(result[(0, 1)], 4);
    /// ```
    ///
    /// Reordering columns:
    ///
    /// ```
    /// # use nalgebra::Matrix2x3;
    /// let m = Matrix2x3::new(
    ///     1.0, 2.0, 3.0,
    ///     4.0, 5.0, 6.0,
    /// );
    ///
    /// // Reverse the column order
    /// let reversed = m.select_columns(&[2, 1, 0]);
    /// assert_eq!(reversed[(0, 0)], 3.0);
    /// assert_eq!(reversed[(0, 1)], 2.0);
    /// assert_eq!(reversed[(0, 2)], 1.0);
    /// ```
    ///
    /// Duplicating columns:
    ///
    /// ```
    /// # use nalgebra::Matrix3x2;
    /// let m = Matrix3x2::new(
    ///     1, 2,
    ///     3, 4,
    ///     5, 6,
    /// );
    ///
    /// // Duplicate the second column
    /// let result = m.select_columns(&[0, 1, 1]);
    /// assert_eq!(result.ncols(), 3);
    /// assert_eq!(result[(0, 0)], 1);
    /// assert_eq!(result[(0, 1)], 2);
    /// assert_eq!(result[(0, 2)], 2);
    /// ```
    ///
    /// # See Also
    /// - [`select_rows`](Self::select_rows) - selects specific rows
    /// - [`remove_column`](Self::remove_column) - removes a single column
    /// - [`remove_columns_at`](Self::remove_columns_at) - removes multiple columns
    #[cfg(any(feature = "std", feature = "alloc"))]
    #[must_use]
    pub fn select_columns<'a, I>(&self, icols: I) -> OMatrix<T, R, Dyn>
    where
        I: IntoIterator<Item = &'a usize>,
        I::IntoIter: ExactSizeIterator,
        DefaultAllocator: Allocator<R, Dyn>,
    {
        let icols = icols.into_iter();
        let nrows = self.shape_generic().0;
        let mut res = Matrix::uninit(nrows, Dyn(icols.len()));

        for (destination, source) in icols.enumerate() {
            // NOTE: this is basically a copy_frow but wrapping the values insnide of MaybeUninit.
            res.column_mut(destination)
                .zip_apply(&self.column(*source), |out, e| *out = MaybeUninit::new(e));
        }

        // Safety: res is now fully initialized.
        unsafe { res.assume_init() }
    }
}

/// # Set rows, columns, and diagonal
impl<T: Scalar, R: Dim, C: Dim, S: RawStorageMut<T, R, C>> Matrix<T, R, C, S> {
    /// Fills the diagonal of this matrix with the content of the given vector.
    ///
    /// This function replaces the diagonal elements (where row index equals column index) with
    /// values from the provided vector. For non-square matrices, only the elements along the
    /// main diagonal (up to the minimum of rows and columns) are modified.
    ///
    /// # Arguments
    /// * `diag` - A vector containing the new diagonal values. Its length must match the
    ///   minimum of the number of rows and columns in the matrix.
    ///
    /// # Panics
    /// Panics if the length of `diag` does not equal `min(nrows, ncols)`.
    ///
    /// # Examples
    ///
    /// Setting the diagonal of a square matrix:
    ///
    /// ```
    /// # use nalgebra::{Matrix3, Vector3};
    /// let mut m = Matrix3::zeros();
    /// let diag = Vector3::new(1.0, 2.0, 3.0);
    /// m.set_diagonal(&diag);
    ///
    /// assert_eq!(m, Matrix3::new(
    ///     1.0, 0.0, 0.0,
    ///     0.0, 2.0, 0.0,
    ///     0.0, 0.0, 3.0,
    /// ));
    /// ```
    ///
    /// Setting the diagonal of a non-square matrix:
    ///
    /// ```
    /// # use nalgebra::{Matrix2x3, Vector2};
    /// let mut m = Matrix2x3::zeros();
    /// let diag = Vector2::new(5.0, 10.0);
    /// m.set_diagonal(&diag);
    ///
    /// assert_eq!(m, Matrix2x3::new(
    ///     5.0, 0.0, 0.0,
    ///     0.0, 10.0, 0.0,
    /// ));
    /// ```
    ///
    /// # See Also
    /// - [`set_partial_diagonal`](Self::set_partial_diagonal) - fills the diagonal from an iterator
    /// - [`fill_diagonal`](Self::fill_diagonal) - fills the diagonal with a single value
    /// - [`diagonal`](crate::Matrix::diagonal) - extracts the diagonal as a vector
    #[inline]
    pub fn set_diagonal<R2: Dim, S2>(&mut self, diag: &Vector<T, R2, S2>)
    where
        R: DimMin<C>,
        S2: RawStorage<T, R2>,
        ShapeConstraint: DimEq<DimMinimum<R, C>, R2>,
    {
        let (nrows, ncols) = self.shape();
        let min_nrows_ncols = cmp::min(nrows, ncols);
        assert_eq!(diag.len(), min_nrows_ncols, "Mismatched dimensions.");

        for i in 0..min_nrows_ncols {
            unsafe { *self.get_unchecked_mut((i, i)) = diag.vget_unchecked(i).clone() }
        }
    }

    /// Fills the diagonal of this matrix with the content of the given iterator.
    ///
    /// This function replaces diagonal elements with values from the provided iterator.
    /// It will fill as many diagonal elements as the iterator yields, up to the
    /// minimum of the number of rows and columns of `self`, and starting with the
    /// diagonal element at index (0, 0). If the iterator yields fewer values than
    /// diagonal elements, only those elements are modified.
    ///
    /// # Arguments
    /// * `diag` - An iterator yielding values to set on the diagonal.
    ///
    /// # Examples
    ///
    /// Filling the entire diagonal:
    ///
    /// ```
    /// # use nalgebra::Matrix3;
    /// let mut m = Matrix3::zeros();
    /// m.set_partial_diagonal([1.0, 2.0, 3.0].into_iter());
    ///
    /// assert_eq!(m, Matrix3::new(
    ///     1.0, 0.0, 0.0,
    ///     0.0, 2.0, 0.0,
    ///     0.0, 0.0, 3.0,
    /// ));
    /// ```
    ///
    /// Filling only part of the diagonal:
    ///
    /// ```
    /// # use nalgebra::Matrix4;
    /// let mut m = Matrix4::zeros();
    /// m.set_partial_diagonal([10.0, 20.0].into_iter());
    ///
    /// assert_eq!(m[(0, 0)], 10.0);
    /// assert_eq!(m[(1, 1)], 20.0);
    /// assert_eq!(m[(2, 2)], 0.0);
    /// assert_eq!(m[(3, 3)], 0.0);
    /// ```
    ///
    /// Using with range iterators:
    ///
    /// ```
    /// # use nalgebra::Matrix3;
    /// let mut m = Matrix3::zeros();
    /// m.set_partial_diagonal(1..=3);
    ///
    /// assert_eq!(m, Matrix3::new(
    ///     1, 0, 0,
    ///     0, 2, 0,
    ///     0, 0, 3,
    /// ));
    /// ```
    ///
    /// # See Also
    /// - [`set_diagonal`](Self::set_diagonal) - fills the diagonal from a vector
    /// - [`fill_diagonal`](Self::fill_diagonal) - fills the diagonal with a single value
    #[inline]
    pub fn set_partial_diagonal(&mut self, diag: impl Iterator<Item = T>) {
        let (nrows, ncols) = self.shape();
        let min_nrows_ncols = cmp::min(nrows, ncols);

        for (i, val) in diag.enumerate().take(min_nrows_ncols) {
            unsafe { *self.get_unchecked_mut((i, i)) = val }
        }
    }

    /// Fills the selected row of this matrix with the content of the given vector.
    ///
    /// This function replaces all elements in the specified row with values from the provided
    /// row vector. The row vector must have the same number of columns as the matrix.
    ///
    /// # Arguments
    /// * `i` - The index of the row to modify (0-indexed).
    /// * `row` - A row vector containing the new values. Must have the same number of columns as the matrix.
    ///
    /// # Panics
    /// Panics if the row index is out of bounds or if the dimensions don't match.
    ///
    /// # Examples
    ///
    /// Setting a row in a matrix:
    ///
    /// ```
    /// # use nalgebra::{Matrix3, RowVector3};
    /// let mut m = Matrix3::zeros();
    /// let row = RowVector3::new(1.0, 2.0, 3.0);
    /// m.set_row(1, &row);
    ///
    /// assert_eq!(m, Matrix3::new(
    ///     0.0, 0.0, 0.0,
    ///     1.0, 2.0, 3.0,
    ///     0.0, 0.0, 0.0,
    /// ));
    /// ```
    ///
    /// Setting multiple rows:
    ///
    /// ```
    /// # use nalgebra::{Matrix2x4, RowVector4};
    /// let mut m = Matrix2x4::zeros();
    /// m.set_row(0, &RowVector4::new(1, 2, 3, 4));
    /// m.set_row(1, &RowVector4::new(5, 6, 7, 8));
    ///
    /// assert_eq!(m, Matrix2x4::new(
    ///     1, 2, 3, 4,
    ///     5, 6, 7, 8,
    /// ));
    /// ```
    ///
    /// # See Also
    /// - [`set_column`](Self::set_column) - sets a column
    /// - [`fill_row`](Self::fill_row) - fills a row with a single value
    /// - [`row_mut`](crate::Matrix::row_mut) - gets a mutable reference to a row
    #[inline]
    pub fn set_row<C2: Dim, S2>(&mut self, i: usize, row: &RowVector<T, C2, S2>)
    where
        S2: RawStorage<T, U1, C2>,
        ShapeConstraint: SameNumberOfColumns<C, C2>,
    {
        self.row_mut(i).copy_from(row);
    }

    /// Fills the selected column of this matrix with the content of the given vector.
    ///
    /// This function replaces all elements in the specified column with values from the provided
    /// column vector. The column vector must have the same number of rows as the matrix.
    ///
    /// # Arguments
    /// * `i` - The index of the column to modify (0-indexed).
    /// * `column` - A column vector containing the new values. Must have the same number of rows as the matrix.
    ///
    /// # Panics
    /// Panics if the column index is out of bounds or if the dimensions don't match.
    ///
    /// # Examples
    ///
    /// Setting a column in a matrix:
    ///
    /// ```
    /// # use nalgebra::{Matrix3, Vector3};
    /// let mut m = Matrix3::zeros();
    /// let col = Vector3::new(1.0, 2.0, 3.0);
    /// m.set_column(1, &col);
    ///
    /// assert_eq!(m, Matrix3::new(
    ///     0.0, 1.0, 0.0,
    ///     0.0, 2.0, 0.0,
    ///     0.0, 3.0, 0.0,
    /// ));
    /// ```
    ///
    /// Setting multiple columns:
    ///
    /// ```
    /// # use nalgebra::{Matrix3x2, Vector3};
    /// let mut m = Matrix3x2::zeros();
    /// m.set_column(0, &Vector3::new(1, 2, 3));
    /// m.set_column(1, &Vector3::new(4, 5, 6));
    ///
    /// assert_eq!(m, Matrix3x2::new(
    ///     1, 4,
    ///     2, 5,
    ///     3, 6,
    /// ));
    /// ```
    ///
    /// Practical use case - updating a transformation matrix:
    ///
    /// ```
    /// # use nalgebra::{Matrix4, Vector4};
    /// let mut transform = Matrix4::identity();
    ///
    /// // Set the translation column (last column)
    /// let translation = Vector4::new(10.0, 20.0, 30.0, 1.0);
    /// transform.set_column(3, &translation);
    ///
    /// assert_eq!(transform[(0, 3)], 10.0);
    /// assert_eq!(transform[(1, 3)], 20.0);
    /// assert_eq!(transform[(2, 3)], 30.0);
    /// ```
    ///
    /// # See Also
    /// - [`set_row`](Self::set_row) - sets a row
    /// - [`fill_column`](Self::fill_column) - fills a column with a single value
    /// - [`column_mut`](crate::Matrix::column_mut) - gets a mutable reference to a column
    #[inline]
    pub fn set_column<R2: Dim, S2>(&mut self, i: usize, column: &Vector<T, R2, S2>)
    where
        S2: RawStorage<T, R2, U1>,
        ShapeConstraint: SameNumberOfRows<R, R2>,
    {
        self.column_mut(i).copy_from(column);
    }
}

/// # In-place filling
impl<T, R: Dim, C: Dim, S: RawStorageMut<T, R, C>> Matrix<T, R, C, S> {
    /// Sets all the elements of this matrix to the value returned by the closure.
    ///
    /// This function calls the provided closure once for each element in the matrix and sets
    /// the element to the returned value. The closure is called for each element independently,
    /// so it can return different values each time (e.g., for generating random matrices).
    ///
    /// # Arguments
    /// * `val` - A closure that returns a value of type `T`. Called once per matrix element.
    ///
    /// # Examples
    ///
    /// Filling with a constant from a closure:
    ///
    /// ```
    /// # use nalgebra::Matrix2x3;
    /// let mut m = Matrix2x3::zeros();
    /// m.fill_with(|| 42);
    ///
    /// assert_eq!(m, Matrix2x3::new(
    ///     42, 42, 42,
    ///     42, 42, 42,
    /// ));
    /// ```
    ///
    /// Filling with sequential values:
    ///
    /// ```
    /// # use nalgebra::Matrix2x3;
    /// use std::cell::Cell;
    ///
    /// let mut m = Matrix2x3::zeros();
    /// let counter = Cell::new(0);
    /// m.fill_with(|| {
    ///     let val = counter.get();
    ///     counter.set(val + 1);
    ///     val
    /// });
    ///
    /// assert_eq!(m, Matrix2x3::new(
    ///     0, 2, 4,
    ///     1, 3, 5,
    /// ));
    /// ```
    ///
    /// Creating a matrix with random values:
    ///
    /// ```no_run
    /// # use nalgebra::Matrix3;
    /// use rand::Rng;
    ///
    /// let mut m = Matrix3::zeros();
    /// let mut rng = rand::thread_rng();
    /// m.fill_with(|| rng.gen_range(0..10));
    ///
    /// // Matrix is now filled with random integers between 0 and 9
    /// assert!(m.iter().all(|&x| x < 10));
    /// ```
    ///
    /// # See Also
    /// - [`fill`](Self::fill) - fills with a single value
    /// - [`from_fn`](crate::Matrix::from_fn) - creates a new matrix using a function
    #[inline]
    pub fn fill_with(&mut self, val: impl Fn() -> T) {
        for e in self.iter_mut() {
            *e = val()
        }
    }

    /// Sets all the elements of this matrix to `val`.
    ///
    /// This function replaces every element in the matrix with the same value.
    /// This is useful for initializing or resetting a matrix to a known state.
    ///
    /// # Arguments
    /// * `val` - The value to set for all matrix elements.
    ///
    /// # Examples
    ///
    /// Filling a matrix with a constant value:
    ///
    /// ```
    /// # use nalgebra::Matrix2x3;
    /// let mut m = Matrix2x3::zeros();
    /// m.fill(7.0);
    ///
    /// assert_eq!(m, Matrix2x3::new(
    ///     7.0, 7.0, 7.0,
    ///     7.0, 7.0, 7.0,
    /// ));
    /// ```
    ///
    /// Resetting a matrix:
    ///
    /// ```
    /// # use nalgebra::Matrix3;
    /// let mut m = Matrix3::new(
    ///     1, 2, 3,
    ///     4, 5, 6,
    ///     7, 8, 9,
    /// );
    /// m.fill(0);
    ///
    /// assert_eq!(m, Matrix3::zeros());
    /// ```
    ///
    /// Use with dynamic matrices:
    ///
    /// ```
    /// # use nalgebra::DMatrix;
    /// let mut m = DMatrix::from_element(3, 4, 1.0);
    /// m.fill(99.0);
    ///
    /// assert!(m.iter().all(|&x| x == 99.0));
    /// ```
    ///
    /// # See Also
    /// - [`fill_with`](Self::fill_with) - fills using a closure
    /// - [`fill_diagonal`](Self::fill_diagonal) - fills only the diagonal
    /// - [`zeros`](crate::Matrix::zeros) - creates a new matrix filled with zeros
    #[inline]
    pub fn fill(&mut self, val: T)
    where
        T: Scalar,
    {
        for e in self.iter_mut() {
            *e = val.clone()
        }
    }

    /// Fills `self` with the identity matrix.
    ///
    /// This function sets all elements to zero except for the diagonal elements, which are set to one.
    /// For non-square matrices, it creates an identity-like matrix where the diagonal (up to the
    /// minimum of rows and columns) is filled with ones.
    ///
    /// # Examples
    ///
    /// Creating an identity matrix:
    ///
    /// ```
    /// # use nalgebra::Matrix3;
    /// let mut m = Matrix3::zeros();
    /// m.fill_with_identity();
    ///
    /// assert_eq!(m, Matrix3::identity());
    /// ```
    ///
    /// Resetting a matrix to identity:
    ///
    /// ```
    /// # use nalgebra::Matrix2;
    /// let mut m = Matrix2::new(
    ///     5.0, 6.0,
    ///     7.0, 8.0,
    /// );
    /// m.fill_with_identity();
    ///
    /// assert_eq!(m, Matrix2::new(
    ///     1.0, 0.0,
    ///     0.0, 1.0,
    /// ));
    /// ```
    ///
    /// Works with non-square matrices:
    ///
    /// ```
    /// # use nalgebra::Matrix2x3;
    /// let mut m = Matrix2x3::zeros();
    /// m.fill_with_identity();
    ///
    /// assert_eq!(m, Matrix2x3::new(
    ///     1.0, 0.0, 0.0,
    ///     0.0, 1.0, 0.0,
    /// ));
    /// ```
    ///
    /// # See Also
    /// - [`identity`](crate::Matrix::identity) - creates a new identity matrix
    /// - [`fill_diagonal`](Self::fill_diagonal) - fills only the diagonal with a value
    /// - [`fill`](Self::fill) - fills the entire matrix with a value
    #[inline]
    pub fn fill_with_identity(&mut self)
    where
        T: Scalar + Zero + One,
    {
        self.fill(T::zero());
        self.fill_diagonal(T::one());
    }

    /// Sets all the diagonal elements of this matrix to `val`.
    ///
    /// This function sets all elements where the row index equals the column index to the specified value.
    /// For non-square matrices, it affects elements up to the minimum of the number of rows and columns.
    /// Other elements remain unchanged.
    ///
    /// # Arguments
    /// * `val` - The value to set for all diagonal elements.
    ///
    /// # Examples
    ///
    /// Filling the diagonal of a square matrix:
    ///
    /// ```
    /// # use nalgebra::Matrix3;
    /// let mut m = Matrix3::zeros();
    /// m.fill_diagonal(5.0);
    ///
    /// assert_eq!(m, Matrix3::new(
    ///     5.0, 0.0, 0.0,
    ///     0.0, 5.0, 0.0,
    ///     0.0, 0.0, 5.0,
    /// ));
    /// ```
    ///
    /// Filling the diagonal of a non-square matrix:
    ///
    /// ```
    /// # use nalgebra::Matrix2x4;
    /// let mut m = Matrix2x4::zeros();
    /// m.fill_diagonal(1);
    ///
    /// assert_eq!(m, Matrix2x4::new(
    ///     1, 0, 0, 0,
    ///     0, 1, 0, 0,
    /// ));
    /// ```
    ///
    /// Practical use - scaling an existing matrix:
    ///
    /// ```
    /// # use nalgebra::Matrix3;
    /// let mut transform = Matrix3::identity();
    /// // Scale by a factor of 2 on all axes
    /// transform.fill_diagonal(2.0);
    ///
    /// let point = transform * nalgebra::Vector3::new(1.0, 2.0, 3.0);
    /// assert_eq!(point, nalgebra::Vector3::new(2.0, 4.0, 6.0));
    /// ```
    ///
    /// # See Also
    /// - [`set_diagonal`](Self::set_diagonal) - sets diagonal from a vector
    /// - [`fill_with_identity`](Self::fill_with_identity) - creates an identity matrix
    /// - [`diagonal`](crate::Matrix::diagonal) - extracts the diagonal
    #[inline]
    pub fn fill_diagonal(&mut self, val: T)
    where
        T: Scalar,
    {
        let (nrows, ncols) = self.shape();
        let n = cmp::min(nrows, ncols);

        for i in 0..n {
            unsafe { *self.get_unchecked_mut((i, i)) = val.clone() }
        }
    }

    /// Sets all the elements of the selected row to `val`.
    ///
    /// This function replaces every element in the specified row with the given value.
    /// This is useful for initializing or modifying specific rows of a matrix.
    ///
    /// # Arguments
    /// * `i` - The index of the row to fill (0-indexed).
    /// * `val` - The value to set for all elements in the row.
    ///
    /// # Panics
    /// Panics if the row index is out of bounds.
    ///
    /// # Examples
    ///
    /// Filling a single row:
    ///
    /// ```
    /// # use nalgebra::Matrix3;
    /// let mut m = Matrix3::zeros();
    /// m.fill_row(1, 5.0);
    ///
    /// assert_eq!(m, Matrix3::new(
    ///     0.0, 0.0, 0.0,
    ///     5.0, 5.0, 5.0,
    ///     0.0, 0.0, 0.0,
    /// ));
    /// ```
    ///
    /// Filling multiple rows with different values:
    ///
    /// ```
    /// # use nalgebra::Matrix3x4;
    /// let mut m = Matrix3x4::zeros();
    /// m.fill_row(0, 1);
    /// m.fill_row(2, 3);
    ///
    /// assert_eq!(m, Matrix3x4::new(
    ///     1, 1, 1, 1,
    ///     0, 0, 0, 0,
    ///     3, 3, 3, 3,
    /// ));
    /// ```
    ///
    /// # See Also
    /// - [`fill_column`](Self::fill_column) - fills a column with a value
    /// - [`set_row`](Self::set_row) - sets a row from a vector
    /// - [`fill`](Self::fill) - fills the entire matrix
    #[inline]
    pub fn fill_row(&mut self, i: usize, val: T)
    where
        T: Scalar,
    {
        assert!(i < self.nrows(), "Row index out of bounds.");
        for j in 0..self.ncols() {
            unsafe { *self.get_unchecked_mut((i, j)) = val.clone() }
        }
    }

    /// Sets all the elements of the selected column to `val`.
    ///
    /// This function replaces every element in the specified column with the given value.
    /// This is useful for initializing or modifying specific columns of a matrix.
    ///
    /// # Arguments
    /// * `j` - The index of the column to fill (0-indexed).
    /// * `val` - The value to set for all elements in the column.
    ///
    /// # Panics
    /// Panics if the column index is out of bounds.
    ///
    /// # Examples
    ///
    /// Filling a single column:
    ///
    /// ```
    /// # use nalgebra::Matrix3;
    /// let mut m = Matrix3::zeros();
    /// m.fill_column(1, 7.0);
    ///
    /// assert_eq!(m, Matrix3::new(
    ///     0.0, 7.0, 0.0,
    ///     0.0, 7.0, 0.0,
    ///     0.0, 7.0, 0.0,
    /// ));
    /// ```
    ///
    /// Filling multiple columns with different values:
    ///
    /// ```
    /// # use nalgebra::Matrix3x4;
    /// let mut m = Matrix3x4::zeros();
    /// m.fill_column(0, 1);
    /// m.fill_column(3, 4);
    ///
    /// assert_eq!(m, Matrix3x4::new(
    ///     1, 0, 0, 4,
    ///     1, 0, 0, 4,
    ///     1, 0, 0, 4,
    /// ));
    /// ```
    ///
    /// # See Also
    /// - [`fill_row`](Self::fill_row) - fills a row with a value
    /// - [`set_column`](Self::set_column) - sets a column from a vector
    /// - [`fill`](Self::fill) - fills the entire matrix
    #[inline]
    pub fn fill_column(&mut self, j: usize, val: T)
    where
        T: Scalar,
    {
        assert!(j < self.ncols(), "Row index out of bounds.");
        for i in 0..self.nrows() {
            unsafe { *self.get_unchecked_mut((i, j)) = val.clone() }
        }
    }

    /// Sets all the elements of the lower-triangular part of this matrix to `val`.
    ///
    /// This function fills the lower triangular portion of the matrix with the specified value.
    /// The `shift` parameter controls which diagonals are affected, allowing you to preserve
    /// the main diagonal and nearby diagonals if desired.
    ///
    /// # Arguments
    /// * `val` - The value to set for elements in the lower triangle.
    /// * `shift` - Controls which diagonals to preserve:
    ///   - If `shift = 0` then the diagonal is overwritten as well.
    ///   - If `shift = 1` then the diagonal is left untouched.
    ///   - If `shift > 1`, then the diagonal and the first `shift - 1` subdiagonals are left untouched.
    ///
    /// # Examples
    ///
    /// Filling lower triangle including diagonal (shift = 0):
    ///
    /// ```
    /// # use nalgebra::Matrix4;
    /// let mut m = Matrix4::identity();
    /// m.fill_lower_triangle(5.0, 0);
    ///
    /// assert_eq!(m, Matrix4::new(
    ///     5.0, 1.0, 0.0, 0.0,
    ///     5.0, 5.0, 1.0, 0.0,
    ///     5.0, 5.0, 5.0, 1.0,
    ///     5.0, 5.0, 5.0, 5.0,
    /// ));
    /// ```
    ///
    /// Filling lower triangle excluding diagonal (shift = 1):
    ///
    /// ```
    /// # use nalgebra::Matrix4;
    /// let mut m = Matrix4::identity();
    /// m.fill_lower_triangle(0.0, 1);
    ///
    /// assert_eq!(m, Matrix4::new(
    ///     1.0, 0.0, 0.0, 0.0,
    ///     0.0, 1.0, 0.0, 0.0,
    ///     0.0, 0.0, 1.0, 0.0,
    ///     0.0, 0.0, 0.0, 1.0,
    /// ));
    /// ```
    ///
    /// Filling with larger shift to preserve multiple diagonals:
    ///
    /// ```
    /// # use nalgebra::Matrix4;
    /// let mut m = Matrix4::from_element(9);
    /// m.fill_lower_triangle(0, 2);
    ///
    /// assert_eq!(m, Matrix4::new(
    ///     9, 9, 9, 9,
    ///     9, 9, 9, 9,
    ///     0, 9, 9, 9,
    ///     0, 0, 9, 9,
    /// ));
    /// ```
    ///
    /// # See Also
    /// - [`fill_upper_triangle`](Self::fill_upper_triangle) - fills the upper triangle
    /// - [`lower_triangle`](Self::lower_triangle) - extracts the lower triangle
    /// - [`fill_lower_triangle_with_upper_triangle`](Self::fill_lower_triangle_with_upper_triangle) - copies upper to lower
    #[inline]
    pub fn fill_lower_triangle(&mut self, val: T, shift: usize)
    where
        T: Scalar,
    {
        for j in 0..self.ncols() {
            for i in (j + shift)..self.nrows() {
                unsafe { *self.get_unchecked_mut((i, j)) = val.clone() }
            }
        }
    }

    /// Sets all the elements of the upper-triangular part of this matrix to `val`.
    ///
    /// This function fills the upper triangular portion of the matrix with the specified value.
    /// The `shift` parameter controls which diagonals are affected, allowing you to preserve
    /// the main diagonal and nearby diagonals if desired.
    ///
    /// # Arguments
    /// * `val` - The value to set for elements in the upper triangle.
    /// * `shift` - Controls which diagonals to preserve:
    ///   - If `shift = 0` then the diagonal is overwritten as well.
    ///   - If `shift = 1` then the diagonal is left untouched.
    ///   - If `shift > 1`, then the diagonal and the first `shift - 1` superdiagonals are left untouched.
    ///
    /// # Examples
    ///
    /// Filling upper triangle including diagonal (shift = 0):
    ///
    /// ```
    /// # use nalgebra::Matrix4;
    /// let mut m = Matrix4::identity();
    /// m.fill_upper_triangle(5.0, 0);
    ///
    /// assert_eq!(m, Matrix4::new(
    ///     5.0, 5.0, 5.0, 5.0,
    ///     1.0, 5.0, 5.0, 5.0,
    ///     0.0, 1.0, 5.0, 5.0,
    ///     0.0, 0.0, 1.0, 5.0,
    /// ));
    /// ```
    ///
    /// Filling upper triangle excluding diagonal (shift = 1):
    ///
    /// ```
    /// # use nalgebra::Matrix4;
    /// let mut m = Matrix4::identity();
    /// m.fill_upper_triangle(0.0, 1);
    ///
    /// assert_eq!(m, Matrix4::new(
    ///     1.0, 0.0, 0.0, 0.0,
    ///     0.0, 1.0, 0.0, 0.0,
    ///     0.0, 0.0, 1.0, 0.0,
    ///     0.0, 0.0, 0.0, 1.0,
    /// ));
    /// ```
    ///
    /// Filling with larger shift to preserve multiple diagonals:
    ///
    /// ```
    /// # use nalgebra::Matrix4;
    /// let mut m = Matrix4::from_element(9);
    /// m.fill_upper_triangle(0, 2);
    ///
    /// assert_eq!(m, Matrix4::new(
    ///     9, 9, 0, 0,
    ///     9, 9, 9, 0,
    ///     9, 9, 9, 9,
    ///     9, 9, 9, 9,
    /// ));
    /// ```
    ///
    /// # See Also
    /// - [`fill_lower_triangle`](Self::fill_lower_triangle) - fills the lower triangle
    /// - [`upper_triangle`](Self::upper_triangle) - extracts the upper triangle
    /// - [`fill_upper_triangle_with_lower_triangle`](Self::fill_upper_triangle_with_lower_triangle) - copies lower to upper
    #[inline]
    pub fn fill_upper_triangle(&mut self, val: T, shift: usize)
    where
        T: Scalar,
    {
        for j in shift..self.ncols() {
            // TODO: is there a more efficient way to avoid the min ?
            // (necessary for rectangular matrices)
            for i in 0..cmp::min(j + 1 - shift, self.nrows()) {
                unsafe { *self.get_unchecked_mut((i, j)) = val.clone() }
            }
        }
    }
}

impl<T: Scalar, D: Dim, S: RawStorageMut<T, D, D>> Matrix<T, D, D, S> {
    /// Copies the upper-triangle of this matrix to its lower-triangular part.
    ///
    /// This function mirrors the upper triangular part (above the diagonal) to the lower triangular part,
    /// making the matrix symmetric. The diagonal elements remain unchanged. This operation is commonly used
    /// when you have computed the upper triangle of a symmetric matrix and need to fill in the lower part.
    ///
    /// # Panics
    /// Panics if the matrix is not square.
    ///
    /// # Examples
    ///
    /// Creating a symmetric matrix from an upper triangle:
    ///
    /// ```
    /// # use nalgebra::Matrix3;
    /// let mut m = Matrix3::new(
    ///     1.0, 2.0, 3.0,
    ///     0.0, 4.0, 5.0,
    ///     0.0, 0.0, 6.0,
    /// );
    /// m.fill_lower_triangle_with_upper_triangle();
    ///
    /// assert_eq!(m, Matrix3::new(
    ///     1.0, 2.0, 3.0,
    ///     2.0, 4.0, 5.0,
    ///     3.0, 5.0, 6.0,
    /// ));
    /// ```
    ///
    /// Ensuring a matrix is symmetric:
    ///
    /// ```
    /// # use nalgebra::Matrix4;
    /// let mut m = Matrix4::identity();
    /// m[(0, 1)] = 7.0;
    /// m[(0, 2)] = 8.0;
    /// m[(1, 2)] = 9.0;
    ///
    /// m.fill_lower_triangle_with_upper_triangle();
    ///
    /// // Now the matrix is symmetric
    /// assert_eq!(m[(1, 0)], 7.0);
    /// assert_eq!(m[(2, 0)], 8.0);
    /// assert_eq!(m[(2, 1)], 9.0);
    /// ```
    ///
    /// # See Also
    /// - [`fill_upper_triangle_with_lower_triangle`](Self::fill_upper_triangle_with_lower_triangle) - mirrors lower to upper
    /// - [`upper_triangle`](Self::upper_triangle) - extracts upper triangle
    /// - [`lower_triangle`](Self::lower_triangle) - extracts lower triangle
    pub fn fill_lower_triangle_with_upper_triangle(&mut self) {
        assert!(self.is_square(), "The input matrix should be square.");

        let dim = self.nrows();
        for j in 0..dim {
            for i in j + 1..dim {
                unsafe {
                    *self.get_unchecked_mut((i, j)) = self.get_unchecked((j, i)).clone();
                }
            }
        }
    }

    /// Copies the lower-triangle of this matrix to its upper-triangular part.
    ///
    /// This function mirrors the lower triangular part (below the diagonal) to the upper triangular part,
    /// making the matrix symmetric. The diagonal elements remain unchanged. This operation is commonly used
    /// when you have computed the lower triangle of a symmetric matrix and need to fill in the upper part.
    ///
    /// # Panics
    /// Panics if the matrix is not square.
    ///
    /// # Examples
    ///
    /// Creating a symmetric matrix from a lower triangle:
    ///
    /// ```
    /// # use nalgebra::Matrix3;
    /// let mut m = Matrix3::new(
    ///     1.0, 0.0, 0.0,
    ///     2.0, 4.0, 0.0,
    ///     3.0, 5.0, 6.0,
    /// );
    /// m.fill_upper_triangle_with_lower_triangle();
    ///
    /// assert_eq!(m, Matrix3::new(
    ///     1.0, 2.0, 3.0,
    ///     2.0, 4.0, 5.0,
    ///     3.0, 5.0, 6.0,
    /// ));
    /// ```
    ///
    /// Ensuring a matrix is symmetric:
    ///
    /// ```
    /// # use nalgebra::Matrix4;
    /// let mut m = Matrix4::identity();
    /// m[(1, 0)] = 7.0;
    /// m[(2, 0)] = 8.0;
    /// m[(2, 1)] = 9.0;
    ///
    /// m.fill_upper_triangle_with_lower_triangle();
    ///
    /// // Now the matrix is symmetric
    /// assert_eq!(m[(0, 1)], 7.0);
    /// assert_eq!(m[(0, 2)], 8.0);
    /// assert_eq!(m[(1, 2)], 9.0);
    /// ```
    ///
    /// # See Also
    /// - [`fill_lower_triangle_with_upper_triangle`](Self::fill_lower_triangle_with_upper_triangle) - mirrors upper to lower
    /// - [`upper_triangle`](Self::upper_triangle) - extracts upper triangle
    /// - [`lower_triangle`](Self::lower_triangle) - extracts lower triangle
    pub fn fill_upper_triangle_with_lower_triangle(&mut self) {
        assert!(self.is_square(), "The input matrix should be square.");

        for j in 1..self.ncols() {
            for i in 0..j {
                unsafe {
                    *self.get_unchecked_mut((i, j)) = self.get_unchecked((j, i)).clone();
                }
            }
        }
    }
}

/// # In-place swapping
impl<T: Scalar, R: Dim, C: Dim, S: RawStorageMut<T, R, C>> Matrix<T, R, C, S> {
    /// Swaps two rows in-place.
    ///
    /// This function exchanges all elements of two rows without allocating new memory.
    /// If the row indices are the same, the matrix remains unchanged.
    ///
    /// # Arguments
    /// * `irow1` - The index of the first row (0-indexed).
    /// * `irow2` - The index of the second row (0-indexed).
    ///
    /// # Panics
    /// Panics if either row index is out of bounds.
    ///
    /// # Examples
    ///
    /// Swapping two rows:
    ///
    /// ```
    /// # use nalgebra::Matrix3x4;
    /// let mut m = Matrix3x4::new(
    ///     1, 2, 3, 4,
    ///     5, 6, 7, 8,
    ///     9, 10, 11, 12,
    /// );
    /// m.swap_rows(0, 2);
    ///
    /// assert_eq!(m, Matrix3x4::new(
    ///     9, 10, 11, 12,
    ///     5, 6, 7, 8,
    ///     1, 2, 3, 4,
    /// ));
    /// ```
    ///
    /// Swapping adjacent rows:
    ///
    /// ```
    /// # use nalgebra::Matrix3;
    /// let mut m = Matrix3::new(
    ///     1.0, 2.0, 3.0,
    ///     4.0, 5.0, 6.0,
    ///     7.0, 8.0, 9.0,
    /// );
    /// m.swap_rows(1, 2);
    ///
    /// assert_eq!(m, Matrix3::new(
    ///     1.0, 2.0, 3.0,
    ///     7.0, 8.0, 9.0,
    ///     4.0, 5.0, 6.0,
    /// ));
    /// ```
    ///
    /// Practical use - row operations in linear algebra:
    ///
    /// ```
    /// # use nalgebra::Matrix2x3;
    /// // Gaussian elimination often requires row swapping
    /// let mut m = Matrix2x3::new(
    ///     0.0, 1.0, 2.0,
    ///     3.0, 4.0, 5.0,
    /// );
    /// // Swap to get a non-zero pivot
    /// m.swap_rows(0, 1);
    /// assert_eq!(m[(0, 0)], 3.0);
    /// ```
    ///
    /// # See Also
    /// - [`swap_columns`](Self::swap_columns) - swaps two columns
    /// - [`swap`](crate::Matrix::swap) - swaps two individual elements
    #[inline]
    pub fn swap_rows(&mut self, irow1: usize, irow2: usize) {
        assert!(irow1 < self.nrows() && irow2 < self.nrows());

        if irow1 != irow2 {
            // TODO: optimize that.
            for i in 0..self.ncols() {
                unsafe { self.swap_unchecked((irow1, i), (irow2, i)) }
            }
        }
        // Otherwise do nothing.
    }

    /// Swaps two columns in-place.
    ///
    /// This function exchanges all elements of two columns without allocating new memory.
    /// If the column indices are the same, the matrix remains unchanged.
    ///
    /// # Arguments
    /// * `icol1` - The index of the first column (0-indexed).
    /// * `icol2` - The index of the second column (0-indexed).
    ///
    /// # Panics
    /// Panics if either column index is out of bounds.
    ///
    /// # Examples
    ///
    /// Swapping two columns:
    ///
    /// ```
    /// # use nalgebra::Matrix3x4;
    /// let mut m = Matrix3x4::new(
    ///     1, 2, 3, 4,
    ///     5, 6, 7, 8,
    ///     9, 10, 11, 12,
    /// );
    /// m.swap_columns(0, 3);
    ///
    /// assert_eq!(m, Matrix3x4::new(
    ///     4, 2, 3, 1,
    ///     8, 6, 7, 5,
    ///     12, 10, 11, 9,
    /// ));
    /// ```
    ///
    /// Swapping adjacent columns:
    ///
    /// ```
    /// # use nalgebra::Matrix3;
    /// let mut m = Matrix3::new(
    ///     1.0, 2.0, 3.0,
    ///     4.0, 5.0, 6.0,
    ///     7.0, 8.0, 9.0,
    /// );
    /// m.swap_columns(1, 2);
    ///
    /// assert_eq!(m, Matrix3::new(
    ///     1.0, 3.0, 2.0,
    ///     4.0, 6.0, 5.0,
    ///     7.0, 9.0, 8.0,
    /// ));
    /// ```
    ///
    /// Practical use - matrix transformations:
    ///
    /// ```
    /// # use nalgebra::Matrix2x3;
    /// let mut m = Matrix2x3::new(
    ///     1.0, 2.0, 3.0,
    ///     4.0, 5.0, 6.0,
    /// );
    /// // Reorder columns
    /// m.swap_columns(0, 2);
    /// assert_eq!(m[(0, 0)], 3.0);
    /// assert_eq!(m[(0, 2)], 1.0);
    /// ```
    ///
    /// # See Also
    /// - [`swap_rows`](Self::swap_rows) - swaps two rows
    /// - [`swap`](crate::Matrix::swap) - swaps two individual elements
    #[inline]
    pub fn swap_columns(&mut self, icol1: usize, icol2: usize) {
        assert!(icol1 < self.ncols() && icol2 < self.ncols());

        if icol1 != icol2 {
            // TODO: optimize that.
            for i in 0..self.nrows() {
                unsafe { self.swap_unchecked((i, icol1), (i, icol2)) }
            }
        }
        // Otherwise do nothing.
    }
}

/*
 *
 * TODO: specialize all the following for slices.
 *
 */
/// # Rows and columns removal
impl<T: Scalar, R: Dim, C: Dim, S: Storage<T, R, C>> Matrix<T, R, C, S> {
    /*
     *
     * Column removal.
     *
     */
    /// Removes the `i`-th column from this matrix.
    ///
    /// This function creates a new matrix with one fewer column by removing the specified column.
    /// All columns after the removed one are shifted left.
    ///
    /// # Arguments
    /// * `i` - The index of the column to remove (0-indexed).
    ///
    /// # Panics
    /// Panics if the column index is out of bounds.
    ///
    /// # Examples
    ///
    /// Removing a column from the middle:
    ///
    /// ```
    /// # use nalgebra::Matrix3x4;
    /// let m = Matrix3x4::new(
    ///     1, 2, 3, 4,
    ///     5, 6, 7, 8,
    ///     9, 10, 11, 12,
    /// );
    ///
    /// let result = m.remove_column(1);
    /// assert_eq!(result.ncols(), 3);
    /// assert_eq!(result, nalgebra::Matrix3::new(
    ///     1, 3, 4,
    ///     5, 7, 8,
    ///     9, 11, 12,
    /// ));
    /// ```
    ///
    /// Removing the first column:
    ///
    /// ```
    /// # use nalgebra::Matrix2x3;
    /// let m = Matrix2x3::new(
    ///     1.0, 2.0, 3.0,
    ///     4.0, 5.0, 6.0,
    /// );
    ///
    /// let result = m.remove_column(0);
    /// assert_eq!(result, nalgebra::Matrix2::new(
    ///     2.0, 3.0,
    ///     5.0, 6.0,
    /// ));
    /// ```
    ///
    /// Removing the last column:
    ///
    /// ```
    /// # use nalgebra::Matrix2x3;
    /// let m = Matrix2x3::new(
    ///     1, 2, 3,
    ///     4, 5, 6,
    /// );
    ///
    /// let result = m.remove_column(2);
    /// assert_eq!(result, nalgebra::Matrix2::new(
    ///     1, 2,
    ///     4, 5,
    /// ));
    /// ```
    ///
    /// # See Also
    /// - [`remove_row`](Self::remove_row) - removes a single row
    /// - [`remove_columns`](Self::remove_columns) - removes multiple consecutive columns
    /// - [`remove_columns_at`](Self::remove_columns_at) - removes non-consecutive columns
    /// - [`insert_column`](Self::insert_column) - adds a column
    #[inline]
    pub fn remove_column(self, i: usize) -> OMatrix<T, R, DimDiff<C, U1>>
    where
        C: DimSub<U1>,
        DefaultAllocator: Reallocator<T, R, C, R, DimDiff<C, U1>>,
    {
        self.remove_fixed_columns::<1>(i)
    }

    /// Removes all columns in `indices`.
    ///
    /// This function creates a new matrix by removing multiple columns specified by their indices.
    /// The indices can be in any order and the resulting matrix will have columns in the original
    /// order (with the specified columns removed).
    ///
    /// # Arguments
    /// * `indices` - A slice of column indices to remove.
    ///
    /// # Panics
    /// Panics if any index is out of bounds.
    ///
    /// # Examples
    ///
    /// Removing multiple columns:
    ///
    /// ```
    /// # use nalgebra::{Matrix3x5, DMatrix};
    /// let m = Matrix3x5::new(
    ///     1, 2, 3, 4, 5,
    ///     6, 7, 8, 9, 10,
    ///     11, 12, 13, 14, 15,
    /// );
    ///
    /// let result = m.remove_columns_at(&[1, 3]);
    /// assert_eq!(result.ncols(), 3);
    /// assert_eq!(result[(0, 0)], 1);
    /// assert_eq!(result[(0, 1)], 3);
    /// assert_eq!(result[(0, 2)], 5);
    /// ```
    ///
    /// Removing non-consecutive columns:
    ///
    /// ```
    /// # use nalgebra::Matrix2x4;
    /// let m = Matrix2x4::new(
    ///     1.0, 2.0, 3.0, 4.0,
    ///     5.0, 6.0, 7.0, 8.0,
    /// );
    ///
    /// let result = m.remove_columns_at(&[0, 2]);
    /// assert_eq!(result.ncols(), 2);
    /// assert_eq!(result[(0, 0)], 2.0);
    /// assert_eq!(result[(0, 1)], 4.0);
    /// ```
    ///
    /// # See Also
    /// - [`remove_column`](Self::remove_column) - removes a single column
    /// - [`remove_rows_at`](Self::remove_rows_at) - removes multiple rows
    /// - [`select_columns`](Self::select_columns) - selects specific columns (opposite operation)
    #[cfg(any(feature = "std", feature = "alloc"))]
    pub fn remove_columns_at(self, indices: &[usize]) -> OMatrix<T, R, Dyn>
    where
        C: DimSub<Dyn, Output = Dyn>,
        DefaultAllocator: Reallocator<T, R, C, R, Dyn>,
    {
        let mut m = self.into_owned();
        let (nrows, ncols) = m.shape_generic();
        let mut offset: usize = 0;
        let mut target: usize = 0;
        while offset + target < ncols.value() {
            if indices.contains(&(target + offset)) {
                // Safety: the resulting pointer is within range.
                let col_ptr = unsafe { m.data.ptr_mut().add((target + offset) * nrows.value()) };
                // Drop every element in the column we are about to overwrite.
                // We use the a similar technique as in `Vec::truncate`.
                let s = ptr::slice_from_raw_parts_mut(col_ptr, nrows.value());
                // Safety: we drop the column in-place, which is OK because we will overwrite these
                //         entries later in the loop, or discard them with the `reallocate_copy`
                //         afterwards.
                unsafe { ptr::drop_in_place(s) };

                offset += 1;
            } else {
                unsafe {
                    let ptr_source = m.data.ptr().add((target + offset) * nrows.value());
                    let ptr_target = m.data.ptr_mut().add(target * nrows.value());

                    // Copy the data, overwriting what we dropped.
                    ptr::copy(ptr_source, ptr_target, nrows.value());
                    target += 1;
                }
            }
        }

        // Safety: The new size is smaller than the old size, so
        //         DefaultAllocator::reallocate_copy will initialize
        //         every element of the new matrix which can then
        //         be assumed to be initialized.
        unsafe {
            let new_data = DefaultAllocator::reallocate_copy(
                nrows,
                ncols.sub(Dyn::from_usize(offset)),
                m.data,
            );

            Matrix::from_data(new_data).assume_init()
        }
    }

    /// Removes all rows in `indices`   
    #[cfg(any(feature = "std", feature = "alloc"))]
    pub fn remove_rows_at(self, indices: &[usize]) -> OMatrix<T, Dyn, C>
    where
        R: DimSub<Dyn, Output = Dyn>,
        DefaultAllocator: Reallocator<T, R, C, Dyn, C>,
    {
        let mut m = self.into_owned();
        let (nrows, ncols) = m.shape_generic();
        let mut offset: usize = 0;
        let mut target: usize = 0;
        while offset + target < nrows.value() * ncols.value() {
            if indices.contains(&((target + offset) % nrows.value())) {
                // Safety: the resulting pointer is within range.
                unsafe {
                    let elt_ptr = m.data.ptr_mut().add(target + offset);
                    // Safety: we drop the component in-place, which is OK because we will overwrite these
                    //         entries later in the loop, or discard them with the `reallocate_copy`
                    //         afterwards.
                    ptr::drop_in_place(elt_ptr)
                };
                offset += 1;
            } else {
                unsafe {
                    let ptr_source = m.data.ptr().add(target + offset);
                    let ptr_target = m.data.ptr_mut().add(target);

                    // Copy the data, overwriting what we dropped in the previous iterations.
                    ptr::copy(ptr_source, ptr_target, 1);
                    target += 1;
                }
            }
        }

        // Safety: The new size is smaller than the old size, so
        //         DefaultAllocator::reallocate_copy will initialize
        //         every element of the new matrix which can then
        //         be assumed to be initialized.
        unsafe {
            let new_data = DefaultAllocator::reallocate_copy(
                nrows.sub(Dyn::from_usize(offset / ncols.value())),
                ncols,
                m.data,
            );

            Matrix::from_data(new_data).assume_init()
        }
    }

    /// Removes `D::dim()` consecutive columns from this matrix, starting with the `i`-th
    /// (included).
    #[inline]
    pub fn remove_fixed_columns<const D: usize>(
        self,
        i: usize,
    ) -> OMatrix<T, R, DimDiff<C, Const<D>>>
    where
        C: DimSub<Const<D>>,
        DefaultAllocator: Reallocator<T, R, C, R, DimDiff<C, Const<D>>>,
    {
        self.remove_columns_generic(i, Const::<D>)
    }

    /// Removes `n` consecutive columns from this matrix, starting with the `i`-th (included).
    #[inline]
    #[cfg(any(feature = "std", feature = "alloc"))]
    pub fn remove_columns(self, i: usize, n: usize) -> OMatrix<T, R, Dyn>
    where
        C: DimSub<Dyn, Output = Dyn>,
        DefaultAllocator: Reallocator<T, R, C, R, Dyn>,
    {
        self.remove_columns_generic(i, Dyn(n))
    }

    /// Removes `nremove.value()` columns from this matrix, starting with the `i`-th (included).
    ///
    /// This is the generic implementation of `.remove_columns(...)` and
    /// `.remove_fixed_columns(...)` which have nicer API interfaces.
    #[inline]
    pub fn remove_columns_generic<D>(self, i: usize, nremove: D) -> OMatrix<T, R, DimDiff<C, D>>
    where
        D: Dim,
        C: DimSub<D>,
        DefaultAllocator: Reallocator<T, R, C, R, DimDiff<C, D>>,
    {
        let mut m = self.into_owned();
        let (nrows, ncols) = m.shape_generic();
        assert!(
            i + nremove.value() <= ncols.value(),
            "Column index out of range."
        );

        let need_column_shifts = nremove.value() != 0 && i + nremove.value() < ncols.value();
        if need_column_shifts {
            // The first `deleted_i * nrows` are left untouched.
            let copied_value_start = i + nremove.value();

            unsafe {
                let ptr_in = m.data.ptr().add(copied_value_start * nrows.value());
                let ptr_out = m.data.ptr_mut().add(i * nrows.value());

                // Drop all the elements of the columns we are about to overwrite.
                // We use the a similar technique as in `Vec::truncate`.
                let s = ptr::slice_from_raw_parts_mut(ptr_out, nremove.value() * nrows.value());
                // Safety: we drop the column in-place, which is OK because we will overwrite these
                //         entries with `ptr::copy` afterward.
                ptr::drop_in_place(s);

                ptr::copy(
                    ptr_in,
                    ptr_out,
                    (ncols.value() - copied_value_start) * nrows.value(),
                );
            }
        } else {
            // All the columns to remove are at the end of the buffer. Drop them.
            unsafe {
                let ptr = m.data.ptr_mut().add(i * nrows.value());
                let s = ptr::slice_from_raw_parts_mut(ptr, nremove.value() * nrows.value());
                ptr::drop_in_place(s)
            };
        }

        // Safety: The new size is smaller than the old size, so
        //         DefaultAllocator::reallocate_copy will initialize
        //         every element of the new matrix which can then
        //         be assumed to be initialized.
        unsafe {
            let new_data = DefaultAllocator::reallocate_copy(nrows, ncols.sub(nremove), m.data);
            Matrix::from_data(new_data).assume_init()
        }
    }

    /*
     *
     * Row removal.
     *
     */
    /// Removes the `i`-th row from this matrix.
    ///
    /// This function creates a new matrix with one fewer row by removing the specified row.
    /// All rows after the removed one are shifted up.
    ///
    /// # Arguments
    /// * `i` - The index of the row to remove (0-indexed).
    ///
    /// # Panics
    /// Panics if the row index is out of bounds.
    ///
    /// # Examples
    ///
    /// Removing a row from the middle:
    ///
    /// ```
    /// # use nalgebra::Matrix4x3;
    /// let m = Matrix4x3::new(
    ///     1, 2, 3,
    ///     4, 5, 6,
    ///     7, 8, 9,
    ///     10, 11, 12,
    /// );
    ///
    /// let result = m.remove_row(1);
    /// assert_eq!(result.nrows(), 3);
    /// assert_eq!(result, nalgebra::Matrix3::new(
    ///     1, 2, 3,
    ///     7, 8, 9,
    ///     10, 11, 12,
    /// ));
    /// ```
    ///
    /// Removing the first row:
    ///
    /// ```
    /// # use nalgebra::Matrix3x2;
    /// let m = Matrix3x2::new(
    ///     1.0, 2.0,
    ///     3.0, 4.0,
    ///     5.0, 6.0,
    /// );
    ///
    /// let result = m.remove_row(0);
    /// assert_eq!(result, nalgebra::Matrix2::new(
    ///     3.0, 4.0,
    ///     5.0, 6.0,
    /// ));
    /// ```
    ///
    /// Removing the last row:
    ///
    /// ```
    /// # use nalgebra::Matrix3x2;
    /// let m = Matrix3x2::new(
    ///     1, 2,
    ///     3, 4,
    ///     5, 6,
    /// );
    ///
    /// let result = m.remove_row(2);
    /// assert_eq!(result, nalgebra::Matrix2::new(
    ///     1, 2,
    ///     3, 4,
    /// ));
    /// ```
    ///
    /// # See Also
    /// - [`remove_column`](Self::remove_column) - removes a single column
    /// - [`remove_rows`](Self::remove_rows) - removes multiple consecutive rows
    /// - [`remove_rows_at`](Self::remove_rows_at) - removes non-consecutive rows
    /// - [`insert_row`](Self::insert_row) - adds a row
    #[inline]
    pub fn remove_row(self, i: usize) -> OMatrix<T, DimDiff<R, U1>, C>
    where
        R: DimSub<U1>,
        DefaultAllocator: Reallocator<T, R, C, DimDiff<R, U1>, C>,
    {
        self.remove_fixed_rows::<1>(i)
    }

    /// Removes `D::dim()` consecutive rows from this matrix, starting with the `i`-th (included).
    #[inline]
    pub fn remove_fixed_rows<const D: usize>(self, i: usize) -> OMatrix<T, DimDiff<R, Const<D>>, C>
    where
        R: DimSub<Const<D>>,
        DefaultAllocator: Reallocator<T, R, C, DimDiff<R, Const<D>>, C>,
    {
        self.remove_rows_generic(i, Const::<D>)
    }

    /// Removes `n` consecutive rows from this matrix, starting with the `i`-th (included).
    #[inline]
    #[cfg(any(feature = "std", feature = "alloc"))]
    pub fn remove_rows(self, i: usize, n: usize) -> OMatrix<T, Dyn, C>
    where
        R: DimSub<Dyn, Output = Dyn>,
        DefaultAllocator: Reallocator<T, R, C, Dyn, C>,
    {
        self.remove_rows_generic(i, Dyn(n))
    }

    /// Removes `nremove.value()` rows from this matrix, starting with the `i`-th (included).
    ///
    /// This is the generic implementation of `.remove_rows(...)` and `.remove_fixed_rows(...)`
    /// which have nicer API interfaces.
    #[inline]
    pub fn remove_rows_generic<D>(self, i: usize, nremove: D) -> OMatrix<T, DimDiff<R, D>, C>
    where
        D: Dim,
        R: DimSub<D>,
        DefaultAllocator: Reallocator<T, R, C, DimDiff<R, D>, C>,
    {
        let mut m = self.into_owned();
        let (nrows, ncols) = m.shape_generic();
        assert!(
            i + nremove.value() <= nrows.value(),
            "Row index out of range."
        );

        if nremove.value() != 0 {
            unsafe {
                compress_rows(
                    m.as_mut_slice(),
                    nrows.value(),
                    ncols.value(),
                    i,
                    nremove.value(),
                );
            }
        }

        // Safety: The new size is smaller than the old size, so
        //         DefaultAllocator::reallocate_copy will initialize
        //         every element of the new matrix which can then
        //         be assumed to be initialized.
        unsafe {
            let new_data = DefaultAllocator::reallocate_copy(nrows.sub(nremove), ncols, m.data);
            Matrix::from_data(new_data).assume_init()
        }
    }
}

/// # Rows and columns insertion
impl<T: Scalar, R: Dim, C: Dim, S: Storage<T, R, C>> Matrix<T, R, C, S> {
    /*
     *
     * Columns insertion.
     *
     */
    /// Inserts a column filled with `val` at the `i-th` position.
    ///
    /// This function creates a new matrix with one additional column. The new column is inserted
    /// at the specified position and filled with the given value. All columns at or after position
    /// `i` are shifted to the right.
    ///
    /// # Arguments
    /// * `i` - The index where the new column should be inserted (0-indexed). Can be equal to the
    ///         number of columns to append at the end.
    /// * `val` - The value to fill the new column with.
    ///
    /// # Panics
    /// Panics if the insertion index is greater than the number of columns.
    ///
    /// # Examples
    ///
    /// Inserting a column in the middle:
    ///
    /// ```
    /// # use nalgebra::Matrix3x2;
    /// let m = Matrix3x2::new(
    ///     1, 2,
    ///     3, 4,
    ///     5, 6,
    /// );
    ///
    /// let result = m.insert_column(1, 0);
    /// assert_eq!(result, nalgebra::Matrix3::new(
    ///     1, 0, 2,
    ///     3, 0, 4,
    ///     5, 0, 6,
    /// ));
    /// ```
    ///
    /// Inserting at the beginning:
    ///
    /// ```
    /// # use nalgebra::Matrix2x3;
    /// let m = Matrix2x3::new(
    ///     1.0, 2.0, 3.0,
    ///     4.0, 5.0, 6.0,
    /// );
    ///
    /// let result = m.insert_column(0, 9.0);
    /// assert_eq!(result.ncols(), 4);
    /// assert_eq!(result[(0, 0)], 9.0);
    /// assert_eq!(result[(1, 0)], 9.0);
    /// ```
    ///
    /// Appending at the end:
    ///
    /// ```
    /// # use nalgebra::Matrix2;
    /// let m = Matrix2::new(
    ///     1, 2,
    ///     3, 4,
    /// );
    ///
    /// let result = m.insert_column(2, 10);
    /// assert_eq!(result, nalgebra::Matrix2x3::new(
    ///     1, 2, 10,
    ///     3, 4, 10,
    /// ));
    /// ```
    ///
    /// # See Also
    /// - [`insert_row`](Self::insert_row) - inserts a row
    /// - [`insert_columns`](Self::insert_columns) - inserts multiple columns
    /// - [`remove_column`](Self::remove_column) - removes a column
    #[inline]
    pub fn insert_column(self, i: usize, val: T) -> OMatrix<T, R, DimSum<C, U1>>
    where
        C: DimAdd<U1>,
        DefaultAllocator: Reallocator<T, R, C, R, DimSum<C, U1>>,
    {
        self.insert_fixed_columns::<1>(i, val)
    }

    /// Inserts `D` columns filled with `val` starting at the `i-th` position.
    #[inline]
    pub fn insert_fixed_columns<const D: usize>(
        self,
        i: usize,
        val: T,
    ) -> OMatrix<T, R, DimSum<C, Const<D>>>
    where
        C: DimAdd<Const<D>>,
        DefaultAllocator: Reallocator<T, R, C, R, DimSum<C, Const<D>>>,
    {
        let mut res = unsafe { self.insert_columns_generic_uninitialized(i, Const::<D>) };
        res.fixed_columns_mut::<D>(i)
            .fill_with(|| MaybeUninit::new(val.clone()));

        // Safety: the result is now fully initialized. The added columns have
        //         been initialized by the `fill_with` above, and the rest have
        //         been initialized by `insert_columns_generic_uninitialized`.
        unsafe { res.assume_init() }
    }

    /// Inserts `n` columns filled with `val` starting at the `i-th` position.
    #[inline]
    #[cfg(any(feature = "std", feature = "alloc"))]
    pub fn insert_columns(self, i: usize, n: usize, val: T) -> OMatrix<T, R, Dyn>
    where
        C: DimAdd<Dyn, Output = Dyn>,
        DefaultAllocator: Reallocator<T, R, C, R, Dyn>,
    {
        let mut res = unsafe { self.insert_columns_generic_uninitialized(i, Dyn(n)) };
        res.columns_mut(i, n)
            .fill_with(|| MaybeUninit::new(val.clone()));

        // Safety: the result is now fully initialized. The added columns have
        //         been initialized by the `fill_with` above, and the rest have
        //         been initialized by `insert_columns_generic_uninitialized`.
        unsafe { res.assume_init() }
    }

    /// Inserts `ninsert.value()` columns starting at the `i-th` place of this matrix.
    ///
    /// # Safety
    /// The output matrix has all its elements initialized except for the the components of the
    /// added columns.
    #[inline]
    pub unsafe fn insert_columns_generic_uninitialized<D>(
        self,
        i: usize,
        ninsert: D,
    ) -> UninitMatrix<T, R, DimSum<C, D>>
    where
        D: Dim,
        C: DimAdd<D>,
        DefaultAllocator: Reallocator<T, R, C, R, DimSum<C, D>>,
    {
        unsafe {
            let m = self.into_owned();
            let (nrows, ncols) = m.shape_generic();
            let mut res = Matrix::from_data(DefaultAllocator::reallocate_copy(
                nrows,
                ncols.add(ninsert),
                m.data,
            ));

            assert!(i <= ncols.value(), "Column insertion index out of range.");

            if ninsert.value() != 0 && i != ncols.value() {
                let ptr_in = res.data.ptr().add(i * nrows.value());
                let ptr_out = res
                    .data
                    .ptr_mut()
                    .add((i + ninsert.value()) * nrows.value());

                ptr::copy(ptr_in, ptr_out, (ncols.value() - i) * nrows.value())
            }

            res
        }
    }

    /*
     *
     * Rows insertion.
     *
     */
    /// Inserts a row filled with `val` at the `i-th` position.
    ///
    /// This function creates a new matrix with one additional row. The new row is inserted
    /// at the specified position and filled with the given value. All rows at or after position
    /// `i` are shifted down.
    ///
    /// # Arguments
    /// * `i` - The index where the new row should be inserted (0-indexed). Can be equal to the
    ///         number of rows to append at the end.
    /// * `val` - The value to fill the new row with.
    ///
    /// # Panics
    /// Panics if the insertion index is greater than the number of rows.
    ///
    /// # Examples
    ///
    /// Inserting a row in the middle:
    ///
    /// ```
    /// # use nalgebra::Matrix2x3;
    /// let m = Matrix2x3::new(
    ///     1, 2, 3,
    ///     4, 5, 6,
    /// );
    ///
    /// let result = m.insert_row(1, 0);
    /// assert_eq!(result, nalgebra::Matrix3::new(
    ///     1, 2, 3,
    ///     0, 0, 0,
    ///     4, 5, 6,
    /// ));
    /// ```
    ///
    /// Inserting at the beginning:
    ///
    /// ```
    /// # use nalgebra::Matrix3x2;
    /// let m = Matrix3x2::new(
    ///     1.0, 2.0,
    ///     3.0, 4.0,
    ///     5.0, 6.0,
    /// );
    ///
    /// let result = m.insert_row(0, 9.0);
    /// assert_eq!(result.nrows(), 4);
    /// assert_eq!(result[(0, 0)], 9.0);
    /// assert_eq!(result[(0, 1)], 9.0);
    /// ```
    ///
    /// Appending at the end:
    ///
    /// ```
    /// # use nalgebra::Matrix2;
    /// let m = Matrix2::new(
    ///     1, 2,
    ///     3, 4,
    /// );
    ///
    /// let result = m.insert_row(2, 10);
    /// assert_eq!(result, nalgebra::Matrix3x2::new(
    ///     1, 2,
    ///     3, 4,
    ///     10, 10,
    /// ));
    /// ```
    ///
    /// # See Also
    /// - [`insert_column`](Self::insert_column) - inserts a column
    /// - [`insert_rows`](Self::insert_rows) - inserts multiple rows
    /// - [`remove_row`](Self::remove_row) - removes a row
    #[inline]
    pub fn insert_row(self, i: usize, val: T) -> OMatrix<T, DimSum<R, U1>, C>
    where
        R: DimAdd<U1>,
        DefaultAllocator: Reallocator<T, R, C, DimSum<R, U1>, C>,
    {
        self.insert_fixed_rows::<1>(i, val)
    }

    /// Inserts `D::dim()` rows filled with `val` starting at the `i-th` position.
    #[inline]
    pub fn insert_fixed_rows<const D: usize>(
        self,
        i: usize,
        val: T,
    ) -> OMatrix<T, DimSum<R, Const<D>>, C>
    where
        R: DimAdd<Const<D>>,
        DefaultAllocator: Reallocator<T, R, C, DimSum<R, Const<D>>, C>,
    {
        let mut res = unsafe { self.insert_rows_generic_uninitialized(i, Const::<D>) };
        res.fixed_rows_mut::<D>(i)
            .fill_with(|| MaybeUninit::new(val.clone()));

        // Safety: the result is now fully initialized. The added rows have
        //         been initialized by the `fill_with` above, and the rest have
        //         been initialized by `insert_rows_generic_uninitialized`.
        unsafe { res.assume_init() }
    }

    /// Inserts `n` rows filled with `val` starting at the `i-th` position.
    #[inline]
    #[cfg(any(feature = "std", feature = "alloc"))]
    pub fn insert_rows(self, i: usize, n: usize, val: T) -> OMatrix<T, Dyn, C>
    where
        R: DimAdd<Dyn, Output = Dyn>,
        DefaultAllocator: Reallocator<T, R, C, Dyn, C>,
    {
        let mut res = unsafe { self.insert_rows_generic_uninitialized(i, Dyn(n)) };
        res.rows_mut(i, n)
            .fill_with(|| MaybeUninit::new(val.clone()));

        // Safety: the result is now fully initialized. The added rows have
        //         been initialized by the `fill_with` above, and the rest have
        //         been initialized by `insert_rows_generic_uninitialized`.
        unsafe { res.assume_init() }
    }

    /// Inserts `ninsert.value()` rows at the `i-th` place of this matrix.
    ///
    /// # Safety
    /// The added rows values are not initialized.
    /// This is the generic implementation of `.insert_rows(...)` and
    /// `.insert_fixed_rows(...)` which have nicer API interfaces.
    #[inline]
    pub unsafe fn insert_rows_generic_uninitialized<D>(
        self,
        i: usize,
        ninsert: D,
    ) -> UninitMatrix<T, DimSum<R, D>, C>
    where
        D: Dim,
        R: DimAdd<D>,
        DefaultAllocator: Reallocator<T, R, C, DimSum<R, D>, C>,
    {
        unsafe {
            let m = self.into_owned();
            let (nrows, ncols) = m.shape_generic();
            let mut res = Matrix::from_data(DefaultAllocator::reallocate_copy(
                nrows.add(ninsert),
                ncols,
                m.data,
            ));

            assert!(i <= nrows.value(), "Row insertion index out of range.");

            if ninsert.value() != 0 {
                extend_rows(
                    res.as_mut_slice(),
                    nrows.value(),
                    ncols.value(),
                    i,
                    ninsert.value(),
                );
            }

            res
        }
    }
}

/// # Resizing and reshaping
impl<T: Scalar, R: Dim, C: Dim, S: Storage<T, R, C>> Matrix<T, R, C, S> {
    /// Resizes this matrix so that it contains `new_nrows` rows and `new_ncols` columns.
    ///
    /// The values are copied such that `self[(i, j)] == result[(i, j)]` for all valid indices.
    /// If the result has more rows and/or columns than `self`, then the extra rows or columns
    /// are filled with `val`. If the result has fewer rows or columns, elements are truncated.
    ///
    /// This function is useful for dynamically adjusting matrix dimensions while preserving
    /// existing data. The result is always a dynamically-sized matrix (`DMatrix`).
    ///
    /// # Arguments
    /// * `new_nrows` - The desired number of rows in the result.
    /// * `new_ncols` - The desired number of columns in the result.
    /// * `val` - The value to fill in new elements if the matrix grows.
    ///
    /// # Examples
    ///
    /// Growing a matrix:
    ///
    /// ```
    /// # use nalgebra::Matrix2;
    /// let m = Matrix2::new(
    ///     1, 2,
    ///     3, 4,
    /// );
    ///
    /// let resized = m.resize(3, 4, 0);
    /// assert_eq!(resized.nrows(), 3);
    /// assert_eq!(resized.ncols(), 4);
    /// assert_eq!(resized[(0, 0)], 1);
    /// assert_eq!(resized[(1, 1)], 4);
    /// assert_eq!(resized[(2, 2)], 0); // new elements filled with 0
    /// ```
    ///
    /// Shrinking a matrix:
    ///
    /// ```
    /// # use nalgebra::Matrix3;
    /// let m = Matrix3::new(
    ///     1.0, 2.0, 3.0,
    ///     4.0, 5.0, 6.0,
    ///     7.0, 8.0, 9.0,
    /// );
    ///
    /// let resized = m.resize(2, 2, 0.0);
    /// assert_eq!(resized.nrows(), 2);
    /// assert_eq!(resized.ncols(), 2);
    /// assert_eq!(resized[(0, 0)], 1.0);
    /// assert_eq!(resized[(1, 1)], 5.0);
    /// ```
    ///
    /// Changing both dimensions:
    ///
    /// ```
    /// # use nalgebra::Matrix2x3;
    /// let m = Matrix2x3::new(
    ///     1, 2, 3,
    ///     4, 5, 6,
    /// );
    ///
    /// let resized = m.resize(3, 2, 99);
    /// assert_eq!(resized.nrows(), 3);
    /// assert_eq!(resized.ncols(), 2);
    /// assert_eq!(resized[(2, 0)], 99);
    /// assert_eq!(resized[(2, 1)], 99);
    /// ```
    ///
    /// # See Also
    /// - [`resize_vertically`](Self::resize_vertically) - resizes only rows
    /// - [`resize_horizontally`](Self::resize_horizontally) - resizes only columns
    /// - [`fixed_resize`](Self::fixed_resize) - resizes to a compile-time size
    /// - [`resize_mut`](crate::Matrix::resize_mut) - resizes in-place (for `DMatrix`)
    #[cfg(any(feature = "std", feature = "alloc"))]
    pub fn resize(self, new_nrows: usize, new_ncols: usize, val: T) -> OMatrix<T, Dyn, Dyn>
    where
        DefaultAllocator: Reallocator<T, R, C, Dyn, Dyn>,
    {
        self.resize_generic(Dyn(new_nrows), Dyn(new_ncols), val)
    }

    /// Resizes this matrix vertically, i.e., so that it contains `new_nrows` rows while keeping the same number of columns.
    ///
    /// The values are copied such that `self[(i, j)] == result[(i, j)]` for all valid indices.
    /// If the result has more rows than `self`, then the extra rows are filled with `val`.
    /// If the result has fewer rows, elements are truncated.
    ///
    /// This is more efficient than `resize` when you only need to change the number of rows.
    ///
    /// # Arguments
    /// * `new_nrows` - The desired number of rows in the result.
    /// * `val` - The value to fill in new elements if rows are added.
    ///
    /// # Examples
    ///
    /// Adding rows to a matrix:
    ///
    /// ```
    /// # use nalgebra::Matrix2x3;
    /// let m = Matrix2x3::new(
    ///     1, 2, 3,
    ///     4, 5, 6,
    /// );
    ///
    /// let resized = m.resize_vertically(4, 0);
    /// assert_eq!(resized.nrows(), 4);
    /// assert_eq!(resized.ncols(), 3);
    /// assert_eq!(resized[(0, 0)], 1);
    /// assert_eq!(resized[(3, 0)], 0); // new row filled with 0
    /// ```
    ///
    /// Removing rows from a matrix:
    ///
    /// ```
    /// # use nalgebra::Matrix4x2;
    /// let m = Matrix4x2::new(
    ///     1.0, 2.0,
    ///     3.0, 4.0,
    ///     5.0, 6.0,
    ///     7.0, 8.0,
    /// );
    ///
    /// let resized = m.resize_vertically(2, 0.0);
    /// assert_eq!(resized.nrows(), 2);
    /// assert_eq!(resized.ncols(), 2);
    /// assert_eq!(resized[(1, 1)], 4.0);
    /// ```
    ///
    /// Practical use - extending a vector:
    ///
    /// ```
    /// # use nalgebra::DVector;
    /// let v = DVector::from_vec(vec![1, 2, 3]);
    /// let extended = v.resize_vertically(5, 0);
    /// assert_eq!(extended.len(), 5);
    /// assert_eq!(extended[3], 0);
    /// assert_eq!(extended[4], 0);
    /// ```
    ///
    /// # See Also
    /// - [`resize`](Self::resize) - resizes both dimensions
    /// - [`resize_horizontally`](Self::resize_horizontally) - resizes only columns
    /// - [`resize_vertically_mut`](crate::Matrix::resize_vertically_mut) - resizes in-place
    #[cfg(any(feature = "std", feature = "alloc"))]
    pub fn resize_vertically(self, new_nrows: usize, val: T) -> OMatrix<T, Dyn, C>
    where
        DefaultAllocator: Reallocator<T, R, C, Dyn, C>,
    {
        let ncols = self.shape_generic().1;
        self.resize_generic(Dyn(new_nrows), ncols, val)
    }

    /// Resizes this matrix horizontally, i.e., so that it contains `new_ncols` columns while keeping the same number of rows.
    ///
    /// The values are copied such that `self[(i, j)] == result[(i, j)]` for all valid indices.
    /// If the result has more columns than `self`, then the extra columns are filled with `val`.
    /// If the result has fewer columns, elements are truncated.
    ///
    /// This is more efficient than `resize` when you only need to change the number of columns.
    ///
    /// # Arguments
    /// * `new_ncols` - The desired number of columns in the result.
    /// * `val` - The value to fill in new elements if columns are added.
    ///
    /// # Examples
    ///
    /// Adding columns to a matrix:
    ///
    /// ```
    /// # use nalgebra::Matrix3x2;
    /// let m = Matrix3x2::new(
    ///     1, 2,
    ///     3, 4,
    ///     5, 6,
    /// );
    ///
    /// let resized = m.resize_horizontally(4, 0);
    /// assert_eq!(resized.nrows(), 3);
    /// assert_eq!(resized.ncols(), 4);
    /// assert_eq!(resized[(0, 0)], 1);
    /// assert_eq!(resized[(0, 3)], 0); // new column filled with 0
    /// ```
    ///
    /// Removing columns from a matrix:
    ///
    /// ```
    /// # use nalgebra::Matrix2x4;
    /// let m = Matrix2x4::new(
    ///     1.0, 2.0, 3.0, 4.0,
    ///     5.0, 6.0, 7.0, 8.0,
    /// );
    ///
    /// let resized = m.resize_horizontally(2, 0.0);
    /// assert_eq!(resized.nrows(), 2);
    /// assert_eq!(resized.ncols(), 2);
    /// assert_eq!(resized[(1, 1)], 6.0);
    /// ```
    ///
    /// # See Also
    /// - [`resize`](Self::resize) - resizes both dimensions
    /// - [`resize_vertically`](Self::resize_vertically) - resizes only rows
    /// - [`resize_horizontally_mut`](crate::Matrix::resize_horizontally_mut) - resizes in-place
    #[cfg(any(feature = "std", feature = "alloc"))]
    pub fn resize_horizontally(self, new_ncols: usize, val: T) -> OMatrix<T, R, Dyn>
    where
        DefaultAllocator: Reallocator<T, R, C, R, Dyn>,
    {
        let nrows = self.shape_generic().0;
        self.resize_generic(nrows, Dyn(new_ncols), val)
    }

    /// Resizes this matrix so that it contains `R2::value()` rows and `C2::value()` columns.
    ///
    /// The values are copied such that `self[(i, j)] == result[(i, j)]` for all valid indices.
    /// If the result has more rows and/or columns than `self`, then the extra rows or columns
    /// are filled with `val`. If the result has fewer rows or columns, elements are truncated.
    ///
    /// This function is similar to [`resize`](Self::resize) but produces a statically-sized matrix
    /// with dimensions known at compile time. This enables stack allocation and additional
    /// compile-time guarantees.
    ///
    /// # Arguments
    /// * `val` - The value to fill in new elements if the matrix grows.
    ///
    /// # Examples
    ///
    /// Growing a matrix to a larger static size:
    ///
    /// ```
    /// # use nalgebra::Matrix2;
    /// let m = Matrix2::new(
    ///     1, 2,
    ///     3, 4,
    /// );
    ///
    /// let resized = m.fixed_resize::<3, 4>(0);
    /// assert_eq!(resized.nrows(), 3);
    /// assert_eq!(resized.ncols(), 4);
    /// assert_eq!(resized[(0, 0)], 1);
    /// assert_eq!(resized[(1, 1)], 4);
    /// assert_eq!(resized[(2, 2)], 0);
    /// ```
    ///
    /// Shrinking a matrix to a smaller static size:
    ///
    /// ```
    /// # use nalgebra::Matrix4;
    /// let m = Matrix4::new(
    ///     1.0, 2.0, 3.0, 4.0,
    ///     5.0, 6.0, 7.0, 8.0,
    ///     9.0, 10.0, 11.0, 12.0,
    ///     13.0, 14.0, 15.0, 16.0,
    /// );
    ///
    /// let resized = m.fixed_resize::<2, 3>(0.0);
    /// assert_eq!(resized.nrows(), 2);
    /// assert_eq!(resized.ncols(), 3);
    /// assert_eq!(resized[(1, 2)], 11.0);
    /// ```
    ///
    /// Converting dynamic to static size:
    ///
    /// ```
    /// # use nalgebra::DMatrix;
    /// let dynamic = DMatrix::from_row_slice(2, 2, &[1, 2, 3, 4]);
    /// let static_mat = dynamic.fixed_resize::<3, 3>(0);
    /// assert_eq!(static_mat.nrows(), 3);
    /// assert_eq!(static_mat[(2, 2)], 0);
    /// ```
    ///
    /// # See Also
    /// - [`resize`](Self::resize) - resizes to dynamic dimensions
    /// - [`resize_generic`](Self::resize_generic) - generic version
    pub fn fixed_resize<const R2: usize, const C2: usize>(
        self,
        val: T,
    ) -> OMatrix<T, Const<R2>, Const<C2>>
    where
        DefaultAllocator: Reallocator<T, R, C, Const<R2>, Const<C2>>,
    {
        self.resize_generic(Const::<R2>, Const::<C2>, val)
    }

    /// Resizes `self` such that it has dimensions `new_nrows × new_ncols`.
    ///
    /// The values are copied such that `self[(i, j)] == result[(i, j)]`. If the result has more
    /// rows and/or columns than `self`, then the extra rows or columns are filled with `val`.
    #[inline]
    pub fn resize_generic<R2: Dim, C2: Dim>(
        self,
        new_nrows: R2,
        new_ncols: C2,
        val: T,
    ) -> OMatrix<T, R2, C2>
    where
        DefaultAllocator: Reallocator<T, R, C, R2, C2>,
    {
        let (nrows, ncols) = self.shape();
        let mut data = self.into_owned();

        if new_nrows.value() == nrows {
            if new_ncols.value() < ncols {
                unsafe {
                    let num_cols_to_delete = ncols - new_ncols.value();
                    let col_ptr = data.data.ptr_mut().add(new_ncols.value() * nrows);
                    let s = ptr::slice_from_raw_parts_mut(col_ptr, num_cols_to_delete * nrows);
                    // Safety: drop the elements of the deleted columns.
                    //         these are the elements that will be truncated
                    //         by the `reallocate_copy` afterward.
                    ptr::drop_in_place(s)
                };
            }

            let res = unsafe { DefaultAllocator::reallocate_copy(new_nrows, new_ncols, data.data) };
            let mut res = Matrix::from_data(res);

            if new_ncols.value() > ncols {
                res.columns_range_mut(ncols..)
                    .fill_with(|| MaybeUninit::new(val.clone()));
            }

            // Safety: the result is now fully initialized by `reallocate_copy` and
            //         `fill_with` (if the output has more columns than the input).
            unsafe { res.assume_init() }
        } else {
            let mut res;

            unsafe {
                if new_nrows.value() < nrows {
                    compress_rows(
                        data.as_mut_slice(),
                        nrows,
                        ncols,
                        new_nrows.value(),
                        nrows - new_nrows.value(),
                    );
                    res = Matrix::from_data(DefaultAllocator::reallocate_copy(
                        new_nrows, new_ncols, data.data,
                    ));
                } else {
                    res = Matrix::from_data(DefaultAllocator::reallocate_copy(
                        new_nrows, new_ncols, data.data,
                    ));
                    extend_rows(
                        res.as_mut_slice(),
                        nrows,
                        new_ncols.value(),
                        nrows,
                        new_nrows.value() - nrows,
                    );
                }
            }

            if new_ncols.value() > ncols {
                res.columns_range_mut(ncols..)
                    .fill_with(|| MaybeUninit::new(val.clone()));
            }

            if new_nrows.value() > nrows {
                res.view_range_mut(nrows.., ..cmp::min(ncols, new_ncols.value()))
                    .fill_with(|| MaybeUninit::new(val.clone()));
            }

            // Safety: the result is now fully initialized by `reallocate_copy` and
            //         `fill_with` (whenever applicable).
            unsafe { res.assume_init() }
        }
    }

    /// Reshapes `self` such that it has dimensions `new_nrows × new_ncols`.
    ///
    /// This will reinterpret `self` as if it is a matrix with `new_nrows` rows and `new_ncols`
    /// columns. The arrangements of the component in the output matrix are the same as what
    /// would be obtained by `Matrix::from_slice_generic(self.as_slice(), new_nrows, new_ncols)`.
    ///
    /// If `self` is a dynamically-sized matrix, then its components are neither copied nor moved.
    /// If `self` is staticyll-sized, then a copy may happen in some situations.
    /// This function will panic if the given dimensions are such that the number of elements of
    /// the input matrix are not equal to the number of elements of the output matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// # use nalgebra::{Matrix3x2, Matrix2x3, DMatrix, Const, Dyn};
    ///
    /// let m1 = Matrix2x3::new(
    ///     1.1, 1.2, 1.3,
    ///     2.1, 2.2, 2.3
    /// );
    /// let m2 = Matrix3x2::new(
    ///     1.1, 2.2,
    ///     2.1, 1.3,
    ///     1.2, 2.3
    /// );
    /// let reshaped = m1.reshape_generic(Const::<3>, Const::<2>);
    /// assert_eq!(reshaped, m2);
    ///
    /// let dm1 = DMatrix::from_row_slice(
    ///     4,
    ///     3,
    ///     &[
    ///         1.0, 0.0, 0.0,
    ///         0.0, 0.0, 1.0,
    ///         0.0, 0.0, 0.0,
    ///         0.0, 1.0, 0.0
    ///     ],
    /// );
    /// let dm2 = DMatrix::from_row_slice(
    ///     6,
    ///     2,
    ///     &[
    ///         1.0, 0.0,
    ///         0.0, 1.0,
    ///         0.0, 0.0,
    ///         0.0, 1.0,
    ///         0.0, 0.0,
    ///         0.0, 0.0,
    ///     ],
    /// );
    /// let reshaped = dm1.reshape_generic(Dyn(6), Dyn(2));
    /// assert_eq!(reshaped, dm2);
    /// ```
    pub fn reshape_generic<R2, C2>(
        self,
        new_nrows: R2,
        new_ncols: C2,
    ) -> Matrix<T, R2, C2, S::Output>
    where
        R2: Dim,
        C2: Dim,
        S: ReshapableStorage<T, R, C, R2, C2>,
    {
        let data = self.data.reshape_generic(new_nrows, new_ncols);
        Matrix::from_data(data)
    }
}

/// # In-place resizing
#[cfg(any(feature = "std", feature = "alloc"))]
impl<T: Scalar> OMatrix<T, Dyn, Dyn> {
    /// Resizes this matrix in-place.
    ///
    /// The values are copied such that `self[(i, j)] == result[(i, j)]` for all valid indices.
    /// If the result has more rows and/or columns than `self`, then the extra rows or columns
    /// are filled with `val`. If the result has fewer rows or columns, elements are truncated.
    ///
    /// This function modifies the matrix in place, which can be more efficient than creating
    /// a new matrix with [`resize`](Self::resize) when working with dynamically-sized matrices.
    ///
    /// # Arguments
    /// * `new_nrows` - The desired number of rows.
    /// * `new_ncols` - The desired number of columns.
    /// * `val` - The value to fill in new elements if the matrix grows.
    ///
    /// # Examples
    ///
    /// Growing a matrix in place:
    ///
    /// ```
    /// # use nalgebra::DMatrix;
    /// let mut m = DMatrix::from_row_slice(2, 2, &[1, 2, 3, 4]);
    /// m.resize_mut(3, 3, 0);
    ///
    /// assert_eq!(m.nrows(), 3);
    /// assert_eq!(m.ncols(), 3);
    /// assert_eq!(m[(0, 0)], 1);
    /// assert_eq!(m[(2, 2)], 0); // new element
    /// ```
    ///
    /// Shrinking a matrix in place:
    ///
    /// ```
    /// # use nalgebra::DMatrix;
    /// let mut m = DMatrix::from_row_slice(3, 3, &[1, 2, 3, 4, 5, 6, 7, 8, 9]);
    /// m.resize_mut(2, 2, 0);
    ///
    /// assert_eq!(m.nrows(), 2);
    /// assert_eq!(m.ncols(), 2);
    /// assert_eq!(m[(1, 1)], 5);
    /// ```
    ///
    /// Practical use - dynamically building a matrix:
    ///
    /// ```
    /// # use nalgebra::DMatrix;
    /// let mut m = DMatrix::zeros(1, 1);
    /// for size in 2..=4 {
    ///     m.resize_mut(size, size, size);
    /// }
    /// assert_eq!(m.nrows(), 4);
    /// assert_eq!(m.ncols(), 4);
    /// ```
    ///
    /// # Note
    /// Defined only for owned fully-dynamic matrices, i.e., `DMatrix`.
    ///
    /// # See Also
    /// - [`resize`](Self::resize) - creates a new resized matrix
    /// - [`resize_vertically_mut`](Self::resize_vertically_mut) - resizes only rows in place
    /// - [`resize_horizontally_mut`](Self::resize_horizontally_mut) - resizes only columns in place
    pub fn resize_mut(&mut self, new_nrows: usize, new_ncols: usize, val: T)
    where
        DefaultAllocator: Reallocator<T, Dyn, Dyn, Dyn, Dyn>,
    {
        // TODO: avoid the clone.
        *self = self.clone().resize(new_nrows, new_ncols, val);
    }
}

#[cfg(any(feature = "std", feature = "alloc"))]
impl<T: Scalar, C: Dim> OMatrix<T, Dyn, C>
where
    DefaultAllocator: Allocator<Dyn, C>,
{
    /// Changes the number of rows of this matrix in-place.
    ///
    /// The values are copied such that `self[(i, j)] == result[(i, j)]` for all valid indices.
    /// If the result has more rows than `self`, then the extra rows are filled with `val`.
    /// If the result has fewer rows, elements are truncated.
    ///
    /// This function modifies the matrix in place, which can be more efficient than creating
    /// a new matrix with [`resize_vertically`](Self::resize_vertically).
    ///
    /// # Arguments
    /// * `new_nrows` - The desired number of rows.
    /// * `val` - The value to fill in new elements if rows are added.
    ///
    /// # Examples
    ///
    /// Growing a vector in place:
    ///
    /// ```
    /// # use nalgebra::DVector;
    /// let mut v = DVector::from_vec(vec![1, 2, 3]);
    /// v.resize_vertically_mut(5, 0);
    ///
    /// assert_eq!(v.len(), 5);
    /// assert_eq!(v[2], 3);
    /// assert_eq!(v[4], 0); // new element
    /// ```
    ///
    /// Shrinking a vector in place:
    ///
    /// ```
    /// # use nalgebra::DVector;
    /// let mut v = DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    /// v.resize_vertically_mut(3, 0.0);
    ///
    /// assert_eq!(v.len(), 3);
    /// assert_eq!(v[2], 3.0);
    /// ```
    ///
    /// Resizing a matrix with dynamic rows:
    ///
    /// ```
    /// # use nalgebra::DMatrix;
    /// let mut m = DMatrix::from_row_slice(2, 3, &[1, 2, 3, 4, 5, 6]);
    /// m.resize_vertically_mut(4, 0);
    ///
    /// assert_eq!(m.nrows(), 4);
    /// assert_eq!(m.ncols(), 3);
    /// assert_eq!(m[(3, 0)], 0);
    /// ```
    ///
    /// # Note
    /// Defined only for owned matrices with a dynamic number of rows (for example, `DVector`).
    ///
    /// # See Also
    /// - [`resize_vertically`](Self::resize_vertically) - creates a new resized matrix
    /// - [`resize_mut`](crate::Matrix::resize_mut) - resizes both dimensions in place
    /// - [`resize_horizontally_mut`](Self::resize_horizontally_mut) - resizes columns in place
    #[cfg(any(feature = "std", feature = "alloc"))]
    pub fn resize_vertically_mut(&mut self, new_nrows: usize, val: T)
    where
        DefaultAllocator: Reallocator<T, Dyn, C, Dyn, C>,
    {
        // TODO: avoid the clone.
        *self = self.clone().resize_vertically(new_nrows, val);
    }
}

#[cfg(any(feature = "std", feature = "alloc"))]
impl<T: Scalar, R: Dim> OMatrix<T, R, Dyn>
where
    DefaultAllocator: Allocator<R, Dyn>,
{
    /// Changes the number of columns of this matrix in-place.
    ///
    /// The values are copied such that `self[(i, j)] == result[(i, j)]` for all valid indices.
    /// If the result has more columns than `self`, then the extra columns are filled with `val`.
    /// If the result has fewer columns, elements are truncated.
    ///
    /// This function modifies the matrix in place, which can be more efficient than creating
    /// a new matrix with [`resize_horizontally`](Self::resize_horizontally).
    ///
    /// # Arguments
    /// * `new_ncols` - The desired number of columns.
    /// * `val` - The value to fill in new elements if columns are added.
    ///
    /// # Examples
    ///
    /// Growing a matrix horizontally in place:
    ///
    /// ```
    /// # use nalgebra::DMatrix;
    /// let mut m = DMatrix::from_row_slice(2, 2, &[1, 2, 3, 4]);
    /// m.resize_horizontally_mut(4, 0);
    ///
    /// assert_eq!(m.nrows(), 2);
    /// assert_eq!(m.ncols(), 4);
    /// assert_eq!(m[(0, 1)], 2);
    /// assert_eq!(m[(0, 3)], 0); // new column element
    /// ```
    ///
    /// Shrinking a matrix horizontally in place:
    ///
    /// ```
    /// # use nalgebra::DMatrix;
    /// let mut m = DMatrix::from_row_slice(2, 4, &[
    ///     1.0, 2.0, 3.0, 4.0,
    ///     5.0, 6.0, 7.0, 8.0,
    /// ]);
    /// m.resize_horizontally_mut(2, 0.0);
    ///
    /// assert_eq!(m.nrows(), 2);
    /// assert_eq!(m.ncols(), 2);
    /// assert_eq!(m[(1, 1)], 6.0);
    /// ```
    ///
    /// Practical use - dynamically extending matrix columns:
    ///
    /// ```
    /// # use nalgebra::DMatrix;
    /// let mut m = DMatrix::from_row_slice(3, 1, &[1, 2, 3]);
    /// m.resize_horizontally_mut(3, 0);
    ///
    /// assert_eq!(m.ncols(), 3);
    /// assert_eq!(m[(0, 0)], 1);
    /// assert_eq!(m[(2, 2)], 0);
    /// ```
    ///
    /// # Note
    /// Defined only for owned matrices with a dynamic number of columns.
    ///
    /// # See Also
    /// - [`resize_horizontally`](Self::resize_horizontally) - creates a new resized matrix
    /// - [`resize_mut`](crate::Matrix::resize_mut) - resizes both dimensions in place
    /// - [`resize_vertically_mut`](Self::resize_vertically_mut) - resizes rows in place
    #[cfg(any(feature = "std", feature = "alloc"))]
    pub fn resize_horizontally_mut(&mut self, new_ncols: usize, val: T)
    where
        DefaultAllocator: Reallocator<T, R, Dyn, R, Dyn>,
    {
        // TODO: avoid the clone.
        *self = self.clone().resize_horizontally(new_ncols, val);
    }
}

// Move the elements of `data` in such a way that the matrix with
// the rows `[i, i + nremove[` deleted is represented in a contiguous
// way in `data` after this method completes.
// Every deleted element are manually dropped by this method.
unsafe fn compress_rows<T: Scalar>(
    data: &mut [T],
    nrows: usize,
    ncols: usize,
    i: usize,
    nremove: usize,
) {
    unsafe {
        let new_nrows = nrows - nremove;

        if nremove == 0 {
            return; // Nothing to remove or drop.
        }

        if new_nrows == 0 || ncols == 0 {
            // The output matrix is empty, drop everything.
            ptr::drop_in_place(data);
            return;
        }

        // Safety: because `nremove != 0`, the pointers given to `ptr::copy`
        //         won’t alias.
        let ptr_in = data.as_ptr();
        let ptr_out = data.as_mut_ptr();

        let mut curr_i = i;

        for k in 0..ncols - 1 {
            // Safety: we drop the row elements in-place because we will overwrite these
            //         entries later with the `ptr::copy`.
            let s = ptr::slice_from_raw_parts_mut(ptr_out.add(curr_i), nremove);
            ptr::drop_in_place(s);
            ptr::copy(
                ptr_in.add(curr_i + (k + 1) * nremove),
                ptr_out.add(curr_i),
                new_nrows,
            );

            curr_i += new_nrows;
        }

        /*
         * Deal with the last column from which less values have to be copied.
         */
        // Safety: we drop the row elements in-place because we will overwrite these
        //         entries later with the `ptr::copy`.
        let s = ptr::slice_from_raw_parts_mut(ptr_out.add(curr_i), nremove);
        ptr::drop_in_place(s);
        let remaining_len = nrows - i - nremove;
        ptr::copy(
            ptr_in.add(nrows * ncols - remaining_len),
            ptr_out.add(curr_i),
            remaining_len,
        );
    }
}

// Moves entries of a matrix buffer to make place for `ninsert` empty rows starting at the `i-th` row index.
// The `data` buffer is assumed to contained at least `(nrows + ninsert) * ncols` elements.
unsafe fn extend_rows<T>(data: &mut [T], nrows: usize, ncols: usize, i: usize, ninsert: usize) {
    unsafe {
        let new_nrows = nrows + ninsert;

        if new_nrows == 0 || ncols == 0 {
            return; // Nothing to do as the output matrix is empty.
        }

        let ptr_in = data.as_ptr();
        let ptr_out = data.as_mut_ptr();

        let remaining_len = nrows - i;
        let mut curr_i = new_nrows * ncols - remaining_len;

        // Deal with the last column from which less values have to be copied.
        ptr::copy(
            ptr_in.add(nrows * ncols - remaining_len),
            ptr_out.add(curr_i),
            remaining_len,
        );

        for k in (0..ncols - 1).rev() {
            curr_i -= new_nrows;

            ptr::copy(ptr_in.add(k * nrows + i), ptr_out.add(curr_i), nrows);
        }
    }
}

/// Extend the number of columns of the `Matrix` with elements from
/// a given iterator.
#[cfg(any(feature = "std", feature = "alloc"))]
impl<T, R, S> Extend<T> for Matrix<T, R, Dyn, S>
where
    T: Scalar,
    R: Dim,
    S: Extend<T>,
{
    /// Extend the number of columns of the `Matrix` with elements
    /// from the given iterator.
    ///
    /// # Example
    /// ```
    /// # use nalgebra::{DMatrix, Dyn, Matrix, OMatrix, Matrix3};
    ///
    /// let data = vec![0, 1, 2,      // column 1
    ///                 3, 4, 5];     // column 2
    ///
    /// let mut matrix = DMatrix::from_vec(3, 2, data);
    ///
    /// matrix.extend(vec![6, 7, 8]); // column 3
    ///
    /// assert!(matrix.eq(&Matrix3::new(0, 3, 6,
    ///                                 1, 4, 7,
    ///                                 2, 5, 8)));
    /// ```
    ///
    /// # Panics
    /// This function panics if the number of elements yielded by the
    /// given iterator is not a multiple of the number of rows of the
    /// `Matrix`.
    ///
    /// ```should_panic
    /// # use nalgebra::{DMatrix, Dyn, OMatrix};
    /// let data = vec![0, 1, 2,  // column 1
    ///                 3, 4, 5]; // column 2
    ///
    /// let mut matrix = DMatrix::from_vec(3, 2, data);
    ///
    /// // The following panics because the vec length is not a multiple of 3.
    /// matrix.extend(vec![6, 7, 8, 9]);
    /// ```
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        self.data.extend(iter);
    }
}

/// Extend the number of rows of the `Vector` with elements from
/// a given iterator.
#[cfg(any(feature = "std", feature = "alloc"))]
impl<T, S> Extend<T> for Matrix<T, Dyn, U1, S>
where
    T: Scalar,
    S: Extend<T>,
{
    /// Extend the number of rows of a `Vector` with elements
    /// from the given iterator.
    ///
    /// # Example
    /// ```
    /// # use nalgebra::DVector;
    /// let mut vector = DVector::from_vec(vec![0, 1, 2]);
    /// vector.extend(vec![3, 4, 5]);
    /// assert!(vector.eq(&DVector::from_vec(vec![0, 1, 2, 3, 4, 5])));
    /// ```
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        self.data.extend(iter);
    }
}

#[cfg(any(feature = "std", feature = "alloc"))]
impl<T, R, S, RV, SV> Extend<Vector<T, RV, SV>> for Matrix<T, R, Dyn, S>
where
    T: Scalar,
    R: Dim,
    S: Extend<Vector<T, RV, SV>>,
    RV: Dim,
    SV: RawStorage<T, RV>,
    ShapeConstraint: SameNumberOfRows<R, RV>,
{
    /// Extends the number of columns of a `Matrix` with `Vector`s
    /// from a given iterator.
    ///
    /// # Example
    /// ```
    /// # use nalgebra::{DMatrix, Vector3, Matrix3x4};
    ///
    /// let data = vec![0, 1, 2,          // column 1
    ///                 3, 4, 5];         // column 2
    ///
    /// let mut matrix = DMatrix::from_vec(3, 2, data);
    ///
    /// matrix.extend(
    ///   vec![Vector3::new(6,  7,  8),   // column 3
    ///        Vector3::new(9, 10, 11)]); // column 4
    ///
    /// assert!(matrix.eq(&Matrix3x4::new(0, 3, 6,  9,
    ///                                   1, 4, 7, 10,
    ///                                   2, 5, 8, 11)));
    /// ```
    ///
    /// # Panics
    /// This function panics if the dimension of each `Vector` yielded
    /// by the given iterator is not equal to the number of rows of
    /// this `Matrix`.
    ///
    /// ```should_panic
    /// # use nalgebra::{DMatrix, Vector2, Matrix3x4};
    /// let mut matrix =
    ///   DMatrix::from_vec(3, 2,
    ///                     vec![0, 1, 2,   // column 1
    ///                          3, 4, 5]); // column 2
    ///
    /// // The following panics because this matrix can only be extended with 3-dimensional vectors.
    /// matrix.extend(
    ///   vec![Vector2::new(6,  7)]); // too few dimensions!
    /// ```
    ///
    /// ```should_panic
    /// # use nalgebra::{DMatrix, Vector4, Matrix3x4};
    /// let mut matrix =
    ///   DMatrix::from_vec(3, 2,
    ///                     vec![0, 1, 2,   // column 1
    ///                          3, 4, 5]); // column 2
    ///
    /// // The following panics because this matrix can only be extended with 3-dimensional vectors.
    /// matrix.extend(
    ///   vec![Vector4::new(6, 7, 8, 9)]); // too few dimensions!
    /// ```
    fn extend<I: IntoIterator<Item = Vector<T, RV, SV>>>(&mut self, iter: I) {
        self.data.extend(iter);
    }
}
