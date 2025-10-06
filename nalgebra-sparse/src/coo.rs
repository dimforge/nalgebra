//! An implementation of the COO sparse matrix format.

#[cfg(feature = "serde-serialize")]
mod coo_serde;

use crate::SparseFormatError;

/// A COO representation of a sparse matrix.
///
/// A COO matrix stores entries in coordinate-form, that is triplets `(i, j, v)`, where `i` and `j`
/// correspond to row and column indices of the entry, and `v` to the value of the entry.
/// The format is of limited use for standard matrix operations. Its main purpose is to facilitate
/// easy construction of other, more efficient matrix formats (such as CSR/COO), and the
/// conversion between different formats.
///
/// # Format
///
/// For given dimensions `nrows` and `ncols`, the matrix is represented by three same-length
/// arrays `row_indices`, `col_indices` and `values` that constitute the coordinate triplets
/// of the matrix. The indices must be in bounds, but *duplicate entries are explicitly allowed*.
/// Upon conversion to other formats, the duplicate entries may be summed together. See the
/// documentation for the respective conversion functions.
///
/// # Examples
///
/// ```
/// use nalgebra_sparse::{coo::CooMatrix, csr::CsrMatrix, csc::CscMatrix};
///
/// // Initialize a matrix with all zeros (no explicitly stored entries).
/// let mut coo = CooMatrix::new(4, 4);
/// // Or initialize it with a set of triplets
/// coo = CooMatrix::try_from_triplets(4, 4, vec![1, 2], vec![0, 1], vec![3.0, 4.0]).unwrap();
///
/// // Push a few triplets
/// coo.push(2, 0, 1.0);
/// coo.push(0, 1, 2.0);
///
/// // Convert to other matrix formats
/// let csr = CsrMatrix::from(&coo);
/// let csc = CscMatrix::from(&coo);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CooMatrix<T> {
    nrows: usize,
    ncols: usize,
    row_indices: Vec<usize>,
    col_indices: Vec<usize>,
    values: Vec<T>,
}

impl<T: na::Scalar> CooMatrix<T> {
    /// Pushes a dense matrix into the sparse one.
    ///
    /// This adds the dense matrix `m` starting at the `r`th row and `c`th column
    /// to the matrix.
    ///
    /// Panics
    /// ------
    ///
    /// Panics if any part of the dense matrix is out of bounds of the sparse matrix
    /// when inserted at `(r, c)`.
    #[inline]
    pub fn push_matrix<R: na::Dim, C: na::Dim, S: nalgebra::storage::RawStorage<T, R, C>>(
        &mut self,
        r: usize,
        c: usize,
        m: &na::Matrix<T, R, C, S>,
    ) {
        let block_nrows = m.nrows();
        let block_ncols = m.ncols();
        let max_row_with_block = r + block_nrows - 1;
        let max_col_with_block = c + block_ncols - 1;
        assert!(max_row_with_block < self.nrows);
        assert!(max_col_with_block < self.ncols);

        self.reserve(block_ncols * block_nrows);

        for (col_idx, col) in m.column_iter().enumerate() {
            for (row_idx, v) in col.iter().enumerate() {
                self.row_indices.push(r + row_idx);
                self.col_indices.push(c + col_idx);
                self.values.push(v.clone());
            }
        }
    }
}

impl<T> CooMatrix<T> {
    /// Construct a zero COO matrix of the given dimensions.
    ///
    /// Specifically, the collection of triplets - corresponding to explicitly stored entries -
    /// is empty, so that the matrix (implicitly) represented by the COO matrix consists of all
    /// zero entries.
    pub fn new(nrows: usize, ncols: usize) -> Self {
        Self {
            nrows,
            ncols,
            row_indices: Vec::new(),
            col_indices: Vec::new(),
            values: Vec::new(),
        }
    }

    /// Construct a zero COO matrix of the given dimensions.
    ///
    /// Specifically, the collection of triplets - corresponding to explicitly stored entries -
    /// is empty, so that the matrix (implicitly) represented by the COO matrix consists of all
    /// zero entries.
    pub fn zeros(nrows: usize, ncols: usize) -> Self {
        Self::new(nrows, ncols)
    }

    /// Try to construct a COO matrix from the given dimensions and a collection of
    /// (i, j, v) triplets.
    ///
    /// Returns an error if either row or column indices contain indices out of bounds,
    /// or if the data arrays do not all have the same length. Note that the COO format
    /// inherently supports duplicate entries.
    pub fn try_from_triplets(
        nrows: usize,
        ncols: usize,
        row_indices: Vec<usize>,
        col_indices: Vec<usize>,
        values: Vec<T>,
    ) -> Result<Self, SparseFormatError> {
        use crate::SparseFormatErrorKind::*;
        if row_indices.len() != col_indices.len() {
            return Err(SparseFormatError::from_kind_and_msg(
                InvalidStructure,
                "Number of row and col indices must be the same.",
            ));
        } else if col_indices.len() != values.len() {
            return Err(SparseFormatError::from_kind_and_msg(
                InvalidStructure,
                "Number of col indices and values must be the same.",
            ));
        }

        let row_indices_in_bounds = row_indices.iter().all(|i| *i < nrows);
        let col_indices_in_bounds = col_indices.iter().all(|j| *j < ncols);

        if !row_indices_in_bounds {
            Err(SparseFormatError::from_kind_and_msg(
                IndexOutOfBounds,
                "Row index out of bounds.",
            ))
        } else if !col_indices_in_bounds {
            Err(SparseFormatError::from_kind_and_msg(
                IndexOutOfBounds,
                "Col index out of bounds.",
            ))
        } else {
            Ok(Self {
                nrows,
                ncols,
                row_indices,
                col_indices,
                values,
            })
        }
    }

    /// Try to construct a COO matrix from the given dimensions and a finite iterator of
    /// (i, j, v) triplets.
    ///
    /// Returns an error if either row or column indices contain indices out of bounds.
    /// Note that the COO format inherently supports duplicate entries, but they are not
    /// eagerly summed.
    ///
    /// Implementation note:
    /// Calls try_from_triplets so each value is scanned twice.
    pub fn try_from_triplets_iter(
        nrows: usize,
        ncols: usize,
        triplets: impl IntoIterator<Item = (usize, usize, T)>,
    ) -> Result<Self, SparseFormatError> {
        let (row_indices, (col_indices, values)) =
            triplets.into_iter().map(|(r, c, v)| (r, (c, v))).unzip();
        Self::try_from_triplets(nrows, ncols, row_indices, col_indices, values)
    }

    /// An iterator over triplets (i, j, v).
    // TODO: Consider giving the iterator a concrete type instead of impl trait...?
    pub fn triplet_iter(&self) -> impl Iterator<Item = (usize, usize, &T)> {
        self.row_indices
            .iter()
            .zip(&self.col_indices)
            .zip(&self.values)
            .map(|((i, j), v)| (*i, *j, v))
    }

    /// A mutable iterator over triplets (i, j, v).
    // TODO: Consider giving the iterator a concrete type instead of impl trait...?
    pub fn triplet_iter_mut(&mut self) -> impl Iterator<Item = (usize, usize, &mut T)> {
        self.row_indices
            .iter()
            .zip(&self.col_indices)
            .zip(self.values.iter_mut())
            .map(|((i, j), v)| (*i, *j, v))
    }

    /// Reserves capacity for COO matrix by at least `additional` elements.
    ///
    /// This increase the capacities of triplet holding arrays by reserving more space to avoid
    /// frequent reallocations in `push` operations.
    ///
    /// ## Panics
    ///
    /// Panics if any of the individual allocation of triplet arrays fails.
    ///
    /// ## Example
    ///
    /// ```
    /// # use nalgebra_sparse::coo::CooMatrix;
    /// let mut coo = CooMatrix::new(4, 4);
    /// // Reserve capacity in advance
    /// coo.reserve(10);
    /// coo.push(1, 0, 3.0);
    /// ```
    pub fn reserve(&mut self, additional: usize) {
        self.row_indices.reserve(additional);
        self.col_indices.reserve(additional);
        self.values.reserve(additional);
    }

    /// Push a single triplet to the matrix.
    ///
    /// This adds the value `v` to the `i`th row and `j`th column in the matrix.
    ///
    /// Panics
    /// ------
    ///
    /// Panics if `i` or `j` is out of bounds.
    #[inline]
    pub fn push(&mut self, i: usize, j: usize, v: T) {
        assert!(i < self.nrows);
        assert!(j < self.ncols);
        self.row_indices.push(i);
        self.col_indices.push(j);
        self.values.push(v);
    }

    /// Clear all triplets from the matrix.
    pub fn clear_triplets(&mut self) {
        self.col_indices.clear();
        self.row_indices.clear();
        self.values.clear();
    }

    /// The number of rows in the matrix.
    #[inline]
    #[must_use]
    pub fn nrows(&self) -> usize {
        self.nrows
    }

    /// The number of columns in the matrix.
    #[inline]
    #[must_use]
    pub fn ncols(&self) -> usize {
        self.ncols
    }

    /// The number of explicitly stored entries in the matrix.
    ///
    /// This number *includes* duplicate entries. For example, if the `CooMatrix` contains duplicate
    /// entries, then it may have a different number of non-zeros as reported by `nnz()` compared
    /// to its CSR representation.
    #[inline]
    #[must_use]
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// The row indices of the explicitly stored entries.
    #[must_use]
    pub fn row_indices(&self) -> &[usize] {
        &self.row_indices
    }

    /// The column indices of the explicitly stored entries.
    #[must_use]
    pub fn col_indices(&self) -> &[usize] {
        &self.col_indices
    }

    /// The values of the explicitly stored entries.
    #[must_use]
    pub fn values(&self) -> &[T] {
        &self.values
    }

    /// Disassembles the matrix into individual triplet arrays.
    ///
    /// Examples
    /// --------
    ///
    /// ```
    /// # use nalgebra_sparse::coo::CooMatrix;
    /// let row_indices = vec![0, 1];
    /// let col_indices = vec![1, 2];
    /// let values = vec![1.0, 2.0];
    /// let coo = CooMatrix::try_from_triplets(2, 3, row_indices, col_indices, values)
    ///     .unwrap();
    ///
    /// let (row_idx, col_idx, val) = coo.disassemble();
    /// assert_eq!(row_idx, vec![0, 1]);
    /// assert_eq!(col_idx, vec![1, 2]);
    /// assert_eq!(val, vec![1.0, 2.0]);
    /// ```
    pub fn disassemble(self) -> (Vec<usize>, Vec<usize>, Vec<T>) {
        (self.row_indices, self.col_indices, self.values)
    }

    /// Removes the `i`th row from the matrix. Beware the cost of the operation is `O(nnz)` and
    /// causes a reallocation.
    ///
    /// Panics
    /// -------
    /// Panics if `i >= nrows`.
    ///
    /// Examples
    /// --------
    ///
    /// ```
    /// # use nalgebra_sparse::coo::CooMatrix;
    /// let row_indices = vec![0, 1];
    /// let col_indices = vec![1, 2];
    /// let values = vec![1.0, 2.0];
    /// let mut coo = CooMatrix::try_from_triplets(2, 3, row_indices, col_indices, values)
    ///     .unwrap();
    ///
    /// coo.remove_row(0);
    /// ```
    pub fn remove_row(&self, i: usize) -> Self
    where
        T: Copy,
    {
        assert!(i < self.nrows);
        let ((new_row_indices, new_col_indices), new_values) =
            self.remove_helper(|((row, _), _)| row != i);
        Self {
            nrows: self.nrows - 1,
            ncols: self.ncols,
            row_indices: new_row_indices,
            col_indices: new_col_indices,
            values: new_values,
        }
    }

    /// Removes the `i`th column from the matrix. Beware the cost of the operation is `O(nnz)` and
    /// causes a reallocation.
    ///
    /// Panics
    /// -------
    /// Panics if `i >= ncols`.
    ///
    /// Examples
    /// --------
    ///
    /// ```
    /// # use nalgebra_sparse::coo::CooMatrix;
    /// let row_indices = vec![0, 1];
    /// let col_indices = vec![1, 2];
    /// let values = vec![1.0, 2.0];
    /// let mut coo = CooMatrix::try_from_triplets(2, 3, row_indices, col_indices, values)
    ///     .unwrap();
    ///
    /// coo.remove_column(0);
    /// ```
    pub fn remove_column(&self, i: usize) -> Self
    where
        T: Copy,
    {
        assert!(i < self.ncols);
        let ((new_row_indices, new_col_indices), new_values) =
            self.remove_helper(|((_, col), _)| col != i);
        Self {
            nrows: self.nrows,
            ncols: self.ncols - 1,
            row_indices: new_row_indices,
            col_indices: new_col_indices,
            values: new_values,
        }
    }

    /// Removes the `i`th row and the `j`th column from the matrix. Beware the cost of the operation
    /// is `O(nnz)` and causes a reallocation. Note that a reallocation can be saved calling this
    /// function rather than successive calls to `remove_row` or `remove_column`.
    ///
    /// Panics
    /// -------
    /// Panics if `i >= nrows` or `j >= ncols`.
    ///
    /// Examples
    /// --------
    ///
    /// ```
    /// # use nalgebra_sparse::coo::CooMatrix;
    /// let row_indices = vec![0, 1];
    /// let col_indices = vec![1, 2];
    /// let values = vec![1.0, 2.0];
    /// let mut coo = CooMatrix::try_from_triplets(2, 3, row_indices, col_indices, values)
    ///     .unwrap();
    ///
    /// coo.remove_row_column(0, 1);
    /// ```
    pub fn remove_row_column(&self, i: usize, j: usize) -> Self
    where
        T: Copy,
    {
        assert!(i < self.nrows);
        assert!(j < self.ncols);
        let ((new_row_indices, new_col_indices), new_values) =
            self.remove_helper(|((row, col), _)| row != i && col != j);
        Self {
            nrows: self.nrows - 1,
            ncols: self.ncols - 1,
            row_indices: new_row_indices,
            col_indices: new_col_indices,
            values: new_values,
        }
    }

    // Helper function for all remove_x functions
    #[inline]
    fn remove_helper<F>(&self, filter_fn: F) -> ((Vec<usize>, Vec<usize>), Vec<T>)
    where
        F: Fn(((usize, usize), T)) -> bool,
        T: Copy,
    {
        self.row_indices
            .iter()
            .zip(self.col_indices.iter())
            .zip(self.values.iter())
            .filter(|((i, j), v)| filter_fn(((**i, **j), **v)))
            .unzip()
    }
}
