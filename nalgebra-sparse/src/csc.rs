//! An implementation of the CSC sparse matrix format.
//!
//! This is the module-level documentation. See [`CscMatrix`] for the main documentation of the
//! CSC implementation.

#[cfg(feature = "serde-serialize")]
mod csc_serde;

use crate::cs;
use crate::cs::{CsLane, CsLaneIter, CsLaneIterMut, CsLaneMut, CsMatrix};
use crate::csr::CsrMatrix;
use crate::pattern::{SparsityPattern, SparsityPatternFormatError, SparsityPatternIter};
use crate::{SparseEntry, SparseEntryMut, SparseFormatError, SparseFormatErrorKind};

use nalgebra::Scalar;
use num_traits::One;
use std::slice::{Iter, IterMut};

/// A CSC representation of a sparse matrix.
///
/// The Compressed Sparse Column (CSC) format is well-suited as a general-purpose storage format
/// for many sparse matrix applications.
///
/// # Usage
///
/// ```
/// use nalgebra_sparse::coo::CooMatrix;
/// use nalgebra_sparse::csc::CscMatrix;
/// use nalgebra::{DMatrix, Matrix3x4};
/// use matrixcompare::assert_matrix_eq;
///
/// // The sparsity patterns of CSC matrices are immutable. This means that you cannot dynamically
/// // change the sparsity pattern of the matrix after it has been constructed. The easiest
/// // way to construct a CSC matrix is to first incrementally construct a COO matrix,
/// // and then convert it to CSC.
///
/// let mut coo = CooMatrix::<f64>::new(3, 3);
/// coo.push(2, 0, 1.0);
/// let csc = CscMatrix::from(&coo);
///
/// // Alternatively, a CSC matrix can be constructed directly from raw CSC data.
/// // Here, we construct a 3x4 matrix
/// let col_offsets = vec![0, 1, 3, 4, 5];
/// let row_indices = vec![0, 0, 2, 2, 0];
/// let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
///
/// // The dense representation of the CSC data, for comparison
/// let dense = Matrix3x4::new(1.0, 2.0, 0.0, 5.0,
///                            0.0, 0.0, 0.0, 0.0,
///                            0.0, 3.0, 4.0, 0.0);
///
/// // The constructor validates the raw CSC data and returns an error if it is invalid.
/// let csc = CscMatrix::try_from_csc_data(3, 4, col_offsets, row_indices, values)
///     .expect("CSC data must conform to format specifications");
/// assert_matrix_eq!(csc, dense);
///
/// // A third approach is to construct a CSC matrix from a pattern and values. Sometimes this is
/// // useful if the sparsity pattern is constructed separately from the values of the matrix.
/// let (pattern, values) = csc.into_pattern_and_values();
/// let csc = CscMatrix::try_from_pattern_and_values(pattern, values)
///     .expect("The pattern and values must be compatible");
///
/// // Once we have constructed our matrix, we can use it for arithmetic operations together with
/// // other CSC matrices and dense matrices/vectors.
/// let x = csc;
/// # #[allow(non_snake_case)]
/// let xTx = x.transpose() * &x;
/// let z = DMatrix::from_fn(4, 8, |i, j| (i as f64) * (j as f64));
/// let w = 3.0 * xTx * z;
///
/// // Although the sparsity pattern of a CSC matrix cannot be changed, its values can.
/// // Here are two different ways to scale all values by a constant:
/// let mut x = x;
/// x *= 5.0;
/// x.values_mut().iter_mut().for_each(|x_i| *x_i *= 5.0);
/// ```
///
/// # Format
///
/// An `m x n` sparse matrix with `nnz` non-zeros in CSC format is represented by the
/// following three arrays:
///
/// - `col_offsets`, an array of integers with length `n + 1`.
/// - `row_indices`, an array of integers with length `nnz`.
/// - `values`, an array of values with length `nnz`.
///
/// The relationship between the arrays is described below.
///
/// - Each consecutive pair of entries `col_offsets[j] .. col_offsets[j + 1]` corresponds to an
///   offset range in `row_indices` that holds the row indices in column `j`.
/// - For an entry represented by the index `idx`, `row_indices[idx]` stores its column index and
///   `values[idx]` stores its value.
///
/// The following invariants must be upheld and are enforced by the data structure:
///
/// - `col_offsets[0] == 0`
/// - `col_offsets[m] == nnz`
/// - `col_offsets` is monotonically increasing.
/// - `0 <= row_indices[idx] < m` for all `idx < nnz`.
/// - The row indices associated with each column are monotonically increasing (see below).
///
/// The CSC format is a standard sparse matrix format (see [Wikipedia article]). The format
/// represents the matrix in a column-by-column fashion. The entries associated with column `j` are
/// determined as follows:
///
/// ```
/// # let col_offsets: Vec<usize> = vec![0, 0];
/// # let row_indices: Vec<usize> = vec![];
/// # let values: Vec<i32> = vec![];
/// # let j = 0;
/// let range = col_offsets[j] .. col_offsets[j + 1];
/// let col_j_rows = &row_indices[range.clone()];
/// let col_j_vals = &values[range];
///
/// // For each pair (i, v) in (col_j_rows, col_j_vals), we obtain a corresponding entry
/// // (i, j, v) in the matrix.
/// assert_eq!(col_j_rows.len(), col_j_vals.len());
/// ```
///
/// In the above example, for each column `j`, the row indices `col_j_cols` must appear in
/// monotonically increasing order. In other words, they must be *sorted*. This criterion is not
/// standard among all sparse matrix libraries, but we enforce this property as it is a crucial
/// assumption for both correctness and performance for many algorithms.
///
/// Note that the CSR and CSC formats are essentially identical, except that CSC stores the matrix
/// column-by-column instead of row-by-row like CSR.
///
/// [Wikipedia article]: https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_column_(CSC_or_CCS)
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CscMatrix<T> {
    // Cols are major, rows are minor in the sparsity pattern
    pub(crate) cs: CsMatrix<T>,
}

impl<T> CscMatrix<T> {
    /// Constructs a CSC representation of the (square) `n x n` identity matrix.
    #[inline]
    pub fn identity(n: usize) -> Self
    where
        T: Scalar + One,
    {
        Self {
            cs: CsMatrix::identity(n),
        }
    }

    /// Create a zero CSC matrix with no explicitly stored entries.
    pub fn zeros(nrows: usize, ncols: usize) -> Self {
        Self {
            cs: CsMatrix::new(ncols, nrows),
        }
    }

    /// Try to construct a CSC matrix from raw CSC data.
    ///
    /// It is assumed that each column contains unique and sorted row indices that are in
    /// bounds with respect to the number of rows in the matrix. If this is not the case,
    /// an error is returned to indicate the failure.
    ///
    /// An error is returned if the data given does not conform to the CSC storage format.
    /// See the documentation for [`CscMatrix`] for more information.
    pub fn try_from_csc_data(
        num_rows: usize,
        num_cols: usize,
        col_offsets: Vec<usize>,
        row_indices: Vec<usize>,
        values: Vec<T>,
    ) -> Result<Self, SparseFormatError> {
        let pattern = SparsityPattern::try_from_offsets_and_indices(
            num_cols,
            num_rows,
            col_offsets,
            row_indices,
        )
        .map_err(pattern_format_error_to_csc_error)?;
        Self::try_from_pattern_and_values(pattern, values)
    }

    /// Try to construct a CSC matrix from raw CSC data with unsorted row indices.
    ///
    /// It is assumed that each column contains unique row indices that are in
    /// bounds with respect to the number of rows in the matrix. If this is not the case,
    /// an error is returned to indicate the failure.
    ///
    /// An error is returned if the data given does not conform to the CSC storage format
    /// with the exception of having unsorted row indices and values.
    /// See the documentation for [`CscMatrix`] for more information.
    pub fn try_from_unsorted_csc_data(
        num_rows: usize,
        num_cols: usize,
        col_offsets: Vec<usize>,
        mut row_indices: Vec<usize>,
        mut values: Vec<T>,
    ) -> Result<Self, SparseFormatError>
    where
        T: Scalar,
    {
        let result = cs::validate_and_optionally_sort_cs_data(
            num_cols,
            num_rows,
            &col_offsets,
            &mut row_indices,
            Some(&mut values),
            true,
        );

        match result {
            Ok(()) => {
                let pattern = unsafe {
                    SparsityPattern::from_offset_and_indices_unchecked(
                        num_cols,
                        num_rows,
                        col_offsets,
                        row_indices,
                    )
                };
                Self::try_from_pattern_and_values(pattern, values)
            }
            Err(err) => Err(err),
        }
    }

    /// Try to construct a CSC matrix from a sparsity pattern and associated non-zero values.
    ///
    /// Returns an error if the number of values does not match the number of minor indices
    /// in the pattern.
    pub fn try_from_pattern_and_values(
        pattern: SparsityPattern,
        values: Vec<T>,
    ) -> Result<Self, SparseFormatError> {
        if pattern.nnz() == values.len() {
            Ok(Self {
                cs: CsMatrix::from_pattern_and_values(pattern, values),
            })
        } else {
            Err(SparseFormatError::from_kind_and_msg(
                SparseFormatErrorKind::InvalidStructure,
                "Number of values and row indices must be the same",
            ))
        }
    }

    /// The number of rows in the matrix.
    #[inline]
    #[must_use]
    pub fn nrows(&self) -> usize {
        self.cs.pattern().minor_dim()
    }

    /// The number of columns in the matrix.
    #[inline]
    #[must_use]
    pub fn ncols(&self) -> usize {
        self.cs.pattern().major_dim()
    }

    /// The number of non-zeros in the matrix.
    ///
    /// Note that this corresponds to the number of explicitly stored entries, *not* the actual
    /// number of algebraically zero entries in the matrix. Explicitly stored entries can still
    /// be zero. Corresponds to the number of entries in the sparsity pattern.
    #[inline]
    #[must_use]
    pub fn nnz(&self) -> usize {
        self.pattern().nnz()
    }

    /// The column offsets defining part of the CSC format.
    #[inline]
    #[must_use]
    pub fn col_offsets(&self) -> &[usize] {
        self.pattern().major_offsets()
    }

    /// The row indices defining part of the CSC format.
    #[inline]
    #[must_use]
    pub fn row_indices(&self) -> &[usize] {
        self.pattern().minor_indices()
    }

    /// The non-zero values defining part of the CSC format.
    #[inline]
    #[must_use]
    pub fn values(&self) -> &[T] {
        self.cs.values()
    }

    /// Mutable access to the non-zero values.
    #[inline]
    pub fn values_mut(&mut self) -> &mut [T] {
        self.cs.values_mut()
    }

    /// An iterator over non-zero triplets (i, j, v).
    ///
    /// The iteration happens in column-major fashion, meaning that j increases monotonically,
    /// and i increases monotonically within each row.
    ///
    /// Examples
    /// --------
    /// ```
    /// # use nalgebra_sparse::csc::CscMatrix;
    /// let col_offsets = vec![0, 2, 3, 4];
    /// let row_indices = vec![0, 2, 1, 0];
    /// let values = vec![1, 3, 2, 4];
    /// let mut csc = CscMatrix::try_from_csc_data(4, 3, col_offsets, row_indices, values)
    ///     .unwrap();
    ///
    /// let triplets: Vec<_> = csc.triplet_iter().map(|(i, j, v)| (i, j, *v)).collect();
    /// assert_eq!(triplets, vec![(0, 0, 1), (2, 0, 3), (1, 1, 2), (0, 2, 4)]);
    /// ```
    pub fn triplet_iter(&self) -> CscTripletIter<'_, T> {
        CscTripletIter {
            pattern_iter: self.pattern().entries(),
            values_iter: self.values().iter(),
        }
    }

    /// A mutable iterator over non-zero triplets (i, j, v).
    ///
    /// Iteration happens in the same order as for [triplet_iter](#method.triplet_iter).
    ///
    /// Examples
    /// --------
    /// ```
    /// # use nalgebra_sparse::csc::CscMatrix;
    /// let col_offsets = vec![0, 2, 3, 4];
    /// let row_indices = vec![0, 2, 1, 0];
    /// let values = vec![1, 3, 2, 4];
    /// // Using the same data as in the `triplet_iter` example
    /// let mut csc = CscMatrix::try_from_csc_data(4, 3, col_offsets, row_indices, values)
    ///     .unwrap();
    ///
    /// // Zero out lower-triangular terms
    /// csc.triplet_iter_mut()
    ///    .filter(|(i, j, _)| j < i)
    ///    .for_each(|(_, _, v)| *v = 0);
    ///
    /// let triplets: Vec<_> = csc.triplet_iter().map(|(i, j, v)| (i, j, *v)).collect();
    /// assert_eq!(triplets, vec![(0, 0, 1), (2, 0, 0), (1, 1, 2), (0, 2, 4)]);
    /// ```
    pub fn triplet_iter_mut(&mut self) -> CscTripletIterMut<'_, T> {
        let (pattern, values) = self.cs.pattern_and_values_mut();
        CscTripletIterMut {
            pattern_iter: pattern.entries(),
            values_mut_iter: values.iter_mut(),
        }
    }

    /// Return the column at the given column index.
    ///
    /// Panics
    /// ------
    /// Panics if column index is out of bounds.
    #[inline]
    #[must_use]
    pub fn col(&self, index: usize) -> CscCol<'_, T> {
        self.get_col(index).expect("Row index must be in bounds")
    }

    /// Mutable column access for the given column index.
    ///
    /// Panics
    /// ------
    /// Panics if column index is out of bounds.
    #[inline]
    pub fn col_mut(&mut self, index: usize) -> CscColMut<'_, T> {
        self.get_col_mut(index)
            .expect("Row index must be in bounds")
    }

    /// Return the column at the given column index, or `None` if out of bounds.
    #[inline]
    #[must_use]
    pub fn get_col(&self, index: usize) -> Option<CscCol<'_, T>> {
        self.cs.get_lane(index).map(|lane| CscCol { lane })
    }

    /// Mutable column access for the given column index, or `None` if out of bounds.
    #[inline]
    #[must_use]
    pub fn get_col_mut(&mut self, index: usize) -> Option<CscColMut<'_, T>> {
        self.cs.get_lane_mut(index).map(|lane| CscColMut { lane })
    }

    /// An iterator over columns in the matrix.
    pub fn col_iter(&self) -> CscColIter<'_, T> {
        CscColIter {
            lane_iter: CsLaneIter::new(self.pattern(), self.values()),
        }
    }

    /// A mutable iterator over columns in the matrix.
    pub fn col_iter_mut(&mut self) -> CscColIterMut<'_, T> {
        let (pattern, values) = self.cs.pattern_and_values_mut();
        CscColIterMut {
            lane_iter: CsLaneIterMut::new(pattern, values),
        }
    }

    /// Disassembles the CSC matrix into its underlying offset, index and value arrays.
    ///
    /// If the matrix contains the sole reference to the sparsity pattern,
    /// then the data is returned as-is. Otherwise, the sparsity pattern is cloned.
    ///
    /// Examples
    /// --------
    ///
    /// ```
    /// # use nalgebra_sparse::csc::CscMatrix;
    /// let col_offsets = vec![0, 2, 3, 4];
    /// let row_indices = vec![0, 2, 1, 0];
    /// let values = vec![1, 3, 2, 4];
    /// let mut csc = CscMatrix::try_from_csc_data(
    ///     4,
    ///     3,
    ///     col_offsets.clone(),
    ///     row_indices.clone(),
    ///     values.clone())
    ///     .unwrap();
    /// let (col_offsets2, row_indices2, values2) = csc.disassemble();
    /// assert_eq!(col_offsets2, col_offsets);
    /// assert_eq!(row_indices2, row_indices);
    /// assert_eq!(values2, values);
    /// ```
    pub fn disassemble(self) -> (Vec<usize>, Vec<usize>, Vec<T>) {
        self.cs.disassemble()
    }

    /// Returns the sparsity pattern and values associated with this matrix.
    pub fn into_pattern_and_values(self) -> (SparsityPattern, Vec<T>) {
        self.cs.into_pattern_and_values()
    }

    /// Returns a reference to the sparsity pattern and a mutable reference to the values.
    #[inline]
    pub fn pattern_and_values_mut(&mut self) -> (&SparsityPattern, &mut [T]) {
        self.cs.pattern_and_values_mut()
    }

    /// Returns a reference to the underlying sparsity pattern.
    #[must_use]
    pub fn pattern(&self) -> &SparsityPattern {
        self.cs.pattern()
    }

    /// Reinterprets the CSC matrix as its transpose represented by a CSR matrix.
    ///
    /// This operation does not touch the CSC data, and is effectively a no-op.
    pub fn transpose_as_csr(self) -> CsrMatrix<T> {
        let (pattern, values) = self.cs.take_pattern_and_values();
        CsrMatrix::try_from_pattern_and_values(pattern, values).unwrap()
    }

    /// Returns an entry for the given row/col indices, or `None` if the indices are out of bounds.
    ///
    /// Each call to this function incurs the cost of a binary search among the explicitly
    /// stored row entries for the given column.
    #[must_use]
    pub fn get_entry(&self, row_index: usize, col_index: usize) -> Option<SparseEntry<'_, T>> {
        self.cs.get_entry(col_index, row_index)
    }

    /// Returns a mutable entry for the given row/col indices, or `None` if the indices are out
    /// of bounds.
    ///
    /// Each call to this function incurs the cost of a binary search among the explicitly
    /// stored row entries for the given column.
    pub fn get_entry_mut(
        &mut self,
        row_index: usize,
        col_index: usize,
    ) -> Option<SparseEntryMut<'_, T>> {
        self.cs.get_entry_mut(col_index, row_index)
    }

    /// Returns an entry for the given row/col indices.
    ///
    /// Same as `get_entry`, except that it directly panics upon encountering row/col indices
    /// out of bounds.
    ///
    /// Panics
    /// ------
    /// Panics if `row_index` or `col_index` is out of bounds.
    #[must_use]
    pub fn index_entry(&self, row_index: usize, col_index: usize) -> SparseEntry<'_, T> {
        self.get_entry(row_index, col_index)
            .expect("Out of bounds matrix indices encountered")
    }

    /// Returns a mutable entry for the given row/col indices.
    ///
    /// Same as `get_entry_mut`, except that it directly panics upon encountering row/col indices
    /// out of bounds.
    ///
    /// Panics
    /// ------
    /// Panics if `row_index` or `col_index` is out of bounds.
    pub fn index_entry_mut(&mut self, row_index: usize, col_index: usize) -> SparseEntryMut<'_, T> {
        self.get_entry_mut(row_index, col_index)
            .expect("Out of bounds matrix indices encountered")
    }

    /// Returns a triplet of slices `(col_offsets, row_indices, values)` that make up the CSC data.
    #[must_use]
    pub fn csc_data(&self) -> (&[usize], &[usize], &[T]) {
        self.cs.cs_data()
    }

    /// Returns a triplet of slices `(col_offsets, row_indices, values)` that make up the CSC data,
    /// where the `values` array is mutable.
    pub fn csc_data_mut(&mut self) -> (&[usize], &[usize], &mut [T]) {
        self.cs.cs_data_mut()
    }

    /// Creates a sparse matrix that contains only the explicit entries decided by the
    /// given predicate.
    #[must_use]
    pub fn filter<P>(&self, predicate: P) -> Self
    where
        T: Clone,
        P: Fn(usize, usize, &T) -> bool,
    {
        // Note: Predicate uses (row, col, value), so we have to switch around since
        // cs uses (major, minor, value)
        Self {
            cs: self
                .cs
                .filter(|col_idx, row_idx, v| predicate(row_idx, col_idx, v)),
        }
    }

    /// Returns a new matrix representing the upper triangular part of this matrix.
    ///
    /// The result includes the diagonal of the matrix.
    #[must_use]
    pub fn upper_triangle(&self) -> Self
    where
        T: Clone,
    {
        self.filter(|i, j, _| i <= j)
    }

    /// Returns a new matrix representing the lower triangular part of this matrix.
    ///
    /// The result includes the diagonal of the matrix.
    #[must_use]
    pub fn lower_triangle(&self) -> Self
    where
        T: Clone,
    {
        self.filter(|i, j, _| i >= j)
    }

    /// Returns the diagonal of the matrix as a sparse matrix.
    #[must_use]
    pub fn diagonal_as_csc(&self) -> Self
    where
        T: Clone,
    {
        Self {
            cs: self.cs.diagonal_as_matrix(),
        }
    }

    /// Compute the transpose of the matrix.
    #[must_use]
    pub fn transpose(&self) -> CscMatrix<T>
    where
        T: Scalar,
    {
        CsrMatrix::from(self).transpose_as_csc()
    }
}

impl<T> Default for CscMatrix<T> {
    fn default() -> Self {
        Self {
            cs: Default::default(),
        }
    }
}

/// Convert pattern format errors into more meaningful CSC-specific errors.
///
/// This ensures that the terminology is consistent: we are talking about rows and columns,
/// not lanes, major and minor dimensions.
fn pattern_format_error_to_csc_error(err: SparsityPatternFormatError) -> SparseFormatError {
    use SparseFormatError as E;
    use SparseFormatErrorKind as K;
    use SparsityPatternFormatError::DuplicateEntry as PatternDuplicateEntry;
    use SparsityPatternFormatError::*;

    match err {
        InvalidOffsetArrayLength => E::from_kind_and_msg(
            K::InvalidStructure,
            "Length of col offset array is not equal to ncols + 1.",
        ),
        InvalidOffsetFirstLast => E::from_kind_and_msg(
            K::InvalidStructure,
            "First or last col offset is inconsistent with format specification.",
        ),
        NonmonotonicOffsets => E::from_kind_and_msg(
            K::InvalidStructure,
            "Col offsets are not monotonically increasing.",
        ),
        NonmonotonicMinorIndices => E::from_kind_and_msg(
            K::InvalidStructure,
            "Row indices are not monotonically increasing (sorted) within each column.",
        ),
        MinorIndexOutOfBounds => {
            E::from_kind_and_msg(K::IndexOutOfBounds, "Row indices are out of bounds.")
        }
        PatternDuplicateEntry => {
            E::from_kind_and_msg(K::DuplicateEntry, "Matrix data contains duplicate entries.")
        }
    }
}

/// Iterator type for iterating over triplets in a CSC matrix.
#[derive(Debug)]
pub struct CscTripletIter<'a, T> {
    pattern_iter: SparsityPatternIter<'a>,
    values_iter: Iter<'a, T>,
}

impl<'a, T> Clone for CscTripletIter<'a, T> {
    fn clone(&self) -> Self {
        CscTripletIter {
            pattern_iter: self.pattern_iter.clone(),
            values_iter: self.values_iter.clone(),
        }
    }
}

impl<'a, T: Clone> CscTripletIter<'a, T> {
    /// Adapts the triplet iterator to return owned values.
    ///
    /// The triplet iterator returns references to the values. This method adapts the iterator
    /// so that the values are cloned.
    #[inline]
    pub fn cloned_values(self) -> impl 'a + Iterator<Item = (usize, usize, T)> {
        self.map(|(i, j, v)| (i, j, v.clone()))
    }
}

impl<'a, T> Iterator for CscTripletIter<'a, T> {
    type Item = (usize, usize, &'a T);

    fn next(&mut self) -> Option<Self::Item> {
        let next_entry = self.pattern_iter.next();
        let next_value = self.values_iter.next();

        match (next_entry, next_value) {
            (Some((i, j)), Some(v)) => Some((j, i, v)),
            _ => None,
        }
    }
}

/// Iterator type for mutably iterating over triplets in a CSC matrix.
#[derive(Debug)]
pub struct CscTripletIterMut<'a, T> {
    pattern_iter: SparsityPatternIter<'a>,
    values_mut_iter: IterMut<'a, T>,
}

impl<'a, T> Iterator for CscTripletIterMut<'a, T> {
    type Item = (usize, usize, &'a mut T);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let next_entry = self.pattern_iter.next();
        let next_value = self.values_mut_iter.next();

        match (next_entry, next_value) {
            (Some((i, j)), Some(v)) => Some((j, i, v)),
            _ => None,
        }
    }
}

/// An immutable representation of a column in a CSC matrix.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CscCol<'a, T> {
    lane: CsLane<'a, T>,
}

/// A mutable representation of a column in a CSC matrix.
///
/// Note that only explicitly stored entries can be mutated. The sparsity pattern belonging
/// to the column cannot be modified.
#[derive(Debug, PartialEq, Eq)]
pub struct CscColMut<'a, T> {
    lane: CsLaneMut<'a, T>,
}

/// Implement the methods common to both CscCol and CscColMut
macro_rules! impl_csc_col_common_methods {
    ($name:ty) => {
        impl<'a, T> $name {
            /// The number of global rows in the column.
            #[inline]
            #[must_use]
            pub fn nrows(&self) -> usize {
                self.lane.minor_dim()
            }

            /// The number of non-zeros in this column.
            #[inline]
            #[must_use]
            pub fn nnz(&self) -> usize {
                self.lane.nnz()
            }

            /// The row indices corresponding to explicitly stored entries in this column.
            #[inline]
            #[must_use]
            pub fn row_indices(&self) -> &[usize] {
                self.lane.minor_indices()
            }

            /// The values corresponding to explicitly stored entries in this column.
            #[inline]
            #[must_use]
            pub fn values(&self) -> &[T] {
                self.lane.values()
            }

            /// Returns an entry for the given global row index.
            ///
            /// Each call to this function incurs the cost of a binary search among the explicitly
            /// stored row entries.
            #[must_use]
            pub fn get_entry(&self, global_row_index: usize) -> Option<SparseEntry<'_, T>> {
                self.lane.get_entry(global_row_index)
            }
        }
    };
}

impl_csc_col_common_methods!(CscCol<'a, T>);
impl_csc_col_common_methods!(CscColMut<'a, T>);

impl<'a, T> CscColMut<'a, T> {
    /// Mutable access to the values corresponding to explicitly stored entries in this column.
    pub fn values_mut(&mut self) -> &mut [T] {
        self.lane.values_mut()
    }

    /// Provides simultaneous access to row indices and mutable values corresponding to the
    /// explicitly stored entries in this column.
    ///
    /// This method primarily facilitates low-level access for methods that process data stored
    /// in CSC format directly.
    pub fn rows_and_values_mut(&mut self) -> (&[usize], &mut [T]) {
        self.lane.indices_and_values_mut()
    }

    /// Returns a mutable entry for the given global row index.
    #[must_use]
    pub fn get_entry_mut(&mut self, global_row_index: usize) -> Option<SparseEntryMut<'_, T>> {
        self.lane.get_entry_mut(global_row_index)
    }
}

/// Column iterator for [`CscMatrix`].
pub struct CscColIter<'a, T> {
    lane_iter: CsLaneIter<'a, T>,
}

impl<'a, T> Iterator for CscColIter<'a, T> {
    type Item = CscCol<'a, T>;

    fn next(&mut self) -> Option<Self::Item> {
        self.lane_iter.next().map(|lane| CscCol { lane })
    }
}

/// Mutable column iterator for [`CscMatrix`].
pub struct CscColIterMut<'a, T> {
    lane_iter: CsLaneIterMut<'a, T>,
}

impl<'a, T> Iterator for CscColIterMut<'a, T>
where
    T: 'a,
{
    type Item = CscColMut<'a, T>;

    fn next(&mut self) -> Option<Self::Item> {
        self.lane_iter.next().map(|lane| CscColMut { lane })
    }
}
