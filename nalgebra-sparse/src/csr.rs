//! An implementation of the CSR sparse matrix format.
//!
//! This is the module-level documentation. See [`CsrMatrix`] for the main documentation of the
//! CSC implementation.

#[cfg(feature = "serde-serialize")]
mod csr_serde;

use crate::cs;
use crate::cs::{CsLane, CsLaneIter, CsLaneIterMut, CsLaneMut, CsMatrix};
use crate::csc::CscMatrix;
use crate::pattern::{SparsityPattern, SparsityPatternFormatError, SparsityPatternIter};
use crate::{SparseEntry, SparseEntryMut, SparseFormatError, SparseFormatErrorKind};

use nalgebra::Scalar;
use num_traits::One;

use std::slice::{Iter, IterMut};

/// A CSR representation of a sparse matrix.
///
/// The Compressed Sparse Row (CSR) format is well-suited as a general-purpose storage format
/// for many sparse matrix applications.
///
/// # Usage
///
/// ```
/// use nalgebra_sparse::coo::CooMatrix;
/// use nalgebra_sparse::csr::CsrMatrix;
/// use nalgebra::{DMatrix, Matrix3x4};
/// use matrixcompare::assert_matrix_eq;
///
/// // The sparsity patterns of CSR matrices are immutable. This means that you cannot dynamically
/// // change the sparsity pattern of the matrix after it has been constructed. The easiest
/// // way to construct a CSR matrix is to first incrementally construct a COO matrix,
/// // and then convert it to CSR.
///
/// let mut coo = CooMatrix::<f64>::new(3, 3);
/// coo.push(2, 0, 1.0);
/// let csr = CsrMatrix::from(&coo);
///
/// // Alternatively, a CSR matrix can be constructed directly from raw CSR data.
/// // Here, we construct a 3x4 matrix
/// let row_offsets = vec![0, 3, 3, 5];
/// let col_indices = vec![0, 1, 3, 1, 2];
/// let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
///
/// // The dense representation of the CSR data, for comparison
/// let dense = Matrix3x4::new(1.0, 2.0, 0.0, 3.0,
///                            0.0, 0.0, 0.0, 0.0,
///                            0.0, 4.0, 5.0, 0.0);
///
/// // The constructor validates the raw CSR data and returns an error if it is invalid.
/// let csr = CsrMatrix::try_from_csr_data(3, 4, row_offsets, col_indices, values)
///     .expect("CSR data must conform to format specifications");
/// assert_matrix_eq!(csr, dense);
///
/// // A third approach is to construct a CSR matrix from a pattern and values. Sometimes this is
/// // useful if the sparsity pattern is constructed separately from the values of the matrix.
/// let (pattern, values) = csr.into_pattern_and_values();
/// let csr = CsrMatrix::try_from_pattern_and_values(pattern, values)
///     .expect("The pattern and values must be compatible");
///
/// // Once we have constructed our matrix, we can use it for arithmetic operations together with
/// // other CSR matrices and dense matrices/vectors.
/// let x = csr;
/// # #[allow(non_snake_case)]
/// let xTx = x.transpose() * &x;
/// let z = DMatrix::from_fn(4, 8, |i, j| (i as f64) * (j as f64));
/// let w = 3.0 * xTx * z;
///
/// // Although the sparsity pattern of a CSR matrix cannot be changed, its values can.
/// // Here are two different ways to scale all values by a constant:
/// let mut x = x;
/// x *= 5.0;
/// x.values_mut().iter_mut().for_each(|x_i| *x_i *= 5.0);
/// ```
///
/// # Format
///
/// An `m x n` sparse matrix with `nnz` non-zeros in CSR format is represented by the
/// following three arrays:
///
/// - `row_offsets`, an array of integers with length `m + 1`.
/// - `col_indices`, an array of integers with length `nnz`.
/// - `values`, an array of values with length `nnz`.
///
/// The relationship between the arrays is described below.
///
/// - Each consecutive pair of entries `row_offsets[i] .. row_offsets[i + 1]` corresponds to an
///   offset range in `col_indices` that holds the column indices in row `i`.
/// - For an entry represented by the index `idx`, `col_indices[idx]` stores its column index and
///   `values[idx]` stores its value.
///
/// The following invariants must be upheld and are enforced by the data structure:
///
/// - `row_offsets[0] == 0`
/// - `row_offsets[m] == nnz`
/// - `row_offsets` is monotonically increasing.
/// - `0 <= col_indices[idx] < n` for all `idx < nnz`.
/// - The column indices associated with each row are monotonically increasing (see below).
///
/// The CSR format is a standard sparse matrix format (see [Wikipedia article]). The format
/// represents the matrix in a row-by-row fashion. The entries associated with row `i` are
/// determined as follows:
///
/// ```
/// # let row_offsets: Vec<usize> = vec![0, 0];
/// # let col_indices: Vec<usize> = vec![];
/// # let values: Vec<i32> = vec![];
/// # let i = 0;
/// let range = row_offsets[i] .. row_offsets[i + 1];
/// let row_i_cols = &col_indices[range.clone()];
/// let row_i_vals = &values[range];
///
/// // For each pair (j, v) in (row_i_cols, row_i_vals), we obtain a corresponding entry
/// // (i, j, v) in the matrix.
/// assert_eq!(row_i_cols.len(), row_i_vals.len());
/// ```
///
/// In the above example, for each row `i`, the column indices `row_i_cols` must appear in
/// monotonically increasing order. In other words, they must be *sorted*. This criterion is not
/// standard among all sparse matrix libraries, but we enforce this property as it is a crucial
/// assumption for both correctness and performance for many algorithms.
///
/// Note that the CSR and CSC formats are essentially identical, except that CSC stores the matrix
/// column-by-column instead of row-by-row like CSR.
///
/// [Wikipedia article]: https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_(CSR,_CRS_or_Yale_format)
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CsrMatrix<T> {
    // Rows are major, cols are minor in the sparsity pattern
    pub(crate) cs: CsMatrix<T>,
}

impl<T> CsrMatrix<T> {
    /// Constructs a CSR representation of the (square) `n x n` identity matrix.
    #[inline]
    pub fn identity(n: usize) -> Self
    where
        T: Scalar + One,
    {
        Self {
            cs: CsMatrix::identity(n),
        }
    }

    /// Create a zero CSR matrix with no explicitly stored entries.
    pub fn zeros(nrows: usize, ncols: usize) -> Self {
        Self {
            cs: CsMatrix::new(nrows, ncols),
        }
    }

    /// Try to construct a CSR matrix from raw CSR data.
    ///
    /// It is assumed that each row contains unique and sorted column indices that are in
    /// bounds with respect to the number of columns in the matrix. If this is not the case,
    /// an error is returned to indicate the failure.
    ///
    /// An error is returned if the data given does not conform to the CSR storage format.
    /// See the documentation for [`CsrMatrix`] for more information.
    pub fn try_from_csr_data(
        num_rows: usize,
        num_cols: usize,
        row_offsets: Vec<usize>,
        col_indices: Vec<usize>,
        values: Vec<T>,
    ) -> Result<Self, SparseFormatError> {
        let pattern = SparsityPattern::try_from_offsets_and_indices(
            num_rows,
            num_cols,
            row_offsets,
            col_indices,
        )
        .map_err(pattern_format_error_to_csr_error)?;
        Self::try_from_pattern_and_values(pattern, values)
    }

    /// Try to construct a CSR matrix from raw CSR data with unsorted column indices.
    ///
    /// It is assumed that each row contains unique column indices that are in
    /// bounds with respect to the number of columns in the matrix. If this is not the case,
    /// an error is returned to indicate the failure.
    ///
    /// An error is returned if the data given does not conform to the CSR storage format
    /// with the exception of having unsorted column indices and values.
    /// See the documentation for [`CsrMatrix`] for more information.
    pub fn try_from_unsorted_csr_data(
        num_rows: usize,
        num_cols: usize,
        row_offsets: Vec<usize>,
        mut col_indices: Vec<usize>,
        mut values: Vec<T>,
    ) -> Result<Self, SparseFormatError>
    where
        T: Scalar,
    {
        let result = cs::validate_and_optionally_sort_cs_data(
            num_rows,
            num_cols,
            &row_offsets,
            &mut col_indices,
            Some(&mut values),
            true,
        );

        match result {
            Ok(()) => {
                let pattern = unsafe {
                    SparsityPattern::from_offset_and_indices_unchecked(
                        num_rows,
                        num_cols,
                        row_offsets,
                        col_indices,
                    )
                };
                Self::try_from_pattern_and_values(pattern, values)
            }
            Err(err) => Err(err),
        }
    }

    /// Try to construct a CSR matrix from a sparsity pattern and associated non-zero values.
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
                "Number of values and column indices must be the same",
            ))
        }
    }

    /// The number of rows in the matrix.
    #[inline]
    #[must_use]
    pub fn nrows(&self) -> usize {
        self.cs.pattern().major_dim()
    }

    /// The number of columns in the matrix.
    #[inline]
    #[must_use]
    pub fn ncols(&self) -> usize {
        self.cs.pattern().minor_dim()
    }

    /// The number of non-zeros in the matrix.
    ///
    /// Note that this corresponds to the number of explicitly stored entries, *not* the actual
    /// number of algebraically zero entries in the matrix. Explicitly stored entries can still
    /// be zero. Corresponds to the number of entries in the sparsity pattern.
    #[inline]
    #[must_use]
    pub fn nnz(&self) -> usize {
        self.cs.pattern().nnz()
    }

    /// The row offsets defining part of the CSR format.
    #[inline]
    #[must_use]
    pub fn row_offsets(&self) -> &[usize] {
        let (offsets, _, _) = self.cs.cs_data();
        offsets
    }

    /// The column indices defining part of the CSR format.
    #[inline]
    #[must_use]
    pub fn col_indices(&self) -> &[usize] {
        let (_, indices, _) = self.cs.cs_data();
        indices
    }

    /// The non-zero values defining part of the CSR format.
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
    /// The iteration happens in row-major fashion, meaning that i increases monotonically,
    /// and j increases monotonically within each row.
    ///
    /// Examples
    /// --------
    /// ```
    /// # use nalgebra_sparse::csr::CsrMatrix;
    /// let row_offsets = vec![0, 2, 3, 4];
    /// let col_indices = vec![0, 2, 1, 0];
    /// let values = vec![1, 2, 3, 4];
    /// let mut csr = CsrMatrix::try_from_csr_data(3, 4, row_offsets, col_indices, values)
    ///     .unwrap();
    ///
    /// let triplets: Vec<_> = csr.triplet_iter().map(|(i, j, v)| (i, j, *v)).collect();
    /// assert_eq!(triplets, vec![(0, 0, 1), (0, 2, 2), (1, 1, 3), (2, 0, 4)]);
    /// ```
    pub fn triplet_iter(&self) -> CsrTripletIter<'_, T> {
        CsrTripletIter {
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
    /// # use nalgebra_sparse::csr::CsrMatrix;
    /// # let row_offsets = vec![0, 2, 3, 4];
    /// # let col_indices = vec![0, 2, 1, 0];
    /// # let values = vec![1, 2, 3, 4];
    /// // Using the same data as in the `triplet_iter` example
    /// let mut csr = CsrMatrix::try_from_csr_data(3, 4, row_offsets, col_indices, values)
    ///     .unwrap();
    ///
    /// // Zero out lower-triangular terms
    /// csr.triplet_iter_mut()
    ///    .filter(|(i, j, _)| j < i)
    ///    .for_each(|(_, _, v)| *v = 0);
    ///
    /// let triplets: Vec<_> = csr.triplet_iter().map(|(i, j, v)| (i, j, *v)).collect();
    /// assert_eq!(triplets, vec![(0, 0, 1), (0, 2, 2), (1, 1, 3), (2, 0, 0)]);
    /// ```
    pub fn triplet_iter_mut(&mut self) -> CsrTripletIterMut<'_, T> {
        let (pattern, values) = self.cs.pattern_and_values_mut();
        CsrTripletIterMut {
            pattern_iter: pattern.entries(),
            values_mut_iter: values.iter_mut(),
        }
    }

    /// Return the row at the given row index.
    ///
    /// Panics
    /// ------
    /// Panics if row index is out of bounds.
    #[inline]
    #[must_use]
    pub fn row(&self, index: usize) -> CsrRow<'_, T> {
        self.get_row(index).expect("Row index must be in bounds")
    }

    /// Mutable row access for the given row index.
    ///
    /// Panics
    /// ------
    /// Panics if row index is out of bounds.
    #[inline]
    pub fn row_mut(&mut self, index: usize) -> CsrRowMut<'_, T> {
        self.get_row_mut(index)
            .expect("Row index must be in bounds")
    }

    /// Return the row at the given row index, or `None` if out of bounds.
    #[inline]
    #[must_use]
    pub fn get_row(&self, index: usize) -> Option<CsrRow<'_, T>> {
        self.cs.get_lane(index).map(|lane| CsrRow { lane })
    }

    /// Mutable row access for the given row index, or `None` if out of bounds.
    #[inline]
    #[must_use]
    pub fn get_row_mut(&mut self, index: usize) -> Option<CsrRowMut<'_, T>> {
        self.cs.get_lane_mut(index).map(|lane| CsrRowMut { lane })
    }

    /// An iterator over rows in the matrix.
    pub fn row_iter(&self) -> CsrRowIter<'_, T> {
        CsrRowIter {
            lane_iter: CsLaneIter::new(self.pattern(), self.values()),
        }
    }

    /// A mutable iterator over rows in the matrix.
    pub fn row_iter_mut(&mut self) -> CsrRowIterMut<'_, T> {
        let (pattern, values) = self.cs.pattern_and_values_mut();
        CsrRowIterMut {
            lane_iter: CsLaneIterMut::new(pattern, values),
        }
    }

    /// Disassembles the CSR matrix into its underlying offset, index and value arrays.
    ///
    /// If the matrix contains the sole reference to the sparsity pattern,
    /// then the data is returned as-is. Otherwise, the sparsity pattern is cloned.
    ///
    /// Examples
    /// --------
    ///
    /// ```
    /// # use nalgebra_sparse::csr::CsrMatrix;
    /// let row_offsets = vec![0, 2, 3, 4];
    /// let col_indices = vec![0, 2, 1, 0];
    /// let values = vec![1, 2, 3, 4];
    /// let mut csr = CsrMatrix::try_from_csr_data(
    ///     3,
    ///     4,
    ///     row_offsets.clone(),
    ///     col_indices.clone(),
    ///     values.clone())
    ///     .unwrap();
    /// let (row_offsets2, col_indices2, values2) = csr.disassemble();
    /// assert_eq!(row_offsets2, row_offsets);
    /// assert_eq!(col_indices2, col_indices);
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

    /// Reinterprets the CSR matrix as its transpose represented by a CSC matrix.
    ///
    /// This operation does not touch the CSR data, and is effectively a no-op.
    pub fn transpose_as_csc(self) -> CscMatrix<T> {
        let (pattern, values) = self.cs.take_pattern_and_values();
        CscMatrix::try_from_pattern_and_values(pattern, values).unwrap()
    }

    /// Returns an entry for the given row/col indices, or `None` if the indices are out of bounds.
    ///
    /// Each call to this function incurs the cost of a binary search among the explicitly
    /// stored column entries for the given row.
    #[must_use]
    pub fn get_entry(&self, row_index: usize, col_index: usize) -> Option<SparseEntry<'_, T>> {
        self.cs.get_entry(row_index, col_index)
    }

    /// Returns a mutable entry for the given row/col indices, or `None` if the indices are out
    /// of bounds.
    ///
    /// Each call to this function incurs the cost of a binary search among the explicitly
    /// stored column entries for the given row.
    pub fn get_entry_mut(
        &mut self,
        row_index: usize,
        col_index: usize,
    ) -> Option<SparseEntryMut<'_, T>> {
        self.cs.get_entry_mut(row_index, col_index)
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

    /// Returns a triplet of slices `(row_offsets, col_indices, values)` that make up the CSR data.
    #[must_use]
    pub fn csr_data(&self) -> (&[usize], &[usize], &[T]) {
        self.cs.cs_data()
    }

    /// Returns a triplet of slices `(row_offsets, col_indices, values)` that make up the CSR data,
    /// where the `values` array is mutable.
    pub fn csr_data_mut(&mut self) -> (&[usize], &[usize], &mut [T]) {
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
        Self {
            cs: self
                .cs
                .filter(|row_idx, col_idx, v| predicate(row_idx, col_idx, v)),
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
    pub fn diagonal_as_csr(&self) -> Self
    where
        T: Clone,
    {
        Self {
            cs: self.cs.diagonal_as_matrix(),
        }
    }

    /// Compute the transpose of the matrix.
    #[must_use]
    pub fn transpose(&self) -> CsrMatrix<T>
    where
        T: Scalar,
    {
        CscMatrix::from(self).transpose_as_csr()
    }
}

impl<T> Default for CsrMatrix<T> {
    fn default() -> Self {
        Self {
            cs: Default::default(),
        }
    }
}

/// Convert pattern format errors into more meaningful CSR-specific errors.
///
/// This ensures that the terminology is consistent: we are talking about rows and columns,
/// not lanes, major and minor dimensions.
fn pattern_format_error_to_csr_error(err: SparsityPatternFormatError) -> SparseFormatError {
    use SparseFormatError as E;
    use SparseFormatErrorKind as K;
    use SparsityPatternFormatError::DuplicateEntry as PatternDuplicateEntry;
    use SparsityPatternFormatError::*;

    match err {
        InvalidOffsetArrayLength => E::from_kind_and_msg(
            K::InvalidStructure,
            "Length of row offset array is not equal to nrows + 1.",
        ),
        InvalidOffsetFirstLast => E::from_kind_and_msg(
            K::InvalidStructure,
            "First or last row offset is inconsistent with format specification.",
        ),
        NonmonotonicOffsets => E::from_kind_and_msg(
            K::InvalidStructure,
            "Row offsets are not monotonically increasing.",
        ),
        NonmonotonicMinorIndices => E::from_kind_and_msg(
            K::InvalidStructure,
            "Column indices are not monotonically increasing (sorted) within each row.",
        ),
        MinorIndexOutOfBounds => {
            E::from_kind_and_msg(K::IndexOutOfBounds, "Column indices are out of bounds.")
        }
        PatternDuplicateEntry => {
            E::from_kind_and_msg(K::DuplicateEntry, "Matrix data contains duplicate entries.")
        }
    }
}

/// Iterator type for iterating over triplets in a CSR matrix.
#[derive(Debug)]
pub struct CsrTripletIter<'a, T> {
    pattern_iter: SparsityPatternIter<'a>,
    values_iter: Iter<'a, T>,
}

impl<'a, T> Clone for CsrTripletIter<'a, T> {
    fn clone(&self) -> Self {
        CsrTripletIter {
            pattern_iter: self.pattern_iter.clone(),
            values_iter: self.values_iter.clone(),
        }
    }
}

impl<'a, T: Clone> CsrTripletIter<'a, T> {
    /// Adapts the triplet iterator to return owned values.
    ///
    /// The triplet iterator returns references to the values. This method adapts the iterator
    /// so that the values are cloned.
    #[inline]
    pub fn cloned_values(self) -> impl 'a + Iterator<Item = (usize, usize, T)> {
        self.map(|(i, j, v)| (i, j, v.clone()))
    }
}

impl<'a, T> Iterator for CsrTripletIter<'a, T> {
    type Item = (usize, usize, &'a T);

    fn next(&mut self) -> Option<Self::Item> {
        let next_entry = self.pattern_iter.next();
        let next_value = self.values_iter.next();

        match (next_entry, next_value) {
            (Some((i, j)), Some(v)) => Some((i, j, v)),
            _ => None,
        }
    }
}

/// Iterator type for mutably iterating over triplets in a CSR matrix.
#[derive(Debug)]
pub struct CsrTripletIterMut<'a, T> {
    pattern_iter: SparsityPatternIter<'a>,
    values_mut_iter: IterMut<'a, T>,
}

impl<'a, T> Iterator for CsrTripletIterMut<'a, T> {
    type Item = (usize, usize, &'a mut T);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let next_entry = self.pattern_iter.next();
        let next_value = self.values_mut_iter.next();

        match (next_entry, next_value) {
            (Some((i, j)), Some(v)) => Some((i, j, v)),
            _ => None,
        }
    }
}

/// An immutable representation of a row in a CSR matrix.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CsrRow<'a, T> {
    lane: CsLane<'a, T>,
}

/// A mutable representation of a row in a CSR matrix.
///
/// Note that only explicitly stored entries can be mutated. The sparsity pattern belonging
/// to the row cannot be modified.
#[derive(Debug, PartialEq, Eq)]
pub struct CsrRowMut<'a, T> {
    lane: CsLaneMut<'a, T>,
}

/// Implement the methods common to both CsrRow and CsrRowMut
macro_rules! impl_csr_row_common_methods {
    ($name:ty) => {
        impl<'a, T> $name {
            /// The number of global columns in the row.
            #[inline]
            #[must_use]
            pub fn ncols(&self) -> usize {
                self.lane.minor_dim()
            }

            /// The number of non-zeros in this row.
            #[inline]
            #[must_use]
            pub fn nnz(&self) -> usize {
                self.lane.nnz()
            }

            /// The column indices corresponding to explicitly stored entries in this row.
            #[inline]
            #[must_use]
            pub fn col_indices(&self) -> &[usize] {
                self.lane.minor_indices()
            }

            /// The values corresponding to explicitly stored entries in this row.
            #[inline]
            #[must_use]
            pub fn values(&self) -> &[T] {
                self.lane.values()
            }

            /// Returns an entry for the given global column index.
            ///
            /// Each call to this function incurs the cost of a binary search among the explicitly
            /// stored column entries.
            #[inline]
            #[must_use]
            pub fn get_entry(&self, global_col_index: usize) -> Option<SparseEntry<'_, T>> {
                self.lane.get_entry(global_col_index)
            }
        }
    };
}

impl_csr_row_common_methods!(CsrRow<'a, T>);
impl_csr_row_common_methods!(CsrRowMut<'a, T>);

impl<'a, T> CsrRowMut<'a, T> {
    /// Mutable access to the values corresponding to explicitly stored entries in this row.
    #[inline]
    pub fn values_mut(&mut self) -> &mut [T] {
        self.lane.values_mut()
    }

    /// Provides simultaneous access to column indices and mutable values corresponding to the
    /// explicitly stored entries in this row.
    ///
    /// This method primarily facilitates low-level access for methods that process data stored
    /// in CSR format directly.
    #[inline]
    pub fn cols_and_values_mut(&mut self) -> (&[usize], &mut [T]) {
        self.lane.indices_and_values_mut()
    }

    /// Returns a mutable entry for the given global column index.
    #[inline]
    #[must_use]
    pub fn get_entry_mut(&mut self, global_col_index: usize) -> Option<SparseEntryMut<'_, T>> {
        self.lane.get_entry_mut(global_col_index)
    }
}

/// Row iterator for [`CsrMatrix`].
pub struct CsrRowIter<'a, T> {
    lane_iter: CsLaneIter<'a, T>,
}

impl<'a, T> Iterator for CsrRowIter<'a, T> {
    type Item = CsrRow<'a, T>;

    fn next(&mut self) -> Option<Self::Item> {
        self.lane_iter.next().map(|lane| CsrRow { lane })
    }
}

/// Mutable row iterator for [`CsrMatrix`].
pub struct CsrRowIterMut<'a, T> {
    lane_iter: CsLaneIterMut<'a, T>,
}

impl<'a, T> Iterator for CsrRowIterMut<'a, T>
where
    T: 'a,
{
    type Item = CsrRowMut<'a, T>;

    fn next(&mut self) -> Option<Self::Item> {
        self.lane_iter.next().map(|lane| CsrRowMut { lane })
    }
}
