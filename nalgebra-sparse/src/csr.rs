//! An implementation of the CSR sparse matrix format.
use crate::{SparseFormatError, SparseFormatErrorKind, SparseEntry, SparseEntryMut};
use crate::pattern::{SparsityPattern, SparsityPatternFormatError, SparsityPatternIter};
use crate::csc::CscMatrix;

use nalgebra::Scalar;
use num_traits::Zero;

use std::sync::Arc;
use std::slice::{IterMut, Iter};
use std::ops::Range;
use std::ptr::slice_from_raw_parts_mut;

/// A CSR representation of a sparse matrix.
///
/// The Compressed Sparse Row (CSR) format is well-suited as a general-purpose storage format
/// for many sparse matrix applications.
///
/// TODO: Storage explanation and examples
///
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CsrMatrix<T> {
    // Rows are major, cols are minor in the sparsity pattern
    sparsity_pattern: Arc<SparsityPattern>,
    values: Vec<T>,
}

impl<T> CsrMatrix<T> {
    /// Create a zero CSR matrix with no explicitly stored entries.
    pub fn new(nrows: usize, ncols: usize) -> Self {
        Self {
            sparsity_pattern: Arc::new(SparsityPattern::new(nrows, ncols)),
            values: vec![],
        }
    }

    /// The number of rows in the matrix.
    #[inline]
    pub fn nrows(&self) -> usize {
        self.sparsity_pattern.major_dim()
    }

    /// The number of columns in the matrix.
    #[inline]
    pub fn ncols(&self) -> usize {
        self.sparsity_pattern.minor_dim()
    }

    /// The number of non-zeros in the matrix.
    ///
    /// Note that this corresponds to the number of explicitly stored entries, *not* the actual
    /// number of algebraically zero entries in the matrix. Explicitly stored entries can still
    /// be zero. Corresponds to the number of entries in the sparsity pattern.
    #[inline]
    pub fn nnz(&self) -> usize {
        self.sparsity_pattern.nnz()
    }

    /// The row offsets defining part of the CSR format.
    #[inline]
    pub fn row_offsets(&self) -> &[usize] {
        self.sparsity_pattern.major_offsets()
    }

    /// The column indices defining part of the CSR format.
    #[inline]
    pub fn col_indices(&self) -> &[usize] {
        self.sparsity_pattern.minor_indices()
    }

    /// The non-zero values defining part of the CSR format.
    #[inline]
    pub fn values(&self) -> &[T] {
        &self.values
    }

    /// Mutable access to the non-zero values.
    #[inline]
    pub fn values_mut(&mut self) -> &mut [T] {
        &mut self.values
    }

    /// Try to construct a CSR matrix from raw CSR data.
    ///
    /// It is assumed that each row contains unique and sorted column indices that are in
    /// bounds with respect to the number of columns in the matrix. If this is not the case,
    /// an error is returned to indicate the failure.
    ///
    /// An error is returned if the data given does not conform to the CSR storage format.
    /// See the documentation for [CsrMatrix](struct.CsrMatrix.html) for more information.
    pub fn try_from_csr_data(
        num_rows: usize,
        num_cols: usize,
        row_offsets: Vec<usize>,
        col_indices: Vec<usize>,
        values: Vec<T>,
    ) -> Result<Self, SparseFormatError> {
        let pattern = SparsityPattern::try_from_offsets_and_indices(
            num_rows, num_cols, row_offsets, col_indices)
            .map_err(pattern_format_error_to_csr_error)?;
        Self::try_from_pattern_and_values(Arc::new(pattern), values)
    }

    /// Try to construct a CSR matrix from a sparsity pattern and associated non-zero values.
    ///
    /// Returns an error if the number of values does not match the number of minor indices
    /// in the pattern.
    pub fn try_from_pattern_and_values(pattern: Arc<SparsityPattern>, values: Vec<T>)
        -> Result<Self, SparseFormatError> {
        if pattern.nnz() == values.len() {
            Ok(Self {
                sparsity_pattern: pattern,
                values,
            })
        } else {
            Err(SparseFormatError::from_kind_and_msg(
                SparseFormatErrorKind::InvalidStructure,
                "Number of values and column indices must be the same"))
        }
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
    pub fn triplet_iter(&self) -> CsrTripletIter<T> {
        CsrTripletIter {
            pattern_iter: self.sparsity_pattern.entries(),
            values_iter: self.values.iter()
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
    pub fn triplet_iter_mut(&mut self) -> CsrTripletIterMut<T> {
        CsrTripletIterMut {
            pattern_iter: self.sparsity_pattern.entries(),
            values_mut_iter: self.values.iter_mut()
        }
    }

    /// Return the row at the given row index.
    ///
    /// Panics
    /// ------
    /// Panics if row index is out of bounds.
    #[inline]
    pub fn row(&self, index: usize) -> CsrRow<T> {
        self.get_row(index)
            .expect("Row index must be in bounds")
    }

    /// Mutable row access for the given row index.
    ///
    /// Panics
    /// ------
    /// Panics if row index is out of bounds.
    #[inline]
    pub fn row_mut(&mut self, index: usize) -> CsrRowMut<T> {
        self.get_row_mut(index)
            .expect("Row index must be in bounds")
    }

    /// Return the row at the given row index, or `None` if out of bounds.
    #[inline]
    pub fn get_row(&self, index: usize) -> Option<CsrRow<T>> {
        let range = self.get_index_range(index)?;
        Some(CsrRow {
            col_indices: &self.sparsity_pattern.minor_indices()[range.clone()],
            values: &self.values[range],
            ncols: self.ncols()
        })
    }

    /// Mutable row access for the given row index, or `None` if out of bounds.
    #[inline]
    pub fn get_row_mut(&mut self, index: usize) -> Option<CsrRowMut<T>> {
        let range = self.get_index_range(index)?;
        Some(CsrRowMut {
            ncols: self.ncols(),
            col_indices: &self.sparsity_pattern.minor_indices()[range.clone()],
            values: &mut self.values[range]
        })
    }

    /// Internal method for simplifying access to a row's data.
    fn get_index_range(&self, row_index: usize) -> Option<Range<usize>> {
        let row_begin = *self.sparsity_pattern.major_offsets().get(row_index)?;
        let row_end = *self.sparsity_pattern.major_offsets().get(row_index + 1)?;
        Some(row_begin .. row_end)
    }

    /// An iterator over rows in the matrix.
    pub fn row_iter(&self) -> CsrRowIter<T> {
        CsrRowIter {
            current_row_idx: 0,
            matrix: self
        }
    }

    /// A mutable iterator over rows in the matrix.
    pub fn row_iter_mut(&mut self) -> CsrRowIterMut<T> {
        CsrRowIterMut {
            current_row_idx: 0,
            pattern: &self.sparsity_pattern,
            remaining_values: self.values.as_mut_ptr()
        }
    }

    /// Returns the underlying vector containing the values for the explicitly stored entries.
    pub fn take_values(self) -> Vec<T> {
        self.values
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
        // Take an Arc to the pattern, which might be the sole reference to the data after
        // taking the values. This is important, because it might let us avoid cloning the data
        // further below.
        let pattern = self.sparsity_pattern;
        let values = self.values;

        // Try to take the pattern out of the `Arc` if possible,
        // otherwise clone the pattern.
        let owned_pattern = Arc::try_unwrap(pattern)
            .unwrap_or_else(|arc| SparsityPattern::clone(&*arc));
        let (offsets, indices) = owned_pattern.disassemble();

        (offsets, indices, values)
    }

    /// Returns the underlying sparsity pattern.
    ///
    /// The sparsity pattern is stored internally inside an `Arc`. This allows users to re-use
    /// the same sparsity pattern for multiple matrices without storing the same pattern multiple
    /// times in memory.
    pub fn pattern(&self) -> &Arc<SparsityPattern> {
        &self.sparsity_pattern
    }

    /// Reinterprets the CSR matrix as its transpose represented by a CSC matrix.
    ///
    /// This operation does not touch the CSR data, and is effectively a no-op.
    pub fn transpose_as_csc(self) -> CscMatrix<T> {
        let pattern = self.sparsity_pattern;
        let values = self.values;
        CscMatrix::try_from_pattern_and_values(pattern, values).unwrap()
    }

    /// Returns an entry for the given row/col indices, or `None` if the indices are out of bounds.
    ///
    /// Each call to this function incurs the cost of a binary search among the explicitly
    /// stored column entries for the given row.
    pub fn get_entry(&self, row_index: usize, col_index: usize) -> Option<SparseEntry<T>> {
        let row_range = self.get_index_range(row_index)?;
        let col_indices = &self.col_indices()[row_range.clone()];
        let values = &self.values()[row_range];
        get_entry_from_slices(self.ncols(), col_indices, values, col_index)
    }

    /// Returns a mutable entry for the given row/col indices, or `None` if the indices are out
    /// of bounds.
    ///
    /// Each call to this function incurs the cost of a binary search among the explicitly
    /// stored column entries for the given row.
    pub fn get_entry_mut(&mut self, row_index: usize, col_index: usize)
                         -> Option<SparseEntryMut<T>> {
        let row_range = self.get_index_range(row_index)?;
        let ncols = self.ncols();
        let (_, col_indices, values) = self.csr_data_mut();
        let col_indices = &col_indices[row_range.clone()];
        let values = &mut values[row_range];
        get_mut_entry_from_slices(ncols, col_indices, values, col_index)
    }

    /// Returns an entry for the given row/col indices.
    ///
    /// Same as `get_entry`, except that it directly panics upon encountering row/col indices
    /// out of bounds.
    ///
    /// Panics
    /// ------
    /// Panics if `row_index` or `col_index` is out of bounds.
    pub fn index_entry(&self, row_index: usize, col_index: usize) -> SparseEntry<T> {
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
    pub fn index_entry_mut(&mut self, row_index: usize, col_index: usize) -> SparseEntryMut<T> {
        self.get_entry_mut(row_index, col_index)
            .expect("Out of bounds matrix indices encountered")
    }

    /// Returns a triplet of slices `(row_offsets, col_indices, values)` that make up the CSR data.
    pub fn csr_data(&self) -> (&[usize], &[usize], &[T]) {
        (self.row_offsets(), self.col_indices(), self.values())
    }

    /// Returns a triplet of slices `(row_offsets, col_indices, values)` that make up the CSR data,
    /// where the `values` array is mutable.
    pub fn csr_data_mut(&mut self) -> (&[usize], &[usize], &mut [T]) {
        let pattern = self.sparsity_pattern.as_ref();
        (pattern.major_offsets(), pattern.minor_indices(), &mut self.values)
    }
}

impl<T> CsrMatrix<T>
where
    T: Scalar + Zero
{
    /// Compute the transpose of the matrix.
    pub fn transpose(&self) -> CsrMatrix<T> {
        CscMatrix::from(self).transpose_as_csr()
    }
}

/// Convert pattern format errors into more meaningful CSR-specific errors.
///
/// This ensures that the terminology is consistent: we are talking about rows and columns,
/// not lanes, major and minor dimensions.
fn pattern_format_error_to_csr_error(err: SparsityPatternFormatError) -> SparseFormatError {
    use SparsityPatternFormatError::*;
    use SparsityPatternFormatError::DuplicateEntry as PatternDuplicateEntry;
    use SparseFormatError as E;
    use SparseFormatErrorKind as K;

    match err {
        InvalidOffsetArrayLength => E::from_kind_and_msg(
            K::InvalidStructure,
            "Length of row offset array is not equal to nrows + 1."),
        InvalidOffsetFirstLast => E::from_kind_and_msg(
            K::InvalidStructure,
            "First or last row offset is inconsistent with format specification."),
        NonmonotonicOffsets => E::from_kind_and_msg(
            K::InvalidStructure,
            "Row offsets are not monotonically increasing."),
        NonmonotonicMinorIndices => E::from_kind_and_msg(
            K::InvalidStructure,
            "Column indices are not monotonically increasing (sorted) within each row."),
        MinorIndexOutOfBounds => E::from_kind_and_msg(
            K::IndexOutOfBounds,
            "Column indices are out of bounds."),
        PatternDuplicateEntry => E::from_kind_and_msg(
            K::DuplicateEntry,
            "Matrix data contains duplicate entries."),
    }
}

/// Iterator type for iterating over triplets in a CSR matrix.
#[derive(Debug)]
pub struct CsrTripletIter<'a, T> {
    pattern_iter: SparsityPatternIter<'a>,
    values_iter: Iter<'a, T>
}

impl<'a, T> Iterator for CsrTripletIter<'a, T> {
    type Item = (usize, usize, &'a T);

    fn next(&mut self) -> Option<Self::Item> {
        let next_entry = self.pattern_iter.next();
        let next_value = self.values_iter.next();

        match (next_entry, next_value) {
            (Some((i, j)), Some(v)) => Some((i, j, v)),
            _ => None
        }
    }
}

/// Iterator type for mutably iterating over triplets in a CSR matrix.
#[derive(Debug)]
pub struct CsrTripletIterMut<'a, T> {
    pattern_iter: SparsityPatternIter<'a>,
    values_mut_iter: IterMut<'a, T>
}

impl<'a, T> Iterator for CsrTripletIterMut<'a, T> {
    type Item = (usize, usize, &'a mut T);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let next_entry = self.pattern_iter.next();
        let next_value = self.values_mut_iter.next();

        match (next_entry, next_value) {
            (Some((i, j)), Some(v)) => Some((i, j, v)),
            _ => None
        }
    }
}

/// An immutable representation of a row in a CSR matrix.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CsrRow<'a, T> {
    ncols: usize,
    col_indices: &'a [usize],
    values: &'a [T],
}

/// A mutable representation of a row in a CSR matrix.
///
/// Note that only explicitly stored entries can be mutated. The sparsity pattern belonging
/// to the row cannot be modified.
#[derive(Debug, PartialEq, Eq)]
pub struct CsrRowMut<'a, T> {
    ncols: usize,
    col_indices: &'a [usize],
    values: &'a mut [T]
}

/// Implement the methods common to both CsrRow and CsrRowMut
macro_rules! impl_csr_row_common_methods {
    ($name:ty) => {
        impl<'a, T> $name {
            /// The number of global columns in the row.
            #[inline]
            pub fn ncols(&self) -> usize {
                self.ncols
            }

            /// The number of non-zeros in this row.
            #[inline]
            pub fn nnz(&self) -> usize {
                self.col_indices.len()
            }

            /// The column indices corresponding to explicitly stored entries in this row.
            #[inline]
            pub fn col_indices(&self) -> &[usize] {
                self.col_indices
            }

            /// The values corresponding to explicitly stored entries in this row.
            #[inline]
            pub fn values(&self) -> &[T] {
                self.values
            }

            /// Returns an entry for the given global column index.
            ///
            /// Each call to this function incurs the cost of a binary search among the explicitly
            /// stored column entries.
            pub fn get_entry(&self, global_col_index: usize) -> Option<SparseEntry<T>> {
                get_entry_from_slices(self.ncols, self.col_indices, self.values, global_col_index)
            }
        }
    }
}

fn get_entry_from_slices<'a, T>(ncols: usize,
                                col_indices: &'a [usize],
                                values: &'a [T],
                                global_col_index: usize) -> Option<SparseEntry<'a, T>> {
    let local_index = col_indices.binary_search(&global_col_index);
    if let Ok(local_index) = local_index {
        Some(SparseEntry::NonZero(&values[local_index]))
    } else if global_col_index < ncols {
        Some(SparseEntry::Zero)
    } else {
        None
    }
}

fn get_mut_entry_from_slices<'a, T>(ncols: usize,
                                    col_indices: &'a [usize],
                                    values: &'a mut [T],
                                    global_col_index: usize) -> Option<SparseEntryMut<'a, T>> {
    let local_index = col_indices.binary_search(&global_col_index);
    if let Ok(local_index) = local_index {
        Some(SparseEntryMut::NonZero(&mut values[local_index]))
    } else if global_col_index < ncols {
        Some(SparseEntryMut::Zero)
    } else {
        None
    }
}

impl_csr_row_common_methods!(CsrRow<'a, T>);
impl_csr_row_common_methods!(CsrRowMut<'a, T>);

impl<'a, T> CsrRowMut<'a, T> {
    /// Mutable access to the values corresponding to explicitly stored entries in this row.
    pub fn values_mut(&mut self) -> &mut [T] {
        self.values
    }

    /// Provides simultaneous access to column indices and mutable values corresponding to the
    /// explicitly stored entries in this row.
    ///
    /// This method primarily facilitates low-level access for methods that process data stored
    /// in CSR format directly.
    pub fn cols_and_values_mut(&mut self) -> (&[usize], &mut [T]) {
        (self.col_indices, self.values)
    }

    /// Returns a mutable entry for the given global column index.
    pub fn get_entry_mut(&mut self, global_col_index: usize) -> Option<SparseEntryMut<T>> {
        get_mut_entry_from_slices(self.ncols, self.col_indices, self.values, global_col_index)
    }
}

/// Row iterator for [CsrMatrix](struct.CsrMatrix.html).
pub struct CsrRowIter<'a, T> {
    // The index of the row that will be returned on the next
    current_row_idx: usize,
    matrix: &'a CsrMatrix<T>
}

impl<'a, T> Iterator for CsrRowIter<'a, T> {
    type Item = CsrRow<'a, T>;

    fn next(&mut self) -> Option<Self::Item> {
        let row = self.matrix.get_row(self.current_row_idx);
        self.current_row_idx += 1;
        row
    }
}

/// Mutable row iterator for [CsrMatrix](struct.CsrMatrix.html).
pub struct CsrRowIterMut<'a, T> {
    current_row_idx: usize,
    pattern: &'a SparsityPattern,
    remaining_values: *mut T,
}

impl<'a, T> Iterator for CsrRowIterMut<'a, T>
where
    T: 'a
{
    type Item = CsrRowMut<'a, T>;

    fn next(&mut self) -> Option<Self::Item> {
        let lane = self.pattern.get_lane(self.current_row_idx);
        let ncols = self.pattern.minor_dim();

        if let Some(col_indices) = lane {
            let count = col_indices.len();

            // Note: I can't think of any way to construct this iterator without unsafe.
            let values_in_row;
            unsafe {
                values_in_row = &mut *slice_from_raw_parts_mut(self.remaining_values, count);
                self.remaining_values = self.remaining_values.add(count);
            }
            self.current_row_idx += 1;

            Some(CsrRowMut {
                ncols,
                col_indices,
                values: values_in_row
            })
        } else {
            None
        }
    }
}