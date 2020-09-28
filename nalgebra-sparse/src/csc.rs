//! An implementation of the CSC sparse matrix format.

use crate::{SparseFormatError, SparseFormatErrorKind};
use crate::pattern::{SparsityPattern, SparsityPatternFormatError, SparsityPatternIter};

use std::sync::Arc;
use std::slice::{IterMut, Iter};
use std::ops::Range;
use num_traits::Zero;
use std::ptr::slice_from_raw_parts_mut;

/// A CSC representation of a sparse matrix.
///
/// The Compressed Sparse Column (CSC) format is well-suited as a general-purpose storage format
/// for many sparse matrix applications.
///
/// TODO: Storage explanation and examples
///
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CscMatrix<T> {
    // Cols are major, rows are minor in the sparsity pattern
    sparsity_pattern: Arc<SparsityPattern>,
    values: Vec<T>,
}

impl<T> CscMatrix<T> {
    /// Create a zero CSC matrix with no explicitly stored entries.
    pub fn new(nrows: usize, ncols: usize) -> Self {
        Self {
            sparsity_pattern: Arc::new(SparsityPattern::new(ncols, nrows)),
            values: vec![],
        }
    }

    /// The number of rows in the matrix.
    #[inline]
    pub fn nrows(&self) -> usize {
        self.sparsity_pattern.minor_dim()
    }

    /// The number of columns in the matrix.
    #[inline]
    pub fn ncols(&self) -> usize {
        self.sparsity_pattern.major_dim()
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

    /// The column offsets defining part of the CSC format.
    #[inline]
    pub fn col_offsets(&self) -> &[usize] {
        self.sparsity_pattern.major_offsets()
    }

    /// The row indices defining part of the CSC format.
    #[inline]
    pub fn row_indices(&self) -> &[usize] {
        self.sparsity_pattern.minor_indices()
    }

    /// The non-zero values defining part of the CSC format.
    #[inline]
    pub fn values(&self) -> &[T] {
        &self.values
    }

    /// Mutable access to the non-zero values.
    #[inline]
    pub fn values_mut(&mut self) -> &mut [T] {
        &mut self.values
    }

    /// Try to construct a CSC matrix from raw CSC data.
    ///
    /// It is assumed that each column contains unique and sorted row indices that are in
    /// bounds with respect to the number of rows in the matrix. If this is not the case,
    /// an error is returned to indicate the failure.
    ///
    /// An error is returned if the data given does not conform to the CSC storage format.
    /// See the documentation for [CscMatrix](struct.CscMatrix.html) for more information.
    pub fn try_from_csc_data(
        num_rows: usize,
        num_cols: usize,
        col_offsets: Vec<usize>,
        row_indices: Vec<usize>,
        values: Vec<T>,
    ) -> Result<Self, SparseFormatError> {
        let pattern = SparsityPattern::try_from_offsets_and_indices(
            num_cols, num_rows, col_offsets, row_indices)
            .map_err(pattern_format_error_to_csc_error)?;
        Self::try_from_pattern_and_values(Arc::new(pattern), values)
    }

    /// Try to construct a CSC matrix from a sparsity pattern and associated non-zero values.
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
                "Number of values and row indices must be the same"))
        }
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
    pub fn triplet_iter(&self) -> CscTripletIter<T> {
        CscTripletIter {
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
    pub fn triplet_iter_mut(&mut self) -> CscTripletIterMut<T> {
        CscTripletIterMut {
            pattern_iter: self.sparsity_pattern.entries(),
            values_mut_iter: self.values.iter_mut()
        }
    }

    /// Return the column at the given column index.
    ///
    /// Panics
    /// ------
    /// Panics if column index is out of bounds.
    #[inline]
    pub fn col(&self, index: usize) -> CscCol<T> {
        self.get_col(index)
            .expect("Row index must be in bounds")
    }

    /// Mutable column access for the given column index.
    ///
    /// Panics
    /// ------
    /// Panics if column index is out of bounds.
    #[inline]
    pub fn col_mut(&mut self, index: usize) -> CscColMut<T> {
        self.get_col_mut(index)
            .expect("Row index must be in bounds")
    }

    /// Return the column at the given column index, or `None` if out of bounds.
    #[inline]
    pub fn get_col(&self, index: usize) -> Option<CscCol<T>> {
        let range = self.get_index_range(index)?;
        Some(CscCol {
            row_indices: &self.sparsity_pattern.minor_indices()[range.clone()],
            values: &self.values[range],
            nrows: self.nrows()
        })
    }

    /// Mutable column access for the given column index, or `None` if out of bounds.
    #[inline]
    pub fn get_col_mut(&mut self, index: usize) -> Option<CscColMut<T>> {
        let range = self.get_index_range(index)?;
        Some(CscColMut {
            nrows: self.nrows(),
            row_indices: &self.sparsity_pattern.minor_indices()[range.clone()],
            values: &mut self.values[range]
        })
    }

    /// Internal method for simplifying access to a column's data.
    fn get_index_range(&self, col_index: usize) -> Option<Range<usize>> {
        let col_begin = *self.sparsity_pattern.major_offsets().get(col_index)?;
        let col_end = *self.sparsity_pattern.major_offsets().get(col_index + 1)?;
        Some(col_begin .. col_end)
    }

    /// An iterator over columns in the matrix.
    pub fn col_iter(&self) -> CscColIter<T> {
        CscColIter {
            current_col_idx: 0,
            matrix: self
        }
    }

    /// A mutable iterator over columns in the matrix.
    pub fn col_iter_mut(&mut self) -> CscColIterMut<T> {
        CscColIterMut {
            current_col_idx: 0,
            pattern: &self.sparsity_pattern,
            remaining_values: self.values.as_mut_ptr()
        }
    }

    /// Returns the underlying vector containing the values for the explicitly stored entries.
    pub fn take_values(self) -> Vec<T> {
        self.values
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
        // Take an Arc to the pattern, which might be the sole reference to the data after
        // taking the values. This is important, because it might let us avoid cloning the data
        // further below.
        let pattern = self.pattern();
        let values = self.take_values();

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
    pub fn pattern(&self) -> Arc<SparsityPattern> {
        Arc::clone(&self.sparsity_pattern)
    }
}

impl<T: Clone + Zero> CscMatrix<T> {
    /// Return the value in the matrix at the given global row/col indices, or `None` if out of
    /// bounds.
    ///
    /// If the indices are in bounds, but no explicitly stored entry is associated with it,
    /// `T::zero()` is returned. Note that this method offers no way of distinguishing
    /// explicitly stored zero entries from zero values that are only implicitly represented.
    ///
    /// Each call to this function incurs the cost of a binary search among the explicitly
    /// stored column entries for the given row.
    #[inline]
    pub fn get(&self, row_index: usize, col_index: usize) -> Option<T> {
        self.get_col(row_index)?.get(col_index)
    }

    /// Same as `get`, but panics if indices are out of bounds.
    ///
    /// Panics
    /// ------
    /// Panics if either index is out of bounds.
    #[inline]
    pub fn index(&self, row_index: usize, col_index: usize) -> T {
        self.get(row_index, col_index).unwrap()
    }
}

/// Convert pattern format errors into more meaningful CSC-specific errors.
///
/// This ensures that the terminology is consistent: we are talking about rows and columns,
/// not lanes, major and minor dimensions.
fn pattern_format_error_to_csc_error(err: SparsityPatternFormatError) -> SparseFormatError {
    use SparsityPatternFormatError::*;
    use SparsityPatternFormatError::DuplicateEntry as PatternDuplicateEntry;
    use SparseFormatError as E;
    use SparseFormatErrorKind as K;

    match err {
        InvalidOffsetArrayLength => E::from_kind_and_msg(
            K::InvalidStructure,
            "Length of col offset array is not equal to ncols + 1."),
        InvalidOffsetFirstLast => E::from_kind_and_msg(
            K::InvalidStructure,
            "First or last col offset is inconsistent with format specification."),
        NonmonotonicOffsets => E::from_kind_and_msg(
            K::InvalidStructure,
            "Col offsets are not monotonically increasing."),
        NonmonotonicMinorIndices => E::from_kind_and_msg(
            K::InvalidStructure,
            "Row indices are not monotonically increasing (sorted) within each column."),
        MinorIndexOutOfBounds => E::from_kind_and_msg(
            K::IndexOutOfBounds,
            "Row indices are out of bounds."),
        PatternDuplicateEntry => E::from_kind_and_msg(
            K::DuplicateEntry,
            "Matrix data contains duplicate entries."),
    }
}

/// Iterator type for iterating over triplets in a CSC matrix.
#[derive(Debug)]
pub struct CscTripletIter<'a, T> {
    pattern_iter: SparsityPatternIter<'a>,
    values_iter: Iter<'a, T>
}

impl<'a, T> Iterator for CscTripletIter<'a, T> {
    type Item = (usize, usize, &'a T);

    fn next(&mut self) -> Option<Self::Item> {
        let next_entry = self.pattern_iter.next();
        let next_value = self.values_iter.next();

        match (next_entry, next_value) {
            (Some((i, j)), Some(v)) => Some((j, i, v)),
            _ => None
        }
    }
}

/// Iterator type for mutably iterating over triplets in a CSC matrix.
#[derive(Debug)]
pub struct CscTripletIterMut<'a, T> {
    pattern_iter: SparsityPatternIter<'a>,
    values_mut_iter: IterMut<'a, T>
}

impl<'a, T> Iterator for CscTripletIterMut<'a, T> {
    type Item = (usize, usize, &'a mut T);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let next_entry = self.pattern_iter.next();
        let next_value = self.values_mut_iter.next();

        match (next_entry, next_value) {
            (Some((i, j)), Some(v)) => Some((j, i, v)),
            _ => None
        }
    }
}

/// An immutable representation of a column in a CSC matrix.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CscCol<'a, T> {
    nrows: usize,
    row_indices: &'a [usize],
    values: &'a [T],
}

/// A mutable representation of a column in a CSC matrix.
///
/// Note that only explicitly stored entries can be mutated. The sparsity pattern belonging
/// to the column cannot be modified.
#[derive(Debug, PartialEq, Eq)]
pub struct CscColMut<'a, T> {
    nrows: usize,
    row_indices: &'a [usize],
    values: &'a mut [T]
}

/// Implement the methods common to both CscCol and CscColMut
macro_rules! impl_csc_col_common_methods {
    ($name:ty) => {
        impl<'a, T> $name {
            /// The number of global rows in the column.
            #[inline]
            pub fn nrows(&self) -> usize {
                self.nrows
            }

            /// The number of non-zeros in this column.
            #[inline]
            pub fn nnz(&self) -> usize {
                self.row_indices.len()
            }

            /// The row indices corresponding to explicitly stored entries in this column.
            #[inline]
            pub fn row_indices(&self) -> &[usize] {
                self.row_indices
            }

            /// The values corresponding to explicitly stored entries in this column.
            #[inline]
            pub fn values(&self) -> &[T] {
                self.values
            }
        }

        impl<'a, T: Clone + Zero> $name {
            /// Return the value in the matrix at the given global row index, or `None` if out of
            /// bounds.
            ///
            /// If the index is in bounds, but no explicitly stored entry is associated with it,
            /// `T::zero()` is returned. Note that this method offers no way of distinguishing
            /// explicitly stored zero entries from zero values that are only implicitly represented.
            ///
            /// Each call to this function incurs the cost of a binary search among the explicitly
            /// stored row entries for the current column.
            pub fn get(&self, global_row_index: usize) -> Option<T> {
                let local_index = self.row_indices().binary_search(&global_row_index);
                if let Ok(local_index) = local_index {
                    Some(self.values[local_index].clone())
                } else if global_row_index < self.nrows {
                    Some(T::zero())
                } else {
                    None
                }
            }
        }
    }
}

impl_csc_col_common_methods!(CscCol<'a, T>);
impl_csc_col_common_methods!(CscColMut<'a, T>);

impl<'a, T> CscColMut<'a, T> {
    /// Mutable access to the values corresponding to explicitly stored entries in this column.
    pub fn values_mut(&mut self) -> &mut [T] {
        self.values
    }

    /// Provides simultaneous access to row indices and mutable values corresponding to the
    /// explicitly stored entries in this column.
    ///
    /// This method primarily facilitates low-level access for methods that process data stored
    /// in CSC format directly.
    pub fn rows_and_values_mut(&mut self) -> (&[usize], &mut [T]) {
        (self.row_indices, self.values)
    }
}

/// Column iterator for [CscMatrix](struct.CscMatrix.html).
pub struct CscColIter<'a, T> {
    // The index of the row that will be returned on the next
    current_col_idx: usize,
    matrix: &'a CscMatrix<T>
}

impl<'a, T> Iterator for CscColIter<'a, T> {
    type Item = CscCol<'a, T>;

    fn next(&mut self) -> Option<Self::Item> {
        let col = self.matrix.get_col(self.current_col_idx);
        self.current_col_idx += 1;
        col
    }
}

/// Mutable column iterator for [CscMatrix](struct.CscMatrix.html).
pub struct CscColIterMut<'a, T> {
    current_col_idx: usize,
    pattern: &'a SparsityPattern,
    remaining_values: *mut T,
}

impl<'a, T> Iterator for CscColIterMut<'a, T>
where
    T: 'a
{
    type Item = CscColMut<'a, T>;

    fn next(&mut self) -> Option<Self::Item> {
        let lane = self.pattern.get_lane(self.current_col_idx);
        let nrows = self.pattern.minor_dim();

        if let Some(row_indices) = lane {
            let count = row_indices.len();

            // Note: I can't think of any way to construct this iterator without unsafe.
            let values_in_row;
            unsafe {
                values_in_row = &mut *slice_from_raw_parts_mut(self.remaining_values, count);
                self.remaining_values = self.remaining_values.add(count);
            }
            self.current_col_idx += 1;

            Some(CscColMut {
                nrows,
                row_indices,
                values: values_in_row
            })
        } else {
            None
        }
    }
}