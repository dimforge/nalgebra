use crate::{SparsityPattern, SparseFormatError};
use crate::iter::SparsityPatternIter;

use std::sync::Arc;
use std::slice::{IterMut, Iter};

/// A CSR representation of a sparse matrix.
///
/// The Compressed Row Storage (CSR) format is well-suited as a general-purpose storage format
/// for many sparse matrix applications.
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
    pub fn column_indices(&self) -> &[usize] {
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
    /// Panics
    /// ------
    /// Panics if the lengths of the provided arrays are not compatible with the CSR format.
    pub fn try_from_csr_data(
        num_rows: usize,
        num_cols: usize,
        row_offsets: Vec<usize>,
        col_indices: Vec<usize>,
        values: Vec<T>,
    ) -> Result<Self, SparseFormatError> {
        assert_eq!(col_indices.len(), values.len(),
                   "Number of values and column indices must be the same");
        let pattern = SparsityPattern::try_from_offsets_and_indices(
            num_rows, num_cols, row_offsets, col_indices)?;
        Ok(Self {
            sparsity_pattern: Arc::new(pattern),
            values,
        })
    }

    /// An iterator over non-zero triplets (i, j, v).
    ///
    /// The iteration happens in row-major fashion, meaning that i increases monotonically,
    /// and j increases monotonically within each row.
    ///
    /// Examples
    /// --------
    /// ```
    /// # use nalgebra_sparse::CsrMatrix;
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
    /// # use nalgebra_sparse::CsrMatrix;
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
}

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