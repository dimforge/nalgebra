//! Sparsity patterns for CSR and CSC matrices.

#[cfg(feature = "serde-serialize")]
mod pattern_serde;

use crate::cs::transpose_cs;
use crate::SparseFormatError;
use std::error::Error;
use std::fmt;

/// A representation of the sparsity pattern of a CSR or CSC matrix.
///
/// CSR and CSC matrices store matrices in a very similar fashion. In fact, in a certain sense,
/// they are transposed. More precisely, when reinterpreting the three data arrays of a CSR
/// matrix as a CSC matrix, we obtain the CSC representation of its transpose.
///
/// [`SparsityPattern`] is an abstraction built on this observation. Whereas CSR matrices
/// store a matrix row-by-row, and a CSC matrix stores a matrix column-by-column, a
/// `SparsityPattern` represents only the index data structure of a matrix *lane-by-lane*.
/// Here, a *lane* is a generalization of rows and columns. We further define *major lanes*
/// and *minor lanes*. The sparsity pattern of a CSR matrix is then obtained by interpreting
/// major/minor as row/column. Conversely, we obtain the sparsity pattern of a CSC matrix by
/// interpreting major/minor as column/row.
///
/// This allows us to use a common abstraction to talk about sparsity patterns of CSR and CSC
/// matrices. This is convenient, because at the abstract level, the invariants of the formats
/// are the same. Hence we may encode the invariants of the index data structure separately from
/// the scalar values of the matrix. This is especially useful in applications where the
/// sparsity pattern is built ahead of the matrix values, or the same sparsity pattern is re-used
/// between different matrices. Finally, we can use `SparsityPattern` to encode adjacency
/// information in graphs.
///
/// # Format
///
/// The format is exactly the same as for the index data structures of CSR and CSC matrices.
/// This means that the sparsity pattern of an `m x n` sparse matrix with `nnz` non-zeros,
/// where in this case `m x n` does *not* mean `rows x columns`, but rather `majors x minors`,
/// is represented by the following two arrays:
///
/// - `major_offsets`, an array of integers with length `m + 1`.
/// - `minor_indices`, an array of integers with length `nnz`.
///
/// The invariants and relationship between `major_offsets` and `minor_indices` remain the same
/// as for `row_offsets` and `col_indices` in the [CSR](`crate::csr::CsrMatrix`) format
/// specification.
#[derive(Debug, Clone, PartialEq, Eq)]
// TODO: Make SparsityPattern parametrized by index type
// (need a solid abstraction for index types though)
pub struct SparsityPattern {
    major_offsets: Vec<usize>,
    minor_indices: Vec<usize>,
    minor_dim: usize,
}

impl SparsityPattern {
    /// Create a sparsity pattern of the given dimensions without explicitly stored entries.
    pub fn zeros(major_dim: usize, minor_dim: usize) -> Self {
        Self {
            major_offsets: vec![0; major_dim + 1],
            minor_indices: vec![],
            minor_dim,
        }
    }
    /// Creates the sparsity pattern of an identity matrix of size `n`.
    pub fn identity(n: usize) -> Self {
        Self {
            major_offsets: (0..=n).collect(),
            minor_indices: (0..n).collect(),
            minor_dim: n,
        }
    }

    /// The offsets for the major dimension.
    #[inline]
    #[must_use]
    pub fn major_offsets(&self) -> &[usize] {
        &self.major_offsets
    }

    /// The indices for the minor dimension.
    #[inline]
    #[must_use]
    pub fn minor_indices(&self) -> &[usize] {
        &self.minor_indices
    }

    /// The number of major lanes in the pattern.
    #[inline]
    #[must_use]
    pub fn major_dim(&self) -> usize {
        assert!(!self.major_offsets.is_empty());
        self.major_offsets.len() - 1
    }

    /// The number of minor lanes in the pattern.
    #[inline]
    #[must_use]
    pub fn minor_dim(&self) -> usize {
        self.minor_dim
    }

    /// The number of "non-zeros", i.e. explicitly stored entries in the pattern.
    #[inline]
    #[must_use]
    pub fn nnz(&self) -> usize {
        self.minor_indices.len()
    }

    /// Get the lane at the given index.
    ///
    /// Panics
    /// ------
    ///
    /// Panics if `major_index` is out of bounds.
    #[inline]
    #[must_use]
    pub fn lane(&self, major_index: usize) -> &[usize] {
        self.get_lane(major_index).unwrap()
    }
    /// Returns an iterator over all lanes in this sparsity pattern, in order.
    /// Does not omit empty lanes.
    #[inline]
    #[must_use]
    pub fn lanes(&self) -> impl Iterator<Item = &[usize]> {
        (0..self.major_offsets.len() - 1).map(move |r| self.lane(r))
    }

    /// Get the lane at the given index, or `None` if out of bounds.
    #[inline]
    #[must_use]
    pub fn get_lane(&self, major_index: usize) -> Option<&[usize]> {
        let offset_begin = *self.major_offsets().get(major_index)?;
        let offset_end = *self.major_offsets().get(major_index + 1)?;
        Some(&self.minor_indices()[offset_begin..offset_end])
    }

    /// Try to construct a sparsity pattern from the given dimensions, major offsets
    /// and minor indices.
    ///
    /// Returns an error if the data does not conform to the requirements.
    pub fn try_from_offsets_and_indices(
        major_dim: usize,
        minor_dim: usize,
        major_offsets: Vec<usize>,
        minor_indices: Vec<usize>,
    ) -> Result<Self, SparsityPatternFormatError> {
        use SparsityPatternFormatError::*;

        if major_offsets.len() != major_dim + 1 {
            return Err(InvalidOffsetArrayLength);
        }

        // Check that the first and last offsets conform to the specification
        {
            let first_offset_ok = *major_offsets.first().unwrap() == 0;
            let last_offset_ok = *major_offsets.last().unwrap() == minor_indices.len();
            if !first_offset_ok || !last_offset_ok {
                return Err(InvalidOffsetFirstLast);
            }
        }

        // Test that each lane has strictly monotonically increasing minor indices, i.e.
        // minor indices within a lane are sorted, unique. In addition, each minor index
        // must be in bounds with respect to the minor dimension.
        {
            for lane_idx in 0..major_dim {
                let range_start = major_offsets[lane_idx];
                let range_end = major_offsets[lane_idx + 1];

                // Test that major offsets are monotonically increasing
                if range_start > range_end {
                    return Err(NonmonotonicOffsets);
                }

                let minor_indices = &minor_indices[range_start..range_end];

                // We test for in-bounds, uniqueness and monotonicity at the same time
                // to ensure that we only visit each minor index once
                let mut iter = minor_indices.iter();
                let mut prev: Option<usize> = None;

                while let Some(next) = iter.next().copied() {
                    if next >= minor_dim {
                        return Err(MinorIndexOutOfBounds);
                    }

                    if let Some(prev) = prev {
                        match prev.cmp(&next) {
                            std::cmp::Ordering::Greater => return Err(NonmonotonicMinorIndices),
                            std::cmp::Ordering::Equal => return Err(DuplicateEntry),
                            std::cmp::Ordering::Less => {}
                        }
                    }
                    prev = Some(next);
                }
            }
        }

        Ok(Self {
            major_offsets,
            minor_indices,
            minor_dim,
        })
    }

    /// Try to construct a sparsity pattern from the given dimensions, major offsets
    /// and minor indices.
    ///
    /// # Panics
    ///
    /// Panics if the number of major offsets is not exactly one greater than the major dimension
    /// or if major offsets do not start with 0 and end with the number of minor indices.
    ///
    /// # Safety
    ///
    /// Assumes that the major offsets and indices adhere to the requirements of being a valid
    /// sparsity pattern.
    /// Specifically, that major offsets is monotonically increasing, and
    /// `major_offsets[i]..major_offsets[i+1]` refers to a major lane in the sparsity pattern,
    /// and `minor_indices[major_offsets[i]..major_offsets[i+1]]` is monotonically increasing.
    pub unsafe fn from_offset_and_indices_unchecked(
        major_dim: usize,
        minor_dim: usize,
        major_offsets: Vec<usize>,
        minor_indices: Vec<usize>,
    ) -> Self {
        assert_eq!(major_offsets.len(), major_dim + 1);

        // Check that the first and last offsets conform to the specification
        {
            let first_offset_ok = *major_offsets.first().unwrap() == 0;
            let last_offset_ok = *major_offsets.last().unwrap() == minor_indices.len();
            assert!(first_offset_ok && last_offset_ok);
        }

        Self {
            major_offsets,
            minor_indices,
            minor_dim,
        }
    }

    /// An iterator over the explicitly stored "non-zero" entries (i, j).
    ///
    /// The iteration happens in a lane-major fashion, meaning that the lane index i
    /// increases monotonically, and the minor index j increases monotonically within each
    /// lane i.
    ///
    /// Examples
    /// --------
    ///
    /// ```
    /// # use nalgebra_sparse::pattern::SparsityPattern;
    /// let offsets = vec![0, 2, 3, 4];
    /// let minor_indices = vec![0, 2, 1, 0];
    /// let pattern = SparsityPattern::try_from_offsets_and_indices(3, 4, offsets, minor_indices)
    ///     .unwrap();
    ///
    /// let entries: Vec<_> = pattern.entries().collect();
    /// assert_eq!(entries, vec![(0, 0), (0, 2), (1, 1), (2, 0)]);
    /// ```
    ///
    #[must_use]
    pub fn entries(&self) -> SparsityPatternIter<'_> {
        SparsityPatternIter::from_pattern(self)
    }

    /// Returns the raw offset and index data for the sparsity pattern.
    ///
    /// Examples
    /// --------
    ///
    /// ```
    /// # use nalgebra_sparse::pattern::SparsityPattern;
    /// let offsets = vec![0, 2, 3, 4];
    /// let minor_indices = vec![0, 2, 1, 0];
    /// let pattern = SparsityPattern::try_from_offsets_and_indices(
    ///         3,
    ///         4,
    ///         offsets.clone(),
    ///         minor_indices.clone())
    ///     .unwrap();
    /// let (offsets2, minor_indices2) = pattern.disassemble();
    /// assert_eq!(offsets2, offsets);
    /// assert_eq!(minor_indices2, minor_indices);
    /// ```
    pub fn disassemble(self) -> (Vec<usize>, Vec<usize>) {
        (self.major_offsets, self.minor_indices)
    }

    /// Computes the transpose of the sparsity pattern.
    ///
    /// This is analogous to matrix transposition, i.e. an entry `(i, j)` becomes `(j, i)` in the
    /// new pattern.
    #[must_use]
    pub fn transpose(&self) -> Self {
        // By using unit () values, we can use the same routines as for CSR/CSC matrices
        let values = vec![(); self.nnz()];
        let (new_offsets, new_indices, _) = transpose_cs(
            self.major_dim(),
            self.minor_dim(),
            self.major_offsets(),
            self.minor_indices(),
            &values,
        );
        // TODO: Skip checks
        Self::try_from_offsets_and_indices(
            self.minor_dim(),
            self.major_dim(),
            new_offsets,
            new_indices,
        )
        .expect("Internal error: Transpose should never fail.")
    }

    /// Computes the output sparsity pattern of `x` in `Ax = b`.
    /// where A's nonzero pattern is given by `self` and the non-zero indices
    /// of vector `b` are specified as a slice.
    /// The output is not necessarily in sorted order, but is topological sort order.
    /// Treats `self` as lower triangular, even if there are elements in the upper triangle.
    /// Acts as if b is one major lane (i.e. CSC matrix and one column)
    pub fn sparse_lower_triangular_solve(&self, b: &[usize], out: &mut Vec<usize>) {
        assert!(b.iter().all(|&i| i < self.major_dim()));
        out.clear();

        // From a given starting column, traverses and finds all reachable indices.
        fn reach(sp: &SparsityPattern, j: usize, out: &mut Vec<usize>) {
            // already traversed
            if out.contains(&j) {
                return;
            }

            out.push(j);
            for &i in sp.lane(j) {
                if i < j {
                    continue;
                }
                reach(sp, i, out);
            }
        }

        for &i in b {
            reach(&self, i, out);
        }
    }

    /// Computes the output sparsity pattern of `x` in `Ax = b`.
    /// where A's nonzero pattern is given by `self` and the non-zero indices
    /// of vector `b` are specified as a slice.
    /// The output is not necessarily in sorted order, but is topological sort order.
    /// Treats `self` as upper triangular, even if there are elements in the lower triangle.
    /// Acts as if b is one major lane (i.e. CSC matrix and one column)
    pub fn sparse_upper_triangular_solve(&self, b: &[usize], out: &mut Vec<usize>) {
        assert!(b.iter().all(|&i| i < self.major_dim()));
        out.clear();

        // From a given starting column, traverses and finds all reachable indices.
        fn reach(sp: &SparsityPattern, j: usize, out: &mut Vec<usize>) {
            // already traversed
            if out.contains(&j) {
                return;
            }

            out.push(j);
            // iteration order here does not matter, but technically it should be rev?
            for &i in sp.lane(j).iter().rev() {
                if i > j {
                    continue;
                }
                reach(sp, i, out);
            }
        }

        for &i in b {
            reach(&self, i, out);
        }
    }

    /// Left-looking Sparse LU decomposition from Gilbert/Peierls.
    /// returns the sparsity pattern of the output
    pub fn left_looking_lu_decomposition(&self) -> SparsityPattern {
        assert_eq!(self.major_dim(), self.minor_dim());
        let n = self.minor_dim();
        let mut sp = SparsityPatternBuilder::new(n, n);
        let mut x = vec![];
        for col in 0..n {
            sp.valid_partial()
                .sparse_lower_triangular_solve(self.lane(col), &mut x);
            x.sort_unstable();
            for &row in &x {
                assert!(sp.insert(col, row).is_ok());
            }
        }
        sp.build()
    }
}

/// A builder that allows for constructing a sparsity pattern.
/// It requires elements to be added in sorted order. Specifically,
/// For each element the major must be >= the previous element's major.
/// If the major is the same, the minor must be in ascending order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SparsityPatternBuilder {
    buf: SparsityPattern,
    major_dim: usize,
}

/// An error when adding into the SparsityPatternBuilder
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum BuilderInsertError {
    ///
    MajorTooLow,
    ///
    MinorTooLow,
}

impl SparsityPatternBuilder {
    /// Constructs a new empty builder.
    pub fn new(major_dim: usize, minor_dim: usize) -> Self {
        Self {
            buf: SparsityPattern {
                major_offsets: vec![0],
                minor_indices: vec![],
                minor_dim,
            },
            major_dim,
        }
    }
    /// The number of non-zero entries inserted into `self`.
    pub fn num_entries(&self) -> usize {
        self.buf.minor_indices.len()
    }

    /// Allows for general assignment of indices
    pub fn insert(&mut self, maj: usize, min: usize) -> Result<(), BuilderInsertError> {
        assert!(maj < self.major_dim);
        assert!(min < self.buf.minor_dim);

        let curr_major = self.buf.major_dim();

        // cannot go backwards in major
        if maj < curr_major {
            return Err(BuilderInsertError::MajorTooLow);
        }
        // cannot go backwards in minor
        if maj == curr_major
            && *self.buf.major_offsets.last().unwrap() < self.buf.minor_indices.len()
            && !self.buf.minor_indices.is_empty()
            && min <= *self.buf.minor_indices.last().unwrap()
        {
            return Err(BuilderInsertError::MinorTooLow);
        }
        // add any advances in row.
        for _ in curr_major..maj {
            self.buf.major_offsets.push(self.buf.minor_indices.len());
        }
        self.buf.minor_indices.push(min);
        Ok(())
    }
    /// Returns a valid partial sparsity pattern.
    /// All the major lanes up to the current insertion will be completed.
    pub(crate) fn valid_partial(&mut self) -> &SparsityPattern {
        if *self.buf.major_offsets.last().unwrap() != self.buf.minor_indices.len() {
            self.buf.major_offsets.push(self.buf.minor_indices.len());
        }
        &self.buf
    }
    /// Consumes self and outputs the constructed `SparsityPattern`.
    /// If elements were added to the last major, but `advance_major`
    /// was not called, will implicitly call `advance_major` then
    /// output the values.
    #[inline]
    pub fn build(mut self) -> SparsityPattern {
        self.buf
            .major_offsets
            .resize(self.major_dim + 1, self.buf.minor_indices.len());
        assert_eq!(self.buf.major_dim(), self.major_dim);
        self.buf
    }
    /// Returns the current major being modified by `self`.
    pub fn current_major(&self) -> usize {
        assert!(!self.buf.major_offsets.is_empty());
        self.buf.major_offsets.len() - 1
    }

    /// Reverts the major index of `self` back to `maj`, deleting any entries ahead of it.
    /// Preserves entries in `maj`.
    pub fn revert_to_major(&mut self, maj: usize) -> bool {
        // preserve maj + 1 elements in self
        if self.buf.major_offsets.len() + 1 <= maj {
            return false;
        }
        let last = self.buf.major_offsets[maj + 1];
        self.buf.major_offsets.truncate(maj + 1);
        self.buf.minor_indices.truncate(last + 1);
        true
    }

    /// Allows for rebuilding part of a sparsity pattern, assuming that
    /// items after maj_start have not been filled in.
    pub fn from(sp: SparsityPattern) -> Self {
        SparsityPatternBuilder {
            major_dim: sp.major_dim(),
            buf: sp,
        }
    }
}

impl Default for SparsityPattern {
    fn default() -> Self {
        Self {
            major_offsets: vec![0],
            minor_indices: vec![],
            minor_dim: 0,
        }
    }
}

/// Error type for `SparsityPattern` format errors.
#[non_exhaustive]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum SparsityPatternFormatError {
    /// Indicates an invalid number of offsets.
    ///
    /// The number of offsets must be equal to (major_dim + 1).
    InvalidOffsetArrayLength,
    /// Indicates that the first or last entry in the offset array did not conform to
    /// specifications.
    ///
    /// The first entry must be 0, and the last entry must be exactly one greater than the
    /// major dimension.
    InvalidOffsetFirstLast,
    /// Indicates that the major offsets are not monotonically increasing.
    NonmonotonicOffsets,
    /// One or more minor indices are out of bounds.
    MinorIndexOutOfBounds,
    /// One or more duplicate entries were detected.
    ///
    /// Two entries are considered duplicates if they are part of the same major lane and have
    /// the same minor index.
    DuplicateEntry,
    /// Indicates that minor indices are not monotonically increasing within each lane.
    NonmonotonicMinorIndices,
}

impl From<SparsityPatternFormatError> for SparseFormatError {
    fn from(err: SparsityPatternFormatError) -> Self {
        use crate::SparseFormatErrorKind;
        use crate::SparseFormatErrorKind::*;
        use SparsityPatternFormatError::DuplicateEntry as PatternDuplicateEntry;
        use SparsityPatternFormatError::*;
        match err {
            InvalidOffsetArrayLength
            | InvalidOffsetFirstLast
            | NonmonotonicOffsets
            | NonmonotonicMinorIndices => {
                SparseFormatError::from_kind_and_error(InvalidStructure, Box::from(err))
            }
            MinorIndexOutOfBounds => {
                SparseFormatError::from_kind_and_error(IndexOutOfBounds, Box::from(err))
            }
            PatternDuplicateEntry => SparseFormatError::from_kind_and_error(
                #[allow(unused_qualifications)]
                SparseFormatErrorKind::DuplicateEntry,
                Box::from(err),
            ),
        }
    }
}

impl fmt::Display for SparsityPatternFormatError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SparsityPatternFormatError::InvalidOffsetArrayLength => {
                write!(f, "Length of offset array is not equal to (major_dim + 1).")
            }
            SparsityPatternFormatError::InvalidOffsetFirstLast => {
                write!(f, "First or last offset is incompatible with format.")
            }
            SparsityPatternFormatError::NonmonotonicOffsets => {
                write!(f, "Offsets are not monotonically increasing.")
            }
            SparsityPatternFormatError::MinorIndexOutOfBounds => {
                write!(f, "A minor index is out of bounds.")
            }
            SparsityPatternFormatError::DuplicateEntry => {
                write!(f, "Input data contains duplicate entries.")
            }
            SparsityPatternFormatError::NonmonotonicMinorIndices => {
                write!(
                    f,
                    "Minor indices are not monotonically increasing within each lane."
                )
            }
        }
    }
}

impl Error for SparsityPatternFormatError {}

/// Iterator type for iterating over entries in a sparsity pattern.
#[derive(Debug, Clone)]
pub struct SparsityPatternIter<'a> {
    // See implementation of Iterator::next for an explanation of how these members are used
    major_offsets: &'a [usize],
    minor_indices: &'a [usize],
    current_lane_idx: usize,
    remaining_minors_in_lane: &'a [usize],
}

impl<'a> SparsityPatternIter<'a> {
    fn from_pattern(pattern: &'a SparsityPattern) -> Self {
        let first_lane_end = pattern.major_offsets().get(1).unwrap_or(&0);
        let minors_in_first_lane = &pattern.minor_indices()[0..*first_lane_end];
        Self {
            major_offsets: pattern.major_offsets(),
            minor_indices: pattern.minor_indices(),
            current_lane_idx: 0,
            remaining_minors_in_lane: minors_in_first_lane,
        }
    }
}

impl<'a> Iterator for SparsityPatternIter<'a> {
    type Item = (usize, usize);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        // We ensure fast iteration across each lane by iteratively "draining" a slice
        // corresponding to the remaining column indices in the particular lane.
        // When we reach the end of this slice, we are at the end of a lane,
        // and we must do some bookkeeping for preparing the iteration of the next lane
        // (or stop iteration if we're through all lanes).
        // This way we can avoid doing unnecessary bookkeeping on every iteration,
        // instead paying a small price whenever we jump to a new lane.
        if let Some(minor_idx) = self.remaining_minors_in_lane.first() {
            let item = Some((self.current_lane_idx, *minor_idx));
            self.remaining_minors_in_lane = &self.remaining_minors_in_lane[1..];
            item
        } else {
            loop {
                // Keep skipping lanes until we found a non-empty lane or there are no more lanes
                if self.current_lane_idx + 2 >= self.major_offsets.len() {
                    // We've processed all lanes, so we're at the end of the iterator
                    // (note: keep in mind that offsets.len() == major_dim() + 1, hence we need +2)
                    return None;
                } else {
                    // Bump lane index and check if the lane is non-empty
                    self.current_lane_idx += 1;
                    let lower = self.major_offsets[self.current_lane_idx];
                    let upper = self.major_offsets[self.current_lane_idx + 1];
                    if upper > lower {
                        self.remaining_minors_in_lane = &self.minor_indices[(lower + 1)..upper];
                        return Some((self.current_lane_idx, self.minor_indices[lower]));
                    }
                }
            }
        }
    }
}
