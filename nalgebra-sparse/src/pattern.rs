use crate::SparseFormatError;

/// A representation of the sparsity pattern of a CSR or COO matrix.
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
    pub fn new(major_dim: usize, minor_dim: usize) -> Self {
        Self {
            major_offsets: vec![0; major_dim + 1],
            minor_indices: vec![],
            minor_dim,
        }
    }

    /// The offsets for the major dimension.
    #[inline(always)]
    pub fn major_offsets(&self) -> &[usize] {
        &self.major_offsets
    }

    /// The indices for the minor dimension.
    #[inline(always)]
    pub fn minor_indices(&self) -> &[usize] {
        &self.minor_indices
    }

    /// The major dimension.
    #[inline(always)]
    pub fn major_dim(&self) -> usize {
        assert!(self.major_offsets.len() > 0);
        self.major_offsets.len() - 1
    }

    /// The minor dimension.
    #[inline(always)]
    pub fn minor_dim(&self) -> usize {
        self.minor_dim
    }

    /// The number of "non-zeros", i.e. explicitly stored entries in the pattern.
    #[inline(always)]
    pub fn nnz(&self) -> usize {
        self.minor_indices.len()
    }

    /// Get the lane at the given index.
    #[inline(always)]
    pub fn lane(&self, major_index: usize) -> Option<&[usize]> {
        let offset_begin = *self.major_offsets().get(major_index)?;
        let offset_end = *self.major_offsets().get(major_index + 1)?;
        Some(&self.minor_indices()[offset_begin..offset_end])
    }

    /// Try to construct a sparsity pattern from the given dimensions, major offsets
    /// and minor indices.
    ///
    /// Returns an error if the data does not conform to the requirements.
    ///
    /// TODO: Maybe we should not do any assertions in any of the construction functions
    pub fn try_from_offsets_and_indices(
        major_dim: usize,
        minor_dim: usize,
        major_offsets: Vec<usize>,
        minor_indices: Vec<usize>,
    ) -> Result<Self, SparseFormatError> {
        assert_eq!(major_offsets.len(), major_dim + 1);
        assert_eq!(*major_offsets.last().unwrap(), minor_indices.len());
        Ok(Self {
            major_offsets,
            minor_indices,
            minor_dim,
        })
    }

    /// An iterator over the explicitly stored "non-zero" entries (i, j).
    ///
    /// The iteration happens in a lane-major fashion, meaning that the lane index i
    /// increases monotonically. and the minor index j increases monotonically within each
    /// lane i.
    ///
    /// Examples
    /// --------
    ///
    /// ```
    /// # use nalgebra_sparse::{SparsityPattern};
    /// let offsets = vec![0, 2, 3, 4];
    /// let minor_indices = vec![0, 2, 1, 0];
    /// let pattern = SparsityPattern::try_from_offsets_and_indices(3, 4, offsets, minor_indices)
    ///     .unwrap();
    ///
    /// let entries: Vec<_> = pattern.entries().collect();
    /// assert_eq!(entries, vec![(0, 0), (0, 2), (1, 1), (2, 0)]);
    /// ```
    ///
    pub fn entries(&self) -> SparsityPatternIter {
        SparsityPatternIter::from_pattern(self)
    }
}

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
        let minors_in_first_lane = &pattern.minor_indices()[0 .. *first_lane_end];
        Self {
            major_offsets: pattern.major_offsets(),
            minor_indices: pattern.minor_indices(),
            current_lane_idx: 0,
            remaining_minors_in_lane: minors_in_first_lane
        }
    }
}

impl<'a> Iterator for SparsityPatternIter<'a> {
    type Item = (usize, usize);

    #[inline(always)]
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
                        self.remaining_minors_in_lane = &self.minor_indices[(lower + 1) .. upper];
                        return Some((self.current_lane_idx, self.minor_indices[lower]))
                    }
                }
            }
        }
    }
}

