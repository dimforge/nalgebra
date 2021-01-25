use std::mem::replace;
use std::ops::Range;

use num_traits::One;

use nalgebra::Scalar;

use crate::pattern::SparsityPattern;
use crate::{SparseEntry, SparseEntryMut};

/// An abstract compressed matrix.
///
/// For the time being, this is only used internally to share implementation between
/// CSR and CSC matrices.
///
/// A CSR matrix is obtained by associating rows with the major dimension, while a CSC matrix
/// is obtained by associating columns with the major dimension.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CsMatrix<T> {
    sparsity_pattern: SparsityPattern,
    values: Vec<T>,
}

impl<T> CsMatrix<T> {
    /// Create a zero matrix with no explicitly stored entries.
    #[inline]
    pub fn new(major_dim: usize, minor_dim: usize) -> Self {
        Self {
            sparsity_pattern: SparsityPattern::zeros(major_dim, minor_dim),
            values: vec![],
        }
    }

    #[inline]
    pub fn pattern(&self) -> &SparsityPattern {
        &self.sparsity_pattern
    }

    #[inline]
    pub fn values(&self) -> &[T] {
        &self.values
    }

    #[inline]
    pub fn values_mut(&mut self) -> &mut [T] {
        &mut self.values
    }

    /// Returns the raw data represented as a tuple `(major_offsets, minor_indices, values)`.
    #[inline]
    pub fn cs_data(&self) -> (&[usize], &[usize], &[T]) {
        let pattern = self.pattern();
        (
            pattern.major_offsets(),
            pattern.minor_indices(),
            &self.values,
        )
    }

    /// Returns the raw data represented as a tuple `(major_offsets, minor_indices, values)`.
    #[inline]
    pub fn cs_data_mut(&mut self) -> (&[usize], &[usize], &mut [T]) {
        let pattern = &mut self.sparsity_pattern;
        (
            pattern.major_offsets(),
            pattern.minor_indices(),
            &mut self.values,
        )
    }

    #[inline]
    pub fn pattern_and_values_mut(&mut self) -> (&SparsityPattern, &mut [T]) {
        (&self.sparsity_pattern, &mut self.values)
    }

    #[inline]
    pub fn from_pattern_and_values(pattern: SparsityPattern, values: Vec<T>) -> Self {
        assert_eq!(
            pattern.nnz(),
            values.len(),
            "Internal error: consumers should verify shape compatibility."
        );
        Self {
            sparsity_pattern: pattern,
            values,
        }
    }

    /// Internal method for simplifying access to a lane's data
    #[inline]
    pub fn get_index_range(&self, row_index: usize) -> Option<Range<usize>> {
        let row_begin = *self.sparsity_pattern.major_offsets().get(row_index)?;
        let row_end = *self.sparsity_pattern.major_offsets().get(row_index + 1)?;
        Some(row_begin..row_end)
    }

    pub fn take_pattern_and_values(self) -> (SparsityPattern, Vec<T>) {
        (self.sparsity_pattern, self.values)
    }

    #[inline]
    pub fn disassemble(self) -> (Vec<usize>, Vec<usize>, Vec<T>) {
        let (offsets, indices) = self.sparsity_pattern.disassemble();
        (offsets, indices, self.values)
    }

    #[inline]
    pub fn into_pattern_and_values(self) -> (SparsityPattern, Vec<T>) {
        (self.sparsity_pattern, self.values)
    }

    /// Returns an entry for the given major/minor indices, or `None` if the indices are out
    /// of bounds.
    pub fn get_entry(&self, major_index: usize, minor_index: usize) -> Option<SparseEntry<T>> {
        let row_range = self.get_index_range(major_index)?;
        let (_, minor_indices, values) = self.cs_data();
        let minor_indices = &minor_indices[row_range.clone()];
        let values = &values[row_range];
        get_entry_from_slices(
            self.pattern().minor_dim(),
            minor_indices,
            values,
            minor_index,
        )
    }

    /// Returns a mutable entry for the given major/minor indices, or `None` if the indices are out
    /// of bounds.
    pub fn get_entry_mut(
        &mut self,
        major_index: usize,
        minor_index: usize,
    ) -> Option<SparseEntryMut<T>> {
        let row_range = self.get_index_range(major_index)?;
        let minor_dim = self.pattern().minor_dim();
        let (_, minor_indices, values) = self.cs_data_mut();
        let minor_indices = &minor_indices[row_range.clone()];
        let values = &mut values[row_range];
        get_mut_entry_from_slices(minor_dim, minor_indices, values, minor_index)
    }

    pub fn get_lane(&self, index: usize) -> Option<CsLane<T>> {
        let range = self.get_index_range(index)?;
        let (_, minor_indices, values) = self.cs_data();
        Some(CsLane {
            minor_indices: &minor_indices[range.clone()],
            values: &values[range],
            minor_dim: self.pattern().minor_dim(),
        })
    }

    #[inline]
    pub fn get_lane_mut(&mut self, index: usize) -> Option<CsLaneMut<T>> {
        let range = self.get_index_range(index)?;
        let minor_dim = self.pattern().minor_dim();
        let (_, minor_indices, values) = self.cs_data_mut();
        Some(CsLaneMut {
            minor_dim,
            minor_indices: &minor_indices[range.clone()],
            values: &mut values[range],
        })
    }

    #[inline]
    pub fn lane_iter(&self) -> CsLaneIter<T> {
        CsLaneIter::new(self.pattern(), self.values())
    }

    #[inline]
    pub fn lane_iter_mut(&mut self) -> CsLaneIterMut<T> {
        CsLaneIterMut::new(&self.sparsity_pattern, &mut self.values)
    }

    #[inline]
    pub fn filter<P>(&self, predicate: P) -> Self
    where
        T: Clone,
        P: Fn(usize, usize, &T) -> bool,
    {
        let (major_dim, minor_dim) = (self.pattern().major_dim(), self.pattern().minor_dim());
        let mut new_offsets = Vec::with_capacity(self.pattern().major_dim() + 1);
        let mut new_indices = Vec::new();
        let mut new_values = Vec::new();

        new_offsets.push(0);
        for (i, lane) in self.lane_iter().enumerate() {
            for (&j, value) in lane.minor_indices().iter().zip(lane.values) {
                if predicate(i, j, value) {
                    new_indices.push(j);
                    new_values.push(value.clone());
                }
            }

            new_offsets.push(new_indices.len());
        }

        // TODO: Avoid checks here
        let new_pattern = SparsityPattern::try_from_offsets_and_indices(
            major_dim,
            minor_dim,
            new_offsets,
            new_indices,
        )
        .expect("Internal error: Sparsity pattern must always be valid.");

        Self::from_pattern_and_values(new_pattern, new_values)
    }

    /// Returns the diagonal of the matrix as a sparse matrix.
    pub fn diagonal_as_matrix(&self) -> Self
    where
        T: Clone,
    {
        // TODO: This might be faster with a binary search for each diagonal entry
        self.filter(|i, j, _| i == j)
    }
}

impl<T: Scalar + One> CsMatrix<T> {
    #[inline]
    pub fn identity(n: usize) -> Self {
        let offsets: Vec<_> = (0..=n).collect();
        let indices: Vec<_> = (0..n).collect();
        let values = vec![T::one(); n];

        // TODO: We should skip checks here
        let pattern =
            SparsityPattern::try_from_offsets_and_indices(n, n, offsets, indices).unwrap();
        Self::from_pattern_and_values(pattern, values)
    }
}

fn get_entry_from_slices<'a, T>(
    minor_dim: usize,
    minor_indices: &'a [usize],
    values: &'a [T],
    global_minor_index: usize,
) -> Option<SparseEntry<'a, T>> {
    let local_index = minor_indices.binary_search(&global_minor_index);
    if let Ok(local_index) = local_index {
        Some(SparseEntry::NonZero(&values[local_index]))
    } else if global_minor_index < minor_dim {
        Some(SparseEntry::Zero)
    } else {
        None
    }
}

fn get_mut_entry_from_slices<'a, T>(
    minor_dim: usize,
    minor_indices: &'a [usize],
    values: &'a mut [T],
    global_minor_indices: usize,
) -> Option<SparseEntryMut<'a, T>> {
    let local_index = minor_indices.binary_search(&global_minor_indices);
    if let Ok(local_index) = local_index {
        Some(SparseEntryMut::NonZero(&mut values[local_index]))
    } else if global_minor_indices < minor_dim {
        Some(SparseEntryMut::Zero)
    } else {
        None
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CsLane<'a, T> {
    minor_dim: usize,
    minor_indices: &'a [usize],
    values: &'a [T],
}

#[derive(Debug, PartialEq, Eq)]
pub struct CsLaneMut<'a, T> {
    minor_dim: usize,
    minor_indices: &'a [usize],
    values: &'a mut [T],
}

pub struct CsLaneIter<'a, T> {
    // The index of the lane that will be returned on the next iteration
    current_lane_idx: usize,
    pattern: &'a SparsityPattern,
    remaining_values: &'a [T],
}

impl<'a, T> CsLaneIter<'a, T> {
    pub fn new(pattern: &'a SparsityPattern, values: &'a [T]) -> Self {
        Self {
            current_lane_idx: 0,
            pattern,
            remaining_values: values,
        }
    }
}

impl<'a, T> Iterator for CsLaneIter<'a, T>
where
    T: 'a,
{
    type Item = CsLane<'a, T>;

    fn next(&mut self) -> Option<Self::Item> {
        let lane = self.pattern.get_lane(self.current_lane_idx);
        let minor_dim = self.pattern.minor_dim();

        if let Some(minor_indices) = lane {
            let count = minor_indices.len();
            let values_in_lane = &self.remaining_values[..count];
            self.remaining_values = &self.remaining_values[count..];
            self.current_lane_idx += 1;

            Some(CsLane {
                minor_dim,
                minor_indices,
                values: values_in_lane,
            })
        } else {
            None
        }
    }
}

pub struct CsLaneIterMut<'a, T> {
    // The index of the lane that will be returned on the next iteration
    current_lane_idx: usize,
    pattern: &'a SparsityPattern,
    remaining_values: &'a mut [T],
}

impl<'a, T> CsLaneIterMut<'a, T> {
    pub fn new(pattern: &'a SparsityPattern, values: &'a mut [T]) -> Self {
        Self {
            current_lane_idx: 0,
            pattern,
            remaining_values: values,
        }
    }
}

impl<'a, T> Iterator for CsLaneIterMut<'a, T>
where
    T: 'a,
{
    type Item = CsLaneMut<'a, T>;

    fn next(&mut self) -> Option<Self::Item> {
        let lane = self.pattern.get_lane(self.current_lane_idx);
        let minor_dim = self.pattern.minor_dim();

        if let Some(minor_indices) = lane {
            let count = minor_indices.len();

            let remaining = replace(&mut self.remaining_values, &mut []);
            let (values_in_lane, remaining) = remaining.split_at_mut(count);
            self.remaining_values = remaining;
            self.current_lane_idx += 1;

            Some(CsLaneMut {
                minor_dim,
                minor_indices,
                values: values_in_lane,
            })
        } else {
            None
        }
    }
}

/// Implement the methods common to both CsLane and CsLaneMut. See the documentation for the
/// methods delegated here by CsrMatrix and CscMatrix members for more information.
macro_rules! impl_cs_lane_common_methods {
    ($name:ty) => {
        impl<'a, T> $name {
            #[inline]
            pub fn minor_dim(&self) -> usize {
                self.minor_dim
            }

            #[inline]
            pub fn nnz(&self) -> usize {
                self.minor_indices.len()
            }

            #[inline]
            pub fn minor_indices(&self) -> &[usize] {
                self.minor_indices
            }

            #[inline]
            pub fn values(&self) -> &[T] {
                self.values
            }

            #[inline]
            pub fn get_entry(&self, global_col_index: usize) -> Option<SparseEntry<T>> {
                get_entry_from_slices(
                    self.minor_dim,
                    self.minor_indices,
                    self.values,
                    global_col_index,
                )
            }
        }
    };
}

impl_cs_lane_common_methods!(CsLane<'a, T>);
impl_cs_lane_common_methods!(CsLaneMut<'a, T>);

impl<'a, T> CsLaneMut<'a, T> {
    pub fn values_mut(&mut self) -> &mut [T] {
        self.values
    }

    pub fn indices_and_values_mut(&mut self) -> (&[usize], &mut [T]) {
        (self.minor_indices, self.values)
    }

    pub fn get_entry_mut(&mut self, global_minor_index: usize) -> Option<SparseEntryMut<T>> {
        get_mut_entry_from_slices(
            self.minor_dim,
            self.minor_indices,
            self.values,
            global_minor_index,
        )
    }
}

/// Helper struct for working with uninitialized data in vectors.
/// TODO: This doesn't belong here.
struct UninitVec<T> {
    vec: Vec<T>,
    len: usize,
}

impl<T> UninitVec<T> {
    pub fn from_len(len: usize) -> Self {
        Self {
            vec: Vec::with_capacity(len),
            // We need to store len separately, because for zero-sized types,
            // Vec::with_capacity(len) does not give vec.capacity() == len
            len,
        }
    }

    /// Sets the element associated with the given index to the provided value.
    ///
    /// Must be called exactly once per index, otherwise results in undefined behavior.
    pub unsafe fn set(&mut self, index: usize, value: T) {
        self.vec.as_mut_ptr().add(index).write(value)
    }

    /// Marks the vector data as initialized by returning a full vector.
    ///
    /// It is undefined behavior to call this function unless *all* elements have been written to
    /// exactly once.
    pub unsafe fn assume_init(mut self) -> Vec<T> {
        self.vec.set_len(self.len);
        self.vec
    }
}

/// Transposes the compressed format.
///
/// This means that major and minor roles are switched. This is used for converting between CSR
/// and CSC formats.
pub fn transpose_cs<T>(
    major_dim: usize,
    minor_dim: usize,
    source_major_offsets: &[usize],
    source_minor_indices: &[usize],
    values: &[T],
) -> (Vec<usize>, Vec<usize>, Vec<T>)
where
    T: Scalar,
{
    assert_eq!(source_major_offsets.len(), major_dim + 1);
    assert_eq!(source_minor_indices.len(), values.len());
    let nnz = values.len();

    // Count the number of occurences of each minor index
    let mut minor_counts = vec![0; minor_dim];
    for minor_idx in source_minor_indices {
        minor_counts[*minor_idx] += 1;
    }
    convert_counts_to_offsets(&mut minor_counts);
    let mut target_offsets = minor_counts;
    target_offsets.push(nnz);
    let mut target_indices = vec![usize::MAX; nnz];

    // We have to use uninitialized storage, because we don't have any kind of "default" value
    // available for `T`. Unfortunately this necessitates some small amount of unsafe code
    let mut target_values = UninitVec::from_len(nnz);

    // Keep track of how many entries we have placed in each target major lane
    let mut current_target_major_counts = vec![0; minor_dim];

    for source_major_idx in 0..major_dim {
        let source_lane_begin = source_major_offsets[source_major_idx];
        let source_lane_end = source_major_offsets[source_major_idx + 1];
        let source_lane_indices = &source_minor_indices[source_lane_begin..source_lane_end];
        let source_lane_values = &values[source_lane_begin..source_lane_end];

        for (&source_minor_idx, val) in source_lane_indices.iter().zip(source_lane_values) {
            // Compute the offset in the target data for this particular source entry
            let target_lane_count = &mut current_target_major_counts[source_minor_idx];
            let entry_offset = target_offsets[source_minor_idx] + *target_lane_count;
            target_indices[entry_offset] = source_major_idx;
            unsafe {
                target_values.set(entry_offset, val.inlined_clone());
            }
            *target_lane_count += 1;
        }
    }

    // At this point, we should have written to each element in target_values exactly once,
    // so initialization should be sound
    let target_values = unsafe { target_values.assume_init() };
    (target_offsets, target_indices, target_values)
}

pub fn convert_counts_to_offsets(counts: &mut [usize]) {
    // Convert the counts to an offset
    let mut offset = 0;
    for i_offset in counts.iter_mut() {
        let count = *i_offset;
        *i_offset = offset;
        offset += count;
    }
}
