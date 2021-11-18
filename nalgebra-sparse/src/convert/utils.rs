//! Module for utility functions used in format conversions.

use nalgebra::Scalar;
use num_traits::Unsigned;
use std::ops::Add;

/// Sort the indices of the given lane.
///
/// The indices and values in `minor_idx` and `values` are sorted according to the
/// minor indices and stored in `minor_idx_result` and `values_result` respectively.
///
/// All input slices are expected to be of the same length. The contents of mutable slices
/// can be arbitrary, as they are anyway overwritten.
pub(crate) fn sort_lane<T: Clone>(
    minor_idx_result: &mut [usize],
    values_result: &mut [T],
    minor_idx: &[usize],
    values: &[T],
    workspace: &mut [usize],
) {
    assert_eq!(minor_idx_result.len(), values_result.len());
    assert_eq!(values_result.len(), minor_idx.len());
    assert_eq!(minor_idx.len(), values.len());
    assert_eq!(values.len(), workspace.len());

    let permutation = workspace;
    // Set permutation to identity
    for (i, p) in permutation.iter_mut().enumerate() {
        *p = i;
    }

    // Compute permutation needed to bring minor indices into sorted order
    // Note: Using sort_unstable here avoids internal allocations, which is crucial since
    // each lane might have a small number of elements
    permutation.sort_unstable_by_key(|idx| minor_idx[*idx]);

    apply_permutation(minor_idx_result, minor_idx, permutation);
    apply_permutation(values_result, values, permutation);
}

fn apply_permutation<T: Clone>(out_slice: &mut [T], in_slice: &[T], permutation: &[usize]) {
    assert_eq!(out_slice.len(), in_slice.len());
    assert_eq!(out_slice.len(), permutation.len());
    for (out_element, old_pos) in out_slice.iter_mut().zip(permutation) {
        *out_element = in_slice[*old_pos].clone();
    }
}

/// Given *sorted* indices and corresponding scalar values, combines duplicates with the given
/// associative combiner and calls the provided produce methods with combined indices and values.
pub(crate) fn combine_duplicates<T: Clone>(
    mut produce_idx: impl FnMut(usize),
    mut produce_value: impl FnMut(T),
    idx_array: &[usize],
    values: &[T],
    combiner: impl Fn(T, T) -> T,
) {
    assert_eq!(idx_array.len(), values.len());

    let mut i = 0;
    while i < idx_array.len() {
        let idx = idx_array[i];
        let mut combined_value = values[i].clone();
        let mut j = i + 1;
        while j < idx_array.len() && idx_array[j] == idx {
            let j_val = values[j].clone();
            combined_value = combiner(combined_value, j_val);
            j += 1;
        }
        produce_idx(idx);
        produce_value(combined_value);
        i = j;
    }
}

/// Helper struct for working with uninitialized data in vectors.
struct UninitVec<T> {
    vec: Vec<T>,
    len: usize,
}

impl<T> UninitVec<T> {
    fn from_len(len: usize) -> Self {
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
    unsafe fn set(&mut self, index: usize, value: T) {
        self.vec.as_mut_ptr().add(index).write(value)
    }

    /// Marks the vector data as initialized by returning a full vector.
    ///
    /// It is undefined behavior to call this function unless *all* elements have been written to
    /// exactly once.
    unsafe fn assume_init(mut self) -> Vec<T> {
        self.vec.set_len(self.len);
        self.vec
    }
}

/// Transposes the compressed raw data `(offsets, indices, data)` by recomputing the offsets and
/// indices for each lane.
///
/// Unlike `CsMatrix::transpose`, this function does not transpose the data for "free," by swapping
/// the shape and changing the storage from CSC -> CSR (or vice-versa).
///
/// This is an expensive recomputation of the matrix, but can be useful for converting between CSC
/// and CSR formats, if you want the same data (but actually want the compressed format to
/// recompute it).
pub(crate) fn transpose_convert<T, Offset, Index>(
    major_dim: usize,
    minor_dim: usize,
    source_major_offsets: &[Offset],
    source_minor_indices: &[Index],
    values: &[T],
) -> (Vec<usize>, Vec<usize>, Vec<T>)
where
    T: Scalar,
    Offset: Add<usize, Output = usize> + Copy + Clone + Into<usize> + Unsigned + Ord,
    Index: Copy + Clone + Into<usize> + Unsigned + Ord,
{
    assert_eq!(source_major_offsets.len(), major_dim + 1);
    assert_eq!(source_minor_indices.len(), values.len());
    let nnz = values.len();

    // Count the number of occurences of each minor index
    let mut minor_counts = vec![0; minor_dim];
    for minor_idx in source_minor_indices {
        minor_counts[(*minor_idx).into()] += 1;
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
        let source_lane_begin = source_major_offsets[source_major_idx].into();
        let source_lane_end = source_major_offsets[source_major_idx + 1].into();
        let source_lane_indices = &source_minor_indices[source_lane_begin..source_lane_end];
        let source_lane_values = &values[source_lane_begin..source_lane_end];

        for (&source_minor_idx, val) in source_lane_indices.iter().zip(source_lane_values) {
            // Compute the offset in the target data for this particular source entry
            let target_lane_count = &mut current_target_major_counts[source_minor_idx.into()];
            let entry_offset = target_offsets[source_minor_idx.into()] + *target_lane_count;
            target_indices[entry_offset] = source_major_idx;
            unsafe {
                target_values.set(entry_offset, val.clone());
            }
            *target_lane_count += 1;
        }
    }

    // At this point, we should have written to each element in target_values exactly once,
    // so initialization should be sound
    let target_values = unsafe { target_values.assume_init() };
    (target_offsets, target_indices, target_values)
}

pub(crate) fn convert_counts_to_offsets(counts: &mut [usize]) {
    // Convert the counts to an offset
    let mut offset = 0;
    for i_offset in counts.iter_mut() {
        let count = *i_offset;
        *i_offset = offset;
        offset += count;
    }
}
