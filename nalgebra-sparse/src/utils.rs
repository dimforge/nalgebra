//! Helper functions for sparse matrix computations

/// permutes entries of in_slice according to permutation slice and puts them to out_slice
#[inline]
pub fn apply_permutation<T: Clone>(out_slice: &mut [T], in_slice: &[T], permutation: &[usize]) {
    assert_eq!(out_slice.len(), in_slice.len());
    assert_eq!(out_slice.len(), permutation.len());
    for (out_element, old_pos) in out_slice.iter_mut().zip(permutation) {
        *out_element = in_slice[*old_pos].clone();
    }
}

/// computes permutation by using provided indices as keys
#[inline]
pub fn compute_sort_permutation(
    minor_index_permutation: &mut [usize],
    minor_idx_in_lane: &[usize],
) {
    assert_eq!(minor_index_permutation.len(), minor_idx_in_lane.len());
    // Set permutation to identity
    for (i, p) in minor_index_permutation.iter_mut().enumerate() {
        *p = i;
    }

    // Compute permutation needed to bring minor indices into sorted order
    // Note: Using sort_unstable here avoids internal allocations, which is crucial since
    // each lane might have a small number of elements
    minor_index_permutation.sort_unstable_by_key(|idx| minor_idx_in_lane[*idx]);
}
