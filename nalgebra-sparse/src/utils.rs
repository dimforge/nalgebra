//! Helper functions for sparse matrix computations

/// Check that the first and last offsets conform to the specification of a CSR matrix
#[inline]
#[must_use]
pub fn first_and_last_offsets_are_ok(
    major_offsets: &Vec<usize>,
    minor_indices: &Vec<usize>,
) -> bool {
    let first_offset_ok = *major_offsets.first().unwrap() == 0;
    let last_offset_ok = *major_offsets.last().unwrap() == minor_indices.len();
    return first_offset_ok && last_offset_ok;
}

/// permutes entries of in_slice according to permutation slice and puts them to out_slice
#[inline]
pub fn apply_permutation<T: Clone>(out_slice: &mut [T], in_slice: &[T], permutation: &[usize]) {
    assert_eq!(out_slice.len(), in_slice.len());
    assert_eq!(out_slice.len(), permutation.len());
    for (out_element, old_pos) in out_slice.iter_mut().zip(permutation) {
        *out_element = in_slice[*old_pos].clone();
    }
}
