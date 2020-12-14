use crate::pattern::SparsityPattern;

use std::mem::swap;
use std::iter;

/// Sparse matrix addition pattern construction, `C <- A + B`.
///
/// Builds the pattern for `C`, which is able to hold the result of the sum `A + B`.
/// The patterns are assumed to have the same major and minor dimensions. In other words,
/// both patterns `A` and `B` must both stem from the same kind of compressed matrix:
/// CSR or CSC.
/// TODO: Explain that output pattern is only used to avoid allocations
pub fn spadd_build_pattern(pattern: &mut SparsityPattern,
                           a: &SparsityPattern,
                           b: &SparsityPattern)
{
    // TODO: Proper error messages
    assert_eq!(a.major_dim(), b.major_dim());
    assert_eq!(a.minor_dim(), b.minor_dim());

    let input_pattern = pattern;
    let mut temp_pattern = SparsityPattern::new(a.major_dim(), b.minor_dim());
    swap(input_pattern, &mut temp_pattern);
    let (mut offsets, mut indices) = temp_pattern.disassemble();

    offsets.clear();
    offsets.reserve(a.major_dim() + 1);
    indices.clear();

    offsets.push(0);

    for lane_idx in 0 .. a.major_dim() {
        let lane_a = a.lane(lane_idx);
        let lane_b = b.lane(lane_idx);
        indices.extend(iterate_union(lane_a, lane_b));
        offsets.push(indices.len());
    }

    // TODO: Consider circumventing format checks? (requires unsafe, should benchmark first)
    let mut new_pattern = SparsityPattern::try_from_offsets_and_indices(
        a.major_dim(), a.minor_dim(), offsets, indices)
        .expect("Pattern must be valid by definition");
    swap(input_pattern, &mut new_pattern);
}

/// Sparse matrix multiplication pattern construction, `C <- A * B`.
pub fn spmm_pattern(a: &SparsityPattern, b: &SparsityPattern) -> SparsityPattern {
    // TODO: Proper error message
    assert_eq!(a.minor_dim(), b.major_dim());

    let mut offsets = Vec::new();
    let mut indices = Vec::new();
    offsets.push(0);

    let mut c_lane_workspace = Vec::new();
    for i in 0 .. a.major_dim() {
        let a_lane_i = a.lane(i);
        let c_lane_i_offset = *offsets.last().unwrap();
        for &k in a_lane_i {
            // We have that the set of elements in lane i in C is given by the union of all
            // B_k, where B_k is the set of indices in lane k of B. More precisely, let C_i
            // denote the set of indices in lane i in C, and similarly for A_i and B_k. Then
            //  C_i = union B_k for all k in A_i
            // We incrementally compute C_i by incrementally computing the union of C_i with
            // B_k until we're through all k in A_i.
            let b_lane_k = b.lane(k);
            let c_lane_i = &indices[c_lane_i_offset..];
            c_lane_workspace.clear();
            c_lane_workspace.extend(iterate_union(c_lane_i, b_lane_k));
            indices.truncate(c_lane_i_offset);
            indices.append(&mut c_lane_workspace);
        }
        offsets.push(indices.len());
    }

    SparsityPattern::try_from_offsets_and_indices(a.major_dim(), b.minor_dim(), offsets, indices)
        .expect("Internal error: Invalid pattern during matrix multiplication pattern construction")
}

/// Iterate over the union of the two sets represented by sorted slices
/// (with unique elements)
fn iterate_union<'a>(mut sorted_a: &'a [usize],
                     mut sorted_b: &'a [usize]) -> impl Iterator<Item=usize> + 'a {
    iter::from_fn(move || {
        if let (Some(a_item), Some(b_item)) = (sorted_a.first(), sorted_b.first()) {
            let item = if a_item < b_item {
                sorted_a = &sorted_a[1 ..];
                a_item
            } else if b_item < a_item {
                sorted_b = &sorted_b[1 ..];
                b_item
            } else {
                // Both lists contain the same element, advance both slices to avoid
                // duplicate entries in the result
                sorted_a = &sorted_a[1 ..];
                sorted_b = &sorted_b[1 ..];
                a_item
            };
            Some(*item)
        } else if let Some(a_item) = sorted_a.first() {
            sorted_a = &sorted_a[1..];
            Some(*a_item)
        } else if let Some(b_item) = sorted_b.first() {
            sorted_b = &sorted_b[1..];
            Some(*b_item)
        } else {
            None
        }
    })
}