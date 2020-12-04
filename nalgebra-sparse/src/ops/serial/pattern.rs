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
        indices.extend(iterate_intersection(lane_a, lane_b));
        offsets.push(indices.len());
    }

    // TODO: Consider circumventing format checks? (requires unsafe, should benchmark first)
    let mut new_pattern = SparsityPattern::try_from_offsets_and_indices(
        a.major_dim(), a.minor_dim(), offsets, indices)
        .expect("Pattern must be valid by definition");
    swap(input_pattern, &mut new_pattern);
}

/// Iterate over the intersection of the two sets represented by sorted slices
/// (with unique elements)
fn iterate_intersection<'a>(mut sorted_a: &'a [usize],
                            mut sorted_b: &'a [usize]) -> impl Iterator<Item=usize> + 'a {
    // TODO: Can use a kind of simultaneous exponential search to speed things up here
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