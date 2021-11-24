//! Module for utility functions used in format conversions.

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

pub(crate) struct CountToOffsetIter<I>
where
    I: Iterator<Item = usize>,
{
    offset: usize,
    count_iter: I,
}

impl<I> CountToOffsetIter<I>
where
    I: Iterator<Item = usize>,
{
    pub(crate) fn new<T: IntoIterator<IntoIter = I, Item = usize>>(counts: T) -> Self {
        CountToOffsetIter {
            offset: 0,
            count_iter: counts.into_iter(),
        }
    }
}

impl<I> Iterator for CountToOffsetIter<I>
where
    I: Iterator<Item = usize>,
{
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        let next = self.count_iter.next()?;

        let current_offset = self.offset;
        self.offset += next;

        Some(current_offset)
    }
}
