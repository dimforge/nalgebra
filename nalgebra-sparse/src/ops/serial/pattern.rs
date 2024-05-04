use crate::pattern::SparsityPattern;

use std::iter;

/// Sparse matrix addition pattern construction, `C <- A + B`.
///
/// Builds the pattern for `C`, which is able to hold the result of the sum `A + B`.
/// The patterns are assumed to have the same major and minor dimensions. In other words,
/// both patterns `A` and `B` must both stem from the same kind of compressed matrix:
/// CSR or CSC.
///
/// # Panics
///
/// Panics if the patterns do not have the same major and minor dimensions.
pub fn spadd_pattern(a: &SparsityPattern, b: &SparsityPattern) -> SparsityPattern {
    assert_eq!(
        a.major_dim(),
        b.major_dim(),
        "Patterns must have identical major dimensions."
    );
    assert_eq!(
        a.minor_dim(),
        b.minor_dim(),
        "Patterns must have identical minor dimensions."
    );

    let mut offsets = Vec::new();
    let mut indices = Vec::new();
    offsets.reserve(a.major_dim() + 1);
    indices.clear();

    offsets.push(0);

    for lane_idx in 0..a.major_dim() {
        let lane_a = a.lane(lane_idx);
        let lane_b = b.lane(lane_idx);
        indices.extend(iterate_union(lane_a, lane_b));
        offsets.push(indices.len());
    }

    // TODO: Consider circumventing format checks? (requires unsafe, should benchmark first)
    SparsityPattern::try_from_offsets_and_indices(a.major_dim(), a.minor_dim(), offsets, indices)
        .expect("Internal error: Pattern must be valid by definition")
}

/// Sparse matrix multiplication pattern construction, `C <- A * B`.
///
/// Assumes that the sparsity patterns both represent CSC matrices, and the result is also
/// represented as the sparsity pattern of a CSC matrix.
///
/// # Panics
///
/// Panics if the patterns, when interpreted as CSC patterns, are not compatible for
/// matrix multiplication.
pub fn spmm_csc_pattern(a: &SparsityPattern, b: &SparsityPattern) -> SparsityPattern {
    // Let C = A * B in CSC format. We note that
    //  C^T = B^T * A^T.
    // Since the interpretation of a CSC matrix in CSR format represents the transpose of the
    // matrix in CSR, we can compute C^T in *CSR format* by switching the order of a and b,
    // which lets us obtain C^T in CSR format. Re-interpreting this as CSC gives us C in CSC format
    spmm_csr_pattern(b, a)
}

/// Sparse matrix multiplication pattern construction, `C <- A * B`.
///
/// Assumes that the sparsity patterns both represent CSR matrices, and the result is also
/// represented as the sparsity pattern of a CSR matrix.
///
/// # Panics
///
/// Panics if the patterns, when interpreted as CSR patterns, are not compatible for
/// matrix multiplication.
pub fn spmm_csr_pattern(a: &SparsityPattern, b: &SparsityPattern) -> SparsityPattern {
    assert_eq!(
        a.minor_dim(),
        b.major_dim(),
        "a and b must have compatible dimensions"
    );

    let mut offsets = Vec::new();
    let mut indices = Vec::new();
    offsets.push(0);

    // Keep a vector of whether we have visited a particular minor index when working
    // on a major lane
    // TODO: Consider using a bitvec or similar here to reduce pressure on memory
    // (would cut memory use to 1/8, which might help reduce cache misses)
    let mut visited = vec![false; b.minor_dim()];

    for i in 0..a.major_dim() {
        let a_lane_i = a.lane(i);
        let c_lane_i_offset = *offsets.last().unwrap();
        for &k in a_lane_i {
            let b_lane_k = b.lane(k);

            for &j in b_lane_k {
                let have_visited_j = &mut visited[j];
                if !*have_visited_j {
                    indices.push(j);
                    *have_visited_j = true;
                }
            }
        }

        let c_lane_i = &mut indices[c_lane_i_offset..];
        c_lane_i.sort_unstable();

        // Reset visits so that visited[j] == false for all j for the next major lane
        for j in c_lane_i {
            visited[*j] = false;
        }

        offsets.push(indices.len());
    }

    SparsityPattern::try_from_offsets_and_indices(a.major_dim(), b.minor_dim(), offsets, indices)
        .expect("Internal error: Invalid pattern during matrix multiplication pattern construction")
}

/// Iterate over the union of the two sets represented by sorted slices
/// (with unique elements)
fn iterate_union<'a>(
    mut sorted_a: &'a [usize],
    mut sorted_b: &'a [usize],
) -> impl Iterator<Item = usize> + 'a {
    iter::from_fn(move || {
        if let (Some(a_item), Some(b_item)) = (sorted_a.first(), sorted_b.first()) {
            let item = match a_item.cmp(b_item) {
                std::cmp::Ordering::Less => {
                    sorted_a = &sorted_a[1..];
                    a_item
                }
                std::cmp::Ordering::Greater => {
                    sorted_b = &sorted_b[1..];
                    b_item
                }
                std::cmp::Ordering::Equal => {
                    // Both lists contain the same element, advance both slices to avoid
                    // duplicate entries in the result
                    sorted_a = &sorted_a[1..];
                    sorted_b = &sorted_b[1..];
                    a_item
                }
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
