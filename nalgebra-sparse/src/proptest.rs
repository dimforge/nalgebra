//! Functionality for integrating `nalgebra-sparse` with `proptest`.
//!
//! **This module is only available if the `proptest-support` feature is enabled**.
//!
//! The strategies provided here are generally expected to be able to generate the entire range
//! of possible outputs given the constraints on dimensions and values. However, there are no
//! particular guarantees on the distribution of possible values.

// Contains some patched code from proptest that we can remove in the (hopefully near) future.
// See docs in file for more details.
mod proptest_patched;

use crate::{
    coo::CooMatrix,
    cs::{CscMatrix, CsrMatrix},
};
use nalgebra::{
    proptest::{matrix, DimRange},
    DMatrix, Dim, Scalar,
};
use proptest::{
    collection::{btree_set, hash_map, vec},
    prelude::*,
    sample::Index,
};
use std::{cmp::min, convert::TryFrom, fmt::Debug, iter::repeat, ops::RangeInclusive};

fn dense_row_major_coord_strategy(
    nrows: usize,
    ncols: usize,
    nnz: usize,
) -> impl Strategy<Value = Vec<(usize, usize)>> {
    assert!(nnz <= nrows * ncols);
    let mut booleans = vec![true; nnz];
    booleans.append(&mut vec![false; (nrows * ncols) - nnz]);
    // Make sure that exactly `nnz` of the booleans are true

    // TODO: We cannot use the below code because of a bug in proptest, see
    // https://github.com/AltSysrq/proptest/pull/217
    // so for now we're using a patched version of the Shuffle adapter
    // (see also docs in `proptest_patched`
    // Just(booleans)
    //     // Need to shuffle to make sure they are randomly distributed
    //     .prop_shuffle()

    proptest_patched::Shuffle(Just(booleans)).prop_map(move |booleans| {
        booleans
            .into_iter()
            .enumerate()
            .filter_map(|(index, is_entry)| {
                if is_entry {
                    // Convert linear index to row/col pair
                    let i = index / ncols;
                    let j = index % ncols;
                    Some((i, j))
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
    })
}

/// A strategy for generating `nnz` triplets.
///
/// This strategy should generally only be used when `nnz` is close to `nrows * ncols`.
fn dense_triplet_strategy<T>(
    value_strategy: T,
    nrows: usize,
    ncols: usize,
    nnz: usize,
) -> impl Strategy<Value = Vec<(usize, usize, T::Value)>>
where
    T: Strategy + Clone + 'static,
    T::Value: Scalar,
{
    assert!(nnz <= nrows * ncols);

    // Construct a number of booleans of which exactly `nnz` are true.
    let booleans: Vec<_> = repeat(true)
        .take(nnz)
        .chain(repeat(false))
        .take(nrows * ncols)
        .collect();

    Just(booleans)
        // Shuffle the booleans so that they are randomly distributed
        .prop_shuffle()
        // Convert the booleans into a list of coordinate pairs
        .prop_map(move |booleans| {
            booleans
                .into_iter()
                .enumerate()
                .filter_map(|(index, is_entry)| {
                    if is_entry {
                        // Convert linear index to row/col pair
                        let i = index / ncols;
                        let j = index % ncols;
                        Some((i, j))
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>()
        })
        // Assign values to each coordinate pair in order to generate a list of triplets
        .prop_flat_map(move |coords| {
            vec![value_strategy.clone(); coords.len()].prop_map(move |values| {
                coords
                    .clone()
                    .into_iter()
                    .zip(values)
                    .map(|((i, j), v)| (i, j, v))
                    .collect::<Vec<_>>()
            })
        })
}

/// A strategy for generating `nnz` triplets.
///
/// This strategy should generally only be used when `nnz << nrows * ncols`. If `nnz` is too
/// close to `nrows * ncols` it may fail due to excessive rejected samples.
fn sparse_triplet_strategy<T>(
    value_strategy: T,
    nrows: usize,
    ncols: usize,
    nnz: usize,
) -> impl Strategy<Value = Vec<(usize, usize, T::Value)>>
where
    T: Strategy + Clone + 'static,
    T::Value: Scalar,
{
    // Have to handle the zero case: proptest doesn't like empty ranges (i.e. 0 .. 0)
    let row_index_strategy = if nrows > 0 { 0..nrows } else { 0..1 };
    let col_index_strategy = if ncols > 0 { 0..ncols } else { 0..1 };
    let coord_strategy = (row_index_strategy, col_index_strategy);
    hash_map(coord_strategy, value_strategy.clone(), nnz)
        .prop_map(|hash_map| {
            let triplets: Vec<_> = hash_map.into_iter().map(|((i, j), v)| (i, j, v)).collect();
            triplets
        })
        // Although order in the hash map is unspecified, it's not necessarily *random*
        // - or, in particular, it does not necessarily sample the whole space of possible outcomes -
        // so we additionally shuffle the triplets
        .prop_shuffle()
}

/// A strategy for producing COO matrices without duplicate entries.
///
/// The values of the matrix are picked from the provided `value_strategy`, while the size of the
/// generated matrices is determined by the ranges `rows` and `cols`. The number of explicitly
/// stored entries is bounded from above by `max_nonzeros`. Note that the matrix might still
/// contain explicitly stored zeroes if the value strategy is capable of generating zero values.
pub fn coo_no_duplicates<T>(
    value_strategy: T,
    rows: impl Into<DimRange>,
    cols: impl Into<DimRange>,
    max_nonzeros: usize,
) -> impl Strategy<Value = CooMatrix<T::Value>>
where
    T: Strategy + Clone + 'static,
    T::Value: Scalar,
{
    (
        rows.into().to_range_inclusive(),
        cols.into().to_range_inclusive(),
    )
        .prop_flat_map(move |(nrows, ncols)| {
            let max_nonzeros = min(max_nonzeros, nrows * ncols);
            let size_range = 0..=max_nonzeros;
            let value_strategy = value_strategy.clone();

            size_range
                .prop_flat_map(move |nnz| {
                    let value_strategy = value_strategy.clone();
                    if nnz as f64 > 0.10 * (nrows as f64) * (ncols as f64) {
                        // If the number of nnz is sufficiently dense, then use the dense
                        // sample strategy
                        dense_triplet_strategy(value_strategy, nrows, ncols, nnz).boxed()
                    } else {
                        // Otherwise, use a hash map strategy so that we can get a sparse sampling
                        // (so that complexity is rather on the order of max_nnz than nrows * ncols)
                        sparse_triplet_strategy(value_strategy, nrows, ncols, nnz).boxed()
                    }
                })
                .prop_map(move |triplets| {
                    let mut coo = CooMatrix::new(nrows, ncols);
                    for (i, j, v) in triplets {
                        coo.push(i, j, v);
                    }
                    coo
                })
        })
}

/// A strategy for producing COO matrices with duplicate entries.
///
/// The values of the matrix are picked from the provided `value_strategy`, while the size of the
/// generated matrices is determined by the ranges `rows` and `cols`. Note that the values
/// only apply to individual entries, and since this strategy can generate duplicate entries,
/// the matrix will generally have values outside the range determined by `value_strategy` when
/// converted to other formats, since the duplicate entries are summed together in this case.
///
/// The number of explicitly stored entries is bounded from above by `max_nonzeros`. The maximum
/// number of duplicate entries is determined by `max_duplicates`. Note that the matrix might still
/// contain explicitly stored zeroes if the value strategy is capable of generating zero values.
pub fn coo_with_duplicates<T>(
    value_strategy: T,
    rows: impl Into<DimRange>,
    cols: impl Into<DimRange>,
    max_nonzeros: usize,
    max_duplicates: usize,
) -> impl Strategy<Value = CooMatrix<T::Value>>
where
    T: Strategy + Clone + 'static,
    T::Value: Scalar,
{
    let coo_strategy = coo_no_duplicates(value_strategy.clone(), rows, cols, max_nonzeros);
    let duplicate_strategy = vec((any::<Index>(), value_strategy.clone()), 0..=max_duplicates);
    (coo_strategy, duplicate_strategy)
        .prop_flat_map(|(coo, duplicates)| {
            let mut triplets: Vec<(usize, usize, T::Value)> = coo
                .triplet_iter()
                .map(|(i, j, v)| (i, j, v.clone()))
                .collect();
            if !triplets.is_empty() {
                let duplicates_iter: Vec<_> = duplicates
                    .into_iter()
                    .map(|(idx, val)| {
                        let (i, j, _) = idx.get(&triplets);
                        (*i, *j, val)
                    })
                    .collect();
                triplets.extend(duplicates_iter);
            }
            // Make sure to shuffle so that the duplicates get mixed in with the non-duplicates
            let shuffled = Just(triplets).prop_shuffle();
            (Just(coo.nrows()), Just(coo.ncols()), shuffled)
        })
        .prop_map(move |(nrows, ncols, triplets)| {
            let mut coo = CooMatrix::new(nrows, ncols);
            for (i, j, v) in triplets {
                coo.push(i, j, v);
            }
            coo
        })
}

fn sparsity_pattern_from_row_major_coords<I>(
    nmajor: usize,
    nminor: usize,
    coords: I,
) -> ((usize, usize), Vec<usize>, Vec<usize>)
where
    I: Iterator<Item = (usize, usize)> + ExactSizeIterator,
{
    let mut minors = Vec::with_capacity(coords.len());
    let mut counts = vec![0usize; nmajor];

    for (i, j) in coords {
        assert!(
            i < nmajor && j < nminor,
            "Generated coords are out of bounds"
        );

        counts[i] += 1;
        minors.push(j);
    }

    let (_, offsets) =
        counts
            .into_iter()
            .fold((0, Vec::with_capacity(nmajor)), |(total, mut v), count| {
                v.push(total);
                (total + count, v)
            });

    let shape = (nmajor, nminor);

    (shape, offsets, minors)
}

/// A strategy for generating sparsity patterns.
pub fn sparsity_pattern(
    major_lanes: impl Into<DimRange>,
    minor_lanes: impl Into<DimRange>,
    max_nonzeros: usize,
) -> impl Strategy<Value = ((usize, usize), Vec<usize>, Vec<usize>)> {
    (
        major_lanes.into().to_range_inclusive(),
        minor_lanes.into().to_range_inclusive(),
    )
        .prop_flat_map(move |(nmajor, nminor)| {
            let max_nonzeros = min(nmajor * nminor, max_nonzeros);
            (Just(nmajor), Just(nminor), 0..=max_nonzeros)
        })
        .prop_flat_map(move |(nmajor, nminor, nnz)| {
            if 10 * nnz < nmajor * nminor {
                // If nnz is small compared to a dense matrix, then use a sparse sampling strategy
                btree_set((0..nmajor, 0..nminor), nnz)
                    .prop_map(move |coords| {
                        sparsity_pattern_from_row_major_coords(nmajor, nminor, coords.into_iter())
                    })
                    .boxed()
            } else {
                // If the required number of nonzeros is sufficiently dense,
                // we instead use a dense sampling
                dense_row_major_coord_strategy(nmajor, nminor, nnz)
                    .prop_map(move |coords| {
                        sparsity_pattern_from_row_major_coords(nmajor, nminor, coords.into_iter())
                    })
                    .boxed()
            }
        })
}

/// A strategy for generating CSR matrices.
pub fn csr<T>(
    value_strategy: T,
    rows: impl Into<DimRange>,
    cols: impl Into<DimRange>,
    max_nonzeros: usize,
) -> impl Strategy<Value = CsrMatrix<T::Value>>
where
    T: Strategy + Clone + 'static,
    T::Value: Scalar,
{
    let rows = rows.into();
    let cols = cols.into();
    sparsity_pattern(
        rows.lower_bound().value()..=rows.upper_bound().value(),
        cols.lower_bound().value()..=cols.upper_bound().value(),
        max_nonzeros,
    )
    .prop_flat_map(move |(shape, offsets, indices)| {
        let nnz = indices.len();
        let data = vec![value_strategy.clone(); nnz];
        (Just(shape), Just(offsets), Just(indices), data)
    })
    .prop_map(|((nmajor, nminor), offsets, indices, data)| {
        CsrMatrix::try_from_parts(nmajor, nminor, offsets, indices, data)
            .expect("Internal error: Generated CsrMatrix is invalid")
    })
}

/// A strategy for generating CSC matrices.
pub fn csc<T>(
    value_strategy: T,
    rows: impl Into<DimRange>,
    cols: impl Into<DimRange>,
    max_nonzeros: usize,
) -> impl Strategy<Value = CscMatrix<T::Value>>
where
    T: Strategy + Clone + 'static,
    T::Value: Scalar,
{
    let rows = rows.into();
    let cols = cols.into();
    sparsity_pattern(
        cols.lower_bound().value()..=cols.upper_bound().value(),
        rows.lower_bound().value()..=rows.upper_bound().value(),
        max_nonzeros,
    )
    .prop_flat_map(move |(shape, offsets, indices)| {
        let nnz = indices.len();
        let data = vec![value_strategy.clone(); nnz];

        (Just(shape), Just(offsets), Just(indices), data)
    })
    .prop_map(|((nmajor, nminor), offsets, indices, data)| {
        CscMatrix::try_from_parts(nminor, nmajor, offsets, indices, data)
            .expect("Internal error: Generated CscMatrix is invalid")
    })
}

/// Range of acceptable matrix dimensions
pub const PROPTEST_MATRIX_DIM: RangeInclusive<usize> = 0..=6;

/// Max number of non-zero elements in a sparse matrix.
pub const PROPTEST_MAX_NNZ: usize = 40;

/// Value strategy for generating explicit values in the sparse matrix.
pub const PROPTEST_I32_VALUE_STRATEGY: RangeInclusive<i32> = -5..=5;

/// Value strategy adapter that maps to a range of values from PROPTEST_I32_VALUE_STRATEGY. This
/// only works if the type `T` can be converted into from an `i32`.
pub fn value_strategy<T>() -> RangeInclusive<T>
where
    T: TryFrom<i32>,
    T::Error: Debug,
{
    let (start, end) = (
        PROPTEST_I32_VALUE_STRATEGY.start(),
        PROPTEST_I32_VALUE_STRATEGY.end(),
    );
    T::try_from(*start).unwrap()..=T::try_from(*end).unwrap()
}

/// A strategy for producing values for a sparse matrix, without including explicit zeros.
pub fn non_zero_i32_value_strategy() -> impl Strategy<Value = i32> {
    let (start, end) = (
        PROPTEST_I32_VALUE_STRATEGY.start(),
        PROPTEST_I32_VALUE_STRATEGY.end(),
    );
    assert!(start < &0);
    assert!(end > &0);
    // Note: we don't use RangeInclusive for the second range, because then we'd have different
    // types, which would require boxing
    (*start..0).prop_union(1..*end + 1)
}

/// Produces a random CSR matrix, which may contain explicit zeros or be zero sized.
pub fn csr_strategy() -> impl Strategy<Value = CsrMatrix<i32>> {
    csr(
        PROPTEST_I32_VALUE_STRATEGY,
        PROPTEST_MATRIX_DIM,
        PROPTEST_MATRIX_DIM,
        PROPTEST_MAX_NNZ,
    )
}

/// Produces a random CSC matrix, which may contain explicit zeros or be zero sized.
pub fn csc_strategy() -> impl Strategy<Value = CscMatrix<i32>> {
    csc(
        PROPTEST_I32_VALUE_STRATEGY,
        PROPTEST_MATRIX_DIM,
        PROPTEST_MATRIX_DIM,
        PROPTEST_MAX_NNZ,
    )
}

/// Produces a random COO matrix, which may contain explicit zeros, be zero sized, or may contain
/// duplicate entries.
pub fn coo_strategy() -> impl Strategy<Value = CooMatrix<i32>> {
    coo_with_duplicates(
        PROPTEST_I32_VALUE_STRATEGY,
        PROPTEST_MATRIX_DIM,
        PROPTEST_MATRIX_DIM,
        PROPTEST_MAX_NNZ,
        2,
    )
}

/// Strategy for producing COO matrices without duplicate entries.
pub fn coo_no_duplicates_strategy() -> impl Strategy<Value = CooMatrix<i32>> {
    coo_no_duplicates(
        PROPTEST_I32_VALUE_STRATEGY,
        PROPTEST_MATRIX_DIM,
        PROPTEST_MATRIX_DIM,
        PROPTEST_MAX_NNZ,
    )
}

/// Avoid generating explicit zero values so that it is possible to reason about sparsity patterns
pub fn non_zero_csr_strategy() -> impl Strategy<Value = CsrMatrix<i32>> {
    csr(
        1..=5,
        PROPTEST_MATRIX_DIM,
        PROPTEST_MATRIX_DIM,
        PROPTEST_MAX_NNZ,
    )
}

/// Avoid generating explicit zero values so that it is possible to reason about sparsity patterns
pub fn non_zero_csc_strategy() -> impl Strategy<Value = CscMatrix<i32>> {
    csc(
        1..=5,
        PROPTEST_MATRIX_DIM,
        PROPTEST_MATRIX_DIM,
        PROPTEST_MAX_NNZ,
    )
}

/// A strategy for generating random dynamically sized matrices.
pub fn dense_strategy() -> impl Strategy<Value = DMatrix<i32>> {
    matrix(
        PROPTEST_I32_VALUE_STRATEGY,
        PROPTEST_MATRIX_DIM,
        PROPTEST_MATRIX_DIM,
    )
}

/// Produces a positive definite CSR matrix
pub fn csr_positive_definite() -> impl Strategy<Value = CsrMatrix<f64>> {
    csc(1.0..=5.0, 1..=10, 1..=10, PROPTEST_MAX_NNZ).prop_map(|x| {
        // Add a small multiple of the identity to ensure positive definiteness
        x.transpose() * x.to_view() + CscMatrix::<f64>::identity(x.ncols())
    })
}

/// Produces a positive definite CSC matrix
pub fn csc_positive_definite() -> impl Strategy<Value = CscMatrix<f64>> {
    csr_positive_definite().prop_map(|csr| csr.transpose_owned())
}
