//! TODO
//!
//! TODO: Clarify that this module needs proptest-support feature

use crate::coo::CooMatrix;
use proptest::prelude::*;
use proptest::collection::{vec, hash_map};
use nalgebra::Scalar;
use std::cmp::min;
use std::iter::repeat;
use proptest::sample::{Index};

/// A strategy for generating `nnz` triplets.
///
/// This strategy should generally only be used when `nnz` is close to `nrows * ncols`.
fn dense_triplet_strategy<T>(value_strategy: T,
                             nrows: usize,
                             ncols: usize,
                             nnz: usize)
                             -> impl Strategy<Value=Vec<(usize, usize, T::Value)>>
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
            vec![value_strategy.clone(); coords.len()]
                .prop_map(move |values| {
                    coords.clone().into_iter()
                        .zip(values)
                        .map(|((i, j), v)| {
                            (i, j, v)
                        })
                        .collect::<Vec<_>>()
                })
        })
}

/// A strategy for generating `nnz` triplets.
///
/// This strategy should generally only be used when `nnz << nrows * ncols`. If `nnz` is too
/// close to `nrows * ncols` it may fail due to excessive rejected samples.
fn sparse_triplet_strategy<T>(value_strategy: T,
                             nrows: usize,
                             ncols: usize,
                             nnz: usize)
                             -> impl Strategy<Value=Vec<(usize, usize, T::Value)>>
    where
        T: Strategy + Clone + 'static,
        T::Value: Scalar,
{
    // Have to handle the zero case: proptest doesn't like empty ranges (i.e. 0 .. 0)
    let row_index_strategy = if nrows > 0 { 0 .. nrows } else { 0 .. 1 };
    let col_index_strategy = if ncols > 0 { 0 .. ncols } else { 0 .. 1 };
    let coord_strategy = (row_index_strategy, col_index_strategy);
    hash_map(coord_strategy, value_strategy.clone(), nnz)
        .prop_map(|hash_map| {
            let triplets: Vec<_> = hash_map
                .into_iter()
                .map(|((i, j), v)| (i, j, v))
                .collect();
            triplets
        })
        // Although order in the hash map is unspecified, it's not necessarily *random*
        // - or, in particular, it does not necessarily sample the whole space of possible outcomes -
        // so we additionally shuffle the triplets
        .prop_shuffle()
}

/// TODO
pub fn coo_no_duplicates<T>(
    value_strategy: T,
    rows: impl Strategy<Value=usize> + 'static,
    cols: impl Strategy<Value=usize> + 'static,
    max_nonzeros: usize) -> impl Strategy<Value=CooMatrix<T::Value>>
where
    T: Strategy + Clone + 'static,
    T::Value: Scalar,
{
    (rows, cols)
        .prop_flat_map(move |(nrows, ncols)| {
            let max_nonzeros = min(max_nonzeros, nrows * ncols);
            let size_range = 0 ..= max_nonzeros;
            let value_strategy = value_strategy.clone();

            size_range.prop_flat_map(move |nnz| {
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

/// TODO
///
/// TODO: Write note on how this strategy only maintains the constraints on values
/// for each triplet, but does not consider the sum of triplets
pub fn coo_with_duplicates<T>(
                 value_strategy: T,
                 rows: impl Strategy<Value=usize> + 'static,
                 cols: impl Strategy<Value=usize> + 'static,
                 max_nonzeros: usize,
                 max_duplicates: usize)
    -> impl Strategy<Value=CooMatrix<T::Value>>
where
    T: Strategy + Clone + 'static,
    T::Value: Scalar,
{
    let coo_strategy = coo_no_duplicates(value_strategy.clone(), rows, cols, max_nonzeros);
    let duplicate_strategy = vec((any::<Index>(), value_strategy.clone()), 0 ..= max_duplicates);
    (coo_strategy, duplicate_strategy)
        .prop_flat_map(|(coo, duplicates)| {
            let mut triplets: Vec<(usize, usize, T::Value)> = coo.triplet_iter()
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