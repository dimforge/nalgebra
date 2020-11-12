//! TODO
//!
//! TODO: Clarify that this module needs proptest-support feature

use crate::coo::CooMatrix;
use proptest::prelude::*;
use proptest::collection::{SizeRange, vec};
use nalgebra::Scalar;

/// TODO
pub fn coo<T>(
                 value_strategy: T,
                 rows: impl Strategy<Value=usize> + 'static,
                 cols: impl Strategy<Value=usize> + 'static,
                 max_nonzeros: usize) -> BoxedStrategy<CooMatrix<T::Value>>
where
    T: Strategy + Clone + 'static,
    T::Value: Scalar,
{
    (rows, cols, (0 ..= max_nonzeros))
        .prop_flat_map(move |(nrows, ncols, nnz)| {
            // If the numbers of rows and columns are small in comparison with the
            // max nnz, it will lead to small matrices essentially always turning out to be dense.
            // To address this, we correct the nnz by computing the modulo with the
            // maximum number of non-zeros (ignoring duplicates) we can have for
            // the given dimensions.
            // This way we can still generate very sparse matrices for small matrices.
            let max_nnz = nrows * ncols;
            let nnz = if max_nnz == 0 { 0 } else { nnz % max_nnz };
            let row_index_strategy = if nrows > 0 { 0 .. nrows } else { 0 .. 1 };
            let col_index_strategy = if ncols > 0 { 0 .. ncols } else { 0 .. 1 };
            let row_indices = vec![row_index_strategy.clone(); nnz];
            let col_indices = vec![col_index_strategy.clone(); nnz];
            let values_strategy = vec![value_strategy.clone(); nnz];

            (Just(nrows), Just(ncols), row_indices, col_indices, values_strategy)
        }).prop_map(|(nrows, ncols, row_indices, col_indices, values)| {
            CooMatrix::try_from_triplets(nrows, ncols, row_indices, col_indices, values)
                .expect("We should always generate valid COO data.")
        }).boxed()
}