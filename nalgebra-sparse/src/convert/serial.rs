//! Serial routines for converting between matrix formats.
//!
//! All routines in this module are single-threaded. At present these routines offer no
//! advantage over using the [`From`] trait, but future changes to the API might offer more
//! control to the user.
use super::utils;
use crate::{
    coo::CooMatrix,
    cs::{CompressedColumnStorage, CompressedRowStorage, CsMatrix, CscMatrix, CsrMatrix},
};
use nalgebra::storage::RawStorage;
use nalgebra::{ClosedAdd, DMatrix, Dim, Matrix, Scalar};
use num_traits::{Unsigned, Zero};
use std::{borrow::Borrow, ops::Add};

/// Converts a dense matrix to [`CooMatrix`].
pub fn convert_dense_coo<T, R, C, S>(dense: &Matrix<T, R, C, S>) -> CooMatrix<T>
where
    T: Scalar + Zero,
    R: Dim,
    C: Dim,
    S: RawStorage<T, R, C>,
{
    let mut coo = CooMatrix::new(dense.nrows(), dense.ncols());

    for (index, v) in dense.iter().enumerate() {
        if v != &T::zero() {
            // We use the fact that matrix iteration is guaranteed to be column-major
            let i = index % dense.nrows();
            let j = index / dense.nrows();
            coo.push(i, j, v.clone());
        }
    }

    coo
}

/// Converts a [`CooMatrix`] to a dense matrix.
pub fn convert_coo_dense<T>(coo: &CooMatrix<T>) -> DMatrix<T>
where
    T: Scalar + Zero + ClosedAdd,
{
    let mut output = DMatrix::<T>::zeros(coo.nrows(), coo.ncols());
    for (i, j, v) in coo.triplet_iter() {
        output[(i, j)] += v.clone();
    }
    output
}

/// Converts a [`CooMatrix`] to a [`CsrMatrix`].
pub fn convert_coo_csr<T>(coo: &CooMatrix<T>) -> CsrMatrix<T, usize>
where
    T: Scalar + Zero,
{
    let (offsets, indices, values) = convert_coo_cs(
        coo.nrows(),
        coo.row_indices(),
        coo.col_indices(),
        coo.values(),
    );

    unsafe { CsrMatrix::from_parts_unchecked(coo.nrows(), coo.ncols(), offsets, indices, values) }
}

/// Converts a [`CsrMatrix`] to a [`CooMatrix`].
pub fn convert_csr_coo<T, O, MO, MI, D, I>(
    csr: &CsMatrix<T, O, MO, MI, D, CompressedRowStorage, I>,
) -> CooMatrix<T>
where
    T: Clone,
    O: Add<usize, Output = usize> + Copy + Clone + Into<usize> + Unsigned + Ord,
    I: Copy + Clone + Into<usize> + Unsigned + Ord,
    MO: Borrow<[O]>,
    MI: Borrow<[I]>,
    D: Borrow<[T]>,
{
    let mut result = CooMatrix::new(csr.nrows(), csr.ncols());
    for (i, j, v) in csr.triplet_iter() {
        result.push(i, j, v.clone());
    }
    result
}

/// Converts a [`CsrMatrix`] to a dense matrix.
pub fn convert_csr_dense<T, O, MO, MI, D, I>(
    csr: &CsMatrix<T, O, MO, MI, D, CompressedRowStorage, I>,
) -> DMatrix<T>
where
    T: Scalar + ClosedAdd + Zero,
    O: Add<usize, Output = usize> + Copy + Clone + Into<usize> + Unsigned + Ord,
    I: Copy + Clone + Into<usize> + Unsigned + Ord,
    MO: Borrow<[O]>,
    MI: Borrow<[I]>,
    D: Borrow<[T]>,
{
    let mut output = DMatrix::zeros(csr.nrows(), csr.ncols());

    for (i, j, v) in csr.triplet_iter() {
        output[(i, j)] += v.clone();
    }

    output
}

/// Converts a dense matrix to a [`CsrMatrix`].
pub fn convert_dense_csr<T, R, C, S>(dense: &Matrix<T, R, C, S>) -> CsrMatrix<T, usize>
where
    T: Scalar + Zero,
    R: Dim,
    C: Dim,
    S: RawStorage<T, R, C>,
{
    let mut row_offsets = Vec::with_capacity(dense.nrows() + 1);
    let mut col_idx = Vec::new();
    let mut values = Vec::new();

    // We have to iterate row-by-row to build the CSR matrix, which is at odds with
    // nalgebra's column-major storage. The alternative would be to perform an initial sweep
    // to count number of non-zeros per row.
    row_offsets.push(0);
    for i in 0..dense.nrows() {
        for j in 0..dense.ncols() {
            let v = dense.index((i, j));
            if v != &T::zero() {
                col_idx.push(j);
                values.push(v.clone());
            }
        }
        row_offsets.push(col_idx.len());
    }

    unsafe {
        CsrMatrix::from_parts_unchecked(dense.nrows(), dense.ncols(), row_offsets, col_idx, values)
    }
}

/// Converts a [`CooMatrix`] to a [`CscMatrix`].
pub fn convert_coo_csc<T>(coo: &CooMatrix<T>) -> CscMatrix<T, usize>
where
    T: Scalar + Zero,
{
    let (offsets, indices, values) = convert_coo_cs(
        coo.ncols(),
        coo.col_indices(),
        coo.row_indices(),
        coo.values(),
    );

    unsafe { CscMatrix::from_parts_unchecked(coo.nrows(), coo.ncols(), offsets, indices, values) }
}

/// Converts a [`CscMatrix`] to a [`CooMatrix`].
pub fn convert_csc_coo<T, O, MO, MI, D, I>(
    csc: &CsMatrix<T, O, MO, MI, D, CompressedColumnStorage, I>,
) -> CooMatrix<T>
where
    T: Scalar,
    O: Add<usize, Output = usize> + Copy + Clone + Into<usize> + Unsigned + Ord,
    I: Copy + Clone + Into<usize> + Unsigned + Ord,
    MO: Borrow<[O]>,
    MI: Borrow<[I]>,
    D: Borrow<[T]>,
{
    let mut coo = CooMatrix::new(csc.nrows(), csc.ncols());
    for (i, j, v) in csc.triplet_iter() {
        coo.push(j, i, v.clone());
    }
    coo
}

/// Converts a [`CscMatrix`] to a dense matrix.
pub fn convert_csc_dense<T, O, MO, MI, D, I>(
    csc: &CsMatrix<T, O, MO, MI, D, CompressedColumnStorage, I>,
) -> DMatrix<T>
where
    T: Scalar + ClosedAdd + Zero,
    O: Add<usize, Output = usize> + Copy + Clone + Into<usize> + Unsigned + Ord,
    I: Copy + Clone + Into<usize> + Unsigned + Ord,
    MO: Borrow<[O]>,
    MI: Borrow<[I]>,
    D: Borrow<[T]>,
{
    let mut output = DMatrix::zeros(csc.nrows(), csc.ncols());

    for (i, j, v) in csc.triplet_iter() {
        output[(j, i)] += v.clone();
    }

    output
}

/// Converts a dense matrix to a [`CscMatrix`].
pub fn convert_dense_csc<T, R, C, S>(dense: &Matrix<T, R, C, S>) -> CscMatrix<T, usize>
where
    T: Scalar + Zero,
    R: Dim,
    C: Dim,
    S: RawStorage<T, R, C>,
{
    let mut col_offsets = Vec::with_capacity(dense.ncols() + 1);
    let mut row_idx = Vec::new();
    let mut values = Vec::new();

    col_offsets.push(0);
    for j in 0..dense.ncols() {
        for i in 0..dense.nrows() {
            let v = dense.index((i, j));
            if v != &T::zero() {
                row_idx.push(i);
                values.push(v.clone());
            }
        }
        col_offsets.push(row_idx.len());
    }

    unsafe {
        CscMatrix::from_parts_unchecked(dense.nrows(), dense.ncols(), col_offsets, row_idx, values)
    }
}

/// Converts a [`CsrMatrix`] to a [`CscMatrix`].
pub fn convert_csr_csc<T, O, MO, MI, D, I>(
    csr: &CsMatrix<T, O, MO, MI, D, CompressedRowStorage, I>,
) -> CscMatrix<T, usize>
where
    T: Clone,
    O: Add<usize, Output = usize> + Copy + Clone + Into<usize> + Unsigned + Ord,
    I: Copy + Clone + Into<usize> + Unsigned + Ord,
    MO: Borrow<[O]>,
    MI: Borrow<[I]>,
    D: Borrow<[T]>,
{
    let (offsets, indices, values) = csr.cs_data();

    let (offsets, indices, values) =
        utils::transpose_convert(csr.nrows(), csr.ncols(), offsets, indices, values);

    unsafe { CscMatrix::from_parts_unchecked(csr.nrows(), csr.ncols(), offsets, indices, values) }
}

/// Converts a [`CscMatrix`] to a [`CsrMatrix`].
pub fn convert_csc_csr<T, O, MO, MI, D, I>(
    csc: &CsMatrix<T, O, MO, MI, D, CompressedColumnStorage, I>,
) -> CsrMatrix<T, usize>
where
    T: Clone,
    O: Add<usize, Output = usize> + Copy + Clone + Into<usize> + Unsigned + Ord,
    I: Copy + Clone + Into<usize> + Unsigned + Ord,
    MO: Borrow<[O]>,
    MI: Borrow<[I]>,
    D: Borrow<[T]>,
{
    let (offsets, indices, values) = csc.cs_data();

    let (offsets, indices, values) =
        utils::transpose_convert(csc.ncols(), csc.nrows(), offsets, indices, values);

    unsafe { CsrMatrix::from_parts_unchecked(csc.nrows(), csc.ncols(), offsets, indices, values) }
}

fn convert_coo_cs<T>(
    major_dim: usize,
    major_indices: &[usize],
    minor_indices: &[usize],
    values: &[T],
) -> (Vec<usize>, Vec<usize>, Vec<T>)
where
    T: Scalar + Zero,
{
    assert_eq!(major_indices.len(), minor_indices.len());
    assert_eq!(minor_indices.len(), values.len());
    let nnz = major_indices.len();

    let (unsorted_major_offsets, unsorted_minor_idx, unsorted_vals) = {
        let mut offsets = vec![0usize; major_dim + 1];
        let mut minor_idx = vec![0usize; nnz];
        let mut vals = vec![T::zero(); nnz];
        coo_to_unsorted_cs(
            &mut offsets,
            &mut minor_idx,
            &mut vals,
            major_dim,
            major_indices,
            minor_indices,
            values,
        );
        (offsets, minor_idx, vals)
    };

    // TODO: If input is sorted and/or without duplicates, we can avoid additional allocations
    // and work. Might want to take advantage of this.

    // At this point, assembly is essentially complete. However, we must ensure
    // that minor indices are sorted within each lane and without duplicates.
    let mut sorted_major_offsets = Vec::new();
    let mut sorted_minor_idx = Vec::new();
    let mut sorted_vals = Vec::new();

    sorted_major_offsets.push(0);

    // We need some temporary storage when working with each lane. Since lanes often have a
    // very small number of non-zero entries, we try to amortize allocations across
    // lanes by reusing workspace vectors
    let mut idx_workspace = Vec::new();
    let mut perm_workspace = Vec::new();
    let mut values_workspace = Vec::new();

    for lane in 0..major_dim {
        let begin = unsorted_major_offsets[lane];
        let end = unsorted_major_offsets[lane + 1];
        let count = end - begin;
        let range = begin..end;

        // Ensure that workspaces can hold enough data
        perm_workspace.resize(count, 0);
        idx_workspace.resize(count, 0);
        values_workspace.resize(count, T::zero());
        utils::sort_lane(
            &mut idx_workspace[..count],
            &mut values_workspace[..count],
            &unsorted_minor_idx[range.clone()],
            &unsorted_vals[range.clone()],
            &mut perm_workspace[..count],
        );

        let sorted_ja_current_len = sorted_minor_idx.len();

        utils::combine_duplicates(
            |idx| sorted_minor_idx.push(idx),
            |val| sorted_vals.push(val),
            &idx_workspace[..count],
            &values_workspace[..count],
            &Add::add,
        );

        let new_col_count = sorted_minor_idx.len() - sorted_ja_current_len;
        sorted_major_offsets.push(sorted_major_offsets.last().unwrap() + new_col_count);
    }

    (sorted_major_offsets, sorted_minor_idx, sorted_vals)
}

/// Converts matrix data given in triplet format to unsorted CSR/CSC, retaining any duplicated
/// indices.
///
/// Here `major/minor` is `row/col` for CSR and `col/row` for CSC.
fn coo_to_unsorted_cs<T: Clone>(
    major_offsets: &mut [usize],
    cs_minor_idx: &mut [usize],
    cs_values: &mut [T],
    major_dim: usize,
    major_indices: &[usize],
    minor_indices: &[usize],
    coo_values: &[T],
) {
    assert_eq!(major_offsets.len(), major_dim + 1);
    assert_eq!(cs_minor_idx.len(), cs_values.len());
    assert_eq!(cs_values.len(), major_indices.len());
    assert_eq!(major_indices.len(), minor_indices.len());
    assert_eq!(minor_indices.len(), coo_values.len());

    // Count the number of occurrences of each row
    for major_idx in major_indices {
        major_offsets[*major_idx] += 1;
    }

    let major_offsets =
        utils::CountToOffsetIter::new(major_offsets.iter().map(|&x| x)).collect::<Vec<_>>();

    {
        // TODO: Instead of allocating a whole new vector storing the current counts,
        // I think it's possible to be a bit more clever by storing each count
        // in the last of the column indices for each row
        let mut current_counts = vec![0usize; major_dim + 1];
        let triplet_iter = major_indices.iter().zip(minor_indices).zip(coo_values);
        for ((i, j), value) in triplet_iter {
            let current_offset = major_offsets[*i] + current_counts[*i];
            cs_minor_idx[current_offset] = *j;
            cs_values[current_offset] = value.clone();
            current_counts[*i] += 1;
        }
    }
}
