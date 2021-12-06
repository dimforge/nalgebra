//! Module holding the various sparse-matrix multiply functions.
//!
//! Sparse-matrix multiplication performance is something that can depend heavily on the underlying
//! data types. In particular, CSR and CSC matrices heavily lean towards optimized access along the
//! major axis of the format (i.e. rows in CSR and columns in CSC).
//!
//! # Performance
//!
//! Because of this, the contained `spmm_x_y` functions may all have different performance
//! characteristics. Where possible, we have attempted to leverage known aspects of the data
//! formats or made effective use of caching intermediate lane data in order to provide the best
//! possible performance. When writing code that necessitates high-performance, once should take
//! care to remember that certain operations are faster than others.
//!
//! In general, one should prefer to pose their problems as a combination of one of the following
//! functions:
//!
//! - CSR times CSC
//! - Dense times CSC
//! - CSR times Dense
//!
//! An example may perhaps be illustrative. Assuming that we have three matrices, `a`, `b`, and
//! `c`, let's look at various ways we might perform the operation `a * b * c`.
//!
//! For the first case, let's assume `a` is CSR, and `b` and `c` are both CSC format. In such a
//! scenario, the following code produces the optimal matrix multiplication:
//!
//! ```rust
//! use nalgebra_sparse::{CsrMatrix, CscMatrix, serial::{spmm_csr_csc}};
//!
//! # fn spmm(a: CsrMatrix<f32, usize>, b: CscMatrix<f32, usize>, c: CscMatrix<f32, usize>) {
//! let product = spmm_csr_csc(spmm_csr_csc(a, b), c);
//!
//! // SLOWER!!!
//! //
//! // let product = spmm_csr_csc(a, spmm_csc_csc(b, c));
//! # }
//! ```
//!
//! By chaining this together with the correct types, we can guarantee that the fastest
//! matrix-multiply operation is used for both products. Let's assume instead that `a` is still
//! CSR, `b` is CSC, and `c` is CSR. This multiplication is tricky because we might accidentally
//! use `spmm_csc_csr`, which is the slowest possible product. Unfortunately we cannot express the
//! entire problem in terms of `spmm_csr_csc` without making a potentially expensive conversion on
//! `c`. So instead, we can formulate the example as:
//!
//! ```rust
//! use nalgebra_sparse::{CsrMatrix, CscMatrix, serial::{spmm_csr_csc}};
//!
//! # fn spmm(a: CsrMatrix<f32, usize>, b: CscMatrix<f32, usize>, c: CscMatrix<f32, usize>) {
//! let product = spmm_csr_csr(spmm_csr_csc(a, b), c);
//!
//! // Avoid this as much as possible, it is the slowest possible form!
//! //
//! // let product = spmm_csr_csr(a, spmm_csc_csr(b, c));
//! # }
//! ```
//!
//! `spmm_csr_csr` is not as fast as `spmm_csr_csc`, because we have to access the second matrix
//! out-of-order relative to its lane ordering. However, given the format of the problem we make
//! the best of the situation and avoid an expensive reconfiguration / re-allocation for the data
//! in `c`.
//!
//! NOTE: These interfaces are primarily meant for advanced users who wish to get the most
//! performance out of their sparse operations. If you're not benchmarking an issue where
//! `spmm_csc_csr` is slowing down your whole pipeline, this may not be worth investing your time
//! in.
//!
//! # Sparse-Dense products
//!
//! Here we have opted to output sparse matrix types when performing sparse-dense matrix
//! multiplications. This is driven from the fact that for any two matrices, the product of those
//! matrices will always have fewer or equal number of non-zero elements. Put in terms of
//! mathematics:
//!
//! ```text
//! C := A * B
//!
//! nnz(C) <= min(nnz(A), nnz(B))
//! ```
//!
//! The direct question about this choice is: when is sparse output from a matrix multiplication
//! the wrong choice? From there, we would ask ourselves, when is sparse representation the wrong
//! choice? And that is something that is largely application dependent! For some applications, 50%
//! density may be too dense to be worth the overhead of the compressed format. On the other hand,
//! others might be comfortable with much higher number of non-zeros in their matrix, since they
//! are using the sparse format primarily for numerical stability (and not successively adding or
//! multiplying by small, almost zero values).
//!
//! The thinking then goes like this: since we know that if either `A` or `B` in the above example
//! are sparse, then another matrix `C` with exactly the same number of non-zero elements as `B`
//! should also be sparse (according to whatever logic / decision structure led to users making `B`
//! sparse). In this way, we avoid making decisions based on heuristics and simply follow what
//! should be natural for the API.
//!
//! Note that at any time one can convert between any of the supported formats. For more info, see
//! the [`convert`](crate::convert) module.

use crate::{
    convert::utils::CountToOffsetIter,
    cs::{CompressedColumnStorage, CompressedRowStorage, CsMatrix, CscMatrix, CsrMatrix},
    error::{OperationError, OperationErrorKind},
};
use nalgebra::{Dim, Matrix, RawStorage, Scalar};
use num_traits::Zero;
use std::{
    borrow::Borrow,
    cmp::Ordering,
    ops::{Add, AddAssign, Mul},
};

/// The fundamental (fastest) sparse-matrix multiply.
///
/// This function takes two arguments, a CSR matrix and a CSC matrix, and performs a sparse-matrix
/// multiplication with the CSR matrix on the left side and the CSC matrix on the right.
///
/// Because of the way that lane access works on the underlying data structures, this is the
/// theoretical fastest matrix product.
///
/// # Errors
///
/// This function fails and produces an [`OperationError`] with kind
/// [`OperationErrorKind::InvalidPattern`] if the two matrices have incompatible shapes for a
/// matrix product.
pub fn spmm_csr_csc<T1, T2, MO1, MO2, MI1, MI2, D1, D2>(
    csr: CsMatrix<T1, MO1, MI1, D1, CompressedRowStorage>,
    csc: CsMatrix<T2, MO2, MI2, D2, CompressedColumnStorage>,
) -> Result<CsrMatrix<<T1 as Mul<T2>>::Output>, OperationError>
where
    T1: Clone + Mul<T2>,
    <T1 as Mul<T2>>::Output: AddAssign + Zero,
    T2: Clone,
    MO1: Borrow<[usize]>,
    MO2: Borrow<[usize]>,
    MI1: Borrow<[usize]>,
    MI2: Borrow<[usize]>,
    D1: Borrow<[T1]>,
    D2: Borrow<[T2]>,
{
    let (rows, lc) = csr.shape();
    let (rr, columns) = csc.shape();

    if lc != rr {
        return Err(OperationError::from_kind_and_message(
            OperationErrorKind::InvalidPattern,
            String::from(
                "The two matrices have incompatible shapes (M × K1 and K2 × N where K1 ≠ K2)",
            ),
        ));
    }

    let (counts, indices_and_data) = csr
        .iter()
        .map(move |lane| {
            let lane = lane.map(|(j, v)| (j, v.clone())).collect::<Vec<_>>();

            let (row_indices, row_data) = csc
                .iter()
                .enumerate()
                .filter_map(|(k, mut sublane)| {
                    let mut lane_iter = lane.iter();

                    let mut lhs = lane_iter.next();
                    let mut rhs = sublane.next();

                    let mut total = <T1 as Mul<T2>>::Output::zero();
                    let mut is_nonzero = false;

                    while lhs.is_some() && rhs.is_some() {
                        let (jl, vl) = lhs.unwrap();
                        let (jr, vr) = rhs.unwrap();

                        match jl.cmp(&jr) {
                            Ordering::Less => {
                                lhs = lane_iter.next();
                            }
                            Ordering::Equal => {
                                total += vl.clone() * vr.clone();
                                is_nonzero = true;
                            }
                            Ordering::Greater => {
                                rhs = sublane.next();
                            }
                        }
                    }

                    if is_nonzero {
                        Some((k, total))
                    } else {
                        None
                    }
                })
                .unzip::<_, _, Vec<_>, Vec<_>>();

            (row_indices.len(), (row_indices, row_data))
        })
        .unzip::<_, _, Vec<_>, Vec<_>>();

    let nnz = counts.iter().sum();
    let offsets = CountToOffsetIter::new(counts).collect();

    let mut indices = Vec::with_capacity(nnz);
    let mut data = Vec::with_capacity(nnz);

    for (mut row_indices, mut row_data) in indices_and_data {
        indices.append(&mut row_indices);
        data.append(&mut row_data);
    }

    Ok(unsafe { CsMatrix::from_parts_unchecked(rows, columns, offsets, indices, data) })
}

/// The fundamental (slowest) sparse-matrix multiply.
///
/// This function takes two arguments, a CSC matrix and a CSR matrix, and performs a sparse-matrix
/// multiplication with the CSC matrix on the left side and the CSR matrix on the right.
///
/// This matrix product is the slowest because lane orientation on both matrices is opposite of the
/// most efficient access pattern. As a result, we end up using the more expensive
/// [`CsMatrix::minor_lane_iter`] in order to iterate through lanes of the underlying data. As a
/// result, there is little that can be done for good cache performance since the access pattern
/// needed for `spmm` is antithetical to the structure of the matrices.
///
/// # Errors
///
/// This function fails and produces an [`OperationError`] with kind
/// [`OperationErrorKind::InvalidPattern`] if the two matrices have incompatible shapes for a
/// matrix product.
pub fn spmm_csc_csr<T1, T2, MO1, MO2, MI1, MI2, D1, D2>(
    csc: CsMatrix<T1, MO1, MI1, D1, CompressedColumnStorage>,
    csr: CsMatrix<T2, MO2, MI2, D2, CompressedRowStorage>,
) -> Result<CsrMatrix<<T1 as Mul<T2>>::Output>, OperationError>
where
    T1: Clone + Mul<T2>,
    <T1 as Mul<T2>>::Output: AddAssign + Zero,
    T2: Clone,
    MO1: Borrow<[usize]>,
    MO2: Borrow<[usize]>,
    MI1: Borrow<[usize]>,
    MI2: Borrow<[usize]>,
    D1: Borrow<[T1]>,
    D2: Borrow<[T2]>,
{
    let (rows, lc) = csc.shape();
    let (rr, columns) = csr.shape();

    if lc != rr {
        return Err(OperationError::from_kind_and_message(
            OperationErrorKind::InvalidPattern,
            String::from(
                "The two matrices have incompatible shapes (M × K1 and K2 × N where K1 ≠ K2)",
            ),
        ));
    }

    let (counts, indices_and_data) = csc
        .minor_lane_iter()
        .map(move |lane| {
            let lane = lane.map(|(j, v)| (j, v.clone())).collect::<Vec<_>>();

            let (row_indices, row_data) = csr
                .minor_lane_iter()
                .enumerate()
                .filter_map(|(k, mut sublane)| {
                    let mut lane_iter = lane.iter();

                    let mut lhs = lane_iter.next();
                    let mut rhs = sublane.next();

                    let mut total = <T1 as Mul<T2>>::Output::zero();
                    let mut is_nonzero = false;

                    while lhs.is_some() && rhs.is_some() {
                        let (jl, vl) = lhs.unwrap();
                        let (jr, vr) = rhs.unwrap();

                        match jl.cmp(&jr) {
                            Ordering::Less => {
                                lhs = lane_iter.next();
                            }
                            Ordering::Equal => {
                                total += vl.clone() * vr.clone();
                                is_nonzero = true;
                            }
                            Ordering::Greater => {
                                rhs = sublane.next();
                            }
                        }
                    }

                    if is_nonzero {
                        Some((k, total))
                    } else {
                        None
                    }
                })
                .unzip::<_, _, Vec<_>, Vec<_>>();

            (row_indices.len(), (row_indices, row_data))
        })
        .unzip::<_, _, Vec<_>, Vec<_>>();

    let nnz = counts.iter().sum();
    let offsets = CountToOffsetIter::new(counts).collect();

    let mut indices = Vec::with_capacity(nnz);
    let mut data = Vec::with_capacity(nnz);

    for (mut row_indices, mut row_data) in indices_and_data {
        indices.append(&mut row_indices);
        data.append(&mut row_data);
    }

    Ok(unsafe { CsMatrix::from_parts_unchecked(rows, columns, offsets, indices, data) })
}

/// The fundamental sparse-matrix multiply for sparse matrices of similar structure.
///
/// This function takes in two arguments, both CSC matrices, and computes the sparse matrix product
/// of the left and right hand sides. In this product, the lane access of the left hand side is the
/// opposite of what we want -- lanes are along columns, not rows. However, the right hand side
/// matrix has the correct layout and so therefore can be made fast and cache-friendly.
///
/// Unlike the `CSR * CSC` case, we have to use [`CsMatrix::minor_lane_iter`] for the left-hand
/// side matrix. However, rather than iterate through that iterator for every row / col pairing of
/// the matrix product we can cache it once to make the minor-lane access far less expensive and
/// cache friendly on re-use. This means that we have a slightly slower matrix product than in the
/// `CSR * CSC` case, but otherwise performs well on average at the cost of some extra memory
/// (retaining an extra copy of the current minor lane of the left hand side matrix).
///
/// # Errors
///
/// This function fails and produces an [`OperationError`] with kind
/// [`OperationErrorKind::InvalidPattern`] if the two matrices have incompatible shapes for a
/// matrix product.
pub fn spmm_csc_csc<T1, T2, MO1, MO2, MI1, MI2, D1, D2>(
    lhs: CsMatrix<T1, MO1, MI1, D1, CompressedColumnStorage>,
    rhs: CsMatrix<T2, MO2, MI2, D2, CompressedColumnStorage>,
) -> Result<CsrMatrix<<T1 as Mul<T2>>::Output>, OperationError>
where
    T1: Clone + Mul<T2>,
    <T1 as Mul<T2>>::Output: AddAssign + Zero,
    T2: Clone,
    MO1: Borrow<[usize]>,
    MO2: Borrow<[usize]>,
    MI1: Borrow<[usize]>,
    MI2: Borrow<[usize]>,
    D1: Borrow<[T1]>,
    D2: Borrow<[T2]>,
{
    let (rows, lc) = lhs.shape();
    let (rr, columns) = rhs.shape();

    if lc != rr {
        return Err(OperationError::from_kind_and_message(
            OperationErrorKind::InvalidPattern,
            String::from(
                "The two matrices have incompatible shapes (M × K1 and K2 × N where K1 ≠ K2)",
            ),
        ));
    }

    let (counts, indices_and_data) = lhs
        .minor_lane_iter()
        .map(move |lane| {
            let lane = lane.map(|(j, v)| (j, v.clone())).collect::<Vec<_>>();

            let (row_indices, row_data) = rhs
                .iter()
                .enumerate()
                .filter_map(|(k, mut sublane)| {
                    let mut lane_iter = lane.iter();

                    let mut lhs = lane_iter.next();
                    let mut rhs = sublane.next();

                    let mut total = <T1 as Mul<T2>>::Output::zero();
                    let mut is_nonzero = false;

                    while lhs.is_some() && rhs.is_some() {
                        let (jl, vl) = lhs.unwrap();
                        let (jr, vr) = rhs.unwrap();

                        match jl.cmp(&jr) {
                            Ordering::Less => {
                                lhs = lane_iter.next();
                            }
                            Ordering::Equal => {
                                total += vl.clone() * vr.clone();
                                is_nonzero = true;
                            }
                            Ordering::Greater => {
                                rhs = sublane.next();
                            }
                        }
                    }

                    if is_nonzero {
                        Some((k, total))
                    } else {
                        None
                    }
                })
                .unzip::<_, _, Vec<_>, Vec<_>>();

            (row_indices.len(), (row_indices, row_data))
        })
        .unzip::<_, _, Vec<_>, Vec<_>>();

    let nnz = counts.iter().sum();
    let offsets = CountToOffsetIter::new(counts).collect();

    let mut indices = Vec::with_capacity(nnz);
    let mut data = Vec::with_capacity(nnz);

    for (mut row_indices, mut row_data) in indices_and_data {
        indices.append(&mut row_indices);
        data.append(&mut row_data);
    }

    Ok(unsafe { CsMatrix::from_parts_unchecked(rows, columns, offsets, indices, data) })
}

/// The fundamental matrix product of two CSR matrices.
///
/// This function behaves identically to the `CSC * CSC` matrix product. It does this by exploiting
/// an identity in mathematics:
///
/// ```text
/// (B' <dot> A')' = A <dot> B
/// ```
///
/// Where the apostrophes in the above identity refer to matrix transposition. Matrix
/// transposition with the [`CsMatrix`] type is "free" in the sense that it doesn't require us to
/// reconfigure or move any data, only re-interpret it (see [`CsMatrix::transpose`] and
/// [`CsMatrix::transpose_owned`]). As a result, we can define the `CSR * CSR` product in terms of
/// the `CSC * CSC` product, and get the exact same performance as a result.
///
/// # Errors
///
/// This function fails and produces an [`OperationError`] with kind
/// [`OperationErrorKind::InvalidPattern`] if the two matrices have incompatible shapes for a
/// matrix product.
pub fn spmm_csr_csr<T1, T2, MO1, MO2, MI1, MI2, D1, D2>(
    lhs: CsMatrix<T1, MO1, MI1, D1, CompressedRowStorage>,
    rhs: CsMatrix<T2, MO2, MI2, D2, CompressedRowStorage>,
) -> Result<CscMatrix<<T2 as Mul<T1>>::Output>, OperationError>
where
    T2: Clone + Mul<T1>,
    <T2 as Mul<T1>>::Output: AddAssign + Zero,
    T1: Clone,
    MO1: Borrow<[usize]>,
    MO2: Borrow<[usize]>,
    MI1: Borrow<[usize]>,
    MI2: Borrow<[usize]>,
    D1: Borrow<[T1]>,
    D2: Borrow<[T2]>,
{
    Ok(spmm_csc_csc(rhs.transpose(), lhs.transpose())?.transpose_owned())
}

/// Sparse-Dense matrix multiplication.
///
/// This function takes in two matrices, one dense and one sparse in CSC format, and computes the
/// `Dense * CSC` matrix product.
///
/// # Errors
///
/// This function fails and produces an [`OperationError`] with kind
/// [`OperationErrorKind::InvalidPattern`] if the two matrices have incompatible shapes for a
/// matrix product.
pub fn spmm_dense_csc<T1, T2, R, C, S, MO, MI, D>(
    dense: Matrix<T1, R, C, S>,
    csc: CsMatrix<T2, MO, MI, D, CompressedColumnStorage>,
) -> Result<CsrMatrix<<T2 as Mul<T1>>::Output>, OperationError>
where
    T1: Scalar,
    R: Dim,
    C: Dim,
    S: RawStorage<T1, R, C>,
    T2: Clone + Mul<T1>,
    <T2 as Mul<T1>>::Output: Add + Zero,
    MO: Borrow<[usize]>,
    MI: Borrow<[usize]>,
    D: Borrow<[T2]>,
{
    let (rows, lc) = dense.shape();
    let (rr, columns) = csc.shape();

    if lc != rr {
        return Err(OperationError::from_kind_and_message(
            OperationErrorKind::InvalidPattern,
            String::from(
                "The two matrices have incompatible shapes (M × K1 and K2 × N where K1 ≠ K2)",
            ),
        ));
    }

    let (counts, indices_and_data) = (0..rows)
        .map(|i| {
            let dense_row = dense.row(i);

            let (row_indices, row_data) = csc
                .iter()
                .enumerate()
                .filter_map(|(k, lane)| {
                    if lane.len() == 0 {
                        None
                    } else {
                        let total = lane.fold(<T2 as Mul<T1>>::Output::zero(), |total, (j, v)| {
                            total + (v.clone() * dense_row[j].clone())
                        });

                        Some((k, total))
                    }
                })
                .unzip::<_, _, Vec<_>, Vec<_>>();

            (row_indices.len(), (row_indices, row_data))
        })
        .unzip::<_, _, Vec<_>, Vec<_>>();

    let nnz = counts.iter().sum();
    let offsets = CountToOffsetIter::new(counts).collect();

    let mut indices = Vec::with_capacity(nnz);
    let mut data = Vec::with_capacity(nnz);

    for (mut row_indices, mut row_data) in indices_and_data {
        indices.append(&mut row_indices);
        data.append(&mut row_data);
    }

    Ok(unsafe { CsMatrix::from_parts_unchecked(rows, columns, offsets, indices, data) })
}

/// Sparse-Dense matrix multiplication.
///
/// This function takes in two matrices, one sparse in CSR format and one dense, and computes the
/// `CSR * Dense` matrix product.
///
/// # Errors
///
/// This function fails and produces an [`OperationError`] with kind
/// [`OperationErrorKind::InvalidPattern`] if the two matrices have incompatible shapes for a
/// matrix product.
pub fn spmm_csr_dense<T1, T2, R, C, S, MO, MI, D>(
    csr: CsMatrix<T1, MO, MI, D, CompressedRowStorage>,
    dense: Matrix<T2, R, C, S>,
) -> Result<CscMatrix<<T1 as Mul<T2>>::Output>, OperationError>
where
    T2: Scalar,
    R: Dim,
    C: Dim,
    S: RawStorage<T2, R, C>,
    T1: Clone + Mul<T2>,
    <T1 as Mul<T2>>::Output: Add + Zero,
    MO: Borrow<[usize]>,
    MI: Borrow<[usize]>,
    D: Borrow<[T1]>,
{
    let (rows, lc) = csr.shape();
    let (rr, columns) = dense.shape();

    if lc != rr {
        return Err(OperationError::from_kind_and_message(
            OperationErrorKind::InvalidPattern,
            String::from(
                "The two matrices have incompatible shapes (M × K1 and K2 × N where K1 ≠ K2)",
            ),
        ));
    }

    // The trick to this function is to exploit the fact that:
    //
    // (B' <dot> A')' = A <dot> B
    //
    // However, we don't want to take an expensive transpose of the dense data -- instead we
    // transpose the CSR matrix directly (rather, we pretend we did, because we just need major
    // lane iteration), and then iterate through columns in the dense matrix instead of rows.
    //
    // This saves us from doing an expensive transpose + alloc on the dense matrix, while still
    // getting the same output. The only major difference is that unlike spmm_dense_csc we output a
    // CscMatrix as a final result (the transpose of the CsrMatrix constructed below).
    let (counts, indices_and_data) = (0..columns)
        .map(|i| {
            let dense_col = dense.column(i);

            let (row_indices, row_data) = csr
                .iter()
                .enumerate()
                .filter_map(|(k, lane)| {
                    if lane.len() == 0 {
                        None
                    } else {
                        let total = lane.fold(<T1 as Mul<T2>>::Output::zero(), |total, (j, v)| {
                            total + (v.clone() * dense_col[j].clone())
                        });

                        Some((k, total))
                    }
                })
                .unzip::<_, _, Vec<_>, Vec<_>>();

            (row_indices.len(), (row_indices, row_data))
        })
        .unzip::<_, _, Vec<_>, Vec<_>>();

    let nnz = counts.iter().sum();
    let offsets = CountToOffsetIter::new(counts).collect();

    let mut indices = Vec::with_capacity(nnz);
    let mut data = Vec::with_capacity(nnz);

    for (mut row_indices, mut row_data) in indices_and_data {
        indices.append(&mut row_indices);
        data.append(&mut row_data);
    }

    // The last step is to construct the final matrix (B' <dot> A').
    //
    // We have to remember that columns and rows are swapped here (so the below is not a typo), and
    // that we're immediately going to take an owned transpose of the data.
    Ok(
        unsafe { CsrMatrix::from_parts_unchecked(columns, rows, offsets, indices, data) }
            .transpose_owned(),
    )
}

/// Sparse-Dense matrix multiplication.
///
/// This function takes in two matrices, one sparse in CSC format and one dense, and computes the
/// `CSC * Dense` matrix product.
///
/// # Errors
///
/// This function fails and produces an [`OperationError`] with kind
/// [`OperationErrorKind::InvalidPattern`] if the two matrices have incompatible shapes for a
/// matrix product.
pub fn spmm_csc_dense<T1, T2, R, C, S, MO, MI, D>(
    csc: CsMatrix<T1, MO, MI, D, CompressedColumnStorage>,
    dense: Matrix<T2, R, C, S>,
) -> Result<CsrMatrix<<T1 as Mul<T2>>::Output>, OperationError>
where
    T2: Scalar,
    R: Dim,
    C: Dim,
    S: RawStorage<T2, R, C>,
    T1: Clone + Mul<T2>,
    <T1 as Mul<T2>>::Output: Add + Zero,
    MO: Borrow<[usize]>,
    MI: Borrow<[usize]>,
    D: Borrow<[T1]>,
{
    let (rows, lc) = csc.shape();
    let (rr, columns) = dense.shape();

    if lc != rr {
        return Err(OperationError::from_kind_and_message(
            OperationErrorKind::InvalidPattern,
            String::from(
                "The two matrices have incompatible shapes (M × K1 and K2 × N where K1 ≠ K2)",
            ),
        ));
    }

    let (counts, indices_and_data) = csc
        .minor_lane_iter()
        .map(|lane| {
            let lane = lane.map(|(j, v)| (j, v.clone())).collect::<Vec<_>>();

            if lane.is_empty() {
                (0, (Vec::new(), Vec::new()))
            } else {
                let (row_indices, row_data) = (0..columns)
                    .map(|k| {
                        let dense_col = dense.column(k);
                        let total = lane
                            .iter()
                            .fold(<T1 as Mul<T2>>::Output::zero(), |total, (j, v)| {
                                total + v.clone() * dense_col[*j].clone()
                            });

                        (k, total)
                    })
                    .unzip::<_, _, Vec<_>, Vec<_>>();

                (row_indices.len(), (row_indices, row_data))
            }
        })
        .unzip::<_, _, Vec<_>, Vec<_>>();

    let nnz = counts.iter().sum();
    let offsets = CountToOffsetIter::new(counts).collect();

    let mut indices = Vec::with_capacity(nnz);
    let mut data = Vec::with_capacity(nnz);

    for (mut row_indices, mut row_data) in indices_and_data {
        indices.append(&mut row_indices);
        data.append(&mut row_data);
    }

    Ok(unsafe { CsMatrix::from_parts_unchecked(rows, columns, offsets, indices, data) })
}

/// Sparse-Dense matrix multiplication.
///
/// This function takes in two matrices, one dense and one sparse in CSC format, and computes the
/// `Dense * CSR` matrix product.
///
/// # Errors
///
/// This function fails and produces an [`OperationError`] with kind
/// [`OperationErrorKind::InvalidPattern`] if the two matrices have incompatible shapes for a
/// matrix product.
pub fn spmm_dense_csr<T1, T2, R, C, S, MO, MI, D>(
    dense: Matrix<T1, R, C, S>,
    csr: CsMatrix<T2, MO, MI, D, CompressedColumnStorage>,
) -> Result<CscMatrix<<T2 as Mul<T1>>::Output>, OperationError>
where
    T1: Scalar,
    R: Dim,
    C: Dim,
    S: RawStorage<T1, R, C>,
    T2: Clone + Mul<T1>,
    <T2 as Mul<T1>>::Output: Add + Zero,
    MO: Borrow<[usize]>,
    MI: Borrow<[usize]>,
    D: Borrow<[T2]>,
{
    let (rows, lc) = dense.shape();
    let (rr, columns) = csr.shape();

    if lc != rr {
        return Err(OperationError::from_kind_and_message(
            OperationErrorKind::InvalidPattern,
            String::from(
                "The two matrices have incompatible shapes (M × K1 and K2 × N where K1 ≠ K2)",
            ),
        ));
    }

    // Like with spmm_csr_dense, this function shares the transposed relationship with
    // spmm_csc_dense. We can exploit this in a similar way, by swapping column / row references
    // for the dense matrix and transposing the final matrix.
    let (counts, indices_and_data) = csr
        .minor_lane_iter()
        .map(|lane| {
            let lane = lane.map(|(j, v)| (j, v.clone())).collect::<Vec<_>>();

            if lane.is_empty() {
                (0, (Vec::new(), Vec::new()))
            } else {
                let (row_indices, row_data) = (0..rows)
                    .map(|k| {
                        let dense_row = dense.row(k);
                        let total = lane
                            .iter()
                            .fold(<T2 as Mul<T1>>::Output::zero(), |total, (j, v)| {
                                total + v.clone() * dense_row[*j].clone()
                            });

                        (k, total)
                    })
                    .unzip::<_, _, Vec<_>, Vec<_>>();

                (row_indices.len(), (row_indices, row_data))
            }
        })
        .unzip::<_, _, Vec<_>, Vec<_>>();

    let nnz = counts.iter().sum();
    let offsets = CountToOffsetIter::new(counts).collect();

    let mut indices = Vec::with_capacity(nnz);
    let mut data = Vec::with_capacity(nnz);

    for (mut row_indices, mut row_data) in indices_and_data {
        indices.append(&mut row_indices);
        data.append(&mut row_data);
    }

    // Columns and rows are intentionally swapped here since we're taking the transpose of the
    // final data.
    Ok(
        unsafe { CsrMatrix::from_parts_unchecked(columns, rows, offsets, indices, data) }
            .transpose_owned(),
    )
}
