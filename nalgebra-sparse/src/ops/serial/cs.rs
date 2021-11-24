use crate::{
    convert::utils::CountToOffsetIter,
    cs::{CompressedColumnStorage, CompressedRowStorage, CsMatrix, CscMatrix, CsrMatrix},
    error::{OperationError, OperationErrorKind},
};
use num_traits::{Unsigned, Zero};
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
pub fn spmm_csr_csc<T1, T2, O1, O2, I1, I2, MO1, MO2, MI1, MI2, D1, D2>(
    csr: CsMatrix<T1, O1, MO1, MI1, D1, CompressedRowStorage, I1>,
    csc: CsMatrix<T2, O2, MO2, MI2, D2, CompressedColumnStorage, I2>,
) -> Result<CsrMatrix<<T1 as Mul<T2>>::Output, usize, usize>, OperationError>
where
    T1: Copy + Mul<T2>,
    <T1 as Mul<T2>>::Output: AddAssign + Zero,
    T2: Copy,
    O1: Add<usize, Output = usize> + Copy + Clone + Into<usize> + Unsigned + Ord,
    O2: Add<usize, Output = usize> + Copy + Clone + Into<usize> + Unsigned + Ord,
    I1: Copy + Clone + Into<usize> + Unsigned + Ord,
    I2: Copy + Clone + Into<usize> + Unsigned + Ord,
    MO1: Borrow<[O1]>,
    MO2: Borrow<[O2]>,
    MI1: Borrow<[I1]>,
    MI2: Borrow<[I2]>,
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
            // See the below comment about index comparisons for why we clone / convert to usize here.
            let lane = lane
                .map(|(i, v)| (i.clone().into(), *v))
                .collect::<Vec<_>>();

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
                        let (jr, &vr) = rhs.unwrap();

                        // The below conversion may seem strange; however, it is necessary.
                        //
                        // For any two matrices with differing index types I1 and I2, where I1 ≠ I2, we
                        // cannot guarantee that there is an ordering between them. Rust provides two
                        // traits to determine ordering:
                        //
                        // - PartialOrd, for claiming that there is a partial ordering between two
                        // types (i.e. not all pairs of I1 / I2 will have a defined ordering)
                        // - Ord, for claiming there is a complete ordering for a single type. (i.e.
                        // all pairs of `a` and `b` of type T will be ordered)
                        //
                        // You might think that we could mandate the trait bound `I1: PartialOrd<I2>`
                        // and avoid the copies / conversion into `usize`. However, the standard
                        // library does not define PartialOrd for any types where the right hand side
                        // is not equal to self, at least for all the unsigned types we care about for
                        // our indices. Instead, clone + converting these up into `usize` means that we
                        // can use `Ord` and `a.cmp(&b)` for all comparisons, which is guaranteed to be
                        // "safe" for any dimension size (Safe in quotes because you shouldn't be able
                        // to make a sparse matrix with dimensions larger than usize).
                        //
                        // This has the effect that we effectively have to clone all of our indices in
                        // order to be able to compare them. To save on the cost of this conversion,
                        // all `jl` indices are converted outside of this loop, and all `jr` indices
                        // are converted inside the loop.
                        let jr = jr.clone().into();

                        match jl.cmp(&jr) {
                            Ordering::Less => {
                                lhs = lane_iter.next();
                                continue;
                            }
                            Ordering::Equal => {
                                total += *vl * vr;
                                is_nonzero = true;
                            }
                            Ordering::Greater => {
                                rhs = sublane.next();
                                continue;
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
pub fn spmm_csc_csr<T1, T2, O1, O2, I1, I2, MO1, MO2, MI1, MI2, D1, D2>(
    csc: CsMatrix<T1, O1, MO1, MI1, D1, CompressedColumnStorage, I1>,
    csr: CsMatrix<T2, O2, MO2, MI2, D2, CompressedRowStorage, I2>,
) -> Result<CsrMatrix<<T1 as Mul<T2>>::Output, usize, usize>, OperationError>
where
    T1: Copy + Mul<T2>,
    <T1 as Mul<T2>>::Output: AddAssign + Zero,
    T2: Copy,
    O1: Add<usize, Output = usize> + Copy + Clone + Into<usize> + Unsigned + Ord,
    O2: Add<usize, Output = usize> + Copy + Clone + Into<usize> + Unsigned + Ord,
    I1: Copy + Clone + Into<usize> + Unsigned + Ord,
    I2: Copy + Clone + Into<usize> + Unsigned + Ord,
    MO1: Borrow<[O1]>,
    MO2: Borrow<[O2]>,
    MI1: Borrow<[I1]>,
    MI2: Borrow<[I2]>,
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
            let lane = lane.map(|(i, v)| (i, *v)).collect::<Vec<_>>();

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
                        let (jr, &vr) = rhs.unwrap();

                        match jl.cmp(&jr) {
                            Ordering::Less => {
                                lhs = lane_iter.next();
                                continue;
                            }
                            Ordering::Equal => {
                                total += *vl * vr;
                                is_nonzero = true;
                            }
                            Ordering::Greater => {
                                rhs = sublane.next();
                                continue;
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
pub fn spmm_csc_csc<T1, T2, O1, O2, I1, I2, MO1, MO2, MI1, MI2, D1, D2>(
    lhs: CsMatrix<T1, O1, MO1, MI1, D1, CompressedColumnStorage, I1>,
    rhs: CsMatrix<T2, O2, MO2, MI2, D2, CompressedColumnStorage, I2>,
) -> Result<CsrMatrix<<T1 as Mul<T2>>::Output, usize, usize>, OperationError>
where
    T1: Copy + Mul<T2>,
    <T1 as Mul<T2>>::Output: AddAssign + Zero,
    T2: Copy,
    O1: Add<usize, Output = usize> + Copy + Clone + Into<usize> + Unsigned + Ord,
    O2: Add<usize, Output = usize> + Copy + Clone + Into<usize> + Unsigned + Ord,
    I1: Copy + Clone + Into<usize> + Unsigned + Ord,
    I2: Copy + Clone + Into<usize> + Unsigned + Ord,
    MO1: Borrow<[O1]>,
    MO2: Borrow<[O2]>,
    MI1: Borrow<[I1]>,
    MI2: Borrow<[I2]>,
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
            let lane = lane.map(|(i, v)| (i, *v)).collect::<Vec<_>>();

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
                        let (jr, &vr) = rhs.unwrap();

                        // See comment in `spmm_csr_csc` for why this is necessary
                        let jr = jr.clone().into();

                        match jl.cmp(&jr) {
                            Ordering::Less => {
                                lhs = lane_iter.next();
                                continue;
                            }
                            Ordering::Equal => {
                                total += *vl * vr;
                                is_nonzero = true;
                            }
                            Ordering::Greater => {
                                rhs = sublane.next();
                                continue;
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
pub fn spmm_csr_csr<T1, T2, O1, O2, I1, I2, MO1, MO2, MI1, MI2, D1, D2>(
    lhs: CsMatrix<T1, O1, MO1, MI1, D1, CompressedRowStorage, I1>,
    rhs: CsMatrix<T2, O2, MO2, MI2, D2, CompressedRowStorage, I2>,
) -> Result<CscMatrix<<T2 as Mul<T1>>::Output, usize, usize>, OperationError>
where
    T2: Copy + Mul<T1>,
    <T2 as Mul<T1>>::Output: AddAssign + Zero,
    T1: Copy,
    O1: Add<usize, Output = usize> + Copy + Clone + Into<usize> + Unsigned + Ord,
    O2: Add<usize, Output = usize> + Copy + Clone + Into<usize> + Unsigned + Ord,
    I1: Copy + Clone + Into<usize> + Unsigned + Ord,
    I2: Copy + Clone + Into<usize> + Unsigned + Ord,
    MO1: Borrow<[O1]>,
    MO2: Borrow<[O2]>,
    MI1: Borrow<[I1]>,
    MI2: Borrow<[I2]>,
    D1: Borrow<[T1]>,
    D2: Borrow<[T2]>,
{
    Ok(spmm_csc_csc(rhs.transpose(), lhs.transpose())?.transpose_owned())
}
