//! Module holding the various sparse-matrix addition functions.
//!
//! Sparse-matrix addition performance can somewhat depend on the underlying data types. In
//! particular, adding two matrices with the same compression kind will tend to perform better than
//! adding two matrices of opposing (row vs. column major) storage layouts.
//!
//! # Performance
//!
//! Because of the above, the contained `spadd_x_y` functions may have slightly different
//! performance characteristics. In general, `spadd_csr_csr` and `spadd_csc_csc` are the fastest
//! possible addition operations one can perform. Mixing formats, such as in `spadd_csc_csr` or
//! `spadd_csr_csc` is slightly slower due to needing to iterate along the minor lane of the
//! secondary matrix.
//!
//! One should prefer to pose their problems as a combination of CSX <-> CSX additions, or
//! dense-sparse additions.

use crate::{
    convert::utils::CountToOffsetIter,
    cs::{CompressedColumnStorage, CompressedRowStorage, CsMatrix, CscMatrix, CsrMatrix},
    error::{OperationError, OperationErrorKind},
};
use nalgebra::{Dim, Matrix, RawStorage, RawStorageMut, Scalar};
use std::{borrow::Borrow, cmp::Ordering, ops::Add};

/// Sparse-sparse matrix addition.
///
/// This function takes two arguments, a CSR matrix and CSC matrix.to_view(), and performs sparse-matrix
/// addition between the two.
///
/// # Errors
///
/// This function fails and produces an [`OperationError`] with kind
/// [`OperationErrorKind::InvalidPattern`] if the two matrices do not have the exact same shape.
pub fn spadd_csr_csc<T1, T2, MO1, MO2, MI1, MI2, D1, D2>(
    csr: CsMatrix<T1, MO1, MI1, D1, CompressedRowStorage>,
    csc: CsMatrix<T2, MO2, MI2, D2, CompressedColumnStorage>,
) -> Result<CsrMatrix<<T1 as Add<T2>>::Output>, OperationError>
where
    T1: Scalar + Into<<T1 as Add<T2>>::Output> + Add<T2>,
    T2: Scalar + Into<<T1 as Add<T2>>::Output>,
    <T1 as Add<T2>>::Output: Scalar,
    MO1: Borrow<[usize]>,
    MO2: Borrow<[usize]>,
    MI1: Borrow<[usize]>,
    MI2: Borrow<[usize]>,
    D1: Borrow<[T1]>,
    D2: Borrow<[T2]>,
{
    let (lrows, lcols) = csr.shape();
    let (rrows, rcols) = csc.shape();

    if lrows != rrows || lcols != rcols {
        return Err(OperationError::from_kind_and_message(
            OperationErrorKind::InvalidPattern,
            String::from("The two matrices have differing shapes (both should be M × N)"),
        ));
    }

    let mut left_iter = csr.triplet_iter();
    let mut right_iter = csc
        .minor_lane_iter()
        .enumerate()
        .flat_map(|(i, lane)| lane.map(move |(j, value)| (i, j, value)));

    let left_val = left_iter.next();
    let right_val = right_iter.next();

    let added_triplets = TripletAdditionIter {
        left_val,
        right_val,
        left_iter,
        right_iter,
    };

    let max_nnz = csr.nnz() + csc.nnz();
    let mut counts = vec![0; lrows];
    let mut indices = Vec::with_capacity(max_nnz);
    let mut data = Vec::with_capacity(max_nnz);

    for (i, j, v) in added_triplets {
        counts[i] += 1;
        indices.push(j);
        data.push(v);
    }

    let offsets = CountToOffsetIter::new(counts).collect();

    Ok(unsafe { CsMatrix::from_parts_unchecked(lrows, lcols, offsets, indices, data) })
}

/// Sparse-sparse matrix addition.
///
/// This function takes two arguments, a CSC matrix and a CSR matrix.to_view(), and performs sparse-matrix
/// addition between the two.
///
/// # Errors
///
/// This function fails and produces an [`OperationError`] with kind
/// [`OperationErrorKind::InvalidPattern`] if the two matrices do not have the exact same shape.
pub fn spadd_csc_csr<T1, T2, MO1, MO2, MI1, MI2, D1, D2>(
    csc: CsMatrix<T1, MO1, MI1, D1, CompressedColumnStorage>,
    csr: CsMatrix<T2, MO2, MI2, D2, CompressedRowStorage>,
) -> Result<CsrMatrix<<T2 as Add<T1>>::Output>, OperationError>
where
    T1: Scalar + Into<<T2 as Add<T1>>::Output>,
    T2: Scalar + Into<<T2 as Add<T1>>::Output> + Add<T1>,
    <T2 as Add<T1>>::Output: Scalar,
    MO1: Borrow<[usize]>,
    MO2: Borrow<[usize]>,
    MI1: Borrow<[usize]>,
    MI2: Borrow<[usize]>,
    D1: Borrow<[T1]>,
    D2: Borrow<[T2]>,
{
    spadd_csr_csc(csr, csc)
}

/// Sparse-sparse matrix addition.
///
/// This function takes two CSC matrices, and performs sparse-matrix addition between the two.
///
/// # Errors
///
/// This function fails and produces an [`OperationError`] with kind
/// [`OperationErrorKind::InvalidPattern`] if the two matrices do not have the exact same shape.
pub fn spadd_csc_csc<T1, T2, MO1, MO2, MI1, MI2, D1, D2>(
    lhs: CsMatrix<T1, MO1, MI1, D1, CompressedColumnStorage>,
    rhs: CsMatrix<T2, MO2, MI2, D2, CompressedColumnStorage>,
) -> Result<CscMatrix<<T1 as Add<T2>>::Output>, OperationError>
where
    T1: Scalar + Into<<T1 as Add<T2>>::Output> + Add<T2>,
    T2: Scalar + Into<<T1 as Add<T2>>::Output>,
    <T1 as Add<T2>>::Output: Scalar,
    MO1: Borrow<[usize]>,
    MO2: Borrow<[usize]>,
    MI1: Borrow<[usize]>,
    MI2: Borrow<[usize]>,
    D1: Borrow<[T1]>,
    D2: Borrow<[T2]>,
{
    let (lrows, lcols) = lhs.shape();
    let (rrows, rcols) = rhs.shape();

    if lrows != rrows || lcols != rcols {
        return Err(OperationError::from_kind_and_message(
            OperationErrorKind::InvalidPattern,
            String::from("The two matrices have differing shapes (both should be M × N)"),
        ));
    }

    let mut left_iter = lhs.triplet_iter();
    let mut right_iter = rhs.triplet_iter();

    let left_val = left_iter.next();
    let right_val = right_iter.next();

    let added_triplets = TripletAdditionIter {
        left_val,
        right_val,
        left_iter,
        right_iter,
    };

    let max_nnz = lhs.nnz() + rhs.nnz();
    let mut counts = vec![0; lcols];
    let mut indices = Vec::with_capacity(max_nnz);
    let mut data = Vec::with_capacity(max_nnz);

    for (i, j, v) in added_triplets {
        counts[i] += 1;
        indices.push(j);
        data.push(v);
    }

    let offsets = CountToOffsetIter::new(counts).collect();

    Ok(unsafe { CsMatrix::from_parts_unchecked(lrows, lcols, offsets, indices, data) })
}

/// Sparse-sparse matrix addition.
///
/// This function takes two CSR matrices, and performs sparse-matrix addition between the two.
///
/// # Errors
///
/// This function fails and produces an [`OperationError`] with kind
/// [`OperationErrorKind::InvalidPattern`] if the two matrices do not have the exact same shape.
pub fn spadd_csr_csr<T1, T2, MO1, MO2, MI1, MI2, D1, D2>(
    lhs: CsMatrix<T1, MO1, MI1, D1, CompressedRowStorage>,
    rhs: CsMatrix<T2, MO2, MI2, D2, CompressedRowStorage>,
) -> Result<CsrMatrix<<T1 as Add<T2>>::Output>, OperationError>
where
    T1: Scalar + Into<<T1 as Add<T2>>::Output> + Add<T2>,
    T2: Scalar + Into<<T1 as Add<T2>>::Output>,
    <T1 as Add<T2>>::Output: Scalar,
    MO1: Borrow<[usize]>,
    MO2: Borrow<[usize]>,
    MI1: Borrow<[usize]>,
    MI2: Borrow<[usize]>,
    D1: Borrow<[T1]>,
    D2: Borrow<[T2]>,
{
    Ok(spadd_csc_csc(lhs.transpose(), rhs.transpose())?.transpose_owned())
}

/// Dense-sparse matrix addition.
///
/// This function takes in two matrices, one dense and one CSC matrix.to_view(), and performs dense-sparse
/// addition between the two.
///
/// # Errors
///
/// Thsi function fails and produces an [`OperationError`] with kind
/// [`OperationErrorKind::InvalidPattern`] if the two matrices do not have the exact same shape.
pub fn spadd_dense_csc<T1, T2, R, C, S, MO, MI, D>(
    mut dense: Matrix<T1, R, C, S>,
    csc: CsMatrix<T2, MO, MI, D, CompressedColumnStorage>,
) -> Result<Matrix<T1, R, C, S>, OperationError>
where
    T1: Scalar + Add<T2, Output = T1>,
    R: Dim,
    C: Dim,
    S: RawStorage<T1, R, C> + RawStorageMut<T1, R, C>,
    T2: Scalar,
    MO: Borrow<[usize]>,
    MI: Borrow<[usize]>,
    D: Borrow<[T2]>,
{
    let (lrows, lcols) = dense.shape();
    let (rrows, rcols) = csc.shape();

    if lrows != rrows || lcols != rcols {
        return Err(OperationError::from_kind_and_message(
            OperationErrorKind::InvalidPattern,
            String::from("The two matrices have differing shapes (both should be M × N)"),
        ));
    }

    for (col, row, val) in csc.triplet_iter() {
        let current = dense.index((row, col)).clone();
        *dense.index_mut((row, col)) = current + val.clone();
    }

    Ok(dense)
}

/// Dense-sparse matrix addition.
///
/// This function takes in two matrices, one dense and one CSC matrix.to_view(), and performs dense-sparse
/// addition between the two.
///
/// # Errors
///
/// Thsi function fails and produces an [`OperationError`] with kind
/// [`OperationErrorKind::InvalidPattern`] if the two matrices do not have the exact same shape.
pub fn spadd_csc_dense<T1, T2, R, C, S, MO, MI, D>(
    csc: CsMatrix<T1, MO, MI, D, CompressedColumnStorage>,
    dense: Matrix<T2, R, C, S>,
) -> Result<Matrix<T2, R, C, S>, OperationError>
where
    T2: Scalar + Add<T1, Output = T2>,
    R: Dim,
    C: Dim,
    S: RawStorage<T2, R, C> + RawStorageMut<T2, R, C>,
    T1: Scalar,
    MO: Borrow<[usize]>,
    MI: Borrow<[usize]>,
    D: Borrow<[T1]>,
{
    spadd_dense_csc(dense, csc)
}

/// Dense-sparse matrix addition.
///
/// This function takes in two matrices, one dense and one CSR matrix.to_view(), and performs dense-sparse
/// addition between the two.
///
/// # Errors
///
/// Thsi function fails and produces an [`OperationError`] with kind
/// [`OperationErrorKind::InvalidPattern`] if the two matrices do not have the exact same shape.
pub fn spadd_dense_csr<T1, T2, R, C, S, MO, MI, D>(
    mut dense: Matrix<T1, R, C, S>,
    csr: CsMatrix<T2, MO, MI, D, CompressedRowStorage>,
) -> Result<Matrix<T1, R, C, S>, OperationError>
where
    T1: Scalar + Add<T2, Output = T1>,
    R: Dim,
    C: Dim,
    S: RawStorage<T1, R, C> + RawStorageMut<T1, R, C>,
    T2: Scalar,
    MO: Borrow<[usize]>,
    MI: Borrow<[usize]>,
    D: Borrow<[T2]>,
{
    let (lrows, lcols) = dense.shape();
    let (rrows, rcols) = csr.shape();

    if lrows != rrows || lcols != rcols {
        return Err(OperationError::from_kind_and_message(
            OperationErrorKind::InvalidPattern,
            String::from("The two matrices have differing shapes (both should be M × N)"),
        ));
    }

    for (row, col, val) in csr.triplet_iter() {
        let current = dense.index((row, col)).clone();
        *dense.index_mut((row, col)) = current + val.clone();
    }

    Ok(dense)
}

/// Dense-sparse matrix addition.
///
/// This function takes in two matrices, one dense and one CSR matrix.to_view(), and performs dense-sparse
/// addition between the two.
///
/// # Errors
///
/// Thsi function fails and produces an [`OperationError`] with kind
/// [`OperationErrorKind::InvalidPattern`] if the two matrices do not have the exact same shape.
pub fn spadd_csr_dense<T1, T2, R, C, S, MO, MI, D>(
    csr: CsMatrix<T1, MO, MI, D, CompressedRowStorage>,
    dense: Matrix<T2, R, C, S>,
) -> Result<Matrix<T2, R, C, S>, OperationError>
where
    T2: Scalar + Add<T1, Output = T2>,
    R: Dim,
    C: Dim,
    S: RawStorage<T2, R, C> + RawStorageMut<T2, R, C>,
    T1: Scalar,
    MO: Borrow<[usize]>,
    MI: Borrow<[usize]>,
    D: Borrow<[T1]>,
{
    spadd_dense_csr(dense, csr)
}

/// Helper type which defines an iterator that takes two triplet iterators (in the same major
/// ordering) and outputs a single triplet iterator of `(major, minor, owned_value)`. This iterator
/// combines duplicate entries via addition i.e. if we have two entries with the same major and
/// minor indices we add the two values together.
///
/// This iterator combinator is not terribly well bound, which is why it is private! First and
/// foremost, `left_val` and `right_val` need to not be `None` immediately, or this will always
/// return `None`. Secondly, this iterator is difficult to use directly outside of the above
/// `spadd_csx_csx` functions because it relies on the fact that both iterators are outputting
/// triplets in the same major ordering. This means that if you want to add a CSC and CSR matrix.to_view(),
/// you cannot do the following:
///
/// ```ignore
/// let mut left_iter = csr.triplet_iter();
/// let mut right_iter = csc.triplet_iter();
///
/// let left_val = left_iter.next();
/// let right_val = right_iter.next();
/// ```
///
/// Because the major and minor ordering of those two are reversed. Instead, you'll want to do
/// something like the following:
///
/// ```ignore
/// let mut left_iter = csr.triplet_iter();
/// let mut right_iter = csc
///     .minor_lane_iter()
///     .enumerate()
///     .flat_map(|(i, lane)| lane.map(move |(j, value)| (i, j, value)));
///
/// let left_val = left_iter.next();
/// let right_val = right_iter.next();
/// ```
///
/// The minor lane iterator is necessary so that you can both traverse the two matrices in
/// the equivalent order (which is needed for the boolean logic below to work) as well as compare
/// `i` to `i` and `j` to `j` between both left and right values.
///
/// Of course, because addition is commutative, one could just reverse the order of the arguments
/// and do the same trick; however, it means that the final result will be a column-major matrix
/// instead of a row-major matrix.
struct TripletAdditionIter<'a, TL, TR, IL, IR>
where
    TL: Clone + Into<<TL as Add<TR>>::Output> + Add<TR>,
    TR: Clone + Into<<TL as Add<TR>>::Output>,
    IL: Iterator<Item = (usize, usize, &'a TL)>,
    IR: Iterator<Item = (usize, usize, &'a TR)>,
{
    left_val: Option<(usize, usize, &'a TL)>,
    right_val: Option<(usize, usize, &'a TR)>,
    left_iter: IL,
    right_iter: IR,
}

impl<'a, TL, TR, IL, IR> Iterator for TripletAdditionIter<'a, TL, TR, IL, IR>
where
    TL: Clone + Into<<TL as Add<TR>>::Output> + Add<TR>,
    TR: Clone + Into<<TL as Add<TR>>::Output>,
    IL: Iterator<Item = (usize, usize, &'a TL)>,
    IR: Iterator<Item = (usize, usize, &'a TR)>,
{
    type Item = (usize, usize, <TL as Add<TR>>::Output);

    fn next(&mut self) -> Option<Self::Item> {
        match (self.left_val, self.right_val) {
            // Neither left nor right are exhausted
            (Some((il, jl, vl)), Some((ir, jr, vr))) => match (il.cmp(&ir), jl.cmp(&jr)) {
                (Ordering::Less, _) | (Ordering::Equal, Ordering::Less) => {
                    self.left_val = self.left_iter.next();
                    Some((il, jl, vl.clone().into()))
                }

                (Ordering::Greater, _) | (Ordering::Equal, Ordering::Greater) => {
                    self.right_val = self.right_iter.next();
                    Some((ir, jr, vr.clone().into()))
                }

                (Ordering::Equal, Ordering::Equal) => {
                    self.left_val = self.left_iter.next();
                    self.right_val = self.right_iter.next();

                    Some((il, jl, vl.clone() + vr.clone()))
                }
            },

            // Only right is exhausted
            (Some((i, j, v)), None) => {
                self.left_val = self.left_iter.next();
                Some((i, j, v.clone().into()))
            }

            // Only left is exhausted
            (None, Some((i, j, v))) => {
                self.right_val = self.right_iter.next();
                Some((i, j, v.clone().into()))
            }

            // Both are exhausted
            (None, None) => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proptest::*;
    use nalgebra::DMatrix;
    use proptest::prelude::*;

    #[test]
    fn spadd_csr_csr_is_commutative() {
        let lhs = CsrMatrix::try_from_parts(
            4,
            4,
            vec![0, 4, 7, 10],
            vec![0, 1, 2, 3, 0, 1, 3, 1, 2, 3, 0, 1, 3],
            vec![1, 2, 3, 4, -1, 2, 5, 4, -2, 6, 2, 4, 6],
        )
        .unwrap();

        let rhs = CsrMatrix::<i32>::identity(4);

        let lhs_plus_rhs = spadd_csr_csr(lhs.to_view(), rhs.to_view()).unwrap();
        let rhs_plus_lhs = spadd_csr_csr(rhs, lhs).unwrap();

        assert_eq!(lhs_plus_rhs.shape(), rhs_plus_lhs.shape());

        let (offsets, indices, data) = lhs_plus_rhs.cs_data();
        let (expected_offsets, expected_indices, expected_data) = rhs_plus_lhs.cs_data();

        assert!(offsets.iter().zip(expected_offsets).all(|(a, b)| a == b));
        assert!(indices.iter().zip(expected_indices).all(|(a, b)| a == b));
        assert!(data.iter().zip(expected_data).all(|(a, b)| a == b));
    }

    #[test]
    fn spadd_csr_csr_is_distributive() {
        let a = CsrMatrix::try_from_parts(
            4,
            4,
            vec![0, 4, 7, 10],
            vec![0, 1, 2, 3, 0, 1, 3, 1, 2, 3, 0, 1, 3],
            vec![1, 2, 3, 4, -1, 2, 5, 4, -2, 6, 2, 4, 6],
        )
        .unwrap();

        let b = CsrMatrix::try_from_parts(
            4,
            4,
            vec![0, 4, 6, 8],
            vec![0, 1, 2, 3, 1, 3, 1, 3, 0, 1, 2, 3],
            vec![6, 4, 2, 8, 1, 7, 2, 6, 4, 1, 6, 3],
        )
        .unwrap();

        let c = CsrMatrix::<i32>::identity(4);

        let a_b_then_c = spadd_csr_csr(
            spadd_csr_csr(a.to_view(), b.to_view()).unwrap(),
            c.to_view(),
        )
        .unwrap();
        let a_then_b_c = spadd_csr_csr(a, spadd_csr_csr(b, c).unwrap()).unwrap();

        assert_eq!(a_b_then_c.shape(), a_then_b_c.shape());

        let (offsets, indices, data) = a_b_then_c.cs_data();
        let (expected_offsets, expected_indices, expected_data) = a_then_b_c.cs_data();

        assert!(offsets.iter().zip(expected_offsets).all(|(a, b)| a == b));
        assert!(indices.iter().zip(expected_indices).all(|(a, b)| a == b));
        assert!(data.iter().zip(expected_data).all(|(a, b)| a == b));
    }

    #[test]
    fn spadd_csr_csc_is_commutative() {
        let lhs = CsrMatrix::try_from_parts(
            4,
            4,
            vec![0, 4, 7, 10],
            vec![0, 1, 2, 3, 0, 1, 3, 1, 2, 3, 0, 1, 3],
            vec![1, 2, 3, 4, -1, 2, 5, 4, -2, 6, 2, 4, 6],
        )
        .unwrap();

        let rhs = CscMatrix::<i32>::identity(4);

        let lhs_plus_rhs = spadd_csr_csc(lhs.to_view(), rhs.to_view()).unwrap();
        let rhs_plus_lhs = spadd_csc_csr(rhs, lhs).unwrap();

        assert_eq!(lhs_plus_rhs.shape(), rhs_plus_lhs.shape());

        let (offsets, indices, data) = lhs_plus_rhs.cs_data();
        let (expected_offsets, expected_indices, expected_data) = rhs_plus_lhs.cs_data();

        assert!(offsets.iter().zip(expected_offsets).all(|(a, b)| a == b));
        assert!(indices.iter().zip(expected_indices).all(|(a, b)| a == b));
        assert!(data.iter().zip(expected_data).all(|(a, b)| a == b));
    }

    #[test]
    fn spadd_csr_csc_is_distributive() {
        let a = CsrMatrix::try_from_parts(
            4,
            4,
            vec![0, 4, 7, 10],
            vec![0, 1, 2, 3, 0, 1, 3, 1, 2, 3, 0, 1, 3],
            vec![1, 2, 3, 4, -1, 2, 5, 4, -2, 6, 2, 4, 6],
        )
        .unwrap();

        let b = CscMatrix::try_from_parts(
            4,
            4,
            vec![0, 4, 6, 8],
            vec![0, 1, 2, 3, 1, 3, 1, 3, 0, 1, 2, 3],
            vec![6, 4, 2, 8, 1, 7, 2, 6, 4, 1, 6, 3],
        )
        .unwrap();

        let c = CsrMatrix::<i32>::identity(4);

        let a_b_then_c = spadd_csr_csr(
            spadd_csr_csc(a.to_view(), b.to_view()).unwrap(),
            c.to_view(),
        )
        .unwrap();
        let a_then_b_c = spadd_csr_csr(a, spadd_csc_csr(b, c).unwrap()).unwrap();

        assert_eq!(a_b_then_c.shape(), a_then_b_c.shape());

        let (offsets, indices, data) = a_b_then_c.cs_data();
        let (expected_offsets, expected_indices, expected_data) = a_then_b_c.cs_data();

        assert!(offsets.iter().zip(expected_offsets).all(|(a, b)| a == b));
        assert!(indices.iter().zip(expected_indices).all(|(a, b)| a == b));
        assert!(data.iter().zip(expected_data).all(|(a, b)| a == b));
    }

    #[test]
    fn spadd_csc_csc_is_commutative() {
        let lhs = CscMatrix::try_from_parts(
            4,
            4,
            vec![0, 4, 7, 10],
            vec![0, 1, 2, 3, 0, 1, 3, 1, 2, 3, 0, 1, 3],
            vec![1, 2, 3, 4, -1, 2, 5, 4, -2, 6, 2, 4, 6],
        )
        .unwrap();

        let rhs = CscMatrix::<i32>::identity(4);

        let lhs_plus_rhs = spadd_csc_csc(lhs.to_view(), rhs.to_view()).unwrap();
        let rhs_plus_lhs = spadd_csc_csc(rhs, lhs).unwrap();

        assert_eq!(lhs_plus_rhs.shape(), rhs_plus_lhs.shape());

        let (offsets, indices, data) = lhs_plus_rhs.cs_data();
        let (expected_offsets, expected_indices, expected_data) = rhs_plus_lhs.cs_data();

        assert!(offsets.iter().zip(expected_offsets).all(|(a, b)| a == b));
        assert!(indices.iter().zip(expected_indices).all(|(a, b)| a == b));
        assert!(data.iter().zip(expected_data).all(|(a, b)| a == b));
    }

    #[test]
    fn spadd_csc_csc_is_distributive() {
        let a = CscMatrix::try_from_parts(
            4,
            4,
            vec![0, 4, 7, 10],
            vec![0, 1, 2, 3, 0, 1, 3, 1, 2, 3, 0, 1, 3],
            vec![1, 2, 3, 4, -1, 2, 5, 4, -2, 6, 2, 4, 6],
        )
        .unwrap();

        let b = CscMatrix::try_from_parts(
            4,
            4,
            vec![0, 4, 6, 8],
            vec![0, 1, 2, 3, 1, 3, 1, 3, 0, 1, 2, 3],
            vec![6, 4, 2, 8, 1, 7, 2, 6, 4, 1, 6, 3],
        )
        .unwrap();

        let c = CscMatrix::<i32>::identity(4);

        let a_b_then_c = spadd_csc_csc(
            spadd_csc_csc(a.to_view(), b.to_view()).unwrap(),
            c.to_view(),
        )
        .unwrap();
        let a_then_b_c = spadd_csc_csc(a, spadd_csc_csc(b, c).unwrap()).unwrap();

        assert_eq!(a_b_then_c.shape(), a_then_b_c.shape());

        let (offsets, indices, data) = a_b_then_c.cs_data();
        let (expected_offsets, expected_indices, expected_data) = a_then_b_c.cs_data();

        assert!(offsets.iter().zip(expected_offsets).all(|(a, b)| a == b));
        assert!(indices.iter().zip(expected_indices).all(|(a, b)| a == b));
        assert!(data.iter().zip(expected_data).all(|(a, b)| a == b));
    }

    #[test]
    fn spadd_csr_dense_is_commutative() {
        let lhs = CsrMatrix::try_from_parts(
            4,
            4,
            vec![0, 4, 7, 10],
            vec![0, 1, 2, 3, 0, 1, 3, 1, 2, 3, 0, 1, 3],
            vec![1, 2, 3, 4, -1, 2, 5, 4, -2, 6, 2, 4, 6],
        )
        .unwrap();

        let rhs = DMatrix::<i32>::identity(4, 4);

        let lhs_plus_rhs = spadd_csr_dense(lhs.to_view(), rhs.clone()).unwrap();
        let rhs_plus_lhs = spadd_dense_csr(rhs, lhs).unwrap();

        assert_eq!(lhs_plus_rhs, rhs_plus_lhs);
    }

    #[test]
    fn spadd_csr_dense_is_distributive() {
        let a = CsrMatrix::try_from_parts(
            4,
            4,
            vec![0, 4, 7, 10],
            vec![0, 1, 2, 3, 0, 1, 3, 1, 2, 3, 0, 1, 3],
            vec![1, 2, 3, 4, -1, 2, 5, 4, -2, 6, 2, 4, 6],
        )
        .unwrap();

        let b = CsrMatrix::try_from_parts(
            4,
            4,
            vec![0, 4, 6, 8],
            vec![0, 1, 2, 3, 1, 3, 1, 3, 0, 1, 2, 3],
            vec![6, 4, 2, 8, 1, 7, 2, 6, 4, 1, 6, 3],
        )
        .unwrap();

        let c = DMatrix::<i32>::identity(4, 4);

        let a_b_then_c =
            spadd_csr_dense(spadd_csr_csr(a.to_view(), b.to_view()).unwrap(), c.clone()).unwrap();
        let a_then_b_c = spadd_csr_dense(a, spadd_csr_dense(b, c).unwrap()).unwrap();

        assert_eq!(a_b_then_c, a_then_b_c);
    }

    #[test]
    fn spadd_csc_dense_is_commutative() {
        let lhs = CscMatrix::try_from_parts(
            4,
            4,
            vec![0, 4, 7, 10],
            vec![0, 1, 2, 3, 0, 1, 3, 1, 2, 3, 0, 1, 3],
            vec![1, 2, 3, 4, -1, 2, 5, 4, -2, 6, 2, 4, 6],
        )
        .unwrap();

        let rhs = DMatrix::<i32>::identity(4, 4);

        let lhs_plus_rhs = spadd_csc_dense(lhs.to_view(), rhs.clone()).unwrap();
        let rhs_plus_lhs = spadd_dense_csc(rhs, lhs).unwrap();

        assert_eq!(lhs_plus_rhs, rhs_plus_lhs);
    }

    #[test]
    fn spadd_csc_dense_is_distributive() {
        let a = CscMatrix::try_from_parts(
            4,
            4,
            vec![0, 4, 7, 10],
            vec![0, 1, 2, 3, 0, 1, 3, 1, 2, 3, 0, 1, 3],
            vec![1, 2, 3, 4, -1, 2, 5, 4, -2, 6, 2, 4, 6],
        )
        .unwrap();

        let b = CscMatrix::try_from_parts(
            4,
            4,
            vec![0, 4, 6, 8],
            vec![0, 1, 2, 3, 1, 3, 1, 3, 0, 1, 2, 3],
            vec![6, 4, 2, 8, 1, 7, 2, 6, 4, 1, 6, 3],
        )
        .unwrap();

        let c = DMatrix::<i32>::identity(4, 4);

        let a_b_then_c =
            spadd_csc_dense(spadd_csc_csc(a.to_view(), b.to_view()).unwrap(), c.clone()).unwrap();
        let a_then_b_c = spadd_csc_dense(a, spadd_csc_dense(b, c).unwrap()).unwrap();

        assert_eq!(a_b_then_c, a_then_b_c);
    }

    #[test]
    fn spadd_csr_csr_agrees_with_dense() {
        let a = CsrMatrix::try_from_parts(
            4,
            4,
            vec![0, 4, 7, 10],
            vec![0, 1, 2, 3, 0, 1, 3, 1, 2, 3, 0, 1, 3],
            vec![1, 2, 3, 4, -1, 2, 5, 4, -2, 6, 2, 4, 6],
        )
        .unwrap();

        let b = CsrMatrix::try_from_parts(
            4,
            4,
            vec![0, 4, 6, 8],
            vec![0, 1, 2, 3, 1, 3, 1, 3, 0, 1, 2, 3],
            vec![6, 4, 2, 8, 1, 7, 2, 6, 4, 1, 6, 3],
        )
        .unwrap();

        let dense_a = DMatrix::from(&a);
        let dense_b = DMatrix::from(&b);

        let sum = DMatrix::from(&spadd_csr_csr(a, b).unwrap());
        let dense_sum = dense_a + dense_b;

        assert_eq!(sum, dense_sum);
    }

    #[test]
    fn spadd_csr_csc_agrees_with_dense() {
        let a = CsrMatrix::try_from_parts(
            4,
            4,
            vec![0, 4, 7, 10],
            vec![0, 1, 2, 3, 0, 1, 3, 1, 2, 3, 0, 1, 3],
            vec![1, 2, 3, 4, -1, 2, 5, 4, -2, 6, 2, 4, 6],
        )
        .unwrap();

        let b = CscMatrix::try_from_parts(
            4,
            4,
            vec![0, 4, 6, 8],
            vec![0, 1, 2, 3, 1, 3, 1, 3, 0, 1, 2, 3],
            vec![6, 4, 2, 8, 1, 7, 2, 6, 4, 1, 6, 3],
        )
        .unwrap();

        let dense_a = DMatrix::from(&a);
        let dense_b = DMatrix::from(&b);

        let sum = DMatrix::from(&spadd_csr_csc(a, b).unwrap());
        let dense_sum = dense_a + dense_b;

        assert_eq!(sum, dense_sum);
    }

    #[test]
    fn spadd_csc_csr_agrees_with_dense() {
        let a = CscMatrix::try_from_parts(
            4,
            4,
            vec![0, 4, 7, 10],
            vec![0, 1, 2, 3, 0, 1, 3, 1, 2, 3, 0, 1, 3],
            vec![1, 2, 3, 4, -1, 2, 5, 4, -2, 6, 2, 4, 6],
        )
        .unwrap();

        let b = CsrMatrix::try_from_parts(
            4,
            4,
            vec![0, 4, 6, 8],
            vec![0, 1, 2, 3, 1, 3, 1, 3, 0, 1, 2, 3],
            vec![6, 4, 2, 8, 1, 7, 2, 6, 4, 1, 6, 3],
        )
        .unwrap();

        let dense_a = DMatrix::from(&a);
        let dense_b = DMatrix::from(&b);

        let sum = DMatrix::from(&spadd_csc_csr(a, b).unwrap());
        let dense_sum = dense_a + dense_b;

        assert_eq!(sum, dense_sum);
    }

    #[test]
    fn spadd_csc_csc_agrees_with_dense() {
        let a = CscMatrix::try_from_parts(
            4,
            4,
            vec![0, 4, 7, 10],
            vec![0, 1, 2, 3, 0, 1, 3, 1, 2, 3, 0, 1, 3],
            vec![1, 2, 3, 4, -1, 2, 5, 4, -2, 6, 2, 4, 6],
        )
        .unwrap();

        let b = CscMatrix::try_from_parts(
            4,
            4,
            vec![0, 4, 6, 8],
            vec![0, 1, 2, 3, 1, 3, 1, 3, 0, 1, 2, 3],
            vec![6, 4, 2, 8, 1, 7, 2, 6, 4, 1, 6, 3],
        )
        .unwrap();

        let dense_a = DMatrix::from(&a);
        let dense_b = DMatrix::from(&b);

        let sum = DMatrix::from(&spadd_csc_csc(a, b).unwrap());
        let dense_sum = dense_a + dense_b;

        assert_eq!(sum, dense_sum);
    }

    proptest! {
        #[test]
        fn spadd_csr_csr_additive_identity(matrix in csr_strategy()) {
            let (nrows, ncols) = matrix.shape();

            let zero = CsrMatrix::<i32>::zeros(nrows, ncols);

            let sum = spadd_csr_csr(matrix.to_view(), zero).unwrap();

            prop_assert_eq!(sum.shape(), matrix.shape());

            let (offsets, indices, data) = matrix.cs_data();
            let (expected_offsets, expected_indices, expected_data) = sum.cs_data();

            prop_assert!(offsets.iter().zip(expected_offsets).all(|(a, b)| a == b));
            prop_assert!(indices.iter().zip(expected_indices).all(|(a, b)| a == b));
            prop_assert!(data.iter().zip(expected_data).all(|(a, b)| a == b));
        }

        #[test]
        fn spadd_csr_csc_additive_identity(matrix in csr_strategy()) {
            let (nrows, ncols) = matrix.shape();

            let zero = CscMatrix::<i32>::zeros(nrows, ncols);

            let sum = spadd_csr_csc(matrix.to_view(), zero).unwrap();

            prop_assert_eq!(sum.shape(), matrix.shape());

            let (offsets, indices, data) = matrix.cs_data();
            let (expected_offsets, expected_indices, expected_data) = sum.cs_data();

            prop_assert!(offsets.iter().zip(expected_offsets).all(|(a, b)| a == b));
            prop_assert!(indices.iter().zip(expected_indices).all(|(a, b)| a == b));
            prop_assert!(data.iter().zip(expected_data).all(|(a, b)| a == b));
        }

        #[test]
        fn spadd_csc_csr_additive_identity(matrix in csc_strategy()) {
            let (nrows, ncols) = matrix.shape();

            let zero = CsrMatrix::<i32>::zeros(nrows, ncols);

            let sum = CscMatrix::from(spadd_csc_csr(matrix.to_view(), zero).unwrap());

            prop_assert_eq!(sum.shape(), matrix.shape());

            let (offsets, indices, data) = matrix.cs_data();
            let (expected_offsets, expected_indices, expected_data) = sum.cs_data();

            prop_assert!(offsets.iter().zip(expected_offsets).all(|(a, b)| a == b));
            prop_assert!(indices.iter().zip(expected_indices).all(|(a, b)| a == b));
            prop_assert!(data.iter().zip(expected_data).all(|(a, b)| a == b));
        }

        #[test]
        fn spadd_csc_csc_additive_identity(matrix in csc_strategy()) {
            let (nrows, ncols) = matrix.shape();

            let zero = CscMatrix::<i32>::zeros(nrows, ncols);

            let sum = spadd_csc_csc(matrix.to_view(), zero).unwrap();

            prop_assert_eq!(sum.shape(), matrix.shape());

            let (offsets, indices, data) = matrix.cs_data();
            let (expected_offsets, expected_indices, expected_data) = sum.cs_data();

            prop_assert!(offsets.iter().zip(expected_offsets).all(|(a, b)| a == b));
            prop_assert!(indices.iter().zip(expected_indices).all(|(a, b)| a == b));
            prop_assert!(data.iter().zip(expected_data).all(|(a, b)| a == b));
        }
    }
}
