//! Module holding the various direct sparse-matrix solvers.
//!
//! A direct solver is one that does not utilize a matrix decomposition in order to solve an
//! equation of the form `A x = B` for some problem where we have known `A` and `B`. Examples of
//! this may be:
//!
//! 1. Lower triangular solvers (where `A` is lower-triangular).
//! 2. Upper triangular solvers (where `A` is upper-triangular).
//!
//! Here we only provide lower triangular and upper triangular solvers. This kind of solver is
//! typically the fastest and most numerically stable form of solver available. For matrices that
//! are not lower-triangular or upper-triangular, it is first recommended to perform a matrix
//! factorization or decomposition of some kind. See the [factorization](crate::factorization)
//! module for a look at what factorizations are provided by this crate.

use crate::{
    cs::{CompressedColumnStorage, CompressedRowStorage, CsMatrix},
    error::{OperationError, OperationErrorKind},
};
use nalgebra::{Dim, Matrix, RawStorage, RawStorageMut, RealField};
use std::{borrow::Borrow, cmp::Ordering};

/// Sparse-dense matrix solver for lower-triangular CSC matrices and a dense right hand side.
///
/// Solves the system `A x = B` where:
///
/// - `A` is a square, CSC matrix that is lower-triangular.
/// - `B` is a dense matrix that has a number of rows equal to the dimensions of `A`.
///
/// NOTE: If `A` is not actually lower-triangular, this function will ignore values on the upper
/// portion of the matrix.
///
/// # Errors
///
/// Returns an [`OperationError`] with kind `OperationErrorKind::InvalidPattern` if `A` is not
/// square, or if `B` has an invalid number of rows.
///
/// Returns an [`OperationError`] with kind `OperationErrorKind::Singular` if `A` contains zero
/// values along the diagonal (i.e. the matrix is singular an non-invertible).
pub fn spsolve_lower_triangular_csc_dense<T, R, C, S, MO, MI, D>(
    csc: CsMatrix<T, MO, MI, D, CompressedColumnStorage>,
    mut dense: Matrix<T, R, C, S>,
) -> Result<Matrix<T, R, C, S>, OperationError>
where
    T: RealField,
    R: Dim,
    C: Dim,
    S: RawStorage<T, R, C> + RawStorageMut<T, R, C>,
    MO: Borrow<[usize]>,
    MI: Borrow<[usize]>,
    D: Borrow<[T]>,
{
    let (nrows, ncols) = csc.shape();

    if nrows != ncols {
        return Err(OperationError::from_kind_and_message(
            OperationErrorKind::InvalidPattern,
            String::from("Lefthand matrix is not square."),
        ));
    }

    let (dense_rows, dense_cols) = dense.shape();

    if dense_rows != ncols {
        return Err(OperationError::from_kind_and_message(
            OperationErrorKind::InvalidPattern,
            format!(
                "The righthand matrix has {} rows but {} rows are needed to solve this system.",
                dense_rows, ncols
            ),
        ));
    }

    for j in 0..dense_cols {
        let mut dense_column = dense.column_mut(j);

        for (k, mut lane) in csc.iter().enumerate() {
            let mut diag = None;

            // We first get the diagonal value, ignoring everything above it.
            while let Some((i, val)) = lane.next() {
                match i.cmp(&k) {
                    Ordering::Less => {
                        // Ignore every value above the diagonal
                        continue;
                    }

                    Ordering::Equal => {
                        // We found the diagonal, break out.
                        diag = Some(val.clone());
                        break;
                    }

                    Ordering::Greater => {
                        // The diagonal doesn't exist. Since we know the pattern is sorted, and
                        // we're greater than the index we're searching for, we know that the
                        // diagonal was not stored (i.e. it is zero).
                        break;
                    }
                }
            }

            if let Some(a_kk) = diag {
                dense_column[k] /= a_kk;

                let x_kj = dense_column[k].clone();

                for (i, l_ik) in lane {
                    let x_ij = &mut dense_column[i];
                    *x_ij -= l_ik.clone() * x_kj.clone();
                }

                dense_column[k] = x_kj;
            } else {
                // If the diagonal was never found (was None) then that can only mean the matrix is
                // singular.
                return Err(OperationError::from_kind_and_message(
                    OperationErrorKind::Singular,
                    String::from("Matrix contains at least one diagonal entry that is zero."),
                ));
            }
        }
    }

    Ok(dense)
}

/// Sparse-dense matrix solver for upper-triangular CSR matrices and a dense right hand side.
///
/// Solves the system `A x = B` where:
///
/// - `A` is a square, CSR matrix that is upper-triangular.
/// - `B` is a dense matrix that has a number of rows equal to the dimensions of `A`.
///
/// NOTE: If `A` is not actually upper-triangular, this function will ignore values on the lower
/// portion of the matrix.
///
/// # Errors
///
/// Returns an [`OperationError`] with kind `OperationErrorKind::InvalidPattern` if `A` is not
/// square, or if `B` has an invalid number of rows.
///
/// Returns an [`OperationError`] with kind `OperationErrorKind::Singular` if `A` contains zero
/// values along the diagonal (i.e. the matrix is singular an non-invertible).
pub fn spsolve_upper_triangular_csr_dense<T, R, C, S, MO, MI, D>(
    csr: CsMatrix<T, MO, MI, D, CompressedRowStorage>,
    mut dense: Matrix<T, R, C, S>,
) -> Result<Matrix<T, R, C, S>, OperationError>
where
    T: RealField,
    R: Dim,
    C: Dim,
    S: RawStorage<T, R, C> + RawStorageMut<T, R, C>,
    MO: Borrow<[usize]>,
    MI: Borrow<[usize]>,
    D: Borrow<[T]>,
{
    let (nrows, ncols) = csr.shape();

    if nrows != ncols {
        return Err(OperationError::from_kind_and_message(
            OperationErrorKind::InvalidPattern,
            String::from("Lefthand matrix is not square."),
        ));
    }

    let (dense_rows, dense_cols) = dense.shape();

    if dense_rows != ncols {
        return Err(OperationError::from_kind_and_message(
            OperationErrorKind::InvalidPattern,
            format!(
                "The righthand matrix has {} rows but {} rows are needed to solve this system.",
                dense_rows, ncols
            ),
        ));
    }

    for j in (0..dense_cols).rev() {
        let mut dense_column = dense.column_mut(j);

        for (k, mut lane) in csr.iter().enumerate().rev() {
            let mut diag = None;

            // We first get the diagonal value, ignoring everything below it.
            while let Some((i, val)) = lane.next() {
                match i.cmp(&k) {
                    Ordering::Less => {
                        // Ignore every value below the diagonal
                        continue;
                    }

                    Ordering::Equal => {
                        // We found the diagonal, break out.
                        diag = Some(val.clone());
                        break;
                    }

                    Ordering::Greater => {
                        // The diagonal doesn't exist. Since we know the pattern is sorted, and
                        // we're greater than the index we're searching for, we know that the
                        // diagonal was not stored (i.e. it is zero).
                        break;
                    }
                }
            }

            if let Some(a_kk) = diag {
                let mut x_kk = dense_column[k].clone();

                for (i, a_ki) in lane {
                    x_kk -= a_ki.clone() * dense_column[i].clone();
                }

                dense_column[k] = x_kk / a_kk;
            } else {
                // If the diagonal was never found (was None) then that can only mean the matrix is
                // singular.
                return Err(OperationError::from_kind_and_message(
                    OperationErrorKind::Singular,
                    String::from("Matrix contains at least one diagonal entry that is zero."),
                ));
            }
        }
    }

    Ok(dense)
}
