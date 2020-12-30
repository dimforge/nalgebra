use crate::csc::CscMatrix;
use crate::ops::Op;
use crate::ops::serial::cs::{spmm_cs_prealloc, spmm_cs_dense, spadd_cs_prealloc};
use crate::ops::serial::OperationError;
use nalgebra::{Scalar, ClosedAdd, ClosedMul, DMatrixSliceMut, DMatrixSlice};
use num_traits::{Zero, One};

use std::borrow::Cow;

/// Sparse-dense matrix-matrix multiplication `C <- beta * C + alpha * op(A) * op(B)`.
pub fn spmm_csc_dense<'a, T>(beta: T,
                             c: impl Into<DMatrixSliceMut<'a, T>>,
                             alpha: T,
                             a: Op<&CscMatrix<T>>,
                             b: Op<impl Into<DMatrixSlice<'a, T>>>)
    where
        T: Scalar + ClosedAdd + ClosedMul + Zero + One
{
    let b = b.convert();
    spmm_csc_dense_(beta, c.into(), alpha, a, b)
}

fn spmm_csc_dense_<T>(beta: T,
                      c: DMatrixSliceMut<T>,
                      alpha: T,
                      a: Op<&CscMatrix<T>>,
                      b: Op<DMatrixSlice<T>>)
    where
        T: Scalar + ClosedAdd + ClosedMul + Zero + One
{
    assert_compatible_spmm_dims!(c, a, b);
    // Need to interpret matrix as transposed since the spmm_cs_dense function assumes CSR layout
    let a = a.transposed().map_same_op(|a| &a.cs);
    spmm_cs_dense(beta, c, alpha, a, b)
}

/// Sparse matrix addition `C <- beta * C + alpha * op(A)`.
///
/// If the pattern of `c` does not accommodate all the non-zero entries in `a`, an error is
/// returned.
pub fn spadd_csc_prealloc<T>(beta: T,
                             c: &mut CscMatrix<T>,
                             alpha: T,
                             a: Op<&CscMatrix<T>>)
                             -> Result<(), OperationError>
    where
        T: Scalar + ClosedAdd + ClosedMul + Zero + One
{
    assert_compatible_spadd_dims!(c, a);
    spadd_cs_prealloc(beta, &mut c.cs, alpha, a.map_same_op(|a| &a.cs))
}


/// Sparse-sparse matrix multiplication, `C <- beta * C + alpha * op(A) * op(B)`.
pub fn spmm_csc_prealloc<T>(
    beta: T,
    c: &mut CscMatrix<T>,
    alpha: T,
    a: Op<&CscMatrix<T>>,
    b: Op<&CscMatrix<T>>)
    -> Result<(), OperationError>
    where
        T: Scalar + ClosedAdd + ClosedMul + Zero + One
{
    assert_compatible_spmm_dims!(c, a, b);

    use Op::{NoOp, Transpose};

    match (&a, &b) {
        (NoOp(ref a), NoOp(ref b)) => {
            // Note: We have to reverse the order for CSC matrices
            spmm_cs_prealloc(beta, &mut c.cs, alpha, &b.cs, &a.cs)
        },
        _ => {
            // Currently we handle transposition by explicitly precomputing transposed matrices
            // and calling the operation again without transposition
            let a_ref: &CscMatrix<T> = a.inner_ref();
            let b_ref: &CscMatrix<T> = b.inner_ref();
            let (a, b) = {
                use Cow::*;
                match (&a, &b) {
                    (NoOp(_), NoOp(_)) => unreachable!(),
                    (Transpose(ref a), NoOp(_)) => (Owned(a.transpose()), Borrowed(b_ref)),
                    (NoOp(_), Transpose(ref b)) => (Borrowed(a_ref), Owned(b.transpose())),
                    (Transpose(ref a), Transpose(ref b)) => (Owned(a.transpose()), Owned(b.transpose()))
                }
            };

            spmm_csc_prealloc(beta, c, alpha, NoOp(a.as_ref()), NoOp(b.as_ref()))
        }
    }
}