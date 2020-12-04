use crate::csr::CsrMatrix;
use crate::ops::{Transpose};
use nalgebra::{Scalar, DMatrixSlice, ClosedAdd, ClosedMul, DMatrixSliceMut};
use num_traits::{Zero, One};

/// Sparse-dense matrix-matrix multiplication `C <- beta * C + alpha * trans(A) * trans(B)`.
pub fn spmm_csr_dense<'a, T>(c: impl Into<DMatrixSliceMut<'a, T>>,
                             beta: T,
                             alpha: T,
                             trans_a: Transpose,
                             a: &CsrMatrix<T>,
                             trans_b: Transpose,
                             b: impl Into<DMatrixSlice<'a, T>>)
    where
        T: Scalar + ClosedAdd + ClosedMul + Zero + One
{
    spmm_csr_dense_(c.into(), beta, alpha, trans_a, a, trans_b, b.into())
}

fn spmm_csr_dense_<T>(mut c: DMatrixSliceMut<T>,
                      beta: T,
                      alpha: T,
                      trans_a: Transpose,
                      a: &CsrMatrix<T>,
                      trans_b: Transpose,
                      b: DMatrixSlice<T>)
where
    T: Scalar + ClosedAdd + ClosedMul + Zero + One
{
    assert_compatible_spmm_dims!(c, a, b, trans_a, trans_b);

    if trans_a.to_bool() {
        // In this case, we have to pre-multiply C by beta
        c *= beta;

        for k in 0..a.nrows() {
            let a_row_k = a.row(k);
            for (&i, a_ki) in a_row_k.col_indices().iter().zip(a_row_k.values()) {
                let gamma_ki = alpha.inlined_clone() * a_ki.inlined_clone();
                let mut c_row_i = c.row_mut(i);
                if trans_b.to_bool() {
                    let b_col_k = b.column(k);
                    for (c_ij, b_jk) in c_row_i.iter_mut().zip(b_col_k.iter()) {
                        *c_ij += gamma_ki.inlined_clone() * b_jk.inlined_clone();
                    }
                } else {
                    let b_row_k = b.row(k);
                    for (c_ij, b_kj) in c_row_i.iter_mut().zip(b_row_k.iter()) {
                        *c_ij += gamma_ki.inlined_clone() * b_kj.inlined_clone();
                    }
                }
            }
        }
    } else {
        for j in 0..c.ncols() {
            let mut c_col_j = c.column_mut(j);
            for (c_ij, a_row_i) in c_col_j.iter_mut().zip(a.row_iter()) {
                let mut dot_ij = T::zero();
                for (&k, a_ik) in a_row_i.col_indices().iter().zip(a_row_i.values()) {
                    let b_contrib =
                        if trans_b.to_bool() { b.index((j, k)) } else { b.index((k, j)) };
                    dot_ij += a_ik.inlined_clone() * b_contrib.inlined_clone();
                }
                *c_ij = beta.inlined_clone() * c_ij.inlined_clone() + alpha.inlined_clone() * dot_ij;
            }
        }
    }
}