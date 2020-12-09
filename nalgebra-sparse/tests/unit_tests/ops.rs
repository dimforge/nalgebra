use nalgebra_sparse::coo::CooMatrix;
use nalgebra_sparse::ops::serial::{spmv_coo, spmm_csr_dense, spadd_build_pattern};
use nalgebra_sparse::ops::{Transpose};
use nalgebra_sparse::csr::CsrMatrix;
use nalgebra_sparse::proptest::{csr, sparsity_pattern};
use nalgebra_sparse::pattern::SparsityPattern;

use nalgebra::{DVector, DMatrix, Scalar, DMatrixSliceMut, DMatrixSlice};
use nalgebra::proptest::matrix;

use proptest::prelude::*;

use std::panic::catch_unwind;
use std::sync::Arc;

#[test]
fn spmv_coo_agrees_with_dense_gemv() {
    let x = DVector::from_column_slice(&[2, 3, 4, 5]);

    let i = vec![0, 0, 1, 1, 2, 2];
    let j = vec![0, 3, 0, 1, 1, 3];
    let v = vec![3, 2, 1, 2, 3, 1];
    let a = CooMatrix::try_from_triplets(3, 4, i, j, v).unwrap();

    let betas = [0, 1, 2];
    let alphas = [0, 1, 2];

    for &beta in &betas {
        for &alpha in &alphas {
            let mut y = DVector::from_column_slice(&[2, 5, 3]);
            let mut y_dense = y.clone();
            spmv_coo(beta, &mut y, alpha, &a, &x);

            y_dense.gemv(alpha, &DMatrix::from(&a), &x, beta);

            assert_eq!(y, y_dense);
        }
    }
}

#[derive(Debug)]
struct SpmmCsrDenseArgs<T: Scalar> {
    c: DMatrix<T>,
    beta: T,
    alpha: T,
    trans_a: Transpose,
    a: CsrMatrix<T>,
    trans_b: Transpose,
    b: DMatrix<T>,
}

/// Returns matrices C, A and B with compatible dimensions such that it can be used
/// in an `spmm` operation `C = beta * C + alpha * trans(A) * trans(B)`.
fn spmm_csr_dense_args_strategy() -> impl Strategy<Value=SpmmCsrDenseArgs<i32>> {
    let max_nnz = 40;
    let value_strategy = -5 ..= 5;
    let c_rows = 0 ..= 6usize;
    let c_cols = 0 ..= 6usize;
    let common_dim = 0 ..= 6usize;
    let trans_strategy = trans_strategy();
    let c_matrix_strategy = matrix(value_strategy.clone(), c_rows, c_cols);

    (c_matrix_strategy, common_dim, trans_strategy.clone(), trans_strategy.clone())
        .prop_flat_map(move |(c, common_dim, trans_a, trans_b)| {
            let a_shape =
                if trans_a.to_bool() { (common_dim, c.nrows()) }
                else { (c.nrows(), common_dim) };
            let b_shape =
                if trans_b.to_bool() { (c.ncols(), common_dim) }
                else { (common_dim, c.ncols()) };
            let a = csr(value_strategy.clone(), Just(a_shape.0), Just(a_shape.1), max_nnz);
            let b = matrix(value_strategy.clone(), b_shape.0, b_shape.1);

            // We use the same values for alpha, beta parameters as for matrix elements
            let alpha = value_strategy.clone();
            let beta = value_strategy.clone();

            (Just(c), beta, alpha, Just(trans_a), a, Just(trans_b), b)
        }).prop_map(|(c, beta, alpha, trans_a, a, trans_b, b)| {
            SpmmCsrDenseArgs {
                c,
                beta,
                alpha,
                trans_a,
                a,
                trans_b,
                b,
            }
        })
}

fn csr_strategy() -> impl Strategy<Value=CsrMatrix<i32>> {
    csr(-5 ..= 5, 0 ..= 6usize, 0 ..= 6usize, 40)
}

fn dense_strategy() -> impl Strategy<Value=DMatrix<i32>> {
    matrix(-5 ..= 5, 0 ..= 6, 0 ..= 6)
}

fn trans_strategy() -> impl Strategy<Value=Transpose> + Clone {
    proptest::bool::ANY.prop_map(Transpose)
}

fn pattern_strategy() -> impl Strategy<Value=SparsityPattern> {
    sparsity_pattern(0 ..= 6usize, 0..= 6usize, 40)
}

/// Constructs pairs (a, b) where a and b have the same dimensions
fn spadd_build_pattern_strategy() -> impl Strategy<Value=(SparsityPattern, SparsityPattern)> {
    pattern_strategy()
        .prop_flat_map(|a| {
            let b = sparsity_pattern(Just(a.major_dim()), Just(a.minor_dim()), 40);
            (Just(a), b)
        })
}

/// Helper function to help us call dense GEMM with our transposition parameters
fn dense_gemm<'a>(c: impl Into<DMatrixSliceMut<'a, i32>>,
                  beta: i32,
                  alpha: i32,
                  trans_a: Transpose,
                  a: impl Into<DMatrixSlice<'a, i32>>,
                  trans_b: Transpose,
                  b: impl Into<DMatrixSlice<'a, i32>>)
{
    let mut c = c.into();
    let a = a.into();
    let b = b.into();

    match (trans_a, trans_b) {
        (Transpose(false), Transpose(false)) => c.gemm(alpha, &a, &b, beta),
        (Transpose(true), Transpose(false)) => c.gemm(alpha, &a.transpose(), &b, beta),
        (Transpose(false), Transpose(true)) => c.gemm(alpha, &a, &b.transpose(), beta),
        (Transpose(true), Transpose(true)) => c.gemm(alpha, &a.transpose(), &b.transpose(), beta)
    };
}

proptest! {

    #[test]
    fn spmm_csr_dense_agrees_with_dense_result(
        SpmmCsrDenseArgs { c, beta, alpha, trans_a, a, trans_b, b }
         in spmm_csr_dense_args_strategy()
    ) {
        let mut spmm_result = c.clone();
        spmm_csr_dense(&mut spmm_result, beta, alpha, trans_a, &a, trans_b, &b);

        let mut gemm_result = c.clone();
        dense_gemm(&mut gemm_result, beta, alpha, trans_a, &DMatrix::from(&a), trans_b, &b);

        prop_assert_eq!(spmm_result, gemm_result);
    }

    #[test]
    fn spmm_csr_dense_panics_on_dim_mismatch(
        (alpha, beta, c, a, b, trans_a, trans_b)
        in (-5 ..= 5, -5 ..= 5, dense_strategy(), csr_strategy(),
            dense_strategy(), trans_strategy(), trans_strategy())
    ) {
        // We refer to `A * B` as the "product"
        let product_rows = if trans_a.to_bool() { a.ncols() } else { a.nrows() };
        let product_cols = if trans_b.to_bool() { b.nrows() } else { b.ncols() };
        // Determine the common dimension in the product
        // from the perspective of a and b, respectively
        let product_a_common = if trans_a.to_bool() { a.nrows() } else { a.ncols() };
        let product_b_common = if trans_b.to_bool() { b.ncols() } else { b.nrows() };

        let dims_are_compatible = product_rows == c.nrows()
            && product_cols == c.ncols()
            && product_a_common == product_b_common;

        // If the dimensions randomly happen to be compatible, then of course we need to
        // skip the test, so we assume that they are not.
        prop_assume!(!dims_are_compatible);

        let result = catch_unwind(|| {
            let mut spmm_result = c.clone();
            spmm_csr_dense(&mut spmm_result, beta, alpha, trans_a, &a, trans_b, &b);
        });

        prop_assert!(result.is_err(),
            "The SPMM kernel executed successfully despite mismatch dimensions");
    }

    #[test]
    fn spadd_build_pattern_test((c, (a, b)) in (pattern_strategy(), spadd_build_pattern_strategy()))
    {
        // (a, b) are dimensionally compatible patterns, whereas c is an *arbitrary* pattern
        let mut pattern_result = c.clone();
        spadd_build_pattern(&mut pattern_result, &a, &b);

        // To verify the pattern, we construct CSR matrices with positive integer entries
        // corresponding to a and b, and convert them to dense matrices.
        // The sum of these dense matrices will then have non-zeros in exactly the same locations
        // as the result of "adding" the sparsity patterns
        let a_csr = CsrMatrix::try_from_pattern_and_values(Arc::new(a.clone()), vec![1; a.nnz()])
            .unwrap();
        let a_dense = DMatrix::from(&a_csr);
        let b_csr = CsrMatrix::try_from_pattern_and_values(Arc::new(b.clone()), vec![1; b.nnz()])
            .unwrap();
        let b_dense = DMatrix::from(&b_csr);
        let c_dense = a_dense + b_dense;
        let c_csr = CsrMatrix::from(&c_dense);

        prop_assert_eq!(&pattern_result, c_csr.pattern().as_ref());
    }
}