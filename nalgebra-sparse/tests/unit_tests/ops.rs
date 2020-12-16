use crate::common::{csr_strategy, PROPTEST_MATRIX_DIM, PROPTEST_MAX_NNZ,
                    PROPTEST_I32_VALUE_STRATEGY};
use nalgebra_sparse::ops::serial::{spmm_csr_dense, spadd_build_pattern, spmm_pattern, spadd_csr, spmm_csr};
use nalgebra_sparse::ops::{Transpose};
use nalgebra_sparse::csr::CsrMatrix;
use nalgebra_sparse::proptest::{csr, sparsity_pattern};
use nalgebra_sparse::pattern::SparsityPattern;

use nalgebra::{DMatrix, Scalar, DMatrixSliceMut, DMatrixSlice};
use nalgebra::proptest::matrix;

use proptest::prelude::*;

use std::panic::catch_unwind;
use std::sync::Arc;

/// Represents the sparsity pattern of a CSR matrix as a dense matrix with 0/1
fn dense_csr_pattern(pattern: &SparsityPattern) -> DMatrix<i32> {
    let boolean_csr = CsrMatrix::try_from_pattern_and_values(
            Arc::new(pattern.clone()),
            vec![1; pattern.nnz()])
        .unwrap();
    DMatrix::from(&boolean_csr)
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
    let max_nnz = PROPTEST_MAX_NNZ;
    let value_strategy = PROPTEST_I32_VALUE_STRATEGY;
    let c_rows = PROPTEST_MATRIX_DIM;
    let c_cols = PROPTEST_MATRIX_DIM;
    let common_dim = PROPTEST_MATRIX_DIM;
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

#[derive(Debug)]
struct SpaddCsrArgs<T> {
    c: CsrMatrix<T>,
    beta: T,
    alpha: T,
    trans_a: Transpose,
    a: CsrMatrix<T>,
}

fn spadd_csr_args_strategy() -> impl Strategy<Value=SpaddCsrArgs<i32>> {
    let value_strategy = PROPTEST_I32_VALUE_STRATEGY;

    spadd_build_pattern_strategy()
        .prop_flat_map(move |(a_pattern, b_pattern)| {
            let mut c_pattern = SparsityPattern::new(a_pattern.major_dim(), b_pattern.major_dim());
            spadd_build_pattern(&mut c_pattern, &a_pattern, &b_pattern);

            let a_values = vec![value_strategy.clone(); a_pattern.nnz()];
            let c_values = vec![value_strategy.clone(); c_pattern.nnz()];
            let alpha = value_strategy.clone();
            let beta = value_strategy.clone();
            (Just(c_pattern), Just(a_pattern), c_values, a_values, alpha, beta, trans_strategy())
        }).prop_map(|(c_pattern, a_pattern, c_values, a_values, alpha, beta, trans_a)| {
            let c = CsrMatrix::try_from_pattern_and_values(Arc::new(c_pattern), c_values).unwrap();
            let a = CsrMatrix::try_from_pattern_and_values(Arc::new(a_pattern), a_values).unwrap();

            let a = if trans_a.to_bool() { a.transpose() } else { a };
            SpaddCsrArgs { c, beta, alpha, trans_a, a }
        })
}

fn dense_strategy() -> impl Strategy<Value=DMatrix<i32>> {
    matrix(PROPTEST_I32_VALUE_STRATEGY, PROPTEST_MATRIX_DIM, PROPTEST_MATRIX_DIM)
}

fn trans_strategy() -> impl Strategy<Value=Transpose> + Clone {
    proptest::bool::ANY.prop_map(Transpose)
}

fn pattern_strategy() -> impl Strategy<Value=SparsityPattern> {
    sparsity_pattern(PROPTEST_MATRIX_DIM, PROPTEST_MATRIX_DIM, PROPTEST_MAX_NNZ)
}

/// Constructs pairs (a, b) where a and b have the same dimensions
fn spadd_build_pattern_strategy() -> impl Strategy<Value=(SparsityPattern, SparsityPattern)> {
    pattern_strategy()
        .prop_flat_map(|a| {
            let b = sparsity_pattern(Just(a.major_dim()), Just(a.minor_dim()), PROPTEST_MAX_NNZ);
            (Just(a), b)
        })
}

/// Constructs pairs (a, b) where a and b have compatible dimensions for a matrix product
fn spmm_pattern_strategy() -> impl Strategy<Value=(SparsityPattern, SparsityPattern)> {
    pattern_strategy()
        .prop_flat_map(|a| {
            let b = sparsity_pattern(Just(a.minor_dim()), PROPTEST_MATRIX_DIM, PROPTEST_MAX_NNZ);
            (Just(a), b)
        })
}

#[derive(Debug)]
struct SpmmCsrArgs<T> {
    c: CsrMatrix<T>,
    beta: T,
    alpha: T,
    trans_a: Transpose,
    a: CsrMatrix<T>,
    trans_b: Transpose,
    b: CsrMatrix<T>
}

fn spmm_csr_args_strategy() -> impl Strategy<Value=SpmmCsrArgs<i32>> {
    spmm_pattern_strategy()
        .prop_flat_map(|(a_pattern, b_pattern)| {
            let a_values = vec![PROPTEST_I32_VALUE_STRATEGY; a_pattern.nnz()];
            let b_values = vec![PROPTEST_I32_VALUE_STRATEGY; b_pattern.nnz()];
            let c_pattern = spmm_pattern(&a_pattern, &b_pattern);
            let c_values = vec![PROPTEST_I32_VALUE_STRATEGY; c_pattern.nnz()];
            let a_pattern = Arc::new(a_pattern);
            let b_pattern = Arc::new(b_pattern);
            let c_pattern = Arc::new(c_pattern);
            let a = a_values.prop_map(move |values|
                CsrMatrix::try_from_pattern_and_values(Arc::clone(&a_pattern), values).unwrap());
            let b = b_values.prop_map(move |values|
                CsrMatrix::try_from_pattern_and_values(Arc::clone(&b_pattern), values).unwrap());
            let c = c_values.prop_map(move |values|
                CsrMatrix::try_from_pattern_and_values(Arc::clone(&c_pattern), values).unwrap());
            let alpha = PROPTEST_I32_VALUE_STRATEGY;
            let beta = PROPTEST_I32_VALUE_STRATEGY;
            (c, beta, alpha, trans_strategy(), a, trans_strategy(), b)
        })
        .prop_map(|(c, beta, alpha, trans_a, a, trans_b, b)| {
            SpmmCsrArgs::<i32> {
                c,
                beta,
                alpha,
                trans_a,
                a: if trans_a.to_bool() { a.transpose() } else { a },
                trans_b,
                b: if trans_b.to_bool() { b.transpose() } else { b }
            }
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

    #[test]
    fn spadd_csr_test(SpaddCsrArgs { c, beta, alpha, trans_a, a } in spadd_csr_args_strategy()) {
        // Test that we get the expected result by comparing to an equivalent dense operation
        // (here we give in the C matrix, so the sparsity pattern is essentially fixed)

        let mut c_sparse = c.clone();
        spadd_csr(&mut c_sparse, beta, alpha, trans_a, &a).unwrap();

        let mut c_dense = DMatrix::from(&c);
        let op_a_dense = DMatrix::from(&a);
        let op_a_dense = if trans_a.to_bool() { op_a_dense.transpose() } else { op_a_dense };
        c_dense = beta * c_dense + alpha * &op_a_dense;

        prop_assert_eq!(&DMatrix::from(&c_sparse), &c_dense);
    }

    #[test]
    fn csr_add_csr(
        // a and b have the same dimensions
        (a, b)
        in csr_strategy()
            .prop_flat_map(|a| {
                let b = csr(-5 ..= 5, Just(a.nrows()), Just(a.ncols()), 40);
                (Just(a), b)
            }))
    {
        // We use the dense result as the ground truth for the arithmetic result
        let c_dense = DMatrix::from(&a) + DMatrix::from(&b);
        // However, it's not enough only to cover the dense result, we also need to verify the
        // sparsity pattern. We can determine the exact sparsity pattern by using
        // dense arithmetic with positive integer values and extracting positive entries.
        let c_dense_pattern = dense_csr_pattern(a.pattern()) + dense_csr_pattern(b.pattern());
        let c_pattern = CsrMatrix::from(&c_dense_pattern).pattern().clone();

        // Check each combination of owned matrices and references
        let c_owned_owned = a.clone() + b.clone();
        prop_assert_eq!(&DMatrix::from(&c_owned_owned), &c_dense);
        prop_assert_eq!(c_owned_owned.pattern(), &c_pattern);

        let c_owned_ref = a.clone() + &b;
        prop_assert_eq!(&DMatrix::from(&c_owned_ref), &c_dense);
        prop_assert_eq!(c_owned_ref.pattern(), &c_pattern);

        let c_ref_owned = &a + b.clone();
        prop_assert_eq!(&DMatrix::from(&c_ref_owned), &c_dense);
        prop_assert_eq!(c_ref_owned.pattern(), &c_pattern);

        let c_ref_ref = &a + &b;
        prop_assert_eq!(&DMatrix::from(&c_ref_ref), &c_dense);
        prop_assert_eq!(c_ref_ref.pattern(), &c_pattern);
    }

    #[test]
    fn spmm_pattern_test((a, b) in spmm_pattern_strategy())
    {
        // (a, b) are multiplication-wise dimensionally compatible patterns
        let c_pattern = spmm_pattern(&a, &b);

        // To verify the pattern, we construct CSR matrices with positive integer entries
        // corresponding to a and b, and convert them to dense matrices.
        // The product of these dense matrices will then have non-zeros in exactly the same locations
        // as the result of "multiplying" the sparsity patterns
        let a_csr = CsrMatrix::try_from_pattern_and_values(Arc::new(a.clone()), vec![1; a.nnz()])
            .unwrap();
        let a_dense = DMatrix::from(&a_csr);
        let b_csr = CsrMatrix::try_from_pattern_and_values(Arc::new(b.clone()), vec![1; b.nnz()])
            .unwrap();
        let b_dense = DMatrix::from(&b_csr);
        let c_dense = a_dense * b_dense;
        let c_csr = CsrMatrix::from(&c_dense);

        prop_assert_eq!(&c_pattern, c_csr.pattern().as_ref());
    }

    #[test]
    fn spmm_csr_test(SpmmCsrArgs { c, beta, alpha, trans_a, a, trans_b, b }
        in spmm_csr_args_strategy()
    ) {
        // Test that we get the expected result by comparing to an equivalent dense operation
        // (here we give in the C matrix, so the sparsity pattern is essentially fixed)
        let mut c_sparse = c.clone();
        spmm_csr(&mut c_sparse, beta, alpha, trans_a, &a, trans_b, &b).unwrap();

        let mut c_dense = DMatrix::from(&c);
        let op_a_dense = DMatrix::from(&a);
        let op_a_dense = if trans_a.to_bool() { op_a_dense.transpose() } else { op_a_dense };
        let op_b_dense = DMatrix::from(&b);
        let op_b_dense = if trans_b.to_bool() { op_b_dense.transpose() } else { op_b_dense };
        c_dense = beta * c_dense + alpha * &op_a_dense * op_b_dense;

        prop_assert_eq!(&DMatrix::from(&c_sparse), &c_dense);
    }

    #[test]
    fn spmm_csr_panics_on_dim_mismatch(
        (alpha, beta, c, a, b, trans_a, trans_b)
        in (PROPTEST_I32_VALUE_STRATEGY,
            PROPTEST_I32_VALUE_STRATEGY,
            csr_strategy(),
            csr_strategy(),
            csr_strategy(),
            trans_strategy(),
            trans_strategy())
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
            spmm_csr(&mut spmm_result, beta, alpha, trans_a, &a, trans_b, &b).unwrap();
        });

        prop_assert!(result.is_err(),
            "The SPMM kernel executed successfully despite mismatch dimensions");
    }

    #[test]
    fn spadd_csr_panics_on_dim_mismatch(
        (alpha, beta, c, a, trans_a)
        in (PROPTEST_I32_VALUE_STRATEGY,
            PROPTEST_I32_VALUE_STRATEGY,
            csr_strategy(),
            csr_strategy(),
            trans_strategy())
    ) {
        let op_a_rows = if trans_a.to_bool() { a.ncols() } else { a.nrows() };
        let op_a_cols = if trans_a.to_bool() { a.nrows() } else { a.ncols() };

        let dims_are_compatible = c.nrows() == op_a_rows && c.ncols() == op_a_cols;

        // If the dimensions randomly happen to be compatible, then of course we need to
        // skip the test, so we assume that they are not.
        prop_assume!(!dims_are_compatible);

        let result = catch_unwind(|| {
            let mut spmm_result = c.clone();
            spadd_csr(&mut spmm_result, beta, alpha, trans_a, &a).unwrap();
        });

        prop_assert!(result.is_err(),
            "The SPMM kernel executed successfully despite mismatch dimensions");
    }
}