use crate::common::{csr_strategy, PROPTEST_MATRIX_DIM, PROPTEST_MAX_NNZ,
                    PROPTEST_I32_VALUE_STRATEGY};
use nalgebra_sparse::ops::serial::{spmm_csr_dense, spadd_pattern, spmm_pattern, spadd_csr, spmm_csr};
use nalgebra_sparse::ops::{Op};
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
    a: Op<CsrMatrix<T>>,
    b: Op<DMatrix<T>>,
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
                if trans_a { (common_dim, c.nrows()) }
                else { (c.nrows(), common_dim) };
            let b_shape =
                if trans_b { (c.ncols(), common_dim) }
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
                a: if trans_a { Op::Transpose(a) } else { Op::NoOp(a) },
                b: if trans_b { Op::Transpose(b) } else { Op::NoOp(b) },
            }
        })
}

#[derive(Debug)]
struct SpaddCsrArgs<T> {
    c: CsrMatrix<T>,
    beta: T,
    alpha: T,
    a: Op<CsrMatrix<T>>,
}

fn spadd_csr_args_strategy() -> impl Strategy<Value=SpaddCsrArgs<i32>> {
    let value_strategy = PROPTEST_I32_VALUE_STRATEGY;

    spadd_pattern_strategy()
        .prop_flat_map(move |(a_pattern, b_pattern)| {
            let c_pattern = spadd_pattern(&a_pattern, &b_pattern);

            let a_values = vec![value_strategy.clone(); a_pattern.nnz()];
            let c_values = vec![value_strategy.clone(); c_pattern.nnz()];
            let alpha = value_strategy.clone();
            let beta = value_strategy.clone();
            (Just(c_pattern), Just(a_pattern), c_values, a_values, alpha, beta, trans_strategy())
        }).prop_map(|(c_pattern, a_pattern, c_values, a_values, alpha, beta, trans_a)| {
            let c = CsrMatrix::try_from_pattern_and_values(Arc::new(c_pattern), c_values).unwrap();
            let a = CsrMatrix::try_from_pattern_and_values(Arc::new(a_pattern), a_values).unwrap();

            let a = if trans_a { Op::Transpose(a.transpose()) } else { Op::NoOp(a) };
            SpaddCsrArgs { c, beta, alpha, a }
        })
}

fn dense_strategy() -> impl Strategy<Value=DMatrix<i32>> {
    matrix(PROPTEST_I32_VALUE_STRATEGY, PROPTEST_MATRIX_DIM, PROPTEST_MATRIX_DIM)
}

fn trans_strategy() -> impl Strategy<Value=bool> + Clone {
    proptest::bool::ANY
}

/// Wraps the values of the given strategy in `Op`, producing both transposed and non-transposed
/// values.
fn op_strategy<S: Strategy>(strategy: S) -> impl Strategy<Value=Op<S::Value>> {
    let is_transposed = proptest::bool::ANY;
    (strategy, is_transposed)
        .prop_map(|(obj, is_trans)| if is_trans {
            Op::Transpose(obj)
        } else {
            Op::NoOp(obj)
        })
}

fn pattern_strategy() -> impl Strategy<Value=SparsityPattern> {
    sparsity_pattern(PROPTEST_MATRIX_DIM, PROPTEST_MATRIX_DIM, PROPTEST_MAX_NNZ)
}

/// Constructs pairs (a, b) where a and b have the same dimensions
fn spadd_pattern_strategy() -> impl Strategy<Value=(SparsityPattern, SparsityPattern)> {
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
    a: Op<CsrMatrix<T>>,
    b: Op<CsrMatrix<T>>,
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
                a: if trans_a { Op::Transpose(a.transpose()) } else { Op::NoOp(a) },
                b: if trans_b { Op::Transpose(b.transpose()) } else { Op::NoOp(b) }
            }
        })
}

/// Helper function to help us call dense GEMM with our `Op` type
fn dense_gemm<'a>(beta: i32,
                  c: impl Into<DMatrixSliceMut<'a, i32>>,
                  alpha: i32,
                  a: Op<impl Into<DMatrixSlice<'a, i32>>>,
                  b: Op<impl Into<DMatrixSlice<'a, i32>>>)
{
    let mut c = c.into();
    let a = a.convert();
    let b = b.convert();

    use Op::{NoOp, Transpose};
    match (a, b) {
        (NoOp(a), NoOp(b)) => c.gemm(alpha, &a, &b, beta),
        (Transpose(a), NoOp(b)) => c.gemm(alpha, &a.transpose(), &b, beta),
        (NoOp(a), Transpose(b)) => c.gemm(alpha, &a, &b.transpose(), beta),
        (Transpose(a), Transpose(b)) => c.gemm(alpha, &a.transpose(), &b.transpose(), beta)
    }
}

proptest! {

    #[test]
    fn spmm_csr_dense_agrees_with_dense_result(
        SpmmCsrDenseArgs { c, beta, alpha, a, b }
         in spmm_csr_dense_args_strategy()
    ) {
        let mut spmm_result = c.clone();
        spmm_csr_dense(beta, &mut spmm_result, alpha, a.as_ref(), b.as_ref());

        let mut gemm_result = c.clone();
        let a_dense = a.map_same_op(|a| DMatrix::from(&a));
        dense_gemm(beta, &mut gemm_result, alpha, a_dense.as_ref(), b.as_ref());

        prop_assert_eq!(spmm_result, gemm_result);
    }

    #[test]
    fn spmm_csr_dense_panics_on_dim_mismatch(
        (alpha, beta, c, a, b)
        in (PROPTEST_I32_VALUE_STRATEGY,
            PROPTEST_I32_VALUE_STRATEGY,
            dense_strategy(),
            op_strategy(csr_strategy()),
            op_strategy(dense_strategy()))
    ) {
        // We refer to `A * B` as the "product"
        let product_rows = match &a {
            Op::NoOp(ref a) => a.nrows(),
            Op::Transpose(ref a) => a.ncols(),
        };
        let product_cols = match &b {
            Op::NoOp(ref b) => b.ncols(),
            Op::Transpose(ref b) => b.nrows(),
        };
        // Determine the common dimension in the product
        // from the perspective of a and b, respectively
        let product_a_common = match &a {
            Op::NoOp(ref a) => a.ncols(),
            Op::Transpose(ref a) => a.nrows(),
        };
        let product_b_common = match &b {
            Op::NoOp(ref b) => b.nrows(),
            Op::Transpose(ref b) => b.ncols()
        };

        let dims_are_compatible = product_rows == c.nrows()
            && product_cols == c.ncols()
            && product_a_common == product_b_common;

        // If the dimensions randomly happen to be compatible, then of course we need to
        // skip the test, so we assume that they are not.
        prop_assume!(!dims_are_compatible);

        let result = catch_unwind(|| {
            let mut spmm_result = c.clone();
            spmm_csr_dense(beta, &mut spmm_result, alpha, a.as_ref(), b.as_ref());
        });

        prop_assert!(result.is_err(),
            "The SPMM kernel executed successfully despite mismatch dimensions");
    }

    #[test]
    fn spadd_pattern_test((a, b) in spadd_pattern_strategy())
    {
        // (a, b) are dimensionally compatible patterns
        let pattern_result = spadd_pattern(&a, &b);

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
    fn spadd_csr_test(SpaddCsrArgs { c, beta, alpha, a } in spadd_csr_args_strategy()) {
        // Test that we get the expected result by comparing to an equivalent dense operation
        // (here we give in the C matrix, so the sparsity pattern is essentially fixed)

        let mut c_sparse = c.clone();
        spadd_csr(beta, &mut c_sparse, alpha, a.as_ref()).unwrap();

        let mut c_dense = DMatrix::from(&c);
        let op_a_dense = match a {
            Op::NoOp(a) => DMatrix::from(&a),
            Op::Transpose(a) => DMatrix::from(&a).transpose(),
        };
        c_dense = beta * c_dense + alpha * &op_a_dense;

        prop_assert_eq!(&DMatrix::from(&c_sparse), &c_dense);
    }

    #[test]
    fn csr_add_csr(
        // a and b have the same dimensions
        (a, b)
        in csr_strategy()
            .prop_flat_map(|a| {
                let b = csr(PROPTEST_I32_VALUE_STRATEGY, Just(a.nrows()), Just(a.ncols()), 40);
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
    fn spmm_csr_test(SpmmCsrArgs { c, beta, alpha, a, b }
        in spmm_csr_args_strategy()
    ) {
        // Test that we get the expected result by comparing to an equivalent dense operation
        // (here we give in the C matrix, so the sparsity pattern is essentially fixed)
        let mut c_sparse = c.clone();
        spmm_csr(beta, &mut c_sparse, alpha, a.as_ref(), b.as_ref()).unwrap();

        let mut c_dense = DMatrix::from(&c);
        let op_a_dense = match a {
            Op::NoOp(ref a) => DMatrix::from(a),
            Op::Transpose(ref a) => DMatrix::from(a).transpose(),
        };
        let op_b_dense = match b {
            Op::NoOp(ref b) => DMatrix::from(b),
            Op::Transpose(ref b) => DMatrix::from(b).transpose(),
        };
        c_dense = beta * c_dense + alpha * &op_a_dense * op_b_dense;

        prop_assert_eq!(&DMatrix::from(&c_sparse), &c_dense);
    }

    #[test]
    fn spmm_csr_panics_on_dim_mismatch(
        (alpha, beta, c, a, b)
        in (PROPTEST_I32_VALUE_STRATEGY,
            PROPTEST_I32_VALUE_STRATEGY,
            csr_strategy(),
            op_strategy(csr_strategy()),
            op_strategy(csr_strategy()))
    ) {
        // We refer to `A * B` as the "product"
        let product_rows = match &a {
            Op::NoOp(ref a) => a.nrows(),
            Op::Transpose(ref a) => a.ncols(),
        };
        let product_cols = match &b {
            Op::NoOp(ref b) => b.ncols(),
            Op::Transpose(ref b) => b.nrows(),
        };
        // Determine the common dimension in the product
        // from the perspective of a and b, respectively
        let product_a_common = match &a {
            Op::NoOp(ref a) => a.ncols(),
            Op::Transpose(ref a) => a.nrows(),
        };
        let product_b_common = match &b {
            Op::NoOp(ref b) => b.nrows(),
            Op::Transpose(ref b) => b.ncols(),
        };

        let dims_are_compatible = product_rows == c.nrows()
            && product_cols == c.ncols()
            && product_a_common == product_b_common;

        // If the dimensions randomly happen to be compatible, then of course we need to
        // skip the test, so we assume that they are not.
        prop_assume!(!dims_are_compatible);

        let result = catch_unwind(|| {
            let mut spmm_result = c.clone();
            spmm_csr(beta, &mut spmm_result, alpha, a.as_ref(), b.as_ref()).unwrap();
        });

        prop_assert!(result.is_err(),
            "The SPMM kernel executed successfully despite mismatch dimensions");
    }

    #[test]
    fn spadd_csr_panics_on_dim_mismatch(
        (alpha, beta, c, op_a)
        in (PROPTEST_I32_VALUE_STRATEGY,
            PROPTEST_I32_VALUE_STRATEGY,
            csr_strategy(),
            op_strategy(csr_strategy()))
    ) {
        let op_a_rows = match &op_a {
            &Op::NoOp(ref a) => a.nrows(),
            &Op::Transpose(ref a) => a.ncols()
        };
        let op_a_cols = match &op_a {
            &Op::NoOp(ref a) => a.ncols(),
            &Op::Transpose(ref a) => a.nrows()
        };

        let dims_are_compatible = c.nrows() == op_a_rows && c.ncols() == op_a_cols;

        // If the dimensions randomly happen to be compatible, then of course we need to
        // skip the test, so we assume that they are not.
        prop_assume!(!dims_are_compatible);

        let result = catch_unwind(|| {
            let mut spmm_result = c.clone();
            spadd_csr(beta, &mut spmm_result, alpha, op_a.as_ref()).unwrap();
        });

        prop_assert!(result.is_err(),
            "The SPMM kernel executed successfully despite mismatch dimensions");
    }

    #[test]
    fn csr_mul_csr(
        // a and b have dimensions compatible for multiplication
        (a, b)
        in csr_strategy()
            .prop_flat_map(|a| {
                let max_nnz = PROPTEST_MAX_NNZ;
                let cols = PROPTEST_MATRIX_DIM;
                let b = csr(PROPTEST_I32_VALUE_STRATEGY, Just(a.ncols()), cols, max_nnz);
                (Just(a), b)
            }))
    {
        // We use the dense result as the ground truth for the arithmetic result
        let c_dense = DMatrix::from(&a) * DMatrix::from(&b);
        // However, it's not enough only to cover the dense result, we also need to verify the
        // sparsity pattern. We can determine the exact sparsity pattern by using
        // dense arithmetic with positive integer values and extracting positive entries.
        let c_dense_pattern = dense_csr_pattern(a.pattern()) * dense_csr_pattern(b.pattern());
        let c_pattern = CsrMatrix::from(&c_dense_pattern).pattern().clone();

        // Check each combination of owned matrices and references
        let c_owned_owned = a.clone() * b.clone();
        prop_assert_eq!(&DMatrix::from(&c_owned_owned), &c_dense);
        prop_assert_eq!(c_owned_owned.pattern(), &c_pattern);

        let c_owned_ref = a.clone() * &b;
        prop_assert_eq!(&DMatrix::from(&c_owned_ref), &c_dense);
        prop_assert_eq!(c_owned_ref.pattern(), &c_pattern);

        let c_ref_owned = &a * b.clone();
        prop_assert_eq!(&DMatrix::from(&c_ref_owned), &c_dense);
        prop_assert_eq!(c_ref_owned.pattern(), &c_pattern);

        let c_ref_ref = &a * &b;
        prop_assert_eq!(&DMatrix::from(&c_ref_ref), &c_dense);
        prop_assert_eq!(c_ref_ref.pattern(), &c_pattern);
    }
}