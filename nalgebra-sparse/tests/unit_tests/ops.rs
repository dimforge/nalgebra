use crate::common::{
    PROPTEST_I32_VALUE_STRATEGY, PROPTEST_MATRIX_DIM, PROPTEST_MAX_NNZ, csc_strategy, csr_strategy,
    non_zero_i32_value_strategy, value_strategy,
};
use nalgebra_sparse::csc::CscMatrix;
use nalgebra_sparse::csr::CsrMatrix;
use nalgebra_sparse::ops::Op;
use nalgebra_sparse::ops::serial::{
    spadd_csc_prealloc, spadd_csr_prealloc, spadd_pattern, spmm_csc_dense, spmm_csc_prealloc,
    spmm_csc_prealloc_unchecked, spmm_csr_dense, spmm_csr_pattern, spmm_csr_prealloc,
    spmm_csr_prealloc_unchecked, spsolve_csc_lower_triangular,
};
use nalgebra_sparse::pattern::SparsityPattern;
use nalgebra_sparse::proptest::{csc, csr, sparsity_pattern};

use nalgebra::proptest::{matrix, vector};
use nalgebra::{DMatrix, DMatrixView, DMatrixViewMut, Scalar};

use proptest::prelude::*;

use matrixcompare::prop_assert_matrix_eq;

use std::panic::catch_unwind;

/// Represents the sparsity pattern of a CSR matrix as a dense matrix with 0/1
fn dense_csr_pattern(pattern: &SparsityPattern) -> DMatrix<i32> {
    let boolean_csr =
        CsrMatrix::try_from_pattern_and_values(pattern.clone(), vec![1; pattern.nnz()]).unwrap();
    DMatrix::from(&boolean_csr)
}

/// Represents the sparsity pattern of a CSC matrix as a dense matrix with 0/1
fn dense_csc_pattern(pattern: &SparsityPattern) -> DMatrix<i32> {
    let boolean_csc =
        CscMatrix::try_from_pattern_and_values(pattern.clone(), vec![1; pattern.nnz()]).unwrap();
    DMatrix::from(&boolean_csc)
}

#[derive(Debug)]
struct SpmmCsrDenseArgs<T: Scalar> {
    c: DMatrix<T>,
    beta: T,
    alpha: T,
    a: Op<CsrMatrix<T>>,
    b: Op<DMatrix<T>>,
}

#[derive(Debug)]
struct SpmmCscDenseArgs<T: Scalar> {
    c: DMatrix<T>,
    beta: T,
    alpha: T,
    a: Op<CscMatrix<T>>,
    b: Op<DMatrix<T>>,
}

/// Returns matrices C, A and B with compatible dimensions such that it can be used
/// in an `spmm` operation `C = beta * C + alpha * trans(A) * trans(B)`.
fn spmm_csr_dense_args_strategy() -> impl Strategy<Value = SpmmCsrDenseArgs<i32>> {
    let max_nnz = PROPTEST_MAX_NNZ;
    let value_strategy = PROPTEST_I32_VALUE_STRATEGY;
    let c_rows = PROPTEST_MATRIX_DIM;
    let c_cols = PROPTEST_MATRIX_DIM;
    let common_dim = PROPTEST_MATRIX_DIM;
    let trans_strategy = trans_strategy();
    let c_matrix_strategy = matrix(value_strategy.clone(), c_rows, c_cols);

    (
        c_matrix_strategy,
        common_dim,
        trans_strategy.clone(),
        trans_strategy.clone(),
    )
        .prop_flat_map(move |(c, common_dim, trans_a, trans_b)| {
            let a_shape = if trans_a {
                (common_dim, c.nrows())
            } else {
                (c.nrows(), common_dim)
            };
            let b_shape = if trans_b {
                (c.ncols(), common_dim)
            } else {
                (common_dim, c.ncols())
            };
            let a = csr(value_strategy.clone(), a_shape.0, a_shape.1, max_nnz);
            let b = matrix(value_strategy.clone(), b_shape.0, b_shape.1);

            // We use the same values for alpha, beta parameters as for matrix elements
            let alpha = value_strategy.clone();
            let beta = value_strategy.clone();

            (Just(c), beta, alpha, Just(trans_a), a, Just(trans_b), b)
        })
        .prop_map(
            |(c, beta, alpha, trans_a, a, trans_b, b)| SpmmCsrDenseArgs {
                c,
                beta,
                alpha,
                a: if trans_a {
                    Op::Transpose(a)
                } else {
                    Op::NoOp(a)
                },
                b: if trans_b {
                    Op::Transpose(b)
                } else {
                    Op::NoOp(b)
                },
            },
        )
}

/// Returns matrices C, A and B with compatible dimensions such that it can be used
/// in an `spmm` operation `C = beta * C + alpha * trans(A) * trans(B)`.
fn spmm_csc_dense_args_strategy() -> impl Strategy<Value = SpmmCscDenseArgs<i32>> {
    spmm_csr_dense_args_strategy().prop_map(|args| SpmmCscDenseArgs {
        c: args.c,
        beta: args.beta,
        alpha: args.alpha,
        a: args.a.map_same_op(|a| CscMatrix::from(&a)),
        b: args.b,
    })
}

#[derive(Debug)]
struct SpaddCsrArgs<T> {
    c: CsrMatrix<T>,
    beta: T,
    alpha: T,
    a: Op<CsrMatrix<T>>,
}

#[derive(Debug)]
struct SpaddCscArgs<T> {
    c: CscMatrix<T>,
    beta: T,
    alpha: T,
    a: Op<CscMatrix<T>>,
}

fn spadd_csr_prealloc_args_strategy() -> impl Strategy<Value = SpaddCsrArgs<i32>> {
    let value_strategy = PROPTEST_I32_VALUE_STRATEGY;

    spadd_pattern_strategy()
        .prop_flat_map(move |(a_pattern, b_pattern)| {
            let c_pattern = spadd_pattern(&a_pattern, &b_pattern);

            let a_values = vec![value_strategy.clone(); a_pattern.nnz()];
            let c_values = vec![value_strategy.clone(); c_pattern.nnz()];
            let alpha = value_strategy.clone();
            let beta = value_strategy.clone();
            (
                Just(c_pattern),
                Just(a_pattern),
                c_values,
                a_values,
                alpha,
                beta,
                trans_strategy(),
            )
        })
        .prop_map(
            |(c_pattern, a_pattern, c_values, a_values, alpha, beta, trans_a)| {
                let c = CsrMatrix::try_from_pattern_and_values(c_pattern, c_values).unwrap();
                let a = CsrMatrix::try_from_pattern_and_values(a_pattern, a_values).unwrap();

                let a = if trans_a {
                    Op::Transpose(a.transpose())
                } else {
                    Op::NoOp(a)
                };
                SpaddCsrArgs { c, beta, alpha, a }
            },
        )
}

fn spadd_csc_prealloc_args_strategy() -> impl Strategy<Value = SpaddCscArgs<i32>> {
    spadd_csr_prealloc_args_strategy().prop_map(|args| SpaddCscArgs {
        c: CscMatrix::from(&args.c),
        beta: args.beta,
        alpha: args.alpha,
        a: args.a.map_same_op(|a| CscMatrix::from(&a)),
    })
}

fn dense_strategy() -> impl Strategy<Value = DMatrix<i32>> {
    matrix(
        PROPTEST_I32_VALUE_STRATEGY,
        PROPTEST_MATRIX_DIM,
        PROPTEST_MATRIX_DIM,
    )
}

fn trans_strategy() -> impl Strategy<Value = bool> + Clone {
    proptest::bool::ANY
}

/// Wraps the values of the given strategy in `Op`, producing both transposed and non-transposed
/// values.
fn op_strategy<S: Strategy>(strategy: S) -> impl Strategy<Value = Op<S::Value>> {
    let is_transposed = proptest::bool::ANY;
    (strategy, is_transposed).prop_map(|(obj, is_trans)| {
        if is_trans {
            Op::Transpose(obj)
        } else {
            Op::NoOp(obj)
        }
    })
}

fn pattern_strategy() -> impl Strategy<Value = SparsityPattern> {
    sparsity_pattern(PROPTEST_MATRIX_DIM, PROPTEST_MATRIX_DIM, PROPTEST_MAX_NNZ)
}

/// Constructs pairs (a, b) where a and b have the same dimensions
fn spadd_pattern_strategy() -> impl Strategy<Value = (SparsityPattern, SparsityPattern)> {
    pattern_strategy().prop_flat_map(|a| {
        let b = sparsity_pattern(a.major_dim(), a.minor_dim(), PROPTEST_MAX_NNZ);
        (Just(a), b)
    })
}

/// Constructs pairs (a, b) where a and b have compatible dimensions for a matrix product
fn spmm_csr_pattern_strategy() -> impl Strategy<Value = (SparsityPattern, SparsityPattern)> {
    pattern_strategy().prop_flat_map(|a| {
        let b = sparsity_pattern(a.minor_dim(), PROPTEST_MATRIX_DIM, PROPTEST_MAX_NNZ);
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

#[derive(Debug)]
struct SpmmCscArgs<T> {
    c: CscMatrix<T>,
    beta: T,
    alpha: T,
    a: Op<CscMatrix<T>>,
    b: Op<CscMatrix<T>>,
}

fn spmm_csr_prealloc_args_strategy() -> impl Strategy<Value = SpmmCsrArgs<i32>> {
    spmm_csr_pattern_strategy()
        .prop_flat_map(|(a_pattern, b_pattern)| {
            let a_values = vec![PROPTEST_I32_VALUE_STRATEGY; a_pattern.nnz()];
            let b_values = vec![PROPTEST_I32_VALUE_STRATEGY; b_pattern.nnz()];
            let c_pattern = spmm_csr_pattern(&a_pattern, &b_pattern);
            let c_values = vec![PROPTEST_I32_VALUE_STRATEGY; c_pattern.nnz()];
            let a = a_values.prop_map(move |values| {
                CsrMatrix::try_from_pattern_and_values(a_pattern.clone(), values).unwrap()
            });
            let b = b_values.prop_map(move |values| {
                CsrMatrix::try_from_pattern_and_values(b_pattern.clone(), values).unwrap()
            });
            let c = c_values.prop_map(move |values| {
                CsrMatrix::try_from_pattern_and_values(c_pattern.clone(), values).unwrap()
            });
            let alpha = PROPTEST_I32_VALUE_STRATEGY;
            let beta = PROPTEST_I32_VALUE_STRATEGY;
            (c, beta, alpha, trans_strategy(), a, trans_strategy(), b)
        })
        .prop_map(
            |(c, beta, alpha, trans_a, a, trans_b, b)| SpmmCsrArgs::<i32> {
                c,
                beta,
                alpha,
                a: if trans_a {
                    Op::Transpose(a.transpose())
                } else {
                    Op::NoOp(a)
                },
                b: if trans_b {
                    Op::Transpose(b.transpose())
                } else {
                    Op::NoOp(b)
                },
            },
        )
}

fn spmm_csc_prealloc_args_strategy() -> impl Strategy<Value = SpmmCscArgs<i32>> {
    // Note: Converting from CSR is simple, but might be significantly slower than
    // writing a common implementation that can be shared between CSR and CSC args
    spmm_csr_prealloc_args_strategy().prop_map(|args| SpmmCscArgs {
        c: CscMatrix::from(&args.c),
        beta: args.beta,
        alpha: args.alpha,
        a: args.a.map_same_op(|a| CscMatrix::from(&a)),
        b: args.b.map_same_op(|b| CscMatrix::from(&b)),
    })
}

fn csc_invertible_diagonal() -> impl Strategy<Value = CscMatrix<f64>> {
    let non_zero_values =
        value_strategy::<f64>().prop_filter("Only non-zeros values accepted", |x| x != &0.0);

    vector(non_zero_values, PROPTEST_MATRIX_DIM).prop_map(|d| {
        let mut matrix = CscMatrix::identity(d.len());
        matrix.values_mut().clone_from_slice(&d.as_slice());
        matrix
    })
}

fn csc_square_with_non_zero_diagonals() -> impl Strategy<Value = CscMatrix<f64>> {
    csc_invertible_diagonal().prop_flat_map(|d| {
        csc(
            value_strategy::<f64>(),
            d.nrows(),
            d.nrows(),
            PROPTEST_MAX_NNZ,
        )
        .prop_map(move |mut c| {
            for (i, j, v) in c.triplet_iter_mut() {
                if i == j {
                    *v = 0.0;
                }
            }

            // Return the sum of a matrix with zero diagonals and an invertible diagonal
            // matrix
            c + &d
        })
    })
}

/// Helper function to help us call dense GEMM with our `Op` type
fn dense_gemm<'a>(
    beta: i32,
    c: impl Into<DMatrixViewMut<'a, i32>>,
    alpha: i32,
    a: Op<impl Into<DMatrixView<'a, i32>>>,
    b: Op<impl Into<DMatrixView<'a, i32>>>,
) {
    let mut c = c.into();
    let a = a.convert();
    let b = b.convert();

    use Op::{NoOp, Transpose};
    match (a, b) {
        (NoOp(a), NoOp(b)) => c.gemm(alpha, &a, &b, beta),
        (Transpose(a), NoOp(b)) => c.gemm(alpha, &a.transpose(), &b, beta),
        (NoOp(a), Transpose(b)) => c.gemm(alpha, &a, &b.transpose(), beta),
        (Transpose(a), Transpose(b)) => c.gemm(alpha, &a.transpose(), &b.transpose(), beta),
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
            Op::NoOp(a) => a.nrows(),
            Op::Transpose(a) => a.ncols(),
        };
        let product_cols = match &b {
            Op::NoOp(b) => b.ncols(),
            Op::Transpose(b) => b.nrows(),
        };
        // Determine the common dimension in the product
        // from the perspective of a and b, respectively
        let product_a_common = match &a {
            Op::NoOp(a) => a.ncols(),
            Op::Transpose(a) => a.nrows(),
        };
        let product_b_common = match &b {
            Op::NoOp(b) => b.nrows(),
            Op::Transpose(b) => b.ncols()
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
        let a_csr = CsrMatrix::try_from_pattern_and_values(a.clone(), vec![1; a.nnz()])
            .unwrap();
        let a_dense = DMatrix::from(&a_csr);
        let b_csr = CsrMatrix::try_from_pattern_and_values(b.clone(), vec![1; b.nnz()])
            .unwrap();
        let b_dense = DMatrix::from(&b_csr);
        let c_dense = a_dense + b_dense;
        let c_csr = CsrMatrix::from(&c_dense);

        prop_assert_eq!(&pattern_result, c_csr.pattern());
    }

    #[test]
    fn spadd_csr_prealloc_test(SpaddCsrArgs { c, beta, alpha, a } in spadd_csr_prealloc_args_strategy()) {
        // Test that we get the expected result by comparing to an equivalent dense operation
        // (here we give in the C matrix, so the sparsity pattern is essentially fixed)

        let mut c_sparse = c.clone();
        spadd_csr_prealloc(beta, &mut c_sparse, alpha, a.as_ref()).unwrap();

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
                let b = csr(PROPTEST_I32_VALUE_STRATEGY, a.nrows(), a.ncols(), PROPTEST_MAX_NNZ);
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
    fn csr_sub_csr(
        // a and b have the same dimensions
        (a, b)
        in csr_strategy()
            .prop_flat_map(|a| {
                let b = csr(PROPTEST_I32_VALUE_STRATEGY, a.nrows(), a.ncols(), PROPTEST_MAX_NNZ);
                (Just(a), b)
            }))
    {
        // See comments in csr_add_csr for rationale for checking the pattern this way
        let c_dense = DMatrix::from(&a) - DMatrix::from(&b);
        let c_dense_pattern = dense_csr_pattern(a.pattern()) + dense_csr_pattern(b.pattern());
        let c_pattern = CsrMatrix::from(&c_dense_pattern).pattern().clone();

        // Check each combination of owned matrices and references
        let c_owned_owned = a.clone() - b.clone();
        prop_assert_eq!(&DMatrix::from(&c_owned_owned), &c_dense);
        prop_assert_eq!(c_owned_owned.pattern(), &c_pattern);

        let c_owned_ref = a.clone() - &b;
        prop_assert_eq!(&DMatrix::from(&c_owned_ref), &c_dense);
        prop_assert_eq!(c_owned_ref.pattern(), &c_pattern);

        let c_ref_owned = &a - b.clone();
        prop_assert_eq!(&DMatrix::from(&c_ref_owned), &c_dense);
        prop_assert_eq!(c_ref_owned.pattern(), &c_pattern);

        let c_ref_ref = &a - &b;
        prop_assert_eq!(&DMatrix::from(&c_ref_ref), &c_dense);
        prop_assert_eq!(c_ref_ref.pattern(), &c_pattern);
    }

    #[test]
    fn spmm_csr_pattern_test((a, b) in spmm_csr_pattern_strategy())
    {
        // (a, b) are multiplication-wise dimensionally compatible patterns
        let c_pattern = spmm_csr_pattern(&a, &b);

        // To verify the pattern, we construct CSR matrices with positive integer entries
        // corresponding to a and b, and convert them to dense matrices.
        // The product of these dense matrices will then have non-zeros in exactly the same locations
        // as the result of "multiplying" the sparsity patterns
        let a_csr = CsrMatrix::try_from_pattern_and_values(a.clone(), vec![1; a.nnz()])
            .unwrap();
        let a_dense = DMatrix::from(&a_csr);
        let b_csr = CsrMatrix::try_from_pattern_and_values(b.clone(), vec![1; b.nnz()])
            .unwrap();
        let b_dense = DMatrix::from(&b_csr);
        let c_dense = a_dense * b_dense;
        let c_csr = CsrMatrix::from(&c_dense);

        prop_assert_eq!(&c_pattern, c_csr.pattern());
    }

    #[test]
    fn spmm_csr_prealloc_unchecked_test(SpmmCsrArgs { c, beta, alpha, a, b }
        in spmm_csr_prealloc_args_strategy()
    ) {
        // Test that we get the expected result by comparing to an equivalent dense operation
        // (here we give in the C matrix, so the sparsity pattern is essentially fixed)
        let mut c_sparse = c.clone();
        spmm_csr_prealloc_unchecked(beta, &mut c_sparse, alpha, a.as_ref(), b.as_ref()).unwrap();

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
    fn spmm_csr_prealloc_test(SpmmCsrArgs { c, beta, alpha, a, b }
        in spmm_csr_prealloc_args_strategy()
    ) {
        // Test that we get the expected result by comparing to an equivalent dense operation
        // (here we give in the C matrix, so the sparsity pattern is essentially fixed)
        let mut c_sparse = c.clone();
        spmm_csr_prealloc(beta, &mut c_sparse, alpha, a.as_ref(), b.as_ref()).unwrap();

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
    fn spmm_csr_prealloc_panics_on_dim_mismatch(
        (alpha, beta, c, a, b)
        in (PROPTEST_I32_VALUE_STRATEGY,
            PROPTEST_I32_VALUE_STRATEGY,
            csr_strategy(),
            op_strategy(csr_strategy()),
            op_strategy(csr_strategy()))
    ) {
        // We refer to `A * B` as the "product"
        let product_rows = match &a {
            Op::NoOp(a) => a.nrows(),
            Op::Transpose(a) => a.ncols(),
        };
        let product_cols = match &b {
            Op::NoOp(b) => b.ncols(),
            Op::Transpose(b) => b.nrows(),
        };
        // Determine the common dimension in the product
        // from the perspective of a and b, respectively
        let product_a_common = match &a {
            Op::NoOp(a) => a.ncols(),
            Op::Transpose(a) => a.nrows(),
        };
        let product_b_common = match &b {
            Op::NoOp(b) => b.nrows(),
            Op::Transpose(b) => b.ncols(),
        };

        let dims_are_compatible = product_rows == c.nrows()
            && product_cols == c.ncols()
            && product_a_common == product_b_common;

        // If the dimensions randomly happen to be compatible, then of course we need to
        // skip the test, so we assume that they are not.
        prop_assume!(!dims_are_compatible);

        let result = catch_unwind(|| {
            let mut spmm_result = c.clone();
            spmm_csr_prealloc(beta, &mut spmm_result, alpha, a.as_ref(), b.as_ref()).unwrap();
        });

        prop_assert!(result.is_err(),
            "The SPMM kernel executed successfully despite mismatch dimensions");
    }

    #[test]
    fn spadd_csr_prealloc_panics_on_dim_mismatch(
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
            spadd_csr_prealloc(beta, &mut spmm_result, alpha, op_a.as_ref()).unwrap();
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
                let b = csr(PROPTEST_I32_VALUE_STRATEGY, a.ncols(), cols, max_nnz);
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

    #[test]
    fn spmm_csc_prealloc_test(SpmmCscArgs { c, beta, alpha, a, b }
        in spmm_csc_prealloc_args_strategy()
    ) {
        // Test that we get the expected result by comparing to an equivalent dense operation
        // (here we give in the C matrix, so the sparsity pattern is essentially fixed)
        let mut c_sparse = c.clone();
        spmm_csc_prealloc(beta, &mut c_sparse, alpha, a.as_ref(), b.as_ref()).unwrap();

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
    fn spmm_csc_prealloc_unchecked_test(SpmmCscArgs { c, beta, alpha, a, b }
        in spmm_csc_prealloc_args_strategy()
    ) {
        // Test that we get the expected result by comparing to an equivalent dense operation
        // (here we give in the C matrix, so the sparsity pattern is essentially fixed)
        let mut c_sparse = c.clone();
        spmm_csc_prealloc_unchecked(beta, &mut c_sparse, alpha, a.as_ref(), b.as_ref()).unwrap();

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
    fn spmm_csc_prealloc_panics_on_dim_mismatch(
        (alpha, beta, c, a, b)
        in (PROPTEST_I32_VALUE_STRATEGY,
            PROPTEST_I32_VALUE_STRATEGY,
            csc_strategy(),
            op_strategy(csc_strategy()),
            op_strategy(csc_strategy()))
    ) {
        // We refer to `A * B` as the "product"
        let product_rows = match &a {
            Op::NoOp(a) => a.nrows(),
            Op::Transpose(a) => a.ncols(),
        };
        let product_cols = match &b {
            Op::NoOp(b) => b.ncols(),
            Op::Transpose(b) => b.nrows(),
        };
        // Determine the common dimension in the product
        // from the perspective of a and b, respectively
        let product_a_common = match &a {
            Op::NoOp(a) => a.ncols(),
            Op::Transpose(a) => a.nrows(),
        };
        let product_b_common = match &b {
            Op::NoOp(b) => b.nrows(),
            Op::Transpose(b) => b.ncols(),
        };

        let dims_are_compatible = product_rows == c.nrows()
            && product_cols == c.ncols()
            && product_a_common == product_b_common;

        // If the dimensions randomly happen to be compatible, then of course we need to
        // skip the test, so we assume that they are not.
        prop_assume!(!dims_are_compatible);

        let result = catch_unwind(|| {
            let mut spmm_result = c.clone();
            spmm_csc_prealloc(beta, &mut spmm_result, alpha, a.as_ref(), b.as_ref()).unwrap();
        });

        prop_assert!(result.is_err(),
            "The SPMM kernel executed successfully despite mismatch dimensions");
    }

    #[test]
    fn csc_mul_csc(
        // a and b have dimensions compatible for multiplication
        (a, b)
        in csc_strategy()
            .prop_flat_map(|a| {
                let max_nnz = PROPTEST_MAX_NNZ;
                let cols = PROPTEST_MATRIX_DIM;
                let b = csc(PROPTEST_I32_VALUE_STRATEGY, a.ncols(), cols, max_nnz);
                (Just(a), b)
            })
            .prop_map(|(a, b)| {
                println!("a: {} x {}, b: {} x {}", a.nrows(), a.ncols(), b.nrows(), b.ncols());
                (a, b)
            }))
    {
        assert_eq!(a.ncols(), b.nrows());
        // We use the dense result as the ground truth for the arithmetic result
        let c_dense = DMatrix::from(&a) * DMatrix::from(&b);
        // However, it's not enough only to cover the dense result, we also need to verify the
        // sparsity pattern. We can determine the exact sparsity pattern by using
        // dense arithmetic with positive integer values and extracting positive entries.
        let c_dense_pattern = dense_csc_pattern(a.pattern()) * dense_csc_pattern(b.pattern());
        let c_pattern = CscMatrix::from(&c_dense_pattern).pattern().clone();

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

    #[test]
    fn spmm_csc_dense_agrees_with_dense_result(
        SpmmCscDenseArgs { c, beta, alpha, a, b }
         in spmm_csc_dense_args_strategy()
    ) {
        let mut spmm_result = c.clone();
        spmm_csc_dense(beta, &mut spmm_result, alpha, a.as_ref(), b.as_ref());

        let mut gemm_result = c.clone();
        let a_dense = a.map_same_op(|a| DMatrix::from(&a));
        dense_gemm(beta, &mut gemm_result, alpha, a_dense.as_ref(), b.as_ref());

        prop_assert_eq!(spmm_result, gemm_result);
    }

    #[test]
    fn spmm_csc_dense_panics_on_dim_mismatch(
        (alpha, beta, c, a, b)
        in (PROPTEST_I32_VALUE_STRATEGY,
            PROPTEST_I32_VALUE_STRATEGY,
            dense_strategy(),
            op_strategy(csc_strategy()),
            op_strategy(dense_strategy()))
    ) {
        // We refer to `A * B` as the "product"
        let product_rows = match &a {
            Op::NoOp(a) => a.nrows(),
            Op::Transpose(a) => a.ncols(),
        };
        let product_cols = match &b {
            Op::NoOp(b) => b.ncols(),
            Op::Transpose(b) => b.nrows(),
        };
        // Determine the common dimension in the product
        // from the perspective of a and b, respectively
        let product_a_common = match &a {
            Op::NoOp(a) => a.ncols(),
            Op::Transpose(a) => a.nrows(),
        };
        let product_b_common = match &b {
            Op::NoOp(b) => b.nrows(),
            Op::Transpose(b) => b.ncols()
        };

        let dims_are_compatible = product_rows == c.nrows()
            && product_cols == c.ncols()
            && product_a_common == product_b_common;

        // If the dimensions randomly happen to be compatible, then of course we need to
        // skip the test, so we assume that they are not.
        prop_assume!(!dims_are_compatible);

        let result = catch_unwind(|| {
            let mut spmm_result = c.clone();
            spmm_csc_dense(beta, &mut spmm_result, alpha, a.as_ref(), b.as_ref());
        });

        prop_assert!(result.is_err(),
            "The SPMM kernel executed successfully despite mismatch dimensions");
    }

    #[test]
    fn spadd_csc_prealloc_test(SpaddCscArgs { c, beta, alpha, a } in spadd_csc_prealloc_args_strategy()) {
        // Test that we get the expected result by comparing to an equivalent dense operation
        // (here we give in the C matrix, so the sparsity pattern is essentially fixed)

        let mut c_sparse = c.clone();
        spadd_csc_prealloc(beta, &mut c_sparse, alpha, a.as_ref()).unwrap();

        let mut c_dense = DMatrix::from(&c);
        let op_a_dense = match a {
            Op::NoOp(a) => DMatrix::from(&a),
            Op::Transpose(a) => DMatrix::from(&a).transpose(),
        };
        c_dense = beta * c_dense + alpha * &op_a_dense;

        prop_assert_eq!(&DMatrix::from(&c_sparse), &c_dense);
    }

    #[test]
    fn spadd_csc_prealloc_panics_on_dim_mismatch(
        (alpha, beta, c, op_a)
        in (PROPTEST_I32_VALUE_STRATEGY,
            PROPTEST_I32_VALUE_STRATEGY,
            csc_strategy(),
            op_strategy(csc_strategy()))
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
            spadd_csc_prealloc(beta, &mut spmm_result, alpha, op_a.as_ref()).unwrap();
        });

        prop_assert!(result.is_err(),
            "The SPMM kernel executed successfully despite mismatch dimensions");
    }

    #[test]
    fn csc_add_csc(
        // a and b have the same dimensions
        (a, b)
        in csc_strategy()
            .prop_flat_map(|a| {
                let b = csc(PROPTEST_I32_VALUE_STRATEGY, a.nrows(), a.ncols(), PROPTEST_MAX_NNZ);
                (Just(a), b)
            }))
    {
        // We use the dense result as the ground truth for the arithmetic result
        let c_dense = DMatrix::from(&a) + DMatrix::from(&b);
        // However, it's not enough only to cover the dense result, we also need to verify the
        // sparsity pattern. We can determine the exact sparsity pattern by using
        // dense arithmetic with positive integer values and extracting positive entries.
        let c_dense_pattern = dense_csc_pattern(a.pattern()) + dense_csc_pattern(b.pattern());
        let c_pattern = CscMatrix::from(&c_dense_pattern).pattern().clone();

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
    fn csc_sub_csc(
        // a and b have the same dimensions
        (a, b)
        in csc_strategy()
            .prop_flat_map(|a| {
                let b = csc(PROPTEST_I32_VALUE_STRATEGY, a.nrows(), a.ncols(), PROPTEST_MAX_NNZ);
                (Just(a), b)
            }))
    {
        // See comments in csc_add_csc for rationale for checking the pattern this way
        let c_dense = DMatrix::from(&a) - DMatrix::from(&b);
        let c_dense_pattern = dense_csc_pattern(a.pattern()) + dense_csc_pattern(b.pattern());
        let c_pattern = CscMatrix::from(&c_dense_pattern).pattern().clone();

        // Check each combination of owned matrices and references
        let c_owned_owned = a.clone() - b.clone();
        prop_assert_eq!(&DMatrix::from(&c_owned_owned), &c_dense);
        prop_assert_eq!(c_owned_owned.pattern(), &c_pattern);

        let c_owned_ref = a.clone() - &b;
        prop_assert_eq!(&DMatrix::from(&c_owned_ref), &c_dense);
        prop_assert_eq!(c_owned_ref.pattern(), &c_pattern);

        let c_ref_owned = &a - b.clone();
        prop_assert_eq!(&DMatrix::from(&c_ref_owned), &c_dense);
        prop_assert_eq!(c_ref_owned.pattern(), &c_pattern);

        let c_ref_ref = &a - &b;
        prop_assert_eq!(&DMatrix::from(&c_ref_ref), &c_dense);
        prop_assert_eq!(c_ref_ref.pattern(), &c_pattern);
    }

    #[test]
    fn csr_mul_scalar((scalar, matrix) in (PROPTEST_I32_VALUE_STRATEGY, csr_strategy())) {
        let dense = DMatrix::from(&matrix);
        let dense_result = dense * scalar;

        let result_owned_owned = matrix.clone() * scalar;
        let result_owned_ref = matrix.clone() * &scalar;
        let result_ref_owned = &matrix * scalar;
        let result_ref_ref = &matrix * &scalar;

        // Check that all the combinations of reference and owned variables return the same
        // result
        prop_assert_eq!(&result_owned_ref, &result_owned_owned);
        prop_assert_eq!(&result_ref_owned, &result_owned_owned);
        prop_assert_eq!(&result_ref_ref, &result_owned_owned);

        // Check that this result is consistent with the dense result, and that the
        // NNZ is the same as before
        prop_assert_eq!(result_owned_owned.nnz(), matrix.nnz());
        prop_assert_eq!(DMatrix::from(&result_owned_owned), dense_result);

        // Finally, check mul-assign
        let mut result_assign_owned = matrix.clone();
        result_assign_owned *= scalar;
        let mut result_assign_ref = matrix.clone();
        result_assign_ref *= &scalar;

        prop_assert_eq!(&result_assign_owned, &result_owned_owned);
        prop_assert_eq!(&result_assign_ref, &result_owned_owned);
    }

    #[test]
    fn csc_mul_scalar((scalar, matrix) in (PROPTEST_I32_VALUE_STRATEGY, csc_strategy())) {
        let dense = DMatrix::from(&matrix);
        let dense_result = dense * scalar;

        let result_owned_owned = matrix.clone() * scalar;
        let result_owned_ref = matrix.clone() * &scalar;
        let result_ref_owned = &matrix * scalar;
        let result_ref_ref = &matrix * &scalar;

        // Check that all the combinations of reference and owned variables return the same
        // result
        prop_assert_eq!(&result_owned_ref, &result_owned_owned);
        prop_assert_eq!(&result_ref_owned, &result_owned_owned);
        prop_assert_eq!(&result_ref_ref, &result_owned_owned);

        // Check that this result is consistent with the dense result, and that the
        // NNZ is the same as before
        prop_assert_eq!(result_owned_owned.nnz(), matrix.nnz());
        prop_assert_eq!(DMatrix::from(&result_owned_owned), dense_result);

        // Finally, check mul-assign
        let mut result_assign_owned = matrix.clone();
        result_assign_owned *= scalar;
        let mut result_assign_ref = matrix.clone();
        result_assign_ref *= &scalar;

        prop_assert_eq!(&result_assign_owned, &result_owned_owned);
        prop_assert_eq!(&result_assign_ref, &result_owned_owned);
    }

    #[test]
    fn scalar_mul_csr((scalar, matrix) in (PROPTEST_I32_VALUE_STRATEGY, csr_strategy())) {
        // For scalar * matrix, we cannot generally implement this for any type T,
        // so we have implemented this for the built in types separately. This requires
        // us to also test these types separately. For validation, we check that
        //  scalar * matrix == matrix * scalar,
        // which is sufficient for correctness if matrix * scalar is correctly implemented
        // (which is tested separately).
        // We only test for i32 here, because with our current implementation, the implementations
        // for different types are completely identical and only rely on basic arithmetic
        // operations
        let result = &matrix * scalar;
        prop_assert_eq!(&(scalar * matrix.clone()), &result);
        prop_assert_eq!(&(scalar * &matrix), &result);
        prop_assert_eq!(&(&scalar * matrix.clone()), &result);
        prop_assert_eq!(&(&scalar * &matrix), &result);
    }

    #[test]
    fn scalar_mul_csc((scalar, matrix) in (PROPTEST_I32_VALUE_STRATEGY, csc_strategy())) {
        // See comments for scalar_mul_csr
        let result = &matrix * scalar;
        prop_assert_eq!(&(scalar * matrix.clone()), &result);
        prop_assert_eq!(&(scalar * &matrix), &result);
        prop_assert_eq!(&(&scalar * matrix.clone()), &result);
        prop_assert_eq!(&(&scalar * &matrix), &result);
    }

    #[test]
    fn csr_neg(csr in csr_strategy()) {
        let result = &csr - 2 * &csr;
        prop_assert_eq!(-&csr, result.clone());
        prop_assert_eq!(-csr, result);
    }

    #[test]
    fn csc_neg(csc in csc_strategy()) {
        let result = &csc - 2 * &csc;
        prop_assert_eq!(-&csc, result.clone());
        prop_assert_eq!(-csc, result);
    }

    #[test]
    fn csr_div((csr, divisor) in (csr_strategy(), non_zero_i32_value_strategy())) {
        let result_owned_owned = csr.clone() / divisor;
        let result_owned_ref = csr.clone() / &divisor;
        let result_ref_owned = &csr / divisor;
        let result_ref_ref = &csr / &divisor;

        // Verify that all results are the same
        prop_assert_eq!(&result_owned_ref, &result_owned_owned);
        prop_assert_eq!(&result_ref_owned, &result_owned_owned);
        prop_assert_eq!(&result_ref_ref, &result_owned_owned);

        // Check that NNZ was left unchanged
        prop_assert_eq!(result_owned_owned.nnz(), csr.nnz());

        // Then compare against the equivalent dense result
        let dense_result = DMatrix::from(&csr) / divisor;
        prop_assert_eq!(DMatrix::from(&result_owned_owned), dense_result);
    }

    #[test]
    fn csc_div((csc, divisor) in (csc_strategy(), non_zero_i32_value_strategy())) {
        let result_owned_owned = csc.clone() / divisor;
        let result_owned_ref = csc.clone() / &divisor;
        let result_ref_owned = &csc / divisor;
        let result_ref_ref = &csc / &divisor;

        // Verify that all results are the same
        prop_assert_eq!(&result_owned_ref, &result_owned_owned);
        prop_assert_eq!(&result_ref_owned, &result_owned_owned);
        prop_assert_eq!(&result_ref_ref, &result_owned_owned);

        // Check that NNZ was left unchanged
        prop_assert_eq!(result_owned_owned.nnz(), csc.nnz());

        // Then compare against the equivalent dense result
        let dense_result = DMatrix::from(&csc) / divisor;
        prop_assert_eq!(DMatrix::from(&result_owned_owned), dense_result);
    }

    #[test]
    fn csr_div_assign((csr, divisor) in (csr_strategy(), non_zero_i32_value_strategy())) {
        let result_owned = {
            let mut csr = csr.clone();
            csr /= divisor;
            csr
        };

        let result_ref = {
            let mut csr = csr.clone();
            csr /= &divisor;
            csr
        };

        let expected_result = csr / divisor;

        prop_assert_eq!(&result_owned, &expected_result);
        prop_assert_eq!(&result_ref, &expected_result);
    }

    #[test]
    fn csc_div_assign((csc, divisor) in (csc_strategy(), non_zero_i32_value_strategy())) {
        let result_owned = {
            let mut csc = csc.clone();
            csc /= divisor;
            csc
        };

        let result_ref = {
            let mut csc = csc.clone();
            csc /= &divisor;
            csc
        };

        let expected_result = csc / divisor;

        prop_assert_eq!(&result_owned, &expected_result);
        prop_assert_eq!(&result_ref, &expected_result);
    }

    #[test]
    fn csr_mul_dense(
        // a and b have dimensions compatible for multiplication
        (a, b)
        in csr_strategy()
            .prop_flat_map(|a| {
                let cols = PROPTEST_MATRIX_DIM;
                let b = matrix(PROPTEST_I32_VALUE_STRATEGY, a.ncols(), cols);
                (Just(a), b)
            }))
    {
        let expected = DMatrix::from(&a) * &b;
        prop_assert_eq!(&a * &b, expected.clone());
        prop_assert_eq!(&a * b.clone(), expected.clone());
        prop_assert_eq!(a.clone() * &b, expected.clone());
        prop_assert_eq!(a.clone() * b.clone(), expected.clone());
    }

    #[test]
    fn csc_mul_dense(
        // a and b have dimensions compatible for multiplication
        (a, b)
        in csc_strategy()
            .prop_flat_map(|a| {
                let cols = PROPTEST_MATRIX_DIM;
                let b = matrix(PROPTEST_I32_VALUE_STRATEGY, a.ncols(), cols);
                (Just(a), b)
            }))
    {
        let expected = DMatrix::from(&a) * &b;
        prop_assert_eq!(&a * &b, expected.clone());
        prop_assert_eq!(&a * b.clone(), expected.clone());
        prop_assert_eq!(a.clone() * &b, expected.clone());
        prop_assert_eq!(a.clone() * b.clone(), expected.clone());
    }

    #[test]
    fn csc_solve_lower_triangular_no_transpose(
        // A CSC matrix `a` and a dimensionally compatible dense matrix `b`
        (a, b)
            in csc_square_with_non_zero_diagonals()
                .prop_flat_map(|a| {
                    let nrows = a.nrows();
                    (Just(a), matrix(value_strategy::<f64>(), nrows, PROPTEST_MATRIX_DIM))
                }))
    {
        let mut x = b.clone();
        spsolve_csc_lower_triangular(Op::NoOp(&a), &mut x).unwrap();

        let a_lower = a.lower_triangle();
        // We're using a high tolerance here because there are some "bad" inputs that can give
        // severe loss of precision.
        prop_assert_matrix_eq!(&a_lower * &x, &b, comp = abs, tol = 1e-4);
    }

    #[test]
    fn csc_solve_lower_triangular_transpose(
        // A CSC matrix `a` and a dimensionally compatible dense matrix `b` (with a transposed)
        (a, b)
            in csc_square_with_non_zero_diagonals()
                .prop_flat_map(|a| {
                    let ncols = a.ncols();
                    (Just(a), matrix(value_strategy::<f64>(), ncols, PROPTEST_MATRIX_DIM))
                }))
    {
        let mut x = b.clone();
        spsolve_csc_lower_triangular(Op::Transpose(&a), &mut x).unwrap();

        let a_lower = a.lower_triangle();
        // We're using a high tolerance here because there are some "bad" inputs that can give
        // severe loss of precision.
        prop_assert_matrix_eq!(&a_lower.transpose() * &x, &b, comp = abs, tol = 1e-4);
    }

}
