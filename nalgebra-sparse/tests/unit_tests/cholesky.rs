#![cfg_attr(rustfmt, rustfmt_skip)]
use crate::common::{value_strategy, PROPTEST_MATRIX_DIM, PROPTEST_MAX_NNZ};
use nalgebra_sparse::csc::CscMatrix;
use nalgebra_sparse::factorization::{CscCholesky};
use nalgebra_sparse::proptest::csc;
use nalgebra::{Matrix5, Vector5, Cholesky, DMatrix};
use nalgebra::proptest::matrix;

use proptest::prelude::*;
use matrixcompare::{assert_matrix_eq, prop_assert_matrix_eq};

fn positive_definite() -> impl Strategy<Value=CscMatrix<f64>> {
    let csc_f64 = csc(value_strategy::<f64>(),
                      PROPTEST_MATRIX_DIM,
                      PROPTEST_MATRIX_DIM,
                      PROPTEST_MAX_NNZ);
    csc_f64
        .prop_map(|x| {
            // Add a small multiple of the identity to ensure positive definiteness
            x.transpose() * &x + CscMatrix::identity(x.ncols())
        })
}

proptest! {
    #[test]
    fn cholesky_correct_for_positive_definite_matrices(
        matrix in positive_definite()
    ) {
        let cholesky = CscCholesky::factor(&matrix).unwrap();
        let l = cholesky.take_l();
        let matrix_reconstructed = &l * l.transpose();

        prop_assert_matrix_eq!(matrix_reconstructed, matrix, comp = abs, tol = 1e-8);

        let is_lower_triangular = l.triplet_iter().all(|(i, j, _)| j <= i);
        prop_assert!(is_lower_triangular);
    }

    #[test]
    fn cholesky_solve_positive_definite(
        (matrix, rhs) in positive_definite()
            .prop_flat_map(|csc| {
                let rhs = matrix(value_strategy::<f64>(), csc.nrows(), PROPTEST_MATRIX_DIM);
                (Just(csc), rhs)
            })
    ) {
        let cholesky = CscCholesky::factor(&matrix).unwrap();

        // solve_mut
        {
            let mut x = rhs.clone();
            cholesky.solve_mut(&mut x);
            prop_assert_matrix_eq!(&matrix * &x, rhs, comp=abs, tol=1e-12);
        }

        // solve
        {
            let x = cholesky.solve(&rhs);
            prop_assert_matrix_eq!(&matrix * &x, rhs, comp=abs, tol=1e-12);
        }
    }

}

// This is a test ported from nalgebra's "sparse" module, for the original CsCholesky impl
#[test]
fn cs_cholesky() {
    let mut a = Matrix5::new(
        40.0, 0.0, 0.0, 0.0, 0.0,
        2.0, 60.0, 0.0, 0.0, 0.0,
        1.0, 0.0, 11.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 50.0, 0.0,
        1.0, 0.0, 0.0, 4.0, 10.0
    );
    a.fill_upper_triangle_with_lower_triangle();
    test_cholesky(a);

    let a = Matrix5::from_diagonal(&Vector5::new(40.0, 60.0, 11.0, 50.0, 10.0));
    test_cholesky(a);

    let mut a = Matrix5::new(
        40.0, 0.0, 0.0, 0.0, 0.0,
        2.0, 60.0, 0.0, 0.0, 0.0,
        1.0, 0.0, 11.0, 0.0, 0.0,
        1.0, 0.0, 0.0, 50.0, 0.0,
        0.0, 0.0, 0.0, 4.0, 10.0
    );
    a.fill_upper_triangle_with_lower_triangle();
    test_cholesky(a);

    let mut a = Matrix5::new(
        2.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 2.0, 0.0, 0.0, 0.0,
        1.0, 1.0, 2.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 2.0, 0.0,
        1.0, 1.0, 0.0, 0.0, 2.0
    );
    a.fill_upper_triangle_with_lower_triangle();
    // Test crate::new, left_looking, and up_looking implementations.
    test_cholesky(a);
}

fn test_cholesky(a: Matrix5<f64>) {
    // TODO: Test "refactor"

    let cs_a = CscMatrix::from(&a);

    let chol_a = Cholesky::new(a).unwrap();
    let chol_cs_a = CscCholesky::factor(&cs_a).unwrap();

    let l = chol_a.l();
    let cs_l = chol_cs_a.take_l();

    let l = DMatrix::from_iterator(l.nrows(), l.ncols(), l.iter().cloned());
    let cs_l_mat = DMatrix::from(&cs_l);
    assert_matrix_eq!(l, cs_l_mat, comp = abs, tol = 1e-12);
}