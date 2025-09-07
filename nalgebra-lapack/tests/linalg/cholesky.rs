use crate::proptest::*;
use na::{Const, DMatrix, Matrix4, Matrix4xX};
use nl::Cholesky;
use proptest::prelude::*;
use std::cmp;

// actually, cholesky needs positive DEFINITE matrices, but those
// are a bit harder to generate, so this should do.
fn positive_semidefinite_dmatrix() -> impl Strategy<Value = DMatrix<f64>> {
    dmatrix().prop_map(|m| &m * m.transpose())
}

fn positive_semidefinite_matrix4() -> impl Strategy<Value = Matrix4<f64>> {
    matrix4().prop_map(|m| &m * m.transpose())
}

fn positive_semidefinite_linear_system() -> impl Strategy<Value = (DMatrix<f64>, DMatrix<f64>)> {
    positive_semidefinite_dmatrix().prop_flat_map(|a| {
        let b = matrix(PROPTEST_F64, a.nrows(), PROPTEST_MATRIX_DIM);
        (Just(a), b)
    })
}

fn positive_semidefinite_linear_system_4() -> impl Strategy<Value = (Matrix4<f64>, Matrix4xX<f64>)>
{
    positive_semidefinite_matrix4().prop_flat_map(|a| {
        let b = matrix(PROPTEST_F64, Const::<4>, PROPTEST_MATRIX_DIM);
        (Just(a), b)
    })
}

proptest! {
    #[test]
    fn cholesky(m in positive_semidefinite_dmatrix()) {
        if let Some(chol) = Cholesky::new(m.clone()) {
            let l = chol.unpack();
            let reconstructed_m = &l * l.transpose();

            prop_assert!(relative_eq!(reconstructed_m, m, epsilon = 1.0e-7));
        }
    }

    #[test]
    fn cholesky_static(m in positive_semidefinite_matrix4()) {
        if let Some(chol) = Cholesky::new(m) {
            let l = chol.unpack();
            let reconstructed_m = &l * l.transpose();

            prop_assert!(relative_eq!(reconstructed_m, m, epsilon = 1.0e-7))
        }
    }

    #[test]
    fn cholesky_solve((a,b) in positive_semidefinite_linear_system()) {

        if let Some(chol) = Cholesky::new(a.clone()) {
            let sol = chol.solve(&b).unwrap();
            prop_assert!(relative_eq!(&a * sol, b, epsilon = 1.0e-6));
        }
    }

    #[test]
    fn cholesky_solve_static((a,b) in positive_semidefinite_linear_system_4()) {
        if let Some(chol) = Cholesky::new(a) {
            let sol = chol.solve(&b).unwrap();
            prop_assert!(relative_eq!(a * sol, b, epsilon = 1.0e-4));
        }
    }

    #[test]
    fn cholesky_inverse(n in PROPTEST_MATRIX_DIM) {
        let n = cmp::min(n, 15); // To avoid slowing down the test too much.
        let m = DMatrix::<f64>::new_random(n, n);
        let m = &m * m.transpose();

        if let Some(m1) = Cholesky::new(m.clone()).unwrap().inverse() {
            let id1 = &m  * &m1;
            let id2 = &m1 * &m;

            prop_assert!(id1.is_identity(1.0e-6) && id2.is_identity(1.0e-6));
        }
    }

    #[test]
    fn cholesky_inverse_static(m in matrix4()) {
        let m = m * m.transpose();
        if let Some(m1) = Cholesky::new(m.clone()).unwrap().inverse() {
            let id1 = &m  * &m1;
            let id2 = &m1 * &m;

            prop_assert!(id1.is_identity(1.0e-4) && id2.is_identity(1.0e-4))
        }
    }
}
