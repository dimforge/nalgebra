use crate::proptest::*;
use core::f64;
use na::{Const, DMatrix, Matrix4, Matrix4xX};
use nl::Cholesky;
use proptest::prelude::*;
use std::cmp;

fn positive_definite_dmatrix() -> impl Strategy<Value = DMatrix<f64>> {
    // @note(geo-ant) to get positive definite matrices we use M*M^T + alpha*I,
    // where alpha is a constant that is chosen so that the eigenvales stay
    // positive.
    dmatrix().prop_map(|m| {
        let alpha = f64::EPSILON.sqrt() * m.norm_squared();
        let nrows = m.nrows();
        &m * m.transpose() + alpha * DMatrix::identity(nrows, nrows)
    })
}

fn positive_definite_matrix4() -> impl Strategy<Value = Matrix4<f64>> {
    matrix4().prop_map(|m| {
        let alpha = f64::EPSILON.sqrt() * m.norm_squared();
        &m * m.transpose() + alpha * Matrix4::identity()
    })
}

fn positive_definite_linear_system() -> impl Strategy<Value = (DMatrix<f64>, DMatrix<f64>)> {
    positive_definite_dmatrix().prop_flat_map(|a| {
        let b = matrix(PROPTEST_F64, a.nrows(), PROPTEST_MATRIX_DIM);
        (Just(a), b)
    })
}

fn positive_definite_linear_system_4() -> impl Strategy<Value = (Matrix4<f64>, Matrix4xX<f64>)> {
    positive_definite_matrix4().prop_flat_map(|a| {
        let b = matrix(PROPTEST_F64, Const::<4>, PROPTEST_MATRIX_DIM);
        (Just(a), b)
    })
}

proptest! {
    #[test]
    fn cholesky(m in positive_definite_dmatrix()) {
        if let Some(chol) = Cholesky::new(m.clone()) {
            let l = chol.unpack();
            let reconstructed_m = &l * l.transpose();

            prop_assert!(relative_eq!(reconstructed_m, m, epsilon = 1.0e-7));
        }
    }

    #[test]
    fn cholesky_static(m in positive_definite_matrix4()) {
        if let Some(chol) = Cholesky::new(m) {
            let l = chol.unpack();
            let reconstructed_m = &l * l.transpose();

            prop_assert!(relative_eq!(reconstructed_m, m, epsilon = 1.0e-7))
        }
    }

    #[test]
    fn cholesky_solve((a,b) in positive_definite_linear_system()) {

        if let Some(chol) = Cholesky::new(a.clone()) {
            let sol = chol.solve(&b).unwrap();
            prop_assert!(relative_eq!(&a * sol, b, epsilon = 1.0e-5));
        }
    }

    #[test]
    fn cholesky_solve_static((a,b) in positive_definite_linear_system_4()) {
        if let Some(chol) = Cholesky::new(a) {
            let sol = chol.solve(&b).unwrap();
            prop_assert!(relative_eq!(a * sol, b, epsilon = 1.0e-5));
        }
    }

    #[test]
    fn cholesky_inverse(a in positive_definite_dmatrix()) {
        let minv = Cholesky::new(a.clone()).unwrap().inverse().unwrap();
        let id1 = &a  * &minv;
        let id2 = &minv * &a;

        prop_assert!(id1.is_identity(1.0e-6) && id2.is_identity(1.0e-6));
    }

    #[test]
    fn cholesky_inverse_static(a in positive_definite_matrix4()) {
        let minv = Cholesky::new(a.clone()).unwrap().inverse().unwrap();
        let id1 = &a  * &minv;
        let id2 = &minv * &a;

        prop_assert!(id1.is_identity(1.0e-6) && id2.is_identity(1.0e-6));
    }
}
