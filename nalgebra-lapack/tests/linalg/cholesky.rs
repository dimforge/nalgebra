use std::cmp;

use na::{DMatrix, DVector, Matrix4x3, Vector4};
use nl::Cholesky;

use crate::proptest::*;
use proptest::{prop_assert, proptest};

proptest! {
    #[test]
    fn cholesky(m in dmatrix()) {
        let m = &m * m.transpose();
        if let Some(chol) = Cholesky::new(m.clone()) {
            let l = chol.unpack();
            let reconstructed_m = &l * l.transpose();

            prop_assert!(relative_eq!(reconstructed_m, m, epsilon = 1.0e-7));
        }
    }

    #[test]
    fn cholesky_static(m in matrix3()) {
        let m = &m * m.transpose();
        if let Some(chol) = Cholesky::new(m) {
            let l = chol.unpack();
            let reconstructed_m = &l * l.transpose();

            prop_assert!(relative_eq!(reconstructed_m, m, epsilon = 1.0e-7))
        }
    }

    #[test]
    fn cholesky_solve(n in PROPTEST_MATRIX_DIM, nb in PROPTEST_MATRIX_DIM) {
        let n  = cmp::min(n, 15);  // To avoid slowing down the test too much.
        let nb = cmp::min(nb, 15); // To avoid slowing down the test too much.
        let m  = DMatrix::<f64>::new_random(n, n);
        let m   = &m * m.transpose();

        if let Some(chol) = Cholesky::new(m.clone()) {
            let b1 = DVector::new_random(n);
            let b2 = DMatrix::new_random(n, nb);

            let sol1 = chol.solve(&b1).unwrap();
            let sol2 = chol.solve(&b2).unwrap();

            prop_assert!(relative_eq!(&m * sol1, b1, epsilon = 1.0e-6));
            prop_assert!(relative_eq!(&m * sol2, b2, epsilon = 1.0e-6));
        }
    }

    #[test]
    fn cholesky_solve_static(m in matrix4()) {
        let m = &m * m.transpose();
        if let Some(chol) = Cholesky::new(m) {
            let b1 = Vector4::new_random();
            let b2 = Matrix4x3::new_random();

            let sol1 = chol.solve(&b1).unwrap();
            let sol2 = chol.solve(&b2).unwrap();

            prop_assert!(relative_eq!(m * sol1, b1, epsilon = 1.0e-7));
            prop_assert!(relative_eq!(m * sol2, b2, epsilon = 1.0e-7));
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

            prop_assert!(id1.is_identity(1.0e-5) && id2.is_identity(1.0e-5))
        }
    }
}
