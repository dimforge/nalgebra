use std::cmp;

use na::{DMatrix, Matrix4};
use nl::Eigen;

use crate::proptest::*;
use proptest::{prop_assert, proptest};

proptest! {
    #[test]
    fn eigensystem(n in PROPTEST_MATRIX_DIM) {
        let n = cmp::min(n, 25);
        let m = DMatrix::<f64>::new_random(n, n);

        if let Some(eig) = Eigen::new(m.clone(), true, true) {
            // TODO: test the complex case too.
            if eig.eigenvalues_are_real() {
                let eigvals                = DMatrix::from_diagonal(&eig.eigenvalues_re);
                let transformed_eigvectors = &m * eig.eigenvectors.as_ref().unwrap();
                let scaled_eigvectors      = eig.eigenvectors.as_ref().unwrap() * &eigvals;

                let transformed_left_eigvectors = m.transpose() * eig.left_eigenvectors.as_ref().unwrap();
                let scaled_left_eigvectors      = eig.left_eigenvectors.as_ref().unwrap() * &eigvals;

                prop_assert!(relative_eq!(transformed_eigvectors, scaled_eigvectors, epsilon = 1.0e-5));
                prop_assert!(relative_eq!(transformed_left_eigvectors, scaled_left_eigvectors, epsilon = 1.0e-5));
            }
        }
    }

    #[test]
    fn eigensystem_static(m in matrix4()) {
        if let Some(eig) = Eigen::new(m, true, true) {
            // TODO: test the complex case too.
            if eig.eigenvalues_are_real() {
                let eigvals                = Matrix4::from_diagonal(&eig.eigenvalues_re);
                let transformed_eigvectors = m * eig.eigenvectors.unwrap();
                let scaled_eigvectors      = eig.eigenvectors.unwrap() * eigvals;

                let transformed_left_eigvectors = m.transpose() * eig.left_eigenvectors.unwrap();
                let scaled_left_eigvectors      = eig.left_eigenvectors.unwrap() * eigvals;

                prop_assert!(relative_eq!(transformed_eigvectors, scaled_eigvectors, epsilon = 1.0e-5));
                prop_assert!(relative_eq!(transformed_left_eigvectors, scaled_left_eigvectors, epsilon = 1.0e-5));
            }
        }
    }
}
