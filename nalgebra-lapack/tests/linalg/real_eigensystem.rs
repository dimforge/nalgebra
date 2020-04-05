use std::cmp;

use na::{DMatrix, Matrix4};
use nl::Eigen;

quickcheck! {
    fn eigensystem(n: usize) -> bool {
        if n != 0 {
            let n = cmp::min(n, 25);
            let m = DMatrix::<f64>::new_random(n, n);

            match Eigen::new(m.clone(), true, true) {
                Some(eig) => {
                    let eigvals                = DMatrix::from_diagonal(&eig.eigenvalues);
                    let transformed_eigvectors = &m * eig.eigenvectors.as_ref().unwrap();
                    let scaled_eigvectors      = eig.eigenvectors.as_ref().unwrap() * &eigvals;

                    let transformed_left_eigvectors = m.transpose() * eig.left_eigenvectors.as_ref().unwrap();
                    let scaled_left_eigvectors      = eig.left_eigenvectors.as_ref().unwrap() * &eigvals;

                    relative_eq!(transformed_eigvectors, scaled_eigvectors, epsilon = 1.0e-7) &&
                    relative_eq!(transformed_left_eigvectors, scaled_left_eigvectors, epsilon = 1.0e-7)
                },
                None => true
            }
        }
        else {
            true
        }
    }

    fn eigensystem_static(m: Matrix4<f64>) -> bool {
        match Eigen::new(m, true, true) {
            Some(eig) => {
                let eigvals                = Matrix4::from_diagonal(&eig.eigenvalues);
                let transformed_eigvectors = m * eig.eigenvectors.unwrap();
                let scaled_eigvectors      = eig.eigenvectors.unwrap() * eigvals;

                let transformed_left_eigvectors = m.transpose() * eig.left_eigenvectors.unwrap();
                let scaled_left_eigvectors      = eig.left_eigenvectors.unwrap() * eigvals;

                relative_eq!(transformed_eigvectors, scaled_eigvectors, epsilon = 1.0e-7) &&
                relative_eq!(transformed_left_eigvectors, scaled_left_eigvectors, epsilon = 1.0e-7)
            },
            None => true
        }
    }
}
