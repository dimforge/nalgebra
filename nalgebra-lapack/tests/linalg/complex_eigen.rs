use crate::proptest::*;
use na::DMatrix;
use nalgebra_lapack::Eigen;
use num_complex::Complex;
use proptest::prelude::*;

fn square_matrix() -> impl Strategy<Value = DMatrix<f64>> {
    PROPTEST_MATRIX_DIM.prop_flat_map(|rows| matrix(PROPTEST_F64, rows..=rows, rows..=rows))
}

// a simple way to get a matrix with only real eigenvalues is to make it symmetic by A^T * A
fn square_matrix_with_real_eigenvalues() -> impl Strategy<Value = DMatrix<f64>> {
    square_matrix().prop_flat_map(|mat| Just(mat.transpose() * mat))
}

proptest! {
    #[test]
    // we have to use a matrix with only real eigenvalues for the get_real_elements()
    // API to make sense. Tbh, @todo(geo-ant) this should be refactored in future,
    // it's a bit unintuitive...
    fn real_eigenvalues_and_eigenvectors(a in square_matrix_with_real_eigenvalues()) {
        let eigen = Eigen::new(a.clone(), true, true).unwrap();
        let (eigenvalues, Some(left_vectors), Some(right_vectors)) = eigen.get_real_elements() else {
            panic!("Eigenvector calculation failed");
        };
        // Test A * v = lambda * v (right-vectors)
        // and v^T A = lambda v^T (left vectors)
        for ((lambda,left_vec), right_vec) in eigenvalues.iter().zip(left_vectors.iter()).zip(right_vectors.iter()) {
            prop_assert!(relative_eq!(left_vec.transpose()* &a , left_vec.transpose().scale(*lambda), epsilon = 1e-6));
            prop_assert!(relative_eq!(&a * right_vec, right_vec.scale(*lambda), epsilon = 1e-6));
        }
    }

    #[test]
    fn complex_eigenvalues_and_eigenvectors(a in square_matrix()) {
        let eigen = Eigen::new(a.clone(), true, true).unwrap();
        let a_complex = a.map(|elem| Complex::new(elem, 0.0));
        let (maybe_eigenvalues, maybe_left_vectors, maybe_right_vectors) = eigen.get_complex_elements();

        let Some(eigenvalues) = maybe_eigenvalues else {
            // this means that the matrix only has real eigenvalues
            // this might be a bit hacky, but we repeat the test above
            let (eigenvalues, Some(left_vectors), Some(right_vectors)) = eigen.get_real_elements() else {
                panic!("Eigenvector calculation failed");
            };
            // Test A * v = lambda * v (right-vectors)
            // and v^T A = lambda v^T (left vectors)
            for ((lambda,left_vec), right_vec) in eigenvalues.iter().zip(left_vectors.iter()).zip(right_vectors.iter()) {
                prop_assert!(relative_eq!(left_vec.transpose()* &a , left_vec.transpose().scale(*lambda), epsilon = 1e-6));
                prop_assert!(relative_eq!(&a * right_vec, right_vec.scale(*lambda), epsilon = 1e-6));
            }
            return Ok(());
        };

        let left_vectors = maybe_left_vectors.unwrap();
        let right_vectors = maybe_right_vectors.unwrap();

        for ((lambda, left_vec), right_vec) in eigenvalues
            .iter()
            .zip(left_vectors.iter())
            .zip(right_vectors.iter())
        {
            // Test A * v = lambda * v (right eigenvectors)
            let av = &a_complex * right_vec;
            let lambda_v = right_vec * *lambda;

            let av_real = av.map(|x| x.re);
            let lambda_v_real = lambda_v.map(|x| x.re);
            prop_assert!(relative_eq!(av_real, lambda_v_real, epsilon = 1e-6));

            let av_imag = av.map(|x| x.im);
            let lambda_v_imag = lambda_v.map(|x| x.im);
            prop_assert!(relative_eq!(av_imag, lambda_v_imag, epsilon = 1e-6));

            // Test v^H * A = lambda * v^H (left eigenvectors, note the H [Hermitian] instead of T [Transpose])
            let vt_a = left_vec.adjoint() * &a_complex;
            let lambda_vt = left_vec.adjoint() * *lambda;

            let vt_a_real = vt_a.map(|x| x.re);
            let lambda_vt_real = lambda_vt.map(|x| x.re);
            prop_assert!(relative_eq!(vt_a_real, lambda_vt_real, epsilon = 1e-6));

            let vt_a_imag = vt_a.map(|x| x.im);
            let lambda_vt_imag = lambda_vt.map(|x| x.im);
            prop_assert!(relative_eq!(vt_a_imag, lambda_vt_imag, epsilon = 1e-6));
        }
    }
}
