use na::{DMatrix, Matrix3x5};
use nl::SVD;

use crate::proptest::*;
use proptest::{prop_assert, proptest};

proptest! {
    #[test]
    fn svd(m in dmatrix()) {
        let svd = SVD::new(m.clone()).unwrap();
        let sm  = DMatrix::from_partial_diagonal(m.nrows(), m.ncols(), svd.singular_values.as_slice());

        let reconstructed_m = &svd.u * sm * &svd.vt;
        let reconstructed_m2 = svd.recompose();

        prop_assert!(relative_eq!(reconstructed_m, m, epsilon = 1.0e-7));
        prop_assert!(relative_eq!(reconstructed_m2, reconstructed_m, epsilon = 1.0e-7));
    }

    #[test]
    fn svd_static(m in matrix3x5()) {
        let svd = SVD::new(m).unwrap();
        let sm  = Matrix3x5::from_partial_diagonal(svd.singular_values.as_slice());

        let reconstructed_m  = &svd.u * &sm * &svd.vt;
        let reconstructed_m2 = svd.recompose();

        prop_assert!(relative_eq!(reconstructed_m, m, epsilon = 1.0e-7));
        prop_assert!(relative_eq!(reconstructed_m2, m, epsilon = 1.0e-7));
    }

    #[test]
    fn pseudo_inverse(m in dmatrix()) {
        let svd = SVD::new(m.clone()).unwrap();
        let im  = svd.pseudo_inverse(1.0e-7);

        if m.nrows() <= m.ncols() {
            prop_assert!((&m * &im).is_identity(1.0e-7));
        }

        if m.nrows() >= m.ncols() {
            prop_assert!((im * m).is_identity(1.0e-7));
        }
    }

    #[test]
    fn pseudo_inverse_static(m in matrix3x5()) {
        let svd = SVD::new(m).unwrap();
        let im  = svd.pseudo_inverse(1.0e-7);

        prop_assert!((m * im).is_identity(1.0e-7))
    }
}
