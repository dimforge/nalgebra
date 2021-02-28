use std::cmp;

use na::DMatrix;
use nl::SymmetricEigen;

use crate::proptest::*;
use proptest::{prop_assert, proptest};

proptest! {
    #[test]
    fn symmetric_eigen(n in PROPTEST_MATRIX_DIM) {
        let n = cmp::max(1, cmp::min(n, 10));
        let m = DMatrix::<f64>::new_random(n, n);
        let eig = SymmetricEigen::new(m.clone());
        let recomp = eig.recompose();
        prop_assert!(relative_eq!(m.lower_triangle(), recomp.lower_triangle(), epsilon = 1.0e-5))
    }

    #[test]
    fn symmetric_eigen_static(m in matrix4()) {
        let eig = SymmetricEigen::new(m);
        let recomp = eig.recompose();
        prop_assert!(relative_eq!(m.lower_triangle(), recomp.lower_triangle(), epsilon = 1.0e-5))
    }
}
