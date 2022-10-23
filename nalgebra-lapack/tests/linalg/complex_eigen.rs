use std::cmp;

use na::{Matrix3};
use nalgebra_lapack::Eigen;

use crate::proptest::*;
use proptest::{prop_assert, proptest};

proptest! {
    //#[test]
    // fn complex_eigen() {
    //     let n = cmp::max(1, cmp::min(n, 10));
    //     let m = DMatrix::<f64>::new_random(n, n);
    //     let eig = SymmetricEigen::new(m.clone());
    //     let recomp = eig.recompose();
    //     prop_assert!(relative_eq!(m.lower_triangle(), recomp.lower_triangle(), epsilon = 1.0e-5))
    // }

}
