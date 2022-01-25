use na::DMatrix;
use nl::{GE, QZ};
use std::cmp;

use crate::proptest::*;
use proptest::{prop_assert, proptest};

proptest! {
    #[test]
    fn qz(n in PROPTEST_MATRIX_DIM) {
        let n = cmp::max(1, cmp::min(n, 10));
        let a = DMatrix::<f64>::new_random(n, n);
        let b = DMatrix::<f64>::new_random(n, n);

        let qz =  QZ::new(a.clone(), b.clone());
        let (vsl,s,t,vsr) = qz.clone().unpack();
        let eigenvalues = qz.eigenvalues();

        let ge = GE::new(a.clone(), b.clone());
        let eigenvalues2 = ge.eigenvalues();

        prop_assert!(relative_eq!(&vsl * s * vsr.transpose(), a.clone(), epsilon = 1.0e-7));
        prop_assert!(relative_eq!(vsl * t * vsr.transpose(), b.clone(), epsilon = 1.0e-7));
        prop_assert!(eigenvalues == eigenvalues2);
    }

    #[test]
    fn qz_static(a in matrix4(), b in matrix4()) {
        let qz = QZ::new(a.clone(), b.clone());
        let ge = GE::new(a.clone(), b.clone());
        let (vsl,s,t,vsr) = qz.unpack();
        let eigenvalues = qz.eigenvalues();
        let eigenvalues2 = ge.eigenvalues();

        prop_assert!(relative_eq!(&vsl * s * vsr.transpose(), a, epsilon = 1.0e-7));
        prop_assert!(relative_eq!(vsl * t * vsr.transpose(), b, epsilon = 1.0e-7));

        prop_assert!(eigenvalues == eigenvalues2);

    }
}
