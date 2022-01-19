use na::DMatrix;
use nl::QZ;
use std::cmp;

use crate::proptest::*;
use proptest::{prop_assert, proptest};

proptest! {
    #[test]
    fn qz(n in PROPTEST_MATRIX_DIM) {
        let n = cmp::max(1, cmp::min(n, 10));
        let a = DMatrix::<f64>::new_random(n, n);
        let b = DMatrix::<f64>::new_random(n, n);

        let (vsl,s,t,vsr) = QZ::new(a.clone(), b.clone()).unpack();

        prop_assert!(relative_eq!(&vsl * s * vsr.transpose(), a, epsilon = 1.0e-7));
        prop_assert!(relative_eq!(vsl * t * vsr.transpose(), b, epsilon = 1.0e-7))
    }

    #[test]
    fn qz_static(a in matrix4(), b in matrix4()) {
        let (vsl,s,t,vsr) = QZ::new(a.clone(), b.clone()).unpack();
        prop_assert!(relative_eq!(&vsl * s * vsr.transpose(), a, epsilon = 1.0e-7));
        prop_assert!(relative_eq!(vsl * t * vsr.transpose(), b, epsilon = 1.0e-7))
    }
}
