use na::DMatrix;
use nl::Schur;
use std::cmp;

use crate::proptest::*;
use proptest::{prop_assert, proptest};

proptest! {
    #[test]
    fn schur(n in PROPTEST_MATRIX_DIM) {
        let n = cmp::max(1, cmp::min(n, 10));
        let m = DMatrix::<f64>::new_random(n, n);

        let (vecs, vals) = Schur::new(m.clone()).unpack();

        prop_assert!(relative_eq!(&vecs * vals * vecs.transpose(), m, epsilon = 1.0e-7))
    }

    #[test]
    fn schur_static(m in matrix4()) {
        let (vecs, vals) = Schur::new(m.clone()).unpack();
        prop_assert!(relative_eq!(vecs * vals * vecs.transpose(), m, epsilon = 1.0e-7))
    }
}
