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

        if let Some(schur) = Schur::try_new(m.clone()) {
            let (vecs, vals) = schur.unpack();
            prop_assert!(relative_eq!(&vecs * vals * vecs.transpose(), m, epsilon = 1.0e-5))
        }
    }

    #[test]
    fn schur_static(m in matrix4()) {
        if let Some(schur) = Schur::try_new(m.clone()) {
            let (vecs, vals) = schur.unpack();
            prop_assert!(relative_eq!(vecs * vals * vecs.transpose(), m, epsilon = 1.0e-5))
        }
    }
}
