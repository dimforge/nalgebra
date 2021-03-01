use std::cmp;

use na::{DMatrix, Matrix4};
use nl::Hessenberg;

use crate::proptest::*;
use proptest::{prop_assert, proptest};

proptest! {
    #[test]
    fn hessenberg(n in PROPTEST_MATRIX_DIM) {
        let n = cmp::min(n, 25);
        let m = DMatrix::<f64>::new_random(n, n);

        if let Some(hess) = Hessenberg::new(m.clone()) {
            let h = hess.h();
            let p = hess.p();

            prop_assert!(relative_eq!(m, &p * h * p.transpose(), epsilon = 1.0e-7))
        }
    }

    #[test]
    fn hessenberg_static(m in matrix4()) {
        if let Some(hess) = Hessenberg::new(m) {
            let h = hess.h();
            let p = hess.p();

            prop_assert!(relative_eq!(m, p * h * p.transpose(), epsilon = 1.0e-7))
        }
    }
}
