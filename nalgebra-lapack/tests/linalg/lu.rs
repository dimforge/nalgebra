use std::cmp;

use na::{DMatrix, DVector, Matrix4x3, Vector4};
use nl::LU;

use crate::proptest::*;
use proptest::{prop_assert, proptest};

proptest! {
    #[test]
    fn lup(m in dmatrix()) {
        let lup = LU::new(m.clone());
        let l = lup.l();
        let u = lup.u();
        let mut computed1 = &l * &u;
        lup.permute(&mut computed1);

        let computed2 = lup.p() * l * u;

        prop_assert!(relative_eq!(computed1, m, epsilon = 1.0e-7));
        prop_assert!(relative_eq!(computed2, m, epsilon = 1.0e-7));
    }

    #[test]
    fn lu_static(m in matrix3x5()) {
        let lup = LU::new(m);
        let l = lup.l();
        let u = lup.u();
        let mut computed1 = l * u;
        lup.permute(&mut computed1);

        let computed2 = lup.p() * l * u;

        prop_assert!(relative_eq!(computed1, m, epsilon = 1.0e-7));
        prop_assert!(relative_eq!(computed2, m, epsilon = 1.0e-7));
    }

    #[test]
    fn lu_solve(n in PROPTEST_MATRIX_DIM, nb in PROPTEST_MATRIX_DIM) {
        let n  = cmp::min(n, 25);  // To avoid slowing down the test too much.
        let nb = cmp::min(nb, 25); // To avoid slowing down the test too much.
        let m  = DMatrix::<f64>::new_random(n, n);

        let lup = LU::new(m.clone());
        let b1 = DVector::new_random(n);
        let b2 = DMatrix::new_random(n, nb);

        let sol1 = lup.solve(&b1).unwrap();
        let sol2 = lup.solve(&b2).unwrap();

        let tr_sol1 = lup.solve_transpose(&b1).unwrap();
        let tr_sol2 = lup.solve_transpose(&b2).unwrap();

        prop_assert!(relative_eq!(&m * sol1, b1, epsilon = 1.0e-7));
        prop_assert!(relative_eq!(&m * sol2, b2, epsilon = 1.0e-7));
        prop_assert!(relative_eq!(m.transpose() * tr_sol1, b1, epsilon = 1.0e-7));
        prop_assert!(relative_eq!(m.transpose() * tr_sol2, b2, epsilon = 1.0e-7));
    }

    #[test]
    fn lu_solve_static(m in matrix4()) {
        let lup = LU::new(m);
        let b1 = Vector4::new_random();
        let b2 = Matrix4x3::new_random();

        let sol1 = lup.solve(&b1).unwrap();
        let sol2 = lup.solve(&b2).unwrap();
        let tr_sol1 = lup.solve_transpose(&b1).unwrap();
        let tr_sol2 = lup.solve_transpose(&b2).unwrap();

        prop_assert!(relative_eq!(m * sol1, b1, epsilon = 1.0e-7));
        prop_assert!(relative_eq!(m * sol2, b2, epsilon = 1.0e-7));
        prop_assert!(relative_eq!(m.transpose() * tr_sol1, b1, epsilon = 1.0e-7));
        prop_assert!(relative_eq!(m.transpose() * tr_sol2, b2, epsilon = 1.0e-7));
    }

    #[test]
    fn lu_inverse(n in PROPTEST_MATRIX_DIM) {
        let n = cmp::min(n, 25); // To avoid slowing down the test too much.
        let m = DMatrix::<f64>::new_random(n, n);

        if let Some(m1) = LU::new(m.clone()).inverse() {
            let id1 = &m  * &m1;
            let id2 = &m1 * &m;

            prop_assert!(id1.is_identity(1.0e-7) && id2.is_identity(1.0e-7));
        }
    }

    #[test]
    fn lu_inverse_static(m in matrix4()) {
        if let Some(m1) = LU::new(m.clone()).inverse() {
            let id1 = &m  * &m1;
            let id2 = &m1 * &m;

            prop_assert!(id1.is_identity(1.0e-5) && id2.is_identity(1.0e-5))
        }
    }
}
