#![cfg(all(feature = "arbitrary", feature = "debug"))]

use std::cmp;
use na::{DMatrix, Matrix4x3, DVector, Vector4};
use na::dimension::U4;
use na::debug::RandomSDP;

quickcheck! {
    fn cholesky(m: RandomSDP<f64>) -> bool {
        let mut m = m.unwrap();

        // Put garbage on the upper triangle to make sure it is not read by the decomposition.
        m.fill_upper_triangle(23.0, 1);

        let l = m.clone().cholesky().unwrap().unpack();
        m.fill_upper_triangle_with_lower_triangle();
        relative_eq!(m, &l * l.transpose(), epsilon = 1.0e-7)
    }

    fn cholesky_static(m: RandomSDP<f64, U4>) -> bool {
        let m = m.unwrap();
        let chol = m.cholesky().unwrap();
        let l    = chol.unpack();

        if !relative_eq!(m, &l * l.transpose(), epsilon = 1.0e-7) {
            false
        }
        else {
            true
        }
    }


    fn cholesky_solve(m: RandomSDP<f64>, nb: usize) -> bool {
        let m  = m.unwrap();
        let n  = m.nrows();
        let nb = cmp::min(nb, 50); // To avoid slowing down the test too much.

        let chol = m.clone().cholesky().unwrap();
        let b1 = DVector::new_random(n);
        let b2 = DMatrix::new_random(n, nb);

        let sol1 = chol.solve(&b1);
        let sol2 = chol.solve(&b2);

        relative_eq!(&m * &sol1, b1, epsilon = 1.0e-7) &&
        relative_eq!(&m * &sol2, b2, epsilon = 1.0e-7)
    }

    fn cholesky_solve_static(m: RandomSDP<f64, U4>) -> bool {
        let m = m.unwrap();
        let chol = m.clone().cholesky().unwrap();
        let b1 = Vector4::new_random();
        let b2 = Matrix4x3::new_random();

        let sol1 = chol.solve(&b1);
        let sol2 = chol.solve(&b2);

        relative_eq!(m * sol1, b1, epsilon = 1.0e-7) &&
        relative_eq!(m * sol2, b2, epsilon = 1.0e-7)
    }

    fn cholesky_inverse(m: RandomSDP<f64>) -> bool {
        let m = m.unwrap();

        let m1 = m.clone().cholesky().unwrap().inverse();
        let id1 = &m  * &m1;
        let id2 = &m1 * &m;

        id1.is_identity(1.0e-7) && id2.is_identity(1.0e-7)
    }

    fn cholesky_inverse_static(m: RandomSDP<f64, U4>) -> bool {
        let m = m.unwrap();
        let m1 = m.clone().cholesky().unwrap().inverse();
        let id1 = &m  * &m1;
        let id2 = &m1 * &m;

        id1.is_identity(1.0e-7) && id2.is_identity(1.0e-7)
    }
}
