#![cfg(feature = "arbitrary")]

use std::cmp;
use na::{DMatrix, Matrix4, Matrix4x3, Matrix5x3, Matrix3x5,
         DVector, Vector4};

quickcheck! {
    fn qr(m: DMatrix<f64>) -> bool {
        let qr = m.clone().qr();
        let q  = qr.q();
        let r  = qr.r();

        relative_eq!(m, &q * r, epsilon = 1.0e-7) &&
        q.is_orthogonal(1.0e-7)
    }

    fn qr_static_5_3(m: Matrix5x3<f64>) -> bool {
        let qr = m.qr();
        let q  = qr.q();
        let r  = qr.r();

        relative_eq!(m, q * r, epsilon = 1.0e-7) &&
        q.is_orthogonal(1.0e-7)
    }

    fn qr_static_3_5(m: Matrix3x5<f64>) -> bool {
        let qr = m.qr();
        let q  = qr.q();
        let r  = qr.r();

        relative_eq!(m, q * r, epsilon = 1.0e-7) &&
        q.is_orthogonal(1.0e-7)
    }

    fn qr_static_square(m: Matrix4<f64>) -> bool {
        let qr = m.qr();
        let q  = qr.q();
        let r  = qr.r();

        println!("{}{}{}{}", q, r, q * r, m);

        relative_eq!(m, q * r, epsilon = 1.0e-7) &&
        q.is_orthogonal(1.0e-7)
    }

    fn qr_solve(n: usize, nb: usize) -> bool {
        if n != 0 && nb != 0 {
            let n  = cmp::min(n, 50);  // To avoid slowing down the test too much.
            let nb = cmp::min(nb, 50); // To avoid slowing down the test too much.
            let m  = DMatrix::<f64>::new_random(n, n);

            let qr = m.clone().qr();
            let b1 = DVector::new_random(n);
            let b2 = DMatrix::new_random(n, nb);

            if qr.is_invertible() {
                let sol1 = qr.solve(&b1).unwrap();
                let sol2 = qr.solve(&b2).unwrap();

                return relative_eq!(&m * sol1, b1, epsilon = 1.0e-6) &&
                    relative_eq!(&m * sol2, b2, epsilon = 1.0e-6)
            }
        }

        return true;
    }

    fn qr_solve_static(m: Matrix4<f64>) -> bool {
         let qr = m.qr();
         let b1 = Vector4::new_random();
         let b2 = Matrix4x3::new_random();

         if qr.is_invertible() {
             let sol1 = qr.solve(&b1).unwrap();
             let sol2 = qr.solve(&b2).unwrap();

             relative_eq!(m * sol1, b1, epsilon = 1.0e-6) &&
             relative_eq!(m * sol2, b2, epsilon = 1.0e-6)
         }
         else {
             false
         }
    }

    fn qr_inverse(n: usize) -> bool {
        let n = cmp::max(1, cmp::min(n, 15)); // To avoid slowing down the test too much.
        let m = DMatrix::<f64>::new_random(n, n);

        if let Some(m1) = m.clone().qr().try_inverse() {
            let id1 = &m  * &m1;
            let id2 = &m1 * &m;

            id1.is_identity(1.0e-5) && id2.is_identity(1.0e-5)
        }
        else {
            true
        }
    }

    fn qr_inverse_static(m: Matrix4<f64>) -> bool {
        let qr  = m.qr();

        if let Some(m1) = qr.try_inverse() {
            let id1 = &m  * &m1;
            let id2 = &m1 * &m;

            id1.is_identity(1.0e-5) && id2.is_identity(1.0e-5)
        }
        else {
            true
        }
    }
}
