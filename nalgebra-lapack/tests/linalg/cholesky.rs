use std::cmp;

use nl::Cholesky;
use na::{DMatrix, DVector, Matrix3, Matrix4, Matrix4x3, Vector4};

quickcheck!{
    fn cholesky(m: DMatrix<f64>) -> bool {
        if m.len() != 0 {
            let m = &m * m.transpose();
            if let Some(chol) = Cholesky::new(m.clone()) {
                let l = chol.unpack();
                let reconstructed_m = &l * l.transpose();

                return relative_eq!(reconstructed_m, m, epsilon = 1.0e-7)
            }
        }
        return true
    }

    fn cholesky_static(m: Matrix3<f64>) -> bool {
        let m = &m * m.transpose();
        if let Some(chol) = Cholesky::new(m) {
            let l = chol.unpack();
            let reconstructed_m = &l * l.transpose();

            relative_eq!(reconstructed_m, m, epsilon = 1.0e-7)
        }
        else {
            false
        }
    }

    fn cholesky_solve(n: usize, nb: usize) -> bool {
        if n != 0 {
            let n  = cmp::min(n, 15);  // To avoid slowing down the test too much.
            let nb = cmp::min(nb, 15); // To avoid slowing down the test too much.
            let m  = DMatrix::<f64>::new_random(n, n);
            let m   = &m * m.transpose();

            if let Some(chol) = Cholesky::new(m.clone()) {
                let b1 = DVector::new_random(n);
                let b2 = DMatrix::new_random(n, nb);

                let sol1 = chol.solve(&b1).unwrap();
                let sol2 = chol.solve(&b2).unwrap();

                return relative_eq!(&m * sol1, b1, epsilon = 1.0e-6) &&
                       relative_eq!(&m * sol2, b2, epsilon = 1.0e-6)
            }
        }

        return true;
    }

    fn cholesky_solve_static(m: Matrix4<f64>) -> bool {
        let m = &m * m.transpose();
        match Cholesky::new(m) {
            Some(chol) => {
                let b1 = Vector4::new_random();
                let b2 = Matrix4x3::new_random();

                let sol1 = chol.solve(&b1).unwrap();
                let sol2 = chol.solve(&b2).unwrap();

                relative_eq!(m * sol1, b1, epsilon = 1.0e-7) &&
                relative_eq!(m * sol2, b2, epsilon = 1.0e-7)
            },
            None => true
        }
    }

    fn cholesky_inverse(n: usize) -> bool {
        if n != 0 {
            let n = cmp::min(n, 15); // To avoid slowing down the test too much.
            let m = DMatrix::<f64>::new_random(n, n);
            let m = &m * m.transpose();

            if let Some(m1) = Cholesky::new(m.clone()).unwrap().inverse() {
                let id1 = &m  * &m1;
                let id2 = &m1 * &m;

                return id1.is_identity(1.0e-6) && id2.is_identity(1.0e-6);
            }
        }

        return true;
    }

    fn cholesky_inverse_static(m: Matrix4<f64>) -> bool {
        let m = m * m.transpose();
        match Cholesky::new(m.clone()).unwrap().inverse() {
            Some(m1) => {
                let id1 = &m  * &m1;
                let id2 = &m1 * &m;

                id1.is_identity(1.0e-5) && id2.is_identity(1.0e-5)
            },
            None => true
        }
    }
}
