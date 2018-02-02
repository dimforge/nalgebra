use std::cmp;

use nl::LU;
use na::{DMatrix, DVector, Matrix3x4, Matrix4, Matrix4x3, Vector4};

quickcheck!{
    fn lup(m: DMatrix<f64>) -> bool {
        if m.len() != 0 {
            let lup = LU::new(m.clone());
            let l = lup.l();
            let u = lup.u();
            let mut computed1 = &l * &u;
            lup.permute(&mut computed1);

            let computed2 = lup.p() * l * u;

            relative_eq!(computed1, m, epsilon = 1.0e-7) &&
            relative_eq!(computed2, m, epsilon = 1.0e-7)
        }
        else {
            true
        }
    }

    fn lu_static(m: Matrix3x4<f64>) -> bool {
        let lup = LU::new(m);
        let l = lup.l();
        let u = lup.u();
        let mut computed1 = l * u;
        lup.permute(&mut computed1);

        let computed2 = lup.p() * l * u;

        relative_eq!(computed1, m, epsilon = 1.0e-7) &&
        relative_eq!(computed2, m, epsilon = 1.0e-7)
    }

    fn lu_solve(n: usize, nb: usize) -> bool {
        if n != 0 {
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

            relative_eq!(&m * sol1, b1, epsilon = 1.0e-7) &&
            relative_eq!(&m * sol2, b2, epsilon = 1.0e-7) &&
            relative_eq!(m.transpose() * tr_sol1, b1, epsilon = 1.0e-7) &&
            relative_eq!(m.transpose() * tr_sol2, b2, epsilon = 1.0e-7)
        }
        else {
            true
        }
    }

    fn lu_solve_static(m: Matrix4<f64>) -> bool {
        let lup = LU::new(m);
        let b1 = Vector4::new_random();
        let b2 = Matrix4x3::new_random();

        let sol1 = lup.solve(&b1).unwrap();
        let sol2 = lup.solve(&b2).unwrap();
        let tr_sol1 = lup.solve_transpose(&b1).unwrap();
        let tr_sol2 = lup.solve_transpose(&b2).unwrap();

        relative_eq!(m * sol1, b1, epsilon = 1.0e-7) &&
        relative_eq!(m * sol2, b2, epsilon = 1.0e-7) &&
        relative_eq!(m.transpose() * tr_sol1, b1, epsilon = 1.0e-7) &&
        relative_eq!(m.transpose() * tr_sol2, b2, epsilon = 1.0e-7)
    }

    fn lu_inverse(n: usize) -> bool {
        if n != 0 {
            let n = cmp::min(n, 25); // To avoid slowing down the test too much.
            let m = DMatrix::<f64>::new_random(n, n);

            if let Some(m1) = LU::new(m.clone()).inverse() {
                let id1 = &m  * &m1;
                let id2 = &m1 * &m;

                return id1.is_identity(1.0e-7) && id2.is_identity(1.0e-7);
            }
        }

        return true;
    }

    fn lu_inverse_static(m: Matrix4<f64>) -> bool {
        match LU::new(m.clone()).inverse() {
            Some(m1) => {
                let id1 = &m  * &m1;
                let id2 = &m1 * &m;

                id1.is_identity(1.0e-5) && id2.is_identity(1.0e-5)
            },
            None => true
        }
    }
}
