#![cfg(feature = "arbitrary")]

use na::{DMatrix, Matrix2, Matrix3x5, Matrix4, Matrix5x3};
use core::helper::{RandScalar, RandComplex};

quickcheck! {
    fn bidiagonal(m: DMatrix<RandComplex<f64>>) -> bool {
        let m = m.map(|e| e.0);
        if m.len() == 0  {
            return true;
        }

        let bidiagonal = m.clone().bidiagonalize();
        let (u, d, v_t) = bidiagonal.unpack();

        println!("{}{}{}", &u, &d, &v_t);
        println!("{:.7}{:.7}", &u * &d * &v_t, m);

        relative_eq!(m, &u * d * &v_t, epsilon = 1.0e-7)
    }

    fn bidiagonal_static_5_3(m: Matrix5x3<RandComplex<f64>>) -> bool {
        let m = m.map(|e| e.0);
        let bidiagonal = m.bidiagonalize();
        let (u, d, v_t) = bidiagonal.unpack();

        println!("{}{}{}", &u, &d, &v_t);
        println!("{:.7}{:.7}", &u * &d * &v_t, m);

        relative_eq!(m, &u * d * &v_t, epsilon = 1.0e-7)
    }

    fn bidiagonal_static_3_5(m: Matrix3x5<RandComplex<f64>>) -> bool {
        let m = m.map(|e| e.0);
        let bidiagonal = m.bidiagonalize();
        let (u, d, v_t) = bidiagonal.unpack();

        println!("{}{}{}", &u, &d, &v_t);
        println!("{:.7}{:.7}", &u * &d * &v_t, m);

        relative_eq!(m, &u * d * &v_t, epsilon = 1.0e-7)
    }

    fn bidiagonal_static_square(m: Matrix4<RandComplex<f64>>) -> bool {
        let m = m.map(|e| e.0);
        let bidiagonal = m.bidiagonalize();
        let (u, d, v_t) = bidiagonal.unpack();

        println!("{}{}{}", &u, &d, &v_t);
        println!("{:.7}{:.7}", &u * &d * &v_t, m);

        relative_eq!(m, &u * d * &v_t, epsilon = 1.0e-7)
    }

    fn bidiagonal_static_square_2x2(m: Matrix2<RandComplex<f64>>) -> bool {
        let m = m.map(|e| e.0);
        let bidiagonal = m.bidiagonalize();
        let (u, d, v_t) = bidiagonal.unpack();

        println!("{}{}{}", &u, &d, &v_t);
        println!("{:.7}{:.7}", &u * &d * &v_t, m);

        relative_eq!(m, &u * d * &v_t, epsilon = 1.0e-7)
    }
}
