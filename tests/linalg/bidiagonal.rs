#![cfg(feature = "arbitrary")]

use na::{DMatrix, Matrix2, Matrix4, Matrix5x3, Matrix3x5};

quickcheck! {
    fn bidiagonal(m: DMatrix<f64>) -> bool {
        if m.len() == 0  {
            return true;
        }

        let bidiagonal = m.clone().bidiagonalize();
        let (u, d, v_t) = bidiagonal.unpack();

        println!("{}{}{}", &u, &d, &v_t);
        println!("{:.7}{:.7}", &u * &d * &v_t, m);

        relative_eq!(m, &u * d * &v_t, epsilon = 1.0e-7)
    }

    fn bidiagonal_static_5_3(m: Matrix5x3<f64>) -> bool {
        let bidiagonal = m.bidiagonalize();
        let (u, d, v_t) = bidiagonal.unpack();

        println!("{}{}{}", &u, &d, &v_t);
        println!("{:.7}{:.7}", &u * &d * &v_t, m);

        relative_eq!(m, &u * d * &v_t, epsilon = 1.0e-7)
    }

    fn bidiagonal_static_3_5(m: Matrix3x5<f64>) -> bool {
        let bidiagonal = m.bidiagonalize();
        let (u, d, v_t) = bidiagonal.unpack();

        println!("{}{}{}", &u, &d, &v_t);
        println!("{:.7}{:.7}", &u * &d * &v_t, m);

        relative_eq!(m, &u * d * &v_t, epsilon = 1.0e-7)
    }

    fn bidiagonal_static_square(m: Matrix4<f64>) -> bool {
        let bidiagonal = m.bidiagonalize();
        let (u, d, v_t) = bidiagonal.unpack();

        println!("{}{}{}", &u, &d, &v_t);
        println!("{:.7}{:.7}", &u * &d * &v_t, m);

        relative_eq!(m, &u * d * &v_t, epsilon = 1.0e-7)
    }

    fn bidiagonal_static_square_2x2(m: Matrix2<f64>) -> bool {
        let bidiagonal = m.bidiagonalize();
        let (u, d, v_t) = bidiagonal.unpack();

        println!("{}{}{}", &u, &d, &v_t);
        println!("{:.7}{:.7}", &u * &d * &v_t, m);

        relative_eq!(m, &u * d * &v_t, epsilon = 1.0e-7)
    }
}
