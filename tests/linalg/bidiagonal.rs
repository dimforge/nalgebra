#![cfg(feature = "arbitrary")]

macro_rules! gen_tests(
    ($module: ident, $scalar: ty) => {
        mod $module {
            use na::{DMatrix, Matrix2, Matrix3x5, Matrix4, Matrix5x3};
            #[allow(unused_imports)]
            use crate::core::helper::{RandScalar, RandComplex};

            quickcheck! {
                fn bidiagonal(m: DMatrix<$scalar>) -> bool {
                    let m = m.map(|e| e.0);
                    if m.len() == 0  {
                        return true;
                    }

                    let bidiagonal = m.clone().bidiagonalize();
                    let (u, d, v_t) = bidiagonal.unpack();

                    relative_eq!(m, &u * d * &v_t, epsilon = 1.0e-7)
                }

                fn bidiagonal_static_5_3(m: Matrix5x3<$scalar>) -> bool {
                    let m = m.map(|e| e.0);
                    let bidiagonal = m.bidiagonalize();
                    let (u, d, v_t) = bidiagonal.unpack();

                    relative_eq!(m, &u * d * &v_t, epsilon = 1.0e-7)
                }

                fn bidiagonal_static_3_5(m: Matrix3x5<$scalar>) -> bool {
                    let m = m.map(|e| e.0);
                    let bidiagonal = m.bidiagonalize();
                    let (u, d, v_t) = bidiagonal.unpack();

                    relative_eq!(m, &u * d * &v_t, epsilon = 1.0e-7)
                }

                fn bidiagonal_static_square(m: Matrix4<$scalar>) -> bool {
                    let m = m.map(|e| e.0);
                    let bidiagonal = m.bidiagonalize();
                    let (u, d, v_t) = bidiagonal.unpack();

                    relative_eq!(m, &u * d * &v_t, epsilon = 1.0e-7)
                }

                fn bidiagonal_static_square_2x2(m: Matrix2<$scalar>) -> bool {
                    let m = m.map(|e| e.0);
                    let bidiagonal = m.bidiagonalize();
                    let (u, d, v_t) = bidiagonal.unpack();

                    relative_eq!(m, &u * d * &v_t, epsilon = 1.0e-7)
                }
            }
        }
    }
);

gen_tests!(complex, RandComplex<f64>);
gen_tests!(f64, RandScalar<f64>);

#[test]
fn bidiagonal_identity() {
    let m = na::DMatrix::<f64>::identity(10, 10);
    let bidiagonal = m.clone().bidiagonalize();
    let (u, d, v_t) = bidiagonal.unpack();
    assert_eq!(m, &u * d * &v_t);

    let m = na::DMatrix::<f64>::identity(10, 15);
    let bidiagonal = m.clone().bidiagonalize();
    let (u, d, v_t) = bidiagonal.unpack();
    assert_eq!(m, &u * d * &v_t);

    let m = na::DMatrix::<f64>::identity(15, 10);
    let bidiagonal = m.clone().bidiagonalize();
    let (u, d, v_t) = bidiagonal.unpack();
    assert_eq!(m, &u * d * &v_t);
}
