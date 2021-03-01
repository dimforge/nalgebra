#![cfg(feature = "proptest-support")]

macro_rules! gen_tests(
    ($module: ident, $scalar: expr) => {
        mod $module {
            #[allow(unused_imports)]
            use crate::core::helper::{RandScalar, RandComplex};

            use crate::proptest::*;
            use proptest::{prop_assert, proptest};

            proptest! {
                #[test]
                fn bidiagonal(m in dmatrix_($scalar)) {
                    let bidiagonal = m.clone().bidiagonalize();
                    let (u, d, v_t) = bidiagonal.unpack();

                    prop_assert!(relative_eq!(m, &u * d * &v_t, epsilon = 1.0e-7))
                }

                #[test]
                fn bidiagonal_static_5_3(m in matrix5x3_($scalar)) {
                    let bidiagonal = m.bidiagonalize();
                    let (u, d, v_t) = bidiagonal.unpack();

                    prop_assert!(relative_eq!(m, &u * d * &v_t, epsilon = 1.0e-7))
                }

                #[test]
                fn bidiagonal_static_3_5(m in matrix3x5_($scalar)) {
                    let bidiagonal = m.bidiagonalize();
                    let (u, d, v_t) = bidiagonal.unpack();

                    prop_assert!(relative_eq!(m, &u * d * &v_t, epsilon = 1.0e-7))
                }

                #[test]
                fn bidiagonal_static_square(m in matrix4_($scalar)) {
                    let bidiagonal = m.bidiagonalize();
                    let (u, d, v_t) = bidiagonal.unpack();

                    prop_assert!(relative_eq!(m, &u * d * &v_t, epsilon = 1.0e-7))
                }

                #[test]
                fn bidiagonal_static_square_2x2(m in matrix2_($scalar)) {
                    let bidiagonal = m.bidiagonalize();
                    let (u, d, v_t) = bidiagonal.unpack();

                    prop_assert!(relative_eq!(m, &u * d * &v_t, epsilon = 1.0e-7))
                }
            }
        }
    }
);

gen_tests!(complex, complex_f64());
gen_tests!(f64, PROPTEST_F64);

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
