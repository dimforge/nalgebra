#![cfg(feature = "proptest-support")]

macro_rules! gen_tests(
    ($module: ident, $scalar: expr, $scalar_type: ty) => {
        mod $module {
            use na::{DMatrix, DVector, Matrix4x3, Matrix5x2, Vector4, Vector5};
            use std::cmp;
            #[allow(unused_imports)]
            use crate::core::helper::{RandScalar, RandComplex};
            use crate::proptest::*;
            use proptest::{prop_assert, proptest};

            proptest! {
                #[test]
                fn qr(m in dmatrix_($scalar)) {
                    let qr = m.clone().qr();
                    let q  = qr.q();
                    let r  = qr.r();

                    prop_assert!(relative_eq!(m, &q * r, epsilon = 1.0e-7));
                    prop_assert!(q.is_orthogonal(1.0e-7));
                }

                #[test]
                fn qr_static_5_3(m in matrix5x3_($scalar)) {
                    let qr = m.qr();
                    let q  = qr.q();
                    let r  = qr.r();

                    prop_assert!(relative_eq!(m, q * r, epsilon = 1.0e-7));
                    prop_assert!(q.is_orthogonal(1.0e-7));
                }

                #[test]
                fn qr_static_3_5(m in matrix3x5_($scalar)) {
                    let qr = m.qr();
                    let q  = qr.q();
                    let r  = qr.r();

                    prop_assert!(relative_eq!(m, q * r, epsilon = 1.0e-7));
                    prop_assert!(q.is_orthogonal(1.0e-7));
                }

                #[test]
                fn qr_static_square(m in matrix4_($scalar)) {
                    let qr = m.qr();
                    let q  = qr.q();
                    let r  = qr.r();

                    prop_assert!(relative_eq!(m, q * r, epsilon = 1.0e-7));
                    prop_assert!(q.is_orthogonal(1.0e-7));
                }

                #[test]
                fn qr_solve(n in PROPTEST_MATRIX_DIM, nb in PROPTEST_MATRIX_DIM) {
                    let m  = DMatrix::<$scalar_type>::new_random(n, n).map(|e| e.0);

                    let qr = m.clone().qr();
                    let b1 = DVector::<$scalar_type>::new_random(n).map(|e| e.0);
                    let b2 = DMatrix::<$scalar_type>::new_random(n, nb).map(|e| e.0);

                    if qr.is_invertible() {
                        let sol1 = qr.solve(&b1).unwrap();
                        let sol2 = qr.solve(&b2).unwrap();

                        prop_assert!(relative_eq!(&m * sol1, b1, epsilon = 1.0e-6));
                        prop_assert!(relative_eq!(&m * sol2, b2, epsilon = 1.0e-6));
                    }
                }

                #[test]
                fn qr_solve_static(m in matrix4_($scalar)) {
                     let qr = m.qr();
                     let b1 = Vector4::<$scalar_type>::new_random().map(|e| e.0);
                     let b2 = Matrix4x3::<$scalar_type>::new_random().map(|e| e.0);

                     if qr.is_invertible() {
                         let sol1 = qr.solve(&b1).unwrap();
                         let sol2 = qr.solve(&b2).unwrap();

                         prop_assert!(relative_eq!(m * sol1, b1, epsilon = 1.0e-6));
                         prop_assert!(relative_eq!(m * sol2, b2, epsilon = 1.0e-6));
                     }
                }

                #[test]
                fn qr_solve_overdetermined(d1 in PROPTEST_MATRIX_DIM, d2 in PROPTEST_MATRIX_DIM, nb in PROPTEST_MATRIX_DIM) {
                    // Build an overdetermined (or square) system: nrows >= ncols.
                    let nrows = cmp::min(cmp::max(d1, d2), 10);
                    let ncols = cmp::min(cmp::min(d1, d2), nrows);
                    let nb    = cmp::min(nb, 10);

                    let m  = DMatrix::<$scalar_type>::new_random(nrows, ncols).map(|e| e.0);
                    let qr = m.clone().qr();
                    let b  = DMatrix::<$scalar_type>::new_random(nrows, nb).map(|e| e.0);

                    // QR gives the least-squares solution, so we cannot assert `m * x == b`.
                    // Instead we check the normal equations `Aᴴ A x == Aᴴ b` that define it.
                    if let Some(x) = qr.solve(&b) {
                        prop_assert!(relative_eq!(
                            m.adjoint() * &m * &x, m.adjoint() * &b,
                            epsilon = 1.0e-5, max_relative = 1.0e-5
                        ));
                    }
                }

                #[test]
                fn qr_solve_overdetermined_static(m in matrix5x3_($scalar)) {
                    let qr = m.qr();
                    let b1 = Vector5::<$scalar_type>::new_random().map(|e| e.0);
                    let b2 = Matrix5x2::<$scalar_type>::new_random().map(|e| e.0);

                    // Overdetermined 5x3: verify the least-squares normal equations.
                    if let Some(sol1) = qr.solve(&b1) {
                        prop_assert!(relative_eq!(
                            m.adjoint() * m * sol1, m.adjoint() * b1,
                            epsilon = 1.0e-5, max_relative = 1.0e-5
                        ));
                    }
                    if let Some(sol2) = qr.solve(&b2) {
                        prop_assert!(relative_eq!(
                            m.adjoint() * m * sol2, m.adjoint() * b2,
                            epsilon = 1.0e-5, max_relative = 1.0e-5
                        ));
                    }
                }

                #[test]
                fn qr_inverse(n in PROPTEST_MATRIX_DIM) {
                    let n = cmp::max(1, cmp::min(n, 15)); // To avoid slowing down the test too much.
                    let m = DMatrix::<$scalar_type>::new_random(n, n).map(|e| e.0);

                    if let Some(m1) = m.clone().qr().try_inverse() {
                        let id1 = &m  * &m1;
                        let id2 = &m1 * &m;

                        prop_assert!(id1.is_identity(1.0e-5));
                        prop_assert!(id2.is_identity(1.0e-5));
                    }
                }

                #[test]
                fn qr_inverse_static(m in matrix4_($scalar)) {
                    let qr = m.qr();

                    if let Some(m1) = qr.try_inverse() {
                        let id1 = &m  * &m1;
                        let id2 = &m1 * &m;

                        prop_assert!(id1.is_identity(1.0e-5));
                        prop_assert!(id2.is_identity(1.0e-5));
                    }
                }
            }
        }
    }
);

gen_tests!(complex, complex_f64(), RandComplex<f64>);
gen_tests!(f64, PROPTEST_F64, RandScalar<f64>);
