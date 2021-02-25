#[cfg_attr(rustfmt, rustfmt_skip)]

use na::Matrix4;

#[test]
fn col_piv_qr() {
    let m = Matrix4::new(
        1.0, -1.0, 2.0, 1.0, -1.0, 3.0, -1.0, -1.0, 3.0, -5.0, 5.0, 3.0, 1.0, 2.0, 1.0, -2.0,
    );
    let col_piv_qr = m.col_piv_qr();
    assert!(relative_eq!(
        col_piv_qr.determinant(),
        0.0,
        epsilon = 1.0e-7
    ));

    let (q, r, p) = col_piv_qr.unpack();

    let mut qr = q * r;
    p.inv_permute_columns(&mut qr);

    assert!(relative_eq!(m, qr, epsilon = 1.0e-7));
}

#[cfg(feature = "arbitrary")]
mod quickcheck_tests {
    macro_rules! gen_tests(
        ($module: ident, $scalar: ty) => {
            mod $module {
                use na::{DMatrix, DVector, Matrix3x5, Matrix4, Matrix4x3, Matrix5x3, Vector4};
                use std::cmp;
                #[allow(unused_imports)]
                use crate::core::helper::{RandScalar, RandComplex};

                quickcheck! {
                    fn col_piv_qr(m: DMatrix<$scalar>) -> bool {
                        let m = m.map(|e| e.0);
                        let col_piv_qr = m.clone().col_piv_qr();
                        let (q, r, p) = col_piv_qr.unpack();
                        let mut qr = &q * &r;
                        p.inv_permute_columns(&mut qr);

                        println!("m: {}", m);
                        println!("col_piv_qr: {}", &q * &r);

                        relative_eq!(m, &qr, epsilon = 1.0e-7) &&
                        q.is_orthogonal(1.0e-7)
                    }

                    fn col_piv_qr_static_5_3(m: Matrix5x3<$scalar>) -> bool {
                        let m = m.map(|e| e.0);
                        let col_piv_qr = m.col_piv_qr();
                        let (q, r, p) = col_piv_qr.unpack();
                        let mut qr = q * r;
                        p.inv_permute_columns(&mut qr);

                        relative_eq!(m, qr, epsilon = 1.0e-7) &&
                        q.is_orthogonal(1.0e-7)
                    }

                    fn col_piv_qr_static_3_5(m: Matrix3x5<$scalar>) -> bool {
                        let m = m.map(|e| e.0);
                        let col_piv_qr = m.col_piv_qr();
                        let (q, r, p) = col_piv_qr.unpack();
                        let mut qr = q * r;
                        p.inv_permute_columns(&mut qr);

                        relative_eq!(m, qr, epsilon = 1.0e-7) &&
                        q.is_orthogonal(1.0e-7)
                    }

                    fn col_piv_qr_static_square(m: Matrix4<$scalar>) -> bool {
                        let m = m.map(|e| e.0);
                        let col_piv_qr = m.col_piv_qr();
                        let (q, r, p) = col_piv_qr.unpack();
                        let mut qr = q * r;
                        p.inv_permute_columns(&mut qr);

                        println!("{}{}{}{}", q, r, qr, m);

                        relative_eq!(m, qr, epsilon = 1.0e-7) &&
                        q.is_orthogonal(1.0e-7)
                    }

                    fn col_piv_qr_solve(n: usize, nb: usize) -> bool {
                        if n != 0 && nb != 0 {
                            let n  = cmp::min(n, 50);  // To avoid slowing down the test too much.
                            let nb = cmp::min(nb, 50); // To avoid slowing down the test too much.
                            let m  = DMatrix::<$scalar>::new_random(n, n).map(|e| e.0);

                            let col_piv_qr = m.clone().col_piv_qr();
                            let b1 = DVector::<$scalar>::new_random(n).map(|e| e.0);
                            let b2 = DMatrix::<$scalar>::new_random(n, nb).map(|e| e.0);

                            if col_piv_qr.is_invertible() {
                                let sol1 = col_piv_qr.solve(&b1).unwrap();
                                let sol2 = col_piv_qr.solve(&b2).unwrap();

                                return relative_eq!(&m * sol1, b1, epsilon = 1.0e-6) &&
                                    relative_eq!(&m * sol2, b2, epsilon = 1.0e-6)
                            }
                        }

                        return true;
                    }

                    fn col_piv_qr_solve_static(m: Matrix4<$scalar>) -> bool {
                         let m = m.map(|e| e.0);
                         let col_piv_qr = m.col_piv_qr();
                         let b1 = Vector4::<$scalar>::new_random().map(|e| e.0);
                         let b2 = Matrix4x3::<$scalar>::new_random().map(|e| e.0);

                         if col_piv_qr.is_invertible() {
                             let sol1 = col_piv_qr.solve(&b1).unwrap();
                             let sol2 = col_piv_qr.solve(&b2).unwrap();

                             relative_eq!(m * sol1, b1, epsilon = 1.0e-6) &&
                             relative_eq!(m * sol2, b2, epsilon = 1.0e-6)
                         }
                         else {
                             false
                         }
                    }

                    fn col_piv_qr_inverse(n: usize) -> bool {
                        let n = cmp::max(1, cmp::min(n, 15)); // To avoid slowing down the test too much.
                        let m = DMatrix::<$scalar>::new_random(n, n).map(|e| e.0);

                        if let Some(m1) = m.clone().col_piv_qr().try_inverse() {
                            let id1 = &m  * &m1;
                            let id2 = &m1 * &m;

                            id1.is_identity(1.0e-5) && id2.is_identity(1.0e-5)
                        }
                        else {
                            true
                        }
                    }

                    fn col_piv_qr_inverse_static(m: Matrix4<$scalar>) -> bool {
                        let m  = m.map(|e| e.0);
                        let col_piv_qr = m.col_piv_qr();

                        if let Some(m1) = col_piv_qr.try_inverse() {
                            let id1 = &m  * &m1;
                            let id2 = &m1 * &m;

                            id1.is_identity(1.0e-5) && id2.is_identity(1.0e-5)
                        }
                        else {
                            true
                        }
                    }
                }
            }
        }
    );

    gen_tests!(complex, RandComplex<f64>);
    gen_tests!(f64, RandScalar<f64>);
}
