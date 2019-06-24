#[cfg_attr(rustfmt, rustfmt_skip)]

use na::Matrix4;

#[test]
fn qrp() {
    let m = Matrix4::new (
            1.0, -1.0, 2.0, 1.0,
            -1.0, 3.0, -1.0, -1.0,
            3.0, -5.0, 5.0, 3.0,
            1.0, 2.0, 1.0, -2.0);
    let qrp = m.qrp();
    assert!(relative_eq!(qrp.determinant(), 0.0, epsilon = 1.0e-7));

    let (q, r, p) = qrp.unpack();

    let mut qr = q * r;
    p.inv_permute_columns(& mut qr);

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
                    fn qrp(m: DMatrix<$scalar>) -> bool {
                        let m = m.map(|e| e.0);
                        let qrp = m.clone().qrp();
                        let q  = qrp.q();
                        let r  = qrp.r();
    
                        println!("m: {}", m);
                        println!("qrp: {}", &q * &r);
    
                        relative_eq!(m, &q * r, epsilon = 1.0e-7) &&
                        q.is_orthogonal(1.0e-7)
                    }
    
                    fn qrp_static_5_3(m: Matrix5x3<$scalar>) -> bool {
                        let m = m.map(|e| e.0);
                        let qrp = m.qrp();
                        let q  = qrp.q();
                        let r  = qrp.r();
    
                        relative_eq!(m, q * r, epsilon = 1.0e-7) &&
                        q.is_orthogonal(1.0e-7)
                    }
    
                    fn qrp_static_3_5(m: Matrix3x5<$scalar>) -> bool {
                        let m = m.map(|e| e.0);
                        let qrp = m.qrp();
                        let q  = qrp.q();
                        let r  = qrp.r();

                    relative_eq!(m, q * r, epsilon = 1.0e-7) &&
                    q.is_orthogonal(1.0e-7)
                }

                    fn qrp_static_square(m: Matrix4<$scalar>) -> bool {
                        let m = m.map(|e| e.0);
                        let qrp = m.qrp();
                        let q  = qrp.q();
                        let r  = qrp.r();
    
                        println!("{}{}{}{}", q, r, q * r, m);
    
                        relative_eq!(m, q * r, epsilon = 1.0e-7) &&
                        q.is_orthogonal(1.0e-7)
                    }
    
                    fn qrp_solve(n: usize, nb: usize) -> bool {
                        if n != 0 && nb != 0 {
                            let n  = cmp::min(n, 50);  // To avoid slowing down the test too much.
                            let nb = cmp::min(nb, 50); // To avoid slowing down the test too much.
                            let m  = DMatrix::<$scalar>::new_random(n, n).map(|e| e.0);
    
                            let qrp = m.clone().qrp();
                            let b1 = DVector::<$scalar>::new_random(n).map(|e| e.0);
                            let b2 = DMatrix::<$scalar>::new_random(n, nb).map(|e| e.0);
    
                            if qrp.is_invertible() {
                                let sol1 = qrp.solve(&b1).unwrap();
                                let sol2 = qrp.solve(&b2).unwrap();
    
                                return relative_eq!(&m * sol1, b1, epsilon = 1.0e-6) &&
                                    relative_eq!(&m * sol2, b2, epsilon = 1.0e-6)
                            }
                        }
    
                        return true;
                    }
    
                    fn qrp_solve_static(m: Matrix4<$scalar>) -> bool {
                         let m = m.map(|e| e.0);
                         let qrp = m.qrp();
                         let b1 = Vector4::<$scalar>::new_random().map(|e| e.0);
                         let b2 = Matrix4x3::<$scalar>::new_random().map(|e| e.0);
    
                         if qrp.is_invertible() {
                             let sol1 = qrp.solve(&b1).unwrap();
                             let sol2 = qrp.solve(&b2).unwrap();
    
                             relative_eq!(m * sol1, b1, epsilon = 1.0e-6) &&
                             relative_eq!(m * sol2, b2, epsilon = 1.0e-6)
                         }
                         else {
                             false
                         }
                    }

                    fn qrp_inverse(n: usize) -> bool {
                        let n = cmp::max(1, cmp::min(n, 15)); // To avoid slowing down the test too much.
                        let m = DMatrix::<$scalar>::new_random(n, n).map(|e| e.0);
    
                        if let Some(m1) = m.clone().qrp().try_inverse() {
                            let id1 = &m  * &m1;
                            let id2 = &m1 * &m;
    
                            id1.is_identity(1.0e-5) && id2.is_identity(1.0e-5)
                        }
                        else {
                            true
                        }
                    }
    
                    fn qrp_inverse_static(m: Matrix4<$scalar>) -> bool {
                        let m  = m.map(|e| e.0);
                        let qrp = m.qrp();
    
                        if let Some(m1) = qrp.try_inverse() {
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

