use na::Matrix3;

#[test]
#[rustfmt::skip]
fn lu_simple() {
    let m = Matrix3::new(
        2.0, -1.0,  0.0,
       -1.0,  2.0, -1.0,
        0.0, -1.0,  2.0);

    let lu = m.lu();
    assert_eq!(lu.determinant(), 4.0);

    let (p, l, u) = lu.unpack();

    let mut lu = l * u;
    p.inv_permute_rows(&mut lu);

    assert!(relative_eq!(m, lu, epsilon = 1.0e-7));
}

#[test]
#[rustfmt::skip]
fn lu_simple_with_pivot() {
    let m = Matrix3::new(
        0.0, -1.0,  2.0,
       -1.0,  2.0, -1.0,
        2.0, -1.0,  0.0);

    let lu = m.lu();
    assert_eq!(lu.determinant(), -4.0);

    let (p, l, u) = lu.unpack();

    let mut lu = l * u;
    p.inv_permute_rows(&mut lu);

    assert!(relative_eq!(m, lu, epsilon = 1.0e-7));
}

#[cfg(feature = "arbitrary")]
mod quickcheck_tests {
    #[allow(unused_imports)]
    use crate::core::helper::{RandComplex, RandScalar};

    macro_rules! gen_tests(
        ($module: ident, $scalar: ty) => {
            mod $module {
                use std::cmp;
                use na::{DMatrix, Matrix4, Matrix4x3, Matrix5x3, Matrix3x5, DVector, Vector4};
                #[allow(unused_imports)]
                use crate::core::helper::{RandScalar, RandComplex};

                quickcheck! {
                    fn lu(m: DMatrix<$scalar>) -> bool {
                        let mut m = m;
                        if m.len() == 0 {
                             m = DMatrix::<$scalar>::new_random(1, 1);
                        }

                        let m = m.map(|e| e.0);

                        let lu = m.clone().lu();
                        let (p, l, u) = lu.unpack();
                        let mut lu = l * u;
                        p.inv_permute_rows(&mut lu);

                        relative_eq!(m, lu, epsilon = 1.0e-7)
                    }

                    fn lu_static_3_5(m: Matrix3x5<$scalar>) -> bool {
                        let m = m.map(|e| e.0);
                        let lu = m.lu();
                        let (p, l, u) = lu.unpack();
                        let mut lu = l * u;
                        p.inv_permute_rows(&mut lu);

                        relative_eq!(m, lu, epsilon = 1.0e-7)
                    }

                    fn lu_static_5_3(m: Matrix5x3<$scalar>) -> bool {
                        let m = m.map(|e| e.0);
                        let lu = m.lu();
                        let (p, l, u) = lu.unpack();
                        let mut lu = l * u;
                        p.inv_permute_rows(&mut lu);

                        relative_eq!(m, lu, epsilon = 1.0e-7)
                    }

                    fn lu_static_square(m: Matrix4<$scalar>) -> bool {
                        let m = m.map(|e| e.0);
                        let lu = m.lu();
                        let (p, l, u) = lu.unpack();
                        let mut lu = l * u;
                        p.inv_permute_rows(&mut lu);

                        relative_eq!(m, lu, epsilon = 1.0e-7)
                    }

                    fn lu_solve(n: usize, nb: usize) -> bool {
                        if n != 0 && nb != 0 {
                            let n  = cmp::min(n, 50);  // To avoid slowing down the test too much.
                            let nb = cmp::min(nb, 50); // To avoid slowing down the test too much.
                            let m  = DMatrix::<$scalar>::new_random(n, n).map(|e| e.0);

                            let lu = m.clone().lu();
                            let b1 = DVector::<$scalar>::new_random(n).map(|e| e.0);
                            let b2 = DMatrix::<$scalar>::new_random(n, nb).map(|e| e.0);

                            let sol1 = lu.solve(&b1);
                            let sol2 = lu.solve(&b2);

                            return (sol1.is_none() || relative_eq!(&m * sol1.unwrap(), b1, epsilon = 1.0e-6)) &&
                                   (sol2.is_none() || relative_eq!(&m * sol2.unwrap(), b2, epsilon = 1.0e-6))
                        }

                        return true;
                    }

                    fn lu_solve_static(m: Matrix4<$scalar>) -> bool {
                         let m = m.map(|e| e.0);
                         let lu = m.lu();
                         let b1 = Vector4::<$scalar>::new_random().map(|e| e.0);
                         let b2 = Matrix4x3::<$scalar>::new_random().map(|e| e.0);

                         let sol1 = lu.solve(&b1);
                         let sol2 = lu.solve(&b2);

                         return (sol1.is_none() || relative_eq!(&m * sol1.unwrap(), b1, epsilon = 1.0e-6)) &&
                                (sol2.is_none() || relative_eq!(&m * sol2.unwrap(), b2, epsilon = 1.0e-6))
                    }

                    fn lu_inverse(n: usize) -> bool {
                        let n = cmp::max(1, cmp::min(n, 15)); // To avoid slowing down the test too much.
                        let m = DMatrix::<$scalar>::new_random(n, n).map(|e| e.0);

                        let mut l = m.lower_triangle();
                        let mut u = m.upper_triangle();

                        // Ensure the matrix is well conditioned for inversion.
                        l.fill_diagonal(na::one());
                        u.fill_diagonal(na::one());
                        let m = l * u;

                        let m1  = m.clone().lu().try_inverse().unwrap();
                        let id1 = &m  * &m1;
                        let id2 = &m1 * &m;

                        return id1.is_identity(1.0e-5) && id2.is_identity(1.0e-5);
                    }

                    fn lu_inverse_static(m: Matrix4<$scalar>) -> bool {
                        let m = m.map(|e| e.0);
                        let lu  = m.lu();

                        if let Some(m1) = lu.try_inverse() {
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
