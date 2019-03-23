#![cfg(all(feature = "arbitrary", feature = "debug"))]


macro_rules! gen_tests(
    ($module: ident, $scalar: ty) => {
        mod $module {
            use na::debug::RandomSDP;
            use na::dimension::{U4, Dynamic};
            use na::{DMatrix, DVector, Matrix4x3, Vector4};
            use rand::random;
            #[allow(unused_imports)]
            use crate::core::helper::{RandScalar, RandComplex};
            use std::cmp;

            quickcheck! {
                fn cholesky(n: usize) -> bool {
                    let m = RandomSDP::new(Dynamic::new(n.max(1).min(50)), || random::<$scalar>().0).unwrap();
                    let l = m.clone().cholesky().unwrap().unpack();
                    relative_eq!(m, &l * l.adjoint(), epsilon = 1.0e-7)
                }

                fn cholesky_static(_m: RandomSDP<f64, U4>) -> bool {
                    let m = RandomSDP::new(U4, || random::<$scalar>().0).unwrap();
                    let chol = m.cholesky().unwrap();
                    let l    = chol.unpack();

                    if !relative_eq!(m, &l * l.adjoint(), epsilon = 1.0e-7) {
                        false
                    }
                    else {
                        true
                    }
                }

                fn cholesky_solve(n: usize, nb: usize) -> bool {
                    let n = n.max(1).min(50);
                    let m = RandomSDP::new(Dynamic::new(n), || random::<$scalar>().0).unwrap();
                    let nb = cmp::min(nb, 50); // To avoid slowing down the test too much.

                    let chol = m.clone().cholesky().unwrap();
                    let b1 = DVector::<$scalar>::new_random(n).map(|e| e.0);
                    let b2 = DMatrix::<$scalar>::new_random(n, nb).map(|e| e.0);

                    let sol1 = chol.solve(&b1);
                    let sol2 = chol.solve(&b2);

                    relative_eq!(&m * &sol1, b1, epsilon = 1.0e-7) &&
                    relative_eq!(&m * &sol2, b2, epsilon = 1.0e-7)
                }

                fn cholesky_solve_static(_n: usize) -> bool {
                    let m = RandomSDP::new(U4, || random::<$scalar>().0).unwrap();
                    let chol = m.clone().cholesky().unwrap();
                    let b1 = Vector4::<$scalar>::new_random().map(|e| e.0);
                    let b2 = Matrix4x3::<$scalar>::new_random().map(|e| e.0);

                    let sol1 = chol.solve(&b1);
                    let sol2 = chol.solve(&b2);

                    relative_eq!(m * sol1, b1, epsilon = 1.0e-7) &&
                    relative_eq!(m * sol2, b2, epsilon = 1.0e-7)
                }

                fn cholesky_inverse(n: usize) -> bool {
                    let m = RandomSDP::new(Dynamic::new(n.max(1).min(50)), || random::<$scalar>().0).unwrap();
                    let m1 = m.clone().cholesky().unwrap().inverse();
                    let id1 = &m  * &m1;
                    let id2 = &m1 * &m;

                    id1.is_identity(1.0e-7) && id2.is_identity(1.0e-7)
                }

                fn cholesky_inverse_static(_n: usize) -> bool {
                    let m = RandomSDP::new(U4, || random::<$scalar>().0).unwrap();
                    let m1 = m.clone().cholesky().unwrap().inverse();
                    let id1 = &m  * &m1;
                    let id2 = &m1 * &m;

                    id1.is_identity(1.0e-7) && id2.is_identity(1.0e-7)
                }
            }
        }
    }
);

gen_tests!(complex, RandComplex<f64>);
gen_tests!(f64, RandScalar<f64>);
