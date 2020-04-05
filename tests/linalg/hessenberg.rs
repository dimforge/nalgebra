#![cfg(feature = "arbitrary")]

use na::Matrix2;

#[test]
fn hessenberg_simple() {
    let m = Matrix2::new(1.0, 0.0, 1.0, 3.0);
    let hess = m.hessenberg();
    let (p, h) = hess.unpack();
    assert!(relative_eq!(m, p * h * p.transpose(), epsilon = 1.0e-7))
}

macro_rules! gen_tests(
    ($module: ident, $scalar: ty) => {
         mod $module {
            use na::{DMatrix, Matrix2, Matrix4};
            use std::cmp;
            #[allow(unused_imports)]
            use crate::core::helper::{RandScalar, RandComplex};

            quickcheck! {
                fn hessenberg(n: usize) -> bool {
                    let n = cmp::max(1, cmp::min(n, 50));
                    let m = DMatrix::<$scalar>::new_random(n, n).map(|e| e.0);

                    let hess = m.clone().hessenberg();
                    let (p, h) = hess.unpack();
                    relative_eq!(m, &p * h * p.adjoint(), epsilon = 1.0e-7)
                }

                fn hessenberg_static_mat2(m: Matrix2<$scalar>) -> bool {
                    let m = m.map(|e| e.0);
                    let hess = m.hessenberg();
                    let (p, h) = hess.unpack();
                    relative_eq!(m, p * h * p.adjoint(), epsilon = 1.0e-7)
                }

                fn hessenberg_static(m: Matrix4<$scalar>) -> bool {
                    let m = m.map(|e| e.0);
                    let hess = m.hessenberg();
                    let (p, h) = hess.unpack();
                    relative_eq!(m, p * h * p.adjoint(), epsilon = 1.0e-7)
                }
            }
         }
    }
);

gen_tests!(complex, RandComplex<f64>);
gen_tests!(f64, RandScalar<f64>);
