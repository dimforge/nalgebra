#![cfg(feature = "proptest-support")]

use na::Matrix2;

#[test]
fn hessenberg_simple() {
    let m = Matrix2::new(1.0, 0.0, 1.0, 3.0);
    let hess = m.hessenberg();
    let (p, h) = hess.unpack();
    assert!(relative_eq!(m, p * h * p.transpose(), epsilon = 1.0e-7))
}

macro_rules! gen_tests(
    ($module: ident, $scalar: expr) => {
         mod $module {
            #[allow(unused_imports)]
            use crate::core::helper::{RandScalar, RandComplex};

            use crate::proptest::*;
            use proptest::{prop_assert, proptest};

            proptest! {
                #[test]
                fn hessenberg(m in dmatrix_($scalar)) {
                    let hess = m.clone().hessenberg();
                    let (p, h) = hess.unpack();
                    prop_assert!(relative_eq!(m, &p * h * p.adjoint(), epsilon = 1.0e-7))
                }

                #[test]
                fn hessenberg_static_mat2(m in matrix2_($scalar)) {
                    let hess = m.hessenberg();
                    let (p, h) = hess.unpack();
                    prop_assert!(relative_eq!(m, p * h * p.adjoint(), epsilon = 1.0e-7))
                }

                #[test]
                fn hessenberg_static(m in matrix4_($scalar)) {
                    let hess = m.hessenberg();
                    let (p, h) = hess.unpack();
                    prop_assert!(relative_eq!(m, p * h * p.adjoint(), epsilon = 1.0e-7))
                }
            }
         }
    }
);

gen_tests!(complex, complex_f64());
gen_tests!(f64, PROPTEST_F64);
