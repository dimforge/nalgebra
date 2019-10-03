#![cfg_attr(rustfmt, rustfmt_skip)]

#[cfg(feature = "arbitrary")]
mod quickcheck_tests {
    macro_rules! gen_tests(
        ($module: ident, $scalar: ty) => {
            mod $module {
                use na::{
                    DMatrix, DVector, ComplexField
                };
                use std::cmp;
                #[allow(unused_imports)]
                use crate::core::helper::{RandScalar, RandComplex};

                quickcheck! {
                    fn polar_recompose(m: DMatrix<$scalar>) -> bool {
                        let m = m.map(|e| e.0);
                        if m.len() > 0 {
                            let polar = m.clone().polar();
                            let recomp_m_l = polar.clone().recompose_left().unwrap();
                            let recomp_m_r = polar.clone().recompose_right().unwrap();

                            relative_eq!(m, recomp_m_l, epsilon = 1.0e-5) &&
                            relative_eq!(m, recomp_m_r, epsilon = 1.0e-5) 
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
