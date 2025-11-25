use na::{Complex, Matrix3};
use num::Zero;

#[test]
#[rustfmt::skip]
fn ldl_simple() {
    let m = Matrix3::new(
        Complex::new(2.0, 0.0), Complex::new(-1.0, 0.5),  Complex::zero(),
        Complex::new(-1.0, -0.5),  Complex::new(2.0, 0.0), Complex::new(-1.0, 0.0),
        Complex::zero(), Complex::new(-1.0, 0.0),  Complex::new(2.0, 0.0));

    let ldl = m.lower_triangle().ldl().unwrap();
    
    // Rebuild
    let p = ldl.l * ldl.d_matrix() * ldl.l.adjoint();

    assert!(relative_eq!(m, p, epsilon = 3.0e-12));
}

#[test]
#[rustfmt::skip]
fn ldl_partial() {
    let m = Matrix3::new(
        Complex::new(2.0, 0.0), Complex::zero(),  Complex::zero(),
        Complex::zero(),  Complex::zero(), Complex::zero(),
        Complex::zero(), Complex::zero(),  Complex::new(2.0, 0.0));

    let ldl = m.lower_triangle().ldl().unwrap();
    
    // Rebuild
    let p = ldl.l * ldl.d_matrix() * ldl.l.adjoint();

    assert!(relative_eq!(m, p, epsilon = 3.0e-12));
}

#[test]
#[rustfmt::skip]
fn ldl_lsqrtd() {
    let m = Matrix3::new(
        Complex::new(2.0, 0.0), Complex::new(-1.0, 0.5),  Complex::zero(),
        Complex::new(-1.0, -0.5),  Complex::new(2.0, 0.0), Complex::new(-1.0, 0.0),
        Complex::zero(), Complex::new(-1.0, 0.0),  Complex::new(2.0, 0.0));

    let chol= m.cholesky().unwrap();
    let ldl = m.ldl().unwrap();
    
    assert!(relative_eq!(ldl.lsqrtd().unwrap(), chol.l(), epsilon = 3.0e-16));
}

#[test]
#[should_panic]
#[rustfmt::skip]
fn ldl_non_sym_panic() {
    let m = Matrix3::new(
        2.0, -1.0,  0.0,
        1.0, -2.0,  3.0,
       -2.0,  1.0,  0.3);

    let ldl = m.ldl().unwrap();
    
    // Rebuild
    let p = ldl.l * ldl.d_matrix() * ldl.l.transpose();

    assert!(relative_eq!(m, p, epsilon = 3.0e-16));
}

#[cfg(feature = "proptest-support")]
mod proptest_tests {
    #[allow(unused_imports)]
    use crate::core::helper::{RandComplex, RandScalar};

    macro_rules! gen_tests(
        ($module: ident, $scalar: expr) => {
            mod $module {
                #[allow(unused_imports)]
                use crate::core::helper::{RandScalar, RandComplex};
                use crate::proptest::*;
                use proptest::{prop_assert, proptest};

                proptest! {
                    #[test]
                    fn ldl(m in dmatrix_($scalar)) {
                        let m = &m * m.adjoint();

                        if let Some(ldl) = m.clone().ldl() {
                            let p = &ldl.l * &ldl.d_matrix() * &ldl.l.transpose();
                            println!("m: {}, p: {}", m, p);

                            prop_assert!(relative_eq!(m, p, epsilon = 1.0e-7));
                        }
                    }

                    #[test]
                    fn ldl_static(m in matrix4_($scalar)) {
                        let m = m.hermitian_part();

                        if let Some(ldl) = m.ldl() {
                            let p = ldl.l * ldl.d_matrix() * ldl.l.transpose();
                            prop_assert!(relative_eq!(m, p, epsilon = 1.0e-7));
                        }
                    }
                }
            }
        }
    );

    gen_tests!(f64, PROPTEST_F64);
}
