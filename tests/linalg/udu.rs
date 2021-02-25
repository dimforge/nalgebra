use na::{Matrix3, UDU};

#[test]
#[rustfmt::skip]
fn udu_simple() {
    let m = Matrix3::new(
        2.0, -1.0,  0.0,
       -1.0,  2.0, -1.0,
        0.0, -1.0,  2.0);

    let udu = UDU::new(m);
    // Rebuild
    let p = udu.u * udu.d_matrix() * udu.u.transpose();

    assert!(relative_eq!(m, p, epsilon = 3.0e-16));
}

#[test]
#[should_panic]
#[rustfmt::skip]
fn udu_non_sym_panic() {
    let m = Matrix3::new(
        2.0, -1.0,  0.0,
        1.0, -2.0,  3.0,
       -2.0,  1.0,  0.0);

    let udu = UDU::new(m);
    // Rebuild
    let p = udu.u * udu.d_matrix() * udu.u.transpose();

    assert!(relative_eq!(m, p, epsilon = 3.0e-16));
}

#[cfg(feature = "arbitrary")]
mod quickcheck_tests {
    #[allow(unused_imports)]
    use crate::core::helper::{RandComplex, RandScalar};

    macro_rules! gen_tests(
        ($module: ident, $scalar: ty) => {
            mod $module {
                use na::{UDU, DMatrix, Matrix4};
                #[allow(unused_imports)]
                use crate::core::helper::{RandScalar, RandComplex};

                quickcheck! {
                    fn udu(n: usize) -> bool {
                        let n = std::cmp::max(1, std::cmp::min(n, 10));
                        let m = DMatrix::<$scalar>::new_random(n, n).map(|e| e.0).hermitian_part();

                        let udu = UDU::new(m.clone());
                        let p = &udu.u * &udu.d_matrix() * &udu.u.transpose();

                        relative_eq!(m, p, epsilon = 1.0e-7)
                    }

                    fn udu_static(m: Matrix4<$scalar>) -> bool {
                        let m = m.map(|e| e.0).hermitian_part();

                        let udu = UDU::new(m.clone());
                        let p = udu.u * udu.d_matrix() * udu.u.transpose();

                        relative_eq!(m, p, epsilon = 1.0e-7)
                    }
                }
            }
        }
    );

    gen_tests!(f64, RandScalar<f64>);
}
