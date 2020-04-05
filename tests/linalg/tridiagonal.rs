#![cfg(feature = "arbitrary")]


macro_rules! gen_tests(
    ($module: ident, $scalar: ty) => {
            mod $module {
            use std::cmp;

            use na::{DMatrix, Matrix2, Matrix4};
            #[allow(unused_imports)]
            use crate::core::helper::{RandScalar, RandComplex};

            quickcheck! {
                fn symm_tridiagonal(n: usize) -> bool {
                    let n = cmp::max(1, cmp::min(n, 50));
                    let m = DMatrix::<$scalar>::new_random(n, n).map(|e| e.0).hermitian_part();
                    let tri = m.clone().symmetric_tridiagonalize();
                    let recomp = tri.recompose();

                    relative_eq!(m.lower_triangle(), recomp.lower_triangle(), epsilon = 1.0e-7)
                }

                fn symm_tridiagonal_singular(n: usize) -> bool {
                    let n = cmp::max(1, cmp::min(n, 4));
                    let mut m = DMatrix::<$scalar>::new_random(n, n).map(|e| e.0).hermitian_part();
                    m.row_mut(n / 2).fill(na::zero());
                    m.column_mut(n / 2).fill(na::zero());
                    let tri = m.clone().symmetric_tridiagonalize();
                    let recomp = tri.recompose();

                    relative_eq!(m.lower_triangle(), recomp.lower_triangle(), epsilon = 1.0e-7)
                }

                fn symm_tridiagonal_static_square(m: Matrix4<$scalar>) -> bool {
                    let m = m.map(|e| e.0).hermitian_part();
                    let tri = m.symmetric_tridiagonalize();
                    let recomp = tri.recompose();

                    relative_eq!(m.lower_triangle(), recomp.lower_triangle(), epsilon = 1.0e-7)
                }

                fn symm_tridiagonal_static_square_2x2(m: Matrix2<$scalar>) -> bool {
                    let m = m.map(|e| e.0).hermitian_part();
                    let tri = m.symmetric_tridiagonalize();
                    let recomp = tri.recompose();

                    relative_eq!(m.lower_triangle(), recomp.lower_triangle(), epsilon = 1.0e-7)
                }
            }
        }
    }
);

gen_tests!(complex, RandComplex<f64>);
gen_tests!(f64, RandScalar<f64>);
