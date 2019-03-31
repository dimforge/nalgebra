#![cfg(feature = "arbitrary")]


macro_rules! gen_tests(
    ($module: ident, $scalar: ty) => {
        mod $module {
            use na::{Matrix4, Matrix4x5, ComplexField};
            #[allow(unused_imports)]
            use crate::core::helper::{RandScalar, RandComplex};

            fn unzero_diagonal<N: ComplexField>(a: &mut Matrix4<N>) {
                for i in 0..4 {
                    if a[(i, i)].norm1() < na::convert(1.0e-7) {
                        a[(i, i)] = N::one();
                    }
                }
            }

            quickcheck! {
                fn solve_lower_triangular(a: Matrix4<$scalar>, b: Matrix4x5<$scalar>) -> bool {
                    let b = b.map(|e| e.0);
                    let mut a = a.map(|e| e.0);
                    unzero_diagonal(&mut a);
                    let tri = a.lower_triangle();
                    let x   = a.solve_lower_triangular(&b).unwrap();

                    relative_eq!(tri * x, b, epsilon = 1.0e-7)
                }

                fn solve_upper_triangular(a: Matrix4<$scalar>, b: Matrix4x5<$scalar>) -> bool {
                    let b = b.map(|e| e.0);
                    let mut a = a.map(|e| e.0);
                    unzero_diagonal(&mut a);
                    let tri = a.upper_triangle();
                    let x   = a.solve_upper_triangular(&b).unwrap();

                    relative_eq!(tri * x, b, epsilon = 1.0e-7)
                }

                fn tr_solve_lower_triangular(a: Matrix4<$scalar>, b: Matrix4x5<$scalar>) -> bool {
                    let b = b.map(|e| e.0);
                    let mut a = a.map(|e| e.0);
                    unzero_diagonal(&mut a);
                    let tri = a.lower_triangle();
                    let x   = a.tr_solve_lower_triangular(&b).unwrap();

                    relative_eq!(tri.transpose() * x, b, epsilon = 1.0e-7)
                }

                fn tr_solve_upper_triangular(a: Matrix4<$scalar>, b: Matrix4x5<$scalar>) -> bool {
                    let b = b.map(|e| e.0);
                    let mut a = a.map(|e| e.0);
                    unzero_diagonal(&mut a);
                    let tri = a.upper_triangle();
                    let x   = a.tr_solve_upper_triangular(&b).unwrap();

                    relative_eq!(tri.transpose() * x, b, epsilon = 1.0e-7)
                }
            }
        }
    }
);

gen_tests!(complex, RandComplex<f64>);
gen_tests!(f64, RandScalar<f64>);
