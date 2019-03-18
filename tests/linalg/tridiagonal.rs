#![cfg(feature = "arbitrary")]

use std::cmp;

use na::{DMatrix, Matrix2, Matrix4};
use core::helper::{RandScalar, RandComplex};

quickcheck! {
    fn symm_tridiagonal(n: usize) -> bool {
        let n = cmp::max(1, cmp::min(n, 50));
        let m = DMatrix::<RandComplex<f64>>::new_random(n, n).map(|e| e.0).hermitian_part();
        let tri = m.clone().symmetric_tridiagonalize();
        let recomp = tri.recompose();

        relative_eq!(m.lower_triangle(), recomp.lower_triangle(), epsilon = 1.0e-7)
    }

    fn symm_tridiagonal_singular(n: usize) -> bool {
        let n = cmp::max(1, cmp::min(n, 4));
        let mut m = DMatrix::<RandComplex<f64>>::new_random(n, n).map(|e| e.0).hermitian_part();
        m.row_mut(n / 2).fill(na::zero());
        m.column_mut(n / 2).fill(na::zero());
        let tri = m.clone().symmetric_tridiagonalize();
        println!("Tri: {:?}", tri);
        let recomp = tri.recompose();
        println!("Recomp: {:?}", recomp);


        relative_eq!(m.lower_triangle(), recomp.lower_triangle(), epsilon = 1.0e-7)
    }

    fn symm_tridiagonal_static_square(m: Matrix4<RandComplex<f64>>) -> bool {
        let m = m.map(|e| e.0).hermitian_part();
        let tri = m.symmetric_tridiagonalize();
        let recomp = tri.recompose();

        relative_eq!(m.lower_triangle(), recomp.lower_triangle(), epsilon = 1.0e-7)
    }

    fn symm_tridiagonal_static_square_2x2(m: Matrix2<RandComplex<f64>>) -> bool {
        let m = m.map(|e| e.0).hermitian_part();
        let tri = m.symmetric_tridiagonalize();
        let recomp = tri.recompose();

        relative_eq!(m.lower_triangle(), recomp.lower_triangle(), epsilon = 1.0e-7)
    }
}
