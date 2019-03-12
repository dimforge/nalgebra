#![cfg(feature = "arbitrary")]

use na::{DMatrix, Matrix2, Matrix4};
use core::helper::{RandScalar, RandComplex};
use std::cmp;


#[test]
fn hessenberg_simple() {
    let m = Matrix2::new(1.0, 0.0, 1.0, 3.0);
    let hess = m.hessenberg();
    let (p, h) = hess.unpack();
    assert!(relative_eq!(m, p * h * p.transpose(), epsilon = 1.0e-7))
}

quickcheck! {
    fn hessenberg(n: usize) -> bool {
        let n = cmp::max(1, cmp::min(n, 50));
        let m = DMatrix::<RandComplex<f64>>::new_random(n, n).map(|e| e.0);

        let hess = m.clone().hessenberg();
        let (p, h) = hess.unpack();
        relative_eq!(m, &p * h * p.conjugate_transpose(), epsilon = 1.0e-7)
    }

    fn hessenberg_static_mat2(m: Matrix2<RandComplex<f64>>) -> bool {
        let m = m.map(|e| e.0);
        let hess = m.hessenberg();
        let (p, h) = hess.unpack();
        relative_eq!(m, p * h * p.conjugate_transpose(), epsilon = 1.0e-7)
    }

    fn hessenberg_static(m: Matrix4<RandComplex<f64>>) -> bool {
        let m = m.map(|e| e.0);
        let hess = m.hessenberg();
        let (p, h) = hess.unpack();
        relative_eq!(m, p * h * p.conjugate_transpose(), epsilon = 1.0e-7)
    }
}
