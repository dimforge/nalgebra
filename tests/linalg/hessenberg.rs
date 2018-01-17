#![cfg(feature = "arbitrary")]

use std::cmp;
use na::{DMatrix, Matrix2, Matrix4};

#[test]
fn hessenberg_simple() {
    let m = Matrix2::new(1.0, 0.0,
                         1.0, 3.0);
    let hess = m.hessenberg();
    let (p, h) = hess.unpack();
    assert!(relative_eq!(m, p * h * p.transpose(), epsilon = 1.0e-7))
}

quickcheck! {
    fn hessenberg(n: usize) -> bool {
        let n = cmp::max(1, cmp::min(n, 50));
        let m = DMatrix::<f64>::new_random(n, n);

        let hess = m.clone().hessenberg();
        let (p, h) = hess.unpack();
        relative_eq!(m, &p * h * p.transpose(), epsilon = 1.0e-7)
    }

    fn hessenberg_static_mat2(m: Matrix2<f64>) -> bool {
        let hess = m.hessenberg();
        let (p, h) = hess.unpack();
        relative_eq!(m, p * h * p.transpose(), epsilon = 1.0e-7)
    }

    fn hessenberg_static(m: Matrix4<f64>) -> bool {
        let hess = m.hessenberg();
        let (p, h) = hess.unpack();
        relative_eq!(m, p * h * p.transpose(), epsilon = 1.0e-7)
    }
}
