use std::cmp;
use nl::RealSchur;
use na::{DMatrix, Matrix4};

quickcheck! {
    fn schur(n: usize) -> bool {
        let n = cmp::max(1, cmp::min(n, 10));
        let m = DMatrix::<f64>::new_random(n, n);

        let (vecs, vals) = RealSchur::new(m.clone()).unpack();

        relative_eq!(&vecs * vals * vecs.transpose(), m, epsilon = 1.0e-7)
    }

    fn schur_static(m: Matrix4<f64>) -> bool {
        let (vecs, vals) = RealSchur::new(m.clone()).unpack();

        relative_eq!(vecs * vals * vecs.transpose(), m, epsilon = 1.0e-7)
    }
}
