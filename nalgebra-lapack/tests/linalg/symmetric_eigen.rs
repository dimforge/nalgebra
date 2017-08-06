use std::cmp;

use nl::SymmetricEigen;
use na::{DMatrix, Matrix4};

quickcheck!{
    fn symmetric_eigen(n: usize) -> bool {
        let n = cmp::max(1, cmp::min(n, 10));
        let m = DMatrix::<f64>::new_random(n, n);
        let eig = SymmetricEigen::new(m.clone());
        let recomp = eig.recompose();
        relative_eq!(m.lower_triangle(), recomp.lower_triangle(), epsilon = 1.0e-5)
    }

    fn symmetric_eigen_static(m: Matrix4<f64>) -> bool {
        let eig = SymmetricEigen::new(m);
        let recomp = eig.recompose();
        relative_eq!(m.lower_triangle(), recomp.lower_triangle(), epsilon = 1.0e-5)
    }
}
