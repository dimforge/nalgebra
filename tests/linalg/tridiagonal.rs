use std::cmp;

use na::{DMatrix, Matrix2, Matrix4, SymmetricTridiagonal};


#[cfg(feature = "arbitrary")]
quickcheck! {
    fn symm_tridiagonal(n: usize) -> bool {
        let n = cmp::max(1, cmp::min(n, 50));
        let m = DMatrix::<f64>::new_random(n, n);
        let tri = SymmetricTridiagonal::new(m.clone());
        let recomp = tri.recompose();

        println!("{}{}", m.lower_triangle(), recomp.lower_triangle());

        relative_eq!(m.lower_triangle(), recomp.lower_triangle(), epsilon = 1.0e-7)
    }

    fn symm_tridiagonal_static_square(m: Matrix4<f64>) -> bool {
        let tri = SymmetricTridiagonal::new(m);
        println!("{}{}", tri.internal_tri(), tri.off_diagonal());
        let recomp = tri.recompose();

        println!("{}{}", m.lower_triangle(), recomp.lower_triangle());

        relative_eq!(m.lower_triangle(), recomp.lower_triangle(), epsilon = 1.0e-7)
    }

    fn symm_tridiagonal_static_square_2x2(m: Matrix2<f64>) -> bool {
        let tri = SymmetricTridiagonal::new(m);
        let recomp = tri.recompose();

        relative_eq!(m.lower_triangle(), recomp.lower_triangle(), epsilon = 1.0e-7)
    }
}
