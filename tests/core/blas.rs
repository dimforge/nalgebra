use std::cmp;

use na::{DVector, DMatrix};


#[cfg(feature = "arbitrary")]
quickcheck! {
    /*
     *
     * Symmetric operators.
     *
     */
    fn gemv_symm(n: usize, alpha: f64, beta: f64) -> bool {
        let n = cmp::max(1, cmp::min(n, 50));
        let a = DMatrix::<f64>::new_random(n, n);
        let a = &a * a.transpose();

        let x = DVector::new_random(n);
        let mut y1 = DVector::new_random(n);
        let mut y2 = y1.clone();

        y1.gemv(alpha, &a, &x, beta);
        y2.gemv_symm(alpha, &a.lower_triangle(), &x, beta);

        if !relative_eq!(y1, y2, epsilon = 1.0e-10) {
            return false;
        }

        y1.gemv(alpha, &a, &x, 0.0);
        y2.gemv_symm(alpha, &a.lower_triangle(), &x, 0.0);

        relative_eq!(y1, y2, epsilon = 1.0e-10)
    }

    fn ger_symm(n: usize, alpha: f64, beta: f64) -> bool {
        let n = cmp::max(1, cmp::min(n, 50));
        let a = DMatrix::<f64>::new_random(n, n);
        let mut a1 = &a * a.transpose();
        let mut a2 = a1.lower_triangle();

        let x = DVector::new_random(n);
        let y = DVector::new_random(n);

        a1.ger(alpha, &x, &y, beta);
        a2.ger_symm(alpha, &x, &y, beta);

        if !relative_eq!(a1.lower_triangle(), a2) {
            return false;
        }

        a1.ger(alpha, &x, &y, 0.0);
        a2.ger_symm(alpha, &x, &y, 0.0);

        relative_eq!(a1.lower_triangle(), a2)
    }
}
