use std::cmp;

use nl::Hessenberg;
use na::{DMatrix, Matrix4};

quickcheck!{
    fn hessenberg(n: usize) -> bool {
        if n != 0 {
            let n = cmp::min(n, 25);
            let m = DMatrix::<f64>::new_random(n, n);

            match Hessenberg::new(m.clone()) {
                Some(hess) => {
                    let h = hess.h();
                    let p = hess.p();

                    relative_eq!(m, &p * h * p.transpose(), epsilon = 1.0e-7)
                },
                None => true
            }
        }
        else {
            true
        }
    }

    fn hessenberg_static(m: Matrix4<f64>) -> bool {
        match Hessenberg::new(m) {
            Some(hess) => {
                let h = hess.h();
                let p = hess.p();

                relative_eq!(m, p * h * p.transpose(), epsilon = 1.0e-7)
            },
            None => true
        }
    }
}
