#![cfg(feature = "arbitrary")]

use na::{Matrix4, Matrix4x5};

fn unzero_diagonal(a: &mut Matrix4<f64>) {
    for i in 0 .. 4 {
        if a[(i, i)] < 1.0e-7 {
            a[(i, i)] = 1.0;
        }
    }
}

quickcheck! {
    fn solve_lower_triangular(a: Matrix4<f64>, b: Matrix4x5<f64>) -> bool {
        let mut a = a;
        unzero_diagonal(&mut a);
        let tri = a.lower_triangle();
        let x   = a.solve_lower_triangular(&b).unwrap();

        println!("{}\n{}\n{}\n{}", tri, x, tri * x, b);

        relative_eq!(tri * x, b, epsilon = 1.0e-7)
    }

    fn solve_upper_triangular(a: Matrix4<f64>, b: Matrix4x5<f64>) -> bool {
        let mut a = a;
        unzero_diagonal(&mut a);
        let tri = a.upper_triangle();
        let x   = a.solve_upper_triangular(&b).unwrap();

        println!("{}\n{}\n{}\n{}", tri, x, tri * x, b);

        relative_eq!(tri * x, b, epsilon = 1.0e-7)
    }

    fn tr_solve_lower_triangular(a: Matrix4<f64>, b: Matrix4x5<f64>) -> bool {
        let mut a = a;
        unzero_diagonal(&mut a);
        let tri = a.lower_triangle();
        let x   = a.tr_solve_lower_triangular(&b).unwrap();

        println!("{}\n{}\n{}\n{}", tri, x, tri * x, b);

        relative_eq!(tri.transpose() * x, b, epsilon = 1.0e-7)
    }

    fn tr_solve_upper_triangular(a: Matrix4<f64>, b: Matrix4x5<f64>) -> bool {
        let mut a = a;
        unzero_diagonal(&mut a);
        let tri = a.upper_triangle();
        let x   = a.tr_solve_upper_triangular(&b).unwrap();

        println!("{}\n{}\n{}\n{}", tri, x, tri * x, b);

        relative_eq!(tri.transpose() * x, b, epsilon = 1.0e-7)
    }
}
