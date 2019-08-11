extern crate nalgebra as na;

#[macro_use]
extern crate quickcheck;

use na::{U3, U4, MatrixMN};
use matrixcompare::compare_matrices;
use matrixcompare::comparators::{ExactElementwiseComparator, AbsoluteElementwiseComparator};

fn compare_integers_fail() {
    println!("Comparing two integer matrices.");

    let a = MatrixMN::<_, U3, U4>::from_row_slice(&[
        0, 1,  2,  3,
        4, 5,  6,  7,
        8, 9, -2, 11
    ]);

    let b = MatrixMN::<_, U3, U4>::from_row_slice(&[
        0, 1,  2,  3,
        4, 5,  6,  7,
        8, 9, 10, 11
    ]);

    if let Some(msg) = compare_matrices(a, b, &ExactElementwiseComparator).panic_message() {
        println!("{}", msg);
    }
}

fn compare_different_size() {
    println!("Comparing matrices of different size.");
    let a = MatrixMN::<_, U3, U3>::from_row_slice(&[
        0, 1,  2,
        4, 5,  6,
        8, 9, 10,
    ]);

    let b = MatrixMN::<_, U3, U4>::from_row_slice(&[
        0, 1,  2,  3,
        4, 5,  6,  7,
        8, 9, 10, 11
    ]);

    if let Some(msg) = compare_matrices(a, b, &ExactElementwiseComparator).panic_message() {
        println!("{}", msg);
    }
}

fn compare_f64_abs_tol_fail() {
    println!("Comparing two f64 matrices.");

    let a = MatrixMN::<f64, U3, U3>::from_row_slice(&[
        0.0, 1.0,  2.0 + 1e-10,
        4.0, 5.0,  6.0,
        8.0, 9.0, 10.0,
    ]);

    let b = MatrixMN::<_, U3, U3>::from_row_slice(&[
        0.0, 1.0,  2.0,
        4.0, 5.0,  6.0,
        8.0, 9.0, 10.0
    ]);

    let cmp = AbsoluteElementwiseComparator { tol: 1e-12 };
    if let Some(msg) = compare_matrices(a, b, &cmp).panic_message() {
        println!("{}", msg);
    }
}

fn main() {
    // This example mostly serves the purpose of demonstrating the kind of error messages
    // that are given upon comparison failure.
    // The more typical use case is using `assert_matrix_eq!` in tests.
    compare_integers_fail();
    println!("======================================================");
    compare_f64_abs_tol_fail();
    println!("======================================================");
    compare_different_size();
}