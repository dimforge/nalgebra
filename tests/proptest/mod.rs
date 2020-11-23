//! Tests for proptest-related functionality.
use nalgebra::base::dimension::*;
use nalgebra::proptest::{matrix, DimRange, MatrixStrategy};
use nalgebra::{DMatrix, DVector, Dim, Matrix3, MatrixMN, Vector3};
use proptest::prelude::*;
use proptest::strategy::ValueTree;
use proptest::test_runner::TestRunner;

/// Generate a proptest that tests that all matrices generated with the
/// provided rows and columns conform to the constraints defined by the
/// input.
macro_rules! generate_matrix_sanity_test {
    ($test_name:ident, $rows:expr, $cols:expr) => {
        proptest! {
            #[test]
            fn $test_name(a in matrix(-5 ..= 5i32, $rows, $cols)) {
                // let a: MatrixMN<_, $rows, $cols> = a;
                let rows_range = DimRange::from($rows);
                let cols_range = DimRange::from($cols);
                prop_assert!(a.nrows() >= rows_range.lower_bound().value()
                          && a.nrows() <= rows_range.upper_bound().value());
                prop_assert!(a.ncols() >= cols_range.lower_bound().value()
                          && a.ncols() <= cols_range.upper_bound().value());
                prop_assert!(a.iter().all(|x_ij| *x_ij >= -5 && *x_ij <= 5));
            }
        }
    };
}

// Test all fixed-size matrices with row/col dimensions up to 3
generate_matrix_sanity_test!(test_matrix_u0_u0, U0, U0);
generate_matrix_sanity_test!(test_matrix_u1_u0, U1, U0);
generate_matrix_sanity_test!(test_matrix_u0_u1, U0, U1);
generate_matrix_sanity_test!(test_matrix_u1_u1, U1, U1);
generate_matrix_sanity_test!(test_matrix_u2_u1, U2, U1);
generate_matrix_sanity_test!(test_matrix_u1_u2, U1, U2);
generate_matrix_sanity_test!(test_matrix_u2_u2, U2, U2);
generate_matrix_sanity_test!(test_matrix_u3_u2, U3, U2);
generate_matrix_sanity_test!(test_matrix_u2_u3, U2, U3);
generate_matrix_sanity_test!(test_matrix_u3_u3, U3, U3);

// Similarly test all heap-allocated but fixed dim ranges
generate_matrix_sanity_test!(test_matrix_0_0, 0, 0);
generate_matrix_sanity_test!(test_matrix_0_1, 0, 1);
generate_matrix_sanity_test!(test_matrix_1_0, 1, 0);
generate_matrix_sanity_test!(test_matrix_1_1, 1, 1);
generate_matrix_sanity_test!(test_matrix_2_1, 2, 1);
generate_matrix_sanity_test!(test_matrix_1_2, 1, 2);
generate_matrix_sanity_test!(test_matrix_2_2, 2, 2);
generate_matrix_sanity_test!(test_matrix_3_2, 3, 2);
generate_matrix_sanity_test!(test_matrix_2_3, 2, 3);
generate_matrix_sanity_test!(test_matrix_3_3, 3, 3);

// Test arbitrary inputs
generate_matrix_sanity_test!(test_matrix_input_1, U5, 1..=5);
generate_matrix_sanity_test!(test_matrix_input_2, 3..=4, 1..=5);
generate_matrix_sanity_test!(test_matrix_input_3, 1..=2, U3);
generate_matrix_sanity_test!(test_matrix_input_4, 3, U4);

#[test]
fn test_matrix_output_types() {
    // Test that the dimension types are correct for the given inputs
    let _: MatrixStrategy<_, U3, U4> = matrix(-5..5, U3, U4);
    let _: MatrixStrategy<_, U3, U3> = matrix(-5..5, U3, U3);
    let _: MatrixStrategy<_, U3, Dynamic> = matrix(-5..5, U3, 1..=5);
    let _: MatrixStrategy<_, Dynamic, U3> = matrix(-5..5, 1..=5, U3);
    let _: MatrixStrategy<_, Dynamic, Dynamic> = matrix(-5..5, 1..=5, 1..=5);
}

// Below we have some tests to ensure that specific instances of MatrixMN are usable
// in a typical proptest scenario where we (implicitly) use the `Arbitrary` trait
proptest! {
    #[test]
    fn ensure_arbitrary_test_compiles_matrix3(_: Matrix3<i32>) {}

    #[test]
    fn ensure_arbitrary_test_compiles_matrixmn_u3_dynamic(_: MatrixMN<i32, U3, Dynamic>) {}

    #[test]
    fn ensure_arbitrary_test_compiles_matrixmn_dynamic_u3(_: MatrixMN<i32, Dynamic, U3>) {}

    #[test]
    fn ensure_arbitrary_test_compiles_dmatrix(_: DMatrix<i32>) {}

    #[test]
    fn ensure_arbitrary_test_compiles_vector3(_: Vector3<i32>) {}

    #[test]
    fn ensure_arbitrary_test_compiles_dvector(_: DVector<i32>) {}
}

#[test]
fn matrix_shrinking_satisfies_constraints() {
    // We use a deterministic test runner to make the test "stable".
    let mut runner = TestRunner::deterministic();

    let strategy = matrix(-1..=2, 1..=3, 2..=4);

    let num_matrices = 25;

    macro_rules! maybeprintln {
        ($($arg:tt)*) => {
            // Uncomment the below line to enable printing of matrix sequences. This is handy
            // for manually inspecting the sequences of simplified matrices.
            // println!($($arg)*)
        };
    }

    maybeprintln!("========================== (begin generation process)");

    for _ in 0..num_matrices {
        let mut tree = strategy
            .new_tree(&mut runner)
            .expect("Tree generation should not fail.");

        let mut current = Some(tree.current());

        maybeprintln!("------------------");

        while let Some(matrix) = current {
            maybeprintln!("{}", matrix);

            assert!(
                matrix.iter().all(|&v| v >= -1 && v <= 2),
                "All matrix elements must satisfy constraints"
            );
            assert!(
                matrix.nrows() >= 1 && matrix.nrows() <= 3,
                "Number of rows in matrix must satisfy constraints."
            );
            assert!(
                matrix.ncols() >= 2 && matrix.ncols() <= 4,
                "Number of columns in matrix must satisfy constraints."
            );

            current = if tree.simplify() {
                Some(tree.current())
            } else {
                None
            }
        }
    }

    maybeprintln!("========================== (end of generation process)");
}

#[cfg(feature = "slow-tests")]
mod slow {
    use super::*;
    use itertools::Itertools;
    use std::collections::HashSet;
    use std::iter::repeat;

    #[cfg(feature = "slow-tests")]
    #[test]
    fn matrix_samples_all_possible_outputs() {
        // Test that the proptest generation covers all possible outputs for a small space of inputs
        // given enough samples.

        // We use a deterministic test runner to make the test "stable".
        let mut runner = TestRunner::deterministic();

        // This number needs to be high enough so that we with high probability sample
        // all possible cases
        let num_generated_matrices = 200000;

        let values = -1..=1;
        let rows = 0..=2;
        let cols = 0..=3;
        let strategy = matrix(values.clone(), rows.clone(), cols.clone());

        // Enumerate all possible combinations
        let mut all_combinations = HashSet::new();
        for nrows in rows {
            for ncols in cols.clone() {
                // For the given number of rows and columns
                let n_values = nrows * ncols;

                if n_values == 0 {
                    // If we have zero rows or columns, the set of matrices with the given
                    // rows and columns is a single element: an empty matrix
                    all_combinations.insert(DMatrix::from_row_slice(nrows, ncols, &[]));
                } else {
                    // Otherwise, we need to sample all possible matrices.
                    // To do this, we generate the values as the (multi) Cartesian product
                    // of the value sets. For example, for a 2x2 matrices, we consider
                    // all possible 4-element arrays that the matrices can take by
                    // considering all elements in the cartesian product
                    //  V x V x V x V
                    // where V is the set of eligible values, e.g. V := -1 ..= 1
                    for matrix_values in repeat(values.clone())
                        .take(n_values)
                        .multi_cartesian_product()
                    {
                        all_combinations.insert(DMatrix::from_row_slice(
                            nrows,
                            ncols,
                            &matrix_values,
                        ));
                    }
                }
            }
        }

        let mut visited_combinations = HashSet::new();
        for _ in 0..num_generated_matrices {
            let tree = strategy
                .new_tree(&mut runner)
                .expect("Tree generation should not fail");
            let matrix = tree.current();
            visited_combinations.insert(matrix.clone());
        }

        assert_eq!(
            visited_combinations, all_combinations,
            "Did not sample all possible values."
        );
    }
}
