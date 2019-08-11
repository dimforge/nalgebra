//! Tests for `matrixcompare` integration.
//!
//! The `matrixcompare` crate itself is responsible for testing the actual comparison.
//! The tests here only check that the necessary trait implementations are correctly implemented,
//! in addition to some sanity checks with example input.

use nalgebra::{U4, U5, MatrixMN, DMatrix};

use matrixcompare::{assert_matrix_eq, DenseAccess};

#[cfg(feature = "arbitrary")]
quickcheck! {
    fn fetch_single_is_equivalent_to_index_f64(matrix: DMatrix<f64>) -> bool {
        for i in 0 .. matrix.nrows() {
            for j in 0 .. matrix.ncols() {
                if matrix.fetch_single(i, j) != *matrix.index((i, j)) {
                    return false;
                }
            }
        }

        true
    }

    fn matrixcompare_shape_agrees_with_matrix(matrix: DMatrix<f64>) -> bool {
        matrix.nrows() == <DMatrix<f64> as matrixcompare::Matrix<f64>>::rows(&matrix)
        &&
        matrix.ncols() == <DMatrix<f64> as matrixcompare::Matrix<f64>>::cols(&matrix)
    }
}

#[test]
fn assert_matrix_eq_dense_positive_comparison() {
    let a = MatrixMN::<_, U4, U5>::from_row_slice(&[
        1210, 1320, 1430, 1540, 1650,
        2310, 2420, 2530, 2640, 2750,
        3410, 3520, 3630, 3740, 3850,
        4510, 4620, 4730, 4840, 4950,
    ]);

    let b = MatrixMN::<_, U4, U5>::from_row_slice(&[
        1210, 1320, 1430, 1540, 1650,
        2310, 2420, 2530, 2640, 2750,
        3410, 3520, 3630, 3740, 3850,
        4510, 4620, 4730, 4840, 4950,
    ]);

    // Test matrices of static size
    assert_matrix_eq!(a, b);
    assert_matrix_eq!(&a, b);
    assert_matrix_eq!(a, &b);
    assert_matrix_eq!(&a, &b);

    // Test matrices of dynamic size
    let a_dyn = a.index((0..4, 0..5));
    let b_dyn = b.index((0..4, 0..5));
    assert_matrix_eq!(a_dyn, b_dyn);
    assert_matrix_eq!(a_dyn, &b_dyn);
    assert_matrix_eq!(&a_dyn, b_dyn);
    assert_matrix_eq!(&a_dyn, &b_dyn);
}

#[test]
#[should_panic]
fn assert_matrix_eq_dense_negative_comparison() {
    let a = MatrixMN::<_, U4, U5>::from_row_slice(&[
        1210, 1320, 1430, 1540, 1650,
        2310, 2420, 2530, 2640, 2750,
        3410, 3520, 3630, 3740, 3850,
        4510, 4620, -4730, 4840, 4950,
    ]);

    let b = MatrixMN::<_, U4, U5>::from_row_slice(&[
        1210, 1320, 1430, 1540, 1650,
        2310, 2420, 2530, 2640, 2750,
        3410, 3520, 3630, 3740, 3850,
        4510, 4620, 4730, 4840, 4950,
    ]);

    assert_matrix_eq!(a, b);
}