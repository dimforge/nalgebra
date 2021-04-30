use nalgebra_macros::{dmatrix, matrix};
use nalgebra::{DMatrix, SMatrix, Matrix3x2, Matrix1x2, Matrix1x3, Matrix1x4, Matrix2x1, Matrix2, Matrix2x3, Matrix2x4, Matrix3x1, Matrix3, Matrix3x4, Matrix4x1, Matrix4x2, Matrix4x3, Matrix4};

fn same_type<T>(_: &T, _: &T) {}

/// Wrapper for `assert_eq` that also asserts that the types are the same
macro_rules! assert_eq_and_type {
    ($left:expr, $right:expr $(,)?) => {
        same_type(&$left, &$right);
        assert_eq!($left, $right);
    };
}

#[test]
fn matrix_small_dims_exhaustive() {
    // 0x0
    assert_eq_and_type!(matrix![], SMatrix::<i32, 0, 0>::zeros());

    // 1xN
    assert_eq_and_type!(matrix![1], SMatrix::<i32, 1, 1>::new(1));
    assert_eq_and_type!(matrix![1, 2], Matrix1x2::new(1, 2));
    assert_eq_and_type!(matrix![1, 2, 3], Matrix1x3::new(1, 2, 3));
    assert_eq_and_type!(matrix![1, 2, 3, 4], Matrix1x4::new(1, 2, 3, 4));

    // 2xN
    assert_eq_and_type!(matrix![1; 2], Matrix2x1::new(1, 2));
    assert_eq_and_type!(matrix![1, 2; 3, 4], Matrix2::new(1, 2, 3, 4));
    assert_eq_and_type!(matrix![1, 2, 3; 4, 5, 6], Matrix2x3::new(1, 2, 3, 4, 5, 6));
    assert_eq_and_type!(matrix![1, 2, 3, 4; 5, 6, 7, 8], Matrix2x4::new(1, 2, 3, 4, 5, 6, 7, 8));

    // 3xN
    assert_eq_and_type!(matrix![1; 2; 3], Matrix3x1::new(1, 2, 3));
    assert_eq_and_type!(matrix![1, 2; 3, 4; 5, 6], Matrix3x2::new(1, 2, 3, 4, 5, 6));
    assert_eq_and_type!(matrix![1, 2, 3; 4, 5, 6; 7, 8, 9], Matrix3::new(1, 2, 3, 4, 5, 6, 7, 8, 9));
    assert_eq_and_type!(matrix![1, 2, 3, 4; 5, 6, 7, 8; 9, 10, 11, 12],
               Matrix3x4::new(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12));

    // 4xN
    assert_eq_and_type!(matrix![1; 2; 3; 4], Matrix4x1::new(1, 2, 3, 4));
    assert_eq_and_type!(matrix![1, 2; 3, 4; 5, 6; 7, 8], Matrix4x2::new(1, 2, 3, 4, 5, 6, 7, 8));
    assert_eq_and_type!(matrix![1, 2, 3; 4, 5, 6; 7, 8, 9; 10, 11, 12],
               Matrix4x3::new(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12));
    assert_eq_and_type!(matrix![1, 2, 3, 4; 5, 6, 7, 8; 9, 10, 11, 12; 13, 14, 15, 16],
               Matrix4::new(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16));
}

#[test]
fn matrix_const_fn() {
    // Ensure that matrix! can be used in const contexts
    const _: SMatrix<i32, 0, 0> = matrix![];
    const _: SMatrix<i32, 1, 2> = matrix![1, 2];
    const _: SMatrix<i32, 2, 3> = matrix![1, 2, 3; 4, 5, 6];
}

#[test]
fn dmatrix_small_dims_exhaustive() {
    // 0x0
    assert_eq_and_type!(dmatrix![], DMatrix::<i32>::zeros(0, 0));

    // 1xN
    assert_eq_and_type!(dmatrix![1], DMatrix::from_row_slice(1, 1, &[1]));
    assert_eq_and_type!(dmatrix![1, 2], DMatrix::from_row_slice(1, 2, &[1, 2]));
    assert_eq_and_type!(dmatrix![1, 2, 3], DMatrix::from_row_slice(1, 3, &[1, 2, 3]));
    assert_eq_and_type!(dmatrix![1, 2, 3, 4], DMatrix::from_row_slice(1, 4, &[1, 2, 3, 4]));

    // 2xN
    assert_eq_and_type!(dmatrix![1; 2], DMatrix::from_row_slice(2, 1,  &[1, 2]));
    assert_eq_and_type!(dmatrix![1, 2; 3, 4], DMatrix::from_row_slice(2, 2,  &[1, 2, 3, 4]));
    assert_eq_and_type!(dmatrix![1, 2, 3; 4, 5, 6], DMatrix::from_row_slice(2, 3,  &[1, 2, 3, 4, 5, 6]));
    assert_eq_and_type!(dmatrix![1, 2, 3, 4; 5, 6, 7, 8], DMatrix::from_row_slice(2, 4,  &[1, 2, 3, 4, 5, 6, 7, 8]));

    // 3xN
    assert_eq_and_type!(dmatrix![1; 2; 3], DMatrix::from_row_slice(3, 1,  &[1, 2, 3]));
    assert_eq_and_type!(dmatrix![1, 2; 3, 4; 5, 6], DMatrix::from_row_slice(3, 2,  &[1, 2, 3, 4, 5, 6]));
    assert_eq_and_type!(dmatrix![1, 2, 3; 4, 5, 6; 7, 8, 9], DMatrix::from_row_slice(3, 3,  &[1, 2, 3, 4, 5, 6, 7, 8, 9]));
    assert_eq_and_type!(dmatrix![1, 2, 3, 4; 5, 6, 7, 8; 9, 10, 11, 12],
               DMatrix::from_row_slice(3, 4,  &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]));

    // 4xN
    assert_eq_and_type!(dmatrix![1; 2; 3; 4], DMatrix::from_row_slice(4, 1,  &[1, 2, 3, 4]));
    assert_eq_and_type!(dmatrix![1, 2; 3, 4; 5, 6; 7, 8], DMatrix::from_row_slice(4, 2,  &[1, 2, 3, 4, 5, 6, 7, 8]));
    assert_eq_and_type!(dmatrix![1, 2, 3; 4, 5, 6; 7, 8, 9; 10, 11, 12],
               DMatrix::from_row_slice(4, 3,  &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]));
    assert_eq_and_type!(dmatrix![1, 2, 3, 4; 5, 6, 7, 8; 9, 10, 11, 12; 13, 14, 15, 16],
               DMatrix::from_row_slice(4, 4,  &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]));
}
