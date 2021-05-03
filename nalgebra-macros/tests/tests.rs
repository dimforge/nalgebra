use nalgebra::{
    DMatrix, DVector, Matrix1x2, Matrix1x3, Matrix1x4, Matrix2, Matrix2x1, Matrix2x3, Matrix2x4,
    Matrix3, Matrix3x1, Matrix3x2, Matrix3x4, Matrix4, Matrix4x1, Matrix4x2, Matrix4x3, SMatrix,
    SVector, Vector1, Vector2, Vector3, Vector4, Vector5, Vector6,
};
use nalgebra_macros::{dmatrix, dvector, matrix, vector};

fn check_statically_same_type<T>(_: &T, _: &T) {}

/// Wrapper for `assert_eq` that also asserts that the types are the same
macro_rules! assert_eq_and_type {
    ($left:expr, $right:expr $(,)?) => {
        check_statically_same_type(&$left, &$right);
        assert_eq!($left, $right);
    };
}

// Skip rustfmt because it just makes the test bloated without making it more readable
#[rustfmt::skip]
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

// Skip rustfmt because it just makes the test bloated without making it more readable
#[rustfmt::skip]
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

// Skip rustfmt because it just makes the test bloated without making it more readable
#[rustfmt::skip]
#[test]
fn vector_small_dims_exhaustive() {
    assert_eq_and_type!(vector![], SVector::<i32, 0>::zeros());
    assert_eq_and_type!(vector![1], Vector1::<i32>::new(1));
    assert_eq_and_type!(vector![1, 2], Vector2::new(1, 2));
    assert_eq_and_type!(vector![1, 2, 3], Vector3::new(1, 2, 3));
    assert_eq_and_type!(vector![1, 2, 3, 4], Vector4::new(1, 2, 3, 4));
    assert_eq_and_type!(vector![1, 2, 3, 4, 5], Vector5::new(1, 2, 3, 4, 5));
    assert_eq_and_type!(vector![1, 2, 3, 4, 5, 6], Vector6::new(1, 2, 3, 4, 5, 6));
}

#[test]
fn vector_const_fn() {
    // Ensure that vector! can be used in const contexts
    const _: SVector<i32, 0> = vector![];
    const _: Vector1<i32> = vector![1];
    const _: Vector2<i32> = vector![1, 2];
    const _: Vector6<i32> = vector![1, 2, 3, 4, 5, 6];
}

// Skip rustfmt because it just makes the test bloated without making it more readable
#[rustfmt::skip]
#[test]
fn dvector_small_dims_exhaustive() {
    assert_eq_and_type!(dvector![], DVector::<i32>::zeros(0));
    assert_eq_and_type!(dvector![1], DVector::from_column_slice(&[1]));
    assert_eq_and_type!(dvector![1, 2], DVector::from_column_slice(&[1, 2]));
    assert_eq_and_type!(dvector![1, 2, 3], DVector::from_column_slice(&[1, 2, 3]));
    assert_eq_and_type!(dvector![1, 2, 3, 4], DVector::from_column_slice(&[1, 2, 3, 4]));
    assert_eq_and_type!(dvector![1, 2, 3, 4, 5], DVector::from_column_slice(&[1, 2, 3, 4, 5]));
    assert_eq_and_type!(dvector![1, 2, 3, 4, 5, 6], DVector::from_column_slice(&[1, 2, 3, 4, 5, 6]));
}

#[test]
fn matrix_trybuild_tests() {
    let t = trybuild::TestCases::new();

    // Verify error message when we give a matrix with mismatched dimensions
    t.compile_fail("tests/trybuild/matrix_mismatched_dimensions.rs");
}

#[test]
fn dmatrix_trybuild_tests() {
    let t = trybuild::TestCases::new();

    // Verify error message when we give a matrix with mismatched dimensions
    t.compile_fail("tests/trybuild/dmatrix_mismatched_dimensions.rs");
}
