use nalgebra::dimension::U1;
use nalgebra::{
    DMatrix, DMatrixView, DVector, Dyn, Matrix, Matrix1x2, Matrix1x3, Matrix1x4, Matrix2,
    Matrix2x1, Matrix2x3, Matrix2x4, Matrix3, Matrix3x1, Matrix3x2, Matrix3x4, Matrix4, Matrix4x1,
    Matrix4x2, Matrix4x3, OMatrix, Point, Point1, Point2, Point3, Point4, Point5, Point6, SMatrix,
    SMatrixView, SMatrixViewMut, SVector, Scalar, Vector1, Vector2, Vector3, Vector4, Vector5,
    Vector6, U2,
};
use nalgebra_macros::{dmatrix, dvector, matrix, point, stack, vector};
use num_traits::Zero;

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

#[test]
fn matrix_trailing_semi() {
    matrix![1, 2;];
    dmatrix![1, 2;];
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

// Skip rustfmt because it just makes the test bloated without making it more readable
#[rustfmt::skip]
#[test]
fn point_small_dims_exhaustive() {
    assert_eq_and_type!(point![], Point::<i32, 0>::origin());
    assert_eq_and_type!(point![1], Point1::<i32>::new(1));
    assert_eq_and_type!(point![1, 2], Point2::new(1, 2));
    assert_eq_and_type!(point![1, 2, 3], Point3::new(1, 2, 3));
    assert_eq_and_type!(point![1, 2, 3, 4], Point4::new(1, 2, 3, 4));
    assert_eq_and_type!(point![1, 2, 3, 4, 5], Point5::new(1, 2, 3, 4, 5));
    assert_eq_and_type!(point![1, 2, 3, 4, 5, 6], Point6::new(1, 2, 3, 4, 5, 6));
}

#[test]
fn vector_const_fn() {
    // Ensure that vector! can be used in const contexts
    const _: SVector<i32, 0> = vector![];
    const _: Vector1<i32> = vector![1];
    const _: Vector2<i32> = vector![1, 2];
    const _: Vector6<i32> = vector![1, 2, 3, 4, 5, 6];
}

#[test]
fn point_const_fn() {
    // Ensure that vector! can be used in const contexts
    const _: Point<i32, 0> = point![];
    const _: Point1<i32> = point![1];
    const _: Point2<i32> = point![1, 2];
    const _: Point6<i32> = point![1, 2, 3, 4, 5, 6];
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
fn vector_trailing_comma() {
    vector![1, 2,];
    point![1, 2,];
    dvector![1, 2,];
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

#[test]
fn matrix_builtin_types() {
    // Check that matrix! compiles for all built-in types
    const _: SMatrix<i8, 2, 2> = matrix![0, 1; 2, 3];
    const _: SMatrix<i16, 2, 2> = matrix![0, 1; 2, 3];
    const _: SMatrix<i32, 2, 2> = matrix![0, 1; 2, 3];
    const _: SMatrix<i64, 2, 2> = matrix![0, 1; 2, 3];
    const _: SMatrix<isize, 2, 2> = matrix![0, 1; 2, 3];
    const _: SMatrix<u8, 2, 2> = matrix![0, 1; 2, 3];
    const _: SMatrix<u16, 2, 2> = matrix![0, 1; 2, 3];
    const _: SMatrix<u32, 2, 2> = matrix![0, 1; 2, 3];
    const _: SMatrix<u64, 2, 2> = matrix![0, 1; 2, 3];
    const _: SMatrix<usize, 2, 2> = matrix![0, 1; 2, 3];
    const _: SMatrix<f32, 2, 2> = matrix![0.0, 1.0; 2.0, 3.0];
    const _: SMatrix<f64, 2, 2> = matrix![0.0, 1.0; 2.0, 3.0];
}

#[test]
fn vector_builtin_types() {
    // Check that vector! compiles for all built-in types
    const _: SVector<i8, 4> = vector![0, 1, 2, 3];
    const _: SVector<i16, 4> = vector![0, 1, 2, 3];
    const _: SVector<i32, 4> = vector![0, 1, 2, 3];
    const _: SVector<i64, 4> = vector![0, 1, 2, 3];
    const _: SVector<isize, 4> = vector![0, 1, 2, 3];
    const _: SVector<u8, 4> = vector![0, 1, 2, 3];
    const _: SVector<u16, 4> = vector![0, 1, 2, 3];
    const _: SVector<u32, 4> = vector![0, 1, 2, 3];
    const _: SVector<u64, 4> = vector![0, 1, 2, 3];
    const _: SVector<usize, 4> = vector![0, 1, 2, 3];
    const _: SVector<f32, 4> = vector![0.0, 1.0, 2.0, 3.0];
    const _: SVector<f64, 4> = vector![0.0, 1.0, 2.0, 3.0];
}

#[test]
fn dmatrix_builtin_types() {
    // Check that dmatrix! compiles for all built-in types
    let _: DMatrix<i8> = dmatrix![0, 1; 2, 3];
    let _: DMatrix<i16> = dmatrix![0, 1; 2, 3];
    let _: DMatrix<i32> = dmatrix![0, 1; 2, 3];
    let _: DMatrix<i64> = dmatrix![0, 1; 2, 3];
    let _: DMatrix<isize> = dmatrix![0, 1; 2, 3];
    let _: DMatrix<u8> = dmatrix![0, 1; 2, 3];
    let _: DMatrix<u16> = dmatrix![0, 1; 2, 3];
    let _: DMatrix<u32> = dmatrix![0, 1; 2, 3];
    let _: DMatrix<u64> = dmatrix![0, 1; 2, 3];
    let _: DMatrix<usize> = dmatrix![0, 1; 2, 3];
    let _: DMatrix<f32> = dmatrix![0.0, 1.0; 2.0, 3.0];
    let _: DMatrix<f64> = dmatrix![0.0, 1.0; 2.0, 3.0];
}

#[test]
fn point_builtin_types() {
    // Check that point! compiles for all built-in types
    const _: Point<i8, 4> = point![0, 1, 2, 3];
    const _: Point<i16, 4> = point![0, 1, 2, 3];
    const _: Point<i32, 4> = point![0, 1, 2, 3];
    const _: Point<i64, 4> = point![0, 1, 2, 3];
    const _: Point<isize, 4> = point![0, 1, 2, 3];
    const _: Point<u8, 4> = point![0, 1, 2, 3];
    const _: Point<u16, 4> = point![0, 1, 2, 3];
    const _: Point<u32, 4> = point![0, 1, 2, 3];
    const _: Point<u64, 4> = point![0, 1, 2, 3];
    const _: Point<usize, 4> = point![0, 1, 2, 3];
    const _: Point<f32, 4> = point![0.0, 1.0, 2.0, 3.0];
    const _: Point<f64, 4> = point![0.0, 1.0, 2.0, 3.0];
}

#[test]
fn dvector_builtin_types() {
    // Check that dvector! compiles for all built-in types
    let _: DVector<i8> = dvector![0, 1, 2, 3];
    let _: DVector<i16> = dvector![0, 1, 2, 3];
    let _: DVector<i32> = dvector![0, 1, 2, 3];
    let _: DVector<i64> = dvector![0, 1, 2, 3];
    let _: DVector<isize> = dvector![0, 1, 2, 3];
    let _: DVector<u8> = dvector![0, 1, 2, 3];
    let _: DVector<u16> = dvector![0, 1, 2, 3];
    let _: DVector<u32> = dvector![0, 1, 2, 3];
    let _: DVector<u64> = dvector![0, 1, 2, 3];
    let _: DVector<usize> = dvector![0, 1, 2, 3];
    let _: DVector<f32> = dvector![0.0, 1.0, 2.0, 3.0];
    let _: DVector<f64> = dvector![0.0, 1.0, 2.0, 3.0];
}

/// Black box function that's just used for testing macros with function call expressions.
fn f<T>(x: T) -> T {
    x
}

#[rustfmt::skip]
#[test]
fn matrix_arbitrary_expressions() {
    // Test that matrix! supports arbitrary expressions for its elements
    let a = matrix![1 + 2       ,     2 * 3;
                    4 * f(5 + 6), 7 - 8 * 9];
    let a_expected = Matrix2::new(1 + 2       ,     2 * 3,
                                  4 * f(5 + 6), 7 - 8 * 9);
    assert_eq_and_type!(a, a_expected);
}

#[rustfmt::skip]
#[test]
fn dmatrix_arbitrary_expressions() {
    // Test that dmatrix! supports arbitrary expressions for its elements
    let a = dmatrix![1 + 2       ,     2 * 3;
                     4 * f(5 + 6), 7 - 8 * 9];
    let a_expected = DMatrix::from_row_slice(2, 2, &[1 + 2       ,     2 * 3,
                                                     4 * f(5 + 6), 7 - 8 * 9]);
    assert_eq_and_type!(a, a_expected);
}

#[rustfmt::skip]
#[test]
fn vector_arbitrary_expressions() {
    // Test that vector! supports arbitrary expressions for its elements
    let a = vector![1 + 2, 2 * 3, 4 * f(5 + 6), 7 - 8 * 9];
    let a_expected = Vector4::new(1 + 2, 2 * 3, 4 * f(5 + 6), 7 - 8 * 9);
    assert_eq_and_type!(a, a_expected);
}

#[rustfmt::skip]
#[test]
fn point_arbitrary_expressions() {
    // Test that point! supports arbitrary expressions for its elements
    let a = point![1 + 2, 2 * 3, 4 * f(5 + 6), 7 - 8 * 9];
    let a_expected = Point4::new(1 + 2, 2 * 3, 4 * f(5 + 6), 7 - 8 * 9);
    assert_eq_and_type!(a, a_expected);
}

#[rustfmt::skip]
#[test]
fn dvector_arbitrary_expressions() {
    // Test that dvector! supports arbitrary expressions for its elements
    let a = dvector![1 + 2, 2 * 3, 4 * f(5 + 6), 7 - 8 * 9];
    let a_expected = DVector::from_column_slice(&[1 + 2, 2 * 3, 4 * f(5 + 6), 7 - 8 * 9]);
    assert_eq_and_type!(a, a_expected);
}

/// Simple implementation that stacks dynamic matrices.
///
/// Used for verifying results of the stack! macro. `None` entries are considered to represent
/// a zero block.
fn stack_dyn<T: Scalar + Zero>(blocks: DMatrix<Option<DMatrix<T>>>) -> DMatrix<T> {
    let row_counts: Vec<usize> = blocks
        .row_iter()
        .map(|block_row| {
            block_row
                .iter()
                .map(|block_or_implicit_zero| {
                    block_or_implicit_zero.as_ref().map(|block| block.nrows())
                })
                .reduce(|nrows1, nrows2| match (nrows1, nrows2) {
                    (Some(_), None) => nrows1,
                    (None, Some(_)) => nrows2,
                    (None, None) => None,
                    (Some(nrows1), Some(nrows2)) if nrows1 == nrows2 => Some(nrows1),
                    _ => panic!("Number of rows must be consistent in each block row"),
                })
                .unwrap_or(Some(0))
                .expect("Each block row must have at least one entry which is not a zero literal")
        })
        .collect();
    let col_counts: Vec<usize> = blocks
        .column_iter()
        .map(|block_col| {
            block_col
                .iter()
                .map(|block_or_implicit_zero| {
                    block_or_implicit_zero.as_ref().map(|block| block.ncols())
                })
                .reduce(|ncols1, ncols2| match (ncols1, ncols2) {
                    (Some(_), None) => ncols1,
                    (None, Some(_)) => ncols2,
                    (None, None) => None,
                    (Some(ncols1), Some(ncols2)) if ncols1 == ncols2 => Some(ncols1),
                    _ => panic!("Number of columns must be consistent in each block column"),
                })
                .unwrap_or(Some(0))
                .expect(
                    "Each block column must have at least one entry which is not a zero literal",
                )
        })
        .collect();

    let nrows_total = row_counts.iter().sum();
    let ncols_total = col_counts.iter().sum();
    let mut output = DMatrix::zeros(nrows_total, ncols_total);

    let mut col_offset = 0;
    for j in 0..blocks.ncols() {
        let mut row_offset = 0;
        for i in 0..blocks.nrows() {
            if let Some(input_ij) = &blocks[(i, j)] {
                let (block_nrows, block_ncols) = input_ij.shape();
                output
                    .view_mut((row_offset, col_offset), (block_nrows, block_ncols))
                    .copy_from(&input_ij);
            }
            row_offset += row_counts[i];
        }
        col_offset += col_counts[j];
    }

    output
}

macro_rules! stack_dyn_convert_to_dmatrix_option {
    (0) => {
        None
    };
    ($entry:expr) => {
        Some($entry.as_view::<Dyn, Dyn, U1, Dyn>().clone_owned())
    };
}

/// Helper macro that compares the result of stack! with a simplified implementation that
/// works only with heap-allocated data.
///
/// This implementation is essentially radically different to the implementation in stack!,
/// so if they both match, then it's a good sign that the stack! impl is correct.
macro_rules! verify_stack {
    ($matrix_type:ty ; [$($($entry:expr),*);*]) => {
        {
            // Our input has the same syntax as the stack! macro (and matrix! macro, for that matter)
            let stack_result: $matrix_type = stack![$($($entry),*);*];
            // Use the dmatrix! macro to nest matrices into each other
            let dyn_result = stack_dyn(
                dmatrix![$($(stack_dyn_convert_to_dmatrix_option!($entry)),*);*]
            );
            // println!("{}", stack_result);
            // println!("{}", dyn_result);
            assert_eq!(stack_result, dyn_result);
        }
    }
}

#[test]
fn stack_simple() {
    let m = stack![
        Matrix2::<usize>::identity(), 0;
        0, &Matrix2::identity();
    ];

    assert_eq_and_type!(m, Matrix4::identity());
}

#[test]
fn stack_diag() {
    let m = stack![
        0, matrix![1, 2; 3, 4;];
        matrix![5, 6; 7, 8;], 0;
    ];

    let res = matrix![
        0, 0, 1, 2;
        0, 0, 3, 4;
        5, 6, 0, 0;
        7, 8, 0, 0;
    ];

    assert_eq_and_type!(m, res);
}

#[test]
fn stack_dynamic() {
    let m = stack![
        matrix![ 1, 2; 3, 4; ], 0;
        0, dmatrix![7, 8, 9; 10, 11, 12; ];
    ];

    let res = dmatrix![
        1, 2, 0, 0, 0;
        3, 4, 0, 0, 0;
        0, 0, 7, 8, 9;
        0, 0, 10, 11, 12;
    ];

    assert_eq_and_type!(m, res);
}

#[test]
fn stack_nested() {
    let m = stack![
        stack![ matrix![1, 2; 3, 4;]; matrix![5, 6;]],
        stack![ matrix![7;9;10;], matrix![11; 12; 13;] ];
    ];

    let res = matrix![
        1, 2, 7, 11;
        3, 4, 9, 12;
        5, 6, 10, 13;
    ];

    assert_eq_and_type!(m, res);
}

#[test]
fn stack_single() {
    let a = matrix![1, 2; 3, 4];
    let b = stack![a];

    assert_eq_and_type!(a, b);
}

#[test]
fn stack_single_row() {
    let a = matrix![1, 2; 3, 4];
    let m = stack![a, a];

    let res = matrix![
        1, 2, 1, 2;
        3, 4, 3, 4;
    ];

    assert_eq_and_type!(m, res);
}

#[test]
fn stack_single_col() {
    let a = matrix![1, 2; 3, 4];
    let m = stack![a; a];

    let res = matrix![
        1, 2;
        3, 4;
        1, 2;
        3, 4;
    ];

    assert_eq_and_type!(m, res);
}

#[test]
#[rustfmt::skip]
fn stack_expr() {
    let a = matrix![1, 2; 3, 4];
    let b = matrix![5, 6; 7, 8];
    let m = stack![a + b; 2i32 * b - a];

    let res = matrix![
         6, 8;
        10, 12;
         4, 4;
         4, 4;
    ];

    assert_eq_and_type!(m, res);
}

#[test]
fn stack_edge_cases() {
    {
        // Empty stack should return zero matrix with specified type
        let _: SMatrix<i32, 0, 0> = stack![];
        let _: SMatrix<f64, 0, 0> = stack![];
    }
}

#[rustfmt::skip]
#[test]
fn stack_many_tests() {
    // s prefix means static, d prefix means dynamic
    // Static matrices
    let s_0x0: SMatrix<i32, 0, 0> = matrix![];
    let s_0x1: SMatrix<i32, 0, 1> = Matrix::default();
    let s_1x0: SMatrix<i32, 1, 0> = Matrix::default();
    let s_1x1: SMatrix<i32, 1, 1> = matrix![1];
    let s_2x2: SMatrix<i32, 2, 2> = matrix![6, 7; 8, 9];
    let s_2x3: SMatrix<i32, 2, 3> = matrix![16, 17, 18; 19, 20, 21];
    let s_3x3: SMatrix<i32, 3, 3> = matrix![28, 29, 30; 31, 32, 33; 34, 35, 36];

    // Dynamic matrices
    let d_0x0: DMatrix<i32> = dmatrix![];
    let d_1x2: DMatrix<i32> = dmatrix![9, 10];
    let d_2x2: DMatrix<i32> = dmatrix![5, 6; 7, 8];
    let d_4x4: DMatrix<i32> = dmatrix![10, 11, 12, 13; 14, 15, 16, 17; 18, 19, 20, 21; 22, 23, 24, 25];

    // Check for weirdness with matrices that have zero row/cols
    verify_stack!(SMatrix<_, 0, 0>; [s_0x0]);
    verify_stack!(SMatrix<_, 0, 1>; [s_0x1]);
    verify_stack!(SMatrix<_, 1, 0>; [s_1x0]);
    verify_stack!(SMatrix<_, 0, 0>; [s_0x0; s_0x0]);
    verify_stack!(SMatrix<_, 0, 0>; [s_0x0, s_0x0; s_0x0, s_0x0]);
    verify_stack!(SMatrix<_, 0, 2>; [s_0x1, s_0x1]);
    verify_stack!(SMatrix<_, 2, 0>; [s_1x0; s_1x0]);
    verify_stack!(SMatrix<_, 1, 0>; [s_1x0, s_1x0]);
    verify_stack!(DMatrix<_>; [d_0x0]);

    // Horizontal stacking
    verify_stack!(SMatrix<_, 1, 2>; [s_1x1, s_1x1]);
    verify_stack!(SMatrix<_, 2, 4>; [s_2x2, s_2x2]);
    verify_stack!(DMatrix<_>; [d_1x2, d_1x2]);

    // Vertical stacking
    verify_stack!(SMatrix<_, 2, 1>; [s_1x1; s_1x1]);
    verify_stack!(SMatrix<_, 4, 2>; [s_2x2; s_2x2]);
    verify_stack!(DMatrix<_>; [d_2x2; d_2x2]);

    // Mix static and dynamic matrices
    verify_stack!(OMatrix<_, U2, Dyn>; [s_2x2, d_2x2]);
    verify_stack!(OMatrix<_, Dyn, U2>; [s_2x2; d_1x2]);

    // Stack more than two matrices
    verify_stack!(SMatrix<_, 1, 3>; [s_1x1, s_1x1, s_1x1]);
    verify_stack!(DMatrix<_>; [d_1x2, d_1x2, d_1x2]);

    // Slightly larger dims
    verify_stack!(SMatrix<_, 3, 6>; [s_3x3, s_3x3]);
    verify_stack!(DMatrix<_>; [d_4x4; d_4x4]);
    verify_stack!(SMatrix<_, 4, 7>; [s_2x2, s_2x3, d_2x2;
                                     d_2x2, s_2x3, s_2x2]);

    // Mix of references and owned
    verify_stack!(OMatrix<_, Dyn, U2>; [&s_2x2; &d_1x2]);
    verify_stack!(SMatrix<_, 4, 7>; [ s_2x2, &s_2x3, d_2x2;
                                     &d_2x2, s_2x3, &s_2x2]);

    // Views
    let s_2x2_v: SMatrixView<_, 2, 2> = s_2x2.as_view();
    let s_2x3_v: SMatrixView<_, 2, 3> = s_2x3.as_view();
    let d_2x2_v: DMatrixView<_> = d_2x2.as_view();
    let mut s_2x2_vm = s_2x2.clone();
    let s_2x2_vm: SMatrixViewMut<_, 2, 2> = s_2x2_vm.as_view_mut();
    let mut s_2x3_vm = s_2x3.clone();
    let s_2x3_vm: SMatrixViewMut<_, 2, 3> = s_2x3_vm.as_view_mut();
    verify_stack!(SMatrix<_, 4, 7>; [ s_2x2_vm, &s_2x3_vm,  d_2x2_v;
                                      &d_2x2_v,   s_2x3_v, &s_2x2_v]);

    // Expressions
    let matrix_fn = |matrix: &DMatrix<_>| matrix.map(|x_ij| x_ij * 3);
    verify_stack!(SMatrix<_, 2, 5>; [ 2 * s_2x2 - 3 * &d_2x2, s_2x3 + 2 * s_2x3]);
    verify_stack!(DMatrix<_>; [ 2 * matrix_fn(&d_2x2) ]);
    verify_stack!(SMatrix<_, 2, 5>; [ (|matrix| 4 * matrix)(s_2x2), s_2x3 ]);
}

#[test]
fn stack_trybuild_tests() {
    let t = trybuild::TestCases::new();

    // Verify error message when a row or column only contains a zero entry
    t.compile_fail("tests/trybuild/stack_empty_row.rs");
    t.compile_fail("tests/trybuild/stack_empty_col.rs");
}
