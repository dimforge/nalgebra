use nalgebra_macros::matrix;
use nalgebra::{SMatrix, Matrix3x2, Matrix1x2, Matrix1x3, Matrix1x4, Matrix2x1, Matrix2, Matrix2x3, Matrix2x4, Matrix3x1, Matrix3, Matrix3x4, Matrix4x1, Matrix4x2, Matrix4x3, Matrix4};

#[test]
fn matrix_small_dims_exhaustive() {
    // 0x0
    assert_eq!(matrix![], SMatrix::<i32, 0, 0>::zeros());

    // 1xN
    assert_eq!(matrix![1], SMatrix::<i32, 1, 1>::new(1));
    assert_eq!(matrix![1, 2], Matrix1x2::new(1, 2));
    assert_eq!(matrix![1, 2, 3], Matrix1x3::new(1, 2, 3));
    assert_eq!(matrix![1, 2, 3, 4], Matrix1x4::new(1, 2, 3, 4));

    // 2xN
    assert_eq!(matrix![1; 2], Matrix2x1::new(1, 2));
    assert_eq!(matrix![1, 2; 3, 4], Matrix2::new(1, 2, 3, 4));
    assert_eq!(matrix![1, 2, 3; 4, 5, 6], Matrix2x3::new(1, 2, 3, 4, 5, 6));
    assert_eq!(matrix![1, 2, 3, 4; 5, 6, 7, 8], Matrix2x4::new(1, 2, 3, 4, 5, 6, 7, 8));

    // 3xN
    assert_eq!(matrix![1; 2; 3], Matrix3x1::new(1, 2, 3));
    assert_eq!(matrix![1, 2; 3, 4; 5, 6], Matrix3x2::new(1, 2, 3, 4, 5, 6));
    assert_eq!(matrix![1, 2, 3; 4, 5, 6; 7, 8, 9], Matrix3::new(1, 2, 3, 4, 5, 6, 7, 8, 9));
    assert_eq!(matrix![1, 2, 3, 4; 5, 6, 7, 8; 9, 10, 11, 12],
               Matrix3x4::new(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12));

    // 4xN
    assert_eq!(matrix![1; 2; 3; 4], Matrix4x1::new(1, 2, 3, 4));
    assert_eq!(matrix![1, 2; 3, 4; 5, 6; 7, 8], Matrix4x2::new(1, 2, 3, 4, 5, 6, 7, 8));
    assert_eq!(matrix![1, 2, 3; 4, 5, 6; 7, 8, 9; 10, 11, 12],
               Matrix4x3::new(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12));
    assert_eq!(matrix![1, 2, 3, 4; 5, 6, 7, 8; 9, 10, 11, 12; 13, 14, 15, 16],
               Matrix4::new(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16));
}