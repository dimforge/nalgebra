use nalgebra_macros::matrix;
use nalgebra::Matrix3x2;

#[test]
fn matrix_basic_construction() {
    let matrix: Matrix3x2<_> = matrix![ 1, 2;
                                        3, 4;
                                        5, 6 ];
    assert_eq!(matrix, Matrix3x2::new(1, 2, 3, 4, 5, 6));
}