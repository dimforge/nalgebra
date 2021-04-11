use nalgebra_macros::matrix;

#[test]
fn basic_usage() {
    matrix![ 1, 3; 4, 5*3];
}