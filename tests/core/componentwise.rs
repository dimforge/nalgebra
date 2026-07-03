use num::float::FloatCore;
use na::{DMatrix, Matrix2, Matrix2x3, Matrix3};

#[test]
fn abs() {
    let a = Matrix2::new(0.0, 1.0, -2.0, -3.0);
    let b = Matrix3::new(1.0, 2.0, 3.0, -2.0, 5.0, -6.0, 7.0, 8.0, 9.0);

    let c = Matrix2::new(0.0, 0.0, 0.0, 0.0);
    let d = Matrix3::new(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);

    assert_eq!(a.abs(), Matrix2::new(0.0, 1.0, 2.0, 3.0));
    assert_eq!(b.abs(), Matrix3::new(1.0, 2.0, 3.0, 2.0, 5.0, 6.0, 7.0, 8.0, 9.0));

    assert_eq!(c.abs(), c);
    assert_eq!(d.abs(), d);
}

#[test]
fn component_pow() {
    let a = Matrix2::new(0.0, 1.0, -2.0, -3.0);
    let b = Matrix3::new(1.0, 2.0, 3.0, -2.0, 5.0, 6.0, 7.0, 8.0, 9.0);

    let c = Matrix2::new(0.0, 0.0, 0.0, 0.0);
    let d = Matrix3::new(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);

    assert_eq!(a.component_pow(2), Matrix2::new(0.0, 1.0, 4.0, 9.0));
    assert_eq!(b.component_pow(2), Matrix3::new(1.0, 4.0, 9.0, 4.0, 25.0, 36.0, 49.0, 64.0, 81.0));

    assert_eq!(c.component_pow(3), Matrix2::new(0.0, 0.0, 0.0, 0.0));
    assert_eq!(d.component_pow(3), Matrix3::new(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0));
}

#[test]
fn component_log10_2x3_matrix() {
    let a = Matrix2x3::new(1.0, 10.0, 100.0, 0.1, 0.01, 1000.0);
    assert_eq!(a.component_log10(), Matrix2x3::new(0.0, 1.0, 2.0, -1.0, -2.0, 3.0));
}

#[test]
fn component_log10_2x3_matrix_strange() {
    let a = Matrix2x3::new(f64::INFINITY, f64::NEG_INFINITY, f64::NAN, f64::MIN, f64::MAX, f64::EPSILON);
    let result = a.component_log10();

    assert!(result[(0, 0)].is_infinite());
    assert!(result[(0, 1)].is_nan());
    assert!(result[(0, 2)].is_nan());
    assert!(result[(1, 0)].is_nan());
    assert_eq!(result[(1, 1)], f64::MAX.log10());
    assert_eq!(result[(1, 2)], f64::EPSILON.log10());
}

#[test]
fn component_log10_d_matrix() {
    let a = DMatrix::from_row_slice(2, 2, &[1.0, 10.0, 0.1, 0.01]);
    assert_eq!(a.component_log10(), DMatrix::from_row_slice(2, 2, &[0.0, 1.0, -1.0, -2.0]));
}

#[test]
fn component_log10_d_matrix_strange() {
    let a = DMatrix::from_row_slice(2, 3, &[f64::INFINITY, f64::NEG_INFINITY, f64::NAN, f64::MIN, f64::MAX, f64::EPSILON]);
    let result = a.component_log10();

    assert!(result[(0, 0)].is_infinite());
    assert!(result[(0, 1)].is_nan());
    assert!(result[(0, 2)].is_nan());
    assert!(result[(1, 0)].is_nan());
    assert_eq!(result[(1, 1)], f64::MAX.log10());
    assert_eq!(result[(1, 2)], f64::EPSILON.log10());
}
