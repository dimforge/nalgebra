use na::{Matrix2, Matrix3};

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