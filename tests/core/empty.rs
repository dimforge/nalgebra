use na::{DMatrix, DVector};

#[test]
fn empty_matrix_mul_vector() {
    // Issue #644
    let m = DMatrix::<f32>::zeros(8, 0);
    let v = DVector::<f32>::zeros(0);
    assert_eq!(m * v, DVector::zeros(8));
}

#[test]
fn empty_matrix_mul_matrix() {
    let m1 = DMatrix::<f32>::zeros(3, 0);
    let m2 = DMatrix::<f32>::zeros(0, 4);
    assert_eq!(m1 * m2, DMatrix::zeros(3, 4));
}

#[test]
fn empty_matrix_tr_mul_vector() {
    let m = DMatrix::<f32>::zeros(0, 5);
    let v = DVector::<f32>::zeros(0);
    assert_eq!(m.tr_mul(&v), DVector::zeros(5));
}

#[test]
fn empty_matrix_tr_mul_matrix() {
    let m1 = DMatrix::<f32>::zeros(0, 3);
    let m2 = DMatrix::<f32>::zeros(0, 4);
    assert_eq!(m1.tr_mul(&m2), DMatrix::zeros(3, 4));
}