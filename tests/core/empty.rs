use na::{DMatrix, DVector};

#[test]
fn empty_mul() {
    // Issue #644
    let m = DMatrix::<f32>::zeros(8, 0);
    let v = DVector::<f32>::zeros(0);
    assert_eq!(m * v, DVector::zeros(8));
}