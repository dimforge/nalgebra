#[cfg(test)]
use crate::DVector;
#[test]
fn test_variance_new() {
    let v = DVector::repeat(10_000, 100000000.1234);
    assert_eq!(v.variance(), 0.0)
}
