use nalgebra::DVector;
#[test]
fn test_variance_new() {
    let long_repeating_vector = DVector::repeat(10_000, 100000000.0);
    assert_eq!(long_repeating_vector.variance(), 0.0);

    let short_vec = DVector::from_vec(vec![1., 2., 3.]);

    assert_eq!(short_vec.variance(), 2.0 / 3.0);
}
