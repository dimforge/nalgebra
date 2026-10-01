use nalgebra_sparse::factorization::LeftLookingLUFactorization;
use nalgebra_sparse::CscBuilder;

#[test]
fn test_basic_lu_factorization() {
    let n = 5;
    let mut a = CscBuilder::new(n, n);
    for i in 0..n {
        assert!(a.insert(i, i, 1.).is_ok());
    }
    // construct an identity matrix as a basic test
    let a = a.build();

    let lu_fact = LeftLookingLUFactorization::new(&a);

    assert_eq!(lu_fact.u(), a);
}

#[test]
fn test_basic_lu_factorization_with_one_more_entry() {
    let n = 3;
    let mut a = CscBuilder::new(n, n);
    for i in 0..n {
        assert!(a.insert(i, i, if i == 0 { 0.5 } else { 1. }).is_ok());
        if i == 0 {
            assert!(a.insert(1, 0, 1.).is_ok());
        }
    }
    // construct an identity matrix as a basic test
    let a = a.build();

    let lu_fact = LeftLookingLUFactorization::new(&a);

    let mut ground_truth = CscBuilder::new(n, n);
    for i in 0..n {
        assert!(ground_truth
            .insert(i, i, if i == 0 { 0.5 } else { 1. })
            .is_ok());
        if i == 0 {
            assert!(ground_truth.insert(1, 0, 2.).is_ok());
        }
    }
    let gt = ground_truth.build();

    assert_eq!(lu_fact.lu(), &gt);
}
