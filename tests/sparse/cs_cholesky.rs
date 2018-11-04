#![cfg_attr(rustfmt, rustfmt_skip)]

use na::{CsMatrix, CsVector, CsCholesky, Cholesky, Matrix5, Vector5};

#[test]
fn cs_cholesky() {
    let mut a = Matrix5::new(
        40.0, 0.0, 0.0, 0.0, 0.0,
        2.0, 60.0, 0.0, 0.0, 0.0,
        1.0, 0.0, 11.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 50.0, 0.0,
        1.0, 0.0, 0.0, 4.0, 10.0
    );
    a.fill_upper_triangle_with_lower_triangle();
    test_cholesky(a);

    let a = Matrix5::from_diagonal(&Vector5::new(40.0, 60.0, 11.0, 50.0, 10.0));
    test_cholesky(a);

    let mut a = Matrix5::new(
        40.0, 0.0, 0.0, 0.0, 0.0,
        2.0, 60.0, 0.0, 0.0, 0.0,
        1.0, 0.0, 11.0, 0.0, 0.0,
        1.0, 0.0, 0.0, 50.0, 0.0,
        0.0, 0.0, 0.0, 4.0, 10.0
    );
    a.fill_upper_triangle_with_lower_triangle();
    test_cholesky(a);

    let mut a = Matrix5::new(
        2.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 2.0, 0.0, 0.0, 0.0,
        1.0, 1.0, 2.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 2.0, 0.0,
        1.0, 1.0, 0.0, 0.0, 2.0
    );
    a.fill_upper_triangle_with_lower_triangle();
    test_cholesky(a);
}


fn test_cholesky(a: Matrix5<f32>) {
    let cs_a: CsMatrix<_, _, _> = a.into();

    let chol_a = Cholesky::new(a).unwrap();
    let chol_cs_a = CsCholesky::new(&cs_a);
    let l = chol_a.l();
    println!("{:?}", chol_cs_a.l());
    let cs_l: Matrix5<_> = chol_cs_a.unwrap_l().unwrap().into();

    println!("{}", l);
    println!("{}", cs_l);

    assert_relative_eq!(l, cs_l);
}
