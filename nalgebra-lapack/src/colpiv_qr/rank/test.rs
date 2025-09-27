#[test]
fn test_calc_max_diag() {
    let a = nalgebra::matrix!
    [
        1.,2.,3.;
        4.,5.,6.;
        7.,8.,-9.;
    ];
    assert_eq!(super::calculate_max_abs_diag(&a), 9.);

    let a = nalgebra::dmatrix!
    [
        1.,2.,3.;
        4.,5.,6.;
        7.,8.,-9.;
        11.,12.,13.;
    ];
    assert_eq!(super::calculate_max_abs_diag(&a), 9.);

    let a = nalgebra::matrix!
    [
        1.,2.,3.,11.;
        4.,5.,6.,12.;
        7.,8.,-9.,13.;
    ];
    assert_eq!(super::calculate_max_abs_diag(&a), 9.);
}
