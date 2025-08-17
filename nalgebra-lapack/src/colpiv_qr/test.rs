use na::OMatrix;

#[test]
fn smoketest_qr_decomposition_for_f32_matrix() {
    let mat: OMatrix<f32, _, _> = nalgebra::matrix!
    [0.,   8.,   1.;
     4.,   5.,   4.;
     9.,   3.,   1.;
     8.,   4.,   9.];

    let _ = super::ColPivQR::new(mat, 1e-10).expect("creating qr decomposition must not fail");
}
