// Regression tests for issue #1520: Miri-detected UB in decomposition methods
// on SMatrix (ArrayStorage). These tests exercise the axcpy_uninit code path
// through stack-allocated matrices to verify the Stacked Borrows fix.
//
// All tests use small SMatrix inputs (2x2 or 3x3) so they complete quickly
// under Miri. Run with: cargo +nightly miri test linalg::miri_smatrix

use na::Matrix2;

#[test]
fn smatrix_cholesky_issue_1520() {
    // Exact reproducer from issue #1520.
    let mat = Matrix2::<f32>::identity();
    let chol = mat.cholesky().expect("Cholesky should succeed on identity");
    let l = chol.l();
    assert_relative_eq!(l * l.transpose(), mat, epsilon = 1.0e-5);
}

#[test]
fn smatrix_lu() {
    let mat = Matrix2::new(1.0f32, 2.0, 3.0, 4.0);
    let lu = mat.lu();
    let (p, l, u) = lu.unpack();
    let mut recomp = l * u;
    p.inv_permute_rows(&mut recomp);
    assert_relative_eq!(mat, recomp, epsilon = 1.0e-5);
}

#[test]
fn smatrix_full_piv_lu() {
    let mat = Matrix2::new(1.0f32, 2.0, 3.0, 4.0);
    let lu = mat.full_piv_lu();
    let (p, l, u, q) = lu.unpack();
    let mut recomp = l * u;
    p.inv_permute_rows(&mut recomp);
    q.inv_permute_columns(&mut recomp);
    assert_relative_eq!(mat, recomp, epsilon = 1.0e-5);
}

#[test]
fn smatrix_qr() {
    let mat = Matrix2::new(1.0f32, 2.0, 3.0, 4.0);
    let qr = mat.qr();
    let (q, r) = (qr.q(), qr.r());
    assert_relative_eq!(mat, q * r, epsilon = 1.0e-5);
}

#[test]
fn smatrix_col_piv_qr() {
    let mat = Matrix2::new(1.0f32, 2.0, 3.0, 4.0);
    let col_piv_qr = mat.col_piv_qr();
    let (q, r, p) = col_piv_qr.unpack();
    let mut qr = q * r;
    p.inv_permute_columns(&mut qr);
    assert_relative_eq!(mat, qr, epsilon = 1.0e-5);
}

#[test]
fn smatrix_hessenberg() {
    let mat = Matrix2::new(1.0f32, 2.0, 3.0, 4.0);
    let hess = mat.hessenberg();
    let (p, h) = hess.unpack();
    assert_relative_eq!(mat, &p * h * p.transpose(), epsilon = 1.0e-5);
}

#[test]
fn smatrix_schur() {
    let mat = Matrix2::new(1.0f32, 2.0, 3.0, 4.0);
    let schur = mat.schur();
    let (vecs, vals) = schur.unpack();
    assert_relative_eq!(mat, vecs * vals * vecs.transpose(), epsilon = 1.0e-5);
}

#[test]
fn smatrix_symmetric_tridiagonal() {
    // Must be symmetric.
    let mat = Matrix2::new(2.0f32, 1.0, 1.0, 3.0);
    let tri = mat.symmetric_tridiagonalize();
    let recomp = tri.recompose();
    assert_relative_eq!(mat.lower_triangle(), recomp.lower_triangle(), epsilon = 1.0e-5);
}

#[test]
fn smatrix_symmetric_eigen() {
    // Must be symmetric.
    let mat = Matrix2::new(2.0f32, 1.0, 1.0, 3.0);
    let eig = mat.symmetric_eigen();
    let recomp = eig.recompose();
    assert_relative_eq!(mat.lower_triangle(), recomp.lower_triangle(), epsilon = 1.0e-5);
}

#[test]
fn smatrix_bidiagonal() {
    let mat = Matrix2::new(1.0f32, 2.0, 3.0, 4.0);
    let bidiag = mat.bidiagonalize();
    let (u, d, v_t) = bidiag.unpack();
    assert_relative_eq!(mat, &u * d * &v_t, epsilon = 1.0e-5);
}

#[test]
fn smatrix_svd() {
    let mat = Matrix2::new(1.0f32, 2.0, 3.0, 4.0);
    let svd = mat.svd(true, true);
    let recomp = svd.recompose().expect("SVD recompose with u and v_t");
    assert_relative_eq!(mat, recomp, epsilon = 1.0e-5);
}
