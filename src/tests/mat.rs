use std::num::{Float, abs};
use std::rand::random;
use na::{Vec1, Vec3, Mat1, Mat2, Mat3, Mat4, Mat5, Mat6, Rot3, DMat, DVec, Indexable};
use na;
use na::decomp_qr;
use std::cmp::{min, max};

macro_rules! test_inv_mat_impl(
  ($t: ty) => (
    for _ in range(0, 10000) {
      let randmat : $t = random();

      assert!(na::approx_eq(&(na::inv(&randmat).unwrap() * randmat), &na::one()));
    }
  );
)

macro_rules! test_transpose_mat_impl(
  ($t: ty) => (
    for _ in range(0, 10000) {
      let randmat : $t = random();

      assert!(na::transpose(&na::transpose(&randmat)) == randmat);
    }
  );
)

macro_rules! test_decomp_qr_impl(
  ($t: ty) => (
    for _ in range(0, 10000) {
      let randmat : $t = random();

      let (q, r) = decomp_qr(&randmat);
      let recomp = q * r;

      assert!(na::approx_eq(&randmat,  &recomp));
    }
  );
)

#[test]
fn test_transpose_mat1() {
    test_transpose_mat_impl!(Mat1<f64>);
}

#[test]
fn test_transpose_mat2() {
    test_transpose_mat_impl!(Mat2<f64>);
}

#[test]
fn test_transpose_mat3() {
    test_transpose_mat_impl!(Mat3<f64>);
}

#[test]
fn test_transpose_mat4() {
    test_transpose_mat_impl!(Mat4<f64>);
}

#[test]
fn test_transpose_mat5() {
    test_transpose_mat_impl!(Mat5<f64>);
}

#[test]
fn test_transpose_mat6() {
    test_transpose_mat_impl!(Mat6<f64>);
}

#[test]
fn test_inv_mat1() {
    test_inv_mat_impl!(Mat1<f64>);
}

#[test]
fn test_inv_mat2() {
    test_inv_mat_impl!(Mat2<f64>);
}

#[test]
fn test_inv_mat3() {
    test_inv_mat_impl!(Mat3<f64>);
}

#[test]
fn test_inv_mat4() {
    test_inv_mat_impl!(Mat4<f64>);
}

#[test]
fn test_inv_mat5() {
    test_inv_mat_impl!(Mat5<f64>);
}

#[test]
fn test_inv_mat6() {
    test_inv_mat_impl!(Mat6<f64>);
}

#[test]
fn test_rotation2() {
    for _ in range(0, 10000) {
        let randmat: na::Rot2<f64> = na::one();
        let ang    = Vec1::new(abs::<f64>(random()) % Float::pi());

        assert!(na::approx_eq(&na::rotation(&na::append_rotation(&randmat, &ang)), &ang));
    }
}

#[test]
fn test_index_mat2() {
  let mat: Mat2<f64> = random();

  assert!(mat.at((0, 1)) == na::transpose(&mat).at((1, 0)));
}

#[test]
fn test_inv_rotation3() {
    for _ in range(0, 10000) {
        let randmat: Rot3<f64> = na::one();
        let dir:     Vec3<f64> = random();
        let ang            = na::normalize(&dir) * (abs::<f64>(random()) % Float::pi());
        let rot            = na::append_rotation(&randmat, &ang);

        assert!(na::approx_eq(&(na::transpose(&rot) * rot), &na::one()));
    }
}

#[test]
fn test_mean_dmat() {
    let mat = DMat::from_row_vec(
        3,
        3,
        [
            1.0f64, 2.0, 3.0,
            4.0f64, 5.0, 6.0,
            7.0f64, 8.0, 9.0,
        ]
    );

    assert!(na::approx_eq(&na::mean(&mat), &DVec::from_vec(3, [4.0f64, 5.0, 6.0])));
}

#[test]
fn test_cov_dmat() {
    let mat = DMat::from_row_vec(
        5,
        3,
        [
            4.0, 2.0, 0.60,
            4.2, 2.1, 0.59,
            3.9, 2.0, 0.58,
            4.3, 2.1, 0.62,
            4.1, 2.2, 0.63
        ]
    );

    let expected = DMat::from_row_vec(
        3,
        3,
        [
            0.025,   0.0075,  0.00175,
            0.0075,  0.007,   0.00135,
            0.00175, 0.00135, 0.00043
        ]
    );

    assert!(na::approx_eq(&na::cov(&mat), &expected));
}

#[test]
fn test_transpose_dmat() {
    let mat = DMat::from_row_vec(
        8,
        4,
        [
            1,  2,  3,  4,
            5,  6,  7,  8,
            9,  10, 11, 12,
            13, 14, 15, 16,
            17, 18, 19, 20,
            21, 22, 23, 24,
            25, 26, 27, 28,
            29, 30, 31, 32
        ]
    );

    assert!(na::transpose(&na::transpose(&mat)) == mat);
}

#[test]
fn test_dmat_from_vec() {
    let mat1 = DMat::from_row_vec(
        8,
        4,
        [
            1,  2,  3,  4,
            5,  6,  7,  8,
            9,  10, 11, 12,
            13, 14, 15, 16,
            17, 18, 19, 20,
            21, 22, 23, 24,
            25, 26, 27, 28,
            29, 30, 31, 32
        ]
    );

    let mat2 = DMat::from_col_vec(
        8,
        4,
        [
            1, 5, 9,  13, 17, 21, 25, 29, 
            2, 6, 10, 14, 18, 22, 26, 30,
            3, 7, 11, 15, 19, 23, 27, 31, 
            4, 8, 12, 16, 20, 24, 28, 32
        ]
    );

    println!("mat1: {:?}, mat2: {:?}", mat1, mat2);

    assert!(mat1 == mat2);
}

#[test]
fn test_decomp_qr() {
    for _ in range(0, 10) {
        let dim1: uint = random();
        let dim2: uint = random();
        let rows = min(40, max(dim1, dim2));
        let cols = min(40, min(dim1, dim2));
        let randmat: DMat<f64> = DMat::new_random(rows, cols);
        let (q, r) = decomp_qr(&randmat);
        let recomp = q * r;

        assert!(na::approx_eq(&randmat,  &recomp));
    }
}

#[test]
fn test_decomp_qr_mat1() {
    test_decomp_qr_impl!(Mat1<f64>);
}

#[test]
fn test_decomp_qr_mat2() {
    test_decomp_qr_impl!(Mat2<f64>);
}

#[test]
fn test_decomp_qr_mat3() {
    test_decomp_qr_impl!(Mat3<f64>);
}

#[test]
fn test_decomp_qr_mat4() {
    test_decomp_qr_impl!(Mat4<f64>);
}

#[test]
fn test_decomp_qr_mat5() {
    test_decomp_qr_impl!(Mat5<f64>);
}

#[test]
fn test_decomp_qr_mat6() {
    test_decomp_qr_impl!(Mat6<f64>);
}
