use std::num::{Real, abs};
use std::rand::random;
use std::cmp::ApproxEq;
use na::{DMat, DVec};
use na::Indexable; // FIXME: get rid of that
use na;

macro_rules! test_inv_mat_impl(
  ($t: ty) => (
    do 10000.times {
      let randmat : $t = random();

      assert!((na::inv(&randmat).unwrap() * randmat).approx_eq(&na::one()));
    }
  );
)

macro_rules! test_transpose_mat_impl(
  ($t: ty) => (
    do 10000.times {
      let randmat : $t = random();

      assert!(na::transpose(&na::transpose(&randmat)) == randmat);
    }
  );
)

#[test]
fn test_transpose_mat1() {
    test_transpose_mat_impl!(na::Mat1<f64>);
}

#[test]
fn test_transpose_mat2() {
    test_transpose_mat_impl!(na::Mat2<f64>);
}

#[test]
fn test_transpose_mat3() {
    test_transpose_mat_impl!(na::Mat3<f64>);
}

#[test]
fn test_transpose_mat4() {
    test_transpose_mat_impl!(na::Mat4<f64>);
}

#[test]
fn test_transpose_mat5() {
    test_transpose_mat_impl!(na::Mat5<f64>);
}

#[test]
fn test_transpose_mat6() {
    test_transpose_mat_impl!(na::Mat6<f64>);
}

#[test]
fn test_inv_mat1() {
    test_inv_mat_impl!(na::Mat1<f64>);
}

#[test]
fn test_inv_mat2() {
    test_inv_mat_impl!(na::Mat2<f64>);
}

#[test]
fn test_inv_mat3() {
    test_inv_mat_impl!(na::Mat3<f64>);
}

#[test]
fn test_inv_mat4() {
    test_inv_mat_impl!(na::Mat4<f64>);
}

#[test]
fn test_inv_mat5() {
    test_inv_mat_impl!(na::Mat5<f64>);
}

#[test]
fn test_inv_mat6() {
    test_inv_mat_impl!(na::Mat6<f64>);
}

#[test]
fn test_rotation2() {
    do 10000.times {
        let randmat: na::Rot2<f64> = na::one();
        let ang    = na::vec1(abs::<f64>(random()) % Real::pi());

        assert!(na::rotation(&na::append_rotation(&randmat, &ang)).approx_eq(&ang));
    }
}

#[test]
fn test_index_mat2() {
  let mat: na::Mat2<f64> = random();

  assert!(mat.at((0, 1)) == na::transpose(&mat).at((1, 0)));
}

#[test]
fn test_inv_rotation3() {
    do 10000.times {
        let randmat: na::Rot3<f64> = na::one();
        let dir:     na::Vec3<f64> = random();
        let ang            = na::normalize(&dir) * (abs::<f64>(random()) % Real::pi());
        let rot            = na::append_rotation(&randmat, &ang);

        assert!((na::transpose(&rot) * rot).approx_eq(&na::one()));
    }
}

#[test]
fn test_mean_dmat() {
    let mat = DMat::from_vec(
        3,
        3,
        [
            1.0f64, 2.0, 3.0,
            4.0f64, 5.0, 6.0,
            7.0f64, 8.0, 9.0,
        ]
    );

    assert!(na::mean(&mat).approx_eq(&DVec::from_vec(3, [4.0f64, 5.0, 6.0])));
}

#[test]
fn test_cov_dmat() {
    let mat = DMat::from_vec(
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

    let expected = DMat::from_vec(
        3,
        3,
        [
            0.025,   0.0075,  0.00175,
            0.0075,  0.007,   0.00135,
            0.00175, 0.00135, 0.00043
        ]
    );

    assert!(na::cov(&mat).approx_eq(&expected));
}
