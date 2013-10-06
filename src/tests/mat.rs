use std::num::{Real, One, abs};
use std::rand::random;
use std::cmp::ApproxEq;
use na::*;

macro_rules! test_inv_mat_impl(
  ($t: ty) => (
    do 10000.times {
      let randmat : $t = random();

      assert!((randmat.inverted().unwrap() * randmat).approx_eq(&One::one()));
    }
  );
)

macro_rules! test_transpose_mat_impl(
  ($t: ty) => (
    do 10000.times {
      let randmat : $t = random();

      assert!(randmat.transposed().transposed().eq(&randmat));
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
    do 10000.times {
        let randmat: Rot2<f64> = One::one();
        let ang     = &Vec1::new(abs::<f64>(random()) % Real::pi());

        assert!(randmat.rotated(ang).rotation().approx_eq(ang));
    }
}

#[test]
fn test_index_mat2() {
  let mat: Mat2<f64> = random();

  assert!(mat.at((0, 1)) == mat.transposed().at((1, 0)));
}

#[test]
fn test_inv_rotation3() {
    do 10000.times {
        let randmat: Rot3<f64> = One::one();
        let dir: Vec3<f64> = random();
        let ang            = &(dir.normalized() * (abs::<f64>(random()) % Real::pi()));
        let rot            = randmat.rotated(ang);

        assert!((rot.transposed() * rot).approx_eq(&One::one()));
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

    assert!(mat.mean().approx_eq(&DVec::from_vec(3, [4.0f64, 5.0, 6.0])));
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

    assert!(mat.cov().approx_eq(&expected));
}
