use std::num::{Real, One, abs};
use std::rand::random;
use std::cmp::ApproxEq;
use traits::inv::Inv;
use traits::rotation::Rotation;
use traits::indexable::Indexable;
use traits::transpose::Transpose;
use traits::norm::Norm;
use vec::{Vec1, Vec3};
use mat::{Mat1, Mat2, Mat3, Mat4, Mat5, Mat6};
use adaptors::rotmat::Rotmat;

macro_rules! test_inv_mat_impl(
  ($t: ty) => (
    do 10000.times {
      let randmat : $t = random();

      assert!((randmat.inverse().unwrap() * randmat).approx_eq(&One::one()));
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
    let randmat: Rotmat<Mat2<f64>> = One::one();
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
    let randmat: Rotmat<Mat3<f64>> = One::one();
    let dir: Vec3<f64> = random();
    let ang            = &(dir.normalized() * (abs::<f64>(random()) % Real::pi()));
    let rot            = randmat.rotated(ang);

    assert!((rot.transposed() * rot).approx_eq(&One::one()));
  }
}
