#[test]
use std::num::{Real, One, abs};
#[test]
use std::rand::random;
#[test]
use std::cmp::ApproxEq;
#[test]
use traits::norm::Norm;
#[test]
use traits::scalar_op::ScalarMul;
#[test]
use traits::inv::Inv;
#[test]
use traits::rotation::{Rotation, Rotatable};
#[test]
use traits::indexable::Indexable;
#[test]
use traits::transpose::Transpose;
#[test]
use vec::{Vec1, Vec3};
#[test]
use mat::{Mat1, Mat2, Mat3, Mat4, Mat5, Mat6};
#[test]
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
    let randmat = One::one::<Rotmat<Mat2<f64>>>();
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
    let randmat        = One::one::<Rotmat<Mat3<f64>>>();
    let dir: Vec3<f64> = random();
    let ang            = &dir.normalized().scalar_mul(&(abs::<f64>(random()) % Real::pi()));
    let rot            = randmat.rotated(ang);

    assert!((rot.transposed() * rot).approx_eq(&One::one()));
  }
}
