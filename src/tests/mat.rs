#[test]
use std::vec;
#[test]
use std::num::{Real, Zero, One, abs};
#[test]
use std::rand::{random};
#[test]
use std::cmp::ApproxEq;
#[test]
use traits::inv::Inv;
#[test]
use traits::rotation::Rotation;
#[test]
use traits::dim::d7;
#[test]
use dim1::vec1::Vec1;
#[test]
use dim1::mat1::Mat1;
#[test]
use dim2::mat2::Mat2;
#[test]
use dim3::mat3::Mat3;
#[test]
use ndim::nmat::NMat;
#[test]
use adaptors::rotmat::Rotmat;
#[test]
use traits::flatten::Flatten;

macro_rules! test_inv_mat_impl(
  ($t: ty) => (
    for 10000.times
    {
      let randmat : $t = random();

      assert!((randmat.inverse() * randmat).approx_eq(&One::one()));
    }
  );
)

macro_rules! test_flatten_impl(
  ($t: ty, $n: ty) => (
    for 10000.times
    {
      let v:     $t    = random();
      let mut l: ~[$n] = vec::from_elem(42 + Flatten::flat_size::<$n, $t>(), Zero::zero::<$n>());

      v.flatten_to(l, 42);

      assert!(Flatten::from_flattened::<$n, $t>(v.flatten(), 0) == v);
      assert!(Flatten::from_flattened::<$n, $t>(l, 42) == v);
    }
  )
)

#[test]
fn test_inv_mat1()
{ test_inv_mat_impl!(Mat1<f64>); }

#[test]
fn test_inv_mat2()
{ test_inv_mat_impl!(Mat2<f64>); }

#[test]
fn test_inv_mat3()
{ test_inv_mat_impl!(Mat3<f64>); }

// FIXME: ICE
// #[test]
// fn test_inv_nmat()
// { test_inv_mat_impl!(NMat<d7, f64>); }

#[test]
fn test_flatten_mat1()
{ test_flatten_impl!(Mat1<f64>, f64); }

#[test]
fn test_flatten_mat2()
{ test_flatten_impl!(Mat2<f64>, f64); }

#[test]
fn test_flatten_mat3()
{ test_flatten_impl!(Mat3<f64>, f64); }

#[test]
fn test_flatten_nmat()
{ test_flatten_impl!(NMat<d7, f64>, f64); }

#[test]
fn test_rotation2()
{
  for 10000.times
  {
    let randmat = One::one::<Rotmat<Mat2<f64>>>();
    let ang     = &Vec1::new(abs::<f64>(random()) % Real::pi());

    assert!(randmat.rotated(ang).rotation().approx_eq(ang));
  }
}
