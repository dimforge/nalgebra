#[test]
use core::num::{One, abs};
#[test]
use core::rand::{random};
#[test]
use core::cmp::ApproxEq;
#[test]
use traits::inv::Inv;
#[test]
use traits::rotation::Rotation;
#[test]
use dim1::vec1::vec1;
#[test]
use dim1::mat1::Mat1;
#[test]
use dim2::mat2::Mat2;
#[test]
use dim3::mat3::Mat3;
#[test]
use adaptors::rotmat::Rotmat;

macro_rules! test_inv_mat_impl(
  ($t:ty) => (
    for uint::range(0u, 10000u) |_|
    {
      let randmat : $t = random();

      assert!((randmat.inverse() * randmat).approx_eq(&One::one()));
    }
  );
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

// FIXME: this one fails with an ICE: node_id_to_type: no type for node [...]
// #[test]
// fn test_inv_nmat()
// { test_inv_mat_impl!(NMat<d7, f64>); }

#[test]
fn test_rotation2()
{
  for uint::range(0u, 10000u) |_|
  {
    let randmat = One::one::<Rotmat<Mat2<f64>>>();
    let ang     = &vec1(abs::<f64>(random()) % f64::consts::pi);

    assert!(randmat.rotated(ang).rotation().approx_eq(ang));
  }
}
