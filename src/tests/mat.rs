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

// FIXME: this one fails with an ICE: node_id_to_type: no type for node [...]
// #[test]
// fn test_inv_nmat()
// {
//   let randmat : NMat<d7, f64> = random();
// 
//   assert!((randmat.inverse() * randmat).approx_eq(&One::one()));
// }

#[test]
fn test_inv_mat1()
{
  for uint::range(0u, 10000u) |_|
  {
    let randmat : Mat1<f64> = random();

    assert!((randmat.inverse() * randmat).approx_eq(&One::one()));
  }
}

#[test]
fn test_inv_mat2()
{
  for uint::range(0u, 10000u) |_|
  {
    let randmat : Mat2<f64> = random();

    assert!((randmat.inverse() * randmat).approx_eq(&One::one()));
  }
}

#[test]
fn test_inv_mat3()
{
  for uint::range(0u, 10000u) |_|
  {
    let randmat : Mat3<f64> = random();

    assert!((randmat.inverse() * randmat).approx_eq(&One::one()));
  }
}

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
