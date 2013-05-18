#[test]
use core::num::{Zero, One};
#[test]
use core::rand::{random};
#[test]
use core::vec::{all, all2};
#[test]
use std::cmp::FuzzyEq;
#[test]
use dim3::vec3::Vec3;
#[test]
use dim2::vec2::Vec2;
#[test]
use dim1::vec1::Vec1;
#[test]
use ndim::nvec::NVec;
#[test]
use traits::dim::d7;
#[test]
use traits::basis::Basis;

#[test]
fn test_cross_vec3()
{
  for uint::range(0u, 10000u) |_|
  {
    let v1 : Vec3<f64> = random();
    let v2 : Vec3<f64> = random();
    let v3 : Vec3<f64> = v1.cross(&v2);

    assert!(v3.dot(&v2).fuzzy_eq(&Zero::zero()));
    assert!(v3.dot(&v1).fuzzy_eq(&Zero::zero()));
  }
}

#[test]
fn test_dot_nvec()
{
  for uint::range(0u, 10000u) |_|
  {
    let v1 : NVec<d7, f64> = random();
    let v2 : NVec<d7, f64> = random();

    assert!(v1.dot(&v2).fuzzy_eq(&v2.dot(&v1)));
  }
}

#[test]
fn test_commut_dot_vec3()
{
  for uint::range(0u, 10000u) |_|
  {
    let v1 : Vec3<f64> = random();
    let v2 : Vec3<f64> = random();

    assert!(v1.dot(&v2).fuzzy_eq(&v2.dot(&v1)));
  }
}

#[test]
fn test_commut_dot_vec2()
{
  for uint::range(0u, 10000u) |_|
  {
    let v1 : Vec2<f64> = random();
    let v2 : Vec2<f64> = random();

    assert!(v1.dot(&v2).fuzzy_eq(&v2.dot(&v1)));
  }
}

#[test]
fn test_commut_dot_vec1()
{
  for uint::range(0u, 10000u) |_|
  {
    let v1 : Vec1<f64> = random();
    let v2 : Vec1<f64> = random();

    assert!(v1.dot(&v2).fuzzy_eq(&v2.dot(&v1)));
  }
}

#[test]
fn test_basis_vec1()
{
  for uint::range(0u, 10000u) |_|
  {
    let basis = Basis::canonical_basis::<Vec1<f64>>();

    // check vectors form an ortogonal basis
    assert!(all2(basis, basis, |e1, e2| e1 == e2 || e1.dot(e2).fuzzy_eq(&Zero::zero())));
    // check vectors form an orthonormal basis
    assert!(all(basis, |e| e.norm().fuzzy_eq(&One::one())));
  }
}

#[test]
fn test_basis_vec2()
{
  for uint::range(0u, 10000u) |_|
  {
    let basis = Basis::canonical_basis::<Vec2<f64>>();

    // check vectors form an ortogonal basis
    assert!(all2(basis, basis, |e1, e2| e1 == e2 || e1.dot(e2).fuzzy_eq(&Zero::zero())));
    // check vectors form an orthonormal basis
    assert!(all(basis, |e| e.norm().fuzzy_eq(&One::one())));
  }
}

#[test]
fn test_basis_vec3()
{
  for uint::range(0u, 10000u) |_|
  {
    let basis = Basis::canonical_basis::<Vec3<f64>>();

    // check vectors form an ortogonal basis
    assert!(all2(basis, basis, |e1, e2| e1 == e2 || e1.dot(e2).fuzzy_eq(&Zero::zero())));
    // check vectors form an orthonormal basis
    assert!(all(basis, |e| e.norm().fuzzy_eq(&One::one())));
  }
}

#[test]
fn test_basis_nvec()
{
  for uint::range(0u, 10000u) |_|
  {
    let basis = Basis::canonical_basis::<NVec<d7, f64>>();

    // check vectors form an ortogonal basis
    assert!(all2(basis, basis, |e1, e2| e1 == e2 || e1.dot(e2).fuzzy_eq(&Zero::zero())));
    // check vectors form an orthonormal basis
    assert!(all(basis, |e| e.norm().fuzzy_eq(&One::one())));
  }
}

#[test]
fn test_subspace_basis_vec1()
{
  for uint::range(0u, 10000u) |_|
  {
    let v : Vec1<f64> = random();
    let v1 = v.normalized();
    let subbasis = v1.orthogonal_subspace_basis();

    // check vectors are orthogonal to v1
    assert!(all(subbasis, |e| v1.dot(e).fuzzy_eq(&Zero::zero())));
    // check vectors form an ortogonal basis
    assert!(all2(subbasis, subbasis, |e1, e2| e1 == e2 || e1.dot(e2).fuzzy_eq(&Zero::zero())));
    // check vectors form an orthonormal basis
    assert!(all(subbasis, |e| e.norm().fuzzy_eq(&One::one())));
  }
}

#[test]
fn test_subspace_basis_vec2()
{
  for uint::range(0u, 10000u) |_|
  {
    let v : Vec2<f64> = random();
    let v1 = v.normalized();
    let subbasis = v1.orthogonal_subspace_basis();

    // check vectors are orthogonal to v1
    assert!(all(subbasis, |e| v1.dot(e).fuzzy_eq(&Zero::zero())));
    // check vectors form an ortogonal basis
    assert!(all2(subbasis, subbasis, |e1, e2| e1 == e2 || e1.dot(e2).fuzzy_eq(&Zero::zero())));
    // check vectors form an orthonormal basis
    assert!(all(subbasis, |e| e.norm().fuzzy_eq(&One::one())));
  }
}

#[test]
fn test_subspace_basis_vec3()
{
  for uint::range(0u, 10000u) |_|
  {
    let v : Vec3<f64> = random();
    let v1 = v.normalized();
    let subbasis = v1.orthogonal_subspace_basis();

    // check vectors are orthogonal to v1
    assert!(all(subbasis, |e| v1.dot(e).fuzzy_eq(&Zero::zero())));
    // check vectors form an ortogonal basis
    assert!(all2(subbasis, subbasis, |e1, e2| e1 == e2 || e1.dot(e2).fuzzy_eq(&Zero::zero())));
    // check vectors form an orthonormal basis
    assert!(all(subbasis, |e| e.norm().fuzzy_eq(&One::one())));
  }
}

// ICE
//
// #[test]
// fn test_subspace_basis_vecn()
// {
//   for uint::range(0u, 10000u) |_|
//   {
//     let v : NVec<d7, f64> = random();
//     let v1 = v.normalized();
//     let subbasis = v1.orthogonal_subspace_basis();
// 
//     // check vectors are orthogonal to v1
//     assert!(all(subbasis, |e| v1.dot(e).fuzzy_eq(&Zero::zero())));
//     // check vectors form an ortogonal basis
//     assert!(all2(subbasis, subbasis, |e1, e2| e1 == e2 || e1.dot(e2).fuzzy_eq(&Zero::zero())));
//     // check vectors form an orthonormal basis
//     assert!(all(subbasis, |e| e.norm().fuzzy_eq(&One::one())));
//   }
// }
