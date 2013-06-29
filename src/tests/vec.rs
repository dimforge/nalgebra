#[test]
use std::iterator::IteratorUtil;
#[test]
use std::num::{Zero, One};
#[test]
use std::rand::{random};
#[test]
use std::cmp::ApproxEq;
#[test]
use vec::{Vec1, Vec2, Vec3, Vec4, Vec5, Vec6};
#[test]
use traits::basis::Basis;
#[test]
use traits::cross::Cross;
#[test]
use traits::dot::Dot;
#[test]
use traits::norm::Norm;
#[test]
use traits::iterable::{Iterable, IterableMut};
#[test]
use traits::scalar_op::{ScalarMul, ScalarDiv, ScalarAdd, ScalarSub};

macro_rules! test_iterator_impl(
  ($t: ty, $n: ty) => (
    for 10000.times
    {
      let v: $t      = random();
      let mut mv: $t = copy v;
      let n: $n      = random();

      let nv: $t = v.iter().transform(|e| e * n).collect();

      for mv.mut_iter().advance |e|
      { *e = *e * n }

      assert!(nv == mv && nv == v.scalar_mul(&n));
    }
  )
)

macro_rules! test_commut_dot_impl(
  ($t: ty) => (
    for 10000.times
    {
      let v1 : $t = random();
      let v2 : $t = random();
    
      assert!(v1.dot(&v2).approx_eq(&v2.dot(&v1)));
    }
  );
)

macro_rules! test_scalar_op_impl(
  ($t: ty, $n: ty) => (
    for 10000.times
    {
      let v1 : $t = random();
      let n  : $n = random();
    
      assert!(v1.scalar_mul(&n).scalar_div(&n).approx_eq(&v1));
      assert!(v1.scalar_div(&n).scalar_mul(&n).approx_eq(&v1));
      assert!(v1.scalar_sub(&n).scalar_add(&n).approx_eq(&v1));
      assert!(v1.scalar_add(&n).scalar_sub(&n).approx_eq(&v1));

      let mut v1 : $t = random();
      let v0 : $t = copy v1;
      let n  : $n = random();

      v1.scalar_mul_inplace(&n);
      v1.scalar_div_inplace(&n);
    
      assert!(v1.approx_eq(&v0));
    }
  );
)

macro_rules! test_basis_impl(
  ($t: ty) => (
    for 10000.times
    {
      let basis = Basis::canonical_basis::<$t>();

      // check vectors form an ortogonal basis
      assert!(
        do basis.iter().zip(basis.iter()).all
        |(e1, e2)| { e1 == e2 || e1.dot(e2).approx_eq(&Zero::zero()) }
      );
      // check vectors form an orthonormal basis
      assert!(basis.iter().all(|e| e.norm().approx_eq(&One::one())));
    }
  );
)

macro_rules! test_subspace_basis_impl(
  ($t: ty) => (
    for 10000.times
    {
      let v : $t   = random();
      let v1       = v.normalized();
      let subbasis = v1.orthogonal_subspace_basis();

      // check vectors are orthogonal to v1
      assert!(subbasis.iter().all(|e| v1.dot(e).approx_eq(&Zero::zero())));
      // check vectors form an ortogonal basis
      assert!(
        do subbasis.iter().zip(subbasis.iter()).all
           |(e1, e2)| { e1 == e2 || e1.dot(e2).approx_eq(&Zero::zero()) }
      );
      // check vectors form an orthonormal basis
      assert!(subbasis.iter().all(|e| e.norm().approx_eq(&One::one())));
    }
  );
)

#[test]
fn test_cross_vec3()
{
  for 10000.times
  {
    let v1 : Vec3<f64> = random();
    let v2 : Vec3<f64> = random();
    let v3 : Vec3<f64> = v1.cross(&v2);

    assert!(v3.dot(&v2).approx_eq(&Zero::zero()));
    assert!(v3.dot(&v1).approx_eq(&Zero::zero()));
  }
}

#[test]
fn test_commut_dot_nvec()
{ test_commut_dot_impl!(Vec6<f64>); }
 
#[test]
fn test_commut_dot_vec3()
{ test_commut_dot_impl!(Vec3<f64>); }

#[test]
fn test_commut_dot_vec2()
{ test_commut_dot_impl!(Vec2<f64>); }

#[test]
fn test_commut_dot_vec1()
{ test_commut_dot_impl!(Vec1<f64>); }

#[test]
fn test_basis_vec1()
{ test_basis_impl!(Vec1<f64>); }

#[test]
fn test_basis_vec2()
{ test_basis_impl!(Vec2<f64>); }

#[test]
fn test_basis_vec3()
{ test_basis_impl!(Vec3<f64>); }

#[test]
fn test_basis_vec4()
{ test_basis_impl!(Vec4<f64>); }

#[test]
fn test_basis_vec5()
{ test_basis_impl!(Vec5<f64>); }

#[test]
fn test_basis_vec6()
{ test_basis_impl!(Vec6<f64>); }

#[test]
fn test_subspace_basis_vec1()
{ test_subspace_basis_impl!(Vec1<f64>); }

#[test]
fn test_subspace_basis_vec2()
{ test_subspace_basis_impl!(Vec2<f64>); }

#[test]
fn test_subspace_basis_vec3()
{ test_subspace_basis_impl!(Vec3<f64>); }

#[test]
fn test_subspace_basis_vec4()
{ test_subspace_basis_impl!(Vec4<f64>); }

#[test]
fn test_subspace_basis_vec5()
{ test_subspace_basis_impl!(Vec5<f64>); }

#[test]
fn test_subspace_basis_vec6()
{ test_subspace_basis_impl!(Vec6<f64>); }

#[test]
fn test_scalar_op_vec1()
{ test_scalar_op_impl!(Vec1<f64>, f64); }

#[test]
fn test_scalar_op_vec2()
{ test_scalar_op_impl!(Vec2<f64>, f64); }
 
#[test]
fn test_scalar_op_vec3()
{ test_scalar_op_impl!(Vec3<f64>, f64); }
 
#[test]
fn test_scalar_op_vec4()
{ test_scalar_op_impl!(Vec4<f64>, f64); }
 
#[test]
fn test_scalar_op_vec5()
{ test_scalar_op_impl!(Vec5<f64>, f64); }
 
#[test]
fn test_scalar_op_vec6()
{ test_scalar_op_impl!(Vec6<f64>, f64); }

#[test]
fn test_iterator_vec1()
{ test_iterator_impl!(Vec1<f64>, f64); }

#[test]
fn test_iterator_vec2()
{ test_iterator_impl!(Vec2<f64>, f64); }
 
#[test]
fn test_iterator_vec3()
{ test_iterator_impl!(Vec3<f64>, f64); }
 
#[test]
fn test_iterator_vec4()
{ test_iterator_impl!(Vec4<f64>, f64); }
 
#[test]
fn test_iterator_vec5()
{ test_iterator_impl!(Vec5<f64>, f64); }
 
#[test]
fn test_iterator_vec6()
{ test_iterator_impl!(Vec6<f64>, f64); }
