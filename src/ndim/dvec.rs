use core::num::{Zero, One, Algebraic};
use core::vec::{map_zip, map, all2, len, from_elem, all};
use core::cmp::ApproxEq;
use traits::ring::Ring;
use traits::division_ring::DivisionRing;
use traits::dot::Dot;
use traits::sub_dot::SubDot;
use traits::norm::Norm;
use traits::translation::Translation;
use traits::workarounds::scalar_op::{ScalarMul, ScalarDiv, ScalarAdd, ScalarSub};

#[deriving(Eq, ToStr, Clone)]
pub struct DVec<T>
{
  at: ~[T]
}

pub fn zero_with_dim<T: Zero + Copy>(dim: uint) -> DVec<T>
{ DVec { at: from_elem(dim, Zero::zero::<T>()) } }

pub fn is_zero<T: Zero>(vec: &DVec<T>) -> bool
{ all(vec.at, |e| e.is_zero()) }

// FIXME: is Clone needed?
impl<T: Copy + DivisionRing + Algebraic + Clone + ApproxEq<T>> DVec<T>
{
  pub fn canonical_basis_with_dim(dim: uint) -> ~[DVec<T>]
  {
    let mut res : ~[DVec<T>] = ~[];

    for uint::range(0u, dim) |i|
    {
      let mut basis_element : DVec<T> = zero_with_dim(dim);

      basis_element.at[i] = One::one();

      res.push(basis_element);
    }

    res
  }

  pub fn orthogonal_subspace_basis(&self) -> ~[DVec<T>]
  {
    // compute the basis of the orthogonal subspace using Gram-Schmidt
    // orthogonalization algorithm
    let     dim              = len(self.at);
    let mut res : ~[DVec<T>] = ~[];

    for uint::range(0u, dim) |i|
    {
      let mut basis_element : DVec<T> = zero_with_dim(len(self.at));

      basis_element.at[i] = One::one();

      if (res.len() == dim - 1)
      { break; }

      let mut elt = basis_element.clone();

      elt -= self.scalar_mul(&basis_element.dot(self));

      for res.each |v|
      { elt -= v.scalar_mul(&elt.dot(v)) };

      if (!elt.sqnorm().approx_eq(&Zero::zero()))
      { res.push(elt.normalized()); }
    }

    assert!(res.len() == dim - 1);

    res
  }
}

impl<T: Copy + Add<T,T>> Add<DVec<T>, DVec<T>> for DVec<T>
{
  fn add(&self, other: &DVec<T>) -> DVec<T>
  {
    assert!(len(self.at) == len(other.at));
    DVec { at: map_zip(self.at, other.at, | a, b | { *a + *b }) }
  }
}

impl<T: Copy + Sub<T,T>> Sub<DVec<T>, DVec<T>> for DVec<T>
{
  fn sub(&self, other: &DVec<T>) -> DVec<T>
  {
    assert!(len(self.at) == len(other.at));
    DVec { at: map_zip(self.at, other.at, | a, b | *a - *b) }
  }
}

impl<T: Copy + Neg<T>> Neg<DVec<T>> for DVec<T>
{
  fn neg(&self) -> DVec<T>
  { DVec { at: map(self.at, |a| -a) } }
}

impl<T: Copy + Ring>
Dot<T> for DVec<T>
{
  fn dot(&self, other: &DVec<T>) -> T
  {
    assert!(len(self.at) == len(other.at));

    let mut res = Zero::zero::<T>();

    for uint::range(0u, len(self.at)) |i|
    { res += self.at[i] * other.at[i]; }

    res
  } 
}

impl<T: Copy + Ring> SubDot<T> for DVec<T>
{
  fn sub_dot(&self, a: &DVec<T>, b: &DVec<T>) -> T
  {
    let mut res = Zero::zero::<T>();

    for uint::range(0u, len(self.at)) |i|
    { res += (self.at[i] - a.at[i]) * b.at[i]; }

    res
  } 
}

impl<T: Copy + Mul<T, T>>
ScalarMul<T> for DVec<T>
{
  fn scalar_mul(&self, s: &T) -> DVec<T>
  { DVec { at: map(self.at, |a| a * *s) } }

  fn scalar_mul_inplace(&mut self, s: &T)
  {
    for uint::range(0u, len(self.at)) |i|
    { self.at[i] *= *s; }
  }
}


impl<T: Copy + Div<T, T>>
ScalarDiv<T> for DVec<T>
{
  fn scalar_div(&self, s: &T) -> DVec<T>
  { DVec { at: map(self.at, |a| a / *s) } }

  fn scalar_div_inplace(&mut self, s: &T)
  {
    for uint::range(0u, len(self.at)) |i|
    { self.at[i] /= *s; }
  }
}

impl<T: Copy + Add<T, T>>
ScalarAdd<T> for DVec<T>
{
  fn scalar_add(&self, s: &T) -> DVec<T>
  { DVec { at: map(self.at, |a| a + *s) } }

  fn scalar_add_inplace(&mut self, s: &T)
  {
    for uint::range(0u, len(self.at)) |i|
    { self.at[i] += *s; }
  }
}

impl<T: Copy + Sub<T, T>>
ScalarSub<T> for DVec<T>
{
  fn scalar_sub(&self, s: &T) -> DVec<T>
  { DVec { at: map(self.at, |a| a - *s) } }

  fn scalar_sub_inplace(&mut self, s: &T)
  {
    for uint::range(0u, len(self.at)) |i|
    { self.at[i] -= *s; }
  }
}

impl<T: Clone + Copy + Add<T, T>> Translation<DVec<T>> for DVec<T>
{
  fn translation(&self) -> DVec<T>
  { self.clone() }

  fn translated(&self, t: &DVec<T>) -> DVec<T>
  { self + *t }

  fn translate(&mut self, t: &DVec<T>)
  { *self = *self + *t; }
}

impl<T: Copy + DivisionRing + Algebraic + Clone>
Norm<T> for DVec<T>
{
  fn sqnorm(&self) -> T
  { self.dot(self) }

  fn norm(&self) -> T
  { self.sqnorm().sqrt() }

  fn normalized(&self) -> DVec<T>
  {
    let mut res : DVec<T> = self.clone();

    res.normalize();

    res
  }

  fn normalize(&mut self) -> T
  {
    let l = self.norm();

    for uint::range(0u, len(self.at)) |i|
    { self.at[i] /= l; }

    l
  }
}

impl<T: ApproxEq<T>> ApproxEq<T> for DVec<T>
{
  fn approx_epsilon() -> T
  { ApproxEq::approx_epsilon::<T, T>() }

  fn approx_eq(&self, other: &DVec<T>) -> bool
  { all2(self.at, other.at, |a, b| a.approx_eq(b)) }

  fn approx_eq_eps(&self, other: &DVec<T>, epsilon: &T) -> bool
  { all2(self.at, other.at, |a, b| a.approx_eq_eps(b, epsilon)) }
}
