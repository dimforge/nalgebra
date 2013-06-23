use std::uint::iterate;
use std::num::{Zero, One, Algebraic};
use std::vec::{map_zip, map, from_elem};
use std::cmp::ApproxEq;
use std::iterator::IteratorUtil;
use traits::ring::Ring;
use traits::division_ring::DivisionRing;
use traits::dot::Dot;
use traits::sub_dot::SubDot;
use traits::norm::Norm;
use traits::translation::Translation;
use traits::scalar_op::{ScalarMul, ScalarDiv, ScalarAdd, ScalarSub};

#[deriving(Eq, Ord, ToStr, Clone)]
pub struct DVec<N>
{
  at: ~[N]
}

#[inline(always)]
pub fn zero_vec_with_dim<N: Zero + Copy>(dim: uint) -> DVec<N>
{ DVec { at: from_elem(dim, Zero::zero::<N>()) } }

#[inline(always)]
pub fn is_zero_vec<N: Zero>(vec: &DVec<N>) -> bool
{ vec.at.iter().all(|e| e.is_zero()) }

// FIXME: is Clone needed?
impl<N: Copy + DivisionRing + Algebraic + Clone + ApproxEq<N>> DVec<N>
{
  pub fn canonical_basis_with_dim(dim: uint) -> ~[DVec<N>]
  {
    let mut res : ~[DVec<N>] = ~[];

    for iterate(0u, dim) |i|
    {
      let mut basis_element : DVec<N> = zero_vec_with_dim(dim);

      basis_element.at[i] = One::one();

      res.push(basis_element);
    }

    res
  }

  pub fn orthogonal_subspace_basis(&self) -> ~[DVec<N>]
  {
    // compute the basis of the orthogonal subspace using Gram-Schmidt
    // orthogonalization algorithm
    let     dim              = self.at.len();
    let mut res : ~[DVec<N>] = ~[];

    for iterate(0u, dim) |i|
    {
      let mut basis_element : DVec<N> = zero_vec_with_dim(self.at.len());

      basis_element.at[i] = One::one();

      if res.len() == dim - 1
      { break; }

      let mut elt = basis_element.clone();

      elt = elt - self.scalar_mul(&basis_element.dot(self));

      for res.iter().advance |v|
      { elt = elt - v.scalar_mul(&elt.dot(v)) };

      if !elt.sqnorm().approx_eq(&Zero::zero())
      { res.push(elt.normalized()); }
    }

    assert!(res.len() == dim - 1);

    res
  }
}

impl<N: Copy + Add<N,N>> Add<DVec<N>, DVec<N>> for DVec<N>
{
  #[inline(always)]
  fn add(&self, other: &DVec<N>) -> DVec<N>
  {
    assert!(self.at.len() == other.at.len());
    DVec { at: map_zip(self.at, other.at, | a, b | { *a + *b }) }
  }
}

impl<N: Copy + Sub<N,N>> Sub<DVec<N>, DVec<N>> for DVec<N>
{
  #[inline(always)]
  fn sub(&self, other: &DVec<N>) -> DVec<N>
  {
    assert!(self.at.len() == other.at.len());
    DVec { at: map_zip(self.at, other.at, | a, b | *a - *b) }
  }
}

impl<N: Copy + Neg<N>> Neg<DVec<N>> for DVec<N>
{
  #[inline(always)]
  fn neg(&self) -> DVec<N>
  { DVec { at: map(self.at, |a| -a) } }
}

impl<N: Copy + Ring>
Dot<N> for DVec<N>
{
  #[inline(always)]
  fn dot(&self, other: &DVec<N>) -> N
  {
    assert!(self.at.len() == other.at.len());

    let mut res = Zero::zero::<N>();

    for iterate(0u, self.at.len()) |i|
    { res = res + self.at[i] * other.at[i]; }

    res
  } 
}

impl<N: Copy + Ring> SubDot<N> for DVec<N>
{
  #[inline(always)]
  fn sub_dot(&self, a: &DVec<N>, b: &DVec<N>) -> N
  {
    let mut res = Zero::zero::<N>();

    for iterate(0u, self.at.len()) |i|
    { res = res + (self.at[i] - a.at[i]) * b.at[i]; }

    res
  } 
}

impl<N: Copy + Mul<N, N>>
ScalarMul<N> for DVec<N>
{
  #[inline(always)]
  fn scalar_mul(&self, s: &N) -> DVec<N>
  { DVec { at: map(self.at, |a| a * *s) } }

  #[inline(always)]
  fn scalar_mul_inplace(&mut self, s: &N)
  {
    for iterate(0u, self.at.len()) |i|
    { self.at[i] = self.at[i] * copy *s; }
  }
}


impl<N: Copy + Div<N, N>>
ScalarDiv<N> for DVec<N>
{
  #[inline(always)]
  fn scalar_div(&self, s: &N) -> DVec<N>
  { DVec { at: map(self.at, |a| a / *s) } }

  #[inline(always)]
  fn scalar_div_inplace(&mut self, s: &N)
  {
    for iterate(0u, self.at.len()) |i|
    { self.at[i] = self.at[i] / copy *s; }
  }
}

impl<N: Copy + Add<N, N>>
ScalarAdd<N> for DVec<N>
{
  #[inline(always)]
  fn scalar_add(&self, s: &N) -> DVec<N>
  { DVec { at: map(self.at, |a| a + *s) } }

  #[inline(always)]
  fn scalar_add_inplace(&mut self, s: &N)
  {
    for iterate(0u, self.at.len()) |i|
    { self.at[i] = self.at[i] + copy *s; }
  }
}

impl<N: Copy + Sub<N, N>>
ScalarSub<N> for DVec<N>
{
  #[inline(always)]
  fn scalar_sub(&self, s: &N) -> DVec<N>
  { DVec { at: map(self.at, |a| a - *s) } }

  #[inline(always)]
  fn scalar_sub_inplace(&mut self, s: &N)
  {
    for iterate(0u, self.at.len()) |i|
    { self.at[i] = self.at[i] - copy *s; }
  }
}

impl<N: Clone + Copy + Add<N, N>> Translation<DVec<N>> for DVec<N>
{
  #[inline(always)]
  fn translation(&self) -> DVec<N>
  { self.clone() }

  #[inline(always)]
  fn translated(&self, t: &DVec<N>) -> DVec<N>
  { self + *t }

  #[inline(always)]
  fn translate(&mut self, t: &DVec<N>)
  { *self = *self + *t; }
}

impl<N: Copy + DivisionRing + Algebraic + Clone>
Norm<N> for DVec<N>
{
  #[inline(always)]
  fn sqnorm(&self) -> N
  { self.dot(self) }

  #[inline(always)]
  fn norm(&self) -> N
  { self.sqnorm().sqrt() }

  #[inline(always)]
  fn normalized(&self) -> DVec<N>
  {
    let mut res : DVec<N> = self.clone();

    res.normalize();

    res
  }

  #[inline(always)]
  fn normalize(&mut self) -> N
  {
    let l = self.norm();

    for iterate(0u, self.at.len()) |i|
    { self.at[i] = self.at[i] / copy l; }

    l
  }
}

impl<N: ApproxEq<N>> ApproxEq<N> for DVec<N>
{
  #[inline(always)]
  fn approx_epsilon() -> N
  { ApproxEq::approx_epsilon::<N, N>() }

  #[inline(always)]
  fn approx_eq(&self, other: &DVec<N>) -> bool
  {
    let mut zip = self.at.iter().zip(other.at.iter());

    do zip.all |(a, b)| { a.approx_eq(b) }
  }

  #[inline(always)]
  fn approx_eq_eps(&self, other: &DVec<N>, epsilon: &N) -> bool
  {
    let mut zip = self.at.iter().zip(other.at.iter());

    do zip.all |(a, b)| { a.approx_eq_eps(b, epsilon) }
  }
}
