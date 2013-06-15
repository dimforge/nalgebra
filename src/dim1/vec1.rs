use std::num::{Zero, One, Algebraic};
use std::rand::{Rand, Rng, RngUtil};
use std::cmp::ApproxEq;
use traits::basis::Basis;
use traits::dim::Dim;
use traits::dot::Dot;
use traits::norm::Norm;
use traits::translation::Translation;
use traits::sub_dot::SubDot;
use traits::flatten::Flatten;
use traits::workarounds::scalar_op::{ScalarMul, ScalarDiv, ScalarAdd, ScalarSub};

#[deriving(Eq, Ord, ToStr)]
pub struct Vec1<N>
{ x : N }


impl<N: Copy> Vec1<N>
{
  #[inline(always)]
  pub fn new(x: N) -> Vec1<N>
  { Vec1 {x: x} }
}

impl<N> Dim for Vec1<N>
{
  #[inline(always)]
  fn dim() -> uint
  { 1 }
}

impl<N:Copy + Add<N,N>> Add<Vec1<N>, Vec1<N>> for Vec1<N>
{
  #[inline(always)]
  fn add(&self, other: &Vec1<N>) -> Vec1<N>
  { Vec1::new(self.x + other.x) }
}

impl<N:Copy + Sub<N,N>> Sub<Vec1<N>, Vec1<N>> for Vec1<N>
{
  #[inline(always)]
  fn sub(&self, other: &Vec1<N>) -> Vec1<N>
  { Vec1::new(self.x - other.x) }
}

impl<N: Copy + Mul<N, N>>
ScalarMul<N> for Vec1<N>
{
  #[inline(always)]
  fn scalar_mul(&self, s: &N) -> Vec1<N>
  { Vec1 { x: self.x * *s } }

  #[inline(always)]
  fn scalar_mul_inplace(&mut self, s: &N)
  { self.x *= *s; }
}


impl<N: Copy + Div<N, N>>
ScalarDiv<N> for Vec1<N>
{
  #[inline(always)]
  fn scalar_div(&self, s: &N) -> Vec1<N>
  { Vec1 { x: self.x / *s } }

  #[inline(always)]
  fn scalar_div_inplace(&mut self, s: &N)
  { self.x /= *s; }
}

impl<N: Copy + Add<N, N>>
ScalarAdd<N> for Vec1<N>
{
  #[inline(always)]
  fn scalar_add(&self, s: &N) -> Vec1<N>
  { Vec1 { x: self.x + *s } }

  #[inline(always)]
  fn scalar_add_inplace(&mut self, s: &N)
  { self.x += *s; }
}

impl<N: Copy + Sub<N, N>>
ScalarSub<N> for Vec1<N>
{
  #[inline(always)]
  fn scalar_sub(&self, s: &N) -> Vec1<N>
  { Vec1 { x: self.x - *s } }

  #[inline(always)]
  fn scalar_sub_inplace(&mut self, s: &N)
  { self.x -= *s; }
}

impl<N: Copy + Add<N, N>> Translation<Vec1<N>> for Vec1<N>
{
  #[inline(always)]
  fn translation(&self) -> Vec1<N>
  { *self }

  #[inline(always)]
  fn translated(&self, t: &Vec1<N>) -> Vec1<N>
  { self + *t }

  #[inline(always)]
  fn translate(&mut self, t: &Vec1<N>)
  { *self += *t }
}

impl<N:Copy + Mul<N, N>> Dot<N> for Vec1<N>
{
  #[inline(always)]
  fn dot(&self, other : &Vec1<N>) -> N
  { self.x * other.x } 
}

impl<N:Copy + Mul<N, N> + Sub<N, N>> SubDot<N> for Vec1<N>
{
  #[inline(always)]
  fn sub_dot(&self, a: &Vec1<N>, b: &Vec1<N>) -> N
  { (self.x - a.x) * b.x } 
}

impl<N:Copy + Mul<N, N> + Add<N, N> + Div<N, N> + Algebraic>
Norm<N> for Vec1<N>
{
  #[inline(always)]
  fn sqnorm(&self) -> N
  { self.dot(self) }

  #[inline(always)]
  fn norm(&self) -> N
  { self.sqnorm().sqrt() }

  #[inline(always)]
  fn normalized(&self) -> Vec1<N>
  { Vec1::new(self.x / self.norm()) }

  #[inline(always)]
  fn normalize(&mut self) -> N
  {
    let l = self.norm();

    self.x /= l;

    l
  }
}

impl<N:Copy + Neg<N>> Neg<Vec1<N>> for Vec1<N>
{
  #[inline(always)]
  fn neg(&self) -> Vec1<N>
  { Vec1::new(-self.x) }
}

impl<N:Copy + Zero> Zero for Vec1<N>
{
  #[inline(always)]
  fn zero() -> Vec1<N>
  {
    let _0 = Zero::zero();
    Vec1::new(_0)
  }

  #[inline(always)]
  fn is_zero(&self) -> bool
  { self.x.is_zero() }
}

impl<N: Copy + One> Basis for Vec1<N>
{
  #[inline(always)]
  fn canonical_basis() -> ~[Vec1<N>]
  { ~[ Vec1::new(One::one()) ] } // FIXME: this should be static

  #[inline(always)]
  fn orthogonal_subspace_basis(&self) -> ~[Vec1<N>]
  { ~[] }
}

impl<N:ApproxEq<N>> ApproxEq<N> for Vec1<N>
{
  #[inline(always)]
  fn approx_epsilon() -> N
  { ApproxEq::approx_epsilon::<N, N>() }

  #[inline(always)]
  fn approx_eq(&self, other: &Vec1<N>) -> bool
  { self.x.approx_eq(&other.x) }

  #[inline(always)]
  fn approx_eq_eps(&self, other: &Vec1<N>, epsilon: &N) -> bool
  { self.x.approx_eq_eps(&other.x, epsilon) }
}

impl<N: Rand + Copy> Rand for Vec1<N>
{
  #[inline(always)]
  fn rand<R: Rng>(rng: &mut R) -> Vec1<N>
  { Vec1::new(rng.gen()) }
}

impl<N: Copy> Flatten<N> for Vec1<N>
{
  #[inline(always)]
  fn flat_size() -> uint
  { 1 }

  #[inline(always)]
  fn from_flattened(l: &[N], off: uint) -> Vec1<N>
  { Vec1::new(l[off]) }

  #[inline(always)]
  fn flatten(&self) -> ~[N]
  { ~[ self.x ] }

  #[inline(always)]
  fn flatten_to(&self, l: &mut [N], off: uint)
  { l[off] = self.x }
}
