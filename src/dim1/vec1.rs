use std::num::{Zero, One, Algebraic, Bounded};
use std::rand::{Rand, Rng, RngUtil};
use std::cmp::ApproxEq;
use traits::basis::Basis;
use traits::dim::Dim;
use traits::dot::Dot;
use traits::norm::Norm;
use traits::translation::{Translation, Translatable};
use traits::sub_dot::SubDot;
use traits::flatten::Flatten;
use traits::scalar_op::{ScalarMul, ScalarDiv, ScalarAdd, ScalarSub};

#[deriving(Eq, Ord, ToStr)]
pub struct Vec1<N>
{ x : N }

impl<N> Vec1<N>
{
  #[inline]
  pub fn new(x: N) -> Vec1<N>
  { Vec1 {x: x} }
}

impl<N> Dim for Vec1<N>
{
  #[inline]
  fn dim() -> uint
  { 1 }
}

impl<N: Add<N, N>> Add<Vec1<N>, Vec1<N>> for Vec1<N>
{
  #[inline]
  fn add(&self, other: &Vec1<N>) -> Vec1<N>
  { Vec1::new(self.x + other.x) }
}

impl<N: Sub<N, N>> Sub<Vec1<N>, Vec1<N>> for Vec1<N>
{
  #[inline]
  fn sub(&self, other: &Vec1<N>) -> Vec1<N>
  { Vec1::new(self.x - other.x) }
}

impl<N: Mul<N, N>>
ScalarMul<N> for Vec1<N>
{
  #[inline]
  fn scalar_mul(&self, s: &N) -> Vec1<N>
  { Vec1 { x: self.x * *s } }

  #[inline]
  fn scalar_mul_inplace(&mut self, s: &N)
  { self.x = self.x * *s; }
}


impl<N: Div<N, N>>
ScalarDiv<N> for Vec1<N>
{
  #[inline]
  fn scalar_div(&self, s: &N) -> Vec1<N>
  { Vec1 { x: self.x / *s } }

  #[inline]
  fn scalar_div_inplace(&mut self, s: &N)
  { self.x = self.x / *s; }
}

impl<N: Add<N, N>>
ScalarAdd<N> for Vec1<N>
{
  #[inline]
  fn scalar_add(&self, s: &N) -> Vec1<N>
  { Vec1 { x: self.x + *s } }

  #[inline]
  fn scalar_add_inplace(&mut self, s: &N)
  { self.x = self.x + *s; }
}

impl<N: Sub<N, N>>
ScalarSub<N> for Vec1<N>
{
  #[inline]
  fn scalar_sub(&self, s: &N) -> Vec1<N>
  { Vec1 { x: self.x - *s } }

  #[inline]
  fn scalar_sub_inplace(&mut self, s: &N)
  { self.x = self.x - *s; }
}

impl<N: Copy + Add<N, N>> Translation<Vec1<N>> for Vec1<N>
{
  #[inline]
  fn translation(&self) -> Vec1<N>
  { copy *self }

  #[inline]
  fn translate(&mut self, t: &Vec1<N>)
  { *self = *self + *t }
}

impl<N: Add<N, N>> Translatable<Vec1<N>, Vec1<N>> for Vec1<N>
{
  #[inline]
  fn translated(&self, t: &Vec1<N>) -> Vec1<N>
  { self + *t }
}

impl<N: Mul<N, N>> Dot<N> for Vec1<N>
{
  #[inline]
  fn dot(&self, other : &Vec1<N>) -> N
  { self.x * other.x } 
}

impl<N: Mul<N, N> + Sub<N, N>> SubDot<N> for Vec1<N>
{
  #[inline]
  fn sub_dot(&self, a: &Vec1<N>, b: &Vec1<N>) -> N
  { (self.x - a.x) * b.x } 
}

impl<N: Mul<N, N> + Add<N, N> + Div<N, N> + Algebraic>
Norm<N> for Vec1<N>
{
  #[inline]
  fn sqnorm(&self) -> N
  { self.dot(self) }

  #[inline]
  fn norm(&self) -> N
  { self.sqnorm().sqrt() }

  #[inline]
  fn normalized(&self) -> Vec1<N>
  { Vec1::new(self.x / self.norm()) }

  #[inline]
  fn normalize(&mut self) -> N
  {
    let l = self.norm();

    self.x = self.x / l;

    l
  }
}

impl<N: Neg<N>> Neg<Vec1<N>> for Vec1<N>
{
  #[inline]
  fn neg(&self) -> Vec1<N>
  { Vec1::new(-self.x) }
}

impl<N: Zero> Zero for Vec1<N>
{
  #[inline]
  fn zero() -> Vec1<N>
  {
    let _0 = Zero::zero();
    Vec1::new(_0)
  }

  #[inline]
  fn is_zero(&self) -> bool
  { self.x.is_zero() }
}

impl<N: One> Basis for Vec1<N>
{
  #[inline]
  fn canonical_basis() -> ~[Vec1<N>]
  { ~[ Vec1::new(One::one()) ] } // FIXME: this should be static

  #[inline]
  fn orthogonal_subspace_basis(&self) -> ~[Vec1<N>]
  { ~[] }
}

impl<N:ApproxEq<N>> ApproxEq<N> for Vec1<N>
{
  #[inline]
  fn approx_epsilon() -> N
  { ApproxEq::approx_epsilon::<N, N>() }

  #[inline]
  fn approx_eq(&self, other: &Vec1<N>) -> bool
  { self.x.approx_eq(&other.x) }

  #[inline]
  fn approx_eq_eps(&self, other: &Vec1<N>, epsilon: &N) -> bool
  { self.x.approx_eq_eps(&other.x, epsilon) }
}

impl<N: Rand> Rand for Vec1<N>
{
  #[inline]
  fn rand<R: Rng>(rng: &mut R) -> Vec1<N>
  { Vec1::new(rng.gen()) }
}

impl<N: Copy> Flatten<N> for Vec1<N>
{
  #[inline]
  fn flat_size() -> uint
  { 1 }

  #[inline]
  fn from_flattened(l: &[N], off: uint) -> Vec1<N>
  { Vec1::new(copy l[off]) }

  #[inline]
  fn flatten(&self) -> ~[N]
  { ~[ copy self.x ] }

  #[inline]
  fn flatten_to(&self, l: &mut [N], off: uint)
  { l[off] = copy self.x }
}

impl<N: Bounded> Bounded for Vec1<N>
{
  #[inline]
  fn max_value() -> Vec1<N>
  { Vec1::new(Bounded::max_value()) }

  #[inline]
  fn min_value() -> Vec1<N>
  { Vec1::new(Bounded::min_value()) }
}
