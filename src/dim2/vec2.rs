use std::num::{Zero, One, Algebraic, Bounded};
use std::rand::{Rand, Rng, RngUtil};
use dim1::vec1::Vec1;
use std::cmp::ApproxEq;
use traits::basis::Basis;
use traits::cross::Cross;
use traits::dim::Dim;
use traits::dot::Dot;
use traits::sub_dot::SubDot;
use traits::norm::Norm;
use traits::flatten::Flatten;
use traits::translation::{Translation, Translatable};
use traits::scalar_op::{ScalarMul, ScalarDiv, ScalarAdd, ScalarSub};

#[deriving(Eq, Ord, ToStr)]
pub struct Vec2<N>
{
  x : N,
  y : N
}

impl<N> Vec2<N>
{
  #[inline]
  pub fn new(x: N, y: N) -> Vec2<N>
  { Vec2 {x: x, y: y} }
}

impl<N> Dim for Vec2<N>
{
  #[inline]
  fn dim() -> uint
  { 2 }
}

impl<N: Add<N,N>> Add<Vec2<N>, Vec2<N>> for Vec2<N>
{
  #[inline]
  fn add(&self, other: &Vec2<N>) -> Vec2<N>
  { Vec2::new(self.x + other.x, self.y + other.y) }
}

impl<N: Sub<N,N>> Sub<Vec2<N>, Vec2<N>> for Vec2<N>
{
  #[inline]
  fn sub(&self, other: &Vec2<N>) -> Vec2<N>
  { Vec2::new(self.x - other.x, self.y - other.y) }
}

impl<N: Mul<N, N>>
ScalarMul<N> for Vec2<N>
{
  #[inline]
  fn scalar_mul(&self, s: &N) -> Vec2<N>
  { Vec2 { x: self.x * *s, y: self.y * *s } }

  #[inline]
  fn scalar_mul_inplace(&mut self, s: &N)
  {
    self.x = self.x * *s;
    self.y = self.y * *s;
  }
}


impl<N: Div<N, N>>
ScalarDiv<N> for Vec2<N>
{
  #[inline]
  fn scalar_div(&self, s: &N) -> Vec2<N>
  { Vec2 { x: self.x / *s, y: self.y / *s } }

  #[inline]
  fn scalar_div_inplace(&mut self, s: &N)
  {
    self.x = self.x / *s;
    self.y = self.y / *s;
  }
}

impl<N: Add<N, N>>
ScalarAdd<N> for Vec2<N>
{
  #[inline]
  fn scalar_add(&self, s: &N) -> Vec2<N>
  { Vec2 { x: self.x + *s, y: self.y + *s } }

  #[inline]
  fn scalar_add_inplace(&mut self, s: &N)
  {
    self.x = self.x + *s;
    self.y = self.y + *s;
  }
}

impl<N: Sub<N, N>>
ScalarSub<N> for Vec2<N>
{
  #[inline]
  fn scalar_sub(&self, s: &N) -> Vec2<N>
  { Vec2 { x: self.x - *s, y: self.y - *s } }

  #[inline]
  fn scalar_sub_inplace(&mut self, s: &N)
  {
    self.x = self.x - *s;
    self.y = self.y - *s;
  }
}

impl<N: Copy + Add<N, N>> Translation<Vec2<N>> for Vec2<N>
{
  #[inline]
  fn translation(&self) -> Vec2<N>
  { copy *self }

  #[inline]
  fn translate(&mut self, t: &Vec2<N>)
  { *self = *self + *t; }
}

impl<N: Add<N, N>> Translatable<Vec2<N>, Vec2<N>> for Vec2<N>
{
  #[inline]
  fn translated(&self, t: &Vec2<N>) -> Vec2<N>
  { self + *t }
}

impl<N: Mul<N, N> + Add<N, N>> Dot<N> for Vec2<N>
{
  #[inline]
  fn dot(&self, other : &Vec2<N>) -> N
  { self.x * other.x + self.y * other.y } 
}

impl<N: Mul<N, N> + Add<N, N> + Sub<N, N>> SubDot<N> for Vec2<N>
{
  #[inline]
  fn sub_dot(&self, a: &Vec2<N>, b: &Vec2<N>) -> N
  { (self.x - a.x) * b.x + (self.y - a.y) * b.y } 
}

impl<N: Mul<N, N> + Add<N, N> + Div<N, N> + Algebraic>
Norm<N> for Vec2<N>
{
  #[inline]
  fn sqnorm(&self) -> N
  { self.dot(self) }

  #[inline]
  fn norm(&self) -> N
  { self.sqnorm().sqrt() }

  #[inline]
  fn normalized(&self) -> Vec2<N>
  {
    let l = self.norm();

    Vec2::new(self.x / l, self.y / l)
  }

  #[inline]
  fn normalize(&mut self) -> N
  {
    let l = self.norm();

    self.x = self.x / l;
    self.y = self.y / l;

    l
  }
}

impl<N: Mul<N, N> + Sub<N, N>> Cross<Vec1<N>> for Vec2<N>
{
  #[inline]
  fn cross(&self, other : &Vec2<N>) -> Vec1<N>
  { Vec1::new(self.x * other.y - self.y * other.x) }
}

impl<N: Neg<N>> Neg<Vec2<N>> for Vec2<N>
{
  #[inline]
  fn neg(&self) -> Vec2<N>
  { Vec2::new(-self.x, -self.y) }
}

impl<N: Zero> Zero for Vec2<N>
{
  #[inline]
  fn zero() -> Vec2<N>
  { Vec2::new(Zero::zero(), Zero::zero()) }

  #[inline]
  fn is_zero(&self) -> bool
  { self.x.is_zero() && self.y.is_zero() }
}

impl<N: Copy + One + Zero + Neg<N>> Basis for Vec2<N>
{
  #[inline]
  fn canonical_basis()     -> ~[Vec2<N>]
  {
    // FIXME: this should be static
    ~[ Vec2::new(One::one(), Zero::zero()),
       Vec2::new(Zero::zero(), One::one()) ]
  }

  #[inline]
  fn orthogonal_subspace_basis(&self) -> ~[Vec2<N>]
  { ~[ Vec2::new(-self.y, copy self.x) ] }
}

impl<N:ApproxEq<N>> ApproxEq<N> for Vec2<N>
{
  #[inline]
  fn approx_epsilon() -> N
  { ApproxEq::approx_epsilon::<N, N>() }

  #[inline]
  fn approx_eq(&self, other: &Vec2<N>) -> bool
  { self.x.approx_eq(&other.x) && self.y.approx_eq(&other.y) }

  #[inline]
  fn approx_eq_eps(&self, other: &Vec2<N>, epsilon: &N) -> bool
  {
    self.x.approx_eq_eps(&other.x, epsilon) &&
    self.y.approx_eq_eps(&other.y, epsilon)
  }
}

impl<N:Rand> Rand for Vec2<N>
{
  #[inline]
  fn rand<R: Rng>(rng: &mut R) -> Vec2<N>
  { Vec2::new(rng.gen(), rng.gen()) }
}

impl<N: Copy> Flatten<N> for Vec2<N>
{
  #[inline]
  fn flat_size() -> uint
  { 2 }

  #[inline]
  fn from_flattened(l: &[N], off: uint) -> Vec2<N>
  { Vec2::new(copy l[off], copy l[off + 1]) }

  #[inline]
  fn flatten(&self) -> ~[N]
  { ~[ copy self.x, copy self.y ] }

  #[inline]
  fn flatten_to(&self, l: &mut [N], off: uint)
  {
    l[off]     = copy self.x;
    l[off + 1] = copy self.y;
  }
}

impl<N: Bounded> Bounded for Vec2<N>
{
  #[inline]
  fn max_value() -> Vec2<N>
  { Vec2::new(Bounded::max_value(), Bounded::max_value()) }

  #[inline]
  fn min_value() -> Vec2<N>
  { Vec2::new(Bounded::min_value(), Bounded::min_value()) }
}
