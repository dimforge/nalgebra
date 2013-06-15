use std::num::{Zero, One, Algebraic};
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
use traits::translation::Translation;
use traits::scalar_op::{ScalarMul, ScalarDiv, ScalarAdd, ScalarSub};

#[deriving(Eq, Ord, ToStr)]
pub struct Vec2<N>
{
  x : N,
  y : N
}

impl<N: Copy> Vec2<N>
{
  #[inline(always)]
  pub fn new(x: N, y: N) -> Vec2<N>
  { Vec2 {x: x, y: y} }
}

impl<N> Dim for Vec2<N>
{
  #[inline(always)]
  fn dim() -> uint
  { 2 }
}

impl<N:Copy + Add<N,N>> Add<Vec2<N>, Vec2<N>> for Vec2<N>
{
  #[inline(always)]
  fn add(&self, other: &Vec2<N>) -> Vec2<N>
  { Vec2::new(self.x + other.x, self.y + other.y) }
}

impl<N:Copy + Sub<N,N>> Sub<Vec2<N>, Vec2<N>> for Vec2<N>
{
  #[inline(always)]
  fn sub(&self, other: &Vec2<N>) -> Vec2<N>
  { Vec2::new(self.x - other.x, self.y - other.y) }
}

impl<N: Copy + Mul<N, N>>
ScalarMul<N> for Vec2<N>
{
  #[inline(always)]
  fn scalar_mul(&self, s: &N) -> Vec2<N>
  { Vec2 { x: self.x * *s, y: self.y * *s } }

  #[inline(always)]
  fn scalar_mul_inplace(&mut self, s: &N)
  {
    self.x *= *s;
    self.y *= *s;
  }
}


impl<N: Copy + Div<N, N>>
ScalarDiv<N> for Vec2<N>
{
  #[inline(always)]
  fn scalar_div(&self, s: &N) -> Vec2<N>
  { Vec2 { x: self.x / *s, y: self.y / *s } }

  #[inline(always)]
  fn scalar_div_inplace(&mut self, s: &N)
  {
    self.x /= *s;
    self.y /= *s;
  }
}

impl<N: Copy + Add<N, N>>
ScalarAdd<N> for Vec2<N>
{
  #[inline(always)]
  fn scalar_add(&self, s: &N) -> Vec2<N>
  { Vec2 { x: self.x + *s, y: self.y + *s } }

  #[inline(always)]
  fn scalar_add_inplace(&mut self, s: &N)
  {
    self.x += *s;
    self.y += *s;
  }
}

impl<N: Copy + Sub<N, N>>
ScalarSub<N> for Vec2<N>
{
  #[inline(always)]
  fn scalar_sub(&self, s: &N) -> Vec2<N>
  { Vec2 { x: self.x - *s, y: self.y - *s } }

  #[inline(always)]
  fn scalar_sub_inplace(&mut self, s: &N)
  {
    self.x -= *s;
    self.y -= *s;
  }
}

impl<N: Copy + Add<N, N>> Translation<Vec2<N>> for Vec2<N>
{
  #[inline(always)]
  fn translation(&self) -> Vec2<N>
  { *self }

  #[inline(always)]
  fn translated(&self, t: &Vec2<N>) -> Vec2<N>
  { self + *t }

  #[inline(always)]
  fn translate(&mut self, t: &Vec2<N>)
  { *self += *t; }
}

impl<N:Copy + Mul<N, N> + Add<N, N>> Dot<N> for Vec2<N>
{
  #[inline(always)]
  fn dot(&self, other : &Vec2<N>) -> N
  { self.x * other.x + self.y * other.y } 
}

impl<N:Copy + Mul<N, N> + Add<N, N> + Sub<N, N>> SubDot<N> for Vec2<N>
{
  #[inline(always)]
  fn sub_dot(&self, a: &Vec2<N>, b: &Vec2<N>) -> N
  { (self.x - a.x) * b.x + (self.y - a.y) * b.y } 
}

impl<N:Copy + Mul<N, N> + Add<N, N> + Div<N, N> + Algebraic>
Norm<N> for Vec2<N>
{
  #[inline(always)]
  fn sqnorm(&self) -> N
  { self.dot(self) }

  #[inline(always)]
  fn norm(&self) -> N
  { self.sqnorm().sqrt() }

  #[inline(always)]
  fn normalized(&self) -> Vec2<N>
  {
    let l = self.norm();

    Vec2::new(self.x / l, self.y / l)
  }

  #[inline(always)]
  fn normalize(&mut self) -> N
  {
    let l = self.norm();

    self.x /= l;
    self.y /= l;

    l
  }
}

impl<N:Copy + Mul<N, N> + Sub<N, N>> Cross<Vec1<N>> for Vec2<N>
{
  #[inline(always)]
  fn cross(&self, other : &Vec2<N>) -> Vec1<N>
  { Vec1::new(self.x * other.y - self.y * other.x) }
}

impl<N:Copy + Neg<N>> Neg<Vec2<N>> for Vec2<N>
{
  #[inline(always)]
  fn neg(&self) -> Vec2<N>
  { Vec2::new(-self.x, -self.y) }
}

impl<N:Copy + Zero> Zero for Vec2<N>
{
  #[inline(always)]
  fn zero() -> Vec2<N>
  {
    let _0 = Zero::zero();
    Vec2::new(_0, _0)
  }

  #[inline(always)]
  fn is_zero(&self) -> bool
  { self.x.is_zero() && self.y.is_zero() }
}

impl<N: Copy + One + Zero + Neg<N>> Basis for Vec2<N>
{
  #[inline(always)]
  fn canonical_basis()     -> ~[Vec2<N>]
  {
    // FIXME: this should be static
    ~[ Vec2::new(One::one(), Zero::zero()),
       Vec2::new(Zero::zero(), One::one()) ]
  }

  #[inline(always)]
  fn orthogonal_subspace_basis(&self) -> ~[Vec2<N>]
  { ~[ Vec2::new(-self.y, self.x) ] }
}

impl<N:ApproxEq<N>> ApproxEq<N> for Vec2<N>
{
  #[inline(always)]
  fn approx_epsilon() -> N
  { ApproxEq::approx_epsilon::<N, N>() }

  #[inline(always)]
  fn approx_eq(&self, other: &Vec2<N>) -> bool
  { self.x.approx_eq(&other.x) && self.y.approx_eq(&other.y) }

  #[inline(always)]
  fn approx_eq_eps(&self, other: &Vec2<N>, epsilon: &N) -> bool
  {
    self.x.approx_eq_eps(&other.x, epsilon) &&
    self.y.approx_eq_eps(&other.y, epsilon)
  }
}

impl<N:Rand + Copy> Rand for Vec2<N>
{
  #[inline(always)]
  fn rand<R: Rng>(rng: &mut R) -> Vec2<N>
  { Vec2::new(rng.gen(), rng.gen()) }
}

impl<N: Copy> Flatten<N> for Vec2<N>
{
  #[inline(always)]
  fn flat_size() -> uint
  { 2 }

  #[inline(always)]
  fn from_flattened(l: &[N], off: uint) -> Vec2<N>
  { Vec2::new(l[off], l[off + 1]) }

  #[inline(always)]
  fn flatten(&self) -> ~[N]
  { ~[ self.x, self.y ] }

  #[inline(always)]
  fn flatten_to(&self, l: &mut [N], off: uint)
  {
    l[off]     = self.x;
    l[off + 1] = self.y;
  }
}
