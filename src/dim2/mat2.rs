use std::num::{One, Zero};
use std::rand::{Rand, Rng, RngUtil};
use std::cmp::ApproxEq;
use std::util::swap;
use traits::dim::Dim;
use traits::inv::Inv;
use traits::transpose::Transpose;
use traits::flatten::Flatten;
use traits::workarounds::rlmul::{RMul, LMul};
use dim2::vec2::Vec2;

#[deriving(Eq, ToStr)]
pub struct Mat2<N>
{
    m11: N, m12: N,
    m21: N, m22: N
}

impl<N: Copy> Mat2<N>
{
  pub fn new(m11: N, m12: N, m21: N, m22: N) -> Mat2<N>
  {
    Mat2
    {
      m11: m11, m12: m12,
      m21: m21, m22: m22,
    }
  }
}

impl<N> Dim for Mat2<N>
{
  fn dim() -> uint
  { 2 }
}

impl<N:Copy + One + Zero> One for Mat2<N>
{
  fn one() -> Mat2<N>
  {
    let (_0, _1) = (Zero::zero(), One::one());
    return Mat2::new(_1, _0,
                     _0, _1)
  }
}

impl<N:Copy + Zero> Zero for Mat2<N>
{
  fn zero() -> Mat2<N>
  {
    let _0 = Zero::zero();
    return Mat2::new(_0, _0,
                     _0, _0)
  }

  fn is_zero(&self) -> bool
  {
    self.m11.is_zero() && self.m12.is_zero() &&
    self.m21.is_zero() && self.m22.is_zero()
  }
}

impl<N:Copy + Mul<N, N> + Add<N, N>> Mul<Mat2<N>, Mat2<N>> for Mat2<N>
{
  fn mul(&self, other: &Mat2<N>) -> Mat2<N>
  {
    Mat2::new(
      self.m11 * other.m11 + self.m12 * other.m21,
      self.m11 * other.m12 + self.m12 * other.m22,
      self.m21 * other.m11 + self.m22 * other.m21,
      self.m21 * other.m12 + self.m22 * other.m22
    )
  }
}

impl<N:Copy + Add<N, N> + Mul<N, N>> RMul<Vec2<N>> for Mat2<N>
{
  fn rmul(&self, other: &Vec2<N>) -> Vec2<N>
  {
    Vec2::new(
      self.m11 * other.x + self.m12 * other.y,
      self.m21 * other.x + self.m22 * other.y
    )
  }
}

impl<N:Copy + Add<N, N> + Mul<N, N>> LMul<Vec2<N>> for Mat2<N>
{
  fn lmul(&self, other: &Vec2<N>) -> Vec2<N>
  {
    Vec2::new(
      self.m11 * other.x + self.m21 * other.y,
      self.m12 * other.x + self.m22 * other.y
    )
  }
}

impl<N:Copy + Mul<N, N> + Div<N, N> + Sub<N, N> + Neg<N> + Zero>
Inv for Mat2<N>
{
  fn inverse(&self) -> Mat2<N>
  {
    let mut res : Mat2<N> = *self;

    res.invert();

    res
  }

  fn invert(&mut self)
  {
    let det = self.m11 * self.m22 - self.m21 * self.m12;

    assert!(!det.is_zero());

    *self = Mat2::new(self.m22 / det , -self.m12 / det,
                      -self.m21 / det, self.m11 / det)
  }
}

impl<N:Copy> Transpose for Mat2<N>
{
  fn transposed(&self) -> Mat2<N>
  {
    Mat2::new(self.m11, self.m21,
              self.m12, self.m22)
  }

  fn transpose(&mut self)
  { swap(&mut self.m21, &mut self.m12); }
}

impl<N:ApproxEq<N>> ApproxEq<N> for Mat2<N>
{
  fn approx_epsilon() -> N
  { ApproxEq::approx_epsilon::<N, N>() }

  fn approx_eq(&self, other: &Mat2<N>) -> bool
  {
    self.m11.approx_eq(&other.m11) &&
    self.m12.approx_eq(&other.m12) &&

    self.m21.approx_eq(&other.m21) &&
    self.m22.approx_eq(&other.m22)
  }

  fn approx_eq_eps(&self, other: &Mat2<N>, epsilon: &N) -> bool
  {
    self.m11.approx_eq_eps(&other.m11, epsilon) &&
    self.m12.approx_eq_eps(&other.m12, epsilon) &&

    self.m21.approx_eq_eps(&other.m21, epsilon) &&
    self.m22.approx_eq_eps(&other.m22, epsilon)
  }
}

impl<N:Rand + Copy> Rand for Mat2<N>
{
  fn rand<R: Rng>(rng: &mut R) -> Mat2<N>
  { Mat2::new(rng.gen(), rng.gen(), rng.gen(), rng.gen()) }
}

impl<N: Copy> Flatten<N> for Mat2<N>
{
  fn flat_size() -> uint
  { 4 }

  fn from_flattened(l: &[N], off: uint) -> Mat2<N>
  { Mat2::new(l[off], l[off + 1], l[off + 2], l[off + 3]) }

  fn flatten(&self) -> ~[N]
  { ~[ self.m11, self.m12, self.m21, self.m22 ] }

  fn flatten_to(&self, l: &mut [N], off: uint)
  {
    l[off]     = self.m11;
    l[off + 1] = self.m12;
    l[off + 2] = self.m21;
    l[off + 3] = self.m22;
  }
}
