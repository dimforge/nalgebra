use core::num::{One, Zero};
use core::rand::{Rand, Rng, RngUtil};
use core::cmp::ApproxEq;
use core::util::swap;
use traits::dim::Dim;
use traits::inv::Inv;
use traits::transpose::Transpose;
use traits::workarounds::rlmul::{RMul, LMul};
use dim2::vec2::{Vec2, vec2};

#[deriving(Eq, ToStr)]
pub struct Mat2<T>
{
    m11: T, m12: T,
    m21: T, m22: T
}

pub fn mat2<T:Copy>(m11: T, m12: T, m21: T, m22: T) -> Mat2<T>
{
  Mat2
  {
    m11: m11, m12: m12,
    m21: m21, m22: m22,
  }
}

impl<T> Dim for Mat2<T>
{
  fn dim() -> uint
  { 2 }
}

impl<T:Copy + One + Zero> One for Mat2<T>
{
  fn one() -> Mat2<T>
  {
    let (_0, _1) = (Zero::zero(), One::one());
    return mat2(_1, _0,
                _0, _1)
  }
}

impl<T:Copy + Zero> Zero for Mat2<T>
{
  fn zero() -> Mat2<T>
  {
    let _0 = Zero::zero();
    return mat2(_0, _0,
                _0, _0)
  }

  fn is_zero(&self) -> bool
  {
    self.m11.is_zero() && self.m12.is_zero() &&
    self.m21.is_zero() && self.m22.is_zero()
  }
}

impl<T:Copy + Mul<T, T> + Add<T, T>> Mul<Mat2<T>, Mat2<T>> for Mat2<T>
{
  fn mul(&self, other: &Mat2<T>) -> Mat2<T>
  {
    mat2
    (self.m11 * other.m11 + self.m12 * other.m21,
     self.m11 * other.m12 + self.m12 * other.m22,
     self.m21 * other.m11 + self.m22 * other.m21,
     self.m21 * other.m12 + self.m22 * other.m22)
  }
}

impl<T:Copy + Add<T, T> + Mul<T, T>> RMul<Vec2<T>> for Mat2<T>
{
  fn rmul(&self, other: &Vec2<T>) -> Vec2<T>
  {
    vec2(
      self.m11 * other.x + self.m12 * other.y,
      self.m21 * other.x + self.m22 * other.y
    )
  }
}

impl<T:Copy + Add<T, T> + Mul<T, T>> LMul<Vec2<T>> for Mat2<T>
{
  fn lmul(&self, other: &Vec2<T>) -> Vec2<T>
  {
    vec2(
      self.m11 * other.x + self.m21 * other.y,
      self.m12 * other.x + self.m22 * other.y
    )
  }
}

impl<T:Copy + Mul<T, T> + Div<T, T> + Sub<T, T> + Neg<T> + Zero>
Inv for Mat2<T>
{
  fn inverse(&self) -> Mat2<T>
  {
    let mut res : Mat2<T> = *self;

    res.invert();

    res
  }

  fn invert(&mut self)
  {
    let det = self.m11 * self.m22 - self.m21 * self.m12;

    assert!(!det.is_zero());

    *self = mat2(self.m22 / det , -self.m12 / det,
                 -self.m21 / det, self.m11 / det)
  }
}

impl<T:Copy> Transpose for Mat2<T>
{
  fn transposed(&self) -> Mat2<T>
  {
    mat2(self.m11, self.m21,
         self.m12, self.m22)
  }

  fn transpose(&mut self)
  { swap(&mut self.m21, &mut self.m12); }
}

impl<T:ApproxEq<T>> ApproxEq<T> for Mat2<T>
{
  fn approx_epsilon() -> T
  { ApproxEq::approx_epsilon::<T, T>() }

  fn approx_eq(&self, other: &Mat2<T>) -> bool
  {
    self.m11.approx_eq(&other.m11) &&
    self.m12.approx_eq(&other.m12) &&

    self.m21.approx_eq(&other.m21) &&
    self.m22.approx_eq(&other.m22)
  }

  fn approx_eq_eps(&self, other: &Mat2<T>, epsilon: &T) -> bool
  {
    self.m11.approx_eq_eps(&other.m11, epsilon) &&
    self.m12.approx_eq_eps(&other.m12, epsilon) &&

    self.m21.approx_eq_eps(&other.m21, epsilon) &&
    self.m22.approx_eq_eps(&other.m22, epsilon)
  }
}

impl<T:Rand + Copy> Rand for Mat2<T>
{
  fn rand<R: Rng>(rng: &mut R) -> Mat2<T>
  { mat2(rng.gen(), rng.gen(), rng.gen(), rng.gen()) }
}
