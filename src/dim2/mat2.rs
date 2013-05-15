use core::num::{One, Zero};
use traits::dim::Dim;
use traits::inv::Inv;
use traits::transpose::Transpose;
use dim2::vec2::Vec2;

#[deriving(Eq)]
pub struct Mat2<T>
{
    m11: T, m12: T,
    m21: T, m22: T
}

pub fn Mat2<T:Copy>(m11: T, m12: T, m21: T, m22: T) -> Mat2<T>
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
    return Mat2(_1, _0,
                _0, _1)
  }
}

impl<T:Copy + Zero> Zero for Mat2<T>
{
  fn zero() -> Mat2<T>
  {
    let _0 = Zero::zero();
    return Mat2(_0, _0,
                _0, _0)
  }

  // fn is_zero(&self) -> bool
  // {
  //   self.m11.is_zero() && self.m11.is_zero() && self.m12.is_zero()
  //   && self.m21.is_zero() && self.m21.is_zero() && self.m22.is_zero()
  // }
}

impl<T:Copy + Mul<T, T> + Add<T, T>> Mul<Mat2<T>, Mat2<T>> for Mat2<T>
{
  fn mul(&self, other: &Mat2<T>) -> Mat2<T>
  {
    Mat2
    (self.m11 * other.m11 + self.m12 * other.m21,
     self.m11 * other.m12 + self.m12 * other.m22,
     self.m21 * other.m11 + self.m22 * other.m21,
     self.m21 * other.m22 + self.m22 * other.m22)
  }
}

// FIXME: implementation of multiple classes for the same struct fails
// with "internal compiler error: Asked to compute kind of a type variable".
//
// impl<T:Copy + Mul<T, T> + Add<T, T>> Mul<Vec2<T>, Vec2<T>> for Mat2<T>
// {
//   fn mul(&self, v: &Vec2<T>) -> Vec2<T>
//   { Vec2(self.m11 * v.x + self.m12 * v.y, self.m21 * v.x + self.m22 * v.y) }
// }

impl<T:Copy + Mul<T, T> + Add<T, T>> Mul<Mat2<T>, Vec2<T>> for Vec2<T>
{
  fn mul(&self, m: &Mat2<T>) -> Vec2<T>
  { Vec2(self.x * m.m11 + self.y * m.m21, self.x * m.m12 + self.y * m.m22) }
}

impl<T:Copy + Mul<T, T> + Div<T, T> + Sub<T, T> + Neg<T> + Eq + Zero>
Inv for Mat2<T>
{
  fn inv(&self) -> Mat2<T>
  {
    let det = self.m11 * self.m22 - self.m21 * self.m12;

    assert!(det != Zero::zero());

    Mat2(self.m22 / det , -self.m12 / det,
         -self.m21 / det, self.m11 / det)
  }
}

impl<T:Copy> Transpose for Mat2<T>
{
  fn transposed(&self) -> Mat2<T>
  {
    Mat2(self.m11, self.m21,
         self.m12, self.m22)
  }

  fn transpose(&mut self)
  { self.m21 <-> self.m12; }
}

impl<T:ToStr> ToStr for Mat2<T>
{
  fn to_str(&self) -> ~str
  {
    ~"Mat2 {"
    + " m11: " + self.m11.to_str()
    + " m12: " + self.m12.to_str()

    + " m21: " + self.m21.to_str()
    + " m22: " + self.m22.to_str()
    + " }"
  }
}
