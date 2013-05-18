use core::num::{One, Zero};
use core::rand::{Rand, Rng, RngUtil};
use std::cmp::FuzzyEq;
use traits::dim::Dim;
use traits::inv::Inv;
use traits::transpose::Transpose;
use traits::workarounds::rlmul::{RMul, LMul};
use dim3::vec3::Vec3;

#[deriving(Eq)]
pub struct Mat3<T>
{
    m11: T, m12: T, m13: T,
    m21: T, m22: T, m23: T,
    m31: T, m32: T, m33: T
}

pub fn Mat3<T:Copy>(m11: T, m12: T, m13: T,
                    m21: T, m22: T, m23: T,
                    m31: T, m32: T, m33: T) -> Mat3<T>
{
  Mat3
  {
    m11: m11, m12: m12, m13: m13,
    m21: m21, m22: m22, m23: m23,
    m31: m31, m32: m32, m33: m33
  }
}

impl<T> Dim for Mat3<T>
{
  fn dim() -> uint
  { 3 }
}

impl<T:Copy + One + Zero> One for Mat3<T>
{
  fn one() -> Mat3<T>
  {
    let (_0, _1) = (Zero::zero(), One::one());
    return Mat3(_1, _0, _0,
                _0, _1, _0,
                _0, _0, _1)
  }
}

impl<T:Copy + Zero> Zero for Mat3<T>
{
  fn zero() -> Mat3<T>
  {
    let _0 = Zero::zero();
    return Mat3(_0, _0, _0,
                _0, _0, _0,
                _0, _0, _0)
  }

  fn is_zero(&self) -> bool
  {
    self.m11.is_zero() && self.m12.is_zero() && self.m13.is_zero() &&
    self.m21.is_zero() && self.m22.is_zero() && self.m23.is_zero() &&
    self.m31.is_zero() && self.m32.is_zero() && self.m33.is_zero()
  }
}

impl<T:Copy + Mul<T, T> + Add<T, T>> Mul<Mat3<T>, Mat3<T>> for Mat3<T>
{
  fn mul(&self, other: &Mat3<T>) -> Mat3<T>
  {
    Mat3(
      self.m11 * other.m11  + self.m12 * other.m21 + self.m13 * other.m31,
      self.m11 * other.m12  + self.m12 * other.m22 + self.m13 * other.m32,
      self.m11 * other.m13  + self.m12 * other.m23 + self.m13 * other.m33,

      self.m21 * other.m11  + self.m22 * other.m21 + self.m23 * other.m31,
      self.m21 * other.m12  + self.m22 * other.m22 + self.m23 * other.m32,
      self.m21 * other.m13  + self.m22 * other.m23 + self.m23 * other.m33,

      self.m31 * other.m11  + self.m32 * other.m21 + self.m33 * other.m31,
      self.m31 * other.m12  + self.m32 * other.m22 + self.m33 * other.m32,
      self.m31 * other.m13  + self.m32 * other.m23 + self.m33 * other.m33
    )
  }
}

impl<T:Copy + Add<T, T> + Mul<T, T>> RMul<Vec3<T>> for Mat3<T>
{
  fn rmul(&self, other: &Vec3<T>) -> Vec3<T>
  {
    Vec3(
      self.m11 * other.x + self.m12 * other.y + self.m13 * other.z,
      self.m21 * other.x + self.m22 * other.y + self.m33 * other.z,
      self.m31 * other.x + self.m32 * other.y + self.m33 * other.z
    )
  }
}

impl<T:Copy + Add<T, T> + Mul<T, T>> LMul<Vec3<T>> for Mat3<T>
{
  fn lmul(&self, other: &Vec3<T>) -> Vec3<T>
  {
    Vec3(
      self.m11 * other.x + self.m21 * other.y + self.m31 * other.z,
      self.m12 * other.x + self.m22 * other.y + self.m32 * other.z,
      self.m13 * other.x + self.m23 * other.y + self.m33 * other.z
    )
  }
}

impl<T:Copy + Mul<T, T> + Quot<T, T> + Sub<T, T> + Add<T, T> + Neg<T> + Zero>
Inv for Mat3<T>
{
  fn inverse(&self) -> Mat3<T>
  {
    let mut res = *self;

    res.invert();

    res
  }

  fn invert(&mut self)
  {
    let minor_m22_m33 = self.m22 * self.m33 - self.m32 * self.m23;
    let minor_m21_m33 = self.m21 * self.m33 - self.m31 * self.m23;
    let minor_m21_m32 = self.m21 * self.m32 - self.m31 * self.m22;

    let det = self.m11 * minor_m22_m33
              - self.m12 * minor_m21_m33
              + self.m13 * minor_m21_m32;

    assert!(!det.is_zero());

    *self = Mat3(
      (minor_m22_m33  / det),
      ((self.m13 * self.m32 - self.m33 * self.m12) / det),
      ((self.m12 * self.m23 - self.m22 * self.m13) / det),

      (-minor_m21_m33 / det),
      ((self.m11 * self.m33 - self.m31 * self.m13) / det),
      ((self.m13 * self.m21 - self.m23 * self.m11) / det),

      (minor_m21_m32  / det),
      ((self.m12 * self.m31 - self.m32 * self.m11) / det),
      ((self.m11 * self.m22 - self.m21 * self.m12) / det)
    )
  }
}

impl<T:Copy> Transpose for Mat3<T>
{
  fn transposed(&self) -> Mat3<T>
  {
    Mat3(self.m11, self.m21, self.m31,
         self.m12, self.m22, self.m32,
         self.m13, self.m23, self.m33)
  }

  fn transpose(&mut self)
  {
    self.m12 <-> self.m21;
    self.m13 <-> self.m31;
    self.m23 <-> self.m32;
  }
}

impl<T:FuzzyEq<T>> FuzzyEq<T> for Mat3<T>
{
  fn fuzzy_eq(&self, other: &Mat3<T>) -> bool
  {
    self.m11.fuzzy_eq(&other.m11) &&
    self.m12.fuzzy_eq(&other.m12) &&
    self.m13.fuzzy_eq(&other.m13) &&

    self.m21.fuzzy_eq(&other.m21) &&
    self.m22.fuzzy_eq(&other.m22) &&
    self.m23.fuzzy_eq(&other.m23) &&

    self.m31.fuzzy_eq(&other.m31) &&
    self.m32.fuzzy_eq(&other.m32) &&
    self.m33.fuzzy_eq(&other.m33)
  }

  fn fuzzy_eq_eps(&self, other: &Mat3<T>, epsilon: &T) -> bool
  {
    self.m11.fuzzy_eq_eps(&other.m11, epsilon) &&
    self.m12.fuzzy_eq_eps(&other.m12, epsilon) &&
    self.m13.fuzzy_eq_eps(&other.m13, epsilon) &&

    self.m21.fuzzy_eq_eps(&other.m21, epsilon) &&
    self.m22.fuzzy_eq_eps(&other.m22, epsilon) &&
    self.m23.fuzzy_eq_eps(&other.m23, epsilon) &&

    self.m31.fuzzy_eq_eps(&other.m31, epsilon) &&
    self.m32.fuzzy_eq_eps(&other.m32, epsilon) &&
    self.m33.fuzzy_eq_eps(&other.m33, epsilon)
  }
}

impl<T:Rand + Copy> Rand for Mat3<T>
{
  fn rand<R: Rng>(rng: &R) -> Mat3<T>
  {
    Mat3(rng.gen(), rng.gen(), rng.gen(),
         rng.gen(), rng.gen(), rng.gen(),
         rng.gen(), rng.gen(), rng.gen())
  }
}

impl<T:ToStr> ToStr for Mat3<T>
{
  fn to_str(&self) -> ~str
  {
    ~"Mat3 {"
    + " m11: " + self.m11.to_str()
    + " m12: " + self.m12.to_str()
    + " m13: " + self.m13.to_str()

    + " m21: " + self.m21.to_str()
    + " m22: " + self.m22.to_str()
    + " m23: " + self.m23.to_str()

    + " m31: " + self.m31.to_str()
    + " m32: " + self.m32.to_str()
    + " m33: " + self.m33.to_str()
    + " }"
  }
}
