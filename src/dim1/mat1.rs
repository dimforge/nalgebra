use core::num::{One, Zero};
use std::cmp::FuzzyEq;
use traits::dim::Dim;
use traits::inv::Inv;
use traits::transpose::Transpose;
use traits::workarounds::rlmul::{RMul, LMul};
use dim1::vec1::Vec1;

#[deriving(Eq)]
pub struct Mat1<T>
{ m11: T }

pub fn Mat1<T:Copy>(m11: T) -> Mat1<T>
{
  Mat1
  { m11: m11 }
}

impl<T> Dim for Mat1<T>
{
  fn dim() -> uint
  { 1 }
}

impl<T:Copy + One> One for Mat1<T>
{
  fn one() -> Mat1<T>
  { return Mat1(One::one()) }
}

impl<T:Copy + Zero> Zero for Mat1<T>
{
  fn zero() -> Mat1<T>
  { Mat1(Zero::zero()) }

  fn is_zero(&self) -> bool
  { self.m11.is_zero() }
}

impl<T:Copy + Mul<T, T> + Add<T, T>> Mul<Mat1<T>, Mat1<T>> for Mat1<T>
{
  fn mul(&self, other: &Mat1<T>) -> Mat1<T>
  { Mat1 (self.m11 * other.m11) }
}

impl<T:Copy + Add<T, T> + Mul<T, T>> RMul<Vec1<T>> for Mat1<T>
{
  fn rmul(&self, other: &Vec1<T>) -> Vec1<T>
  { Vec1(self.m11 * other.x) }
}

impl<T:Copy + Add<T, T> + Mul<T, T>> LMul<Vec1<T>> for Mat1<T>
{
  fn lmul(&self, other: &Vec1<T>) -> Vec1<T>
  { Vec1(self.m11 * other.x) }
}

impl<T:Copy + Mul<T, T> + Quot<T, T> + Sub<T, T> + Neg<T> + Zero + One>
Inv for Mat1<T>
{
  fn inverse(&self) -> Mat1<T>
  {
    let mut res : Mat1<T> = *self;

    res.invert();

    res
  }

  fn invert(&mut self)
  {
    assert!(!self.m11.is_zero());

    self.m11 = One::one::<T>() / self.m11
  }
}

impl<T:Copy> Transpose for Mat1<T>
{
  fn transposed(&self) -> Mat1<T>
  { *self }

  fn transpose(&mut self)
  { }
}

impl<T:FuzzyEq<T>> FuzzyEq<T> for Mat1<T>
{
  fn fuzzy_eq(&self, other: &Mat1<T>) -> bool
  { self.m11.fuzzy_eq(&other.m11) }

  fn fuzzy_eq_eps(&self, other: &Mat1<T>, epsilon: &T) -> bool
  { self.m11.fuzzy_eq_eps(&other.m11, epsilon) }
}

impl<T:ToStr> ToStr for Mat1<T>
{
  fn to_str(&self) -> ~str
  {
    ~"Mat1 {"
    + " m11: " + self.m11.to_str()
    + " }"
  }
}
