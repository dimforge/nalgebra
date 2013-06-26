use std::num::{One, Zero};
use std::rand::{Rand, Rng, RngUtil};
use std::cmp::ApproxEq;
use std::util::swap;
use traits::dim::Dim;
use traits::inv::Inv;
use traits::transpose::Transpose;
use traits::flatten::Flatten;
use traits::rlmul::{RMul, LMul};
use dim3::vec3::Vec3;

#[deriving(Eq, ToStr)]
pub struct Mat3<N>
{
    m11: N, m12: N, m13: N,
    m21: N, m22: N, m23: N,
    m31: N, m32: N, m33: N
}

impl<N> Mat3<N>
{
  #[inline(always)]
  pub fn new(m11: N, m12: N, m13: N,
             m21: N, m22: N, m23: N,
             m31: N, m32: N, m33: N) -> Mat3<N>
  {
    Mat3
    {
      m11: m11, m12: m12, m13: m13,
      m21: m21, m22: m22, m23: m23,
      m31: m31, m32: m32, m33: m33
    }
  }
}

impl<N> Dim for Mat3<N>
{
  #[inline(always)]
  fn dim() -> uint
  { 3 }
}

impl<N: Copy + One + Zero> One for Mat3<N>
{
  #[inline(always)]
  fn one() -> Mat3<N>
  {
    let (_0, _1) = (Zero::zero(), One::one());
    return Mat3::new(copy _1, copy _0, copy _0,
                     copy _0, copy _1, copy _0,
                     copy _0, _0,      _1)
  }
}

impl<N: Copy + Zero> Zero for Mat3<N>
{
  #[inline(always)]
  fn zero() -> Mat3<N>
  {
    let _0 = Zero::zero();
    return Mat3::new(copy _0, copy _0, copy _0,
                     copy _0, copy _0, copy _0,
                     copy _0, copy _0, _0)
  }

  #[inline(always)]
  fn is_zero(&self) -> bool
  {
    self.m11.is_zero() && self.m12.is_zero() && self.m13.is_zero() &&
    self.m21.is_zero() && self.m22.is_zero() && self.m23.is_zero() &&
    self.m31.is_zero() && self.m32.is_zero() && self.m33.is_zero()
  }
}

impl<N: Mul<N, N> + Add<N, N>> Mul<Mat3<N>, Mat3<N>> for Mat3<N>
{
  #[inline(always)]
  fn mul(&self, other: &Mat3<N>) -> Mat3<N>
  {
    Mat3::new(
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

impl<N: Add<N, N> + Mul<N, N>> RMul<Vec3<N>> for Mat3<N>
{
  #[inline(always)]
  fn rmul(&self, other: &Vec3<N>) -> Vec3<N>
  {
    Vec3::new(
      self.m11 * other.x + self.m12 * other.y + self.m13 * other.z,
      self.m21 * other.x + self.m22 * other.y + self.m33 * other.z,
      self.m31 * other.x + self.m32 * other.y + self.m33 * other.z
    )
  }
}

impl<N: Add<N, N> + Mul<N, N>> LMul<Vec3<N>> for Mat3<N>
{
  #[inline(always)]
  fn lmul(&self, other: &Vec3<N>) -> Vec3<N>
  {
    Vec3::new(
      self.m11 * other.x + self.m21 * other.y + self.m31 * other.z,
      self.m12 * other.x + self.m22 * other.y + self.m32 * other.z,
      self.m13 * other.x + self.m23 * other.y + self.m33 * other.z
    )
  }
}

impl<N:Copy + Mul<N, N> + Div<N, N> + Sub<N, N> + Add<N, N> + Neg<N> + Zero>
Inv for Mat3<N>
{
  #[inline(always)]
  fn inverse(&self) -> Mat3<N>
  {
    let mut res = copy *self;

    res.invert();

    res
  }

  #[inline(always)]
  fn invert(&mut self)
  {
    let minor_m22_m33 = self.m22 * self.m33 - self.m32 * self.m23;
    let minor_m21_m33 = self.m21 * self.m33 - self.m31 * self.m23;
    let minor_m21_m32 = self.m21 * self.m32 - self.m31 * self.m22;

    let det = self.m11 * minor_m22_m33
              - self.m12 * minor_m21_m33
              + self.m13 * minor_m21_m32;

    assert!(!det.is_zero());

    *self = Mat3::new(
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

impl<N:Copy> Transpose for Mat3<N>
{
  #[inline(always)]
  fn transposed(&self) -> Mat3<N>
  {
    Mat3::new(copy self.m11, copy self.m21, copy self.m31,
              copy self.m12, copy self.m22, copy self.m32,
              copy self.m13, copy self.m23, copy self.m33)
  }

  #[inline(always)]
  fn transpose(&mut self)
  {
    swap(&mut self.m12, &mut self.m21);
    swap(&mut self.m13, &mut self.m31);
    swap(&mut self.m23, &mut self.m32);
  }
}

impl<N:ApproxEq<N>> ApproxEq<N> for Mat3<N>
{
  #[inline(always)]
  fn approx_epsilon() -> N
  { ApproxEq::approx_epsilon::<N, N>() }

  #[inline(always)]
  fn approx_eq(&self, other: &Mat3<N>) -> bool
  {
    self.m11.approx_eq(&other.m11) &&
    self.m12.approx_eq(&other.m12) &&
    self.m13.approx_eq(&other.m13) &&

    self.m21.approx_eq(&other.m21) &&
    self.m22.approx_eq(&other.m22) &&
    self.m23.approx_eq(&other.m23) &&

    self.m31.approx_eq(&other.m31) &&
    self.m32.approx_eq(&other.m32) &&
    self.m33.approx_eq(&other.m33)
  }

  #[inline(always)]
  fn approx_eq_eps(&self, other: &Mat3<N>, epsilon: &N) -> bool
  {
    self.m11.approx_eq_eps(&other.m11, epsilon) &&
    self.m12.approx_eq_eps(&other.m12, epsilon) &&
    self.m13.approx_eq_eps(&other.m13, epsilon) &&

    self.m21.approx_eq_eps(&other.m21, epsilon) &&
    self.m22.approx_eq_eps(&other.m22, epsilon) &&
    self.m23.approx_eq_eps(&other.m23, epsilon) &&

    self.m31.approx_eq_eps(&other.m31, epsilon) &&
    self.m32.approx_eq_eps(&other.m32, epsilon) &&
    self.m33.approx_eq_eps(&other.m33, epsilon)
  }
}

impl<N: Rand> Rand for Mat3<N>
{
  #[inline(always)]
  fn rand<R: Rng>(rng: &mut R) -> Mat3<N>
  {
    Mat3::new(rng.gen(), rng.gen(), rng.gen(),
              rng.gen(), rng.gen(), rng.gen(),
              rng.gen(), rng.gen(), rng.gen())
  }
}

impl<N: Copy> Flatten<N> for Mat3<N>
{
  #[inline(always)]
  fn flat_size() -> uint
  { 9 }

  #[inline(always)]
  fn from_flattened(l: &[N], off: uint) -> Mat3<N>
  { Mat3::new(copy l[off + 0], copy l[off + 1], copy l[off + 2],
              copy l[off + 3], copy l[off + 4], copy l[off + 5],
              copy l[off + 6], copy l[off + 7], copy l[off + 8]) }

  #[inline(always)]
  fn flatten(&self) -> ~[N]
  {
    ~[
      copy self.m11, copy self.m12, copy self.m13,
      copy self.m21, copy self.m22, copy self.m23,
      copy self.m31, copy self.m32, copy self.m33
    ]
  }

  #[inline(always)]
  fn flatten_to(&self, l: &mut [N], off: uint)
  {
    l[off + 0] = copy self.m11;
    l[off + 1] = copy self.m12;
    l[off + 2] = copy self.m13;
    l[off + 3] = copy self.m21;
    l[off + 4] = copy self.m22;
    l[off + 5] = copy self.m23;
    l[off + 6] = copy self.m31;
    l[off + 7] = copy self.m32;
    l[off + 8] = copy self.m33;
  }
}
