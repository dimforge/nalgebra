use std::num::{One, Zero};
use std::rand::{Rand, Rng, RngUtil};
use std::cmp::ApproxEq;
use traits::dim::Dim;
use traits::inv::Inv;
use traits::transpose::Transpose;
use traits::workarounds::rlmul::{RMul, LMul};
use dim1::vec1::Vec1;

#[deriving(Eq, ToStr)]
pub struct Mat1<N>
{ m11: N }

impl<N: Copy> Mat1<N>
{
  pub fn new(m11: N) -> Mat1<N>
  {
    Mat1
    { m11: m11 }
  }
}

impl<N> Dim for Mat1<N>
{
  fn dim() -> uint
  { 1 }
}

impl<N:Copy + One> One for Mat1<N>
{
  fn one() -> Mat1<N>
  { return Mat1::new(One::one()) }
}

impl<N:Copy + Zero> Zero for Mat1<N>
{
  fn zero() -> Mat1<N>
  { Mat1::new(Zero::zero()) }

  fn is_zero(&self) -> bool
  { self.m11.is_zero() }
}

impl<N:Copy + Mul<N, N> + Add<N, N>> Mul<Mat1<N>, Mat1<N>> for Mat1<N>
{
  fn mul(&self, other: &Mat1<N>) -> Mat1<N>
  { Mat1::new(self.m11 * other.m11) }
}

impl<N:Copy + Add<N, N> + Mul<N, N>> RMul<Vec1<N>> for Mat1<N>
{
  fn rmul(&self, other: &Vec1<N>) -> Vec1<N>
  { Vec1::new(self.m11 * other.x) }
}

impl<N:Copy + Add<N, N> + Mul<N, N>> LMul<Vec1<N>> for Mat1<N>
{
  fn lmul(&self, other: &Vec1<N>) -> Vec1<N>
  { Vec1::new(self.m11 * other.x) }
}

impl<N:Copy + Mul<N, N> + Div<N, N> + Sub<N, N> + Neg<N> + Zero + One>
Inv for Mat1<N>
{
  fn inverse(&self) -> Mat1<N>
  {
    let mut res : Mat1<N> = *self;

    res.invert();

    res
  }

  fn invert(&mut self)
  {
    assert!(!self.m11.is_zero());

    self.m11 = One::one::<N>() / self.m11
  }
}

impl<N:Copy> Transpose for Mat1<N>
{
  fn transposed(&self) -> Mat1<N>
  { *self }

  fn transpose(&mut self)
  { }
}

impl<N:ApproxEq<N>> ApproxEq<N> for Mat1<N>
{
  fn approx_epsilon() -> N
  { ApproxEq::approx_epsilon::<N, N>() }

  fn approx_eq(&self, other: &Mat1<N>) -> bool
  { self.m11.approx_eq(&other.m11) }

  fn approx_eq_eps(&self, other: &Mat1<N>, epsilon: &N) -> bool
  { self.m11.approx_eq_eps(&other.m11, epsilon) }
}

impl<N:Rand + Copy> Rand for Mat1<N>
{
  fn rand<R: Rng>(rng: &mut R) -> Mat1<N>
  { Mat1::new(rng.gen()) }
}
