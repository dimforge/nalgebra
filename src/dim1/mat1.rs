use std::num::{One, Zero};
use std::rand::{Rand, Rng, RngUtil};
use std::cmp::ApproxEq;
use traits::dim::Dim;
use traits::inv::Inv;
use traits::transpose::Transpose;
use traits::flatten::Flatten;
use traits::rlmul::{RMul, LMul};
use dim1::vec1::Vec1;

#[deriving(Eq, ToStr)]
pub struct Mat1<N>
{ m11: N }

impl<N> Mat1<N>
{
  #[inline(always)]
  pub fn new(m11: N) -> Mat1<N>
  {
    Mat1
    { m11: m11 }
  }
}

impl<N> Dim for Mat1<N>
{
  #[inline(always)]
  fn dim() -> uint
  { 1 }
}

impl<N: One> One for Mat1<N>
{
  #[inline(always)]
  fn one() -> Mat1<N>
  { return Mat1::new(One::one()) }
}

impl<N: Zero> Zero for Mat1<N>
{
  #[inline(always)]
  fn zero() -> Mat1<N>
  { Mat1::new(Zero::zero()) }

  #[inline(always)]
  fn is_zero(&self) -> bool
  { self.m11.is_zero() }
}

impl<N: Mul<N, N> + Add<N, N>> Mul<Mat1<N>, Mat1<N>> for Mat1<N>
{
  #[inline(always)]
  fn mul(&self, other: &Mat1<N>) -> Mat1<N>
  { Mat1::new(self.m11 * other.m11) }
}

impl<N: Add<N, N> + Mul<N, N>> RMul<Vec1<N>> for Mat1<N>
{
  #[inline(always)]
  fn rmul(&self, other: &Vec1<N>) -> Vec1<N>
  { Vec1::new(self.m11 * other.x) }
}

impl<N: Add<N, N> + Mul<N, N>> LMul<Vec1<N>> for Mat1<N>
{
  #[inline(always)]
  fn lmul(&self, other: &Vec1<N>) -> Vec1<N>
  { Vec1::new(self.m11 * other.x) }
}

impl<N:Copy + Mul<N, N> + Div<N, N> + Sub<N, N> + Neg<N> + Zero + One>
Inv for Mat1<N>
{
  #[inline(always)]
  fn inverse(&self) -> Mat1<N>
  {
    let mut res : Mat1<N> = copy *self;

    res.invert();

    res
  }

  #[inline(always)]
  fn invert(&mut self)
  {
    assert!(!self.m11.is_zero());

    self.m11 = One::one::<N>() / self.m11
  }
}

impl<N: Copy> Transpose for Mat1<N>
{
  #[inline(always)]
  fn transposed(&self) -> Mat1<N>
  { copy *self }

  #[inline(always)]
  fn transpose(&mut self)
  { }
}

impl<N:ApproxEq<N>> ApproxEq<N> for Mat1<N>
{
  #[inline(always)]
  fn approx_epsilon() -> N
  { ApproxEq::approx_epsilon::<N, N>() }

  #[inline(always)]
  fn approx_eq(&self, other: &Mat1<N>) -> bool
  { self.m11.approx_eq(&other.m11) }

  #[inline(always)]
  fn approx_eq_eps(&self, other: &Mat1<N>, epsilon: &N) -> bool
  { self.m11.approx_eq_eps(&other.m11, epsilon) }
}

impl<N: Rand > Rand for Mat1<N>
{
  #[inline(always)]
  fn rand<R: Rng>(rng: &mut R) -> Mat1<N>
  { Mat1::new(rng.gen()) }
}

impl<N: Copy> Flatten<N> for Mat1<N>
{
  #[inline(always)]
  fn flat_size() -> uint
  { 1 }

  #[inline(always)]
  fn from_flattened(l: &[N], off: uint) -> Mat1<N>
  { Mat1::new(copy l[off]) }

  #[inline(always)]
  fn flatten(&self) -> ~[N]
  { ~[ copy self.m11 ] }

  #[inline(always)]
  fn flatten_to(&self, l: &mut [N], off: uint)
  { l[off] = copy self.m11 }
}
