use std::uint::iterate;
use std::num::{One, Zero};
use std::rand::{Rand, Rng, RngUtil};
use std::cmp::ApproxEq;
use traits::dim::Dim;
use traits::inv::Inv;
use traits::division_ring::DivisionRing;
use traits::transpose::Transpose;
use traits::flatten::Flatten;
use traits::rlmul::{RMul, LMul};
use ndim::dmat::{DMat, one_mat_with_dim, zero_mat_with_dim, is_zero_mat};
use ndim::nvec::NVec;

// D is a phantom type parameter, used only as a dimensional token.
// Its allows use to encode the vector dimension at the type-level.
// It can be anything implementing the Dim trait. However, to avoid confusion,
// using d0, d1, d2, d3 and d4 tokens are prefered.
#[deriving(Eq, ToStr)]
pub struct NMat<D, N>
{ mij: DMat<N> }

impl<D: Dim, N: Copy> NMat<D, N>
{
  #[inline]
  fn offset(i: uint, j: uint) -> uint
  { i * Dim::dim::<D>() + j }

  #[inline]
  fn set(&mut self, i: uint, j: uint, t: &N)
  { self.mij.set(i, j, t) }
}

impl<D: Dim, N> Dim for NMat<D, N>
{
  #[inline]
  fn dim() -> uint
  { Dim::dim::<D>() }
}

impl<D: Dim, N: Copy> Index<(uint, uint), N> for NMat<D, N>
{
  #[inline]
  fn index(&self, &idx: &(uint, uint)) -> N
  { self.mij[idx] }
}

impl<D: Dim, N: Copy + One + Zero> One for NMat<D, N>
{
  #[inline]
  fn one() -> NMat<D, N>
  { NMat { mij: one_mat_with_dim(Dim::dim::<D>()) } }
}

impl<D: Dim, N: Copy + Zero> Zero for NMat<D, N>
{
  #[inline]
  fn zero() -> NMat<D, N>
  { NMat { mij: zero_mat_with_dim(Dim::dim::<D>()) } }

  #[inline]
  fn is_zero(&self) -> bool
  { is_zero_mat(&self.mij) }
}

impl<D: Dim, N: Copy + Mul<N, N> + Add<N, N> + Zero>
Mul<NMat<D, N>, NMat<D, N>> for NMat<D, N>
{
  fn mul(&self, other: &NMat<D, N>) -> NMat<D, N>
  {
    let     dim = Dim::dim::<D>();
    let mut res = Zero::zero::<NMat<D, N>>();

    for iterate(0u, dim) |i|
    {
      for iterate(0u, dim) |j|
      {
        let mut acc: N = Zero::zero();

        for iterate(0u, dim) |k|
        { acc = acc + self[(i, k)] * other[(k, j)]; }

        res.set(i, j, &acc);
      }
    }

    res
  }
}

impl<D: Dim, N: Copy + Add<N, N> + Mul<N, N> + Zero>
RMul<NVec<D, N>> for NMat<D, N>
{
  fn rmul(&self, other: &NVec<D, N>) -> NVec<D, N>
  {
    let     dim              = Dim::dim::<D>();
    let mut res : NVec<D, N> = Zero::zero();

    for iterate(0u, dim) |i|
    {
      for iterate(0u, dim) |j|
      { res.at.at[i] = res.at.at[i] + other.at.at[j] * self[(i, j)]; }
    }

    res
  }
}

impl<D: Dim, N: Copy + Add<N, N> + Mul<N, N> + Zero>
LMul<NVec<D, N>> for NMat<D, N>
{
  fn lmul(&self, other: &NVec<D, N>) -> NVec<D, N>
  {
    let     dim              = Dim::dim::<D>();
    let mut res : NVec<D, N> = Zero::zero();

    for iterate(0u, dim) |i|
    {
      for iterate(0u, dim) |j|
      { res.at.at[i] = res.at.at[i] + other.at.at[j] * self[(j, i)]; }
    }

    res
  }
}

impl<D: Dim, N: Clone + Copy + Eq + DivisionRing>
Inv for NMat<D, N>
{
  #[inline]
  fn inverse(&self) -> NMat<D, N>
  { NMat { mij: self.mij.inverse() } }

  #[inline]
  fn invert(&mut self)
  { self.mij.invert() }
}

impl<D: Dim, N:Copy> Transpose for NMat<D, N>
{
  #[inline]
  fn transposed(&self) -> NMat<D, N>
  {
    let mut res = copy *self;

    res.transpose();

    res
  }

  #[inline]
  fn transpose(&mut self)
  { self.mij.transpose() }
}

impl<D, N: ApproxEq<N>> ApproxEq<N> for NMat<D, N>
{
  #[inline]
  fn approx_epsilon() -> N
  { ApproxEq::approx_epsilon::<N, N>() }

  #[inline]
  fn approx_eq(&self, other: &NMat<D, N>) -> bool
  { self.mij.approx_eq(&other.mij) }

  #[inline]
  fn approx_eq_eps(&self, other: &NMat<D, N>, epsilon: &N) -> bool
  { self.mij.approx_eq_eps(&other.mij, epsilon) }
}

impl<D: Dim, N: Rand + Zero + Copy> Rand for NMat<D, N>
{
  fn rand<R: Rng>(rng: &mut R) -> NMat<D, N>
  {
    let     dim = Dim::dim::<D>();
    let mut res : NMat<D, N> = Zero::zero();

    for iterate(0u, dim) |i|
    {
      for iterate(0u, dim) |j|
      { res.set(i, j, &rng.gen()); }
    }

    res
  }
}

impl<D: Dim, N: Zero + Copy> Flatten<N> for NMat<D, N>
{
  #[inline]
  fn flat_size() -> uint
  { Dim::dim::<D>() * Dim::dim::<D>() }

  #[inline]
  fn from_flattened(l: &[N], off: uint) -> NMat<D, N>
  {
    let     dim = Dim::dim::<D>();
    let mut res = Zero::zero::<NMat<D, N>>();

    for iterate(0u, dim * dim) |i|
    { res.mij.mij[i] = copy l[off + i] }

    res
  }

  #[inline]
  fn flatten(&self) -> ~[N]
  {
    let     dim = Dim::dim::<D>();
    let mut res = ~[];

    for iterate(0u, dim * dim) |i|
    { res.push(copy self.mij.mij[i]) }

    res
  }

  #[inline]
  fn flatten_to(&self, l: &mut [N], off: uint)
  {
    let dim = Dim::dim::<D>();

    for iterate(0u, dim * dim) |i|
    { l[off + i] = copy self.mij.mij[i] }
  }
}
