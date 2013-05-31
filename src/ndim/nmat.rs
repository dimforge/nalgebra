use core::num::{One, Zero};
use core::rand::{Rand, Rng, RngUtil};
use core::cmp::ApproxEq;
use traits::dim::Dim;
use traits::inv::Inv;
use traits::transpose::Transpose;
use traits::workarounds::rlmul::{RMul, LMul};
use ndim::dmat::{DMat, one_mat_with_dim, zero_mat_with_dim, is_zero_mat};
use ndim::nvec::NVec;

// D is a phantom type parameter, used only as a dimensional token.
// Its allows use to encode the vector dimension at the type-level.
// It can be anything implementing the Dim trait. However, to avoid confusion,
// using d0, d1, d2, d3 and d4 tokens are prefered.
#[deriving(Eq, ToStr)]
pub struct NMat<D, T>
{ mij: DMat<T> }

impl<D: Dim, T: Copy> NMat<D, T>
{
  fn offset(i: uint, j: uint) -> uint
  { i * Dim::dim::<D>() + j }

  fn set(&mut self, i: uint, j: uint, t: &T)
  { self.mij.set(i, j, t) }
}

impl<D: Dim, T> Dim for NMat<D, T>
{
  fn dim() -> uint
  { Dim::dim::<D>() }
}

impl<D: Dim, T: Copy> Index<(uint, uint), T> for NMat<D, T>
{
  fn index(&self, &idx: &(uint, uint)) -> T
  { self.mij[idx] }
}

impl<D: Dim, T: Copy + One + Zero> One for NMat<D, T>
{
  fn one() -> NMat<D, T>
  { NMat { mij: one_mat_with_dim(Dim::dim::<D>()) } }
}

impl<D: Dim, T: Copy + Zero> Zero for NMat<D, T>
{
  fn zero() -> NMat<D, T>
  { NMat { mij: zero_mat_with_dim(Dim::dim::<D>()) } }

  fn is_zero(&self) -> bool
  { is_zero_mat(&self.mij) }
}

impl<D: Dim, T: Copy + Mul<T, T> + Add<T, T> + Zero>
Mul<NMat<D, T>, NMat<D, T>> for NMat<D, T>
{
  fn mul(&self, other: &NMat<D, T>) -> NMat<D, T>
  {
    let     dim = Dim::dim::<D>();
    let mut res = Zero::zero::<NMat<D, T>>();

    for uint::range(0u, dim) |i|
    {
      for uint::range(0u, dim) |j|
      {
        let mut acc: T = Zero::zero();

        for uint::range(0u, dim) |k|
        { acc += self[(i, k)] * other[(k, j)]; }

        res.set(i, j, &acc);
      }
    }

    res
  }
}

impl<D: Dim, T: Copy + Add<T, T> + Mul<T, T> + Zero>
RMul<NVec<D, T>> for NMat<D, T>
{
  fn rmul(&self, other: &NVec<D, T>) -> NVec<D, T>
  {
    let     dim              = Dim::dim::<D>();
    let mut res : NVec<D, T> = Zero::zero();

    for uint::range(0u, dim) |i|
    {
      for uint::range(0u, dim) |j|
      { res.at.at[i] = res.at.at[i] + other.at.at[j] * self[(i, j)]; }
    }

    res
  }
}

impl<D: Dim, T: Copy + Add<T, T> + Mul<T, T> + Zero>
LMul<NVec<D, T>> for NMat<D, T>
{
  fn lmul(&self, other: &NVec<D, T>) -> NVec<D, T>
  {
    let     dim              = Dim::dim::<D>();
    let mut res : NVec<D, T> = Zero::zero();

    for uint::range(0u, dim) |i|
    {
      for uint::range(0u, dim) |j|
      { res.at.at[i] = res.at.at[i] + other.at.at[j] * self[(j, i)]; }
    }

    res
  }
}

impl<D: Dim,
     T: Clone + Copy + Eq + One + Zero +
        Mul<T, T> + Div<T, T> + Sub<T, T> + Neg<T>>
Inv for NMat<D, T>
{
  fn inverse(&self) -> NMat<D, T>
  { NMat { mij: self.mij.inverse() } }

  fn invert(&mut self)
  { self.mij.invert() }
}

impl<D: Dim, T:Copy> Transpose for NMat<D, T>
{
  fn transposed(&self) -> NMat<D, T>
  {
    let mut res = copy *self;

    res.transpose();

    res
  }

  fn transpose(&mut self)
  { self.mij.transpose() }
}

impl<D, T: ApproxEq<T>> ApproxEq<T> for NMat<D, T>
{
  fn approx_epsilon() -> T
  { ApproxEq::approx_epsilon::<T, T>() }

  fn approx_eq(&self, other: &NMat<D, T>) -> bool
  { self.mij.approx_eq(&other.mij) }

  fn approx_eq_eps(&self, other: &NMat<D, T>, epsilon: &T) -> bool
  { self.mij.approx_eq_eps(&other.mij, epsilon) }
}

impl<D: Dim, T: Rand + Zero + Copy> Rand for NMat<D, T>
{
  fn rand<R: Rng>(rng: &mut R) -> NMat<D, T>
  {
    let     dim = Dim::dim::<D>();
    let mut res : NMat<D, T> = Zero::zero();

    for uint::range(0u, dim) |i|
    {
      for uint::range(0u, dim) |j|
      { res.set(i, j, &rng.gen()); }
    }

    res
  }
}
