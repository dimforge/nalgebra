use core::num::{One, Zero};
use core::rand::{Rand, Rng, RngUtil};
use core::vec::{from_elem, swap, all, all2};
use core::cmp::ApproxEq;
use traits::dim::Dim;
use traits::inv::Inv;
use traits::transpose::Transpose;
use traits::workarounds::rlmul::{RMul, LMul};
use ndim::nvec::NVec;

// D is a phantom type parameter, used only as a dimensional token.
// Its allows use to encode the vector dimension at the type-level.
// It can be anything implementing the Dim trait. However, to avoid confusion,
// using d0, d1, d2, d3 and d4 tokens are prefered.
#[deriving(Eq, ToStr)]
pub struct NMat<D, T>
{
  mij: ~[T]
}

impl<D, T: Clone> Clone for NMat<D, T>
{
  fn clone(&self) -> NMat<D, T>
  { NMat{ mij: self.mij.clone() } }
}

impl<D: Dim, T: Copy> NMat<D, T>
{
  fn offset(i: uint, j: uint) -> uint
  { i * Dim::dim::<D>() + j }

  fn set(&mut self, i: uint, j: uint, t: &T)
  { self.mij[NMat::offset::<D, T>(i, j)] = *t }
}

impl<D: Dim, T> Dim for NMat<D, T>
{
  fn dim() -> uint
  { Dim::dim::<D>() }
}

impl<D: Dim, T: Copy> Index<(uint, uint), T> for NMat<D, T>
{
  fn index(&self, &(i, j): &(uint, uint)) -> T
  { self.mij[NMat::offset::<D, T>(i, j)] }
}

impl<D: Dim, T: Copy + One + Zero> One for NMat<D, T>
{
  fn one() -> NMat<D, T>
  {
    let     dim = Dim::dim::<D>();
    let mut res = NMat{ mij: from_elem(dim * dim, Zero::zero()) };
    let     _1  = One::one::<T>();

    for uint::range(0u, dim) |i|
    { res.set(i, i, &_1); }

    res
  }
}

impl<D: Dim, T: Copy + Zero> Zero for NMat<D, T>
{
  fn zero() -> NMat<D, T>
  {
    let dim = Dim::dim::<D>();

    NMat{ mij: from_elem(dim * dim, Zero::zero()) }
  }

  fn is_zero(&self) -> bool
  { all(self.mij, |e| e.is_zero()) }
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
      { res.at[i] = res.at[i] + other.at[j] * self[(i, j)]; }
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
      { res.at[i] = res.at[i] + other.at[j] * self[(j, i)]; }
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
  {
    let mut res : NMat<D, T> = self.clone();

    res.invert();

    res
  }

  fn invert(&mut self)
  {
    let     dim = Dim::dim::<D>();
    let mut res = One::one::<NMat<D, T>>();
    let     _0T = Zero::zero::<T>();

    // inversion using Gauss-Jordan elimination
    for uint::range(0u, dim) |k|
    {
      // search a non-zero value on the k-th column
      // FIXME: is it worth it to spend some more time searching for the max
      // instead?

      // FIXME: this is kind of uggly…
      // … but we cannot use position_between since we are iterating on one
      // columns
      let mut n0 = 0u; // index of a non-zero entry

      while (n0 != dim)
      {
        if (self[(n0, k)] != _0T)
        { break; }

        n0 += 1;
      }

      assert!(n0 != dim); // non inversible matrix

      // swap pivot line
      if (n0 != k)
      {
        for uint::range(0u, dim) |j|
        {
          swap(self.mij,
               NMat::offset::<D, T>(n0, j),
               NMat::offset::<D, T>(k, j));
          swap(res.mij,
               NMat::offset::<D, T>(n0, j),
               NMat::offset::<D, T>(k, j));
        }
      }

      let pivot = self[(k, k)];

      for uint::range(k, dim) |j|
      {
        // FIXME: not to putting selfal exression directly on the nuction call
        // is uggly but does not seem to compile any more…
        let selfval = &(self[(k, j)] / pivot);
        let resval  = &(res[(k, j)] / pivot);

        self.set(k, j, selfval);
        res.set(k, j, resval);
      }

      for uint::range(0u, dim) |l|
      {
        if (l != k)
        {
          let normalizer = self[(l, k)] / pivot;

          for uint::range(k, dim) |j|
          {
            let selfval = &(self[(l, j)] - self[(k, j)] * normalizer);
            let resval  = &(res[(l, j)] - res[(k, j)] * normalizer);

            self.set(k, j, selfval);
            res.set(k, j, resval);
          }
        }
      }
    }
  }
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
  {
    let dim = Dim::dim::<D>();

    for uint::range(1u, dim) |i|
    {
      for uint::range(0u, dim - 1) |j|
      {
        swap(self.mij,
             NMat::offset::<D, T>(i, j),
             NMat::offset::<D, T>(j, i));
      }
    }
  }
}

impl<D, T: ApproxEq<T>> ApproxEq<T> for NMat<D, T>
{
  fn approx_epsilon() -> T
  { ApproxEq::approx_epsilon::<T, T>() }

  fn approx_eq(&self, other: &NMat<D, T>) -> bool
  { all2(self.mij, other.mij, |a, b| a.approx_eq(b)) }

  fn approx_eq_eps(&self, other: &NMat<D, T>, epsilon: &T) -> bool
  { all2(self.mij, other.mij, |a, b| a.approx_eq_eps(b, epsilon)) }
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
