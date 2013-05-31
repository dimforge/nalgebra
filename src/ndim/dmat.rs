use core::num::{One, Zero};
use core::vec::{from_elem, swap, all, all2, len};
use core::cmp::ApproxEq;
use traits::inv::Inv;
use traits::transpose::Transpose;
use traits::workarounds::rlmul::{RMul, LMul};
use ndim::dvec::{DVec, zero_vec_with_dim};

#[deriving(Eq, ToStr, Clone)]
pub struct DMat<T>
{
  dim: uint, // FIXME: handle more than just square matrices
  mij: ~[T]
}

pub fn zero_mat_with_dim<T: Zero + Copy>(dim: uint) -> DMat<T>
{ DMat { dim: dim, mij: from_elem(dim * dim, Zero::zero()) } }

pub fn is_zero_mat<T: Zero>(mat: &DMat<T>) -> bool
{ all(mat.mij, |e| e.is_zero()) }

pub fn one_mat_with_dim<T: Copy + One + Zero>(dim: uint) -> DMat<T>
{
  let mut res = zero_mat_with_dim(dim);
  let     _1  = One::one::<T>();

  for uint::range(0u, dim) |i|
  { res.set(i, i, &_1); }

  res
}

impl<T: Copy> DMat<T>
{
  pub fn offset(&self, i: uint, j: uint) -> uint
  { i * self.dim + j }

  pub fn set(&mut self, i: uint, j: uint, t: &T)
  {
    assert!(i < self.dim);
    assert!(j < self.dim);
    self.mij[self.offset(i, j)] = *t
  }
}

impl<T: Copy> Index<(uint, uint), T> for DMat<T>
{
  fn index(&self, &(i, j): &(uint, uint)) -> T
  { self.mij[self.offset(i, j)] }
}

impl<T: Copy + Mul<T, T> + Add<T, T> + Zero>
Mul<DMat<T>, DMat<T>> for DMat<T>
{
  fn mul(&self, other: &DMat<T>) -> DMat<T>
  {
    assert!(self.dim == other.dim);

    let     dim = self.dim;
    let mut res = zero_mat_with_dim(dim);

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

impl<T: Copy + Add<T, T> + Mul<T, T> + Zero>
RMul<DVec<T>> for DMat<T>
{
  fn rmul(&self, other: &DVec<T>) -> DVec<T>
  {
    assert!(self.dim == len(other.at));

    let     dim           = self.dim;
    let mut res : DVec<T> = zero_vec_with_dim(dim);

    for uint::range(0u, dim) |i|
    {
      for uint::range(0u, dim) |j|
      { res.at[i] = res.at[i] + other.at[j] * self[(i, j)]; }
    }

    res
  }
}

impl<T: Copy + Add<T, T> + Mul<T, T> + Zero>
LMul<DVec<T>> for DMat<T>
{
  fn lmul(&self, other: &DVec<T>) -> DVec<T>
  {
    assert!(self.dim == len(other.at));

    let     dim           = self.dim;
    let mut res : DVec<T> = zero_vec_with_dim(dim);

    for uint::range(0u, dim) |i|
    {
      for uint::range(0u, dim) |j|
      { res.at[i] = res.at[i] + other.at[j] * self[(j, i)]; }
    }

    res
  }
}

impl<T: Clone + Copy + Eq + One + Zero +
        Mul<T, T> + Div<T, T> + Sub<T, T> + Neg<T>>
Inv for DMat<T>
{
  fn inverse(&self) -> DMat<T>
  {
    let mut res : DMat<T> = self.clone();

    res.invert();

    res
  }

  fn invert(&mut self)
  {
    let     dim = self.dim;
    let mut res = one_mat_with_dim::<T>(dim);
    let     _0T = Zero::zero::<T>();

    // inversion using Gauss-Jordan elimination
    for uint::range(0u, dim) |k|
    {
      // search a non-zero value on the k-th column
      // FIXME: would it be worth it to spend some more time searching for the
      // max instead?

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
          let off_n0_j = self.offset(n0, j);
          let off_k_j  = self.offset(k, j);

          swap(self.mij, off_n0_j, off_k_j);
          swap(res.mij,  off_n0_j, off_k_j);
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

impl<T:Copy> Transpose for DMat<T>
{
  fn transposed(&self) -> DMat<T>
  {
    let mut res = copy *self;

    res.transpose();

    res
  }

  fn transpose(&mut self)
  {
    let dim = self.dim;

    for uint::range(1u, dim) |i|
    {
      for uint::range(0u, dim - 1) |j|
      {
        let off_i_j = self.offset(i, j);
        let off_j_i = self.offset(j, i);

        swap(self.mij, off_i_j, off_j_i);
      }
    }
  }
}

impl<T: ApproxEq<T>> ApproxEq<T> for DMat<T>
{
  fn approx_epsilon() -> T
  { ApproxEq::approx_epsilon::<T, T>() }

  fn approx_eq(&self, other: &DMat<T>) -> bool
  { all2(self.mij, other.mij, |a, b| a.approx_eq(b)) }

  fn approx_eq_eps(&self, other: &DMat<T>, epsilon: &T) -> bool
  { all2(self.mij, other.mij, |a, b| a.approx_eq_eps(b, epsilon)) }
}
