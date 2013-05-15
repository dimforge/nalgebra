use core::num::{One, Zero};
use core::vec::{from_elem, swap};
use traits::dim::Dim;
use traits::inv::Inv;
use traits::transpose::Transpose;
// use ndim::nvec::NVec;

// D is a phantom type parameter, used only as a dimensional token.
// Its allows use to encode the vector dimension at the type-level.
// It can be anything implementing the Dim trait. However, to avoid confusion,
// using d0, d1, d2, d3 and d4 tokens are prefered.
#[deriving(Eq)]
pub struct NMat<D, T>
{
  mij: ~[T]
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

impl<D: Dim, T:Copy> Index<(uint, uint), T> for NMat<D, T>
{
  fn index(&self, &(i, j): &(uint, uint)) -> T
  { self.mij[NMat::offset::<D, T>(i, j)] }
}

impl<D: Dim, T:Copy + One + Zero> One for NMat<D, T>
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

impl<D: Dim, T:Copy + Zero> Zero for NMat<D, T>
{
  fn zero() -> NMat<D, T>
  {
    let dim = Dim::dim::<D>();

    NMat{ mij: from_elem(dim * dim, Zero::zero()) }
  }
}

impl<D: Dim, T:Copy + Mul<T, T> + Add<T, T> + Zero>
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

impl<D: Dim,
     T: Copy + Mul<T, T> + Div<T, T> + Sub<T, T> + Neg<T> + Eq + One + Zero>
Inv for NMat<D, T>
{
  fn inv(&self) -> NMat<D, T>
  {
    let mut cpy = copy *self;
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
      // … but we cannot use position_betwee since we are iterating on one
      // columns
      let mut n0 = 0u; // index of a non-zero entry

      while (n0 != dim)
      {
        if (cpy[(n0, k)] != _0T)
        { break; }

        n0 += 1;
      }

      assert!(n0 != dim); // non inversible matrix

      // swap pivot line
      if (n0 != k)
      {
        for uint::range(0u, dim) |j|
        {
          swap(cpy.mij,
               NMat::offset::<D, T>(n0, j),
               NMat::offset::<D, T>(k, j));
          swap(res.mij,
               NMat::offset::<D, T>(n0, j),
               NMat::offset::<D, T>(k, j));
        }
      }

      let pivot = cpy[(k, k)];

      for uint::range(k, dim) |j|
      {
        cpy.set(k, j, &(cpy[(k, j)] / pivot));
        res.set(k, j, &(res[(k, j)] / pivot));
      }

      for uint::range(0u, dim) |l|
      {
        if (l != k)
        {
          let normalizer = cpy[(l, k)] / pivot;

          for uint::range(k, dim) |j|
          {
            cpy.set(k, j, &(cpy[(l, j)] - cpy[(k, j)] * normalizer));
            res.set(k, j, &(res[(l, j)] - res[(k, j)] * normalizer));
          }
        }
      }
    }

    res
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

impl<D: Dim, T:ToStr> ToStr for NMat<D, T>
{
  fn to_str(&self) -> ~str
  { ~"Mat" + Dim::dim::<D>().to_str() + " {" + self.mij.to_str() + " }" }
}
