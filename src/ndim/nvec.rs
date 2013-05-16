use core::vec::{map_zip, from_elem, map, all, all2};
use core::num::{Zero, Algebraic};
use std::cmp::FuzzyEq;
use traits::dim::Dim;
use traits::dot::Dot;

// D is a phantom parameter, used only as a dimensional token.
// Its allows use to encode the vector dimension at the type-level.
// It can be anything implementing the Dim trait. However, to avoid confusion,
// using d0, d1, d2, d3 and d4 tokens are prefered.
// FIXME: it might be possible to implement type-level integers and use them
// here?
#[deriving(Eq)]
pub struct NVec<D, T>
{
  at: ~[T]
}


impl<D: Dim, T> Dim for NVec<D, T>
{
  fn dim() -> uint
  { Dim::dim::<D>() }
}

impl<D, T:Copy + Add<T,T>> Add<NVec<D, T>, NVec<D, T>> for NVec<D, T>
{
  fn add(&self, other: &NVec<D, T>) -> NVec<D, T>
  { NVec { at: map_zip(self.at, other.at, | a, b | { *a + *b }) } }
}

impl<D, T:Copy + Sub<T,T>> Sub<NVec<D, T>, NVec<D, T>> for NVec<D, T>
{
  fn sub(&self, other: &NVec<D, T>) -> NVec<D, T>
  { NVec { at: map_zip(self.at, other.at, | a, b | *a - *b) } }
}

impl<D, T:Copy + Neg<T>> Neg<NVec<D, T>> for NVec<D, T>
{
  fn neg(&self) -> NVec<D, T>
  { NVec { at: map(self.at, |a| -a) } }
}

impl<D: Dim, T:Copy + Mul<T, T> + Add<T, T> + Algebraic + Zero>
Dot<T> for NVec<D, T>
{
  fn dot(&self, other: &NVec<D, T>) -> T
  {
    let mut res = Zero::zero::<T>();

    for uint::range(0u, Dim::dim::<D>()) |i|
    { res += self.at[i] * other.at[i]; }

    res
  } 

  fn sqnorm(&self) -> T
  { self.dot(self) }

  fn norm(&self) -> T
  { self.sqnorm().sqrt() }
}

// FIXME: I dont really know how te generalize the cross product int
// n-dimensions…
// impl<T:Copy + Mul<T, T> + Sub<T, T>> Cross<T> for NVec<D, T>
// {
//   fn cross(&self, other: &NVec<D, T>) -> T
//   { self.x * other.y - self.y * other.x }
// }

impl<D: Dim, T:Copy + Zero> Zero for NVec<D, T>
{
  fn zero() -> NVec<D, T>
  {
    let _0 = Zero::zero();

    NVec { at: from_elem(Dim::dim::<D>(), _0) }
  }

  fn is_zero(&self) -> bool
  {
    all(self.at, |e| e.is_zero())
  }
}

impl<D, T:FuzzyEq<T>> FuzzyEq<T> for NVec<D, T>
{
  fn fuzzy_eq(&self, other: &NVec<D, T>) -> bool
  { all2(self.at, other.at, |a, b| a.fuzzy_eq(b)) }

  fn fuzzy_eq_eps(&self, other: &NVec<D, T>, epsilon: &T) -> bool
  { all2(self.at, other.at, |a, b| a.fuzzy_eq_eps(b, epsilon)) }
}

impl<D: Dim, T:ToStr> ToStr for NVec<D, T>
{
  fn to_str(&self) -> ~str
  { ~"Vec" + Dim::dim::<D>().to_str() + self.at.to_str() }
}
