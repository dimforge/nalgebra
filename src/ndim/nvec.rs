use core::num::{Zero, One, Algebraic};
use core::rand::{Rand, Rng, RngUtil};
use core::vec::{map_zip, from_elem, map, all, all2};
use core::cmp::ApproxEq;
use traits::basis::Basis;
use traits::dim::Dim;
use traits::dot::Dot;
use traits::norm::Norm;
use traits::translation::Translation;
use traits::workarounds::scalar_op::{ScalarMul, ScalarDiv, ScalarAdd, ScalarSub};

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

impl<D, T: Clone> Clone for NVec<D, T>
{
  fn clone(&self) -> NVec<D, T>
  { NVec{ at: self.at.clone() } }
}

impl<D, T: Copy + Add<T,T>> Add<NVec<D, T>, NVec<D, T>> for NVec<D, T>
{
  fn add(&self, other: &NVec<D, T>) -> NVec<D, T>
  { NVec { at: map_zip(self.at, other.at, | a, b | { *a + *b }) } }
}

impl<D, T: Copy + Sub<T,T>> Sub<NVec<D, T>, NVec<D, T>> for NVec<D, T>
{
  fn sub(&self, other: &NVec<D, T>) -> NVec<D, T>
  { NVec { at: map_zip(self.at, other.at, | a, b | *a - *b) } }
}

impl<D, T: Copy + Neg<T>> Neg<NVec<D, T>> for NVec<D, T>
{
  fn neg(&self) -> NVec<D, T>
  { NVec { at: map(self.at, |a| -a) } }
}

impl<D: Dim, T: Copy + Mul<T, T> + Add<T, T> + Algebraic + Zero>
Dot<T> for NVec<D, T>
{
  fn dot(&self, other: &NVec<D, T>) -> T
  {
    let mut res = Zero::zero::<T>();

    for uint::range(0u, Dim::dim::<D>()) |i|
    { res += self.at[i] * other.at[i]; }

    res
  } 
}

impl<D: Dim, T: Copy + Mul<T, T>>
ScalarMul<T> for NVec<D, T>
{
  fn scalar_mul(&self, s: &T) -> NVec<D, T>
  { NVec { at: map(self.at, |a| a * *s) } }

  fn scalar_mul_inplace(&mut self, s: &T)
  {
    for uint::range(0u, Dim::dim::<D>()) |i|
    { self.at[i] *= *s; }
  }
}


impl<D: Dim, T: Copy + Div<T, T>>
ScalarDiv<T> for NVec<D, T>
{
  fn scalar_div(&self, s: &T) -> NVec<D, T>
  { NVec { at: map(self.at, |a| a / *s) } }

  fn scalar_div_inplace(&mut self, s: &T)
  {
    for uint::range(0u, Dim::dim::<D>()) |i|
    { self.at[i] /= *s; }
  }
}

impl<D: Dim, T: Copy + Add<T, T>>
ScalarAdd<T> for NVec<D, T>
{
  fn scalar_add(&self, s: &T) -> NVec<D, T>
  { NVec { at: map(self.at, |a| a + *s) } }

  fn scalar_add_inplace(&mut self, s: &T)
  {
    for uint::range(0u, Dim::dim::<D>()) |i|
    { self.at[i] += *s; }
  }
}

impl<D: Dim, T: Copy + Sub<T, T>>
ScalarSub<T> for NVec<D, T>
{
  fn scalar_sub(&self, s: &T) -> NVec<D, T>
  { NVec { at: map(self.at, |a| a - *s) } }

  fn scalar_sub_inplace(&mut self, s: &T)
  {
    for uint::range(0u, Dim::dim::<D>()) |i|
    { self.at[i] -= *s; }
  }
}

impl<D: Dim, T: Clone + Copy + Add<T, T>> Translation<NVec<D, T>> for NVec<D, T>
{
  fn translation(&self) -> NVec<D, T>
  { self.clone() }

  fn translated(&self, t: &NVec<D, T>) -> NVec<D, T>
  { self + *t }

  fn translate(&mut self, t: &NVec<D, T>)
  { *self = *self + *t; }
}

impl<D: Dim, T: Copy + Mul<T, T> + Add<T, T> + Div<T, T> + Algebraic + Zero +
                Clone>
Norm<T> for NVec<D, T>
{
  fn sqnorm(&self) -> T
  { self.dot(self) }

  fn norm(&self) -> T
  { self.sqnorm().sqrt() }

  fn normalized(&self) -> NVec<D, T>
  {
    let mut res : NVec<D, T> = self.clone();

    res.normalize();

    res
  }

  fn normalize(&mut self) -> T
  {
    let l = self.norm();

    for uint::range(0u, Dim::dim::<D>()) |i|
    { self.at[i] /= l; }

    l
  }
}

impl<D: Dim,
     T: Copy + One + Zero + Neg<T> + Ord + Mul<T, T> + Sub<T, T> + Add<T, T> +
        Div<T, T> + Algebraic + Clone + ApproxEq<T>>
Basis for NVec<D, T>
{
  fn canonical_basis() -> ~[NVec<D, T>]
  {
    let     dim = Dim::dim::<D>();
    let mut res : ~[NVec<D, T>] = ~[];

    for uint::range(0u, dim) |i|
    {
      let mut basis_element : NVec<D, T> = Zero::zero();

      basis_element.at[i] = One::one();

      res.push(basis_element);
    }

    res
  }

  fn orthogonal_subspace_basis(&self) -> ~[NVec<D, T>]
  {
    // compute the basis of the orthogonal subspace using Gram-Schmidt
    // orthogonalization algorithm
    let     dim = Dim::dim::<D>();
    let mut res : ~[NVec<D, T>] = ~[];

    for uint::range(0u, dim) |i|
    {
      let mut basis_element : NVec<D, T> = Zero::zero();

      basis_element.at[i] = One::one();

      if (res.len() == dim - 1)
      { break; }

      let mut elt = basis_element.clone();

      elt -= self.scalar_mul(&basis_element.dot(self));

      for res.each |v|
      { elt -= v.scalar_mul(&elt.dot(v)) };

      if (!elt.sqnorm().approx_eq(&Zero::zero()))
      { res.push(elt.normalized()); }
    }

    assert!(res.len() == dim - 1);

    res
  }
}

// FIXME: I dont really know how te generalize the cross product int
// n-dimensions…
// impl<T: Copy + Mul<T, T> + Sub<T, T>> Cross<T> for NVec<D, T>
// {
//   fn cross(&self, other: &NVec<D, T>) -> T
//   { self.x * other.y - self.y * other.x }
// }

impl<D: Dim, T: Copy + Zero> Zero for NVec<D, T>
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

impl<D, T: ApproxEq<T>> ApproxEq<T> for NVec<D, T>
{
  fn approx_epsilon() -> T
  { ApproxEq::approx_epsilon::<T, T>() }

  fn approx_eq(&self, other: &NVec<D, T>) -> bool
  { all2(self.at, other.at, |a, b| a.approx_eq(b)) }

  fn approx_eq_eps(&self, other: &NVec<D, T>, epsilon: &T) -> bool
  { all2(self.at, other.at, |a, b| a.approx_eq_eps(b, epsilon)) }
}

impl<D: Dim, T: Rand + Zero + Copy> Rand for NVec<D, T>
{
  fn rand<R: Rng>(rng: &mut R) -> NVec<D, T>
  {
    let     dim = Dim::dim::<D>();
    let mut res : NVec<D, T> = Zero::zero();

    for uint::range(0u, dim) |i|
    { res.at[i] = rng.gen() }

    res
  }
}

impl<D: Dim, T: ToStr> ToStr for NVec<D, T>
{
  fn to_str(&self) -> ~str
  { ~"Vec" + Dim::dim::<D>().to_str() + self.at.to_str() }
}
