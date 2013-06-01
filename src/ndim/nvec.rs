use std::uint::iterate;
use std::num::{Zero, Algebraic};
use std::rand::{Rand, Rng, RngUtil};
use std::vec::{map};
use std::cmp::ApproxEq;
use ndim::dvec::{DVec, zero_vec_with_dim, is_zero_vec};
use traits::basis::Basis;
use traits::ring::Ring;
use traits::division_ring::DivisionRing;
use traits::dim::Dim;
use traits::dot::Dot;
use traits::sub_dot::SubDot;
use traits::norm::Norm;
use traits::translation::Translation;
use traits::workarounds::scalar_op::{ScalarMul, ScalarDiv, ScalarAdd, ScalarSub};

// D is a phantom parameter, used only as a dimensional token.
// Its allows use to encode the vector dimension at the type-level.
// It can be anything implementing the Dim trait. However, to avoid confusion,
// using d0, d1, d2, d3, ..., d7 (or your own dn) are prefered.
// FIXME: it might be possible to implement type-level integers and use them
// here?
#[deriving(Eq, ToStr)]
pub struct NVec<D, T>
{ at: DVec<T> }


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
  { NVec { at: self.at + other.at } }
}

impl<D, T: Copy + Sub<T,T>> Sub<NVec<D, T>, NVec<D, T>> for NVec<D, T>
{
  fn sub(&self, other: &NVec<D, T>) -> NVec<D, T>
  { NVec { at: self.at - other.at } }
}

impl<D, T: Copy + Neg<T>> Neg<NVec<D, T>> for NVec<D, T>
{
  fn neg(&self) -> NVec<D, T>
  { NVec { at: -self.at } }
}

impl<D: Dim, T: Copy + Ring>
Dot<T> for NVec<D, T>
{
  fn dot(&self, other: &NVec<D, T>) -> T
  { self.at.dot(&other.at) } 
}

impl<D: Dim, T: Copy + Ring> SubDot<T> for NVec<D, T>
{
  fn sub_dot(&self, a: &NVec<D, T>, b: &NVec<D, T>) -> T
  { self.at.sub_dot(&a.at, &b.at) } 
}

impl<D: Dim, T: Copy + Mul<T, T>>
ScalarMul<T> for NVec<D, T>
{
  fn scalar_mul(&self, s: &T) -> NVec<D, T>
  { NVec { at: self.at.scalar_mul(s) } }

  fn scalar_mul_inplace(&mut self, s: &T)
  { self.at.scalar_mul_inplace(s) }
}


impl<D: Dim, T: Copy + Div<T, T>>
ScalarDiv<T> for NVec<D, T>
{
  fn scalar_div(&self, s: &T) -> NVec<D, T>
  { NVec { at: self.at.scalar_div(s) } }

  fn scalar_div_inplace(&mut self, s: &T)
  { self.at.scalar_div_inplace(s) }
}

impl<D: Dim, T: Copy + Add<T, T>>
ScalarAdd<T> for NVec<D, T>
{
  fn scalar_add(&self, s: &T) -> NVec<D, T>
  { NVec { at: self.at.scalar_add(s) } }

  fn scalar_add_inplace(&mut self, s: &T)
  { self.at.scalar_add_inplace(s) }
}

impl<D: Dim, T: Copy + Sub<T, T>>
ScalarSub<T> for NVec<D, T>
{
  fn scalar_sub(&self, s: &T) -> NVec<D, T>
  { NVec { at: self.at.scalar_sub(s) } }

  fn scalar_sub_inplace(&mut self, s: &T)
  { self.scalar_sub_inplace(s) }
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

impl<D: Dim, T: Copy + DivisionRing + Algebraic + Clone>
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
  { self.at.normalize() }
}

impl<D: Dim,
     T: Copy + DivisionRing + Algebraic + Clone + ApproxEq<T>>
Basis for NVec<D, T>
{
  fn canonical_basis() -> ~[NVec<D, T>]
  { map(DVec::canonical_basis_with_dim(Dim::dim::<D>()), |&e| NVec { at: e }) }

  fn orthogonal_subspace_basis(&self) -> ~[NVec<D, T>]
  { map(self.at.orthogonal_subspace_basis(), |&e| NVec { at: e }) }
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
  { NVec { at: zero_vec_with_dim(Dim::dim::<D>()) } }

  fn is_zero(&self) -> bool
  { is_zero_vec(&self.at) }
}

impl<D, T: ApproxEq<T>> ApproxEq<T> for NVec<D, T>
{
  fn approx_epsilon() -> T
  { ApproxEq::approx_epsilon::<T, T>() }

  fn approx_eq(&self, other: &NVec<D, T>) -> bool
  { self.at.approx_eq(&other.at) }

  fn approx_eq_eps(&self, other: &NVec<D, T>, epsilon: &T) -> bool
  { self.at.approx_eq_eps(&other.at, epsilon) }
}

impl<D: Dim, T: Rand + Zero + Copy> Rand for NVec<D, T>
{
  fn rand<R: Rng>(rng: &mut R) -> NVec<D, T>
  {
    let     dim = Dim::dim::<D>();
    let mut res : NVec<D, T> = Zero::zero();

    for iterate(0u, dim) |i|
    { res.at.at[i] = rng.gen() }

    res
  }
}
