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
use traits::flatten::Flatten;
use traits::workarounds::scalar_op::{ScalarMul, ScalarDiv, ScalarAdd, ScalarSub};

// D is a phantom parameter, used only as a dimensional token.
// Its allows use to encode the vector dimension at the type-level.
// It can be anything implementing the Dim trait. However, to avoid confusion,
// using d0, d1, d2, d3, ..., d7 (or your own dn) are prefered.
// FIXME: it might be possible to implement type-level integers and use them
// here?
#[deriving(Eq, ToStr)]
pub struct NVec<D, N>
{ at: DVec<N> }


impl<D: Dim, N> Dim for NVec<D, N>
{
  #[inline(always)]
  fn dim() -> uint
  { Dim::dim::<D>() }
}

impl<D, N: Clone> Clone for NVec<D, N>
{
  #[inline(always)]
  fn clone(&self) -> NVec<D, N>
  { NVec{ at: self.at.clone() } }
}

impl<D, N: Copy + Add<N,N>> Add<NVec<D, N>, NVec<D, N>> for NVec<D, N>
{
  #[inline(always)]
  fn add(&self, other: &NVec<D, N>) -> NVec<D, N>
  { NVec { at: self.at + other.at } }
}

impl<D, N: Copy + Sub<N,N>> Sub<NVec<D, N>, NVec<D, N>> for NVec<D, N>
{
  #[inline(always)]
  fn sub(&self, other: &NVec<D, N>) -> NVec<D, N>
  { NVec { at: self.at - other.at } }
}

impl<D, N: Copy + Neg<N>> Neg<NVec<D, N>> for NVec<D, N>
{
  #[inline(always)]
  fn neg(&self) -> NVec<D, N>
  { NVec { at: -self.at } }
}

impl<D: Dim, N: Copy + Ring>
Dot<N> for NVec<D, N>
{
  #[inline(always)]
  fn dot(&self, other: &NVec<D, N>) -> N
  { self.at.dot(&other.at) } 
}

impl<D: Dim, N: Copy + Ring> SubDot<N> for NVec<D, N>
{
  #[inline(always)]
  fn sub_dot(&self, a: &NVec<D, N>, b: &NVec<D, N>) -> N
  { self.at.sub_dot(&a.at, &b.at) } 
}

impl<D: Dim, N: Copy + Mul<N, N>>
ScalarMul<N> for NVec<D, N>
{
  #[inline(always)]
  fn scalar_mul(&self, s: &N) -> NVec<D, N>
  { NVec { at: self.at.scalar_mul(s) } }

  #[inline(always)]
  fn scalar_mul_inplace(&mut self, s: &N)
  { self.at.scalar_mul_inplace(s) }
}


impl<D: Dim, N: Copy + Div<N, N>>
ScalarDiv<N> for NVec<D, N>
{
  #[inline(always)]
  fn scalar_div(&self, s: &N) -> NVec<D, N>
  { NVec { at: self.at.scalar_div(s) } }

  #[inline(always)]
  fn scalar_div_inplace(&mut self, s: &N)
  { self.at.scalar_div_inplace(s) }
}

impl<D: Dim, N: Copy + Add<N, N>>
ScalarAdd<N> for NVec<D, N>
{
  #[inline(always)]
  fn scalar_add(&self, s: &N) -> NVec<D, N>
  { NVec { at: self.at.scalar_add(s) } }

  #[inline(always)]
  fn scalar_add_inplace(&mut self, s: &N)
  { self.at.scalar_add_inplace(s) }
}

impl<D: Dim, N: Copy + Sub<N, N>>
ScalarSub<N> for NVec<D, N>
{
  #[inline(always)]
  fn scalar_sub(&self, s: &N) -> NVec<D, N>
  { NVec { at: self.at.scalar_sub(s) } }

  #[inline(always)]
  fn scalar_sub_inplace(&mut self, s: &N)
  { self.scalar_sub_inplace(s) }
}

impl<D: Dim, N: Clone + Copy + Add<N, N>> Translation<NVec<D, N>> for NVec<D, N>
{
  #[inline(always)]
  fn translation(&self) -> NVec<D, N>
  { self.clone() }

  #[inline(always)]
  fn translated(&self, t: &NVec<D, N>) -> NVec<D, N>
  { self + *t }

  #[inline(always)]
  fn translate(&mut self, t: &NVec<D, N>)
  { *self = *self + *t; }
}

impl<D: Dim, N: Copy + DivisionRing + Algebraic + Clone>
Norm<N> for NVec<D, N>
{
  #[inline(always)]
  fn sqnorm(&self) -> N
  { self.dot(self) }

  #[inline(always)]
  fn norm(&self) -> N
  { self.sqnorm().sqrt() }

  #[inline(always)]
  fn normalized(&self) -> NVec<D, N>
  {
    let mut res : NVec<D, N> = self.clone();

    res.normalize();

    res
  }

  #[inline(always)]
  fn normalize(&mut self) -> N
  { self.at.normalize() }
}

impl<D: Dim,
     N: Copy + DivisionRing + Algebraic + Clone + ApproxEq<N>>
Basis for NVec<D, N>
{
  #[inline(always)]
  fn canonical_basis() -> ~[NVec<D, N>]
  { map(DVec::canonical_basis_with_dim(Dim::dim::<D>()), |&e| NVec { at: e }) }

  #[inline(always)]
  fn orthogonal_subspace_basis(&self) -> ~[NVec<D, N>]
  { map(self.at.orthogonal_subspace_basis(), |&e| NVec { at: e }) }
}

// FIXME: I dont really know how te generalize the cross product int
// n-dimensions…
// impl<N: Copy + Mul<N, N> + Sub<N, N>> Cross<N> for NVec<D, N>
// {
//   fn cross(&self, other: &NVec<D, N>) -> N
//   { self.x * other.y - self.y * other.x }
// }

impl<D: Dim, N: Copy + Zero> Zero for NVec<D, N>
{
  #[inline(always)]
  fn zero() -> NVec<D, N>
  { NVec { at: zero_vec_with_dim(Dim::dim::<D>()) } }

  #[inline(always)]
  fn is_zero(&self) -> bool
  { is_zero_vec(&self.at) }
}

impl<D, N: ApproxEq<N>> ApproxEq<N> for NVec<D, N>
{
  #[inline(always)]
  fn approx_epsilon() -> N
  { ApproxEq::approx_epsilon::<N, N>() }

  #[inline(always)]
  fn approx_eq(&self, other: &NVec<D, N>) -> bool
  { self.at.approx_eq(&other.at) }

  #[inline(always)]
  fn approx_eq_eps(&self, other: &NVec<D, N>, epsilon: &N) -> bool
  { self.at.approx_eq_eps(&other.at, epsilon) }
}

impl<D: Dim, N: Rand + Zero + Copy> Rand for NVec<D, N>
{
  #[inline(always)]
  fn rand<R: Rng>(rng: &mut R) -> NVec<D, N>
  {
    let     dim = Dim::dim::<D>();
    let mut res : NVec<D, N> = Zero::zero();

    for iterate(0u, dim) |i|
    { res.at.at[i] = rng.gen() }

    res
  }
}

impl<D: Dim, N: Zero + Copy> Flatten<N> for NVec<D, N>
{
  #[inline(always)]
  fn flat_size() -> uint
  { Dim::dim::<D>() }

  #[inline(always)]
  fn from_flattened(l: &[N], off: uint) -> NVec<D, N>
  {
    let     dim = Dim::dim::<D>();
    let mut res = Zero::zero::<NVec<D, N>>();

    for iterate(0u, dim) |i|
    { res.at.at[i] = l[off + i] }

    res
  }

  #[inline(always)]
  fn flatten(&self) -> ~[N]
  {
    let     dim = Dim::dim::<D>();
    let mut res = ~[];

    for iterate(0u, dim) |i|
    { res.push(self.at.at[i]) }

    res
  }

  #[inline(always)]
  fn flatten_to(&self, l: &mut [N], off: uint)
  {
    let dim = Dim::dim::<D>();

    for iterate(0u, dim) |i|
    { l[off + i] = self.at.at[i] }
  }
}
