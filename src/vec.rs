use std::num::{abs, Zero, One, Algebraic, Bounded};
use std::rand::{Rand, Rng, RngUtil};
use std::vec::{VecIterator, VecMutIterator};
use std::iterator::{Iterator, FromIterator};
use std::cmp::ApproxEq;
use traits::iterable::{Iterable, IterableMut, FromAnyIterator};
use traits::basis::Basis;
use traits::cross::Cross;
use traits::dim::Dim;
use traits::dot::Dot;
use traits::sub_dot::SubDot;
use traits::norm::Norm;
use traits::translation::{Translation, Translatable};
use traits::scalar_op::{ScalarMul, ScalarDiv, ScalarAdd, ScalarSub};
use std::uint::iterate;

use std::iterator::IteratorUtil;
use traits::ring::Ring;
use traits::division_ring::DivisionRing;

// FIXME: is there a way to split this file:
//  − one file for macros
//  − another file for macro calls and specializations
// ?

macro_rules! new_impl(
  ($t: ident, $dim: expr) => (
    impl<N> $t<N>
    {
      #[inline]
      pub fn new(at: [N, ..$dim]) -> $t<N>
      { $t { at: at } }
    }
  )
)

macro_rules! new_repeat_impl(
  ($t: ident, $param: ident, [ $($elem: ident)|+ ]) => (
    impl<N: Copy> $t<N>
    {
      #[inline]
      pub fn new_repeat($param: N) -> $t<N>
      { $t{ at: [ $( copy $elem, )+ ] } }
    }
  )
)

macro_rules! iterable_impl(
  ($t: ident) => (
    impl<N> Iterable<N> for $t<N>
    {
      fn iter<'l>(&'l self) -> VecIterator<'l, N>
      { self.at.iter() }
    }
  )
)

macro_rules! iterable_mut_impl(
  ($t: ident) => (
    impl<N> IterableMut<N> for $t<N>
    {
      fn mut_iter<'l>(&'l mut self) -> VecMutIterator<'l, N>
      { self.at.mut_iter() }
    }
  )
)

macro_rules! eq_impl(
  ($t: ident) => (
    impl<N: Eq> Eq for $t<N>
    {
      #[inline]
      fn eq(&self, other: &$t<N>) -> bool
      { self.at.iter().zip(other.at.iter()).all(|(a, b)| a == b) }
    
      #[inline]
      fn ne(&self, other: &$t<N>) -> bool
      { self.at.iter().zip(other.at.iter()).all(|(a, b)| a != b) }
    }
  )
)

macro_rules! dim_impl(
  ($t: ident, $dim: expr) => (
    impl<N> Dim for $t<N>
    {
      #[inline]
      fn dim() -> uint
      { $dim }
    }
  )
)

// FIXME: add the possibility to specialize that
macro_rules! basis_impl(
  ($t: ident, $dim: expr) => (
    impl<N: Copy + DivisionRing + Algebraic + ApproxEq<N>> Basis for $t<N>
    {
      pub fn canonical_basis() -> ~[$t<N>]
      {
        let mut res : ~[$t<N>] = ~[];
    
        for iterate(0u, $dim) |i|
        {
          let mut basis_element : $t<N> = Zero::zero();
    
          basis_element.at[i] = One::one();
    
          res.push(basis_element);
        }
    
        res
      }
    
      pub fn orthogonal_subspace_basis(&self) -> ~[$t<N>]
      {
        // compute the basis of the orthogonal subspace using Gram-Schmidt
        // orthogonalization algorithm
        let mut res : ~[$t<N>] = ~[];
    
        for iterate(0u, $dim) |i|
        {
          let mut basis_element : $t<N> = Zero::zero();
    
          basis_element.at[i] = One::one();
    
          if res.len() == $dim - 1
          { break; }
    
          let mut elt = copy basis_element;
    
          elt = elt - self.scalar_mul(&basis_element.dot(self));
    
          for res.iter().advance |v|
          { elt = elt - v.scalar_mul(&elt.dot(v)) };
    
          if !elt.sqnorm().approx_eq(&Zero::zero())
          { res.push(elt.normalized()); }
        }
    
        res
      }
    }
  )
)

macro_rules! add_impl(
  ($t: ident) => (
    impl<N: Copy + Add<N,N>> Add<$t<N>, $t<N>> for $t<N>
    {
      #[inline]
      fn add(&self, other: &$t<N>) -> $t<N>
      {
        self.at.iter()
               .zip(other.at.iter())
               .transform(|(a, b)| { *a + *b })
               .collect()
      }
    }
  )
)

macro_rules! sub_impl(
  ($t: ident) => (
    impl<N: Copy + Sub<N,N>> Sub<$t<N>, $t<N>> for $t<N>
    {
      #[inline]
      fn sub(&self, other: &$t<N>) -> $t<N>
      {
        self.at.iter()
               .zip(other.at.iter())
               .transform(| (a, b) | { *a - *b })
               .collect()
      }
    }
  )
)

macro_rules! neg_impl(
  ($t: ident) => (
    impl<N: Neg<N>> Neg<$t<N>> for $t<N>
    {
      #[inline]
      fn neg(&self) -> $t<N>
      { self.at.iter().transform(|a| -a).collect() }
    }
  )
)

macro_rules! dot_impl(
  ($t: ident, $dim: expr) => (
    impl<N: Ring> Dot<N> for $t<N>
    {
      #[inline]
      fn dot(&self, other: &$t<N>) -> N
      {
        let mut res = Zero::zero::<N>();
    
        for iterate(0u, $dim) |i|
        { res = res + self.at[i] * other.at[i]; }
    
        res
      } 
    }
  )
)

macro_rules! sub_dot_impl(
  ($t: ident, $dim: expr) => (
    impl<N: Ring> SubDot<N> for $t<N>
    {
      #[inline]
      fn sub_dot(&self, a: &$t<N>, b: &$t<N>) -> N
      {
        let mut res = Zero::zero::<N>();
    
        for iterate(0u, $dim) |i|
        { res = res + (self.at[i] - a.at[i]) * b.at[i]; }
    
        res
      } 
    }
  )
)

macro_rules! scalar_mul_impl(
  ($t: ident, $dim: expr) => (
    impl<N: Mul<N, N>> ScalarMul<N> for $t<N>
    {
      #[inline]
      fn scalar_mul(&self, s: &N) -> $t<N>
      { self.at.iter().transform(|a| a * *s).collect() }
    
      #[inline]
      fn scalar_mul_inplace(&mut self, s: &N)
      {
        for iterate(0u, $dim) |i|
        { self.at[i] = self.at[i] * *s; }
      }
    }
  )
)


macro_rules! scalar_div_impl(
  ($t: ident, $dim: expr) => (
    impl<N: Div<N, N>> ScalarDiv<N> for $t<N>
    {
      #[inline]
      fn scalar_div(&self, s: &N) -> $t<N>
      { self.at.iter().transform(|a| a / *s).collect() }
    
      #[inline]
      fn scalar_div_inplace(&mut self, s: &N)
      {
        for iterate(0u, $dim) |i|
        { self.at[i] = self.at[i] / *s; }
      }
    }
  )
)

macro_rules! scalar_add_impl(
  ($t: ident, $dim: expr) => (
    impl<N: Add<N, N>> ScalarAdd<N> for $t<N>
    {
      #[inline]
      fn scalar_add(&self, s: &N) -> $t<N>
      { self.at.iter().transform(|a| a + *s).collect() }
    
      #[inline]
      fn scalar_add_inplace(&mut self, s: &N)
      {
        for iterate(0u, $dim) |i|
        { self.at[i] = self.at[i] + *s; }
      }
    }
  )
)

macro_rules! scalar_sub_impl(
  ($t: ident, $dim: expr) => (
    impl<N: Sub<N, N>> ScalarSub<N> for $t<N>
    {
      #[inline]
      fn scalar_sub(&self, s: &N) -> $t<N>
      { self.at.iter().transform(|a| a - *s).collect() }
    
      #[inline]
      fn scalar_sub_inplace(&mut self, s: &N)
      {
        for iterate(0u, $dim) |i|
        { self.at[i] = self.at[i] - *s; }
      }
    }
  )
)

macro_rules! translation_impl(
  ($t: ident) => (
    impl<N: Copy + Add<N, N> + Neg<N>> Translation<$t<N>> for $t<N>
    {
      #[inline]
      fn translation(&self) -> $t<N>
      { copy *self }

      #[inline]
      fn inv_translation(&self) -> $t<N>
      { -self }
    
      #[inline]
      fn translate_by(&mut self, t: &$t<N>)
      { *self = *self + *t; }
    }
  )
)

macro_rules! translatable_impl(
  ($t: ident) => (
    impl<N: Add<N, N> + Copy> Translatable<$t<N>, $t<N>> for $t<N>
    {
      #[inline]
      fn translated(&self, t: &$t<N>) -> $t<N>
      { self + *t }
    }
  )
)

macro_rules! norm_impl(
  ($t: ident, $dim: expr) => (
    impl<N: Copy + DivisionRing + Algebraic> Norm<N> for $t<N>
    {
      #[inline]
      fn sqnorm(&self) -> N
      { self.dot(self) }
    
      #[inline]
      fn norm(&self) -> N
      { self.sqnorm().sqrt() }
    
      #[inline]
      fn normalized(&self) -> $t<N>
      {
        let mut res : $t<N> = copy *self;
    
        res.normalize();
    
        res
      }
    
      #[inline]
      fn normalize(&mut self) -> N
      {
        let l = self.norm();
    
        for iterate(0u, $dim) |i|
        { self.at[i] = self.at[i] / l; }
    
        l
      }
    }
  )
)

macro_rules! approx_eq_impl(
  ($t: ident) => (
    impl<N: ApproxEq<N>> ApproxEq<N> for $t<N>
    {
      #[inline]
      fn approx_epsilon() -> N
      { ApproxEq::approx_epsilon::<N, N>() }
    
      #[inline]
      fn approx_eq(&self, other: &$t<N>) -> bool
      {
        let mut zip = self.at.iter().zip(other.at.iter());
    
        do zip.all |(a, b)| { a.approx_eq(b) }
      }
    
      #[inline]
      fn approx_eq_eps(&self, other: &$t<N>, epsilon: &N) -> bool
      {
        let mut zip = self.at.iter().zip(other.at.iter());
    
        do zip.all |(a, b)| { a.approx_eq_eps(b, epsilon) }
      }
    }
  )
)

macro_rules! zero_impl(
  ($t: ident) => (
    impl<N: Copy + Zero> Zero for $t<N>
    {
      #[inline]
      fn zero() -> $t<N>
      { $t::new_repeat(Zero::zero()) }
    
      #[inline]
      fn is_zero(&self) -> bool
      { self.at.iter().all(|e| e.is_zero()) }
    }
  )
)

macro_rules! rand_impl(
  ($t: ident, $param: ident, [ $($elem: ident)|+ ]) => (
    impl<N: Rand> Rand for $t<N>
    {
      #[inline]
      fn rand<R: Rng>($param: &mut R) -> $t<N>
      { $t::new([ $( $elem.gen(), )+ ]) }
    }
  )
)

macro_rules! from_any_iterator_impl(
  ($t: ident, $param: ident, [ $($elem: ident)|+ ]) => (
    impl<N: Copy> FromAnyIterator<N> for $t<N>
    {
      fn from_iterator<'l>($param: &mut VecIterator<'l, N>) -> $t<N>
      { $t { at: [ $( copy *$elem.next().unwrap(), )+ ] } }

      fn from_mut_iterator<'l>($param: &mut VecMutIterator<'l, N>) -> $t<N>
      { $t { at: [ $( copy *$elem.next().unwrap(), )+ ] } }
    }
  )
)

macro_rules! from_iterator_impl(
  ($t: ident, $param: ident, [ $($elem: ident)|+ ]) => (
    impl<N, Iter: Iterator<N>> FromIterator<N, Iter> for $t<N>
    {
      fn from_iterator($param: &mut Iter) -> $t<N>
      { $t { at: [ $( $elem.next().unwrap(), )+ ] } }
    }
  )
)

macro_rules! bounded_impl(
  ($t: ident) => (
    impl<N: Bounded + Copy> Bounded for $t<N>
    {
      #[inline]
      fn max_value() -> $t<N>
      { $t::new_repeat(Bounded::max_value()) }
    
      #[inline]
      fn min_value() -> $t<N>
      { $t::new_repeat(Bounded::min_value()) }
    }
  )
)

#[deriving(Ord, ToStr)]
pub struct Vec1<N>
{ at: [N, ..1] }

new_impl!(Vec1, 1)
new_repeat_impl!(Vec1, elem, [elem])
dim_impl!(Vec1, 1)
eq_impl!(Vec1)
// (specialized) basis_impl!(Vec1, 1)
add_impl!(Vec1)
sub_impl!(Vec1)
neg_impl!(Vec1)
dot_impl!(Vec1, 1)
sub_dot_impl!(Vec1, 1)
scalar_mul_impl!(Vec1, 1)
scalar_div_impl!(Vec1, 1)
scalar_add_impl!(Vec1, 1)
scalar_sub_impl!(Vec1, 1)
translation_impl!(Vec1)
translatable_impl!(Vec1)
norm_impl!(Vec1, 1)
approx_eq_impl!(Vec1)
zero_impl!(Vec1)
rand_impl!(Vec1, rng, [rng])
from_iterator_impl!(Vec1, iterator, [iterator])
from_any_iterator_impl!(Vec1, iterator, [iterator])
bounded_impl!(Vec1)
iterable_impl!(Vec1)
iterable_mut_impl!(Vec1)

#[deriving(Ord, ToStr)]
pub struct Vec2<N>
{ at: [N, ..2] }

new_impl!(Vec2, 2)
new_repeat_impl!(Vec2, elem, [elem | elem])
dim_impl!(Vec2, 2)
eq_impl!(Vec2)
// (specialized) basis_impl!(Vec2, 2)
add_impl!(Vec2)
sub_impl!(Vec2)
neg_impl!(Vec2)
dot_impl!(Vec2, 2)
sub_dot_impl!(Vec2, 2)
scalar_mul_impl!(Vec2, 2)
scalar_div_impl!(Vec2, 2)
scalar_add_impl!(Vec2, 2)
scalar_sub_impl!(Vec2, 2)
translation_impl!(Vec2)
translatable_impl!(Vec2)
norm_impl!(Vec2, 2)
approx_eq_impl!(Vec2)
zero_impl!(Vec2)
rand_impl!(Vec2, rng, [rng | rng])
from_iterator_impl!(Vec2, iterator, [iterator | iterator])
from_any_iterator_impl!(Vec2, iterator, [iterator | iterator])
bounded_impl!(Vec2)
iterable_impl!(Vec2)
iterable_mut_impl!(Vec2)

#[deriving(Ord, ToStr)]
pub struct Vec3<N>
{ at: [N, ..3] }

new_impl!(Vec3, 3)
new_repeat_impl!(Vec3, elem, [elem | elem | elem])
dim_impl!(Vec3, 3)
eq_impl!(Vec3)
// (specialized) basis_impl!(Vec3, 3)
add_impl!(Vec3)
sub_impl!(Vec3)
neg_impl!(Vec3)
dot_impl!(Vec3, 3)
sub_dot_impl!(Vec3, 3)
scalar_mul_impl!(Vec3, 3)
scalar_div_impl!(Vec3, 3)
scalar_add_impl!(Vec3, 3)
scalar_sub_impl!(Vec3, 3)
translation_impl!(Vec3)
translatable_impl!(Vec3)
norm_impl!(Vec3, 3)
approx_eq_impl!(Vec3)
zero_impl!(Vec3)
rand_impl!(Vec3, rng, [rng | rng | rng])
from_iterator_impl!(Vec3, iterator, [iterator | iterator | iterator])
from_any_iterator_impl!(Vec3, iterator, [iterator | iterator | iterator])
bounded_impl!(Vec3)
iterable_impl!(Vec3)
iterable_mut_impl!(Vec3)

#[deriving(Ord, ToStr)]
pub struct Vec4<N>
{ at: [N, ..4] }

new_impl!(Vec4, 4)
new_repeat_impl!(Vec4, elem, [elem | elem | elem | elem])
dim_impl!(Vec4, 4)
eq_impl!(Vec4)
basis_impl!(Vec4, 4)
add_impl!(Vec4)
sub_impl!(Vec4)
neg_impl!(Vec4)
dot_impl!(Vec4, 4)
sub_dot_impl!(Vec4, 4)
scalar_mul_impl!(Vec4, 4)
scalar_div_impl!(Vec4, 4)
scalar_add_impl!(Vec4, 4)
scalar_sub_impl!(Vec4, 4)
translation_impl!(Vec4)
translatable_impl!(Vec4)
norm_impl!(Vec4, 4)
approx_eq_impl!(Vec4)
zero_impl!(Vec4)
rand_impl!(Vec4, rng, [rng | rng | rng | rng])
from_iterator_impl!(Vec4, iterator, [iterator | iterator | iterator | iterator])
from_any_iterator_impl!(Vec4, iterator, [iterator | iterator | iterator | iterator])
bounded_impl!(Vec4)
iterable_impl!(Vec4)
iterable_mut_impl!(Vec4)

#[deriving(Ord, ToStr)]
pub struct Vec5<N>
{ at: [N, ..5] }

new_impl!(Vec5, 5)
new_repeat_impl!(Vec5, elem, [elem | elem | elem | elem | elem])
dim_impl!(Vec5, 5)
eq_impl!(Vec5)
basis_impl!(Vec5, 5)
add_impl!(Vec5)
sub_impl!(Vec5)
neg_impl!(Vec5)
dot_impl!(Vec5, 5)
sub_dot_impl!(Vec5, 5)
scalar_mul_impl!(Vec5, 5)
scalar_div_impl!(Vec5, 5)
scalar_add_impl!(Vec5, 5)
scalar_sub_impl!(Vec5, 5)
translation_impl!(Vec5)
translatable_impl!(Vec5)
norm_impl!(Vec5, 5)
approx_eq_impl!(Vec5)
zero_impl!(Vec5)
rand_impl!(Vec5, rng, [rng | rng | rng | rng | rng])
from_iterator_impl!(Vec5, iterator, [iterator | iterator | iterator | iterator | iterator])
from_any_iterator_impl!(Vec5, iterator, [iterator | iterator | iterator | iterator | iterator])
bounded_impl!(Vec5)
iterable_impl!(Vec5)
iterable_mut_impl!(Vec5)

#[deriving(Ord, ToStr)]
pub struct Vec6<N>
{ at: [N, ..6] }

new_impl!(Vec6, 6)
new_repeat_impl!(Vec6, elem, [elem | elem | elem | elem | elem | elem])
dim_impl!(Vec6, 6)
eq_impl!(Vec6)
basis_impl!(Vec6, 6)
add_impl!(Vec6)
sub_impl!(Vec6)
neg_impl!(Vec6)
dot_impl!(Vec6, 6)
sub_dot_impl!(Vec6, 6)
scalar_mul_impl!(Vec6, 6)
scalar_div_impl!(Vec6, 6)
scalar_add_impl!(Vec6, 6)
scalar_sub_impl!(Vec6, 6)
translation_impl!(Vec6)
translatable_impl!(Vec6)
norm_impl!(Vec6, 6)
approx_eq_impl!(Vec6)
zero_impl!(Vec6)
rand_impl!(Vec6, rng, [rng | rng | rng | rng | rng | rng])
from_iterator_impl!(Vec6, iterator, [iterator | iterator | iterator | iterator | iterator | iterator])
from_any_iterator_impl!(Vec6, iterator, [iterator | iterator | iterator | iterator | iterator | iterator])
bounded_impl!(Vec6)
iterable_impl!(Vec6)
iterable_mut_impl!(Vec6)

impl<N: Mul<N, N> + Sub<N, N>> Cross<Vec1<N>> for Vec2<N>
{
  #[inline]
  fn cross(&self, other : &Vec2<N>) -> Vec1<N>
  { Vec1::new([self.at[0] * other.at[1] - self.at[1] * other.at[0]]) }
}

impl<N: Mul<N, N> + Sub<N, N>> Cross<Vec3<N>> for Vec3<N>
{
  #[inline]
  fn cross(&self, other : &Vec3<N>) -> Vec3<N>
  {
    Vec3::new(
      [self.at[1] * other.at[2] - self.at[2] * other.at[1],
       self.at[2] * other.at[0] - self.at[0] * other.at[2],
       self.at[0] * other.at[1] - self.at[1] * other.at[0]]
    )
  }
}

impl<N: One> Basis for Vec1<N>
{
  #[inline]
  fn canonical_basis() -> ~[Vec1<N>]
  { ~[ Vec1::new([One::one()]) ] } // FIXME: this should be static

  #[inline]
  fn orthogonal_subspace_basis(&self) -> ~[Vec1<N>]
  { ~[] }
}

impl<N: Copy + One + Zero + Neg<N>> Basis for Vec2<N>
{
  #[inline]
  fn canonical_basis()     -> ~[Vec2<N>]
  {
    // FIXME: this should be static
    ~[ Vec2::new([One::one(), Zero::zero()]),
       Vec2::new([Zero::zero(), One::one()]) ]
  }

  #[inline]
  fn orthogonal_subspace_basis(&self) -> ~[Vec2<N>]
  { ~[ Vec2::new([-self.at[1], copy self.at[0]]) ] }
}

impl<N: Copy + DivisionRing + Ord + Algebraic>
Basis for Vec3<N>
{
  #[inline]
  fn canonical_basis() -> ~[Vec3<N>]
  {
    // FIXME: this should be static
    ~[ Vec3::new([One::one(), Zero::zero(), Zero::zero()]),
       Vec3::new([Zero::zero(), One::one(), Zero::zero()]),
       Vec3::new([Zero::zero(), Zero::zero(), One::one()]) ]
  }

  #[inline]
  fn orthogonal_subspace_basis(&self) -> ~[Vec3<N>]
  {
      let a = 
        if abs(copy self.at[0]) > abs(copy self.at[1])
        { Vec3::new([copy self.at[2], Zero::zero(), -copy self.at[0]]).normalized() }
        else
        { Vec3::new([Zero::zero(), -self.at[2], copy self.at[1]]).normalized() };

      ~[ a.cross(self), a ]
  }
}
