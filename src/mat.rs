use std::uint::iterate;
use std::num::{One, Zero};
use std::vec::swap;
use std::cmp::ApproxEq;
use std::rand::{Rand, Rng, RngUtil};
use std::iterator::IteratorUtil;
use vec::{Vec1, Vec2, Vec3, Vec4, Vec5, Vec6};
use traits::dim::Dim;
use traits::ring::Ring;
use traits::inv::Inv;
use traits::division_ring::DivisionRing;
use traits::transpose::Transpose;
use traits::rlmul::{RMul, LMul};
use traits::transformation::Transform;

macro_rules! mat_impl(
  ($t: ident, $dim: expr) => (
    impl<N> $t<N>
    {
      #[inline]
      pub fn new(mij: [N, ..$dim * $dim]) -> $t<N>
      { $t { mij: mij } }
    }
  )
)

macro_rules! one_impl(
  ($t: ident, [ $($value: ident)|+ ] ) => (
    impl<N: Copy + One + Zero> One for $t<N>
    {
      #[inline]
      fn one() -> $t<N>
      {
        let (_0, _1) = (Zero::zero::<N>(), One::one::<N>());
        return $t::new( [ $( copy $value, )+ ] )
      }
    }
  )
)


macro_rules! zero_impl(
  ($t: ident, [ $($value: ident)|+ ] ) => (
    impl<N: Copy + Zero> Zero for $t<N>
    {
      #[inline]
      fn zero() -> $t<N>
      {
        let _0 = Zero::zero();
        return $t::new( [ $( copy $value, )+ ] )
      }

     #[inline]
     fn is_zero(&self) -> bool
     { self.mij.iter().all(|e| e.is_zero()) }
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

macro_rules! mat_indexing_impl(
  ($t: ident, $dim: expr) => (
    impl<N: Copy> $t<N>
    {
      #[inline]
      pub fn offset(&self, i: uint, j: uint) -> uint
      { i * $dim + j }
    
      #[inline]
      pub fn set(&mut self, i: uint, j: uint, t: &N)
      {
        self.mij[self.offset(i, j)] = copy *t
      }
    
      #[inline]
      pub fn at(&self, i: uint, j: uint) -> N
      {
        copy self.mij[self.offset(i, j)]
      }
    }
  )
)

macro_rules! mul_impl(
  ($t: ident, $dim: expr) => (
    impl<N: Copy + Ring>
    Mul<$t<N>, $t<N>> for $t<N>
    {
      fn mul(&self, other: &$t<N>) -> $t<N>
      {
        let mut res: $t<N> = Zero::zero();
    
        for iterate(0u, $dim) |i|
        {
          for iterate(0u, $dim) |j|
          {
            let mut acc = Zero::zero::<N>();
    
            for iterate(0u, $dim) |k|
            { acc = acc + self.at(i, k) * other.at(k, j); }
    
            res.set(i, j, &acc);
          }
        }
    
        res
      }
    }
  )
)

macro_rules! rmul_impl(
  ($t: ident, $v: ident, $dim: expr) => (
    impl<N: Copy + Ring>
    RMul<$v<N>> for $t<N>
    {
      fn rmul(&self, other: &$v<N>) -> $v<N>
      {
        let mut res : $v<N> = Zero::zero();
    
        for iterate(0u, $dim) |i|
        {
          for iterate(0u, $dim) |j|
          { res.at[i] = res.at[i] + other.at[j] * self.at(i, j); }
        }
    
        res
      }
    }
  )
)

macro_rules! lmul_impl(
  ($t: ident, $v: ident, $dim: expr) => (
    impl<N: Copy + Ring>
    LMul<$v<N>> for $t<N>
    {
      fn lmul(&self, other: &$v<N>) -> $v<N>
      {
    
        let mut res : $v<N> = Zero::zero();
    
        for iterate(0u, $dim) |i|
        {
          for iterate(0u, $dim) |j|
          { res.at[i] = res.at[i] + other.at[j] * self.at(j, i); }
        }
    
        res
      }
    }
  )
)

macro_rules! transform_impl(
  ($t: ident, $v: ident) => (
    impl<N: Copy + DivisionRing + Eq>
    Transform<$v<N>> for $t<N>
    {
      #[inline]
      fn transform_vec(&self, v: &$v<N>) -> $v<N>
      { self.rmul(v) }
    
      #[inline]
      fn inv_transform(&self, v: &$v<N>) -> $v<N>
      { self.inverse().transform_vec(v) }
    }
  )
)

macro_rules! inv_impl(
  ($t: ident, $dim: expr) => (
    impl<N: Copy + Eq + DivisionRing>
    Inv for $t<N>
    {
      #[inline]
      fn inverse(&self) -> $t<N>
      {
        let mut res : $t<N> = copy *self;
    
        res.invert();
    
        res
      }
    
      fn invert(&mut self)
      {
        let mut res: $t<N> = One::one();
        let     _0N: N     = Zero::zero();
    
        // inversion using Gauss-Jordan elimination
        for iterate(0u, $dim) |k|
        {
          // search a non-zero value on the k-th column
          // FIXME: would it be worth it to spend some more time searching for the
          // max instead?
    
          let mut n0 = k; // index of a non-zero entry
    
          while (n0 != $dim)
          {
            if self.at(n0, k) != _0N
            { break; }
    
            n0 = n0 + 1;
          }
    
          // swap pivot line
          if n0 != k
          {
            for iterate(0u, $dim) |j|
            {
              let off_n0_j = self.offset(n0, j);
              let off_k_j  = self.offset(k, j);
    
              swap(self.mij, off_n0_j, off_k_j);
              swap(res.mij,  off_n0_j, off_k_j);
            }
          }
    
          let pivot = self.at(k, k);
    
          for iterate(k, $dim) |j|
          {
            let selfval = &(self.at(k, j) / pivot);
            self.set(k, j, selfval);
          }
    
          for iterate(0u, $dim) |j|
          {
            let resval  = &(res.at(k, j)   / pivot);
            res.set(k, j, resval);
          }
    
          for iterate(0u, $dim) |l|
          {
            if l != k
            {
              let normalizer = self.at(l, k);
    
              for iterate(k, $dim) |j|
              {
                let selfval = &(self.at(l, j) - self.at(k, j) * normalizer);
                self.set(l, j, selfval);
              }
    
              for iterate(0u, $dim) |j|
              {
                let resval  = &(res.at(l, j) - res.at(k, j) * normalizer);
                res.set(l, j, resval);
              }
            }
          }
        }
    
        *self = res;
      }
    }
  )
)

macro_rules! transpose_impl(
  ($t: ident, $dim: expr) => (
    impl<N: Copy> Transpose for $t<N>
    {
      #[inline]
      fn transposed(&self) -> $t<N>
      {
        let mut res = copy *self;
    
        res.transpose();
    
        res
      }
    
      fn transpose(&mut self)
      {
        for iterate(1u, $dim) |i|
        {
          for iterate(0u, $dim - 1) |j|
          {
            let off_i_j = self.offset(i, j);
            let off_j_i = self.offset(j, i);
    
            swap(self.mij, off_i_j, off_j_i);
          }
        }
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
        let mut zip = self.mij.iter().zip(other.mij.iter());
    
        do zip.all |(a, b)| { a.approx_eq(b) }
      }
    
      #[inline]
      fn approx_eq_eps(&self, other: &$t<N>, epsilon: &N) -> bool
      {
        let mut zip = self.mij.iter().zip(other.mij.iter());
    
        do zip.all |(a, b)| { a.approx_eq_eps(b, epsilon) }
      }
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

#[deriving(ToStr)]
pub struct Mat1<N>
{ mij: [N, ..1 * 1] }

mat_impl!(Mat1, 1)
one_impl!(Mat1, [ _1 ])
zero_impl!(Mat1, [ _0  ])
dim_impl!(Mat1, 1)
mat_indexing_impl!(Mat1, 1)
mul_impl!(Mat1, 1)
rmul_impl!(Mat1, Vec1, 1)
lmul_impl!(Mat1, Vec1, 1)
transform_impl!(Mat1, Vec1)
// inv_impl!(Mat1, 1)
transpose_impl!(Mat1, 1)
approx_eq_impl!(Mat1)
rand_impl!(Mat1, rng, [ rng ])

#[deriving(ToStr)]
pub struct Mat2<N>
{ mij: [N, ..2 * 2] }

mat_impl!(Mat2, 2)
one_impl!(Mat2, [ _1 | _0 |
                      _0 | _1 ])
zero_impl!(Mat2, [ _0 | _0 |
                       _0 | _0 ])
dim_impl!(Mat2, 2)
mat_indexing_impl!(Mat2, 2)
mul_impl!(Mat2, 2)
rmul_impl!(Mat2, Vec2, 2)
lmul_impl!(Mat2, Vec2, 2)
transform_impl!(Mat2, Vec2)
// inv_impl!(Mat2, 2)
transpose_impl!(Mat2, 2)
approx_eq_impl!(Mat2)
rand_impl!(Mat2, rng, [ rng | rng |
                            rng | rng ])

#[deriving(ToStr)]
pub struct Mat3<N>
{ mij: [N, ..3 * 3] }

mat_impl!(Mat3, 3)
one_impl!(Mat3, [ _1 | _0 | _0 |
                      _0 | _1 | _0 |
                      _0 | _0 | _1 ])
zero_impl!(Mat3, [ _0 | _0 | _0 |
                       _0 | _0 | _0 |
                       _0 | _0 | _0 ])
dim_impl!(Mat3, 3)
mat_indexing_impl!(Mat3, 3)
mul_impl!(Mat3, 3)
rmul_impl!(Mat3, Vec3, 3)
lmul_impl!(Mat3, Vec3, 3)
transform_impl!(Mat3, Vec3)
// inv_impl!(Mat3, 3)
transpose_impl!(Mat3, 3)
approx_eq_impl!(Mat3)
rand_impl!(Mat3, rng, [ rng | rng | rng |
                            rng | rng | rng |
                            rng | rng | rng])

#[deriving(ToStr)]
pub struct Mat4<N>
{ mij: [N, ..4 * 4] }

mat_impl!(Mat4, 4)
one_impl!(Mat4, [
          _1 | _0 | _0 | _0 |
          _0 | _1 | _0 | _0 |
          _0 | _0 | _1 | _0 |
          _0 | _0 | _0 | _1
          ])
zero_impl!(Mat4, [
          _0 | _0 | _0 | _0 |
          _0 | _0 | _0 | _0 |
          _0 | _0 | _0 | _0 |
          _0 | _0 | _0 | _0
          ])
dim_impl!(Mat4, 4)
mat_indexing_impl!(Mat4, 4)
mul_impl!(Mat4, 4)
rmul_impl!(Mat4, Vec4, 4)
lmul_impl!(Mat4, Vec4, 4)
transform_impl!(Mat4, Vec4)
inv_impl!(Mat4, 4)
transpose_impl!(Mat4, 4)
approx_eq_impl!(Mat4)
rand_impl!(Mat4, rng, [
           rng | rng | rng | rng |
           rng | rng | rng | rng |
           rng | rng | rng | rng |
           rng | rng | rng | rng
           ])

#[deriving(ToStr)]
pub struct Mat5<N>
{ mij: [N, ..5 * 5] }

mat_impl!(Mat5, 5)
one_impl!(Mat5, [
          _1 | _0 | _0 | _0 | _0 |
          _0 | _1 | _0 | _0 | _0 |
          _0 | _0 | _1 | _0 | _0 |
          _0 | _0 | _0 | _1 | _0 |
          _0 | _0 | _0 | _0 | _1
          ])
zero_impl!(Mat5, [
          _0 | _0 | _0 | _0 | _0 |
          _0 | _0 | _0 | _0 | _0 |
          _0 | _0 | _0 | _0 | _0 |
          _0 | _0 | _0 | _0 | _0 |
          _0 | _0 | _0 | _0 | _0
          ])
dim_impl!(Mat5, 5)
mat_indexing_impl!(Mat5, 5)
mul_impl!(Mat5, 5)
rmul_impl!(Mat5, Vec5, 5)
lmul_impl!(Mat5, Vec5, 5)
transform_impl!(Mat5, Vec5)
inv_impl!(Mat5, 5)
transpose_impl!(Mat5, 5)
approx_eq_impl!(Mat5)
rand_impl!(Mat5, rng, [
           rng | rng | rng | rng | rng |
           rng | rng | rng | rng | rng |
           rng | rng | rng | rng | rng |
           rng | rng | rng | rng | rng |
           rng | rng | rng | rng | rng
           ])

#[deriving(ToStr)]
pub struct Mat6<N>
{ mij: [N, ..6 * 6] }

mat_impl!(Mat6, 6)
one_impl!(Mat6, [
          _1 | _0 | _0 | _0 | _0 | _0 |
          _0 | _1 | _0 | _0 | _0 | _0 |
          _0 | _0 | _1 | _0 | _0 | _0 |
          _0 | _0 | _0 | _1 | _0 | _0 |
          _0 | _0 | _0 | _0 | _1 | _0 |
          _0 | _0 | _0 | _0 | _0 | _1
          ])
zero_impl!(Mat6, [
          _0 | _0 | _0 | _0 | _0 | _0 |
          _0 | _0 | _0 | _0 | _0 | _0 |
          _0 | _0 | _0 | _0 | _0 | _0 |
          _0 | _0 | _0 | _0 | _0 | _0 |
          _0 | _0 | _0 | _0 | _0 | _0 |
          _0 | _0 | _0 | _0 | _0 | _0
          ])
dim_impl!(Mat6, 6)
mat_indexing_impl!(Mat6, 6)
mul_impl!(Mat6, 6)
rmul_impl!(Mat6, Vec6, 6)
lmul_impl!(Mat6, Vec6, 6)
transform_impl!(Mat6, Vec6)
inv_impl!(Mat6, 6)
transpose_impl!(Mat6, 6)
approx_eq_impl!(Mat6)
rand_impl!(Mat6, rng, [
           rng | rng | rng | rng | rng | rng |
           rng | rng | rng | rng | rng | rng |
           rng | rng | rng | rng | rng | rng |
           rng | rng | rng | rng | rng | rng |
           rng | rng | rng | rng | rng | rng |
           rng | rng | rng | rng | rng | rng
           ])

// some specializations:
impl<N: Copy + DivisionRing>
Inv for Mat1<N>
{
  #[inline]
  fn inverse(&self) -> Mat1<N>
  {
    let mut res : Mat1<N> = copy *self;

    res.invert();

    res
  }

  #[inline]
  fn invert(&mut self)
  {
    assert!(!self.mij[0].is_zero());

    self.mij[0] = One::one::<N>() / self.mij[0]
  }
}

impl<N: Copy + DivisionRing>
Inv for Mat2<N>
{
  #[inline]
  fn inverse(&self) -> Mat2<N>
  {
    let mut res : Mat2<N> = copy *self;

    res.invert();

    res
  }

  #[inline]
  fn invert(&mut self)
  {
    let det = self.mij[0 * 2 + 0] * self.mij[1 * 2 + 1] - self.mij[1 * 2 + 0] * self.mij[0 * 2 + 1];

    assert!(!det.is_zero());

    *self = Mat2::new([self.mij[1 * 2 + 1] / det , -self.mij[0 * 2 + 1] / det,
                           -self.mij[1 * 2 + 0] / det, self.mij[0 * 2 + 0] / det])
  }
}

impl<N: Copy + DivisionRing>
Inv for Mat3<N>
{
  #[inline]
  fn inverse(&self) -> Mat3<N>
  {
    let mut res = copy *self;

    res.invert();

    res
  }

  #[inline]
  fn invert(&mut self)
  {
    let minor_m12_m23 = self.mij[1 * 3 + 1] * self.mij[2 * 3 + 2] - self.mij[2 * 3 + 1] * self.mij[1 * 3 + 2];
    let minor_m11_m23 = self.mij[1 * 3 + 0] * self.mij[2 * 3 + 2] - self.mij[2 * 3 + 0] * self.mij[1 * 3 + 2];
    let minor_m11_m22 = self.mij[1 * 3 + 0] * self.mij[2 * 3 + 1] - self.mij[2 * 3 + 0] * self.mij[1 * 3 + 1];

    let det = self.mij[0 * 3 + 0] * minor_m12_m23
              - self.mij[0 * 3 + 1] * minor_m11_m23
              + self.mij[0 * 3 + 2] * minor_m11_m22;

    assert!(!det.is_zero());

    *self = Mat3::new( [
      (minor_m12_m23  / det),
      ((self.mij[0 * 3 + 2] * self.mij[2 * 3 + 1] - self.mij[2 * 3 + 2] * self.mij[0 * 3 + 1]) / det),
      ((self.mij[0 * 3 + 1] * self.mij[1 * 3 + 2] - self.mij[1 * 3 + 1] * self.mij[0 * 3 + 2]) / det),

      (-minor_m11_m23 / det),
      ((self.mij[0 * 3 + 0] * self.mij[2 * 3 + 2] - self.mij[2 * 3 + 0] * self.mij[0 * 3 + 2]) / det),
      ((self.mij[0 * 3 + 2] * self.mij[1 * 3 + 0] - self.mij[1 * 3 + 2] * self.mij[0 * 3 + 0]) / det),

      (minor_m11_m22  / det),
      ((self.mij[0 * 3 + 1] * self.mij[2 * 3 + 0] - self.mij[2 * 3 + 1] * self.mij[0 * 3 + 0]) / det),
      ((self.mij[0 * 3 + 0] * self.mij[1 * 3 + 1] - self.mij[1 * 3 + 0] * self.mij[0 * 3 + 1]) / det)
    ] )
  }
}
