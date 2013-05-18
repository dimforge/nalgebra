use core::num::{Zero, One, Algebraic};
use core::rand::{Rand, Rng, RngUtil};
use std::cmp::FuzzyEq;
use traits::dot::Dot;
use traits::dim::Dim;
use traits::basis::Basis;
use traits::norm::Norm;

#[deriving(Eq)]
pub struct Vec1<T>
{ x : T }

pub fn Vec1<T:Copy>(x: T) -> Vec1<T>
{ Vec1 {x: x} }

impl<T> Dim for Vec1<T>
{
  fn dim() -> uint
  { 1 }
}

impl<T:Copy + Add<T,T>> Add<Vec1<T>, Vec1<T>> for Vec1<T>
{
  fn add(&self, other: &Vec1<T>) -> Vec1<T>
  { Vec1(self.x + other.x) }
}

impl<T:Copy + Sub<T,T>> Sub<Vec1<T>, Vec1<T>> for Vec1<T>
{
  fn sub(&self, other: &Vec1<T>) -> Vec1<T>
  { Vec1(self.x - other.x) }
}

impl<T:Copy + Mul<T, T> + Add<T, T> + Algebraic> Dot<T> for Vec1<T>
{
  fn dot(&self, other : &Vec1<T>) -> T
  { self.x * other.x } 
}

impl<T:Copy + Mul<T, T> + Add<T, T> + Quot<T, T> + Algebraic>
Norm<T> for Vec1<T>
{
  fn sqnorm(&self) -> T
  { self.dot(self) }

  fn norm(&self) -> T
  { self.sqnorm().sqrt() }

  fn normalized(&self) -> Vec1<T>
  { Vec1(self.x / self.norm()) }

  fn normalize(&mut self) -> T
  {
    let l = self.norm();

    self.x /= l;

    l
  }
}

impl<T:Copy + Neg<T>> Neg<Vec1<T>> for Vec1<T>
{
  fn neg(&self) -> Vec1<T>
  { Vec1(-self.x) }
}

impl<T:Copy + Zero> Zero for Vec1<T>
{
  fn zero() -> Vec1<T>
  {
    let _0 = Zero::zero();
    Vec1(_0)
  }

  fn is_zero(&self) -> bool
  { self.x.is_zero() }
}

impl<T: Copy + One> Basis for Vec1<T>
{
  fn canonical_basis()     -> ~[Vec1<T>]
  { ~[ Vec1(One::one()) ] } // FIXME: this should be static

  fn orthogonal_subspace_basis(&self) -> ~[Vec1<T>]
  { ~[] }
}

impl<T:FuzzyEq<T>> FuzzyEq<T> for Vec1<T>
{
  fn fuzzy_eq(&self, other: &Vec1<T>) -> bool
  { self.x.fuzzy_eq(&other.x) }

  fn fuzzy_eq_eps(&self, other: &Vec1<T>, epsilon: &T) -> bool
  { self.x.fuzzy_eq_eps(&other.x, epsilon) }
}

impl<T:Rand + Copy> Rand for Vec1<T>
{
  fn rand<R: Rng>(rng: &R) -> Vec1<T>
  { Vec1(rng.gen()) }
}

impl<T:ToStr> ToStr for Vec1<T>
{
  fn to_str(&self) -> ~str
  { ~"Vec1 { x : " + self.x.to_str() + " }" }
}
