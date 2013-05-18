use core::num::{Zero, One, Algebraic};
use core::rand::{Rand, Rng, RngUtil};
use std::cmp::FuzzyEq;
use traits::dot::Dot;
use traits::dim::Dim;
use traits::cross::Cross;
use traits::basis::Basis;
use traits::norm::Norm;
use dim1::vec1::Vec1;

#[deriving(Eq)]
pub struct Vec2<T>
{
  x : T,
  y : T
}

pub fn Vec2<T:Copy>(x: T, y: T) -> Vec2<T>
{ Vec2 {x: x, y: y} }

impl<T> Dim for Vec2<T>
{
  fn dim() -> uint
  { 2 }
}

impl<T:Copy + Add<T,T>> Add<Vec2<T>, Vec2<T>> for Vec2<T>
{
  fn add(&self, other: &Vec2<T>) -> Vec2<T>
  { Vec2(self.x + other.x, self.y + other.y) }
}

impl<T:Copy + Sub<T,T>> Sub<Vec2<T>, Vec2<T>> for Vec2<T>
{
  fn sub(&self, other: &Vec2<T>) -> Vec2<T>
  { Vec2(self.x - other.x, self.y - other.y) }
}

impl<T:Copy + Mul<T, T> + Add<T, T> + Algebraic> Dot<T> for Vec2<T>
{
  fn dot(&self, other : &Vec2<T>) -> T
  { self.x * other.x + self.y * other.y } 
}

impl<T:Copy + Mul<T, T> + Add<T, T> + Quot<T, T> + Algebraic>
Norm<T> for Vec2<T>
{
  fn sqnorm(&self) -> T
  { self.dot(self) }

  fn norm(&self) -> T
  { self.sqnorm().sqrt() }

  fn normalized(&self) -> Vec2<T>
  {
    let l = self.norm();

    Vec2(self.x / l, self.y / l)
  }

  fn normalize(&mut self) -> T
  {
    let l = self.norm();

    self.x /= l;
    self.y /= l;

    l
  }
}

impl<T:Copy + Mul<T, T> + Sub<T, T>> Cross<Vec1<T>> for Vec2<T>
{
  fn cross(&self, other : &Vec2<T>) -> Vec1<T>
  { Vec1(self.x * other.y - self.y * other.x) }
}

impl<T:Copy + Neg<T>> Neg<Vec2<T>> for Vec2<T>
{
  fn neg(&self) -> Vec2<T>
  { Vec2(-self.x, -self.y) }
}

impl<T:Copy + Zero> Zero for Vec2<T>
{
  fn zero() -> Vec2<T>
  {
    let _0 = Zero::zero();
    Vec2(_0, _0)
  }

  fn is_zero(&self) -> bool
  { self.x.is_zero() && self.y.is_zero() }
}

impl<T: Copy + One + Zero + Neg<T>> Basis for Vec2<T>
{
  fn canonical_basis()     -> ~[Vec2<T>]
  {
    // FIXME: this should be static
    ~[ Vec2(One::one(), Zero::zero()),
       Vec2(Zero::zero(), One::one()) ]
  }

  fn orthogonal_subspace_basis(&self) -> ~[Vec2<T>]
  { ~[ Vec2(-self.y, self.x) ] }
}

impl<T:FuzzyEq<T>> FuzzyEq<T> for Vec2<T>
{
  fn fuzzy_eq(&self, other: &Vec2<T>) -> bool
  { self.x.fuzzy_eq(&other.x) && self.y.fuzzy_eq(&other.y) }

  fn fuzzy_eq_eps(&self, other: &Vec2<T>, epsilon: &T) -> bool
  {
    self.x.fuzzy_eq_eps(&other.x, epsilon) &&
    self.y.fuzzy_eq_eps(&other.y, epsilon)
  }
}

impl<T:Rand + Copy> Rand for Vec2<T>
{
  fn rand<R: Rng>(rng: &R) -> Vec2<T>
  { Vec2(rng.gen(), rng.gen()) }
}

impl<T:ToStr> ToStr for Vec2<T>
{
  fn to_str(&self) -> ~str
  { ~"Vec2 { x : " + self.x.to_str() + ", y : " + self.y.to_str() + " }" }
}
