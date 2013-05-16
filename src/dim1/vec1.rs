use core::num::{Zero, Algebraic};
use traits::dot::Dot;
use traits::dim::Dim;

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

  fn sqnorm(&self) -> T
  { self.dot(self) }

  fn norm(&self) -> T
  { self.sqnorm().sqrt() }
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

impl<T:ToStr> ToStr for Vec1<T>
{
  fn to_str(&self) -> ~str
  { ~"Vec1 { x : " + self.x.to_str() + " }" }
}
