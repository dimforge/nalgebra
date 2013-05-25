use core::num::{Zero, One, Algebraic};
use core::rand::{Rand, Rng, RngUtil};
use core::cmp::ApproxEq;
use traits::basis::Basis;
use traits::dim::Dim;
use traits::dot::Dot;
use traits::norm::Norm;
use traits::translation::Translation;
use traits::sub_dot::SubDot;
use traits::workarounds::scalar_op::{ScalarMul, ScalarDiv, ScalarAdd, ScalarSub};

#[deriving(Eq, ToStr)]
pub struct Vec1<T>
{ x : T }

pub fn vec1<T:Copy>(x: T) -> Vec1<T>
{ Vec1 {x: x} }

impl<T> Dim for Vec1<T>
{
  fn dim() -> uint
  { 1 }
}

impl<T:Copy + Add<T,T>> Add<Vec1<T>, Vec1<T>> for Vec1<T>
{
  fn add(&self, other: &Vec1<T>) -> Vec1<T>
  { vec1(self.x + other.x) }
}

impl<T:Copy + Sub<T,T>> Sub<Vec1<T>, Vec1<T>> for Vec1<T>
{
  fn sub(&self, other: &Vec1<T>) -> Vec1<T>
  { vec1(self.x - other.x) }
}

impl<T: Copy + Mul<T, T>>
ScalarMul<T> for Vec1<T>
{
  fn scalar_mul(&self, s: &T) -> Vec1<T>
  { Vec1 { x: self.x * *s } }

  fn scalar_mul_inplace(&mut self, s: &T)
  { self.x *= *s; }
}


impl<T: Copy + Div<T, T>>
ScalarDiv<T> for Vec1<T>
{
  fn scalar_div(&self, s: &T) -> Vec1<T>
  { Vec1 { x: self.x / *s } }

  fn scalar_div_inplace(&mut self, s: &T)
  { self.x /= *s; }
}

impl<T: Copy + Add<T, T>>
ScalarAdd<T> for Vec1<T>
{
  fn scalar_add(&self, s: &T) -> Vec1<T>
  { Vec1 { x: self.x + *s } }

  fn scalar_add_inplace(&mut self, s: &T)
  { self.x += *s; }
}

impl<T: Copy + Sub<T, T>>
ScalarSub<T> for Vec1<T>
{
  fn scalar_sub(&self, s: &T) -> Vec1<T>
  { Vec1 { x: self.x - *s } }

  fn scalar_sub_inplace(&mut self, s: &T)
  { self.x -= *s; }
}

impl<T: Copy + Add<T, T>> Translation<Vec1<T>> for Vec1<T>
{
  fn translation(&self) -> Vec1<T>
  { *self }

  fn translated(&self, t: &Vec1<T>) -> Vec1<T>
  { self + *t }

  fn translate(&mut self, t: &Vec1<T>)
  { *self += *t }
}

impl<T:Copy + Mul<T, T>> Dot<T> for Vec1<T>
{
  fn dot(&self, other : &Vec1<T>) -> T
  { self.x * other.x } 
}

impl<T:Copy + Mul<T, T> + Sub<T, T>> SubDot<T> for Vec1<T>
{
  fn sub_dot(&self, a: &Vec1<T>, b: &Vec1<T>) -> T
  { (self.x - a.x) * b.x } 
}

impl<T:Copy + Mul<T, T> + Add<T, T> + Div<T, T> + Algebraic>
Norm<T> for Vec1<T>
{
  fn sqnorm(&self) -> T
  { self.dot(self) }

  fn norm(&self) -> T
  { self.sqnorm().sqrt() }

  fn normalized(&self) -> Vec1<T>
  { vec1(self.x / self.norm()) }

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
  { vec1(-self.x) }
}

impl<T:Copy + Zero> Zero for Vec1<T>
{
  fn zero() -> Vec1<T>
  {
    let _0 = Zero::zero();
    vec1(_0)
  }

  fn is_zero(&self) -> bool
  { self.x.is_zero() }
}

impl<T: Copy + One> Basis for Vec1<T>
{
  fn canonical_basis()     -> ~[Vec1<T>]
  { ~[ vec1(One::one()) ] } // FIXME: this should be static

  fn orthogonal_subspace_basis(&self) -> ~[Vec1<T>]
  { ~[] }
}

impl<T:ApproxEq<T>> ApproxEq<T> for Vec1<T>
{
  fn approx_epsilon() -> T
  { ApproxEq::approx_epsilon::<T, T>() }

  fn approx_eq(&self, other: &Vec1<T>) -> bool
  { self.x.approx_eq(&other.x) }

  fn approx_eq_eps(&self, other: &Vec1<T>, epsilon: &T) -> bool
  { self.x.approx_eq_eps(&other.x, epsilon) }
}

impl<T:Rand + Copy> Rand for Vec1<T>
{
  fn rand<R: Rng>(rng: &mut R) -> Vec1<T>
  { vec1(rng.gen()) }
}
