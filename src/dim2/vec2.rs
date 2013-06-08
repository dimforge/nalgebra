use std::num::{Zero, One, Algebraic};
use std::rand::{Rand, Rng, RngUtil};
use dim1::vec1::Vec1;
use std::cmp::ApproxEq;
use traits::basis::Basis;
use traits::cross::Cross;
use traits::dim::Dim;
use traits::dot::Dot;
use traits::sub_dot::SubDot;
use traits::norm::Norm;
use traits::translation::Translation;
use traits::workarounds::scalar_op::{ScalarMul, ScalarDiv, ScalarAdd, ScalarSub};

#[deriving(Eq, ToStr)]
pub struct Vec2<T>
{
  x : T,
  y : T
}

impl<T: Copy> Vec2<T>
{
  pub fn new(x: T, y: T) -> Vec2<T>
  { Vec2 {x: x, y: y} }
}

impl<T> Dim for Vec2<T>
{
  fn dim() -> uint
  { 2 }
}

impl<T:Copy + Add<T,T>> Add<Vec2<T>, Vec2<T>> for Vec2<T>
{
  fn add(&self, other: &Vec2<T>) -> Vec2<T>
  { Vec2::new(self.x + other.x, self.y + other.y) }
}

impl<T:Copy + Sub<T,T>> Sub<Vec2<T>, Vec2<T>> for Vec2<T>
{
  fn sub(&self, other: &Vec2<T>) -> Vec2<T>
  { Vec2::new(self.x - other.x, self.y - other.y) }
}

impl<T: Copy + Mul<T, T>>
ScalarMul<T> for Vec2<T>
{
  fn scalar_mul(&self, s: &T) -> Vec2<T>
  { Vec2 { x: self.x * *s, y: self.y * *s } }

  fn scalar_mul_inplace(&mut self, s: &T)
  {
    self.x *= *s;
    self.y *= *s;
  }
}


impl<T: Copy + Div<T, T>>
ScalarDiv<T> for Vec2<T>
{
  fn scalar_div(&self, s: &T) -> Vec2<T>
  { Vec2 { x: self.x / *s, y: self.y / *s } }

  fn scalar_div_inplace(&mut self, s: &T)
  {
    self.x /= *s;
    self.y /= *s;
  }
}

impl<T: Copy + Add<T, T>>
ScalarAdd<T> for Vec2<T>
{
  fn scalar_add(&self, s: &T) -> Vec2<T>
  { Vec2 { x: self.x + *s, y: self.y + *s } }

  fn scalar_add_inplace(&mut self, s: &T)
  {
    self.x += *s;
    self.y += *s;
  }
}

impl<T: Copy + Sub<T, T>>
ScalarSub<T> for Vec2<T>
{
  fn scalar_sub(&self, s: &T) -> Vec2<T>
  { Vec2 { x: self.x - *s, y: self.y - *s } }

  fn scalar_sub_inplace(&mut self, s: &T)
  {
    self.x -= *s;
    self.y -= *s;
  }
}

impl<T: Copy + Add<T, T>> Translation<Vec2<T>> for Vec2<T>
{
  fn translation(&self) -> Vec2<T>
  { *self }

  fn translated(&self, t: &Vec2<T>) -> Vec2<T>
  { self + *t }

  fn translate(&mut self, t: &Vec2<T>)
  { *self += *t; }
}

impl<T:Copy + Mul<T, T> + Add<T, T>> Dot<T> for Vec2<T>
{
  fn dot(&self, other : &Vec2<T>) -> T
  { self.x * other.x + self.y * other.y } 
}

impl<T:Copy + Mul<T, T> + Add<T, T> + Sub<T, T>> SubDot<T> for Vec2<T>
{
  fn sub_dot(&self, a: &Vec2<T>, b: &Vec2<T>) -> T
  { (self.x - a.x) * b.x + (self.y - a.y) * b.y } 
}

impl<T:Copy + Mul<T, T> + Add<T, T> + Div<T, T> + Algebraic>
Norm<T> for Vec2<T>
{
  fn sqnorm(&self) -> T
  { self.dot(self) }

  fn norm(&self) -> T
  { self.sqnorm().sqrt() }

  fn normalized(&self) -> Vec2<T>
  {
    let l = self.norm();

    Vec2::new(self.x / l, self.y / l)
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
  { Vec1::new(self.x * other.y - self.y * other.x) }
}

impl<T:Copy + Neg<T>> Neg<Vec2<T>> for Vec2<T>
{
  fn neg(&self) -> Vec2<T>
  { Vec2::new(-self.x, -self.y) }
}

impl<T:Copy + Zero> Zero for Vec2<T>
{
  fn zero() -> Vec2<T>
  {
    let _0 = Zero::zero();
    Vec2::new(_0, _0)
  }

  fn is_zero(&self) -> bool
  { self.x.is_zero() && self.y.is_zero() }
}

impl<T: Copy + One + Zero + Neg<T>> Basis for Vec2<T>
{
  fn canonical_basis()     -> ~[Vec2<T>]
  {
    // FIXME: this should be static
    ~[ Vec2::new(One::one(), Zero::zero()),
       Vec2::new(Zero::zero(), One::one()) ]
  }

  fn orthogonal_subspace_basis(&self) -> ~[Vec2<T>]
  { ~[ Vec2::new(-self.y, self.x) ] }
}

impl<T:ApproxEq<T>> ApproxEq<T> for Vec2<T>
{
  fn approx_epsilon() -> T
  { ApproxEq::approx_epsilon::<T, T>() }

  fn approx_eq(&self, other: &Vec2<T>) -> bool
  { self.x.approx_eq(&other.x) && self.y.approx_eq(&other.y) }

  fn approx_eq_eps(&self, other: &Vec2<T>, epsilon: &T) -> bool
  {
    self.x.approx_eq_eps(&other.x, epsilon) &&
    self.y.approx_eq_eps(&other.y, epsilon)
  }
}

impl<T:Rand + Copy> Rand for Vec2<T>
{
  fn rand<R: Rng>(rng: &mut R) -> Vec2<T>
  { Vec2::new(rng.gen(), rng.gen()) }
}
