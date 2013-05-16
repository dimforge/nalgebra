use core::num::{Zero, Algebraic};
use traits::dim::Dim;
use traits::dot::Dot;
use traits::cross::Cross;

#[deriving(Eq)]
pub struct Vec3<T>
{
  x : T,
  y : T,
  z : T
}

pub fn Vec3<T:Copy>(x: T, y: T, z: T) -> Vec3<T>
{ Vec3 {x: x, y: y, z: z} }

impl<T> Dim for Vec3<T>
{
  fn dim() -> uint
  { 3 }
}

impl<T:Copy + Add<T,T>> Add<Vec3<T>, Vec3<T>> for Vec3<T>
{
  fn add(&self, other: &Vec3<T>) -> Vec3<T>
  { Vec3(self.x + other.x, self.y + other.y, self.z + other.z) }
}

impl<T:Copy + Sub<T,T>> Sub<Vec3<T>, Vec3<T>> for Vec3<T>
{
  fn sub(&self, other: &Vec3<T>) -> Vec3<T>
  { Vec3(self.x - other.x, self.y - other.y, self.z - other.z) }
}

impl<T:Copy + Neg<T>> Neg<Vec3<T>> for Vec3<T>
{
  fn neg(&self) -> Vec3<T>
  { Vec3(-self.x, -self.y, -self.z) }
}

impl<T:Copy + Mul<T, T> + Add<T, T> + Algebraic> Dot<T> for Vec3<T>
{
  fn dot(&self, other : &Vec3<T>) -> T
  { self.x * other.x + self.y * other.y + self.z * other.z } 

  fn sqnorm(&self) -> T
  { self.dot(self) }

  fn norm(&self) -> T
  { self.sqnorm().sqrt() }
}

impl<T:Copy + Mul<T, T> + Sub<T, T>> Cross<Vec3<T>> for Vec3<T>
{
  fn cross(&self, other : &Vec3<T>) -> Vec3<T>
  {
    Vec3(
      self.y * other.z - self.z * other.y,
      self.z * other.x - self.x * other.z,
      self.x * other.y - self.y * other.x
    )
  }
}

impl<T:Copy + Zero> Zero for Vec3<T>
{
  fn zero() -> Vec3<T>
  {
    let _0 = Zero::zero();
    Vec3(_0, _0, _0)
  }

  fn is_zero(&self) -> bool
  { self.x.is_zero() && self.y.is_zero() && self.z.is_zero() }
}

impl<T:ToStr> ToStr for Vec3<T>
{
  fn to_str(&self) -> ~str
  {
    ~"Vec3 "
    + "{ x : " + self.x.to_str()
    + ", y : " + self.y.to_str()
    + ", z : " + self.z.to_str()
    + " }"
  }
}
