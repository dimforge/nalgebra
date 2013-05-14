use traits::dot::Dot;
use traits::sqrt::Sqrt;

#[deriving(Eq)]
pub struct Vec3<T>
{
  x : T,
  y : T,
  z : T
}

pub fn Vec3<T:Copy>(x: T, y: T, z: T) -> Vec3<T>
{ Vec3 {x: x, y: y, z: z} }


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

impl<T:Copy + Mul<T, T> + Add<T, T> + Sqrt> Dot<T> for Vec3<T>
{
  fn dot(&self, other : &Vec3<T>) -> T
  { self.x * other.x + self.y * other.y + self.z * other.z } 

  fn sqnorm(&self) -> T
  { self.dot(self) }

  fn norm(&self) -> T
  { self.sqnorm().sqrt() }
}
