use traits::dot::Dot;
use traits::sqrt::Sqrt;

#[deriving(Eq)]
pub struct Vec2<T>
{
  x : T,
  y : T
}

pub fn Vec2<T:Copy>(x: T, y: T) -> Vec2<T>
{ Vec2 {x: x, y: y} }


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

impl<T:Copy + Mul<T, T> + Add<T, T> + Sqrt> Dot<T> for Vec2<T>
{
  fn dot(&self, other : &Vec2<T>) -> T
  { self.x * other.x + self.y * other.y } 

  fn sqnorm(&self) -> T
  { self.dot(self) }

  fn norm(&self) -> T
  { self.sqnorm().sqrt() }
}

impl<T:ToStr> ToStr for Vec2<T>
{
  fn to_str(&self) -> ~str
  { ~"Vec2 { x : " + self.x.to_str() + ", y : " + self.y.to_str() + " }" }
}
