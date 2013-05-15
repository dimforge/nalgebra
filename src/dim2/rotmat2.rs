use core::num::{One, Zero};
use traits::workarounds::trigonometric::Trigonometric;
use traits::dim::Dim;
use dim2::mat2::Mat2;
// use dim2::vec2::Vec2;

// FIXME: using a newtype did not compile (due a compiler bug)
// pub type Rotmat2<T> = (Mat2<T>)
#[deriving(Eq)]
pub struct Rotmat2<T>
{
  priv submat: Mat2<T>
}

pub fn Rotmat2<T:Copy + Trigonometric + Neg<T>>(angle: T) -> Rotmat2<T>
{
  let coa = angle.cos();
  let sia = angle.sin();

  Rotmat2
  { submat: Mat2(coa, -sia, sia, coa) }
}

impl<T> Dim for Rotmat2<T>
{
  fn dim() -> uint
  { 2 }
}
 
impl<T:Copy + One + Zero> One for Rotmat2<T>
{
  fn one() -> Rotmat2<T>
  { Rotmat2 { submat: One::one() } }
}

impl<T:Copy + Mul<T, T> + Add<T, T>> Mul<Rotmat2<T>, Rotmat2<T>> for Rotmat2<T>
{
  fn mul(&self, other: &Rotmat2<T>) -> Rotmat2<T>
  { Rotmat2 { submat: self.submat.mul(&other.submat) } }
}

// FIXME: implementation of the same classes for the same struct fails
// with "internal compiler error: Asked to compute kind of a type variable".
//
// impl<T:Copy + Mul<T, T> + Add<T, T>> Mul<Vec2<T>, Vec2<T>> for Rotmat2<T>
// {
//   fn mul(&self, v: &Vec2<T>) -> Vec2<T>
//   { Vec2(self.m11 * v.x + self.m12 * v.y, self.m21 * v.x + self.m22 * v.y) }
// }

// FIXME: implementation of the same classes for the same struct (here Vec2<T>)
// fails with "internal compiler error: Asked to compute kind of a type
// variable".
//
// impl<T:Copy + Mul<T, T> + Add<T, T>> Mul<Rotmat2<T>, Vec2<T>> for Vec2<T>
// {
//   fn mul(&self, m: &Rotmat2<T>) -> Vec2<T>
//   { self.mult(&m.submat) }
// }

/*
impl<T:Copy + Mul<T, T> + Div<T, T> + Sub<T, T> + Neg<T> + Eq + Zero>
Inv<T> for Rotmat2<T>
{
  fn inv(&self) -> Rotmat2<T>
  {
    // transpose
    Rotmat2(
      Mat2(
        self.submat.m11, self.submat.m21,
        self.submat.m12, self.submat.m22
      )
    )
  }
}
*/

/*
impl<T:ToStr> ToStr for Rotmat2<T>
{
  fn to_str(&self) -> ~str
  {
    ~"Rotmat2 {"
    + " m11: " + self.m11.to_str()
    + " m12: " + self.m12.to_str()

    + " m21: " + self.m21.to_str()
    + " m22: " + self.m22.to_str()
    + " }"
  }
}
*/
