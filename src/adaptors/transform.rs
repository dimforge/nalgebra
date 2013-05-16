use core::num::{One, Zero};
use traits::dim::Dim;
use traits::inv::Inv;
use traits::transpose::Transpose;
use traits::workarounds::rlmul::{RMul, LMul};

pub struct Transform<M, V>
{
  submat   : M,
  subtrans : V
}

pub fn Transform<M: Copy, V: Copy>(mat: &M, trans: &V) -> Transform<M, V>
{ Transform { submat: *mat, subtrans: *trans } }

impl<M:Dim, V> Dim for Transform<M, V>
{
  fn dim() -> uint
  { Dim::dim::<M>() }
}

impl<M:Copy + One, V:Copy + Zero> One for Transform<M, V>
{
  fn one() -> Transform<M, V>
  { Transform { submat: One::one(), subtrans: Zero::zero() } }
}

impl<M:Copy + Zero, V:Copy + Zero> Zero for Transform<M, V>
{
  fn zero() -> Transform<M, V>
  { Transform { submat: Zero::zero(), subtrans: Zero::zero() } }

  fn is_zero(&self) -> bool
  { self.submat.is_zero() && self.subtrans.is_zero() }
}

impl<M:Copy + RMul<V> + Mul<M, M>, V:Copy + Add<V, V>>
Mul<Transform<M, V>, Transform<M, V>> for Transform<M, V>
{
  fn mul(&self, other: &Transform<M, V>) -> Transform<M, V>
  {
    Transform { submat: self.submat * other.submat,
                subtrans: self.subtrans + self.submat.rmul(&other.subtrans) }
  }
}

impl<M: RMul<V>, V> RMul<V> for Transform<M, V>
{
  fn rmul(&self, other: &V) -> V
  { self.submat.rmul(other) }
}

impl<M: LMul<V>, V> LMul<V> for Transform<M, V>
{
  fn lmul(&self, other: &V) -> V
  { self.submat.lmul(other) }
}

impl<M:Copy + Transpose + Inv + RMul<V>, V:Copy + Neg<V>>
Inv for Transform<M, V>
{
  fn invert(&mut self)
  {
    self.submat.invert();
    self.subtrans = self.submat.rmul(&-self.subtrans);
  }

  fn inverse(&self) -> Transform<M, V>
  {
    let mut res = *self;

    res.invert();

    res
  }
}

impl<M:ToStr, V:ToStr> ToStr for Transform<M, V>
{
  fn to_str(&self) -> ~str
  {
    ~"Transform {" + " submat: "    + self.submat.to_str()    +
                     " subtrans: "  + self.subtrans.to_str()  + " }"
  }
}
