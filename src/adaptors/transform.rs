use core::num::{One, Zero};
use core::rand::{Rand, Rng, RngUtil};
use std::cmp::FuzzyEq;
use traits::dim::Dim;
use traits::inv::Inv;
use traits::rotation::Rotation;
use traits::translation::Translation;
use traits::transpose::Transpose;
use traits::delta_transform::DeltaTransform;
use traits::workarounds::rlmul::{RMul, LMul};

pub struct Transform<M, V>
{
  priv submat   : M,
  priv subtrans : V
}

pub fn transform<M: Copy, V: Copy>(mat: &M, trans: &V)
-> Transform<M, V>
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

impl<M: Copy, V: Copy + Translation<V>> Translation<V> for Transform<M, V>
{
  fn translation(&self) -> V
  { self.subtrans.translation() }

  fn translated(&self, t: &V) -> Transform<M, V>
  { transform(&self.submat, &self.subtrans.translated(t)) }

  fn translate(&mut self, t: &V)
  { self.subtrans.translate(t) }
}

impl<M: Rotation<V> + Copy + RMul<V> + One, V: Copy>
Rotation<V> for Transform<M, V>
{
  fn rotation(&self) -> V
  { self.submat.rotation() }

  fn rotated(&self, rot: &V) -> Transform<M, V>
  {
    // FIXME: this does not seem opitmal
    let delta = One::one::<M>().rotated(rot);

    transform(&self.submat.rotated(rot), &delta.rmul(&self.subtrans))
  }

  fn rotate(&mut self, rot: &V)
  {
    // FIXME: this does not seem opitmal
    let delta = One::one::<M>().rotated(rot);
    self.submat.rotate(rot);
    self.subtrans = delta.rmul(&self.subtrans);
  }
}

impl<M: Copy, V> DeltaTransform<M> for Transform<M, V>
{
  fn delta_transform(&self) -> M
  { self.submat }
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

impl<T, M:FuzzyEq<T>, V:FuzzyEq<T>> FuzzyEq<T> for Transform<M, V>
{
  fn fuzzy_eq(&self, other: &Transform<M, V>) -> bool
  {
    self.submat.fuzzy_eq(&other.submat) &&
    self.subtrans.fuzzy_eq(&other.subtrans)
  }

  fn fuzzy_eq_eps(&self, other: &Transform<M, V>, epsilon: &T) -> bool
  {
    self.submat.fuzzy_eq_eps(&other.submat, epsilon) &&
    self.subtrans.fuzzy_eq_eps(&other.subtrans, epsilon)
  }
}

impl<M: Rand + Copy, V: Rand + Copy> Rand for Transform<M, V>
{
  fn rand<R: Rng>(rng: &R) -> Transform<M, V>
  { transform(&rng.gen(), &rng.gen()) }
}

impl<M:ToStr, V:ToStr> ToStr for Transform<M, V>
{
  fn to_str(&self) -> ~str
  {
    ~"Transform {" + " submat: "    + self.submat.to_str()    +
                     " subtrans: "  + self.subtrans.to_str()  + " }"
  }
}
