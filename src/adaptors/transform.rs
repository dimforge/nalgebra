use std::num::{One, Zero};
use std::rand::{Rand, Rng, RngUtil};
use std::cmp::ApproxEq;
use traits::dim::Dim;
use traits::inv::Inv;
use traits::rotation::{Rotation, Rotatable};
use traits::translation::{Translation, Translatable};
use traits::transformation::{Transformation, Transformable};
use traits::transpose::Transpose;
use traits::delta_transform::{DeltaTransform, DeltaTransformVector};
use traits::rlmul::{RMul, LMul};

#[deriving(Eq, ToStr)]
pub struct Transform<M, V>
{
  priv submat   : M,
  priv subtrans : V
}

impl<M, V> Transform<M, V>
{
  #[inline(always)]
  pub fn new(mat: M, trans: V) -> Transform<M, V>
  { Transform { submat: mat, subtrans: trans } }
}

impl<M:Dim, V> Dim for Transform<M, V>
{
  #[inline(always)]
  fn dim() -> uint
  { Dim::dim::<M>() }
}

impl<M: One, V: Zero> One for Transform<M, V>
{
  #[inline(always)]
  fn one() -> Transform<M, V>
  { Transform { submat: One::one(), subtrans: Zero::zero() } }
}

impl<M: Zero, V: Zero> Zero for Transform<M, V>
{
  #[inline(always)]
  fn zero() -> Transform<M, V>
  { Transform { submat: Zero::zero(), subtrans: Zero::zero() } }

  #[inline(always)]
  fn is_zero(&self) -> bool
  { self.submat.is_zero() && self.subtrans.is_zero() }
}

impl<M: RMul<V> + Mul<M, M>, V: Add<V, V>>
Mul<Transform<M, V>, Transform<M, V>> for Transform<M, V>
{
  #[inline(always)]
  fn mul(&self, other: &Transform<M, V>) -> Transform<M, V>
  {
    Transform { submat: self.submat * other.submat,
                subtrans: self.subtrans + self.submat.rmul(&other.subtrans) }
  }
}

impl<M: RMul<V>, V: Add<V, V>> RMul<V> for Transform<M, V>
{
  #[inline(always)]
  fn rmul(&self, other: &V) -> V
  { self.submat.rmul(other) + self.subtrans }
}

impl<M: LMul<V>, V: Add<V, V>> LMul<V> for Transform<M, V>
{
  #[inline(always)]
  fn lmul(&self, other: &V) -> V
  { self.submat.lmul(other) + self.subtrans }
}

impl<M, V: Translation<V>> Translation<V> for Transform<M, V>
{
  #[inline(always)]
  fn translation(&self) -> V
  { self.subtrans.translation() }

  #[inline(always)]
  fn translate(&mut self, t: &V)
  { self.subtrans.translate(t) }
}

impl<M: Copy, V: Translatable<V, Res>, Res: Translation<V>>
Translatable<V, Transform<M, Res>> for Transform<M, V>
{
  #[inline(always)]
  fn translated(&self, t: &V) -> Transform<M, Res>
  { Transform::new(copy self.submat, self.subtrans.translated(t)) }
}

impl<M: Rotation<AV> + RMul<V> + One,
     V,
     AV>
Rotation<AV> for Transform<M, V>
{
  #[inline(always)]
  fn rotation(&self) -> AV
  { self.submat.rotation() }

  #[inline(always)]
  fn rotate(&mut self, rot: &AV)
  {
    // FIXME: this does not seem opitmal
    let mut delta = One::one::<M>();
    delta.rotate(rot);
    self.submat.rotate(rot);
    self.subtrans = delta.rmul(&self.subtrans);
  }
}

impl<M: Rotatable<AV, Res> + One,
     Res: Rotation<AV> + RMul<V>,
     V,
     AV>
Rotatable<AV, Transform<Res, V>> for Transform<M, V>
{
  #[inline(always)]
  fn rotated(&self, rot: &AV) -> Transform<Res, V>
  {
    // FIXME: this does not seem opitmal
    let delta = One::one::<M>().rotated(rot);

    Transform::new(self.submat.rotated(rot), delta.rmul(&self.subtrans))
  }
}

impl<M: RMul<V> + Mul<M, M> + Copy, V: Add<V, V> + Copy> Transformation<Transform<M, V>> for Transform<M, V>
{
  fn transformation(&self) -> Transform<M, V>
  { copy *self }

  fn transform_by(&mut self, other: &Transform<M, V>)
  { *self = other * *self; }
}

// FIXME: constraints are too restrictive.
// Should be: Transformable<M2, // Transform<Res, V> ...
impl<M: RMul<V> + Mul<M, M>, V: Add<V, V>>
Transformable<Transform<M, V>, Transform<M, V>> for Transform<M, V>
{
  fn transformed(&self, t: &Transform<M, V>) -> Transform<M, V>
  { t * *self }
}

impl<M: Copy, V> DeltaTransform<M> for Transform<M, V>
{
  #[inline(always)]
  fn delta_transform(&self) -> M
  { copy self.submat }
}

impl<M: RMul<V>, V> DeltaTransformVector<V> for Transform<M, V>
{
  #[inline(always)]
  fn delta_transform_vector(&self, v: &V) -> V
  { self.submat.rmul(v) }
}

impl<M:Copy + Transpose + Inv + RMul<V>, V: Copy + Neg<V>>
Inv for Transform<M, V>
{
  #[inline(always)]
  fn invert(&mut self)
  {
    self.submat.invert();
    self.subtrans = self.submat.rmul(&-self.subtrans);
  }

  #[inline(always)]
  fn inverse(&self) -> Transform<M, V>
  {
    let mut res = copy *self;

    res.invert();

    res
  }
}

impl<N: ApproxEq<N>, M:ApproxEq<N>, V:ApproxEq<N>>
ApproxEq<N> for Transform<M, V>
{
  #[inline(always)]
  fn approx_epsilon() -> N
  { ApproxEq::approx_epsilon::<N, N>() }

  #[inline(always)]
  fn approx_eq(&self, other: &Transform<M, V>) -> bool
  {
    self.submat.approx_eq(&other.submat) &&
    self.subtrans.approx_eq(&other.subtrans)
  }

  #[inline(always)]
  fn approx_eq_eps(&self, other: &Transform<M, V>, epsilon: &N) -> bool
  {
    self.submat.approx_eq_eps(&other.submat, epsilon) &&
    self.subtrans.approx_eq_eps(&other.subtrans, epsilon)
  }
}

impl<M: Rand, V: Rand> Rand for Transform<M, V>
{
  #[inline(always)]
  fn rand<R: Rng>(rng: &mut R) -> Transform<M, V>
  { Transform::new(rng.gen(), rng.gen()) }
}
