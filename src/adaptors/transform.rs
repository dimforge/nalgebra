use std::num::{One, Zero};
use std::rand::{Rand, Rng, RngUtil};
use std::cmp::ApproxEq;
use traits::dim::Dim;
use traits::inv::Inv;
use traits::rotation::{Rotation, Rotate, Rotatable};
use traits::translation::{Translation, Translate, Translatable};
use traits::transformation;
use traits::transformation::{Transformation, Transformable};
use traits::rlmul::{RMul, LMul};
use traits::homogeneous::{ToHomogeneous, FromHomogeneous};
use traits::column::Column;

#[deriving(Eq, ToStr)]
pub struct Transform<M, V>
{
  priv submat   : M,
  priv subtrans : V
}

impl<M, V> Transform<M, V>
{
  #[inline]
  pub fn new(mat: M, trans: V) -> Transform<M, V>
  { Transform { submat: mat, subtrans: trans } }
}

impl<M:Dim, V> Dim for Transform<M, V>
{
  #[inline]
  fn dim() -> uint
  { Dim::dim::<M>() }
}

impl<M: One, V: Zero> One for Transform<M, V>
{
  #[inline]
  fn one() -> Transform<M, V>
  { Transform { submat: One::one(), subtrans: Zero::zero() } }
}

impl<M: Zero, V: Zero> Zero for Transform<M, V>
{
  #[inline]
  fn zero() -> Transform<M, V>
  { Transform { submat: Zero::zero(), subtrans: Zero::zero() } }

  #[inline]
  fn is_zero(&self) -> bool
  { self.submat.is_zero() && self.subtrans.is_zero() }
}

impl<M: RMul<V> + Mul<M, M>, V: Add<V, V>>
Mul<Transform<M, V>, Transform<M, V>> for Transform<M, V>
{
  #[inline]
  fn mul(&self, other: &Transform<M, V>) -> Transform<M, V>
  {
    Transform { submat: self.submat * other.submat,
                subtrans: self.subtrans + self.submat.rmul(&other.subtrans) }
  }
}

impl<M: RMul<V>, V: Add<V, V>> RMul<V> for Transform<M, V>
{
  #[inline]
  fn rmul(&self, other: &V) -> V
  { self.submat.rmul(other) + self.subtrans }
}

impl<M: LMul<V>, V: Add<V, V>> LMul<V> for Transform<M, V>
{
  #[inline]
  fn lmul(&self, other: &V) -> V
  { self.submat.lmul(other) + self.subtrans }
}

impl<M, V: Translation<V>> Translation<V> for Transform<M, V>
{
  #[inline]
  fn translation(&self) -> V
  { self.subtrans.translation() }

  #[inline]
  fn inv_translation(&self) -> V
  { self.subtrans.inv_translation() }

  #[inline]
  fn translate_by(&mut self, t: &V)
  { self.subtrans.translate_by(t) }
}

impl<M: Translate<V>, V, _0> Translate<V> for Transform<M, _0>
{
  #[inline]
  fn translate(&self, v: &V) -> V
  { self.submat.translate(v) }

  #[inline]
  fn inv_translate(&self, v: &V) -> V
  { self.submat.inv_translate(v) }
}

impl<M: Copy, V: Translatable<V, V> + Translation<V>>
Translatable<V, Transform<M, V>> for Transform<M, V>
{
  #[inline]
  fn translated(&self, t: &V) -> Transform<M, V>
  { Transform::new(copy self.submat, self.subtrans.translated(t)) }
}

impl<M: Rotation<AV> + RMul<V> + One,
     V,
     AV>
Rotation<AV> for Transform<M, V>
{
  #[inline]
  fn rotation(&self) -> AV
  { self.submat.rotation() }

  #[inline]
  fn inv_rotation(&self) -> AV
  { self.submat.inv_rotation() }


  #[inline]
  fn rotate_by(&mut self, rot: &AV)
  {
    // FIXME: this does not seem opitmal
    let mut delta = One::one::<M>();
    delta.rotate_by(rot);
    self.submat.rotate_by(rot);
    self.subtrans = delta.rmul(&self.subtrans);
  }
}

impl<M: Rotate<V>, V, _0> Rotate<V> for Transform<M, _0>
{
  #[inline]
  fn rotate(&self, v: &V) -> V
  { self.submat.rotate(v) }

  #[inline]
  fn inv_rotate(&self, v: &V) -> V
  { self.submat.inv_rotate(v) }
}

impl<M: Rotatable<AV, Res> + One,
     Res: Rotation<AV> + RMul<V> + One,
     V,
     AV>
Rotatable<AV, Transform<Res, V>> for Transform<M, V>
{
  #[inline]
  fn rotated(&self, rot: &AV) -> Transform<Res, V>
  {
    // FIXME: this does not seem opitmal
    let delta = One::one::<M>().rotated(rot);

    Transform::new(self.submat.rotated(rot), delta.rmul(&self.subtrans))
  }
}

impl<M: Inv + RMul<V> + Mul<M, M> + Copy, V: Add<V, V> + Neg<V> + Copy>
Transformation<Transform<M, V>> for Transform<M, V>
{
  fn transformation(&self) -> Transform<M, V>
  { copy *self }

  fn inv_transformation(&self) -> Transform<M, V>
  { self.inverse() }

  fn transform_by(&mut self, other: &Transform<M, V>)
  { *self = other * *self; }
}

impl<M: transformation::Transform<V>, V: Add<V, V> + Sub<V, V>>
transformation::Transform<V> for Transform<M, V>
{
  #[inline]
  fn transform_vec(&self, v: &V) -> V
  { self.submat.transform_vec(v) + self.subtrans }

  #[inline]
  fn inv_transform(&self, v: &V) -> V
  { self.submat.inv_transform(&(v - self.subtrans)) }
}


// FIXME: constraints are too restrictive.
// Should be: Transformable<M2, // Transform<Res, V> ...
impl<M: RMul<V> + Mul<M, M> + Inv, V: Add<V, V> + Neg<V>>
Transformable<Transform<M, V>, Transform<M, V>> for Transform<M, V>
{
  fn transformed(&self, t: &Transform<M, V>) -> Transform<M, V>
  { t * *self }
}

impl<M: Copy + Inv + RMul<V>, V: Copy + Neg<V>>
Inv for Transform<M, V>
{
  #[inline]
  fn invert(&mut self)
  {
    self.submat.invert();
    self.subtrans = self.submat.rmul(&-self.subtrans);
  }

  #[inline]
  fn inverse(&self) -> Transform<M, V>
  {
    let mut res = copy *self;

    res.invert();

    res
  }
}

impl<M: ToHomogeneous<M2>, M2: Dim + Column<V>, V: Copy>
ToHomogeneous<M2> for Transform<M, V>
{
  fn to_homogeneous(&self) -> M2
  {
    let mut res = self.submat.to_homogeneous();

    // copy the translation
    let dim = Dim::dim::<M2>();

    res.set_column(dim - 1, copy self.subtrans);

    res
  }
}

impl<M: Column<V> + Dim, M2: FromHomogeneous<M>, V: Copy>
FromHomogeneous<M> for Transform<M2, V>
{
  fn from_homogeneous(m: &M) -> Transform<M2, V>
  {
    Transform::new(FromHomogeneous::from_homogeneous(m),
                   m.column(Dim::dim::<M>() - 1))
  }
}

impl<N: ApproxEq<N>, M:ApproxEq<N>, V:ApproxEq<N>>
ApproxEq<N> for Transform<M, V>
{
  #[inline]
  fn approx_epsilon() -> N
  { ApproxEq::approx_epsilon::<N, N>() }

  #[inline]
  fn approx_eq(&self, other: &Transform<M, V>) -> bool
  {
    self.submat.approx_eq(&other.submat) &&
    self.subtrans.approx_eq(&other.subtrans)
  }

  #[inline]
  fn approx_eq_eps(&self, other: &Transform<M, V>, epsilon: &N) -> bool
  {
    self.submat.approx_eq_eps(&other.submat, epsilon) &&
    self.subtrans.approx_eq_eps(&other.subtrans, epsilon)
  }
}

impl<M: Rand, V: Rand> Rand for Transform<M, V>
{
  #[inline]
  fn rand<R: Rng>(rng: &mut R) -> Transform<M, V>
  { Transform::new(rng.gen(), rng.gen()) }
}
