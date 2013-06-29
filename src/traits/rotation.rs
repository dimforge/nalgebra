use traits::translation::{Translation, Translatable};

/// Trait of object which represent a rotation, and to wich new rotations can
/// be appended. A rotation is assumed to be an isomitry without translation
/// and without reflexion.
pub trait Rotation<V>
{
  /// Gets the rotation associated with this object.
  fn rotation(&self) -> V;

  fn inv_rotation(&self) -> V;

  /// In-place version of `rotated`.
  fn rotate_by(&mut self, &V);
}

pub trait Rotatable<V, Res: Rotation<V>>
{
  /// Appends a rotation from an alternative representation. Such
  /// representation has the same format as the one returned by `rotation`.
  fn rotated(&self, &V) -> Res;
}

pub trait Rotate<V>
{
  fn rotate(&self, &V)     -> V;
  fn inv_rotate(&self, &V) -> V;
}

/**
 * Applies a rotation centered on a specific point.
 *
 *   - `m`:       the object to be rotated.
 *   - `ammount`: the rotation to apply.
 *   - `point`:   the center of rotation.
 */
#[inline]
pub fn rotated_wrt_point<M: Translatable<LV, M2>,
                         M2: Rotation<AV> + Translation<LV>,
                         LV: Neg<LV>,
                         AV>
       (m: &M, ammount: &AV, center: &LV) -> M2
{
  let mut res = m.translated(&-center);

  res.rotate_by(ammount);
  res.translate_by(center);

  res
}

#[inline]
pub fn rotate_wrt_point<M:  Rotation<AV> + Translation<LV>,
                        LV: Neg<LV>,
                        AV>
       (m: &mut M, ammount: &AV, center: &LV)
{
  m.translate_by(&-center);
  m.rotate_by(ammount);
  m.translate_by(center);
}

/**
 * Applies a rotation centered on the input translation.
 *
 *   - `m`:       the object to be rotated.
 *   - `ammount`: the rotation to apply.
 */
#[inline]
pub fn rotated_wrt_center<M: Translatable<LV, M2> + Translation<LV>,
                          M2: Rotation<AV> + Translation<LV>,
                          LV: Neg<LV>,
                          AV>
       (m: &M, ammount: &AV) -> M2
{ rotated_wrt_point(m, ammount, &m.translation()) }

/**
 * Applies a rotation centered on the input translation.
 *
 *   - `m`:       the object to be rotated.
 *   - `ammount`: the rotation to apply.
 */
#[inline]
pub fn rotate_wrt_center<M: Translatable<LV, M> + Translation<LV> + Rotation<AV>,
                         LV: Neg<LV>,
                         AV>
       (m: &mut M, ammount: &AV)
{
  let t = m.translation();

  rotate_wrt_point(m, ammount, &t)
}
