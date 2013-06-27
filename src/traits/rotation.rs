use traits::translation::{Translation, Translatable};

/// Trait of object which represent a rotation, and to wich new rotations can
/// be appended. A rotation is assumed to be an isomitry without translation
/// and without reflexion.
pub trait Rotation<V>
{
  /// Gets the rotation associated with this object.
  fn rotation(&self) -> V;

  /// In-place version of `rotated`.
  fn rotate(&mut self, &V);
}

pub trait Rotatable<V, Res: Rotation<V>>
{
  /// Appends a rotation from an alternative representation. Such
  /// representation has the same format as the one returned by `rotation`.
  fn rotated(&self, &V) -> Res;
}

/**
 * Applies a rotation centered on a specific point.
 *
 *   - `m`:       the object to be rotated.
 *   - `ammount`: the rotation to apply.
 *   - `point`:   the center of rotation.
 */
#[inline(always)]
pub fn rotate_wrt_point<M: Translatable<LV, M2>,
                        M2: Rotation<AV> + Translation<LV>,
                        LV: Neg<LV>,
                        AV>
       (m: &M, ammount: &AV, center: &LV) -> M2
{
  let mut res = m.translated(&-center);

  res.rotate(ammount);
  res.translate(center);

  res
}
