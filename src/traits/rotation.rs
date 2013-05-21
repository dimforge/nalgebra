/// Trait of object which represent a rotation, and to wich new rotations can
/// be appended. A rotation is assumed to be an isomitry without translation
/// and without reflexion.
pub trait Rotation<V>
{
  /// Gets the rotation associated with this object.
  fn rotation(&self)   -> V;

  /// Appends a rotation from an alternative representation. Such
  /// representation has the same format as the one returned by `rotation`.
  fn rotated(&self, &V) -> Self;

  /// In-place version of `rotated`.
  fn rotate(&mut self, &V);
}
