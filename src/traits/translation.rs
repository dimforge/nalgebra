/// Trait of object which represent a translation, and to wich new translation
/// can be appended.
pub trait Translation<V, Res>
{
  /// Gets the translation associated with this object.
  fn translation(&self) -> V;

  /// Appends a translation from an alternative representation. Such
  /// representation has the same format as the one returned by `translation`.
  fn translated(&self, &V) -> Res;

  /// In-place version of `translate`.
  fn translate(&mut self, &V);
}
