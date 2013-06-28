/// Trait of object which represent a translation, and to wich new translation
/// can be appended.
pub trait Translation<V>
{
  // FIXME: add a "from translation: translantion(V) -> Self ?
  /// Gets the translation associated with this object.
  fn translation(&self) -> V;

  fn inv_translation(&self) -> V;

  /// In-place version of `translate`.
  fn translate_by(&mut self, &V);
}

pub trait Translate<V>
{
  fn translate(&self, &V)     -> V;
  fn inv_translate(&self, &V) -> V;
}

pub trait Translatable<V, Res: Translation<V>>
{
  /// Appends a translation from an alternative representation. Such
  /// representation has the same format as the one returned by `translation`.
  fn translated(&self, &V) -> Res;
}
