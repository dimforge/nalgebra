/// Trait of object which represent a translation, and to wich new translation
/// can be appended.
pub trait Translation<V> {
    // FIXME: add a "from translation: translantion(V) -> Self ?
    /// Gets the translation associated with this object.
    fn translation(&self) -> V;

    /// Gets the inverse translation associated with this object.
    fn inv_translation(&self) -> V;

    /// In-place version of `translated`.
    fn translate_by(&mut self, &V);

    /// Appends a translation.
    fn translated(&self, &V) -> Self;

    /// Sets the translation.
    fn set_translation(&mut self, V);
}

/// Trait of objects able to rotate other objects. This is typically implemented by matrices which
/// rotate vectors.
pub trait Translate<V> {
    /// Apply a translation to an object.
    fn translate(&self, &V)     -> V;
    /// Apply an inverse translation to an object.
    fn inv_translate(&self, &V) -> V;
}
