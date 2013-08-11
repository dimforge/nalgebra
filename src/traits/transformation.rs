/// Trait of object which represent a transformation, and to wich new transformations can
/// be appended. A transformation is assumed to be an isomitry without translation
/// and without reflexion.
pub trait Transformation<M> {
    /// Gets the transformation associated with this object.
    fn transformation(&self) -> M;

    /// Gets the inverse transformation associated with this object.
    fn inv_transformation(&self) -> M;

    /// In-place version of `transformed` (see the `Transformable` trait).
    fn transform_by(&mut self, &M);
}

/// Trait of objects able to transform other objects. This is typically implemented by matrices which
/// transform vectors.
pub trait Transform<V> {
    /// Apply a transformation to an object.
    fn transform(&self, &V) -> V;
    /// Apply an inverse transformation to an object.
    fn inv_transform(&self, &V) -> V;
}

/// Trait of objects which can be put on an alternate form which represent a transformation. This is
/// typically implemented by structures requiring an internal restructuration to be able to
/// represent a transformation.
pub trait Transformable<M, Res: Transformation<M>> {
    /// Appends a transformation from an alternative representation. Such
    /// representation has the same format as the one returned by `transformation`.
    fn transformed(&self, &M) -> Res;
}
