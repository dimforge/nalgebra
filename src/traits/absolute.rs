/// Trait of objects having an absolute value.
/// This is useful of the object and the absolute value do not have the same type.
pub trait Absolute<A> {
    /// Compute some absolute representation of this object.
    /// Typically, this will make all component of a matrix or vector positive.
    fn absolute(&self) -> A;
}
