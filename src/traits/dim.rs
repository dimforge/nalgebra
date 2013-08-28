/**
 * Trait of objects having a spacial dimension.
 */
pub trait Dim {
    /// The dimension of the object.
    fn dim(unused_self: Option<Self>) -> uint;
}
