/// Trait of vectors which can be converted to another vector. Used to change the type of a vector
/// components.
pub trait VecCast<V> {
    /// Converts `v` to have the type `V`.
    fn from(v: Self) -> V;
}
