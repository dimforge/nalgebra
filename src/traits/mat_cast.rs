/// Trait of matrices which can be converted to another matrix. Used to change the type of a matrix
/// components.
pub trait MatCast<M> {
    /// Converts `m` to have the type `M`.
    fn from(m: Self) -> M;
}
