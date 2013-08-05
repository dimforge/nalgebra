// FIXME: this trait should not be on nalgebra.
// however, it is needed because std::ops::Index is (strangely) to poor: it
// does not have a function to set values.
// Also, using Index with tuples crashes.
/// This is a workaround trait.
///
/// It exists because the `Index` trait cannot be used to express write access.
/// Thus, this is the same as the `Index` trait but without the syntactic sugar and with a method
/// to write to a specific index.
pub trait Indexable<Index, Res>
{
    /// Reads the `i`-th element of `self`.
    fn at(&self, i: Index) -> Res;
    /// Writes to the `i`-th element of `self`.
    fn set(&mut self, i: Index, Res);
    /// Swaps the `i`-th element of `self` with its `j`-th element.
    fn swap(&mut self, i: Index, j: Index);
}
