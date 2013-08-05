/// Traits to access columns of a matrix.
pub trait Column<C> {
    /// Reads the `i`-th column of `self`.
    fn column(&self, i: uint) -> C;
    /// Writes the `i`-th column of `self`.
    fn set_column(&mut self, i: uint, C);
}
