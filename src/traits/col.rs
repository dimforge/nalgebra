/// Traits to access columns of a matrix or vector.
pub trait Col<C> {
    /// The number of columun of this matrix or vector.
    fn num_cols(&self) -> uint;
    /// Reads the `i`-th column of `self`.
    fn col(&self, i: uint) -> C;
    /// Writes the `i`-th column of `self`.
    fn set_col(&mut self, i: uint, C);
}
