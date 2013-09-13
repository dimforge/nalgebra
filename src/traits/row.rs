/// Traits to access rows of a matrix or vector.
pub trait Row<R> {
    /// The number of columun of this matrix or vector.
    fn num_rows(&self) -> uint;
    /// Reads the `i`-th row of `self`.
    fn row(&self, i: uint) -> R;
    /// Writes the `i`-th row of `self`.
    fn set_row(&mut self, i: uint, R);
}
