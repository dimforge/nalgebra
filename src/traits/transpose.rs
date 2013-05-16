// FIXME: valid only for square matrices…
pub trait Transpose
{
  /// Computes the transpose of a matrix.
  fn transposed(&self) -> Self;
  /// Inplace version of `transposed`.
  fn transpose(&mut self);
}
