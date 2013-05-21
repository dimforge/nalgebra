// FIXME: valid only for square matrices…
/// Trait of objects which can be transposed. Note that, for the moment, this
/// does not allow the implementation by non-square matrix (or anything which
/// is not stable by transposition).
pub trait Transpose
{
  /// Computes the transpose of a matrix.
  fn transposed(&self) -> Self;

  /// In-place version of `transposed`.
  fn transpose(&mut self);
}
