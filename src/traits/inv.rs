pub trait Inv
{
  /// Returns the inverse of an element.
  fn inverse(&self) -> Self;
  /// Inplace version of `inverse`.
  fn invert(&mut self);
}
