/**
 * Trait of objects having a L² norm and which can be normalized.
 */
pub trait Norm<T>
{
  /// Computes the norm a an object.
  fn norm(&self)       -> T;

  /**
   * Computes the squared norm of an object.
   * 
   * Computes the squared norm of an object. Computation of the squared norm
   * is usually faster than the norm itself.
   */
  fn sqnorm(&self)     -> T;

  /// Gets the normalized version of the argument.
  fn normalized(&self) -> Self;

  /// In-place version of `normalized`.
  fn normalize(&mut self)  -> T;
}
