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

  /// Returns the normalized version of the argument.
  fn normalized(&self) -> Self;

  /// Inplace version of `normalized`.
  fn normalize(&mut self)  -> T;
}
