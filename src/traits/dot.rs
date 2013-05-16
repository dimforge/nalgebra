pub trait Dot<T>
{
  /// Computes the dot (inner) product of two objects.
  fn dot(&self, &Self) -> T;
  /// Computes the norm a an object.
  fn norm(&self)       -> T;
  /**
   * Computes the squared norm of an object.
   * 
   * Computes the squared norm of an object. Computation of the squared norm
   * is usually faster than the norm itself.
   */
  fn sqnorm(&self)     -> T; // { self.dot(self); }
}
