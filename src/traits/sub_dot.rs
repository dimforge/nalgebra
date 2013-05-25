pub trait SubDot<T>
{
  /**
   * Short-cut to compute the projecton of a point on a vector, but without
   * computing intermediate vectors.
   * This must be equivalent to:
   *
   *   (a - b).dot(c)
   *
   */
  fn sub_dot(&self, b: &Self, c: &Self) -> T;
}
