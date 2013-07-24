use traits::dot::Dot;

/// Traits of objects with a subtract and a dot product. Exists only for optimization purpose.
pub trait SubDot<N> : Sub<Self, Self> + Dot<N>
{
  /**
   * Short-cut to compute the projection of a point on a vector, but without
   * computing intermediate vectors.
   * This must be equivalent to:
   *
   *   (a - b).dot(c)
   *
   */
  fn sub_dot(&self, b: &Self, c: &Self) -> N;
}
