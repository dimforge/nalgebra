/**
 * Trait of objects having a dot product (also called inner product).
 */
pub trait Dot<N>
{
  /// Computes the dot (inner) product of two objects.
  fn dot(&self, &Self) -> N;
}
