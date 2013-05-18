pub trait Dot<T>
{
  /// Computes the dot (inner) product of two objects.
  fn dot(&self, &Self) -> T;
}
