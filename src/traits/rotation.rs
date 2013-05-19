pub trait Rotation<V>
{
  fn rotation(&self)   -> V;
  fn rotated(&self, &V) -> Self;
  fn rotate(&mut self, &V);
}
