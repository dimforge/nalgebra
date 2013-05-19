pub trait Translation<V>
{
  fn translation(&self)   -> V;
  fn translated(&self, &V) -> Self;
  fn translate(&mut self, &V);
}
