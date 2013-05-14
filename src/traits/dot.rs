pub trait Dot<T>
{
  fn dot(&self, &Self) -> T;
  fn norm(&self)       -> T;
  fn sqnorm(&self)     -> T; // { self.dot(self); }
}
