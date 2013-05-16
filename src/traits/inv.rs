pub trait Inv
{
  fn inverse(&self) -> Self;
  fn invert(&mut self);
}
