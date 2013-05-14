// FIXME: valid only for square matrices…
pub trait Transpose
{
  fn transposed(&self) -> Self;
  fn transpose(&mut self);
}
