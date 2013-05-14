// FIXME: this does not seem to exist already
// but it will surely be added someday.

pub trait Sqrt
{
  fn sqrt(&self) -> Self;
}

impl Sqrt for f64
{
  fn sqrt(&self) -> f64 { f64::sqrt(*self) }
}

impl Sqrt for f32
{
  fn sqrt(&self) -> f32 { f32::sqrt(*self) }
}
