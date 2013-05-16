// Trigonometric is available in core, but compilation fails with internal
// error.

use core::f64::{cos, sin};

pub trait Trigonometric
{
  fn cos(Self) -> Self;
  fn sin(Self) -> Self;
}

impl Trigonometric for f64
{
  fn cos(a: f64) -> f64
  { cos(a) }

  fn sin(a: f64) -> f64
  { sin(a) }
}
