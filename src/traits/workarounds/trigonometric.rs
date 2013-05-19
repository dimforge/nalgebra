// Trigonometric is available in core, but compilation fails with internal
// error.

use core::f64::{cos, sin, atan};

pub trait Trigonometric
{
  fn cos(Self)  -> Self;
  fn sin(Self)  -> Self;
  fn atan(Self) -> Self;
}

impl Trigonometric for f64
{
  fn cos(a: f64) -> f64
  { cos(a) }

  fn sin(a: f64) -> f64
  { sin(a) }

  fn atan(a: f64) -> f64
  { atan(a) }
}
