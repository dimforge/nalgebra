use approx::AbsDiffEq;
use num::{Bounded, FromPrimitive, Signed};

use na::Scalar;
use simba::scalar::{ClosedAdd, ClosedMul, ClosedSub};
use std::cmp::PartialOrd;

/// A number that can either be an integer or a float.
pub trait Number:
    Scalar
    + Copy
    + PartialOrd
    + ClosedAdd
    + ClosedSub
    + ClosedMul
    + AbsDiffEq<Epsilon = Self>
    + Signed
    + FromPrimitive
    + Bounded
{
}

impl<
        T: Scalar
            + Copy
            + PartialOrd
            + ClosedAdd
            + ClosedSub
            + ClosedMul
            + AbsDiffEq<Epsilon = Self>
            + Signed
            + FromPrimitive
            + Bounded,
    > Number for T
{
}
