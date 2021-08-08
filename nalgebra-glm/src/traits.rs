use approx::AbsDiffEq;
use num::{Bounded, Signed};

use na::Scalar;
use simba::scalar::{ClosedAdd, ClosedMul, ClosedSub, RealField, SupersetOf};
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
    + Bounded
    + SupersetOf<f64>
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
            + Bounded
            + SupersetOf<f64>,
    > Number for T
{
}

/// A number that can be any float type.
pub trait RealNumber: Number + RealField {}

impl<T: Number + RealField> RealNumber for T {}
