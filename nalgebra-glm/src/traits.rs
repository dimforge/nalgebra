use approx::AbsDiffEq;
use num::{Bounded, FromPrimitive, Signed};

use na::allocator::Allocator;
use na::{DimMin, DimName, Scalar, U1};
use simba::scalar::{ClosedAdd, ClosedMul, ClosedSub};
use std::cmp::PartialOrd;

/// A type-level number representing a vector, matrix row, or matrix column, dimension.
pub trait Dimension: DimName + DimMin<Self, Output = Self> {}
impl<D: DimName + DimMin<D, Output = Self>> Dimension for D {}

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

#[doc(hidden)]
pub trait Alloc<N: Scalar, R: Dimension, C: Dimension = U1>:
    Allocator<N, R>
    + Allocator<N, C>
    + Allocator<N, U1, R>
    + Allocator<N, U1, C>
    + Allocator<N, R, C>
    + Allocator<N, C, R>
    + Allocator<N, R, R>
    + Allocator<N, C, C>
    + Allocator<bool, R>
    + Allocator<bool, C>
    + Allocator<f32, R>
    + Allocator<f32, C>
    + Allocator<u32, R>
    + Allocator<u32, C>
    + Allocator<i32, R>
    + Allocator<i32, C>
    + Allocator<f64, R>
    + Allocator<f64, C>
    + Allocator<u64, R>
    + Allocator<u64, C>
    + Allocator<i64, R>
    + Allocator<i64, C>
    + Allocator<i16, R>
    + Allocator<i16, C>
    + Allocator<(usize, usize), R>
    + Allocator<(usize, usize), C>
{
}

impl<N: Scalar, R: Dimension, C: Dimension, T> Alloc<N, R, C> for T where
    T: Allocator<N, R>
        + Allocator<N, C>
        + Allocator<N, U1, R>
        + Allocator<N, U1, C>
        + Allocator<N, R, C>
        + Allocator<N, C, R>
        + Allocator<N, R, R>
        + Allocator<N, C, C>
        + Allocator<bool, R>
        + Allocator<bool, C>
        + Allocator<f32, R>
        + Allocator<f32, C>
        + Allocator<u32, R>
        + Allocator<u32, C>
        + Allocator<i32, R>
        + Allocator<i32, C>
        + Allocator<f64, R>
        + Allocator<f64, C>
        + Allocator<u64, R>
        + Allocator<u64, C>
        + Allocator<i64, R>
        + Allocator<i64, C>
        + Allocator<i16, R>
        + Allocator<i16, C>
        + Allocator<(usize, usize), R>
        + Allocator<(usize, usize), C>
{
}
