use std::cmp::{PartialOrd, PartialEq};

use alga::general::Ring;
use na::{Scalar, DimName, U1};
use na::allocator::Allocator;


pub trait Number: Scalar + Ring + PartialOrd + PartialEq {
}

impl<T: Scalar + Ring + PartialOrd + PartialEq> Number for T {
}

#[doc(hidden)]
pub trait Alloc<N: Scalar, R: DimName, C: DimName = U1>:
Allocator<N, R> + Allocator<N, C> + Allocator<N, U1, R> + Allocator<N, U1, C> + Allocator<N, R, C> + Allocator<N, C, R> + Allocator<N, R, R> + Allocator<N, C, C> +
Allocator<bool, R> + Allocator<bool, C> +
Allocator<f32, R> + Allocator<f32, C> +
Allocator<u32, R> + Allocator<u32, C> +
Allocator<i32, R> + Allocator<i32, C> +
Allocator<f64, R> + Allocator<f64, C> +
Allocator<u64, R> + Allocator<u64, C> +
Allocator<i64, R> + Allocator<i64, C> +
Allocator<i16, R> + Allocator<i16, C>
{
}

impl<N: Scalar, R: DimName, C: DimName, T>
    Alloc<N, R, C> for T
where T: Allocator<N, R> + Allocator<N, C> + Allocator<N, U1, R> + Allocator<N, U1, C> + Allocator<N, R, C> + Allocator<N, C, R> + Allocator<N, R, R> + Allocator<N, C, C> +
         Allocator<bool, R> + Allocator<bool, C> +
         Allocator<f32, R> + Allocator<f32, C> +
         Allocator<u32, R> + Allocator<u32, C> +
         Allocator<i32, R> + Allocator<i32, C> +
         Allocator<f64, R> + Allocator<f64, C> +
         Allocator<u64, R> + Allocator<u64, C> +
         Allocator<i64, R> + Allocator<i64, C> +
         Allocator<i16, R> + Allocator<i16, C> {
}