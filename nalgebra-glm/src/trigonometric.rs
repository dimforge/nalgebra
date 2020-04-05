use na::{self, DefaultAllocator, RealField};

use crate::aliases::TVec;
use crate::traits::{Alloc, Dimension};

/// Component-wise arc-cosinus.
pub fn acos<N: RealField, D: Dimension>(x: &TVec<N, D>) -> TVec<N, D>
where
    DefaultAllocator: Alloc<N, D>,
{
    x.map(|e| e.acos())
}

/// Component-wise hyperbolic arc-cosinus.
pub fn acosh<N: RealField, D: Dimension>(x: &TVec<N, D>) -> TVec<N, D>
where
    DefaultAllocator: Alloc<N, D>,
{
    x.map(|e| e.acosh())
}

/// Component-wise arc-sinus.
pub fn asin<N: RealField, D: Dimension>(x: &TVec<N, D>) -> TVec<N, D>
where
    DefaultAllocator: Alloc<N, D>,
{
    x.map(|e| e.asin())
}

/// Component-wise hyperbolic arc-sinus.
pub fn asinh<N: RealField, D: Dimension>(x: &TVec<N, D>) -> TVec<N, D>
where
    DefaultAllocator: Alloc<N, D>,
{
    x.map(|e| e.asinh())
}

/// Component-wise arc-tangent of `y / x`.
pub fn atan2<N: RealField, D: Dimension>(y: &TVec<N, D>, x: &TVec<N, D>) -> TVec<N, D>
where
    DefaultAllocator: Alloc<N, D>,
{
    y.zip_map(x, |y, x| y.atan2(x))
}

/// Component-wise arc-tangent.
pub fn atan<N: RealField, D: Dimension>(y_over_x: &TVec<N, D>) -> TVec<N, D>
where
    DefaultAllocator: Alloc<N, D>,
{
    y_over_x.map(|e| e.atan())
}

/// Component-wise hyperbolic arc-tangent.
pub fn atanh<N: RealField, D: Dimension>(x: &TVec<N, D>) -> TVec<N, D>
where
    DefaultAllocator: Alloc<N, D>,
{
    x.map(|e| e.atanh())
}

/// Component-wise cosinus.
pub fn cos<N: RealField, D: Dimension>(angle: &TVec<N, D>) -> TVec<N, D>
where
    DefaultAllocator: Alloc<N, D>,
{
    angle.map(|e| e.cos())
}

/// Component-wise hyperbolic cosinus.
pub fn cosh<N: RealField, D: Dimension>(angle: &TVec<N, D>) -> TVec<N, D>
where
    DefaultAllocator: Alloc<N, D>,
{
    angle.map(|e| e.cosh())
}

/// Component-wise conversion from radians to degrees.
pub fn degrees<N: RealField, D: Dimension>(radians: &TVec<N, D>) -> TVec<N, D>
where
    DefaultAllocator: Alloc<N, D>,
{
    radians.map(|e| e * na::convert(180.0) / N::pi())
}

/// Component-wise conversion fro degrees to radians.
pub fn radians<N: RealField, D: Dimension>(degrees: &TVec<N, D>) -> TVec<N, D>
where
    DefaultAllocator: Alloc<N, D>,
{
    degrees.map(|e| e * N::pi() / na::convert(180.0))
}

/// Component-wise sinus.
pub fn sin<N: RealField, D: Dimension>(angle: &TVec<N, D>) -> TVec<N, D>
where
    DefaultAllocator: Alloc<N, D>,
{
    angle.map(|e| e.sin())
}

/// Component-wise hyperbolic sinus.
pub fn sinh<N: RealField, D: Dimension>(angle: &TVec<N, D>) -> TVec<N, D>
where
    DefaultAllocator: Alloc<N, D>,
{
    angle.map(|e| e.sinh())
}

/// Component-wise tangent.
pub fn tan<N: RealField, D: Dimension>(angle: &TVec<N, D>) -> TVec<N, D>
where
    DefaultAllocator: Alloc<N, D>,
{
    angle.map(|e| e.tan())
}

/// Component-wise hyperbolic tangent.
pub fn tanh<N: RealField, D: Dimension>(angle: &TVec<N, D>) -> TVec<N, D>
where
    DefaultAllocator: Alloc<N, D>,
{
    angle.map(|e| e.tanh())
}
