use na::{self, Real, DefaultAllocator};

use aliases::Vec;
use traits::{Alloc, Dimension};


/// Componentwise arc-cosinus.
pub fn acos<N: Real, D: Dimension>(x: &Vec<N, D>) -> Vec<N, D>
    where DefaultAllocator: Alloc<N, D> {
    x.map(|e| e.acos())
}

/// Componentwise hyperbolic arc-cosinus.
pub fn acosh<N: Real, D: Dimension>(x: &Vec<N, D>) -> Vec<N, D>
    where DefaultAllocator: Alloc<N, D> {
    x.map(|e| e.acosh())
}

/// Componentwise arc-sinus.
pub fn asin<N: Real, D: Dimension>(x: &Vec<N, D>) -> Vec<N, D>
    where DefaultAllocator: Alloc<N, D> {
    x.map(|e| e.asin())
}

/// Componentwise hyperbolic arc-sinus.
pub fn asinh<N: Real, D: Dimension>(x: &Vec<N, D>) -> Vec<N, D>
    where DefaultAllocator: Alloc<N, D> {
    x.map(|e| e.asinh())
}

/// Componentwise arc-tangent of `y / x`.
pub fn atan2<N: Real, D: Dimension>(y: &Vec<N, D>, x: &Vec<N, D>) -> Vec<N, D>
    where DefaultAllocator: Alloc<N, D> {
    y.zip_map(x, |y, x| y.atan2(x))
}

/// Componentwise arc-tangent.
pub fn atan<N: Real, D: Dimension>(y_over_x: &Vec<N, D>) -> Vec<N, D>
    where DefaultAllocator: Alloc<N, D> {
    y_over_x.map(|e| e.atan())
}

/// Componentwise hyperbolic arc-tangent.
pub fn atanh<N: Real, D: Dimension>(x: &Vec<N, D>) -> Vec<N, D>
    where DefaultAllocator: Alloc<N, D> {
    x.map(|e| e.atanh())
}

/// Componentwise cosinus.
pub fn cos<N: Real, D: Dimension>(angle: &Vec<N, D>) -> Vec<N, D>
    where DefaultAllocator: Alloc<N, D> {
    angle.map(|e| e.cos())
}

/// Componentwise hyperbolic cosinus.
pub fn cosh<N: Real, D: Dimension>(angle: &Vec<N, D>) -> Vec<N, D>
    where DefaultAllocator: Alloc<N, D> {
    angle.map(|e| e.cosh())
}

/// Componentwise conversion from radians to degrees.
pub fn degrees<N: Real, D: Dimension>(radians: &Vec<N, D>) -> Vec<N, D>
    where DefaultAllocator: Alloc<N, D> {
    radians.map(|e| e * na::convert(180.0) / N::pi())
}

/// Componentwise conversion fro degrees to radians.
pub fn radians<N: Real, D: Dimension>(degrees: &Vec<N, D>) -> Vec<N, D>
    where DefaultAllocator: Alloc<N, D> {
    degrees.map(|e| e * N::pi() / na::convert(180.0))
}

/// Componentwise sinus.
pub fn sin<N: Real, D: Dimension>(angle: &Vec<N, D>) -> Vec<N, D>
    where DefaultAllocator: Alloc<N, D> {
    angle.map(|e| e.sin())
}

/// Componentwise hyperbolic sinus.
pub fn sinh<N: Real, D: Dimension>(angle: &Vec<N, D>) -> Vec<N, D>
    where DefaultAllocator: Alloc<N, D> {
    angle.map(|e| e.sinh())
}

/// Componentwise tangent.
pub fn tan<N: Real, D: Dimension>(angle: &Vec<N, D>) -> Vec<N, D>
    where DefaultAllocator: Alloc<N, D> {
    angle.map(|e| e.tan())
}

/// Componentwise hyperbolic tangent.
pub fn tanh<N: Real, D: Dimension>(angle: &Vec<N, D>) -> Vec<N, D>
    where DefaultAllocator: Alloc<N, D> {
    angle.map(|e| e.tanh())
}
