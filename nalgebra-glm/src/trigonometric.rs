use na::{self, Real, DimName, DefaultAllocator};

use aliases::Vec;
use traits::Alloc;

pub fn acos<N: Real, D: DimName>(x: &Vec<N, D>) -> Vec<N, D>
    where DefaultAllocator: Alloc<N, D> {
    x.map(|e| e.acos())
}
pub fn acosh<N: Real, D: DimName>(x: &Vec<N, D>) -> Vec<N, D>
    where DefaultAllocator: Alloc<N, D> {
    x.map(|e| e.acosh())
}
pub fn asin<N: Real, D: DimName>(x: &Vec<N, D>) -> Vec<N, D>
    where DefaultAllocator: Alloc<N, D> {
    x.map(|e| e.asin())
}
pub fn asinh<N: Real, D: DimName>(x: &Vec<N, D>) -> Vec<N, D>
    where DefaultAllocator: Alloc<N, D> {
    x.map(|e| e.asinh())
}
pub fn atan2<N: Real, D: DimName>(y: &Vec<N, D>, x: &Vec<N, D>) -> Vec<N, D>
    where DefaultAllocator: Alloc<N, D> {
    y.zip_map(x, |y, x| y.atan2(x))
}
pub fn atan<N: Real, D: DimName>(y_over_x: &Vec<N, D>) -> Vec<N, D>
    where DefaultAllocator: Alloc<N, D> {
    y_over_x.map(|e| e.atan())
}
pub fn atanh<N: Real, D: DimName>(x: &Vec<N, D>) -> Vec<N, D>
    where DefaultAllocator: Alloc<N, D> {
    x.map(|e| e.atanh())
}

pub fn cos<N: Real, D: DimName>(angle: &Vec<N, D>) -> Vec<N, D>
    where DefaultAllocator: Alloc<N, D> {
    angle.map(|e| e.cos())
}

pub fn cosh<N: Real, D: DimName>(angle: &Vec<N, D>) -> Vec<N, D>
    where DefaultAllocator: Alloc<N, D> {
    angle.map(|e| e.cosh())
}

pub fn degrees<N: Real, D: DimName>(radians: &Vec<N, D>) -> Vec<N, D>
    where DefaultAllocator: Alloc<N, D> {
    radians.map(|e| e * na::convert(180.0) / N::pi())
}

pub fn radians<N: Real, D: DimName>(degrees: &Vec<N, D>) -> Vec<N, D>
    where DefaultAllocator: Alloc<N, D> {
    degrees.map(|e| e * N::pi() / na::convert(180.0))
}

pub fn sin<N: Real, D: DimName>(angle: &Vec<N, D>) -> Vec<N, D>
    where DefaultAllocator: Alloc<N, D> {
    angle.map(|e| e.sin())
}

pub fn sinh<N: Real, D: DimName>(angle: &Vec<N, D>) -> Vec<N, D>
    where DefaultAllocator: Alloc<N, D> {
    angle.map(|e| e.sinh())
}

pub fn tan<N: Real, D: DimName>(angle: &Vec<N, D>) -> Vec<N, D>
    where DefaultAllocator: Alloc<N, D> {
    angle.map(|e| e.tan())
}

pub fn tanh<N: Real, D: DimName>(angle: &Vec<N, D>) -> Vec<N, D>
    where DefaultAllocator: Alloc<N, D> {
    angle.map(|e| e.tanh())
}