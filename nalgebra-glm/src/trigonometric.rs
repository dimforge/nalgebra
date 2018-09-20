use na::{self, Real, DefaultAllocator};

use aliases::Vec;
use traits::{Alloc, Dimension};

pub fn acos<N: Real, D: Dimension>(x: &Vec<N, D>) -> Vec<N, D>
    where DefaultAllocator: Alloc<N, D> {
    x.map(|e| e.acos())
}

pub fn acosh<N: Real, D: Dimension>(x: &Vec<N, D>) -> Vec<N, D>
    where DefaultAllocator: Alloc<N, D> {
    x.map(|e| e.acosh())
}

pub fn asin<N: Real, D: Dimension>(x: &Vec<N, D>) -> Vec<N, D>
    where DefaultAllocator: Alloc<N, D> {
    x.map(|e| e.asin())
}

pub fn asinh<N: Real, D: Dimension>(x: &Vec<N, D>) -> Vec<N, D>
    where DefaultAllocator: Alloc<N, D> {
    x.map(|e| e.asinh())
}

pub fn atan2<N: Real, D: Dimension>(y: &Vec<N, D>, x: &Vec<N, D>) -> Vec<N, D>
    where DefaultAllocator: Alloc<N, D> {
    y.zip_map(x, |y, x| y.atan2(x))
}

pub fn atan<N: Real, D: Dimension>(y_over_x: &Vec<N, D>) -> Vec<N, D>
    where DefaultAllocator: Alloc<N, D> {
    y_over_x.map(|e| e.atan())
}

pub fn atanh<N: Real, D: Dimension>(x: &Vec<N, D>) -> Vec<N, D>
    where DefaultAllocator: Alloc<N, D> {
    x.map(|e| e.atanh())
}

pub fn cos<N: Real, D: Dimension>(angle: &Vec<N, D>) -> Vec<N, D>
    where DefaultAllocator: Alloc<N, D> {
    angle.map(|e| e.cos())
}

pub fn cosh<N: Real, D: Dimension>(angle: &Vec<N, D>) -> Vec<N, D>
    where DefaultAllocator: Alloc<N, D> {
    angle.map(|e| e.cosh())
}

pub fn degrees<N: Real, D: Dimension>(radians: &Vec<N, D>) -> Vec<N, D>
    where DefaultAllocator: Alloc<N, D> {
    radians.map(|e| e * na::convert(180.0) / N::pi())
}

pub fn radians<N: Real, D: Dimension>(degrees: &Vec<N, D>) -> Vec<N, D>
    where DefaultAllocator: Alloc<N, D> {
    degrees.map(|e| e * N::pi() / na::convert(180.0))
}

pub fn sin<N: Real, D: Dimension>(angle: &Vec<N, D>) -> Vec<N, D>
    where DefaultAllocator: Alloc<N, D> {
    angle.map(|e| e.sin())
}

pub fn sinh<N: Real, D: Dimension>(angle: &Vec<N, D>) -> Vec<N, D>
    where DefaultAllocator: Alloc<N, D> {
    angle.map(|e| e.sinh())
}

pub fn tan<N: Real, D: Dimension>(angle: &Vec<N, D>) -> Vec<N, D>
    where DefaultAllocator: Alloc<N, D> {
    angle.map(|e| e.tan())
}

pub fn tanh<N: Real, D: Dimension>(angle: &Vec<N, D>) -> Vec<N, D>
    where DefaultAllocator: Alloc<N, D> {
    angle.map(|e| e.tanh())
}
