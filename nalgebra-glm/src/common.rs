use na::{Scalar, DefaultAllocator};

use aliases::Vec;
use traits::{Dimension, Alloc};


pub fn abs<T>(x: T) -> T {
    unimplemented!()

}

pub fn abs2<N: Scalar, D: Dimension>(x: &Vec<N, D>) -> Vec<N, D>
    where DefaultAllocator: Alloc<N, D> {
        unimplemented!()

}

pub fn ceil<N: Scalar, D: Dimension>(x: &Vec<N, D>) -> Vec<N, D>
    where DefaultAllocator: Alloc<N, D> {
        unimplemented!()

}

pub fn clamp<T>(x: T, minVal: T, maxVal: T) -> T {
    unimplemented!()

}

pub fn clamp2<N: Scalar, D: Dimension>(x: &Vec<N, D>,minVal: N, maxVal: N) -> Vec<N, D>
    where DefaultAllocator: Alloc<N, D> {
        unimplemented!()

}

pub fn clamp3<N: Scalar, D: Dimension>(x: &Vec<N, D>, minVal: &Vec<N, D>, maxVal: &Vec<N, D>) -> Vec<N, D>
    where DefaultAllocator: Alloc<N, D> {
        unimplemented!()

}

pub fn floatBitsToInt(v: f32) -> i32 {
    unimplemented!()

}

pub fn floatBitsToInt2<D: Dimension>(v: Vec<f32, D>) -> Vec<i32, D>
    where DefaultAllocator: Alloc<f32, D> {
    unimplemented!()

}

pub fn floatBitsToUint(v: f32) -> u32 {
        unimplemented!()

}

pub fn floatBitsToUint2<D: Dimension>(v: &Vec<f32, D>) -> Vec<u32, D>
    where DefaultAllocator: Alloc<f32, D> {
        unimplemented!()

}
pub fn floor<N: Scalar, D: Dimension>(x: &Vec<N, D>) -> Vec<N, D>
    where DefaultAllocator: Alloc<N, D> {
        unimplemented!()

}
pub fn fma<T>(a: T, b: T, c: T) -> T {
    unimplemented!()

}

pub fn fract<T>(x: T) -> T {
    unimplemented!()

}

pub fn fract2<N: Scalar, D: Dimension>(x: &Vec<N, D>) -> Vec<N, D>
    where DefaultAllocator: Alloc<N, D> {
        unimplemented!()

}

pub fn frexp<T, I>(x: T, exp: I) -> T {
    unimplemented!()

}

pub fn intBitsToFloat<N: Scalar, D: Dimension>(v: i32) -> f32 {
    unimplemented!()

}

pub fn intBitsToFloat2<D: Dimension>(v: &Vec<u32, D>) -> Vec<f32, D>
    where DefaultAllocator: Alloc<f32, D> {
        unimplemented!()

}

pub fn isinf<N: Scalar, D: Dimension>(x: &Vec<N, D>) -> Vec<bool, D>
    where DefaultAllocator: Alloc<N, D> {
        unimplemented!()

}

pub fn isnan<N: Scalar, D: Dimension>(x: &Vec<N, D>) -> Vec<bool, D>
    where DefaultAllocator: Alloc<N, D> {
        unimplemented!()

}

pub fn ldexp<T, I>(x: T, exp: I) -> T {
    unimplemented!()

}

pub fn max<T>(x: T, y: T) -> T {
    unimplemented!()

}

pub fn max2<N: Scalar, D: Dimension>(x: &Vec<N, D>,y: N) -> Vec<N, D>
    where DefaultAllocator: Alloc<N, D> {
        unimplemented!()

}
pub fn max3<N: Scalar, D: Dimension>(x: &Vec<N, D>, y: &Vec<N, D>) -> Vec<N, D>
    where DefaultAllocator: Alloc<N, D> {
        unimplemented!()

}
pub fn min<T>(x: T, y: T) -> T {
    unimplemented!()

}

pub fn min2<N: Scalar, D: Dimension>(x: &Vec<N, D>,y: N) -> Vec<N, D>
    where DefaultAllocator: Alloc<N, D> {
        unimplemented!()

}
pub fn min3<N: Scalar, D: Dimension>(x: &Vec<N, D>, y: &Vec<N, D>) -> Vec<N, D>
    where DefaultAllocator: Alloc<N, D> {
        unimplemented!()

}
pub fn mix<T>(x: T, y: T, a: T) -> T {
    unimplemented!()

}

pub fn mod_<N: Scalar, D: Dimension>(x: &Vec<N, D>, y: &Vec<N, D>) -> Vec<N, D>
    where DefaultAllocator: Alloc<N, D> {
    unimplemented!()

}
pub fn modf<T>(x: T, i: &T) -> T {
    unimplemented!()

}

pub fn round<N: Scalar, D: Dimension>(x: &Vec<N, D>) -> Vec<N, D>
    where DefaultAllocator: Alloc<N, D> {
        unimplemented!()

}
pub fn roundEven<N: Scalar, D: Dimension>(x: &Vec<N, D>) -> Vec<N, D>
    where DefaultAllocator: Alloc<N, D> {
        unimplemented!()

}
pub fn sign<N: Scalar, D: Dimension>(x: &Vec<N, D>) -> Vec<N, D>
    where DefaultAllocator: Alloc<N, D> {
        unimplemented!()

}

pub fn smoothstep<T>(edge0: T, edge1: T, x: T) -> T {
    unimplemented!()

}

pub fn step<T>(edge: T, x: T) -> T {
    unimplemented!()

}

pub fn step2<N: Scalar, D: Dimension>(edge: N, x: &Vec<N, D>) -> Vec<N, D>
    where DefaultAllocator: Alloc<N, D> {
        unimplemented!()

}
pub fn step3<N: Scalar, D: Dimension>(edge: &Vec<N, D>, x: &Vec<N, D>) -> Vec<N, D>
    where DefaultAllocator: Alloc<N, D> {
        unimplemented!()

}
pub fn trunc<N: Scalar, D: Dimension>(x: &Vec<N, D>) -> Vec<N, D>
    where DefaultAllocator: Alloc<N, D> {
        unimplemented!()

}
pub fn uintBitsToFloat(v: u32) -> f32 {
    unimplemented!()

}
pub fn uintBitsToFloat2<D: Dimension>(v: &Vec<u32, D>) -> Vec<f32, D>
    where DefaultAllocator: Alloc<f32, D> {
    unimplemented!()
}