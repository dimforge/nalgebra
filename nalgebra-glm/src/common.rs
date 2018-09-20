use std::mem;
use num::FromPrimitive;
use na::{self, Scalar, Real, DefaultAllocator};

use aliases::{Vec, Mat};
use traits::{Number, Dimension, Alloc};


pub fn abs<N: Number, R: Dimension, C: Dimension>(x: &Mat<N, R, C>) -> Mat<N, R, C>
    where DefaultAllocator: Alloc<N, R, C> {
     x.abs()
}

pub fn ceil<N: Real, D: Dimension>(x: &Vec<N, D>) -> Vec<N, D>
    where DefaultAllocator: Alloc<N, D> {
    x.map(|x| x.ceil())
}

pub fn clamp<N: Number>(x: N, minVal: N, maxVal: N) -> N {
    na::clamp(x, minVal, maxVal)
}

pub fn clamp2<N: Number, D: Dimension>(x: &Vec<N, D>,minVal: N, maxVal: N) -> Vec<N, D>
    where DefaultAllocator: Alloc<N, D> {
    x.map(|x| na::clamp(x, minVal, maxVal))
}

pub fn clamp3<N: Number, D: Dimension>(x: &Vec<N, D>, minVal: &Vec<N, D>, maxVal: &Vec<N, D>) -> Vec<N, D>
    where DefaultAllocator: Alloc<N, D> {
    na::clamp(x.clone(), minVal.clone(), maxVal.clone())
}

pub fn float_bits_to_int(v: f32) -> i32 {
    unsafe { mem::transmute(v) }
}

pub fn float_bits_to_int_vec<D: Dimension>(v: Vec<f32, D>) -> Vec<i32, D>
    where DefaultAllocator: Alloc<f32, D> {
    v.map(|v| float_bits_to_int(v))
}

pub fn float_bits_to_uint(v: f32) -> u32 {
    unsafe { mem::transmute(v) }
}

pub fn float_bits_to_uint_vec<D: Dimension>(v: &Vec<f32, D>) -> Vec<u32, D>
    where DefaultAllocator: Alloc<f32, D> {
    v.map(|v| float_bits_to_uint(v))
}

pub fn floor<N: Real, D: Dimension>(x: &Vec<N, D>) -> Vec<N, D>
    where DefaultAllocator: Alloc<N, D> {
    x.map(|x| x.floor())
}

// FIXME: should be implemented for Vec/Mat?
pub fn fma<N: Number>(a: N, b: N, c: N) -> N {
    // FIXME: use an actual FMA
    a * b + c
}

pub fn fract<N: Real>(x: N) -> N {
    x.fract()
}

pub fn fract2<N: Real, D: Dimension>(x: &Vec<N, D>) -> Vec<N, D>
    where DefaultAllocator: Alloc<N, D> {
    x.map(|x| x.fract())
}

//// FIXME: should be implemented for Vec/Mat?
///// Returns the (significant, exponent) of this float number.
//pub fn frexp<N: Real>(x: N, exp: N) -> (N, N) {
//    // FIXME: is there a better approach?
//    let e = x.log2().ceil();
//    (x * (-e).exp2(), e)
//}

pub fn int_bits_to_float(v: i32) -> f32 {
    unsafe { mem::transmute(v) }

}

pub fn int_bits_to_float_vec<D: Dimension>(v: &Vec<i32, D>) -> Vec<f32, D>
    where DefaultAllocator: Alloc<f32, D> {
    v.map(|v| int_bits_to_float(v))
}

//pub fn isinf<N: Scalar, D: Dimension>(x: &Vec<N, D>) -> Vec<bool, D>
//    where DefaultAllocator: Alloc<N, D> {
//        unimplemented!()
//
//}
//
//pub fn isnan<N: Scalar, D: Dimension>(x: &Vec<N, D>) -> Vec<bool, D>
//    where DefaultAllocator: Alloc<N, D> {
//        unimplemented!()
//
//}

///// Returns the (significant, exponent) of this float number.
//pub fn ldexp<N: Real>(x: N, exp: N) -> N {
//    // FIXME: is there a better approach?
//    x * (exp).exp2()
//}

pub fn max<N: Number>(x: N, y: N) -> N {
    na::sup(&x, &y)
}

pub fn max2<N: Number, D: Dimension>(x: &Vec<N, D>, y: N) -> Vec<N, D>
    where DefaultAllocator: Alloc<N, D> {
    x.map(|x| na::sup(&x, &y))
}

pub fn max3<N: Number, D: Dimension>(x: &Vec<N, D>, y: &Vec<N, D>) -> Vec<N, D>
    where DefaultAllocator: Alloc<N, D> {
    na::sup(x, y)
}

pub fn min<N: Number>(x: N, y: N) -> N {
    na::inf(&x, &y)
}

pub fn min2<N: Number, D: Dimension>(x: &Vec<N, D>,y: N) -> Vec<N, D>
    where DefaultAllocator: Alloc<N, D> {
    x.map(|x| na::inf(&x, &y))
}

pub fn min3<N: Number, D: Dimension>(x: &Vec<N, D>, y: &Vec<N, D>) -> Vec<N, D>
    where DefaultAllocator: Alloc<N, D> {
    na::inf(x, y)
}

pub fn mix<N: Number>(x: N, y: N, a: N) -> N {
    x * (N::one() - a) + y * a
}

pub fn mod_<N: Number, D: Dimension>(x: &Vec<N, D>, y: &Vec<N, D>) -> Vec<N, D>
    where DefaultAllocator: Alloc<N, D> {
    x.zip_map(y, |x, y| x % y)
}

pub fn modf<N: Number>(x: N, i: N) -> N {
    x % i
}

pub fn round<N: Real, D: Dimension>(x: &Vec<N, D>) -> Vec<N, D>
    where DefaultAllocator: Alloc<N, D> {
    x.map(|x| x.round())

}

//pub fn roundEven<N: Scalar, D: Dimension>(x: &Vec<N, D>) -> Vec<N, D>
//    where DefaultAllocator: Alloc<N, D> {
//        unimplemented!()
//}

pub fn sign<N: Number, D: Dimension>(x: &Vec<N, D>) -> Vec<N, D>
    where DefaultAllocator: Alloc<N, D> {
    x.map(|x| x.signum())
}

pub fn smoothstep<N: Number>(edge0: N, edge1: N, x: N) -> N {
    let _3: N = FromPrimitive::from_f64(3.0).unwrap();
    let _2: N = FromPrimitive::from_f64(2.0).unwrap();
    let t = na::clamp((x - edge0) / (edge1 - edge0), N::zero(), N::one());
    t * t * (_3 - t * _2)
}

pub fn step<N: Number>(edge: N, x: N) -> N {
    if edge > x {
        N::zero()
    } else {
        N::one()
    }
}

pub fn step2<N: Number, D: Dimension>(edge: N, x: &Vec<N, D>) -> Vec<N, D>
    where DefaultAllocator: Alloc<N, D> {
    x.map(|x| step(edge, x))
}

pub fn step3<N: Number, D: Dimension>(edge: &Vec<N, D>, x: &Vec<N, D>) -> Vec<N, D>
    where DefaultAllocator: Alloc<N, D> {
    edge.zip_map(x, |edge, x| step(edge, x))
}

pub fn trunc<N: Real, D: Dimension>(x: &Vec<N, D>) -> Vec<N, D>
    where DefaultAllocator: Alloc<N, D> {
    x.map(|x| x.trunc())
}

pub fn uint_bits_to_float(v: u32) -> f32 {
    unsafe { mem::transmute(v) }

}

pub fn uint_bits_to_float_vec<D: Dimension>(v: &Vec<u32, D>) -> Vec<f32, D>
    where DefaultAllocator: Alloc<f32, D> {
    v.map(|v| uint_bits_to_float(v))
}