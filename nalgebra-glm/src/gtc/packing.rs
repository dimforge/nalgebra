use na::{Scalar, Real, DefaultAllocator, U3, U4};

use traits::{Alloc, Dimension};
use aliases::*;


pub fn packF2x11_1x10(v: &Vec3) -> i32 {
    unimplemented!()
}

pub fn packF3x9_E1x5(v: &Vec3) -> i32 {
    unimplemented!()
}

pub fn packHalf<D: Dimension>(v: &TVec<f32, D>) -> TVec<u16, D>
    where DefaultAllocator: Alloc<u16, D> {
    unimplemented!()
}

pub fn packHalf1x16(v: f32) -> u16 {
    unimplemented!()
}

pub fn packHalf4x16(v: &Vec4) -> u64 {
    unimplemented!()
}

pub fn packI3x10_1x2(v: &IVec4) -> i32 {
    unimplemented!()
}

pub fn packInt2x16(v: &I16Vec2) -> i32 {
    unimplemented!()
}

pub fn packInt2x32(v: &I32Vec2) -> i64 {
    unimplemented!()
}

pub fn packInt2x8(v: &I8Vec2) -> i16 {
    unimplemented!()
}

pub fn packInt4x16(v: &I16Vec4) -> i64 {
    unimplemented!()
}

pub fn packInt4x8(v: &I8Vec4) -> i32 {
    unimplemented!()
}

pub fn packRGBM<N: Scalar>(rgb: &TVec3<N>) -> TVec4<N> {
    unimplemented!()
}

pub fn packSnorm<I: Scalar, N: Real, D: Dimension>(v: TVec<N, D>) -> TVec<I, D>
    where DefaultAllocator: Alloc<N, D> + Alloc<I, D> {
    unimplemented!()
}

pub fn packSnorm1x16(v: f32) -> u16 {
    unimplemented!()
}

pub fn packSnorm1x8(s: f32) -> u8 {
    unimplemented!()
}

pub fn packSnorm2x8(v: &Vec2) -> u16 {
    unimplemented!()
}

pub fn packSnorm3x10_1x2(v: &Vec4) -> i32 {
    unimplemented!()
}

pub fn packSnorm4x16(v: &Vec4) -> u64 {
    unimplemented!()
}

pub fn packU3x10_1x2(v: &UVec4) -> i32 {
    unimplemented!()
}

pub fn packUint2x16(v: &U16Vec2) -> u32 {
    unimplemented!()
}

pub fn packUint2x32(v: &U32Vec2) -> u64 {
    unimplemented!()
}

pub fn packUint2x8(v: &U8Vec2) -> u16 {
    unimplemented!()
}

pub fn packUint4x16(v: &U16Vec4) -> u64 {
    unimplemented!()
}

pub fn packUint4x8(v: &U8Vec4) -> i32 {
    unimplemented!()
}

pub fn packUnorm<UI: Scalar, N: Real, D: Dimension>(v: &TVec<N, D>) -> TVec<UI, D>
    where DefaultAllocator: Alloc<N, D> + Alloc<UI, D> {
    unimplemented!()
}

pub fn packUnorm1x16(v: f32) -> u16 {
    unimplemented!()
}

pub fn packUnorm1x5_1x6_1x5(v: &Vec3) -> u16 {
    unimplemented!()
}

pub fn packUnorm1x8(v: f32) -> u8 {
    unimplemented!()
}

pub fn packUnorm2x3_1x2(v: &Vec3) -> u8 {
    unimplemented!()
}

pub fn packUnorm2x4(v: &Vec2) -> u8 {
    unimplemented!()
}

pub fn packUnorm2x8(v: &Vec2) -> u16 {
    unimplemented!()
}

pub fn packUnorm3x10_1x2(v: &Vec4) -> i32 {
    unimplemented!()
}

pub fn packUnorm3x5_1x1(v: &Vec4) -> u16 {
    unimplemented!()
}

pub fn packUnorm4x16(v: &Vec4) -> u64 {
    unimplemented!()
}

pub fn packUnorm4x4(v: &Vec4) -> u16 {
    unimplemented!()
}

pub fn unpackF2x11_1x10(p: i32) -> Vec3 {
    unimplemented!()
}

pub fn unpackF3x9_E1x5(p: i32) -> Vec3 {
    unimplemented!()
}

pub fn unpackHalf<N: Scalar, D: Dimension>(p: TVec<i16, D>) -> TVec<N, D>
    where DefaultAllocator: Alloc<N, D> {
    unimplemented!()
}

pub fn unpackHalf1x16(v: u16) -> f32 {
    unimplemented!()
}

pub fn unpackHalf4x16(p: u64) -> Vec4 {
    unimplemented!()
}

pub fn unpackI3x10_1x2(p: i32) -> IVec4 {
    unimplemented!()
}

pub fn unpackInt2x16(p: i32) -> I16Vec2 {
    unimplemented!()
}

pub fn unpackInt2x32(p: i64) -> I32Vec2 {
    unimplemented!()
}

pub fn unpackInt2x8(p: i16) -> I8Vec2 {
    unimplemented!()
}

pub fn unpackInt4x16(p: i64) -> I16Vec4 {
    unimplemented!()
}

pub fn unpackInt4x8(p: i32) -> I8Vec4 {
    unimplemented!()
}

pub fn unpackRGBM<N: Scalar>(rgbm: &TVec4<N>) -> TVec3<N> {
    unimplemented!()
}

pub fn unpackSnorm<I: Scalar, N: Real, D: Dimension>(v: &TVec<I, D>) -> TVec<N, D>
    where DefaultAllocator: Alloc<N, D> + Alloc<I, D> {
    unimplemented!()
}

pub fn unpackSnorm1x16(p: u16) -> f32 {
    unimplemented!()
}

pub fn unpackSnorm1x8(p: u8) -> f32 {
    unimplemented!()
}

pub fn unpackSnorm2x8(p: u16) -> Vec2 {
    unimplemented!()
}

pub fn unpackSnorm3x10_1x2(p: i32) -> Vec4 {
    unimplemented!()
}

pub fn unpackSnorm4x16(p: u64) -> Vec4 {
    unimplemented!()
}

pub fn unpackU3x10_1x2(p: i32) -> UVec4 {
    unimplemented!()
}

pub fn unpackUint2x16(p: u32) -> U16Vec2 {
    unimplemented!()
}

pub fn unpackUint2x32(p: u64) -> U32Vec2 {
    unimplemented!()
}

pub fn unpackUint2x8(p: u16) -> U8Vec2 {
    unimplemented!()
}

pub fn unpackUint4x16(p: u64) -> U16Vec4 {
    unimplemented!()
}

pub fn unpackUint4x8(p: i32) -> U8Vec4 {
    unimplemented!()
}

pub fn unpackUnorm<UI: Scalar, N: Real, D: Dimension>(v: &TVec<UI, D>) -> TVec<N, D>
    where DefaultAllocator: Alloc<N, D> + Alloc<UI, D> {
    unimplemented!()
}

pub fn unpackUnorm1x16(p: u16) -> f32 {
    unimplemented!()
}

pub fn unpackUnorm1x5_1x6_1x5(p: u16) -> Vec3 {
    unimplemented!()
}

pub fn unpackUnorm1x8(p: u8) -> f32 {
    unimplemented!()
}

pub fn unpackUnorm2x3_1x2(p: u8) -> Vec3 {
    unimplemented!()
}

pub fn unpackUnorm2x4(p: u8) -> Vec2 {
    unimplemented!()
}

pub fn unpackUnorm2x8(p: u16) -> Vec2 {
    unimplemented!()
}

pub fn unpackUnorm3x10_1x2(p: i32) -> Vec4 {
    unimplemented!()
}

pub fn unpackUnorm3x5_1x1(p: u16) -> Vec4 {
    unimplemented!()
}

pub fn unpackUnorm4x16(p: u64) -> Vec4 {
    unimplemented!()
}

pub fn unpackUnorm4x4(p: u16) -> Vec4 {
    unimplemented!()
}
