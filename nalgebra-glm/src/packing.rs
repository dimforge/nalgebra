use na::Scalar;

use crate::aliases::{Vec2, Vec4, UVec2};


pub fn packDouble2x32<N: Scalar + Copy>(v: &UVec2) -> f64 {
    unimplemented!()
}

pub fn packHalf2x16<N: Scalar + Copy>(v: &Vec2) -> u32 {
    unimplemented!()
}

pub fn packSnorm2x16<N: Scalar + Copy>(v: &Vec2) -> u32 {
    unimplemented!()
}

pub fn packSnorm4x8<N: Scalar + Copy>(v: &Vec4) -> u32 {
    unimplemented!()
}

pub fn packUnorm2x16<N: Scalar + Copy>(v: &Vec2) -> u32 {
    unimplemented!()
}

pub fn packUnorm4x8<N: Scalar + Copy>(v: &Vec4) -> u32 {
    unimplemented!()
}

pub fn unpackDouble2x32<N: Scalar + Copy>(v: f64) -> UVec2 {
    unimplemented!()
}

pub fn unpackHalf2x16<N: Scalar + Copy>(v: u32) -> Vec2 {
    unimplemented!()
}

pub fn unpackSnorm2x16<N: Scalar + Copy>(p: u32) -> Vec2 {
    unimplemented!()
}

pub fn unpackSnorm4x8<N: Scalar + Copy>(p: u32) -> Vec4 {
    unimplemented!()
}

pub fn unpackUnorm2x16<N: Scalar + Copy>(p: u32) -> Vec2 {
    unimplemented!()
}

pub fn unpackUnorm4x8<N: Scalar + Copy>(p: u32) -> Vec4 {
    unimplemented!()
}
