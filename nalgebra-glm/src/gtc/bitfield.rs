use na::{Scalar, DefaultAllocator};

use traits::{Alloc, Dimension};
use aliases::*;

pub fn bitfieldDeinterleave(x: u16) -> U8Vec2 {
    unimplemented!()
}

pub fn bitfieldDeinterleave2(x: u32) -> U16Vec2 {
    unimplemented!()
}

pub fn bitfieldDeinterleave3(x: u64) -> U32Vec2 {
    unimplemented!()
}

pub fn bitfieldFillOne<IU>(Value: IU, FirstBit: i32, BitCount: i32) -> IU {
    unimplemented!()
}

pub fn bitfieldFillOne2<N: Scalar, D: Dimension>(Value: &TVec<N, D>, FirstBit: i32, BitCount: i32) -> TVec<N, D>
    where DefaultAllocator: Alloc<N, D> {
    unimplemented!()
}

pub fn bitfieldFillZero<IU>(Value: IU, FirstBit: i32, BitCount: i32) -> IU {
    unimplemented!()
}

pub fn bitfieldFillZero2<N: Scalar, D: Dimension>(Value: &TVec<N, D>, FirstBit: i32, BitCount: i32) -> TVec<N, D>
    where DefaultAllocator: Alloc<N, D> {
    unimplemented!()
}

pub fn bitfieldInterleave(x: i8, y: i8) -> i16 {
    unimplemented!()
}

pub fn bitfieldInterleave2(x: u8, y: u8) -> u16 {
    unimplemented!()
}

pub fn bitfieldInterleave3(v: &U8Vec2) -> u16 {
    unimplemented!()
}

pub fn bitfieldInterleave4(x: i16, y: i16) -> i32 {
    unimplemented!()
}

pub fn bitfieldInterleave5(x: u16, y: u16) -> u32 {
    unimplemented!()
}

pub fn bitfieldInterleave6(v: &U16Vec2) -> u32 {
    unimplemented!()
}

pub fn bitfieldInterleave7(x: i32, y: i32) -> i64 {
    unimplemented!()
}

pub fn bitfieldInterleave8(x: u32, y: u32) -> u64 {
    unimplemented!()
}

pub fn bitfieldInterleave9(v: &U32Vec2) -> u64 {
    unimplemented!()
}

pub fn bitfieldInterleave10(x: i8, y: i8, z: i8) -> i32 {
    unimplemented!()
}

pub fn bitfieldInterleave11(x: u8, y: u8, z: u8) -> u32 {
    unimplemented!()
}

pub fn bitfieldInterleave12(x: i16, y: i16, z: i16) -> i64 {
    unimplemented!()
}

pub fn bitfieldInterleave13(x: u16, y: u16, z: u16) -> u64 {
    unimplemented!()
}

pub fn bitfieldInterleave14(x: i32, y: i32, z: i32) -> i64 {
    unimplemented!()
}

pub fn bitfieldInterleave15(x: u32, y: u32, z: u32) -> u64 {
    unimplemented!()
}

pub fn bitfieldInterleave16(x: i8, y: i8, z: i8, w: i8) -> i32 {
    unimplemented!()
}

pub fn bitfieldInterleave17(x: u8, y: u8, z: u8, w: u8) -> u32 {
    unimplemented!()
}

pub fn bitfieldInterleave18(x: i16, y: i16, z: i16, w: i16) -> i64 {
    unimplemented!()
}

pub fn bitfieldInterleave19(x: u16, y: u16, z: u16, w: u16) -> u64 {
    unimplemented!()
}

pub fn bitfieldRotateLeft<IU>(In: IU, Shift: i32) -> IU {
    unimplemented!()
}

pub fn bitfieldRotateLeft2<N: Scalar, D: Dimension>(In: &TVec<N, D>, Shift: i32) -> TVec<N, D>
    where DefaultAllocator: Alloc<N, D> {
    unimplemented!()
}

pub fn bitfieldRotateRight<IU>(In: IU, Shift: i32) -> IU {
    unimplemented!()
}

pub fn bitfieldRotateRight2<N: Scalar, D: Dimension>(In: &TVec<N, D>, Shift: i32) -> TVec<N, D>
    where DefaultAllocator: Alloc<N, D> {
    unimplemented!()
}

pub fn mask<IU>(Bits: IU) -> IU {
    unimplemented!()
}

pub fn mask2<N: Scalar, D: Dimension>(v: &TVec<N, D>) -> TVec<N, D>
    where DefaultAllocator: Alloc<N, D> {
    unimplemented!()
}
