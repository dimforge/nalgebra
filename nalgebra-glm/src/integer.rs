use na::{Scalar, Real, U3, DefaultAllocator};

use traits::{Number, Alloc, Dimension};
use aliases::Vec;

pub fn bitCount<T>(v: T) -> i32 {
    unimplemented!()
}

pub fn bitCount2<N: Scalar, D: Dimension>(v: &Vec<N, D>) -> Vec<i32, D>
    where DefaultAllocator: Alloc<N, D> {
    unimplemented!()
}

pub fn bitfieldExtract<N: Scalar, D: Dimension>(Value: &Vec<N, D>, Offset: i32, Bits: i32) -> Vec<N, D>
    where DefaultAllocator: Alloc<N, D> {
    unimplemented!()
}

pub fn bitfieldInsert<N: Scalar, D: Dimension>(Base: &Vec<N, D>, Insert: &Vec<N, D>, Offset: i32, Bits: i32) -> Vec<N, D>
    where DefaultAllocator: Alloc<N, D> {
    unimplemented!()
}

pub fn bitfieldReverse<N: Scalar, D: Dimension>(v: &Vec<N, D>) -> Vec<N, D>
    where DefaultAllocator: Alloc<N, D> {
    unimplemented!()
}

pub fn findLSB<IU>(x: IU) -> u32 {
    unimplemented!()
}

pub fn findLSB2<N: Scalar, D: Dimension>(v: &Vec<N, D>) -> Vec<i32, D>
    where DefaultAllocator: Alloc<N, D> {
    unimplemented!()
}

pub fn findMSB<IU>(x: IU) -> i32 {
    unimplemented!()
}

pub fn findMSB2<N: Scalar, D: Dimension>(v: &Vec<N, D>) -> Vec<i32, D>
    where DefaultAllocator: Alloc<N, D> {
    unimplemented!()
}

pub fn imulExtended<N: Scalar, D: Dimension>(x: &Vec<i32, D>, y: &Vec<i32, D>, msb: &Vec<i32, D>, lsb: &Vec<i32, D>)
    where DefaultAllocator: Alloc<N, D> {
    unimplemented!()
}

pub fn uaddCarry<N: Scalar, D: Dimension>(x: &Vec<u32, D>, y: &Vec<u32, D>, carry: &Vec<u32, D>) -> Vec<u32, D>
    where DefaultAllocator: Alloc<N, D> {
    unimplemented!()
}

pub fn umulExtended<N: Scalar, D: Dimension>(x: &Vec<u32, D>, y: &Vec<u32, D>, msb: &Vec<u32, D>, lsb: &Vec<u32, D>)
    where DefaultAllocator: Alloc<N, D> {
unimplemented!()
}

pub fn usubBorrow<N: Scalar, D: Dimension>(x: &Vec<u32, D>, y: &Vec<u32, D>, borrow: &Vec<u32, D>) -> Vec<u32, D>
    where DefaultAllocator: Alloc<N, D> {
    unimplemented!()
}
