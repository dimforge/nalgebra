use na::{DefaultAllocator, RealNumber, Scalar, U3};

use crate::aliases::TVec;
use crate::traits::{Alloc, Dimension, Number};

pub fn bitCount<T>(v: T) -> i32 {
    unimplemented!()
}

pub fn bitCount2<T: Scalar, const D: usize>(v: &TVec<T, D>) -> TVec<i32, D>
where
    DefaultAllocator: Alloc<T, D>,
{
    unimplemented!()
}

pub fn bitfieldExtract<T: Scalar, const D: usize>(
    Value: &TVec<T, D>,
    Offset: i32,
    Bits: i32,
) -> TVec<T, D>
where
    DefaultAllocator: Alloc<T, D>,
{
    unimplemented!()
}

pub fn bitfieldInsert<T: Scalar, const D: usize>(
    Base: &TVec<T, D>,
    Insert: &TVec<T, D>,
    Offset: i32,
    Bits: i32,
) -> TVec<T, D>
where
    DefaultAllocator: Alloc<T, D>,
{
    unimplemented!()
}

pub fn bitfieldReverse<T: Scalar, const D: usize>(v: &TVec<T, D>) -> TVec<T, D>
where
    DefaultAllocator: Alloc<T, D>,
{
    unimplemented!()
}

pub fn findLSB<IU>(x: IU) -> u32 {
    unimplemented!()
}

pub fn findLSB2<T: Scalar, const D: usize>(v: &TVec<T, D>) -> TVec<i32, D>
where
    DefaultAllocator: Alloc<T, D>,
{
    unimplemented!()
}

pub fn findMSB<IU>(x: IU) -> i32 {
    unimplemented!()
}

pub fn findMSB2<T: Scalar, const D: usize>(v: &TVec<T, D>) -> TVec<i32, D>
where
    DefaultAllocator: Alloc<T, D>,
{
    unimplemented!()
}

pub fn imulExtended<T: Scalar, const D: usize>(
    x: &TVec<i32, D>,
    y: &TVec<i32, D>,
    msb: &TVec<i32, D>,
    lsb: &TVec<i32, D>,
) where
    DefaultAllocator: Alloc<T, D>,
{
    unimplemented!()
}

pub fn uaddCarry<T: Scalar, const D: usize>(
    x: &TVec<u32, D>,
    y: &TVec<u32, D>,
    carry: &TVec<u32, D>,
) -> TVec<u32, D>
where
    DefaultAllocator: Alloc<T, D>,
{
    unimplemented!()
}

pub fn umulExtended<T: Scalar, const D: usize>(
    x: &TVec<u32, D>,
    y: &TVec<u32, D>,
    msb: &TVec<u32, D>,
    lsb: &TVec<u32, D>,
) where
    DefaultAllocator: Alloc<T, D>,
{
    unimplemented!()
}

pub fn usubBorrow<T: Scalar, const D: usize>(
    x: &TVec<u32, D>,
    y: &TVec<u32, D>,
    borrow: &TVec<u32, D>,
) -> TVec<u32, D>
where
    DefaultAllocator: Alloc<T, D>,
{
    unimplemented!()
}
