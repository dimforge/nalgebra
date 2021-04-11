use na::{DefaultAllocator, RealField, Scalar, U3};

use crate::aliases::TVec;
use crate::traits::{Alloc, Dimension, Number};

pub fn ceilMultiple<T>(v: T, Multiple: T) -> T {
    unimplemented!()
}

pub fn ceilMultiple2<T: Scalar, const D: usize>(v: &TVec<T, D>, Multiple: &TVec<T, D>) -> TVec<T, D>
where
    DefaultAllocator: Alloc<T, D>,
{
    unimplemented!()
}

pub fn ceilPowerOfTwo<IU>(v: IU) -> IU {
    unimplemented!()
}

pub fn ceilPowerOfTwo2<T: Scalar, const D: usize>(v: &TVec<T, D>) -> TVec<T, D>
where
    DefaultAllocator: Alloc<T, D>,
{
    unimplemented!()
}

pub fn floorMultiple<T>(v: T, Multiple: T) -> T {
    unimplemented!()
}

pub fn floorMultiple2<T: Scalar, const D: usize>(
    v: &TVec<T, D>,
    Multiple: &TVec<T, D>,
) -> TVec<T, D>
where
    DefaultAllocator: Alloc<T, D>,
{
    unimplemented!()
}

pub fn floorPowerOfTwo<IU>(v: IU) -> IU {
    unimplemented!()
}

pub fn floorPowerOfTwo2<T: Scalar, const D: usize>(v: &TVec<T, D>) -> TVec<T, D>
where
    DefaultAllocator: Alloc<T, D>,
{
    unimplemented!()
}

pub fn isMultiple<IU>(v: IU, Multiple: IU) -> bool {
    unimplemented!()
}

pub fn isMultiple2<T: Scalar, const D: usize>(v: &TVec<T, D>, Multiple: T) -> TVec<bool, D>
where
    DefaultAllocator: Alloc<T, D>,
{
    unimplemented!()
}

pub fn isMultiple3<T: Scalar, const D: usize>(
    v: &TVec<T, D>,
    Multiple: &TVec<T, D>,
) -> TVec<bool, D>
where
    DefaultAllocator: Alloc<T, D>,
{
    unimplemented!()
}

pub fn isPowerOfTwo2<IU>(v: IU) -> bool {
    unimplemented!()
}

pub fn isPowerOfTwo<T: Scalar, const D: usize>(v: &TVec<T, D>) -> TVec<bool, D>
where
    DefaultAllocator: Alloc<T, D>,
{
    unimplemented!()
}

pub fn roundMultiple<T>(v: T, Multiple: T) -> T {
    unimplemented!()
}

pub fn roundMultiple2<T: Scalar, const D: usize>(
    v: &TVec<T, D>,
    Multiple: &TVec<T, D>,
) -> TVec<T, D>
where
    DefaultAllocator: Alloc<T, D>,
{
    unimplemented!()
}

pub fn roundPowerOfTwo<IU>(v: IU) -> IU {
    unimplemented!()
}

pub fn roundPowerOfTwo2<T: Scalar, const D: usize>(v: &TVec<T, D>) -> TVec<T, D>
where
    DefaultAllocator: Alloc<T, D>,
{
    unimplemented!()
}
