use na::{Scalar, Real, U3, DefaultAllocator};

use traits::{Number, Alloc, Dimension};
use aliases::TVec;


pub fn ceilMultiple<T>(v: T, Multiple: T) -> T {
    unimplemented!()
}

pub fn ceilMultiple2<N: Scalar, D: Dimension>(v: &TVec<N, D>, Multiple: &TVec<N, D>) -> TVec<N, D>
    where DefaultAllocator: Alloc<N, D> {
    unimplemented!()
}

pub fn ceilPowerOfTwo<IU>(v: IU) -> IU {
    unimplemented!()
}

pub fn ceilPowerOfTwo2<N: Scalar, D: Dimension>(v: &TVec<N, D>) -> TVec<N, D>
    where DefaultAllocator: Alloc<N, D> {
    unimplemented!()
}

pub fn floorMultiple<T>(v: T, Multiple: T) -> T {
    unimplemented!()
}

pub fn floorMultiple2<N: Scalar, D: Dimension>(v: &TVec<N, D>, Multiple: &TVec<N, D>) -> TVec<N, D>
    where DefaultAllocator: Alloc<N, D> {
    unimplemented!()
}

pub fn floorPowerOfTwo<IU>(v: IU) -> IU {
    unimplemented!()
}

pub fn floorPowerOfTwo2<N: Scalar, D: Dimension>(v: &TVec<N, D>) -> TVec<N, D>
    where DefaultAllocator: Alloc<N, D> {
    unimplemented!()
}

pub fn isMultiple<IU>(v: IU, Multiple: IU) -> bool {
    unimplemented!()
}

pub fn isMultiple2<N: Scalar, D: Dimension>(v: &TVec<N, D>,Multiple: N) -> TVec<bool, D>
    where DefaultAllocator: Alloc<N, D> {
    unimplemented!()
}

pub fn isMultiple3<N: Scalar, D: Dimension>(v: &TVec<N, D>, Multiple: &TVec<N, D>) -> TVec<bool, D>
    where DefaultAllocator: Alloc<N, D> {
    unimplemented!()
}

pub fn isPowerOfTwo2<IU>(v: IU) -> bool {
    unimplemented!()
}

pub fn isPowerOfTwo<N: Scalar, D: Dimension>(v: &TVec<N, D>) -> TVec<bool, D>
    where DefaultAllocator: Alloc<N, D> {
    unimplemented!()
}

pub fn roundMultiple<T>(v: T, Multiple: T) -> T {
    unimplemented!()
}

pub fn roundMultiple2<N: Scalar, D: Dimension>(v: &TVec<N, D>, Multiple: &TVec<N, D>) -> TVec<N, D>
    where DefaultAllocator: Alloc<N, D> {
    unimplemented!()
}

pub fn roundPowerOfTwo<IU>(v: IU) -> IU {
    unimplemented!()
}

pub fn roundPowerOfTwo2<N: Scalar, D: Dimension>(v: &TVec<N, D>) -> TVec<N, D>
    where DefaultAllocator: Alloc<N, D> {
    unimplemented!()
}
