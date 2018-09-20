use na::{Scalar, DimName, DefaultAllocator};

use traits::Alloc;
use aliases::Vec;

pub fn fmax<N: Scalar, D: DimName>(a: &Vec<N, D>,b: N) -> Vec<N, D>
    where DefaultAllocator: Alloc<N, D> {
    unimplemented!()
}

pub fn fmax2<N: Scalar, D: DimName>(a: &Vec<N, D>, b: &Vec<N, D>) -> Vec<N, D>
    where DefaultAllocator: Alloc<N, D> {
    unimplemented!()
}

pub fn fmax3<N: Scalar, D: DimName>(a: &Vec<N, D>, b: &Vec<N, D>, c: &Vec<N, D>) -> Vec<N, D>
    where DefaultAllocator: Alloc<N, D> {
    unimplemented!()
}

pub fn fmax4<N: Scalar, D: DimName>(a: &Vec<N, D>, b: &Vec<N, D>, c: &Vec<N, D>, d: &Vec<N, D>) -> Vec<N, D>
    where DefaultAllocator: Alloc<N, D> {
    unimplemented!()
}

pub fn fmin<N: Scalar, D: DimName>(x: &Vec<N, D>,y: N) -> Vec<N, D>
    where DefaultAllocator: Alloc<N, D> {
    unimplemented!()
}

pub fn fmin2<N: Scalar, D: DimName>(x: &Vec<N, D>, y: &Vec<N, D>) -> Vec<N, D>
    where DefaultAllocator: Alloc<N, D> {
    unimplemented!()
}

pub fn fmin3<N: Scalar, D: DimName>(a: &Vec<N, D>, b: &Vec<N, D>, c: &Vec<N, D>) -> Vec<N, D>
    where DefaultAllocator: Alloc<N, D> {
    unimplemented!()
}

pub fn fmin4<N: Scalar, D: DimName>(a: &Vec<N, D>, b: &Vec<N, D>, c: &Vec<N, D>, d: &Vec<N, D>) -> Vec<N, D>
    where DefaultAllocator: Alloc<N, D> {
    unimplemented!()
}

pub fn max3<N: Scalar, D: DimName>(x: &Vec<N, D>, y: &Vec<N, D>, z: &Vec<N, D>) -> Vec<N, D>
    where DefaultAllocator: Alloc<N, D> {
    unimplemented!()
}

pub fn max4<N: Scalar, D: DimName>(x: &Vec<N, D>, y: &Vec<N, D>, z: &Vec<N, D>, w: &Vec<N, D>) -> Vec<N, D>
    where DefaultAllocator: Alloc<N, D> {
    unimplemented!()
}

pub fn min3<N: Scalar, D: DimName>(a: &Vec<N, D>, b: &Vec<N, D>, c: &Vec<N, D>) -> Vec<N, D>
    where DefaultAllocator: Alloc<N, D> {
    unimplemented!()
}

pub fn min4<N: Scalar, D: DimName>(a: &Vec<N, D>, b: &Vec<N, D>, c: &Vec<N, D>, d: &Vec<N, D>) -> Vec<N, D>
    where DefaultAllocator: Alloc<N, D> {
    unimplemented!()
}
