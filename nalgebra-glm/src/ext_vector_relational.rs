use na::{Scalar, DimName, DefaultAllocator};

use traits::Alloc;
use aliases::Vec;

pub fn equal<N: Scalar, D: DimName>(x: &Vec<N, D>, y: &Vec<N, D>, epsilon: N) -> Vec<bool, D>
    where DefaultAllocator: Alloc<N, D> {
    unimplemented!()
}

pub fn equal2<N: Scalar, D: DimName>(x: &Vec<N, D>, y: &Vec<N, D>, epsilon: &Vec<N, D>) -> Vec<bool, D>
    where DefaultAllocator: Alloc<N, D> {
    unimplemented!()
}

pub fn notEqual<N: Scalar, D: DimName>(x: &Vec<N, D>, y: &Vec<N, D>, epsilon: N) -> Vec<bool, D>
    where DefaultAllocator: Alloc<N, D> {
    unimplemented!()
}

pub fn notEqual2<N: Scalar, D: DimName>(x: &Vec<N, D>, y: &Vec<N, D>, epsilon: &Vec<N, D>) -> Vec<bool, D>
    where DefaultAllocator: Alloc<N, D> {
    unimplemented!()
}
