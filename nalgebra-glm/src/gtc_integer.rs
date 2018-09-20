use na::{DimName, Scalar, DefaultAllocator};

use traits::Alloc;
use aliases::Vec;

pub fn iround<N: Scalar, D: DimName>(x: &Vec<N, D>) -> Vec<i64, D>
    where DefaultAllocator: Alloc<N, D> {
    unimplemented!()
}

pub fn log2<I>(x: I) -> I {
    unimplemented!()
}

pub fn uround<N: Scalar, D: DimName>(x: &Vec<N, D>) -> Vec<u64, D>
    where DefaultAllocator: Alloc<N, D> {
    unimplemented!()
}