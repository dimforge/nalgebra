//use na::{Scalar, DefaultAllocator};
//
//use crate::traits::{Alloc, Dimension};
//use crate::aliases::TVec;

//pub fn iround<N: Scalar + Copy, D: Dimension>(x: &TVec<N, D>) -> TVec<i32, D>
//    where DefaultAllocator: Alloc<N, D> {
//    x.map(|x| x.round())
//}
//
//pub fn log2<I>(x: I) -> I {
//    unimplemented!()
//}
//
//pub fn uround<N: Scalar + Copy, D: Dimension>(x: &TVec<N, D>) -> TVec<u32, D>
//    where DefaultAllocator: Alloc<N, D> {
//    unimplemented!()
//}
