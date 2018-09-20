//use na::{Scalar, DefaultAllocator};
//
//use traits::{Alloc, Dimension};
//use aliases::Vec;

//pub fn iround<N: Scalar, D: Dimension>(x: &Vec<N, D>) -> Vec<i32, D>
//    where DefaultAllocator: Alloc<N, D> {
//    x.map(|x| x.round())
//}
//
//pub fn log2<I>(x: I) -> I {
//    unimplemented!()
//}
//
//pub fn uround<N: Scalar, D: Dimension>(x: &Vec<N, D>) -> Vec<u32, D>
//    where DefaultAllocator: Alloc<N, D> {
//    unimplemented!()
//}