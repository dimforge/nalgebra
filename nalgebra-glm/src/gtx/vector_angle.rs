use na::{DefaultAllocator, RealField};

use crate::aliases::TVec;
use crate::traits::{Alloc, Dimension};

/// The angle between two vectors.
pub fn angle<N: RealField, D: Dimension>(x: &TVec<N, D>, y: &TVec<N, D>) -> N
where
    DefaultAllocator: Alloc<N, D>,
{
    x.angle(y)
}

//pub fn oriented_angle<N: RealField>(x: &TVec2<N>, y: &TVec2<N>) -> N {
//    unimplemented!()
//}
//
//pub fn oriented_angle_ref<N: RealField>(x: &TVec3<N>, y: &TVec3<N>, refv: &TVec3<N>) -> N {
//    unimplemented!()
//}
