use na::{DefaultAllocator, Real};

use aliases::TVec;
use traits::{Alloc, Dimension};

/// The angle between two vectors.
pub fn angle<N: Real, D: Dimension>(x: &TVec<N, D>, y: &TVec<N, D>) -> N
where DefaultAllocator: Alloc<N, D> {
    x.angle(y)
}

//pub fn oriented_angle<N: Real>(x: &TVec2<N>, y: &TVec2<N>) -> N {
//    unimplemented!()
//}
//
//pub fn oriented_angle_ref<N: Real>(x: &TVec3<N>, y: &TVec3<N>, refv: &TVec3<N>) -> N {
//    unimplemented!()
//}
