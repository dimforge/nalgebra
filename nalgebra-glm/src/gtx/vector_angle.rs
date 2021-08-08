use crate::RealNumber;

use crate::aliases::TVec;

/// The angle between two vectors.
pub fn angle<T: RealNumber, const D: usize>(x: &TVec<T, D>, y: &TVec<T, D>) -> T {
    x.angle(y)
}

//pub fn oriented_angle<T: RealNumber>(x: &TVec2<T>, y: &TVec2<T>) -> T {
//    unimplemented!()
//}
//
//pub fn oriented_angle_ref<T: RealNumber>(x: &TVec3<T>, y: &TVec3<T>, refv: &TVec3<T>) -> T {
//    unimplemented!()
//}
