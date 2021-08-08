use na::{self, Unit};

use crate::aliases::Qua;
use crate::RealNumber;

/// The conjugate of `q`.
pub fn quat_conjugate<T: RealNumber>(q: &Qua<T>) -> Qua<T> {
    q.conjugate()
}

/// The inverse of `q`.
pub fn quat_inverse<T: RealNumber>(q: &Qua<T>) -> Qua<T> {
    q.try_inverse().unwrap_or_else(na::zero)
}

//pub fn quat_isinf<T: RealNumber>(x: &Qua<T>) -> TVec<bool, U4> {
//    x.coords.map(|e| e.is_inf())
//}

//pub fn quat_isnan<T: RealNumber>(x: &Qua<T>) -> TVec<bool, U4> {
//    x.coords.map(|e| e.is_nan())
//}

/// Interpolate linearly between `x` and `y`.
pub fn quat_lerp<T: RealNumber>(x: &Qua<T>, y: &Qua<T>, a: T) -> Qua<T> {
    x.lerp(y, a)
}

//pub fn quat_mix<T: RealNumber>(x: &Qua<T>, y: &Qua<T>, a: T) -> Qua<T> {
//    x * (T::one() - a) + y * a
//}

/// Interpolate spherically between `x` and `y`.
pub fn quat_slerp<T: RealNumber>(x: &Qua<T>, y: &Qua<T>, a: T) -> Qua<T> {
    Unit::new_normalize(*x)
        .slerp(&Unit::new_normalize(*y), a)
        .into_inner()
}
