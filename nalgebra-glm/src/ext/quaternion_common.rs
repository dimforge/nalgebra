use na::{self, RealField, Unit};

use crate::aliases::Qua;

/// The conjugate of `q`.
pub fn quat_conjugate<T: RealField>(q: &Qua<T>) -> Qua<T> {
    q.conjugate()
}

/// The inverse of `q`.
pub fn quat_inverse<T: RealField>(q: &Qua<T>) -> Qua<T> {
    q.try_inverse().unwrap_or_else(na::zero)
}

//pub fn quat_isinf<T: RealField>(x: &Qua<T>) -> TVec<bool, U4> {
//    x.coords.map(|e| e.is_inf())
//}

//pub fn quat_isnan<T: RealField>(x: &Qua<T>) -> TVec<bool, U4> {
//    x.coords.map(|e| e.is_nan())
//}

/// Interpolate linearly between `x` and `y`.
pub fn quat_lerp<T: RealField>(x: &Qua<T>, y: &Qua<T>, a: T) -> Qua<T> {
    x.lerp(y, a)
}

//pub fn quat_mix<T: RealField>(x: &Qua<T>, y: &Qua<T>, a: T) -> Qua<T> {
//    x * (T::one() - a) + y * a
//}

/// Interpolate spherically between `x` and `y`.
pub fn quat_slerp<T: RealField>(x: &Qua<T>, y: &Qua<T>, a: T) -> Qua<T> {
    Unit::new_normalize(*x)
        .slerp(&Unit::new_normalize(*y), a)
        .into_inner()
}
