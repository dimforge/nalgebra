use na::{self, Real, Unit};

use aliases::Qua;

/// The conjugate of `q`.
pub fn quat_conjugate<N: Real>(q: &Qua<N>) -> Qua<N> {
    q.conjugate()
}

/// The inverse of `q`.
pub fn quat_inverse<N: Real>(q: &Qua<N>) -> Qua<N> {
    q.try_inverse().unwrap_or_else(na::zero)
}

//pub fn quat_isinf<N: Real>(x: &Qua<N>) -> TVec<bool, U4> {
//    x.coords.map(|e| e.is_inf())
//}

//pub fn quat_isnan<N: Real>(x: &Qua<N>) -> TVec<bool, U4> {
//    x.coords.map(|e| e.is_nan())
//}

/// Interpolate linearly between `x` and `y`.
pub fn quat_lerp<N: Real>(x: &Qua<N>, y: &Qua<N>, a: N) -> Qua<N> {
    x.lerp(y, a)
}

//pub fn quat_mix<N: Real>(x: &Qua<N>, y: &Qua<N>, a: N) -> Qua<N> {
//    x * (N::one() - a) + y * a
//}

/// Interpolate spherically between `x` and `y`.
pub fn quat_slerp<N: Real>(x: &Qua<N>, y: &Qua<N>, a: N) -> Qua<N> {
    Unit::new_normalize(*x)
        .slerp(&Unit::new_normalize(*y), a)
        .into_inner()
}
