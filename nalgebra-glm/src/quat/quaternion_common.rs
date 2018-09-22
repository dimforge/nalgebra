use na::{self, Real, Unit};

use aliases::Qua;

/// The conjugate of `q`.
pub fn conjugate<N: Real>(q: &Qua<N>) -> Qua<N> {
    q.conjugate()
}

/// The inverse of `q`.
pub fn inverse<N: Real>(q: &Qua<N>) -> Qua<N> {
    q.try_inverse().unwrap_or(na::zero())
}

//pub fn isinf<N: Real>(x: &Qua<N>) -> Vec<bool, U4> {
//    x.coords.map(|e| e.is_inf())
//}

//pub fn isnan<N: Real>(x: &Qua<N>) -> Vec<bool, U4> {
//    x.coords.map(|e| e.is_nan())
//}

/// Interpolate linearly between `x` and `y`.
pub fn lerp<N: Real>(x: &Qua<N>, y: &Qua<N>, a: N) -> Qua<N> {
    x.lerp(y, a)
}

//pub fn mix<N: Real>(x: &Qua<N>, y: &Qua<N>, a: N) -> Qua<N> {
//    x * (N::one() - a) + y * a
//}

/// Interpolate spherically between `x` and `y`.
pub fn slerp<N: Real>(x: &Qua<N>, y: &Qua<N>, a: N) -> Qua<N> {
    Unit::new_normalize(*x).slerp(&Unit::new_normalize(*y), a).unwrap()
}
