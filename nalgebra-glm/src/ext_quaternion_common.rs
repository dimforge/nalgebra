use na::{self, Real, Unit, U4};

use aliases::{Vec, Qua};

pub fn conjugate<N: Real>(q: &Qua<N>) -> Qua<N> {
    q.conjugate()
}

pub fn inverse<N: Real>(q: &Qua<N>) -> Qua<N> {
    q.try_inverse().unwrap_or(na::zero())
}

//pub fn isinf<N: Real>(x: &Qua<N>) -> Vec<bool, U4> {
//    x.coords.map(|e| e.is_inf())
//}

//pub fn isnan<N: Real>(x: &Qua<N>) -> Vec<bool, U4> {
//    x.coords.map(|e| e.is_nan())
//}

pub fn lerp<N: Real>(x: &Qua<N>, y: &Qua<N>, a: N) -> Qua<N> {
    x.lerp(y, a)
}

pub fn mix<N: Real>(x: &Qua<N>, y: &Qua<N>, a: N) -> Qua<N> {
    x * (N::one() - a) + y * a
}

pub fn slerp<N: Real>(x: &Qua<N>, y: &Qua<N>, a: N) -> Qua<N> {
    Unit::new_normalize(*x).slerp(&Unit::new_normalize(*y), a).unwrap()
}
