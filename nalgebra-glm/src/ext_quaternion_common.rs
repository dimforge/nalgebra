use na::{Real, U4};

use aliases::{Vec, Qua};

pub fn conjugate<N: Real>(q: &Qua<N>) -> Qua<N> {
    unimplemented!()
}

pub fn inverse<N: Real>(q: &Qua<N>) -> Qua<N> {
    unimplemented!()
}

pub fn isinf<N: Real>(x: &Qua<N>) -> Vec<bool, U4> {
    unimplemented!()
}

pub fn isnan<N: Real>(x: &Qua<N>) -> Vec<bool, U4> {
    unimplemented!()
}

pub fn lerp<N: Real>(x: &Qua<N>, y: &Qua<N>,a: N) -> Qua<N> {
    unimplemented!()
}

pub fn mix<N: Real>(x: &Qua<N>, y: &Qua<N>,a: N) -> Qua<N> {
    unimplemented!()
}

pub fn slerp<N: Real>(x: &Qua<N>, y: &Qua<N>,a: N) -> Qua<N> {
    unimplemented!()
}
