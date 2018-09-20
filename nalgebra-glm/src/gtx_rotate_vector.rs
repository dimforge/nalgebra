use na::{Real, U2, U3, U4};

pub fn orientation<N: Real>(Normal: &Vec<N, U3>, Up: &Vec<N, U3>) -> Mat<N, U4, U4> {
    unimplemented!()
}

pub fn rotate2<N: Real>(v: &Vec<N, U2>, angle: N) -> Vec<N, U2> {
    unimplemented!()
}

pub fn rotate<N: Real>(v: &Vec<N, U3>, angle: N, normal: &Vec<N, U3>) -> Vec<N, U3> {
    unimplemented!()
}

pub fn rotate4<N: Real>(v: &Vec<N, U4>, angle: N, normal: &Vec<N, U3>) -> Vec<N, U4> {
    unimplemented!()
}

pub fn rotateX<N: Real>(v: &Vec<N, U3>, angle: N) -> Vec<N, U3> {
    unimplemented!()
}

pub fn rotateX4<N: Real>(v: &Vec<N, U4>, angle: N) -> Vec<N, U4> {
    unimplemented!()
}

pub fn rotateY<N: Real>(v: &Vec<N, U3>, angle: N) -> Vec<N, U3> {
    unimplemented!()
}

pub fn rotateY4<N: Real>(v: &Vec<N, U4>, angle: N) -> Vec<N, U4> {
    unimplemented!()
}

pub fn rotateZ<N: Real>(v: &Vec<N, U3>, angle: N) -> Vec<N, U3> {
    unimplemented!()
}

pub fn rotateZ4<N: Real>(v: &Vec<N, U4>, angle: N) -> Vec<N, U4> {
    unimplemented!()
}

pub fn slerp<N: Real>(x: &Vec<N, U3>, y: &Vec<N, U3>, a: N) -> Vec<N, U3> {
    unimplemented!()
}
