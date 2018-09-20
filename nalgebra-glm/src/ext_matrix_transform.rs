use na::{Scalar, U3, U4};

use aliases::{Mat, Vec};

pub fn identity<T>() -> T {
    unimplemented!()
}

pub fn lookAt<N: Scalar>(eye: &Vec<N, U3>, center: &Vec<N, U3>, up: &Vec<N, U3>) -> Mat<N, U4, U4> {
    unimplemented!()
}

pub fn lookAtLH<N: Scalar>(eye: &Vec<N, U3>, center: &Vec<N, U3>, up: &Vec<N, U3>) -> Mat<N, U4, U4> {
    unimplemented!()
}

pub fn lookAtRH<N: Scalar>(eye: &Vec<N, U3>, center: &Vec<N, U3>, up: &Vec<N, U3>) -> Mat<N, U4, U4> {
    unimplemented!()
}

pub fn rotate<N: Scalar>(m: &Mat<N, U4, U4>,angle: N, axis: &Vec<N, U3>) -> Mat<N, U4, U4> {
    unimplemented!()
}

pub fn scale<N: Scalar>(m: &Mat<N, U4, U4>, v: &Vec<N, U3>) -> Mat<N, U4, U4> {
    unimplemented!()
}

pub fn translate<N: Scalar>(m: &Mat<N, U4, U4>, v: &Vec<N, U3>) -> Mat<N, U4, U4> {
    unimplemented!()
}
