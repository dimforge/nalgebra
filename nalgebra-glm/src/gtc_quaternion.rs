use na::{Real, U3, U4};

use aliases::{Qua, Vec, Mat};


pub fn eulerAngles<N: Real>(x: &Qua<N>) -> Vec<N, U3> {
    unimplemented!()
}

pub fn greaterThan<N: Real>(x: &Qua<N>, y: &Qua<N>) -> Vec<bool, U4> {
    unimplemented!()
}

pub fn greaterThanEqual<N: Real>(x: &Qua<N>, y: &Qua<N>) -> Vec<bool, U4> {
    unimplemented!()
}

pub fn lessThan<N: Real>(x: &Qua<N>, y: &Qua<N>) -> Vec<bool, U4> {
    unimplemented!()
}

pub fn lessThanEqual<N: Real>(x: &Qua<N>, y: &Qua<N>) -> Vec<bool, U4> {
    unimplemented!()
}

pub fn mat3_cast<N: Real>(x: &Qua<N>) -> Mat<N, U3, U3> {
    unimplemented!()
}

pub fn mat4_cast<N: Real>(x: &Qua<N>) -> Mat<N, U4, U4> {
    unimplemented!()
}

pub fn pitch<N: Real>(x: &Qua<N>) -> N {
    unimplemented!()
}

pub fn quat_cast<N: Real>(x: &Mat<N, U3, U3>) -> Qua<N> {
    unimplemented!()
}

pub fn quat_cast2<N: Real>(x: &Mat<N, U4, U4>) -> Qua<N> {
    unimplemented!()
}

pub fn quatLookAt<N: Real>(direction: &Vec<N, U3>, up: &Vec<N, U3>) -> Qua<N> {
    unimplemented!()
}

pub fn quatLookAtLH<N: Real>(direction: &Vec<N, U3>, up: &Vec<N, U3>) -> Qua<N> {
    unimplemented!()
}

pub fn quatLookAtRH<N: Real>(direction: &Vec<N, U3>, up: &Vec<N, U3>) -> Qua<N> {
    unimplemented!()
}

pub fn roll<N: Real>(x: &Qua<N>) -> N {
    unimplemented!()
}

pub fn yaw<N: Real>(x: &Qua<N>) -> N {
    unimplemented!()
}
