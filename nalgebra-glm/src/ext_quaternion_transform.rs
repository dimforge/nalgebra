use na::{Real, U3, UnitQuaternion, Unit};

use aliases::{Vec, Qua};

pub fn exp<N: Real>(q: &Qua<N>) -> Qua<N> {
    q.exp()
}

pub fn log<N: Real>(q: &Qua<N>) -> Qua<N> {
    q.ln()
}

pub fn pow<N: Real>(q: &Qua<N>, y: N) -> Qua<N> {
    q.powf(y)
}

pub fn rotate<N: Real>(q: &Qua<N>, angle: N, axis: &Vec<N, U3>) -> Qua<N> {
    q * UnitQuaternion::from_axis_angle(&Unit::new_normalize(*axis), angle).unwrap()
}

pub fn sqrt<N: Real>(q: &Qua<N>) -> Qua<N> {
    unimplemented!()
}