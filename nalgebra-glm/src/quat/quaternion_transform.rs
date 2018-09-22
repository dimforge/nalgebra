use na::{Real, U3, UnitQuaternion, Unit};

use aliases::{Vec, Qua};

/// Computes the quaternion exponential.
pub fn exp<N: Real>(q: &Qua<N>) -> Qua<N> {
    q.exp()
}

/// Computes the quaternion logarithm.
pub fn log<N: Real>(q: &Qua<N>) -> Qua<N> {
    q.ln()
}

/// Raises the quaternion `q` to the power `y`.
pub fn pow<N: Real>(q: &Qua<N>, y: N) -> Qua<N> {
    q.powf(y)
}

/// Builds a quaternion from an axis and an angle, and right-multiply it to the quaternion `q`.
pub fn rotate<N: Real>(q: &Qua<N>, angle: N, axis: &Vec<N, U3>) -> Qua<N> {
    q * UnitQuaternion::from_axis_angle(&Unit::new_normalize(*axis), angle).unwrap()
}

//pub fn sqrt<N: Real>(q: &Qua<N>) -> Qua<N> {
//    unimplemented!()
//}