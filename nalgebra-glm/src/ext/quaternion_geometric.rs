use na::RealField;

use crate::aliases::Qua;

/// Multiplies two quaternions.
pub fn quat_cross<N: RealField>(q1: &Qua<N>, q2: &Qua<N>) -> Qua<N> {
    q1 * q2
}

/// The scalar product of two quaternions.
pub fn quat_dot<N: RealField>(x: &Qua<N>, y: &Qua<N>) -> N {
    x.dot(y)
}

/// The magnitude of the quaternion `q`.
pub fn quat_length<N: RealField>(q: &Qua<N>) -> N {
    q.norm()
}

/// The magnitude of the quaternion `q`.
pub fn quat_magnitude<N: RealField>(q: &Qua<N>) -> N {
    q.norm()
}

/// Normalizes the quaternion `q`.
pub fn quat_normalize<N: RealField>(q: &Qua<N>) -> Qua<N> {
    q.normalize()
}
