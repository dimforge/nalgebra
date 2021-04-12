use na::RealField;

use crate::aliases::Qua;

/// Multiplies two quaternions.
pub fn quat_cross<T: RealField>(q1: &Qua<T>, q2: &Qua<T>) -> Qua<T> {
    q1 * q2
}

/// The scalar product of two quaternions.
pub fn quat_dot<T: RealField>(x: &Qua<T>, y: &Qua<T>) -> T {
    x.dot(y)
}

/// The magnitude of the quaternion `q`.
pub fn quat_length<T: RealField>(q: &Qua<T>) -> T {
    q.norm()
}

/// The magnitude of the quaternion `q`.
pub fn quat_magnitude<T: RealField>(q: &Qua<T>) -> T {
    q.norm()
}

/// Normalizes the quaternion `q`.
pub fn quat_normalize<T: RealField>(q: &Qua<T>) -> Qua<T> {
    q.normalize()
}
