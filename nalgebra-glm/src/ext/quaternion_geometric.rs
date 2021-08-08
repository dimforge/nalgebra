use crate::RealNumber;

use crate::aliases::Qua;

/// Multiplies two quaternions.
pub fn quat_cross<T: RealNumber>(q1: &Qua<T>, q2: &Qua<T>) -> Qua<T> {
    q1 * q2
}

/// The scalar product of two quaternions.
pub fn quat_dot<T: RealNumber>(x: &Qua<T>, y: &Qua<T>) -> T {
    x.dot(y)
}

/// The magnitude of the quaternion `q`.
pub fn quat_length<T: RealNumber>(q: &Qua<T>) -> T {
    q.norm()
}

/// The magnitude of the quaternion `q`.
pub fn quat_magnitude<T: RealNumber>(q: &Qua<T>) -> T {
    q.norm()
}

/// Normalizes the quaternion `q`.
pub fn quat_normalize<T: RealNumber>(q: &Qua<T>) -> Qua<T> {
    q.normalize()
}
