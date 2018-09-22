use na::Real;

use aliases::Qua;

/// Multiplies two quaternions.
pub fn cross<N: Real>(q1: &Qua<N>, q2: &Qua<N>) -> Qua<N> {
    q1 * q2
}

/// The scalar product of two quaternions.
pub fn dot<N: Real>(x: &Qua<N>, y: &Qua<N>) -> N {
    x.dot(y)
}

/// The magnitude of the quaternion `q`.
pub fn length<N: Real>(q: &Qua<N>) -> N {
    q.norm()
}

/// The magnitude of the quaternion `q`.
pub fn magnitude<N: Real>(q: &Qua<N>) -> N {
    q.norm()
}

/// Normalizes the quaternion `q`.
pub fn normalize<N: Real>(q: &Qua<N>) -> Qua<N> {
    q.normalize()
}