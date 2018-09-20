use na::Real;

use aliases::Qua;

pub fn cross<N: Real>(q1: &Qua<N>, q2: &Qua<N>) -> Qua<N> {
    q1 * q2
}
pub fn dot<N: Real>(x: &Qua<N>, y: &Qua<N>) -> N {
    x.dot(y)
}
pub fn length<N: Real>(q: &Qua<N>) -> N {
    q.norm()
}
pub fn normalize<N: Real>(q: &Qua<N>) -> Qua<N> {
    q.normalize()
}