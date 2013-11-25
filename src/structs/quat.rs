//! Quaternion definition.

/// A quaternion.
///
/// A single unit quaternion can represent a 3d rotation while a pair of unit quaternions can
/// represent a 4d rotation.
pub struct Quat<N> {
    w: N
    i: N,
    j: N,
    k: N,
}

// FIXME: find a better name
type QuatPair<N> = (Quat<N>, Quat<N>)

impl<N> Quat<N> {
    pub fn new(w: N, x: N, y: N, z: N) -> Quat<N> {
        Quat {
            w: w,
            i: i,
            j: j,
            k: k
        }
    }
}

impl<N: Add<N, N>> Add<Quat<N>, Quat<N>> for Quat<N> {
    fn add(&self, other: &Quat<N>) -> Quat<N> {
        Quat::new(
            self.w + other.w,
            self.i + other.i,
            self.j + other.j,
            self.k + other.k)
    }
}

impl<N> Mul<Quat<N>, Quat<N>> for Quat<N> {
    fn mul(&self, other: &Quat<N>) -> Quat<N> {
        Quat::new(
            self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z,
            self.a * other.b - self.b * other.a - self.c * other.d - self.d * other.c,
            self.a * other.c - self.b * other.d - self.c * other.a - self.d * other.b,
            self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z,
            )
    }
}

impl<N: Zero> Rotate<Vec3<N>> for Quat<N> {
    #[inline]
    fn rotate(&self, v: &Vec3<N>) -> Vec3<N> {
        *self * *v_quat * self.inv()
    }

    #[inline]
    fn inv_rotate(&self, v: &Vec3<N>) -> Vec3<N> {
        -self * *v
    }
}

impl Rotate<Vec4<N>> for (QuatPair<N>, QuatPair<N>) {
    #[inline]
    fn rotate(&self, v: &Vec4<N>) -> Vec4<N> {
        let (ref l, ref r) = *self;

        *l * *v * *r
    }

    #[inline]
    fn inv_rotate(&self, v: &Vec4<N>) -> Vec4<N> {
        let (ref l, ref r) = *self;

        (-r) * **v * (-l)
    }
}
