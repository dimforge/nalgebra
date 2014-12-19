//! Quaternion definition.

#![allow(missing_docs)] // we allow missing to avoid having to document the dispatch trait.

use std::mem;
use std::num;
use std::rand::{Rand, Rng};
use std::slice::{Items, MutItems};
use structs::{Vec3, Pnt3, Rot3, Mat3};
use traits::operations::{ApproxEq, Inv, POrd, POrdering, Axpy, ScalarAdd, ScalarSub, ScalarMul,
                         ScalarDiv};
use traits::structure::{Cast, Indexable, Iterable, IterableMut, Dim, Shape, BaseFloat, BaseNum, Zero,
                        One, Bounded};
use traits::geometry::{Norm, Rotation, Rotate, Transform};

/// A quaternion.
#[deriving(Eq, PartialEq, Encodable, Decodable, Clone, Hash, Rand, Show, Copy)]
pub struct Quat<N> {
    /// The scalar component of the quaternion.
    pub w: N,
    /// The first vector component of the quaternion.
    pub i: N,
    /// The second vector component of the quaternion.
    pub j: N,
    /// The third vector component of the quaternion.
    pub k: N
}

impl<N> Quat<N> {
    /// Creates a new quaternion from its components.
    #[inline]
    pub fn new(w: N, i: N, j: N, k: N) -> Quat<N> {
        Quat {
            w: w,
            i: i,
            j: j,
            k: k
        }
    }

    /// The vector part `(i, j, k)` of this quaternion.
    #[inline]
    pub fn vector<'a>(&'a self) -> &'a Vec3<N> {
        // FIXME: do this require a `repr(C)` ?
        unsafe {
            mem::transmute(&self.i)
        }
    }

    /// The scalar part `w` of this quaternion.
    #[inline]
    pub fn scalar<'a>(&'a self) -> &'a N {
        &self.w
    }
}

impl<N: Neg<N>> Quat<N> {
    /// Replaces this quaternion by its conjugate.
    #[inline]
    pub fn conjugate(&mut self) {
        self.i = -self.i;
        self.j = -self.j;
        self.k = -self.k;
    }
}

impl<N: BaseFloat + ApproxEq<N>> Inv for Quat<N> {
    #[inline]
    fn inv_cpy(&self) -> Option<Quat<N>> {
        let mut res = *self;

        if res.inv() {
            Some(res)
        }
        else {
            None
        }
    }

    #[inline]
    fn inv(&mut self) -> bool {
        let sqnorm = Norm::sqnorm(self);

        if ApproxEq::approx_eq(&sqnorm, &::zero()) {
            false
        }
        else {
            self.conjugate();
            self.w = self.w / sqnorm;
            self.i = self.i / sqnorm;
            self.j = self.j / sqnorm;
            self.k = self.k / sqnorm;

            true
        }
    }
}

impl<N: BaseFloat> Norm<N> for Quat<N> {
    #[inline]
    fn sqnorm(&self) -> N {
        self.w * self.w + self.i * self.i + self.j * self.j + self.k * self.k
    }

    #[inline]
    fn normalize_cpy(&self) -> Quat<N> {
        let n = self.norm();
        Quat::new(self.w / n, self.i / n, self.j / n, self.k / n)
    }

    #[inline]
    fn normalize(&mut self) -> N {
        let n = Norm::norm(self);

        self.w = self.w / n;
        self.i = self.i / n;
        self.j = self.j / n;
        self.k = self.k / n;

        n
    }
}

impl<N: Copy + Mul<N, N> + Sub<N, N> + Add<N, N>> Mul<Quat<N>, Quat<N>> for Quat<N> {
    #[inline]
    fn mul(self, right: Quat<N>) -> Quat<N> {
        Quat::new(
            self.w * right.w - self.i * right.i - self.j * right.j - self.k * right.k,
            self.w * right.i + self.i * right.w + self.j * right.k - self.k * right.j,
            self.w * right.j - self.i * right.k + self.j * right.w + self.k * right.i,
            self.w * right.k + self.i * right.j - self.j * right.i + self.k * right.w)
    }
}

impl<N: ApproxEq<N> + BaseFloat> Div<Quat<N>, Quat<N>> for Quat<N> {
    #[inline]
    fn div(self, right: Quat<N>) -> Quat<N> {
        self * right.inv_cpy().expect("Unable to invert the denominator.")
    }
}

/// A unit quaternion that can represent a 3D rotation.
#[deriving(Eq, PartialEq, Encodable, Decodable, Clone, Hash, Show, Copy)]
pub struct UnitQuat<N> {
    q: Quat<N>
}

impl<N: BaseFloat> UnitQuat<N> {
    /// Creates a new unit quaternion from the axis-angle representation of a rotation.
    #[inline]
    pub fn new(axisangle: Vec3<N>) -> UnitQuat<N> {
        let sqang = Norm::sqnorm(&axisangle);

        if ::is_zero(&sqang) {
            ::one()
        }
        else {
            let ang    = sqang.sqrt();
            let (s, c) = (ang / num::cast(2.0f64).unwrap()).sin_cos();

            let s_ang = s / ang;

            unsafe {
                UnitQuat::new_with_unit_quat(
                    Quat::new(
                        c,
                        axisangle.x * s_ang,
                        axisangle.y * s_ang,
                        axisangle.z * s_ang)
                )
            }
        }
    }

    /// Creates a new unit quaternion from a quaternion.
    ///
    /// The input quaternion will be normalized.
    #[inline]
    pub fn new_with_quat(q: Quat<N>) -> UnitQuat<N> {
        let mut q = q;
        let _ = q.normalize();

        UnitQuat {
            q: q
        }
    }

    /// Creates a new unit quaternion from Euler angles.
    ///
    /// The primitive rotations are applied in order: 1 roll − 2 pitch − 3 yaw.
    #[inline]
    pub fn new_with_euler_angles(roll: N, pitch: N, yaw: N) -> UnitQuat<N> {
        let _0_5: N  = num::cast(0.5f64).unwrap();
        let (sr, cr) = (roll * _0_5).sin_cos();
        let (sp, cp) = (pitch * _0_5).sin_cos();
        let (sy, cy) = (yaw * _0_5).sin_cos();

        unsafe {
            UnitQuat::new_with_unit_quat(
                Quat::new(
                    cr * cp * cy + sr * sp * sy,
                    sr * cp * cy - cr * sp * sy,
                    cr * sp * cy + sr * cp * sy,
                    cr * cp * sy - sr * sp * cy)
            )
        }
    }

    /// Builds a rotation matrix from this quaternion.
    pub fn to_rot(&self) -> Rot3<N> {
        let _2: N = num::cast(2.0f64).unwrap();
        let ww = self.q.w * self.q.w;
        let ii = self.q.i * self.q.i;
        let jj = self.q.j * self.q.j;
        let kk = self.q.k * self.q.k;
        let ij = _2 * self.q.i * self.q.j;
        let wk = _2 * self.q.w * self.q.k;
        let wj = _2 * self.q.w * self.q.j;
        let ik = _2 * self.q.i * self.q.k;
        let jk = _2 * self.q.j * self.q.k;
        let wi = _2 * self.q.w * self.q.i;

        unsafe {
            Rot3::new_with_mat(
                Mat3::new(
                    ww + ii - jj - kk, ij - wk,           wj + ik,
                    wk + ij,           ww - ii + jj - kk, jk - wi,
                    ik - wj,           wi + jk,           ww - ii - jj + kk
                )
            )
        }
    }
}

impl<N> UnitQuat<N> {
    /// Creates a new unit quaternion from a quaternion.
    ///
    /// This is unsafe because the input quaternion will not be normalized.
    #[inline]
    pub unsafe fn new_with_unit_quat(q: Quat<N>) -> UnitQuat<N> {
        UnitQuat {
            q: q
        }
    }

    /// The `Quat` representation of this unit quaternion.
    #[inline]
    pub fn quat<'a>(&'a self) -> &'a Quat<N> {
        &self.q
    }
}

impl<N: BaseNum> One for UnitQuat<N> {
    #[inline]
    fn one() -> UnitQuat<N> {
        unsafe {
            UnitQuat::new_with_unit_quat(Quat::new(::one(), ::zero(), ::zero(), ::zero()))
        }
    }
}

impl<N: Copy + Neg<N>> Inv for UnitQuat<N> {
    #[inline]
    fn inv_cpy(&self) -> Option<UnitQuat<N>> {
        let mut cpy = *self;

        cpy.inv();
        Some(cpy)
    }

    #[inline]
    fn inv(&mut self) -> bool {
        self.q.conjugate();

        true
    }
}

impl<N: Rand + BaseFloat> Rand for UnitQuat<N> {
    #[inline]
    fn rand<R: Rng>(rng: &mut R) -> UnitQuat<N> {
        UnitQuat::new(rng.gen())
    }
}

impl<N: ApproxEq<N>> ApproxEq<N> for UnitQuat<N> {
    #[inline]
    fn approx_epsilon(_: Option<UnitQuat<N>>) -> N {
        ApproxEq::approx_epsilon(None::<N>)
    }

    #[inline]
    fn approx_eq_eps(&self, other: &UnitQuat<N>, eps: &N) -> bool {
        ApproxEq::approx_eq_eps(&self.q, &other.q, eps)
    }
}

impl<N: BaseFloat + ApproxEq<N>> Div<UnitQuat<N>, UnitQuat<N>> for UnitQuat<N> {
    #[inline]
    fn div(self, other: UnitQuat<N>) -> UnitQuat<N> {
        UnitQuat { q: self.q / other.q }
    }
}

impl<N: BaseNum> Mul<UnitQuat<N>, UnitQuat<N>> for UnitQuat<N> {
    #[inline]
    fn mul(self, right: UnitQuat<N>) -> UnitQuat<N> {
        UnitQuat { q: self.q * right.q }
    }
}

impl<N: BaseNum> Mul<Vec3<N>, Vec3<N>> for UnitQuat<N> {
    #[inline]
    fn mul(self, right: Vec3<N>) -> Vec3<N> {
        let _2: N = ::one::<N>() + ::one();
        let mut t = ::cross(self.q.vector(), &right);
        t.x = t.x * _2;
        t.y = t.y * _2;
        t.z = t.z * _2;

        Vec3::new(t.x * self.q.w, t.y * self.q.w, t.z * self.q.w) + ::cross(self.q.vector(), &t) + right
    }
}

impl<N: BaseNum> Mul<Pnt3<N>, Pnt3<N>> for UnitQuat<N> {
    #[inline]
    fn mul(self, right: Pnt3<N>) -> Pnt3<N> {
        ::orig::<Pnt3<N>>() + self * *right.as_vec()
    }
}

impl<N: BaseNum> Mul<UnitQuat<N>, Vec3<N>> for Vec3<N> {
    #[inline]
    fn mul(self, right: UnitQuat<N>) -> Vec3<N> {
        let mut inv_quat = right;

        inv_quat.inv();

        inv_quat * self
    }
}

impl<N: BaseNum> Mul<UnitQuat<N>, Pnt3<N>> for Pnt3<N> {
    #[inline]
    fn mul(self, right: UnitQuat<N>) -> Pnt3<N> {
        ::orig::<Pnt3<N>>() + *self.as_vec() * right
    }
}

impl<N: BaseFloat> Rotation<Vec3<N>> for UnitQuat<N> {
    #[inline]
    fn rotation(&self) -> Vec3<N> {
        let _2 = ::one::<N>() + ::one();
        let mut v = *self.q.vector();
        let ang = _2 * v.normalize().atan2(self.q.w);

        if ::is_zero(&ang) {
            ::zero()
        }
        else {
            Vec3::new(v.x * ang, v.y * ang, v.z * ang)
        }
    }

    #[inline]
    fn inv_rotation(&self) -> Vec3<N> {
        -self.rotation()
    }

    #[inline]
    fn append_rotation(&mut self, amount: &Vec3<N>) {
        *self = Rotation::append_rotation_cpy(self, amount)
    }

    #[inline]
    fn append_rotation_cpy(&self, amount: &Vec3<N>) -> UnitQuat<N> {
        *self * UnitQuat::new(*amount)
    }

    #[inline]
    fn prepend_rotation(&mut self, amount: &Vec3<N>) {
        *self = Rotation::prepend_rotation_cpy(self, amount)
    }

    #[inline]
    fn prepend_rotation_cpy(&self, amount: &Vec3<N>) -> UnitQuat<N> {
        UnitQuat::new(*amount) * *self
    }

    #[inline]
    fn set_rotation(&mut self, v: Vec3<N>) {
        *self = UnitQuat::new(v)
    }
}

impl<N: BaseNum> Rotate<Vec3<N>> for UnitQuat<N> {
    #[inline]
    fn rotate(&self, v: &Vec3<N>) -> Vec3<N> {
        *self * *v
    }

    #[inline]
    fn inv_rotate(&self, v: &Vec3<N>) -> Vec3<N> {
        *v * *self
    }
}

impl<N: BaseNum> Rotate<Pnt3<N>> for UnitQuat<N> {
    #[inline]
    fn rotate(&self, p: &Pnt3<N>) -> Pnt3<N> {
        *self * *p
    }

    #[inline]
    fn inv_rotate(&self, p: &Pnt3<N>) -> Pnt3<N> {
        *p * *self
    }
}

impl<N: BaseNum> Transform<Vec3<N>> for UnitQuat<N> {
    #[inline]
    fn transform(&self, v: &Vec3<N>) -> Vec3<N> {
        *self * *v
    }

    #[inline]
    fn inv_transform(&self, v: &Vec3<N>) -> Vec3<N> {
        *v * *self
    }
}

impl<N: BaseNum> Transform<Pnt3<N>> for UnitQuat<N> {
    #[inline]
    fn transform(&self, p: &Pnt3<N>) -> Pnt3<N> {
        *self * *p
    }

    #[inline]
    fn inv_transform(&self, p: &Pnt3<N>) -> Pnt3<N> {
        *p * *self
    }
}

ord_impl!(Quat, w, i, j, k);
vec_axis_impl!(Quat, w, i, j, k);
vec_cast_impl!(Quat, w, i, j, k);
as_array_impl!(Quat, 4);
index_impl!(Quat);
indexable_impl!(Quat, 4);
at_fast_impl!(Quat, 4);
new_repeat_impl!(Quat, val, w, i, j, k);
dim_impl!(Quat, 3);
container_impl!(Quat);
add_impl!(Quat, w, i, j, k);
sub_impl!(Quat, w, i, j, k);
scalar_add_impl!(Quat, w, i, j, k);
scalar_sub_impl!(Quat, w, i, j, k);
scalar_mul_impl!(Quat, w, i, j, k);
scalar_div_impl!(Quat, w, i, j, k);
neg_impl!(Quat, w, i, j, k);
scalar_ops_impl!(Quat, w, i, j, k);
zero_one_impl!(Quat, w, i, j, k);
approx_eq_impl!(Quat, w, i, j, k);
from_iterator_impl!(Quat, iterator, iterator, iterator, iterator);
bounded_impl!(Quat, w, i, j, k);
axpy_impl!(Quat, w, i, j, k);
iterable_impl!(Quat, 4);
iterable_mut_impl!(Quat, 4);

dim_impl!(UnitQuat, 3);
