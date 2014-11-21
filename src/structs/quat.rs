//! Quaternion definition.

#![allow(missing_docs)] // we allow missing to avoid having to document the dispatch trait.

use std::mem;
use std::num;
use std::rand::{Rand, Rng};
use std::slice::{Items, MutItems};
use structs::{Vec3, Pnt3, Rot3, Mat3, Vec3MulRhs, Pnt3MulRhs};
use traits::operations::{ApproxEq, Inv, POrd, POrdering, Axpy, ScalarAdd, ScalarSub, ScalarMul,
                         ScalarDiv};
use traits::structure::{Cast, Indexable, Iterable, IterableMut, Dim, Shape, BaseFloat, BaseNum, Zero,
                        One, Bounded};
use traits::geometry::{Norm, Cross, Rotation, Rotate, Transform};

/// A quaternion.
#[deriving(Eq, PartialEq, Encodable, Decodable, Clone, Hash, Rand, Show)]
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

impl<N: BaseFloat + ApproxEq<N> + Clone> Inv for Quat<N> {
    #[inline]
    fn inv_cpy(m: &Quat<N>) -> Option<Quat<N>> {
        let mut res = m.clone();

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
    fn sqnorm(q: &Quat<N>) -> N {
        q.w * q.w + q.i * q.i + q.j * q.j + q.k * q.k
    }

    #[inline]
    fn normalize_cpy(v: &Quat<N>) -> Quat<N> {
        let n = Norm::norm(v);
        Quat::new(v.w / n, v.i / n, v.j / n, v.k / n)
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

impl<N: Mul<N, N> + Sub<N, N> + Add<N, N>> QuatMulRhs<N, Quat<N>> for Quat<N> {
    #[inline]
    fn binop(left: &Quat<N>, right: &Quat<N>) -> Quat<N> {
        Quat::new(
            left.w * right.w - left.i * right.i - left.j * right.j - left.k * right.k,
            left.w * right.i + left.i * right.w + left.j * right.k - left.k * right.j,
            left.w * right.j - left.i * right.k + left.j * right.w + left.k * right.i,
            left.w * right.k + left.i * right.j - left.j * right.i + left.k * right.w)
    }
}

impl<N: ApproxEq<N> + BaseFloat + Clone> QuatDivRhs<N, Quat<N>> for Quat<N> {
    #[inline]
    fn binop(left: &Quat<N>, right: &Quat<N>) -> Quat<N> {
        *left * Inv::inv_cpy(right).expect("Unable to invert the denominator.")
    }
}

/// A unit quaternion that can represent a 3D rotation.
#[deriving(Eq, PartialEq, Encodable, Decodable, Clone, Hash, Show)]
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

impl<N: BaseNum + Clone> One for UnitQuat<N> {
    #[inline]
    fn one() -> UnitQuat<N> {
        unsafe {
            UnitQuat::new_with_unit_quat(Quat::new(::one(), ::zero(), ::zero(), ::zero()))
        }
    }
}

impl<N: Clone + Neg<N>> Inv for UnitQuat<N> {
    #[inline]
    fn inv_cpy(m: &UnitQuat<N>) -> Option<UnitQuat<N>> {
        let mut cpy = m.clone();

        cpy.inv();
        Some(cpy)
    }

    #[inline]
    fn inv(&mut self) -> bool {
        self.q.conjugate();

        true
    }
}

impl<N: Clone + Rand + BaseFloat> Rand for UnitQuat<N> {
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
    fn approx_eq(a: &UnitQuat<N>, b: &UnitQuat<N>) -> bool {
        ApproxEq::approx_eq(&a.q, &b.q)
    }

    #[inline]
    fn approx_eq_eps(a: &UnitQuat<N>, b: &UnitQuat<N>, eps: &N) -> bool {
        ApproxEq::approx_eq_eps(&a.q, &b.q, eps)
    }
}

impl<N: BaseFloat + ApproxEq<N> + Clone> Div<UnitQuat<N>, UnitQuat<N>> for UnitQuat<N> {
    #[inline]
    fn div(&self, other: &UnitQuat<N>) -> UnitQuat<N> {
        UnitQuat { q: self.q / other.q }
    }
}

impl<N: BaseNum + Clone> UnitQuatMulRhs<N, UnitQuat<N>> for UnitQuat<N> {
    #[inline]
    fn binop(left: &UnitQuat<N>, right: &UnitQuat<N>) -> UnitQuat<N> {
        UnitQuat { q: left.q * right.q }
    }
}

impl<N: BaseNum + Clone> UnitQuatMulRhs<N, Vec3<N>> for Vec3<N> {
    #[inline]
    fn binop(left: &UnitQuat<N>, right: &Vec3<N>) -> Vec3<N> {
        let _2: N = ::one::<N>() + ::one();
        let mut t = Cross::cross(left.q.vector(), right);
        t.x = t.x * _2;
        t.y = t.y * _2;
        t.z = t.z * _2;

        Vec3::new(t.x * left.q.w, t.y * left.q.w, t.z * left.q.w) +
        Cross::cross(left.q.vector(), &t) +
        *right
    }
}

impl<N: BaseNum + Clone> UnitQuatMulRhs<N, Pnt3<N>> for Pnt3<N> {
    #[inline]
    fn binop(left: &UnitQuat<N>, right: &Pnt3<N>) -> Pnt3<N> {
        ::orig::<Pnt3<N>>() + *left * *right.as_vec()
    }
}

impl<N: BaseNum + Clone> Vec3MulRhs<N, Vec3<N>> for UnitQuat<N> {
    #[inline]
    fn binop(left: &Vec3<N>, right: &UnitQuat<N>) -> Vec3<N> {
        let mut inv_quat = right.clone();
        inv_quat.inv();

        inv_quat * *left
    }
}

impl<N: BaseNum + Clone> Pnt3MulRhs<N, Pnt3<N>> for UnitQuat<N> {
    #[inline]
    fn binop(left: &Pnt3<N>, right: &UnitQuat<N>) -> Pnt3<N> {
        ::orig::<Pnt3<N>>() + *left.as_vec() * *right
    }
}

impl<N: BaseFloat + Clone> Rotation<Vec3<N>> for UnitQuat<N> {
    #[inline]
    fn rotation(&self) -> Vec3<N> {
        let _2 = ::one::<N>() + ::one();
        let mut v = self.q.vector().clone();
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
    fn append_rotation_cpy(t: &UnitQuat<N>, amount: &Vec3<N>) -> UnitQuat<N> {
        *t * UnitQuat::new(amount.clone())
    }

    #[inline]
    fn prepend_rotation(&mut self, amount: &Vec3<N>) {
        *self = Rotation::prepend_rotation_cpy(self, amount)
    }

    #[inline]
    fn prepend_rotation_cpy(t: &UnitQuat<N>, amount: &Vec3<N>) -> UnitQuat<N> {
        UnitQuat::new(amount.clone()) * *t
    }

    #[inline]
    fn set_rotation(&mut self, v: Vec3<N>) {
        *self = UnitQuat::new(v)
    }
}

impl<N: BaseNum + Clone> Rotate<Vec3<N>> for UnitQuat<N> {
    #[inline]
    fn rotate(&self, v: &Vec3<N>) -> Vec3<N> {
        *self * *v
    }

    #[inline]
    fn inv_rotate(&self, v: &Vec3<N>) -> Vec3<N> {
        *v * *self
    }
}

impl<N: BaseNum + Clone> Rotate<Pnt3<N>> for UnitQuat<N> {
    #[inline]
    fn rotate(&self, p: &Pnt3<N>) -> Pnt3<N> {
        *self * *p
    }

    #[inline]
    fn inv_rotate(&self, p: &Pnt3<N>) -> Pnt3<N> {
        *p * *self
    }
}

impl<N: BaseNum + Clone> Transform<Vec3<N>> for UnitQuat<N> {
    #[inline]
    fn transform(&self, v: &Vec3<N>) -> Vec3<N> {
        *self * *v
    }

    #[inline]
    fn inv_transform(&self, v: &Vec3<N>) -> Vec3<N> {
        *v * *self
    }
}

impl<N: BaseNum + Clone> Transform<Pnt3<N>> for UnitQuat<N> {
    #[inline]
    fn transform(&self, p: &Pnt3<N>) -> Pnt3<N> {
        *self * *p
    }

    #[inline]
    fn inv_transform(&self, p: &Pnt3<N>) -> Pnt3<N> {
        *p * *self
    }
}

double_dispatch_binop_decl_trait!(Quat, QuatMulRhs)
double_dispatch_binop_decl_trait!(Quat, QuatDivRhs)
double_dispatch_binop_decl_trait!(Quat, QuatAddRhs)
double_dispatch_binop_decl_trait!(Quat, QuatSubRhs)
double_dispatch_cast_decl_trait!(Quat, QuatCast)
mul_redispatch_impl!(Quat, QuatMulRhs)
div_redispatch_impl!(Quat, QuatDivRhs)
add_redispatch_impl!(Quat, QuatAddRhs)
sub_redispatch_impl!(Quat, QuatSubRhs)
cast_redispatch_impl!(Quat, QuatCast)
ord_impl!(Quat, w, i, j, k)
vec_axis_impl!(Quat, w, i, j, k)
vec_cast_impl!(Quat, QuatCast, w, i, j, k)
as_array_impl!(Quat, 4)
index_impl!(Quat)
indexable_impl!(Quat, 4)
at_fast_impl!(Quat, 4)
new_repeat_impl!(Quat, val, w, i, j, k)
dim_impl!(Quat, 3)
container_impl!(Quat)
add_impl!(Quat, QuatAddRhs, w, i, j, k)
sub_impl!(Quat, QuatSubRhs, w, i, j, k)
neg_impl!(Quat, w, i, j, k)
scalar_ops_impl!(Quat, w, i, j, k)
vec_mul_scalar_impl!(Quat, f64, QuatMulRhs, w, i, j, k)
vec_mul_scalar_impl!(Quat, f32, QuatMulRhs, w, i, j, k)
vec_mul_scalar_impl!(Quat, u64, QuatMulRhs, w, i, j, k)
vec_mul_scalar_impl!(Quat, u32, QuatMulRhs, w, i, j, k)
vec_mul_scalar_impl!(Quat, u16, QuatMulRhs, w, i, j, k)
vec_mul_scalar_impl!(Quat, u8, QuatMulRhs, w, i, j, k)
vec_mul_scalar_impl!(Quat, i64, QuatMulRhs, w, i, j, k)
vec_mul_scalar_impl!(Quat, i32, QuatMulRhs, w, i, j, k)
vec_mul_scalar_impl!(Quat, i16, QuatMulRhs, w, i, j, k)
vec_mul_scalar_impl!(Quat, i8, QuatMulRhs, w, i, j, k)
vec_mul_scalar_impl!(Quat, uint, QuatMulRhs, w, i, j, k)
vec_mul_scalar_impl!(Quat, int, QuatMulRhs, w, i, j, k)
vec_div_scalar_impl!(Quat, f64, QuatDivRhs, w, i, j, k)
vec_div_scalar_impl!(Quat, f32, QuatDivRhs, w, i, j, k)
vec_div_scalar_impl!(Quat, u64, QuatDivRhs, w, i, j, k)
vec_div_scalar_impl!(Quat, u32, QuatDivRhs, w, i, j, k)
vec_div_scalar_impl!(Quat, u16, QuatDivRhs, w, i, j, k)
vec_div_scalar_impl!(Quat, u8, QuatDivRhs, w, i, j, k)
vec_div_scalar_impl!(Quat, i64, QuatDivRhs, w, i, j, k)
vec_div_scalar_impl!(Quat, i32, QuatDivRhs, w, i, j, k)
vec_div_scalar_impl!(Quat, i16, QuatDivRhs, w, i, j, k)
vec_div_scalar_impl!(Quat, i8, QuatDivRhs, w, i, j, k)
vec_div_scalar_impl!(Quat, uint, QuatDivRhs, w, i, j, k)
vec_div_scalar_impl!(Quat, int, QuatDivRhs, w, i, j, k)
vec_add_scalar_impl!(Quat, f64, QuatAddRhs, w, i, j, k)
vec_add_scalar_impl!(Quat, f32, QuatAddRhs, w, i, j, k)
vec_add_scalar_impl!(Quat, u64, QuatAddRhs, w, i, j, k)
vec_add_scalar_impl!(Quat, u32, QuatAddRhs, w, i, j, k)
vec_add_scalar_impl!(Quat, u16, QuatAddRhs, w, i, j, k)
vec_add_scalar_impl!(Quat, u8, QuatAddRhs, w, i, j, k)
vec_add_scalar_impl!(Quat, i64, QuatAddRhs, w, i, j, k)
vec_add_scalar_impl!(Quat, i32, QuatAddRhs, w, i, j, k)
vec_add_scalar_impl!(Quat, i16, QuatAddRhs, w, i, j, k)
vec_add_scalar_impl!(Quat, i8, QuatAddRhs, w, i, j, k)
vec_add_scalar_impl!(Quat, uint, QuatAddRhs, w, i, j, k)
vec_add_scalar_impl!(Quat, int, QuatAddRhs, w, i, j, k)
vec_sub_scalar_impl!(Quat, f64, QuatSubRhs, w, i, j, k)
vec_sub_scalar_impl!(Quat, f32, QuatSubRhs, w, i, j, k)
vec_sub_scalar_impl!(Quat, u64, QuatSubRhs, w, i, j, k)
vec_sub_scalar_impl!(Quat, u32, QuatSubRhs, w, i, j, k)
vec_sub_scalar_impl!(Quat, u16, QuatSubRhs, w, i, j, k)
vec_sub_scalar_impl!(Quat, u8, QuatSubRhs, w, i, j, k)
vec_sub_scalar_impl!(Quat, i64, QuatSubRhs, w, i, j, k)
vec_sub_scalar_impl!(Quat, i32, QuatSubRhs, w, i, j, k)
vec_sub_scalar_impl!(Quat, i16, QuatSubRhs, w, i, j, k)
vec_sub_scalar_impl!(Quat, i8, QuatSubRhs, w, i, j, k)
vec_sub_scalar_impl!(Quat, uint, QuatSubRhs, w, i, j, k)
vec_sub_scalar_impl!(Quat, int, QuatSubRhs, w, i, j, k)
zero_one_impl!(Quat, w, i, j, k)
approx_eq_impl!(Quat, w, i, j, k)
from_iterator_impl!(Quat, iterator, iterator, iterator, iterator)
bounded_impl!(Quat, w, i, j, k)
axpy_impl!(Quat, w, i, j, k)
iterable_impl!(Quat, 4)
iterable_mut_impl!(Quat, 4)

double_dispatch_binop_decl_trait!(UnitQuat, UnitQuatMulRhs)
mul_redispatch_impl!(UnitQuat, UnitQuatMulRhs)
dim_impl!(UnitQuat, 3)
