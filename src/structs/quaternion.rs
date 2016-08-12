//! Quaternion definition.

use std::fmt;
use std::mem;
use std::slice::{Iter, IterMut};
use std::ops::{Add, Sub, Mul, Div, Neg, AddAssign, SubAssign, MulAssign, DivAssign, Index, IndexMut};
use std::iter::{FromIterator, IntoIterator};
use rand::{Rand, Rng};
use num::{Zero, One};
use structs::{Vector3, Point3, Rotation3, Matrix3, Unit};
use traits::operations::{ApproxEq, Inverse, PartialOrder, PartialOrdering, Axpy};
use traits::structure::{Cast, Indexable, Iterable, IterableMut, Dimension, Shape, BaseFloat, BaseNum,
                        Bounded, Repeat};
use traits::geometry::{Norm, Rotation, RotationMatrix, Rotate, RotationTo, Transform};

#[cfg(feature="arbitrary")]
use quickcheck::{Arbitrary, Gen};


/// A quaternion. See `UnitQuaternion` for a quaternion that can be used as a rotation.
#[repr(C)]
#[derive(Eq, PartialEq, RustcEncodable, RustcDecodable, Clone, Hash, Debug, Copy)]
pub struct Quaternion<N> {
    /// The scalar component of the quaternion.
    pub w: N,
    /// The first vector component of the quaternion.
    pub i: N,
    /// The second vector component of the quaternion.
    pub j: N,
    /// The third vector component of the quaternion.
    pub k: N
}

impl<N: Copy> Quaternion<N> {
    /// The vector part `(i, j, k)` of this quaternion.
    #[inline]
    pub fn vector(&self) -> &Vector3<N> {
        unsafe { mem::transmute(&self.i) }
    }

    /// The scalar part `w` of this quaternion.
    #[inline]
    pub fn scalar(&self) -> N {
        self.w
    }
}

impl<N: BaseNum + Neg<Output = N>> Quaternion<N> {
    /// Compute the conjugate of this quaternion.
    #[inline]
    pub fn conjugate(&self) -> Quaternion<N> {
        Quaternion { w: self.w, i: -self.i, j: -self.j, k: -self.k }
    }

    /// Replaces this quaternion by its conjugate.
    #[inline]
    pub fn conjugate_mut(&mut self) {
        self.i = -self.i;
        self.j = -self.j;
        self.k = -self.k;
    }
}

impl<N: BaseFloat> Quaternion<N> {
    /// Creates a new quaternion from its scalar and vector parts.
    pub fn from_parts(scalar: N, vector: Vector3<N>) -> Quaternion<N> {
        Quaternion::new(scalar, vector.x, vector.y, vector.z)
    }

    /// Creates a new quaternion from its polar decomposition.
    ///
    /// Note that `axis` is assumed to be a unit vector.
    pub fn from_polar_decomposition(scalar: N, theta: N, axis: Unit<Vector3<N>>) -> Quaternion<N> {
        let rot = UnitQuaternion::from_axisangle(axis, theta * ::cast(2.0));

        rot.unwrap() * scalar
    }

    /// The polar decomposition of this quaternion.
    ///
    /// Returns, from left to right: the quaternion norm, the half rotation angle, the rotation
    /// axis. If the rotation angle is zero, the rotation axis is set to the `y` axis.
    pub fn polar_decomposition(&self) -> (N, N, Unit<Vector3<N>>) {
        let nn = ::norm_squared(self);

        let default_axis = Unit::from_unit_value_unchecked(Vector3::y());

        if ApproxEq::approx_eq(&nn, &::zero()) {
            (::zero(), ::zero(), default_axis)
        }
        else {
            let n   = nn.sqrt();
            let nq  = *self / n;
            let v   = *self.vector();
            let vnn = ::norm_squared(&v);

            if ApproxEq::approx_eq(&vnn, &::zero()) {
                (n, ::zero(), default_axis)
            }
            else {
                let angle = self.scalar().acos();
                let vn    = n.sqrt();
                let axis  = Unit::from_unit_value_unchecked(v / vn);

                (n, angle, axis)
            }
        }
    }
}

impl<N: BaseFloat> Inverse for Quaternion<N> {
    #[inline]
    fn inverse(&self) -> Option<Quaternion<N>> {
        let mut res = *self;

        if res.inverse_mut() {
            Some(res)
        }
        else {
            None
        }
    }

    #[inline]
    fn inverse_mut(&mut self) -> bool {
        let norm_squared = Norm::norm_squared(self);

        if ApproxEq::approx_eq(&norm_squared, &::zero()) {
            false
        }
        else {
            self.conjugate_mut();
            *self /= norm_squared;

            true
        }
    }
}

impl<N: BaseFloat> Norm for Quaternion<N> {
    type NormType = N;

    #[inline]
    fn norm_squared(&self) -> N {
        self.w * self.w + self.i * self.i + self.j * self.j + self.k * self.k
    }

    #[inline]
    fn normalize(&self) -> Quaternion<N> {
        let n = self.norm();
        *self / n
    }

    #[inline]
    fn normalize_mut(&mut self) -> N {
        let n = ::norm(self);
        *self /= n;

        n
    }

    #[inline]
    fn try_normalize(&self, min_norm: N) -> Option<Quaternion<N>> {
        let n = ::norm(self);

        if n <= min_norm {
            None
        }
        else {
            Some(*self / n)
        }
    }
}

impl<N> Mul<Quaternion<N>> for Quaternion<N>
    where N: Copy + Mul<N, Output = N> + Sub<N, Output = N> + Add<N, Output = N> {
    type Output = Quaternion<N>;

    #[inline]
    fn mul(self, right: Quaternion<N>) -> Quaternion<N> {
        Quaternion::new(
            self.w * right.w - self.i * right.i - self.j * right.j - self.k * right.k,
            self.w * right.i + self.i * right.w + self.j * right.k - self.k * right.j,
            self.w * right.j - self.i * right.k + self.j * right.w + self.k * right.i,
            self.w * right.k + self.i * right.j - self.j * right.i + self.k * right.w)
    }
}

impl<N> MulAssign<Quaternion<N>> for Quaternion<N>
    where N: Copy + Mul<N, Output = N> + Sub<N, Output = N> + Add<N, Output = N> {
    #[inline]
    fn mul_assign(&mut self, right: Quaternion<N>) {
        *self = *self * right;
    }
}

impl<N: BaseFloat> Div<Quaternion<N>> for Quaternion<N> {
    type Output = Quaternion<N>;

    #[inline]
    fn div(self, right: Quaternion<N>) -> Quaternion<N> {
        self * right.inverse().expect("Unable to invert the denominator.")
    }
}

impl<N: BaseFloat> DivAssign<Quaternion<N>> for Quaternion<N> {
    #[inline]
    fn div_assign(&mut self, right: Quaternion<N>) {
        *self *= right.inverse().expect("Unable to invert the denominator.")
    }
}

impl<N: BaseFloat> Quaternion<N> {
    /// Compute the exponential of a quaternion.
    #[inline]
    pub fn exp(&self) -> Self {
        let v = *self.vector();
        let nn = v.norm_squared();

        if nn.is_zero() {
            ::one()
        }
        else {
            let n  = nn.sqrt();
            let nv = v / n * n.sin();
            Quaternion::from_parts(n.cos(), nv) * self.scalar().exp()
        }
    }

    /// Compute the natural logarithm of a quaternion.
    #[inline]
    pub fn ln(&self) -> Self {
        let n = self.norm();
        let v = self.vector();
        let s = self.scalar();

        Quaternion::from_parts(n.ln(), v.normalize() *  (s / n).acos())
    }

    /// Raise the quaternion to a given floating power.
    #[inline]
    pub fn powf(&self, n: N) -> Self {
        (self.ln() * n).exp()
    }
}

impl<T> One for Quaternion<T> where T: Copy + One + Zero + Sub<T, Output = T> + Add<T, Output = T> {
    #[inline]
  fn one() -> Self {
    Quaternion::new(T::one(), T::zero(), T::zero(), T::zero())
  }
}

impl<N: fmt::Display> fmt::Display for Quaternion<N> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Quaternion {} − ({}, {}, {})", self.w, self.i, self.j, self.k)
    }
}

/// A unit quaternions. May be used to represent a rotation.
pub type UnitQuaternion<N> = Unit<Quaternion<N>>;

// /// A unit quaternion that can represent a 3D rotation.
// #[repr(C)]
// #[derive(Eq, PartialEq, RustcEncodable, RustcDecodable, Clone, Hash, Debug, Copy)]
// pub struct UnitQuaternion<N> {
//     q: Quaternion<N>
// }

impl<N: BaseFloat> UnitQuaternion<N> {
    /// Creates a new quaternion from a unit vector (the rotation axis) and an angle
    /// (the rotation angle).
    #[inline]
    pub fn from_axisangle(axis: Unit<Vector3<N>>, angle: N) -> UnitQuaternion<N> {
        let (sang, cang) = (angle / ::cast(2.0)).sin_cos();

        let q = Quaternion::from_parts(cang, axis.unwrap() * sang);
        Unit::from_unit_value_unchecked(q)
    }

    /// Same as `::from_axisangle` with the axis multiplied with the angle.
    #[inline]
    pub fn from_scaled_axis(axis: Vector3<N>) -> UnitQuaternion<N> {
        let two: N = ::cast(2.0);
        let q = Quaternion::from_parts(::zero(), axis * two).exp();
        UnitQuaternion::from_unit_value_unchecked(q)
    }

    /// Creates a new unit quaternion from a quaternion.
    ///
    /// The input quaternion will be normalized.
    #[inline]
    pub fn from_quaternion(q: &Quaternion<N>) -> UnitQuaternion<N> {
        Unit::new(&q)
    }

    /// Creates a new unit quaternion from Euler angles.
    ///
    /// The primitive rotations are applied in order: 1 roll − 2 pitch − 3 yaw.
    #[inline]
    pub fn from_euler_angles(roll: N, pitch: N, yaw: N) -> UnitQuaternion<N> {
        let (sr, cr) = (roll  * ::cast(0.5)).sin_cos();
        let (sp, cp) = (pitch * ::cast(0.5)).sin_cos();
        let (sy, cy) = (yaw   * ::cast(0.5)).sin_cos();

        let q = Quaternion::new(
                    cr * cp * cy + sr * sp * sy,
                    sr * cp * cy - cr * sp * sy,
                    cr * sp * cy + sr * cp * sy,
                    cr * cp * sy - sr * sp * cy);

        Unit::from_unit_value_unchecked(q)
    }

    /// The rotation angle of this unit quaternion.
    #[inline]
    pub fn angle(&self) -> N {
        self.as_ref().scalar().acos()
    }

    /// The rotation axis of this unit quaternion or `None` if the rotation is zero.
    #[inline]
    pub fn axis(&self) -> Option<Unit<Vector3<N>>> {
        Unit::try_new(self.as_ref().vector(), ::zero())
    }

    /// Builds a rotation matrix from this quaternion.
    pub fn to_rotation_matrix(&self) -> Rotation3<N> {
        let ww = self.as_ref().w * self.as_ref().w;
        let ii = self.as_ref().i * self.as_ref().i;
        let jj = self.as_ref().j * self.as_ref().j;
        let kk = self.as_ref().k * self.as_ref().k;
        let ij = self.as_ref().i * self.as_ref().j * ::cast(2.0);
        let wk = self.as_ref().w * self.as_ref().k * ::cast(2.0);
        let wj = self.as_ref().w * self.as_ref().j * ::cast(2.0);
        let ik = self.as_ref().i * self.as_ref().k * ::cast(2.0);
        let jk = self.as_ref().j * self.as_ref().k * ::cast(2.0);
        let wi = self.as_ref().w * self.as_ref().i * ::cast(2.0);

        Rotation3::new_with_matrix_unchecked(
            Matrix3::new(
                ww + ii - jj - kk, ij - wk,           wj + ik,
                wk + ij,           ww - ii + jj - kk, jk - wi,
                ik - wj,           wi + jk,           ww - ii - jj + kk
            )
        )
    }
}

impl<N: BaseNum> One for UnitQuaternion<N> {
    #[inline]
    fn one() -> UnitQuaternion<N> {
        let one = Quaternion::new(::one(), ::zero(), ::zero(), ::zero());
        UnitQuaternion::from_unit_value_unchecked(one)
    }
}

impl<N: BaseNum + Neg<Output = N>> Inverse for UnitQuaternion<N> {
    #[inline]
    fn inverse(&self) -> Option<UnitQuaternion<N>> {
        let mut cpy = *self;

        cpy.inverse_mut();
        Some(cpy)
    }

    #[inline]
    fn inverse_mut(&mut self) -> bool {
        *self = Unit::from_unit_value_unchecked(self.as_ref().conjugate());

        true
    }
}

impl<N: Rand + BaseFloat> Rand for UnitQuaternion<N> {
    #[inline]
    fn rand<R: Rng>(rng: &mut R) -> UnitQuaternion<N> {
        UnitQuaternion::new(&rng.gen())
    }
}

impl<N: ApproxEq<N>> ApproxEq<N> for UnitQuaternion<N> {
    #[inline]
    fn approx_epsilon(_: Option<UnitQuaternion<N>>) -> N {
        ApproxEq::approx_epsilon(None::<N>)
    }

    #[inline]
    fn approx_ulps(_: Option<UnitQuaternion<N>>) -> u32 {
        ApproxEq::approx_ulps(None::<N>)
    }

    #[inline]
    fn approx_eq_eps(&self, other: &UnitQuaternion<N>, eps: &N) -> bool {
        ApproxEq::approx_eq_eps(self.as_ref(), other.as_ref(), eps)
    }

    #[inline]
    fn approx_eq_ulps(&self, other: &UnitQuaternion<N>, ulps: u32) -> bool {
        ApproxEq::approx_eq_ulps(self.as_ref(), other.as_ref(), ulps)
    }
}

impl<N: BaseFloat> Div<UnitQuaternion<N>> for UnitQuaternion<N> {
    type Output = UnitQuaternion<N>;

    #[inline]
    fn div(self, other: UnitQuaternion<N>) -> UnitQuaternion<N> {
        Unit::from_unit_value_unchecked(self.unwrap() / other.unwrap())
    }
}

impl<N: BaseFloat> DivAssign<UnitQuaternion<N>> for UnitQuaternion<N> {
    #[inline]
    fn div_assign(&mut self, other: UnitQuaternion<N>) {
        *self = Unit::from_unit_value_unchecked(*self.as_ref() / *other.as_ref())
    }
}

impl<N: BaseNum> Mul<UnitQuaternion<N>> for UnitQuaternion<N> {
    type Output = UnitQuaternion<N>;

    #[inline]
    fn mul(self, right: UnitQuaternion<N>) -> UnitQuaternion<N> {
        Unit::from_unit_value_unchecked(self.unwrap() * right.unwrap())
    }
}

impl<N: BaseNum> MulAssign<UnitQuaternion<N>> for UnitQuaternion<N> {
    #[inline]
    fn mul_assign(&mut self, right: UnitQuaternion<N>) {
        *self = Unit::from_unit_value_unchecked(*self.as_ref() * *right.as_ref())
    }
}

impl<N: BaseNum> Mul<Vector3<N>> for UnitQuaternion<N> {
    type Output = Vector3<N>;

    #[inline]
    fn mul(self, right: Vector3<N>) -> Vector3<N> {
        let two: N = ::one::<N>() + ::one();
        let t = ::cross(self.as_ref().vector(), &right);

        t * (two * self.as_ref().w) + ::cross(self.as_ref().vector(), &t) + right
    }
}

impl<N: BaseNum> Mul<Point3<N>> for UnitQuaternion<N> {
    type Output = Point3<N>;

    #[inline]
    fn mul(self, right: Point3<N>) -> Point3<N> {
        ::origin::<Point3<N>>() + self * *right.as_vector()
    }
}

impl<N: BaseNum + Neg<Output = N>> Mul<UnitQuaternion<N>> for Vector3<N> {
    type Output = Vector3<N>;

    #[inline]
    fn mul(self, right: UnitQuaternion<N>) -> Vector3<N> {
        let mut inverse_quaternion = right;

        inverse_quaternion.inverse_mut();

        inverse_quaternion * self
    }
}

impl<N: BaseNum + Neg<Output = N>> Mul<UnitQuaternion<N>> for Point3<N> {
    type Output = Point3<N>;

    #[inline]
    fn mul(self, right: UnitQuaternion<N>) -> Point3<N> {
        ::origin::<Point3<N>>() + *self.as_vector() * right
    }
}

impl<N: BaseNum + Neg<Output = N>> MulAssign<UnitQuaternion<N>> for Vector3<N> {
    #[inline]
    fn mul_assign(&mut self, right: UnitQuaternion<N>) {
        *self = *self * right
    }
}

impl<N: BaseNum + Neg<Output = N>> MulAssign<UnitQuaternion<N>> for Point3<N> {
    #[inline]
    fn mul_assign(&mut self, right: UnitQuaternion<N>) {
        *self = *self * right
    }
}

impl<N: BaseFloat> Rotation<Vector3<N>> for UnitQuaternion<N> {
    #[inline]
    fn rotation(&self) -> Vector3<N> {
        if let Some(v) = self.axis() {
            v.unwrap() * self.angle()
        }
        else {
            ::zero()
        }
    }

    #[inline]
    fn inverse_rotation(&self) -> Vector3<N> {
        -self.rotation()
    }

    #[inline]
    fn append_rotation_mut(&mut self, amount: &Vector3<N>) {
        *self = Rotation::append_rotation(self, amount)
    }

    #[inline]
    fn append_rotation(&self, amount: &Vector3<N>) -> UnitQuaternion<N> {
        *self * UnitQuaternion::from_scaled_axis(*amount)
    }

    #[inline]
    fn prepend_rotation_mut(&mut self, amount: &Vector3<N>) {
        *self = Rotation::prepend_rotation(self, amount)
    }

    #[inline]
    fn prepend_rotation(&self, amount: &Vector3<N>) -> UnitQuaternion<N> {
        UnitQuaternion::from_scaled_axis(*amount) * *self
    }

    #[inline]
    fn set_rotation(&mut self, v: Vector3<N>) {
        *self = UnitQuaternion::from_scaled_axis(v);
    }
}

impl<N: BaseFloat> RotationMatrix<N, Vector3<N>, Vector3<N>> for UnitQuaternion<N> {
    type Output = Rotation3<N>;

    #[inline]
    fn to_rotation_matrix(&self) -> Rotation3<N> {
        self.to_rotation_matrix()
    }
}

impl<N: BaseNum + Neg<Output = N>> Rotate<Vector3<N>> for UnitQuaternion<N> {
    #[inline]
    fn rotate(&self, v: &Vector3<N>) -> Vector3<N> {
        *self * *v
    }

    #[inline]
    fn inverse_rotate(&self, v: &Vector3<N>) -> Vector3<N> {
        *v * *self
    }
}

impl<N: BaseNum + Neg<Output = N>> Rotate<Point3<N>> for UnitQuaternion<N> {
    #[inline]
    fn rotate(&self, p: &Point3<N>) -> Point3<N> {
        *self * *p
    }

    #[inline]
    fn inverse_rotate(&self, p: &Point3<N>) -> Point3<N> {
        *p * *self
    }
}

impl<N: BaseFloat> RotationTo for UnitQuaternion<N> {
    type AngleType = N;
    type DeltaRotationType = UnitQuaternion<N>;

    #[inline]
    fn angle_to(&self, other: &Self) -> N {
        let delta = self.rotation_to(other);

        delta.as_ref().w.acos() * ::cast(2.0)
    }

    #[inline]
    fn rotation_to(&self, other: &Self) -> UnitQuaternion<N> {
        *other / *self
    }
}

impl<N: BaseNum + Neg<Output = N>> Transform<Vector3<N>> for UnitQuaternion<N> {
    #[inline]
    fn transform(&self, v: &Vector3<N>) -> Vector3<N> {
        *self * *v
    }

    #[inline]
    fn inverse_transform(&self, v: &Vector3<N>) -> Vector3<N> {
        *v * *self
    }
}

impl<N: BaseNum + Neg<Output = N>> Transform<Point3<N>> for UnitQuaternion<N> {
    #[inline]
    fn transform(&self, p: &Point3<N>) -> Point3<N> {
        *self * *p
    }

    #[inline]
    fn inverse_transform(&self, p: &Point3<N>) -> Point3<N> {
        *p * *self
    }
}

impl<N: fmt::Display> fmt::Display for UnitQuaternion<N> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Unit quaternion {} − ({}, {}, {})",
               self.as_ref().w, self.as_ref().i, self.as_ref().j, self.as_ref().k)
    }
}

/*
 *
 * Dimension
 *
 */
impl<N> Dimension for UnitQuaternion<N> {
    #[inline]
    fn dimension(_: Option<UnitQuaternion<N>>) -> usize {
        3
    }
}

#[cfg(feature="arbitrary")]
impl<N: Arbitrary + BaseFloat> Arbitrary for UnitQuaternion<N> {
    fn arbitrary<G: Gen>(g: &mut G) -> UnitQuaternion<N> {
        UnitQuaternion::new(&Arbitrary::arbitrary(g))
    }
}

impl<N: BaseFloat> UnitQuaternion<N> {
    /// Compute the exponential of a quaternion.
    ///
    /// Note that this function yields a `Quaternion<N>` because it looses the unit property.
    pub fn exp(&self) -> Quaternion<N> {
        self.as_ref().exp()
    }

    /// Compute the natural logarithm of a quaternion.
    ///
    /// Note that this function yields a `Quaternion<N>` because it looses the unit property.  The
    /// vector part of the return value corresponds to the axis-angle representation (divided by
    /// 2.0) of this unit quaternion.
    pub fn ln(&self) -> Quaternion<N> {
        if let Some(v) = self.axis() {
            Quaternion::from_parts(::zero(), v.unwrap() * self.angle())
        }
        else {
            ::zero()
        }
    }

    /// Raise this unit quaternion to a given floating power.
    ///
    /// If this unit quaternion represents a rotation by `theta`, then the resulting quaternion
    /// rotates by `n * theta`.
    pub fn powf(&self, n: N) -> Self {
        if let Some(v) = self.axis() {
            UnitQuaternion::from_axisangle(v, self.angle() * n)
        }
        else {
            ::one()
        }
    }
}

componentwise_zero!(Quaternion, w, i, j, k);
component_basis_element!(Quaternion, w, i, j, k);
pointwise_add!(Quaternion, w, i, j, k);
pointwise_sub!(Quaternion, w, i, j, k);
from_iterator_impl!(Quaternion, iterator, iterator, iterator, iterator);
vectorlike_impl!(Quaternion, 4, w, i, j, k);
