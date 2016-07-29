//! Quaternion definition.

use std::fmt;
use std::mem;
use std::slice::{Iter, IterMut};
use std::ops::{Add, Sub, Mul, Div, Neg, AddAssign, SubAssign, MulAssign, DivAssign, Index, IndexMut};
use std::iter::{FromIterator, IntoIterator};
use rand::{Rand, Rng};
use num::{Zero, One};
use structs::{Vector3, Point3, Rotation3, Matrix3};
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

impl<N> Quaternion<N> {
    /// Creates a new quaternion from its components.
    #[inline]
    pub fn new(w: N, i: N, j: N, k: N) -> Quaternion<N> {
        Quaternion {
            w: w,
            i: i,
            j: j,
            k: k
        }
    }

    /// The vector part `(i, j, k)` of this quaternion.
    #[inline]
    pub fn vector<'a>(&'a self) -> &'a Vector3<N> {
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

impl<N: Neg<Output = N> + Copy> Quaternion<N> {
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

impl<N: BaseFloat + ApproxEq<N>> Inverse for Quaternion<N> {
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
            self.w = self.w / norm_squared;
            self.i = self.i / norm_squared;
            self.j = self.j / norm_squared;
            self.k = self.k / norm_squared;

            true
        }
    }
}

impl<N: BaseFloat> Norm<N> for Quaternion<N> {
    #[inline]
    fn norm_squared(&self) -> N {
        self.w * self.w + self.i * self.i + self.j * self.j + self.k * self.k
    }

    #[inline]
    fn normalize(&self) -> Quaternion<N> {
        let n = self.norm();
        Quaternion::new(self.w / n, self.i / n, self.j / n, self.k / n)
    }

    #[inline]
    fn normalize_mut(&mut self) -> N {
        let n = Norm::norm(self);

        self.w = self.w / n;
        self.i = self.i / n;
        self.j = self.j / n;
        self.k = self.k / n;

        n
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

impl<N: ApproxEq<N> + BaseFloat> Div<Quaternion<N>> for Quaternion<N> {
    type Output = Quaternion<N>;

    #[inline]
    fn div(self, right: Quaternion<N>) -> Quaternion<N> {
        self * right.inverse().expect("Unable to invert the denominator.")
    }
}

impl<N: ApproxEq<N> + BaseFloat> DivAssign<Quaternion<N>> for Quaternion<N> {
    #[inline]
    fn div_assign(&mut self, right: Quaternion<N>) {
        *self *= right.inverse().expect("Unable to invert the denominator.")
    }
}

impl<N: fmt::Display> fmt::Display for Quaternion<N> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Quaternion {} − ({}, {}, {})", self.w, self.i, self.j, self.k)
    }
}

rand_impl!(Quaternion, w, i, j, k);


/// A unit quaternion that can represent a 3D rotation.
#[repr(C)]
#[derive(Eq, PartialEq, RustcEncodable, RustcDecodable, Clone, Hash, Debug, Copy)]
pub struct UnitQuaternion<N> {
    q: Quaternion<N>
}

impl<N: BaseFloat> UnitQuaternion<N> {
    /// Creates a new unit quaternion from the axis-angle representation of a rotation.
    #[inline]
    pub fn new(axisangle: Vector3<N>) -> UnitQuaternion<N> {
        let sqang = Norm::norm_squared(&axisangle);

        if ::is_zero(&sqang) {
            ::one()
        }
        else {
            let ang    = sqang.sqrt();
            let (s, c) = (ang / Cast::from(2.0)).sin_cos();

            let s_ang = s / ang;

            unsafe {
                UnitQuaternion::new_with_unit_quaternion(
                    Quaternion::new(
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
    pub fn new_with_quaternion(q: Quaternion<N>) -> UnitQuaternion<N> {
        UnitQuaternion { q: q.normalize() }
    }

    /// Creates a new unit quaternion from Euler angles.
    ///
    /// The primitive rotations are applied in order: 1 roll − 2 pitch − 3 yaw.
    #[inline]
    pub fn new_with_euler_angles(roll: N, pitch: N, yaw: N) -> UnitQuaternion<N> {
        let _0_5: N  = Cast::from(0.5);
        let (sr, cr) = (roll * _0_5).sin_cos();
        let (sp, cp) = (pitch * _0_5).sin_cos();
        let (sy, cy) = (yaw * _0_5).sin_cos();

        unsafe {
            UnitQuaternion::new_with_unit_quaternion(
                Quaternion::new(
                    cr * cp * cy + sr * sp * sy,
                    sr * cp * cy - cr * sp * sy,
                    cr * sp * cy + sr * cp * sy,
                    cr * cp * sy - sr * sp * cy)
            )
        }
    }

    /// Builds a rotation matrix from this quaternion.
    pub fn to_rotation_matrix(&self) -> Rotation3<N> {
        let _2: N = Cast::from(2.0);
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
            Rotation3::new_with_matrix(
                Matrix3::new(
                    ww + ii - jj - kk, ij - wk,           wj + ik,
                    wk + ij,           ww - ii + jj - kk, jk - wi,
                    ik - wj,           wi + jk,           ww - ii - jj + kk
                )
            )
        }
    }
}


impl<N> UnitQuaternion<N> {
    /// Creates a new unit quaternion from a quaternion.
    ///
    /// This is unsafe because the input quaternion will not be normalized.
    #[inline]
    pub unsafe fn new_with_unit_quaternion(q: Quaternion<N>) -> UnitQuaternion<N> {
        UnitQuaternion {
            q: q
        }
    }

    /// The `Quaternion` representation of this unit quaternion.
    #[inline]
    pub fn quaternion<'a>(&'a self) -> &'a Quaternion<N> {
        &self.q
    }
}

impl<N: BaseNum> One for UnitQuaternion<N> {
    #[inline]
    fn one() -> UnitQuaternion<N> {
        unsafe {
            UnitQuaternion::new_with_unit_quaternion(Quaternion::new(::one(), ::zero(), ::zero(), ::zero()))
        }
    }
}

impl<N: Copy + Neg<Output = N>> Inverse for UnitQuaternion<N> {
    #[inline]
    fn inverse(&self) -> Option<UnitQuaternion<N>> {
        let mut cpy = *self;

        cpy.inverse_mut();
        Some(cpy)
    }

    #[inline]
    fn inverse_mut(&mut self) -> bool {
        self.q.conjugate_mut();

        true
    }
}

impl<N: Rand + BaseFloat> Rand for UnitQuaternion<N> {
    #[inline]
    fn rand<R: Rng>(rng: &mut R) -> UnitQuaternion<N> {
        UnitQuaternion::new(rng.gen())
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
        ApproxEq::approx_eq_eps(&self.q, &other.q, eps)
    }

    #[inline]
    fn approx_eq_ulps(&self, other: &UnitQuaternion<N>, ulps: u32) -> bool {
        ApproxEq::approx_eq_ulps(&self.q, &other.q, ulps)
    }
}

impl<N: BaseFloat + ApproxEq<N>> Div<UnitQuaternion<N>> for UnitQuaternion<N> {
    type Output = UnitQuaternion<N>;

    #[inline]
    fn div(self, other: UnitQuaternion<N>) -> UnitQuaternion<N> {
        UnitQuaternion { q: self.q / other.q }
    }
}

impl<N: BaseFloat + ApproxEq<N>> DivAssign<UnitQuaternion<N>> for UnitQuaternion<N> {
    #[inline]
    fn div_assign(&mut self, other: UnitQuaternion<N>) {
        self.q /= other.q
    }
}

impl<N: BaseNum> Mul<UnitQuaternion<N>> for UnitQuaternion<N> {
    type Output = UnitQuaternion<N>;

    #[inline]
    fn mul(self, right: UnitQuaternion<N>) -> UnitQuaternion<N> {
        UnitQuaternion { q: self.q * right.q }
    }
}

impl<N: BaseNum> MulAssign<UnitQuaternion<N>> for UnitQuaternion<N> {
    #[inline]
    fn mul_assign(&mut self, right: UnitQuaternion<N>) {
        self.q *= right.q
    }
}

impl<N: BaseNum> Mul<Vector3<N>> for UnitQuaternion<N> {
    type Output = Vector3<N>;

    #[inline]
    fn mul(self, right: Vector3<N>) -> Vector3<N> {
        let _2: N = ::one::<N>() + ::one();
        let mut t = ::cross(self.q.vector(), &right);
        t.x = t.x * _2;
        t.y = t.y * _2;
        t.z = t.z * _2;

        Vector3::new(t.x * self.q.w, t.y * self.q.w, t.z * self.q.w) + ::cross(self.q.vector(), &t) + right
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
        let _2 = ::one::<N>() + ::one();
        let mut v = *self.q.vector();
        let ang = _2 * v.normalize_mut().atan2(self.q.w);

        if ::is_zero(&ang) {
            ::zero()
        }
        else {
            Vector3::new(v.x * ang, v.y * ang, v.z * ang)
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
        *self * UnitQuaternion::new(*amount)
    }

    #[inline]
    fn prepend_rotation_mut(&mut self, amount: &Vector3<N>) {
        *self = Rotation::prepend_rotation(self, amount)
    }

    #[inline]
    fn prepend_rotation(&self, amount: &Vector3<N>) -> UnitQuaternion<N> {
        UnitQuaternion::new(*amount) * *self
    }

    #[inline]
    fn set_rotation(&mut self, v: Vector3<N>) {
        *self = UnitQuaternion::new(v)
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

impl<N: BaseFloat + ApproxEq<N>> RotationTo for UnitQuaternion<N> {
    type AngleType = N;
    type DeltaRotationType = UnitQuaternion<N>;

    #[inline]
    fn angle_to(&self, other: &Self) -> N {
        let delta = self.rotation_to(other);
        let _2    = ::one::<N>() + ::one();

        _2 * delta.q.vector().norm().atan2(delta.q.w)
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
        write!(f, "Unit quaternion {} − ({}, {}, {})", self.q.w, self.q.i, self.q.j, self.q.k)
    }
}

#[cfg(feature="arbitrary")]
impl<N: Arbitrary + BaseFloat> Arbitrary for UnitQuaternion<N> {
    fn arbitrary<G: Gen>(g: &mut G) -> UnitQuaternion<N> {
        UnitQuaternion::new(Arbitrary::arbitrary(g))
    }
}

impl Quaternion<f32> {
  /// Compute the exponential of a quaternion.
  ///
  /// This function yields a `UnitQuaternion<f32>`.
  pub fn exp(&self) -> UnitQuaternion<f32> {
    let v = Vector3::new(self.i, self.j, self.k);
    let n = v.norm();
    let z = v / n * n.sin();

    UnitQuaternion::new_with_quaternion(Quaternion::new(n.cos(), z[0], z[1], z[2]))
  }
}

impl UnitQuaternion<f32> {
  /// Compute the natural logarithm (a.k.a ln()) of a unit quaternion.
  ///
  /// Becareful, this function yields a `Quaternion<f32>`, losing the unit property.
  pub fn log(&self) -> Quaternion<f32> {
    let q = self.quaternion();
    (Quaternion { w: 0., .. *q }).normalize() * q.w.acos()
  }

  /// Raise the quaternion to a given floating power.
  pub fn powf(&self, n: f32) -> Self {
    (self.log() * n).exp()
  }
}

pord_impl!(Quaternion, w, i, j, k);
vec_axis_impl!(Quaternion, w, i, j, k);
vec_cast_impl!(Quaternion, w, i, j, k);
conversion_impl!(Quaternion, 4);
index_impl!(Quaternion);
indexable_impl!(Quaternion, 4);
at_fast_impl!(Quaternion, 4);
repeat_impl!(Quaternion, val, w, i, j, k);
dim_impl!(Quaternion, 3);
container_impl!(Quaternion);
add_impl!(Quaternion, w, i, j, k);
sub_impl!(Quaternion, w, i, j, k);
scalar_add_impl!(Quaternion, w, i, j, k);
scalar_sub_impl!(Quaternion, w, i, j, k);
scalar_mul_impl!(Quaternion, w, i, j, k);
scalar_div_impl!(Quaternion, w, i, j, k);
neg_impl!(Quaternion, w, i, j, k);
zero_one_impl!(Quaternion, w, i, j, k);
approx_eq_impl!(Quaternion, w, i, j, k);
from_iterator_impl!(Quaternion, iterator, iterator, iterator, iterator);
bounded_impl!(Quaternion, w, i, j, k);
axpy_impl!(Quaternion, w, i, j, k);
iterable_impl!(Quaternion, 4);
iterable_mut_impl!(Quaternion, 4);
arbitrary_impl!(Quaternion, w, i, j, k);

dim_impl!(UnitQuaternion, 3);
