use approx::{AbsDiffEq, RelativeEq, UlpsEq};
use num::Zero;
use std::fmt;
use std::hash;
#[cfg(feature = "abomonation-serialize")]
use std::io::{Result as IOResult, Write};

#[cfg(feature = "serde-serialize")]
use base::storage::Owned;
#[cfg(feature = "serde-serialize")]
use serde::{Deserialize, Deserializer, Serialize, Serializer};

#[cfg(feature = "abomonation-serialize")]
use abomonation::Abomonation;

use alga::general::Real;

use base::dimension::{U1, U3, U4};
use base::storage::{CStride, RStride};
use base::{Matrix3, MatrixN, MatrixSlice, MatrixSliceMut, Unit, Vector3, Vector4};

use geometry::Rotation;

/// A quaternion. See the type alias `UnitQuaternion = Unit<Quaternion>` for a quaternion
/// that may be used as a rotation.
#[repr(C)]
#[derive(Debug)]
pub struct Quaternion<N: Real> {
    /// This quaternion as a 4D vector of coordinates in the `[ x, y, z, w ]` storage order.
    pub coords: Vector4<N>,
}

#[cfg(feature = "abomonation-serialize")]
impl<N: Real> Abomonation for Quaternion<N>
where Vector4<N>: Abomonation
{
    unsafe fn entomb<W: Write>(&self, writer: &mut W) -> IOResult<()> {
        self.coords.entomb(writer)
    }

    fn extent(&self) -> usize {
        self.coords.extent()
    }

    unsafe fn exhume<'a, 'b>(&'a mut self, bytes: &'b mut [u8]) -> Option<&'b mut [u8]> {
        self.coords.exhume(bytes)
    }
}

impl<N: Real + Eq> Eq for Quaternion<N> {}

impl<N: Real> PartialEq for Quaternion<N> {
    fn eq(&self, rhs: &Self) -> bool {
        self.coords == rhs.coords ||
        // Account for the double-covering of S², i.e. q = -q
        self.as_vector().iter().zip(rhs.as_vector().iter()).all(|(a, b)| *a == -*b)
    }
}

impl<N: Real + hash::Hash> hash::Hash for Quaternion<N> {
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        self.coords.hash(state)
    }
}

impl<N: Real> Copy for Quaternion<N> {}

impl<N: Real> Clone for Quaternion<N> {
    #[inline]
    fn clone(&self) -> Self {
        Self::from(self.coords.clone())
    }
}

#[cfg(feature = "serde-serialize")]
impl<N: Real> Serialize for Quaternion<N>
where Owned<N, U4>: Serialize
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where S: Serializer {
        self.coords.serialize(serializer)
    }
}

#[cfg(feature = "serde-serialize")]
impl<'a, N: Real> Deserialize<'a> for Quaternion<N>
where Owned<N, U4>: Deserialize<'a>
{
    fn deserialize<Des>(deserializer: Des) -> Result<Self, Des::Error>
    where Des: Deserializer<'a> {
        let coords = Vector4::<N>::deserialize(deserializer)?;

        Ok(Self::from(coords))
    }
}

impl<N: Real> Quaternion<N> {
    /// Moves this unit quaternion into one that owns its data.
    #[inline]
    #[deprecated(note = "This method is a no-op and will be removed in a future release.")]
    pub fn into_owned(self) -> Self {
        self
    }

    /// Clones this unit quaternion into one that owns its data.
    #[inline]
    #[deprecated(note = "This method is a no-op and will be removed in a future release.")]
    pub fn clone_owned(&self) -> Self {
        Self::from(self.coords.clone_owned())
    }

    /// Normalizes this quaternion.
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::Quaternion;
    /// let q = Quaternion::new(1.0, 2.0, 3.0, 4.0);
    /// let q_normalized = q.normalize();
    /// relative_eq!(q_normalized.norm(), 1.0);
    /// ```
    #[inline]
    pub fn normalize(&self) -> Self {
        Self::from(self.coords.normalize())
    }

    /// The imaginary part of this quaternion.
    #[inline]
    pub fn imag(&self) -> Vector3<N> {
        self.coords.xyz()
    }

    /// The conjugate of this quaternion.
    ///
    /// # Example
    /// ```
    /// # use nalgebra::Quaternion;
    /// let q = Quaternion::new(1.0, 2.0, 3.0, 4.0);
    /// let conj = q.conjugate();
    /// assert!(conj.i == -2.0 && conj.j == -3.0 && conj.k == -4.0 && conj.w == 1.0);
    /// ```
    #[inline]
    pub fn conjugate(&self) -> Self {
        Self::from_parts(self.w, -self.imag())
    }

    /// Inverts this quaternion if it is not zero.
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::Quaternion;
    /// let q = Quaternion::new(1.0, 2.0, 3.0, 4.0);
    /// let inv_q = q.try_inverse();
    ///
    /// assert!(inv_q.is_some());
    /// assert_relative_eq!(inv_q.unwrap() * q, Quaternion::identity());
    ///
    /// //Non-invertible case
    /// let q = Quaternion::new(0.0, 0.0, 0.0, 0.0);
    /// let inv_q = q.try_inverse();
    ///
    /// assert!(inv_q.is_none());
    /// ```
    #[inline]
    pub fn try_inverse(&self) -> Option<Self> {
        let mut res = Self::from(self.coords.clone_owned());

        if res.try_inverse_mut() {
            Some(res)
        } else {
            None
        }
    }

    /// Linear interpolation between two quaternion.
    ///
    /// Computes `self * (1 - t) + other * t`.
    ///
    /// # Example
    /// ```
    /// # use nalgebra::Quaternion;
    /// let q1 = Quaternion::new(1.0, 2.0, 3.0, 4.0);
    /// let q2 = Quaternion::new(10.0, 20.0, 30.0, 40.0);
    ///
    /// assert_eq!(q1.lerp(&q2, 0.1), Quaternion::new(1.9, 3.8, 5.7, 7.6));
    /// ```
    #[inline]
    pub fn lerp(&self, other: &Self, t: N) -> Self {
        self * (N::one() - t) + other * t
    }

    /// The vector part `(i, j, k)` of this quaternion.
    ///
    /// # Example
    /// ```
    /// # use nalgebra::Quaternion;
    /// let q = Quaternion::new(1.0, 2.0, 3.0, 4.0);
    /// assert_eq!(q.vector()[0], 2.0);
    /// assert_eq!(q.vector()[1], 3.0);
    /// assert_eq!(q.vector()[2], 4.0);
    /// ```
    #[inline]
    pub fn vector(&self) -> MatrixSlice<N, U3, U1, RStride<N, U4, U1>, CStride<N, U4, U1>> {
        self.coords.fixed_rows::<U3>(0)
    }

    /// The scalar part `w` of this quaternion.
    ///
    /// # Example
    /// ```
    /// # use nalgebra::Quaternion;
    /// let q = Quaternion::new(1.0, 2.0, 3.0, 4.0);
    /// assert_eq!(q.scalar(), 1.0);
    /// ```
    #[inline]
    pub fn scalar(&self) -> N {
        self.coords[3]
    }

    /// Reinterprets this quaternion as a 4D vector.
    ///
    /// # Example
    /// ```
    /// # use nalgebra::{Vector4, Quaternion};
    /// let q = Quaternion::new(1.0, 2.0, 3.0, 4.0);
    /// // Recall that the quaternion is stored internally as (i, j, k, w)
    /// // while the ::new constructor takes the arguments as (w, i, j, k).
    /// assert_eq!(*q.as_vector(), Vector4::new(2.0, 3.0, 4.0, 1.0));
    /// ```
    #[inline]
    pub fn as_vector(&self) -> &Vector4<N> {
        &self.coords
    }

    /// The norm of this quaternion.
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::Quaternion;
    /// let q = Quaternion::new(1.0, 2.0, 3.0, 4.0);
    /// assert_relative_eq!(q.norm(), 5.47722557, epsilon = 1.0e-6);
    /// ```
    #[inline]
    pub fn norm(&self) -> N {
        self.coords.norm()
    }

    /// A synonym for the norm of this quaternion.
    ///
    /// Aka the length.
    /// This is the same as `.norm()`
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::Quaternion;
    /// let q = Quaternion::new(1.0, 2.0, 3.0, 4.0);
    /// assert_relative_eq!(q.magnitude(), 5.47722557, epsilon = 1.0e-6);
    /// ```
    #[inline]
    pub fn magnitude(&self) -> N {
        self.norm()
    }

    /// The squared norm of this quaternion.
    ///
    /// # Example
    /// ```
    /// # use nalgebra::Quaternion;
    /// let q = Quaternion::new(1.0, 2.0, 3.0, 4.0);
    /// assert_eq!(q.magnitude_squared(), 30.0);
    /// ```
    #[inline]
    pub fn norm_squared(&self) -> N {
        self.coords.norm_squared()
    }

    /// A synonym for the squared norm of this quaternion.
    ///
    /// Aka the squared length.
    /// This is the same as `.norm_squared()`
    ///
    /// # Example
    /// ```
    /// # use nalgebra::Quaternion;
    /// let q = Quaternion::new(1.0, 2.0, 3.0, 4.0);
    /// assert_eq!(q.magnitude_squared(), 30.0);
    /// ```
    #[inline]
    pub fn magnitude_squared(&self) -> N {
        self.norm_squared()
    }

    /// The dot product of two quaternions.
    ///
    /// # Example
    /// ```
    /// # use nalgebra::Quaternion;
    /// let q1 = Quaternion::new(1.0, 2.0, 3.0, 4.0);
    /// let q2 = Quaternion::new(5.0, 6.0, 7.0, 8.0);
    /// assert_eq!(q1.dot(&q2), 70.0);
    /// ```
    #[inline]
    pub fn dot(&self, rhs: &Self) -> N {
        self.coords.dot(&rhs.coords)
    }

    /// The polar decomposition of this quaternion.
    ///
    /// Returns, from left to right: the quaternion norm, the half rotation angle, the rotation
    /// axis. If the rotation angle is zero, the rotation axis is set to `None`.
    ///
    /// # Example
    /// ```
    /// # use std::f32;
    /// # use nalgebra::{Vector3, Quaternion};
    /// let q = Quaternion::new(0.0, 5.0, 0.0, 0.0);
    /// let (norm, half_ang, axis) = q.polar_decomposition();
    /// assert_eq!(norm, 5.0);
    /// assert_eq!(half_ang, f32::consts::FRAC_PI_2);
    /// assert_eq!(axis, Some(Vector3::x_axis()));
    /// ```
    pub fn polar_decomposition(&self) -> (N, N, Option<Unit<Vector3<N>>>) {
        if let Some((q, n)) = Unit::try_new_and_get(*self, N::zero()) {
            if let Some(axis) = Unit::try_new(self.vector().clone_owned(), N::zero()) {
                let angle = q.angle() / ::convert(2.0f64);

                (n, angle, Some(axis))
            } else {
                (n, N::zero(), None)
            }
        } else {
            (N::zero(), N::zero(), None)
        }
    }

    /// Compute the natural logarithm of a quaternion.
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::Quaternion;
    /// let q = Quaternion::new(2.0, 5.0, 0.0, 0.0);
    /// assert_relative_eq!(q.ln(), Quaternion::new(1.683647, 1.190289, 0.0, 0.0), epsilon = 1.0e-6)
    /// ```
    #[inline]
    pub fn ln(&self) -> Self {
        let n = self.norm();
        let v = self.vector();
        let s = self.scalar();

        Self::from_parts(n.ln(), v.normalize() * (s / n).acos())
    }

    /// Compute the exponential of a quaternion.
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::Quaternion;
    /// let q = Quaternion::new(1.683647, 1.190289, 0.0, 0.0);
    /// assert_relative_eq!(q.exp(), Quaternion::new(2.0, 5.0, 0.0, 0.0), epsilon = 1.0e-5)
    /// ```
    #[inline]
    pub fn exp(&self) -> Self {
        self.exp_eps(N::default_epsilon())
    }

    /// Compute the exponential of a quaternion. Returns the identity if the vector part of this quaternion
    /// has a norm smaller than `eps`.
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::Quaternion;
    /// let q = Quaternion::new(1.683647, 1.190289, 0.0, 0.0);
    /// assert_relative_eq!(q.exp_eps(1.0e-6), Quaternion::new(2.0, 5.0, 0.0, 0.0), epsilon = 1.0e-5);
    ///
    /// // Singular case.
    /// let q = Quaternion::new(0.0000001, 0.0, 0.0, 0.0);
    /// assert_eq!(q.exp_eps(1.0e-6), Quaternion::identity());
    /// ```
    #[inline]
    pub fn exp_eps(&self, eps: N) -> Self {
        let v = self.vector();
        let nn = v.norm_squared();

        if nn <= eps * eps {
            Self::identity()
        } else {
            let w_exp = self.scalar().exp();
            let n = nn.sqrt();
            let nv = v * (w_exp * n.sin() / n);

            Self::from_parts(w_exp * n.cos(), nv)
        }
    }

    /// Raise the quaternion to a given floating power.
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::Quaternion;
    /// let q = Quaternion::new(1.0, 2.0, 3.0, 4.0);
    /// assert_relative_eq!(q.powf(1.5), Quaternion::new( -6.2576659, 4.1549037, 6.2323556, 8.3098075), epsilon = 1.0e-6);
    /// ```
    #[inline]
    pub fn powf(&self, n: N) -> Self {
        (self.ln() * n).exp()
    }

    /// Transforms this quaternion into its 4D vector form (Vector part, Scalar part).
    ///
    /// # Example
    /// ```
    /// # use nalgebra::{Quaternion, Vector4};
    /// let mut q = Quaternion::identity();
    /// *q.as_vector_mut() = Vector4::new(1.0, 2.0, 3.0, 4.0);
    /// assert!(q.i == 1.0 && q.j == 2.0 && q.k == 3.0 && q.w == 4.0);
    /// ```
    #[inline]
    pub fn as_vector_mut(&mut self) -> &mut Vector4<N> {
        &mut self.coords
    }

    /// The mutable vector part `(i, j, k)` of this quaternion.
    ///
    /// # Example
    /// ```
    /// # use nalgebra::{Quaternion, Vector4};
    /// let mut q = Quaternion::identity();
    /// {
    ///     let mut v = q.vector_mut();
    ///     v[0] = 2.0;
    ///     v[1] = 3.0;
    ///     v[2] = 4.0;
    /// }
    /// assert!(q.i == 2.0 && q.j == 3.0 && q.k == 4.0 && q.w == 1.0);
    /// ```
    #[inline]
    pub fn vector_mut(
        &mut self,
    ) -> MatrixSliceMut<N, U3, U1, RStride<N, U4, U1>, CStride<N, U4, U1>> {
        self.coords.fixed_rows_mut::<U3>(0)
    }

    /// Replaces this quaternion by its conjugate.
    ///
    /// # Example
    /// ```
    /// # use nalgebra::Quaternion;
    /// let mut q = Quaternion::new(1.0, 2.0, 3.0, 4.0);
    /// q.conjugate_mut();
    /// assert!(q.i == -2.0 && q.j == -3.0 && q.k == -4.0 && q.w == 1.0);
    /// ```
    #[inline]
    pub fn conjugate_mut(&mut self) {
        self.coords[0] = -self.coords[0];
        self.coords[1] = -self.coords[1];
        self.coords[2] = -self.coords[2];
    }

    /// Inverts this quaternion in-place if it is not zero.
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::Quaternion;
    /// let mut q = Quaternion::new(1.0, 2.0, 3.0, 4.0);
    ///
    /// assert!(q.try_inverse_mut());
    /// assert_relative_eq!(q * Quaternion::new(1.0, 2.0, 3.0, 4.0), Quaternion::identity());
    ///
    /// //Non-invertible case
    /// let mut q = Quaternion::new(0.0, 0.0, 0.0, 0.0);
    /// assert!(!q.try_inverse_mut());
    /// ```
    #[inline]
    pub fn try_inverse_mut(&mut self) -> bool {
        let norm_squared = self.norm_squared();

        if relative_eq!(&norm_squared, &N::zero()) {
            false
        } else {
            self.conjugate_mut();
            self.coords /= norm_squared;

            true
        }
    }

    /// Normalizes this quaternion.
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::Quaternion;
    /// let mut q = Quaternion::new(1.0, 2.0, 3.0, 4.0);
    /// q.normalize_mut();
    /// assert_relative_eq!(q.norm(), 1.0);
    /// ```
    #[inline]
    pub fn normalize_mut(&mut self) -> N {
        self.coords.normalize_mut()
    }
}

impl<N: Real + AbsDiffEq<Epsilon = N>> AbsDiffEq for Quaternion<N> {
    type Epsilon = N;

    #[inline]
    fn default_epsilon() -> Self::Epsilon {
        N::default_epsilon()
    }

    #[inline]
    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        self.as_vector().abs_diff_eq(other.as_vector(), epsilon) ||
        // Account for the double-covering of S², i.e. q = -q
        self.as_vector().iter().zip(other.as_vector().iter()).all(|(a, b)| a.abs_diff_eq(&-*b, epsilon))
    }
}

impl<N: Real + RelativeEq<Epsilon = N>> RelativeEq for Quaternion<N> {
    #[inline]
    fn default_max_relative() -> Self::Epsilon {
        N::default_max_relative()
    }

    #[inline]
    fn relative_eq(
        &self,
        other: &Self,
        epsilon: Self::Epsilon,
        max_relative: Self::Epsilon,
    ) -> bool
    {
        self.as_vector().relative_eq(other.as_vector(), epsilon, max_relative) ||
        // Account for the double-covering of S², i.e. q = -q
        self.as_vector().iter().zip(other.as_vector().iter()).all(|(a, b)| a.relative_eq(&-*b, epsilon, max_relative))
    }
}

impl<N: Real + UlpsEq<Epsilon = N>> UlpsEq for Quaternion<N> {
    #[inline]
    fn default_max_ulps() -> u32 {
        N::default_max_ulps()
    }

    #[inline]
    fn ulps_eq(&self, other: &Self, epsilon: Self::Epsilon, max_ulps: u32) -> bool {
        self.as_vector().ulps_eq(other.as_vector(), epsilon, max_ulps) ||
        // Account for the double-covering of S², i.e. q = -q.
        self.as_vector().iter().zip(other.as_vector().iter()).all(|(a, b)| a.ulps_eq(&-*b, epsilon, max_ulps))
    }
}

impl<N: Real + fmt::Display> fmt::Display for Quaternion<N> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "Quaternion {} − ({}, {}, {})",
            self[3], self[0], self[1], self[2]
        )
    }
}

/// A unit quaternions. May be used to represent a rotation.
pub type UnitQuaternion<N> = Unit<Quaternion<N>>;

impl<N: Real> UnitQuaternion<N> {
    /// Moves this unit quaternion into one that owns its data.
    #[inline]
    #[deprecated(
        note = "This method is unnecessary and will be removed in a future release. Use `.clone()` instead."
    )]
    pub fn into_owned(self) -> Self {
        self
    }

    /// Clones this unit quaternion into one that owns its data.
    #[inline]
    #[deprecated(
        note = "This method is unnecessary and will be removed in a future release. Use `.clone()` instead."
    )]
    pub fn clone_owned(&self) -> Self {
        *self
    }

    /// The rotation angle in [0; pi] of this unit quaternion.
    ///
    /// # Example
    /// ```
    /// # use nalgebra::{Unit, UnitQuaternion, Vector3};
    /// let axis = Unit::new_normalize(Vector3::new(1.0, 2.0, 3.0));
    /// let rot = UnitQuaternion::from_axis_angle(&axis, 1.78);
    /// assert_eq!(rot.angle(), 1.78);
    /// ```
    #[inline]
    pub fn angle(&self) -> N {
        let w = self.quaternion().scalar().abs();

        // Handle inaccuracies that make break `.acos`.
        if w >= N::one() {
            N::zero()
        } else {
            w.acos() * ::convert(2.0f64)
        }
    }

    /// The underlying quaternion.
    ///
    /// Same as `self.as_ref()`.
    ///
    /// # Example
    /// ```
    /// # use nalgebra::{UnitQuaternion, Quaternion};
    /// let axis = UnitQuaternion::identity();
    /// assert_eq!(*axis.quaternion(), Quaternion::new(1.0, 0.0, 0.0, 0.0));
    /// ```
    #[inline]
    pub fn quaternion(&self) -> &Quaternion<N> {
        self.as_ref()
    }

    /// Compute the conjugate of this unit quaternion.
    ///
    /// # Example
    /// ```
    /// # use nalgebra::{Unit, UnitQuaternion, Vector3};
    /// let axis = Unit::new_normalize(Vector3::new(1.0, 2.0, 3.0));
    /// let rot = UnitQuaternion::from_axis_angle(&axis, 1.78);
    /// let conj = rot.conjugate();
    /// assert_eq!(conj, UnitQuaternion::from_axis_angle(&-axis, 1.78));
    /// ```
    #[inline]
    pub fn conjugate(&self) -> Self {
        Self::new_unchecked(self.as_ref().conjugate())
    }

    /// Inverts this quaternion if it is not zero.
    ///
    /// # Example
    /// ```
    /// # use nalgebra::{Unit, UnitQuaternion, Vector3};
    /// let axis = Unit::new_normalize(Vector3::new(1.0, 2.0, 3.0));
    /// let rot = UnitQuaternion::from_axis_angle(&axis, 1.78);
    /// let inv = rot.inverse();
    /// assert_eq!(rot * inv, UnitQuaternion::identity());
    /// assert_eq!(inv * rot, UnitQuaternion::identity());
    /// ```
    #[inline]
    pub fn inverse(&self) -> Self {
        self.conjugate()
    }

    /// The rotation angle needed to make `self` and `other` coincide.
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{UnitQuaternion, Vector3};
    /// let rot1 = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), 1.0);
    /// let rot2 = UnitQuaternion::from_axis_angle(&Vector3::x_axis(), 0.1);
    /// assert_relative_eq!(rot1.angle_to(&rot2), 1.0045657, epsilon = 1.0e-6);
    /// ```
    #[inline]
    pub fn angle_to(&self, other: &Self) -> N {
        let delta = self.rotation_to(other);
        delta.angle()
    }

    /// The unit quaternion needed to make `self` and `other` coincide.
    ///
    /// The result is such that: `self.rotation_to(other) * self == other`.
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{UnitQuaternion, Vector3};
    /// let rot1 = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), 1.0);
    /// let rot2 = UnitQuaternion::from_axis_angle(&Vector3::x_axis(), 0.1);
    /// let rot_to = rot1.rotation_to(&rot2);
    /// assert_relative_eq!(rot_to * rot1, rot2, epsilon = 1.0e-6);
    /// ```
    #[inline]
    pub fn rotation_to(&self, other: &Self) -> Self{
        other / self
    }

    /// Linear interpolation between two unit quaternions.
    ///
    /// The result is not normalized.
    ///
    /// # Example
    /// ```
    /// # use nalgebra::{UnitQuaternion, Quaternion};
    /// let q1 = UnitQuaternion::new_normalize(Quaternion::new(1.0, 0.0, 0.0, 0.0));
    /// let q2 = UnitQuaternion::new_normalize(Quaternion::new(0.0, 1.0, 0.0, 0.0));
    /// assert_eq!(q1.lerp(&q2, 0.1), Quaternion::new(0.9, 0.1, 0.0, 0.0));
    /// ```
    #[inline]
    pub fn lerp(&self, other: &Self, t: N) -> Quaternion<N> {
        self.as_ref().lerp(other.as_ref(), t)
    }

    /// Normalized linear interpolation between two unit quaternions.
    ///
    /// This is the same as `self.lerp` except that the result is normalized.
    ///
    /// # Example
    /// ```
    /// # use nalgebra::{UnitQuaternion, Quaternion};
    /// let q1 = UnitQuaternion::new_normalize(Quaternion::new(1.0, 0.0, 0.0, 0.0));
    /// let q2 = UnitQuaternion::new_normalize(Quaternion::new(0.0, 1.0, 0.0, 0.0));
    /// assert_eq!(q1.nlerp(&q2, 0.1), UnitQuaternion::new_normalize(Quaternion::new(0.9, 0.1, 0.0, 0.0)));
    /// ```
    #[inline]
    pub fn nlerp(&self, other: &Self, t: N) -> Self {
        let mut res = self.lerp(other, t);
        let _ = res.normalize_mut();

        Self::new_unchecked(res)
    }

    /// Spherical linear interpolation between two unit quaternions.
    ///
    /// Panics if the angle between both quaternion is 180 degrees (in which case the interpolation
    /// is not well-defined). Use `.try_slerp` instead to avoid the panic.
    #[inline]
    pub fn slerp(&self, other: &Self, t: N) -> Self {
        Unit::new_unchecked(Quaternion::from(
            Unit::new_unchecked(self.coords)
                .slerp(&Unit::new_unchecked(other.coords), t)
                .into_inner(),
        ))
    }

    /// Computes the spherical linear interpolation between two unit quaternions or returns `None`
    /// if both quaternions are approximately 180 degrees apart (in which case the interpolation is
    /// not well-defined).
    ///
    /// # Arguments
    /// * `self`: the first quaternion to interpolate from.
    /// * `other`: the second quaternion to interpolate toward.
    /// * `t`: the interpolation parameter. Should be between 0 and 1.
    /// * `epsilon`: the value below which the sinus of the angle separating both quaternion
    /// must be to return `None`.
    #[inline]
    pub fn try_slerp(
        &self,
        other: &Self,
        t: N,
        epsilon: N,
    ) -> Option<Self>
    {
        Unit::new_unchecked(self.coords)
            .try_slerp(&Unit::new_unchecked(other.coords), t, epsilon)
            .map(|q| Unit::new_unchecked(Quaternion::from(q.into_inner())))
    }

    /// Compute the conjugate of this unit quaternion in-place.
    #[inline]
    pub fn conjugate_mut(&mut self) {
        self.as_mut_unchecked().conjugate_mut()
    }

    /// Inverts this quaternion if it is not zero.
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{UnitQuaternion, Vector3, Unit};
    /// let axisangle = Vector3::new(0.1, 0.2, 0.3);
    /// let mut rot = UnitQuaternion::new(axisangle);
    /// rot.inverse_mut();
    /// assert_relative_eq!(rot * UnitQuaternion::new(axisangle), UnitQuaternion::identity());
    /// assert_relative_eq!(UnitQuaternion::new(axisangle) * rot, UnitQuaternion::identity());
    /// ```
    #[inline]
    pub fn inverse_mut(&mut self) {
        self.as_mut_unchecked().conjugate_mut()
    }

    /// The rotation axis of this unit quaternion or `None` if the rotation is zero.
    ///
    /// # Example
    /// ```
    /// # use nalgebra::{UnitQuaternion, Vector3, Unit};
    /// let axis = Unit::new_normalize(Vector3::new(1.0, 2.0, 3.0));
    /// let angle = 1.2;
    /// let rot = UnitQuaternion::from_axis_angle(&axis, angle);
    /// assert_eq!(rot.axis(), Some(axis));
    ///
    /// // Case with a zero angle.
    /// let rot = UnitQuaternion::from_axis_angle(&axis, 0.0);
    /// assert!(rot.axis().is_none());
    /// ```
    #[inline]
    pub fn axis(&self) -> Option<Unit<Vector3<N>>> {
        let v = if self.quaternion().scalar() >= N::zero() {
            self.as_ref().vector().clone_owned()
        } else {
            -self.as_ref().vector()
        };

        Unit::try_new(v, N::zero())
    }

    /// The rotation axis of this unit quaternion multiplied by the rotation angle.
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{UnitQuaternion, Vector3, Unit};
    /// let axisangle = Vector3::new(0.1, 0.2, 0.3);
    /// let rot = UnitQuaternion::new(axisangle);
    /// assert_relative_eq!(rot.scaled_axis(), axisangle, epsilon = 1.0e-6);
    /// ```
    #[inline]
    pub fn scaled_axis(&self) -> Vector3<N> {
        if let Some(axis) = self.axis() {
            axis.into_inner() * self.angle()
        } else {
            Vector3::zero()
        }
    }

    /// The rotation axis and angle in ]0, pi] of this unit quaternion.
    ///
    /// Returns `None` if the angle is zero.
    ///
    /// # Example
    /// ```
    /// # use nalgebra::{UnitQuaternion, Vector3, Unit};
    /// let axis = Unit::new_normalize(Vector3::new(1.0, 2.0, 3.0));
    /// let angle = 1.2;
    /// let rot = UnitQuaternion::from_axis_angle(&axis, angle);
    /// assert_eq!(rot.axis_angle(), Some((axis, angle)));
    ///
    /// // Case with a zero angle.
    /// let rot = UnitQuaternion::from_axis_angle(&axis, 0.0);
    /// assert!(rot.axis_angle().is_none());
    /// ```
    #[inline]
    pub fn axis_angle(&self) -> Option<(Unit<Vector3<N>>, N)> {
        if let Some(axis) = self.axis() {
            Some((axis, self.angle()))
        } else {
            None
        }
    }

    /// Compute the exponential of a quaternion.
    ///
    /// Note that this function yields a `Quaternion<N>` because it loses the unit property.
    #[inline]
    pub fn exp(&self) -> Quaternion<N> {
        self.as_ref().exp()
    }

    /// Compute the natural logarithm of a quaternion.
    ///
    /// Note that this function yields a `Quaternion<N>` because it loses the unit property.
    /// The vector part of the return value corresponds to the axis-angle representation (divided
    /// by 2.0) of this unit quaternion.
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{Vector3, UnitQuaternion};
    /// let axisangle = Vector3::new(0.1, 0.2, 0.3);
    /// let q = UnitQuaternion::new(axisangle);
    /// assert_relative_eq!(q.ln().vector().into_owned(), axisangle, epsilon = 1.0e-6);
    /// ```
    #[inline]
    pub fn ln(&self) -> Quaternion<N> {
        if let Some(v) = self.axis() {
            Quaternion::from_parts(N::zero(), v.into_inner() * self.angle())
        } else {
            Quaternion::zero()
        }
    }

    /// Raise the quaternion to a given floating power.
    ///
    /// This returns the unit quaternion that identifies a rotation with axis `self.axis()` and
    /// angle `self.angle() × n`.
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{UnitQuaternion, Vector3, Unit};
    /// let axis = Unit::new_normalize(Vector3::new(1.0, 2.0, 3.0));
    /// let angle = 1.2;
    /// let rot = UnitQuaternion::from_axis_angle(&axis, angle);
    /// let pow = rot.powf(2.0);
    /// assert_relative_eq!(pow.axis().unwrap(), axis, epsilon = 1.0e-6);
    /// assert_eq!(pow.angle(), 2.4);
    /// ```
    #[inline]
    pub fn powf(&self, n: N) -> Self {
        if let Some(v) = self.axis() {
            Self::from_axis_angle(&v, self.angle() * n)
        } else {
            Self::identity()
        }
    }

    /// Builds a rotation matrix from this unit quaternion.
    ///
    /// # Example
    ///
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use std::f32;
    /// # use nalgebra::{UnitQuaternion, Vector3, Matrix3};
    /// let q = UnitQuaternion::from_axis_angle(&Vector3::z_axis(), f32::consts::FRAC_PI_6);
    /// let rot = q.to_rotation_matrix();
    /// let expected = Matrix3::new(0.8660254, -0.5,      0.0,
    ///                             0.5,       0.8660254, 0.0,
    ///                             0.0,       0.0,       1.0);
    ///
    /// assert_relative_eq!(*rot.matrix(), expected, epsilon = 1.0e-6);
    /// ```
    #[inline]
    pub fn to_rotation_matrix(&self) -> Rotation<N, U3> {
        let i = self.as_ref()[0];
        let j = self.as_ref()[1];
        let k = self.as_ref()[2];
        let w = self.as_ref()[3];

        let ww = w * w;
        let ii = i * i;
        let jj = j * j;
        let kk = k * k;
        let ij = i * j * ::convert(2.0f64);
        let wk = w * k * ::convert(2.0f64);
        let wj = w * j * ::convert(2.0f64);
        let ik = i * k * ::convert(2.0f64);
        let jk = j * k * ::convert(2.0f64);
        let wi = w * i * ::convert(2.0f64);

        Rotation::from_matrix_unchecked(Matrix3::new(
            ww + ii - jj - kk,
            ij - wk,
            wj + ik,
            wk + ij,
            ww - ii + jj - kk,
            jk - wi,
            ik - wj,
            wi + jk,
            ww - ii - jj + kk,
        ))
    }

    /// Converts this unit quaternion into its equivalent Euler angles.
    ///
    /// The angles are produced in the form (roll, pitch, yaw).
    #[inline]
    #[deprecated(note = "This is renamed to use `.euler_angles()`.")]
    pub fn to_euler_angles(&self) -> (N, N, N) {
        self.euler_angles()
    }

    /// Retrieves the euler angles corresponding to this unit quaternion.
    ///
    /// The angles are produced in the form (roll, pitch, yaw).
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::UnitQuaternion;
    /// let rot = UnitQuaternion::from_euler_angles(0.1, 0.2, 0.3);
    /// let euler = rot.euler_angles();
    /// assert_relative_eq!(euler.0, 0.1, epsilon = 1.0e-6);
    /// assert_relative_eq!(euler.1, 0.2, epsilon = 1.0e-6);
    /// assert_relative_eq!(euler.2, 0.3, epsilon = 1.0e-6);
    /// ```
    #[inline]
    pub fn euler_angles(&self) -> (N, N, N) {
        self.to_rotation_matrix().euler_angles()
    }

    /// Converts this unit quaternion into its equivalent homogeneous transformation matrix.
    ///
    /// # Example
    ///
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use std::f32;
    /// # use nalgebra::{UnitQuaternion, Vector3, Matrix4};
    /// let rot = UnitQuaternion::from_axis_angle(&Vector3::z_axis(), f32::consts::FRAC_PI_6);
    /// let expected = Matrix4::new(0.8660254, -0.5,      0.0, 0.0,
    ///                             0.5,       0.8660254, 0.0, 0.0,
    ///                             0.0,       0.0,       1.0, 0.0,
    ///                             0.0,       0.0,       0.0, 1.0);
    ///
    /// assert_relative_eq!(rot.to_homogeneous(), expected, epsilon = 1.0e-6);
    /// ```
    #[inline]
    pub fn to_homogeneous(&self) -> MatrixN<N, U4> {
        self.to_rotation_matrix().to_homogeneous()
    }
}

impl<N: Real + fmt::Display> fmt::Display for UnitQuaternion<N> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if let Some(axis) = self.axis() {
            let axis = axis.into_inner();
            write!(
                f,
                "UnitQuaternion angle: {} − axis: ({}, {}, {})",
                self.angle(),
                axis[0],
                axis[1],
                axis[2]
            )
        } else {
            write!(
                f,
                "UnitQuaternion angle: {} − axis: (undefined)",
                self.angle()
            )
        }
    }
}

impl<N: Real + AbsDiffEq<Epsilon = N>> AbsDiffEq for UnitQuaternion<N> {
    type Epsilon = N;

    #[inline]
    fn default_epsilon() -> Self::Epsilon {
        N::default_epsilon()
    }

    #[inline]
    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        self.as_ref().abs_diff_eq(other.as_ref(), epsilon)
    }
}

impl<N: Real + RelativeEq<Epsilon = N>> RelativeEq for UnitQuaternion<N> {
    #[inline]
    fn default_max_relative() -> Self::Epsilon {
        N::default_max_relative()
    }

    #[inline]
    fn relative_eq(
        &self,
        other: &Self,
        epsilon: Self::Epsilon,
        max_relative: Self::Epsilon,
    ) -> bool
    {
        self.as_ref()
            .relative_eq(other.as_ref(), epsilon, max_relative)
    }
}

impl<N: Real + UlpsEq<Epsilon = N>> UlpsEq for UnitQuaternion<N> {
    #[inline]
    fn default_max_ulps() -> u32 {
        N::default_max_ulps()
    }

    #[inline]
    fn ulps_eq(&self, other: &Self, epsilon: Self::Epsilon, max_ulps: u32) -> bool {
        self.as_ref().ulps_eq(other.as_ref(), epsilon, max_ulps)
    }
}
