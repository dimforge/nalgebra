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

use Quaternion;

/// A quaternion. See the type alias `UnitDualQuaternion = Unit<DualQuaternion>` for a quaternion
/// that may be used as a rotation.
#[repr(C)]
#[derive(Debug)]
pub struct DualQuaternion<N: Real> {
    /// This quaternion as a 4D vector of coordinates in the `[ x, y, z, w ]` storage order.
    pub coords: Vector4<N>,
    pub re: Quaternion<N>,
    pub du: Quaternion<N>,
}

// #[cfg(feature = "abomonation-serialize")]
// impl<N: Real> Abomonation for DualQuaternion<N>
// where Matrix4x2<N>: Abomonation
// {
//     unsafe fn entomb<W: Write>(&self, writer: &mut W) -> IOResult<()> {
//         self.coords.entomb(writer)
//     }

//     fn extent(&self) -> usize {
//         self.coords.extent()
//     }

//     unsafe fn exhume<'a, 'b>(&'a mut self, bytes: &'b mut [u8]) -> Option<&'b mut [u8]> {
//         self.coords.exhume(bytes)
//     }
// }

impl<N: Real + Eq> Eq for DualQuaternion<N> {}

impl<N: Real> PartialEq for DualQuaternion<N> {
    fn eq(&self, rhs: &Self) -> bool {
        self.re == rhs.re && self.du == rhs.du
    }
}

impl<N: Real + hash::Hash> hash::Hash for DualQuaternion<N> {
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        self.re.hash(state)
    }
}

impl<N: Real> Copy for DualQuaternion<N> {}

impl<N: Real> Clone for DualQuaternion<N> {
    #[inline]
    fn clone(&self) -> Self {
        // DualQuaternion::from(self.coords.clone())
        panic!("")
    }
}

// #[cfg(feature = "serde-serialize")]
// impl<N: Real> Serialize for DualQuaternion<N>
// where Owned<N, U4>: Serialize
// {
//     fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
//     where S: Serializer {
//         self.coords.serialize(serializer)
//     }
// }

// #[cfg(feature = "serde-serialize")]
// impl<'a, N: Real> Deserialize<'a> for DualQuaternion<N>
// where Owned<N, U4>: Deserialize<'a>
// {
//     fn deserialize<Des>(deserializer: Des) -> Result<Self, Des::Error>
//     where Des: Deserializer<'a> {
//         let coords = Vector4::<N>::deserialize(deserializer)?;

//         Ok(DualQuaternion::from(coords))
//     }
// }

// impl<N: Real> DualQuaternion<N> {
//     /// Moves this unit quaternion into one that owns its data.
//     #[inline]
//     #[deprecated(note = "This method is a no-op and will be removed in a future release.")]
//     pub fn into_owned(self) -> DualQuaternion<N> {
//         self
//     }

//     /// Clones this unit quaternion into one that owns its data.
//     #[inline]
//     #[deprecated(note = "This method is a no-op and will be removed in a future release.")]
//     pub fn clone_owned(&self) -> DualQuaternion<N> {
//         DualQuaternion::from(self.coords.clone_owned())
//     }

//     /// Normalizes this quaternion.
//     ///
//     /// # Example
//     /// ```
//     /// # #[macro_use] extern crate approx;
//     /// # use nalgebra::DualQuaternion;
//     /// let q = DualQuaternion::new(1.0, 2.0, 3.0, 4.0);
//     /// let q_normalized = q.normalize();
//     /// relative_eq!(q_normalized.norm(), 1.0);
//     /// ```
//     #[inline]
//     pub fn normalize(&self) -> DualQuaternion<N> {
//         DualQuaternion::from(self.coords.normalize())
//     }

//     /// The conjugate of this quaternion.
//     ///
//     /// # Example
//     /// ```
//     /// # use nalgebra::DualQuaternion;
//     /// let q = DualQuaternion::new(1.0, 2.0, 3.0, 4.0);
//     /// let conj = q.conjugate();
//     /// assert!(conj.i == -2.0 && conj.j == -3.0 && conj.k == -4.0 && conj.w == 1.0);
//     /// ```
//     #[inline]
//     pub fn conjugate(&self) -> DualQuaternion<N> {
//         let v = Vector4::new(
//             -self.coords[0],
//             -self.coords[1],
//             -self.coords[2],
//             self.coords[3],
//         );
//         DualQuaternion::from(v)
//     }

//     /// Inverts this quaternion if it is not zero.
//     ///
//     /// # Example
//     /// ```
//     /// # #[macro_use] extern crate approx;
//     /// # use nalgebra::DualQuaternion;
//     /// let q = DualQuaternion::new(1.0, 2.0, 3.0, 4.0);
//     /// let inv_q = q.try_inverse();
//     ///
//     /// assert!(inv_q.is_some());
//     /// assert_relative_eq!(inv_q.unwrap() * q, DualQuaternion::identity());
//     ///
//     /// //Non-invertible case
//     /// let q = DualQuaternion::new(0.0, 0.0, 0.0, 0.0);
//     /// let inv_q = q.try_inverse();
//     ///
//     /// assert!(inv_q.is_none());
//     /// ```
//     #[inline]
//     pub fn try_inverse(&self) -> Option<DualQuaternion<N>> {
//         let mut res = DualQuaternion::from(self.coords.clone_owned());

//         if res.try_inverse_mut() {
//             Some(res)
//         } else {
//             None
//         }
//     }

//     /// Linear interpolation between two quaternion.
//     ///
//     /// Computes `self * (1 - t) + other * t`.
//     ///
//     /// # Example
//     /// ```
//     /// # use nalgebra::DualQuaternion;
//     /// let q1 = DualQuaternion::new(1.0, 2.0, 3.0, 4.0);
//     /// let q2 = DualQuaternion::new(10.0, 20.0, 30.0, 40.0);
//     ///
//     /// assert_eq!(q1.lerp(&q2, 0.1), DualQuaternion::new(1.9, 3.8, 5.7, 7.6));
//     /// ```
//     #[inline]
//     pub fn lerp(&self, other: &DualQuaternion<N>, t: N) -> DualQuaternion<N> {
//         self * (N::one() - t) + other * t
//     }

//     /// The vector part `(i, j, k)` of this quaternion.
//     ///
//     /// # Example
//     /// ```
//     /// # use nalgebra::DualQuaternion;
//     /// let q = DualQuaternion::new(1.0, 2.0, 3.0, 4.0);
//     /// assert_eq!(q.vector()[0], 2.0);
//     /// assert_eq!(q.vector()[1], 3.0);
//     /// assert_eq!(q.vector()[2], 4.0);
//     /// ```
//     #[inline]
//     pub fn vector(&self) -> MatrixSlice<N, U3, U1, RStride<N, U4, U1>, CStride<N, U4, U1>> {
//         self.coords.fixed_rows::<U3>(0)
//     }

//     /// The scalar part `w` of this quaternion.
//     ///
//     /// # Example
//     /// ```
//     /// # use nalgebra::DualQuaternion;
//     /// let q = DualQuaternion::new(1.0, 2.0, 3.0, 4.0);
//     /// assert_eq!(q.scalar(), 1.0);
//     /// ```
//     #[inline]
//     pub fn scalar(&self) -> N {
//         self.coords[3]
//     }

//     /// Reinterprets this quaternion as a 4D vector.
//     ///
//     /// # Example
//     /// ```
//     /// # use nalgebra::{Vector4, DualQuaternion};
//     /// let q = DualQuaternion::new(1.0, 2.0, 3.0, 4.0);
//     /// // Recall that the quaternion is stored internally as (i, j, k, w)
//     /// // while the ::new constructor takes the arguments as (w, i, j, k).
//     /// assert_eq!(*q.as_vector(), Vector4::new(2.0, 3.0, 4.0, 1.0));
//     /// ```
//     #[inline]
//     pub fn as_vector(&self) -> &Vector4<N> {
//         &self.coords
//     }

//     /// The norm of this quaternion.
//     ///
//     /// # Example
//     /// ```
//     /// # #[macro_use] extern crate approx;
//     /// # use nalgebra::DualQuaternion;
//     /// let q = DualQuaternion::new(1.0, 2.0, 3.0, 4.0);
//     /// assert_relative_eq!(q.norm(), 5.47722557, epsilon = 1.0e-6);
//     /// ```
//     #[inline]
//     pub fn norm(&self) -> N {
//         self.coords.norm()
//     }

//     /// A synonym for the norm of this quaternion.
//     ///
//     /// Aka the length.
//     /// This is the same as `.norm()`
//     ///
//     /// # Example
//     /// ```
//     /// # #[macro_use] extern crate approx;
//     /// # use nalgebra::DualQuaternion;
//     /// let q = DualQuaternion::new(1.0, 2.0, 3.0, 4.0);
//     /// assert_relative_eq!(q.magnitude(), 5.47722557, epsilon = 1.0e-6);
//     /// ```
//     #[inline]
//     pub fn magnitude(&self) -> N {
//         self.norm()
//     }

//     /// The squared norm of this quaternion.
//     ///
//     /// # Example
//     /// ```
//     /// # use nalgebra::DualQuaternion;
//     /// let q = DualQuaternion::new(1.0, 2.0, 3.0, 4.0);
//     /// assert_eq!(q.magnitude_squared(), 30.0);
//     /// ```
//     #[inline]
//     pub fn norm_squared(&self) -> N {
//         self.coords.norm_squared()
//     }

//     /// A synonym for the squared norm of this quaternion.
//     ///
//     /// Aka the squared length.
//     /// This is the same as `.norm_squared()`
//     ///
//     /// # Example
//     /// ```
//     /// # use nalgebra::DualQuaternion;
//     /// let q = DualQuaternion::new(1.0, 2.0, 3.0, 4.0);
//     /// assert_eq!(q.magnitude_squared(), 30.0);
//     /// ```
//     #[inline]
//     pub fn magnitude_squared(&self) -> N {
//         self.norm_squared()
//     }

//     /// The dot product of two quaternions.
//     ///
//     /// # Example
//     /// ```
//     /// # use nalgebra::DualQuaternion;
//     /// let q1 = DualQuaternion::new(1.0, 2.0, 3.0, 4.0);
//     /// let q2 = DualQuaternion::new(5.0, 6.0, 7.0, 8.0);
//     /// assert_eq!(q1.dot(&q2), 70.0);
//     /// ```
//     #[inline]
//     pub fn dot(&self, rhs: &Self) -> N {
//         self.coords.dot(&rhs.coords)
//     }

//     /// The polar decomposition of this quaternion.
//     ///
//     /// Returns, from left to right: the quaternion norm, the half rotation angle, the rotation
//     /// axis. If the rotation angle is zero, the rotation axis is set to `None`.
//     ///
//     /// # Example
//     /// ```
//     /// # use std::f32;
//     /// # use nalgebra::{Vector3, DualQuaternion};
//     /// let q = DualQuaternion::new(0.0, 5.0, 0.0, 0.0);
//     /// let (norm, half_ang, axis) = q.polar_decomposition();
//     /// assert_eq!(norm, 5.0);
//     /// assert_eq!(half_ang, f32::consts::FRAC_PI_2);
//     /// assert_eq!(axis, Some(Vector3::x_axis()));
//     /// ```
//     pub fn polar_decomposition(&self) -> (N, N, Option<Unit<Vector3<N>>>) {
//         if let Some((q, n)) = Unit::try_new_and_get(*self, N::zero()) {
//             if let Some(axis) = Unit::try_new(self.vector().clone_owned(), N::zero()) {
//                 let angle = q.angle() / ::convert(2.0f64);

//                 (n, angle, Some(axis))
//             } else {
//                 (n, N::zero(), None)
//             }
//         } else {
//             (N::zero(), N::zero(), None)
//         }
//     }

//     /// Compute the natural logarithm of a quaternion.
//     ///
//     /// # Example
//     /// ```
//     /// # #[macro_use] extern crate approx;
//     /// # use nalgebra::DualQuaternion;
//     /// let q = DualQuaternion::new(2.0, 5.0, 0.0, 0.0);
//     /// assert_relative_eq!(q.ln(), DualQuaternion::new(1.683647, 1.190289, 0.0, 0.0), epsilon = 1.0e-6)
//     /// ```
//     #[inline]
//     pub fn ln(&self) -> DualQuaternion<N> {
//         let n = self.norm();
//         let v = self.vector();
//         let s = self.scalar();

//         DualQuaternion::from_parts(n.ln(), v.normalize() * (s / n).acos())
//     }

//     /// Compute the exponential of a quaternion.
//     ///
//     /// # Example
//     /// ```
//     /// # #[macro_use] extern crate approx;
//     /// # use nalgebra::DualQuaternion;
//     /// let q = DualQuaternion::new(1.683647, 1.190289, 0.0, 0.0);
//     /// assert_relative_eq!(q.exp(), DualQuaternion::new(2.0, 5.0, 0.0, 0.0), epsilon = 1.0e-5)
//     /// ```
//     #[inline]
//     pub fn exp(&self) -> DualQuaternion<N> {
//         self.exp_eps(N::default_epsilon())
//     }

//     /// Compute the exponential of a quaternion. Returns the identity if the vector part of this quaternion
//     /// has a norm smaller than `eps`.
//     ///
//     /// # Example
//     /// ```
//     /// # #[macro_use] extern crate approx;
//     /// # use nalgebra::DualQuaternion;
//     /// let q = DualQuaternion::new(1.683647, 1.190289, 0.0, 0.0);
//     /// assert_relative_eq!(q.exp_eps(1.0e-6), DualQuaternion::new(2.0, 5.0, 0.0, 0.0), epsilon = 1.0e-5);
//     ///
//     /// // Singular case.
//     /// let q = DualQuaternion::new(0.0000001, 0.0, 0.0, 0.0);
//     /// assert_eq!(q.exp_eps(1.0e-6), DualQuaternion::identity());
//     /// ```
//     #[inline]
//     pub fn exp_eps(&self, eps: N) -> DualQuaternion<N> {
//         let v = self.vector();
//         let nn = v.norm_squared();

//         if nn <= eps * eps {
//             DualQuaternion::identity()
//         } else {
//             let w_exp = self.scalar().exp();
//             let n = nn.sqrt();
//             let nv = v * (w_exp * n.sin() / n);

//             DualQuaternion::from_parts(w_exp * n.cos(), nv)
//         }
//     }

//     /// Raise the quaternion to a given floating power.
//     ///
//     /// # Example
//     /// ```
//     /// # #[macro_use] extern crate approx;
//     /// # use nalgebra::DualQuaternion;
//     /// let q = DualQuaternion::new(1.0, 2.0, 3.0, 4.0);
//     /// assert_relative_eq!(q.powf(1.5), DualQuaternion::new( -6.2576659, 4.1549037, 6.2323556, 8.3098075), epsilon = 1.0e-6);
//     /// ```
//     #[inline]
//     pub fn powf(&self, n: N) -> DualQuaternion<N> {
//         (self.ln() * n).exp()
//     }

//     /// Transforms this quaternion into its 4D vector form (Vector part, Scalar part).
//     ///
//     /// # Example
//     /// ```
//     /// # use nalgebra::{DualQuaternion, Vector4};
//     /// let mut q = DualQuaternion::identity();
//     /// *q.as_vector_mut() = Vector4::new(1.0, 2.0, 3.0, 4.0);
//     /// assert!(q.i == 1.0 && q.j == 2.0 && q.k == 3.0 && q.w == 4.0);
//     /// ```
//     #[inline]
//     pub fn as_vector_mut(&mut self) -> &mut Vector4<N> {
//         &mut self.coords
//     }

//     /// The mutable vector part `(i, j, k)` of this quaternion.
//     ///
//     /// # Example
//     /// ```
//     /// # use nalgebra::{DualQuaternion, Vector4};
//     /// let mut q = DualQuaternion::identity();
//     /// {
//     ///     let mut v = q.vector_mut();
//     ///     v[0] = 2.0;
//     ///     v[1] = 3.0;
//     ///     v[2] = 4.0;
//     /// }
//     /// assert!(q.i == 2.0 && q.j == 3.0 && q.k == 4.0 && q.w == 1.0);
//     /// ```
//     #[inline]
//     pub fn vector_mut(
//         &mut self,
//     ) -> MatrixSliceMut<N, U3, U1, RStride<N, U4, U1>, CStride<N, U4, U1>> {
//         self.coords.fixed_rows_mut::<U3>(0)
//     }

//     /// Replaces this quaternion by its conjugate.
//     ///
//     /// # Example
//     /// ```
//     /// # use nalgebra::DualQuaternion;
//     /// let mut q = DualQuaternion::new(1.0, 2.0, 3.0, 4.0);
//     /// q.conjugate_mut();
//     /// assert!(q.i == -2.0 && q.j == -3.0 && q.k == -4.0 && q.w == 1.0);
//     /// ```
//     #[inline]
//     pub fn conjugate_mut(&mut self) {
//         self.coords[0] = -self.coords[0];
//         self.coords[1] = -self.coords[1];
//         self.coords[2] = -self.coords[2];
//     }

//     /// Inverts this quaternion in-place if it is not zero.
//     ///
//     /// # Example
//     /// ```
//     /// # #[macro_use] extern crate approx;
//     /// # use nalgebra::DualQuaternion;
//     /// let mut q = DualQuaternion::new(1.0, 2.0, 3.0, 4.0);
//     ///
//     /// assert!(q.try_inverse_mut());
//     /// assert_relative_eq!(q * DualQuaternion::new(1.0, 2.0, 3.0, 4.0), DualQuaternion::identity());
//     ///
//     /// //Non-invertible case
//     /// let mut q = DualQuaternion::new(0.0, 0.0, 0.0, 0.0);
//     /// assert!(!q.try_inverse_mut());
//     /// ```
//     #[inline]
//     pub fn try_inverse_mut(&mut self) -> bool {
//         let norm_squared = self.norm_squared();

//         if relative_eq!(&norm_squared, &N::zero()) {
//             false
//         } else {
//             self.conjugate_mut();
//             self.coords /= norm_squared;

//             true
//         }
//     }

//     /// Normalizes this quaternion.
//     ///
//     /// # Example
//     /// ```
//     /// # #[macro_use] extern crate approx;
//     /// # use nalgebra::DualQuaternion;
//     /// let mut q = DualQuaternion::new(1.0, 2.0, 3.0, 4.0);
//     /// q.normalize_mut();
//     /// assert_relative_eq!(q.norm(), 1.0);
//     /// ```
//     #[inline]
//     pub fn normalize_mut(&mut self) -> N {
//         self.coords.normalize_mut()
//     }
// }

// impl<N: Real + AbsDiffEq<Epsilon = N>> AbsDiffEq for DualQuaternion<N> {
//     type Epsilon = N;

//     #[inline]
//     fn default_epsilon() -> Self::Epsilon {
//         N::default_epsilon()
//     }

//     #[inline]
//     fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
//         self.as_vector().abs_diff_eq(other.as_vector(), epsilon) ||
//         // Account for the double-covering of S², i.e. q = -q
//        self.as_vector().iter().zip(other.as_vector().iter()).all(|(a, b)| a.abs_diff_eq(&-*b, epsilon))
//     }
// }

// impl<N: Real + RelativeEq<Epsilon = N>> RelativeEq for DualQuaternion<N> {
//     #[inline]
//     fn default_max_relative() -> Self::Epsilon {
//         N::default_max_relative()
//     }

//     #[inline]
//     fn relative_eq(
//         &self,
//         other: &Self,
//         epsilon: Self::Epsilon,
//         max_relative: Self::Epsilon,
//     ) -> bool
//     {
//         self.as_vector().relative_eq(other.as_vector(), epsilon, max_relative) ||
//         // Account for the double-covering of S², i.e. q = -q
//        self.as_vector().iter().zip(other.as_vector().iter()).all(|(a, b)| a.relative_eq(&-*b, epsilon, max_relative))
//     }
// }

// impl<N: Real + UlpsEq<Epsilon = N>> UlpsEq for DualQuaternion<N> {
//     #[inline]
//     fn default_max_ulps() -> u32 {
//         N::default_max_ulps()
//     }

//     #[inline]
//     fn ulps_eq(&self, other: &Self, epsilon: Self::Epsilon, max_ulps: u32) -> bool {
//         self.as_vector().ulps_eq(other.as_vector(), epsilon, max_ulps) ||
//         // Account for the double-covering of S², i.e. q = -q.
//        self.as_vector().iter().zip(other.as_vector().iter()).all(|(a, b)| a.ulps_eq(&-*b, epsilon, max_ulps))
//     }
// }

// impl<N: Real + fmt::Display> fmt::Display for DualQuaternion<N> {
//     fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
//         write!(
//             f,
//             "DualQuaternion {} − ({}, {}, {})",
//             self[3], self[0], self[1], self[2]
//         )
//     }
// }

// /// A unit quaternions. May be used to represent a rotation.
pub type UnitDualQuaternion<N> = Unit<DualQuaternion<N>>;

// impl<N: Real> UnitDualQuaternion<N> {
//     /// Moves this unit quaternion into one that owns its data.
//     #[inline]
//     #[deprecated(
//         note = "This method is unnecessary and will be removed in a future release. Use `.clone()` instead."
//     )]
//     pub fn into_owned(self) -> UnitDualQuaternion<N> {
//         self
//     }

//     /// Clones this unit quaternion into one that owns its data.
//     #[inline]
//     #[deprecated(
//         note = "This method is unnecessary and will be removed in a future release. Use `.clone()` instead."
//     )]
//     pub fn clone_owned(&self) -> UnitDualQuaternion<N> {
//         *self
//     }

//     /// The rotation angle in [0; pi] of this unit quaternion.
//     ///
//     /// # Example
//     /// ```
//     /// # use nalgebra::{Unit, UnitDualQuaternion, Vector3};
//     /// let axis = Unit::new_normalize(Vector3::new(1.0, 2.0, 3.0));
//     /// let rot = UnitDualQuaternion::from_axis_angle(&axis, 1.78);
//     /// assert_eq!(rot.angle(), 1.78);
//     /// ```
//     #[inline]
//     pub fn angle(&self) -> N {
//         let w = self.quaternion().scalar().abs();

//         // Handle inaccuracies that make break `.acos`.
//         if w >= N::one() {
//             N::zero()
//         } else {
//             w.acos() * ::convert(2.0f64)
//         }
//     }

//     /// The underlying quaternion.
//     ///
//     /// Same as `self.as_ref()`.
//     ///
//     /// # Example
//     /// ```
//     /// # use nalgebra::{UnitDualQuaternion, DualQuaternion};
//     /// let axis = UnitDualQuaternion::identity();
//     /// assert_eq!(*axis.quaternion(), DualQuaternion::new(1.0, 0.0, 0.0, 0.0));
//     /// ```
//     #[inline]
//     pub fn quaternion(&self) -> &DualQuaternion<N> {
//         self.as_ref()
//     }

//     /// Compute the conjugate of this unit quaternion.
//     ///
//     /// # Example
//     /// ```
//     /// # use nalgebra::{Unit, UnitDualQuaternion, Vector3};
//     /// let axis = Unit::new_normalize(Vector3::new(1.0, 2.0, 3.0));
//     /// let rot = UnitDualQuaternion::from_axis_angle(&axis, 1.78);
//     /// let conj = rot.conjugate();
//     /// assert_eq!(conj, UnitDualQuaternion::from_axis_angle(&-axis, 1.78));
//     /// ```
//     #[inline]
//     pub fn conjugate(&self) -> UnitDualQuaternion<N> {
//         UnitDualQuaternion::new_unchecked(self.as_ref().conjugate())
//     }

//     /// Inverts this quaternion if it is not zero.
//     ///
//     /// # Example
//     /// ```
//     /// # use nalgebra::{Unit, UnitDualQuaternion, Vector3};
//     /// let axis = Unit::new_normalize(Vector3::new(1.0, 2.0, 3.0));
//     /// let rot = UnitDualQuaternion::from_axis_angle(&axis, 1.78);
//     /// let inv = rot.inverse();
//     /// assert_eq!(rot * inv, UnitDualQuaternion::identity());
//     /// assert_eq!(inv * rot, UnitDualQuaternion::identity());
//     /// ```
//     #[inline]
//     pub fn inverse(&self) -> UnitDualQuaternion<N> {
//         self.conjugate()
//     }

//     /// The rotation angle needed to make `self` and `other` coincide.
//     ///
//     /// # Example
//     /// ```
//     /// # #[macro_use] extern crate approx;
//     /// # use nalgebra::{UnitDualQuaternion, Vector3};
//     /// let rot1 = UnitDualQuaternion::from_axis_angle(&Vector3::y_axis(), 1.0);
//     /// let rot2 = UnitDualQuaternion::from_axis_angle(&Vector3::x_axis(), 0.1);
//     /// assert_relative_eq!(rot1.angle_to(&rot2), 1.0045657, epsilon = 1.0e-6);
//     /// ```
//     #[inline]
//     pub fn angle_to(&self, other: &UnitDualQuaternion<N>) -> N {
//         let delta = self.rotation_to(other);
//         delta.angle()
//     }

//     /// The unit quaternion needed to make `self` and `other` coincide.
//     ///
//     /// The result is such that: `self.rotation_to(other) * self == other`.
//     ///
//     /// # Example
//     /// ```
//     /// # #[macro_use] extern crate approx;
//     /// # use nalgebra::{UnitDualQuaternion, Vector3};
//     /// let rot1 = UnitDualQuaternion::from_axis_angle(&Vector3::y_axis(), 1.0);
//     /// let rot2 = UnitDualQuaternion::from_axis_angle(&Vector3::x_axis(), 0.1);
//     /// let rot_to = rot1.rotation_to(&rot2);
//     /// assert_relative_eq!(rot_to * rot1, rot2, epsilon = 1.0e-6);
//     /// ```
//     #[inline]
//     pub fn rotation_to(&self, other: &UnitDualQuaternion<N>) -> UnitDualQuaternion<N> {
//         other / self
//     }

//     /// Linear interpolation between two unit quaternions.
//     ///
//     /// The result is not normalized.
//     ///
//     /// # Example
//     /// ```
//     /// # use nalgebra::{UnitDualQuaternion, DualQuaternion};
//     /// let q1 = UnitDualQuaternion::new_normalize(DualQuaternion::new(1.0, 0.0, 0.0, 0.0));
//     /// let q2 = UnitDualQuaternion::new_normalize(DualQuaternion::new(0.0, 1.0, 0.0, 0.0));
//     /// assert_eq!(q1.lerp(&q2, 0.1), DualQuaternion::new(0.9, 0.1, 0.0, 0.0));
//     /// ```
//     #[inline]
//     pub fn lerp(&self, other: &UnitDualQuaternion<N>, t: N) -> DualQuaternion<N> {
//         self.as_ref().lerp(other.as_ref(), t)
//     }

//     /// Normalized linear interpolation between two unit quaternions.
//     ///
//     /// This is the same as `self.lerp` except that the result is normalized.
//     ///
//     /// # Example
//     /// ```
//     /// # use nalgebra::{UnitDualQuaternion, DualQuaternion};
//     /// let q1 = UnitDualQuaternion::new_normalize(DualQuaternion::new(1.0, 0.0, 0.0, 0.0));
//     /// let q2 = UnitDualQuaternion::new_normalize(DualQuaternion::new(0.0, 1.0, 0.0, 0.0));
//     /// assert_eq!(q1.nlerp(&q2, 0.1), UnitDualQuaternion::new_normalize(DualQuaternion::new(0.9, 0.1, 0.0, 0.0)));
//     /// ```
//     #[inline]
//     pub fn nlerp(&self, other: &UnitDualQuaternion<N>, t: N) -> UnitDualQuaternion<N> {
//         let mut res = self.lerp(other, t);
//         let _ = res.normalize_mut();

//         UnitDualQuaternion::new_unchecked(res)
//     }

//     /// Spherical linear interpolation between two unit quaternions.
//     ///
//     /// Panics if the angle between both quaternion is 180 degrees (in which case the interpolation
//     /// is not well-defined). Use `.try_slerp` instead to avoid the panic.
//     #[inline]
//     pub fn slerp(&self, other: &UnitDualQuaternion<N>, t: N) -> UnitDualQuaternion<N> {
//         Unit::new_unchecked(DualQuaternion::from(
//             Unit::new_unchecked(self.coords)
//                 .slerp(&Unit::new_unchecked(other.coords), t)
//                 .into_inner(),
//         ))
//     }

//     /// Computes the spherical linear interpolation between two unit quaternions or returns `None`
//     /// if both quaternions are approximately 180 degrees apart (in which case the interpolation is
//     /// not well-defined).
//     ///
//     /// # Arguments
//     /// * `self`: the first quaternion to interpolate from.
//     /// * `other`: the second quaternion to interpolate toward.
//     /// * `t`: the interpolation parameter. Should be between 0 and 1.
//     /// * `epsilon`: the value below which the sinus of the angle separating both quaternion
//     /// must be to return `None`.
//     #[inline]
//     pub fn try_slerp(
//         &self,
//         other: &UnitDualQuaternion<N>,
//         t: N,
//         epsilon: N,
//     ) -> Option<UnitDualQuaternion<N>>
//     {
//         Unit::new_unchecked(self.coords)
//             .try_slerp(&Unit::new_unchecked(other.coords), t, epsilon)
//             .map(|q| Unit::new_unchecked(DualQuaternion::from(q.into_inner())))
//     }

//     /// Compute the conjugate of this unit quaternion in-place.
//     #[inline]
//     pub fn conjugate_mut(&mut self) {
//         self.as_mut_unchecked().conjugate_mut()
//     }

//     /// Inverts this quaternion if it is not zero.
//     ///
//     /// # Example
//     /// ```
//     /// # #[macro_use] extern crate approx;
//     /// # use nalgebra::{UnitDualQuaternion, Vector3, Unit};
//     /// let axisangle = Vector3::new(0.1, 0.2, 0.3);
//     /// let mut rot = UnitDualQuaternion::new(axisangle);
//     /// rot.inverse_mut();
//     /// assert_relative_eq!(rot * UnitDualQuaternion::new(axisangle), UnitDualQuaternion::identity());
//     /// assert_relative_eq!(UnitDualQuaternion::new(axisangle) * rot, UnitDualQuaternion::identity());
//     /// ```
//     #[inline]
//     pub fn inverse_mut(&mut self) {
//         self.as_mut_unchecked().conjugate_mut()
//     }

//     /// The rotation axis of this unit quaternion or `None` if the rotation is zero.
//     ///
//     /// # Example
//     /// ```
//     /// # use nalgebra::{UnitDualQuaternion, Vector3, Unit};
//     /// let axis = Unit::new_normalize(Vector3::new(1.0, 2.0, 3.0));
//     /// let angle = 1.2;
//     /// let rot = UnitDualQuaternion::from_axis_angle(&axis, angle);
//     /// assert_eq!(rot.axis(), Some(axis));
//     ///
//     /// // Case with a zero angle.
//     /// let rot = UnitDualQuaternion::from_axis_angle(&axis, 0.0);
//     /// assert!(rot.axis().is_none());
//     /// ```
//     #[inline]
//     pub fn axis(&self) -> Option<Unit<Vector3<N>>> {
//         let v = if self.quaternion().scalar() >= N::zero() {
//             self.as_ref().vector().clone_owned()
//         } else {
//             -self.as_ref().vector()
//         };

//         Unit::try_new(v, N::zero())
//     }

//     /// The rotation axis of this unit quaternion multiplied by the rotation angle.
//     ///
//     /// # Example
//     /// ```
//     /// # #[macro_use] extern crate approx;
//     /// # use nalgebra::{UnitDualQuaternion, Vector3, Unit};
//     /// let axisangle = Vector3::new(0.1, 0.2, 0.3);
//     /// let rot = UnitDualQuaternion::new(axisangle);
//     /// assert_relative_eq!(rot.scaled_axis(), axisangle, epsilon = 1.0e-6);
//     /// ```
//     #[inline]
//     pub fn scaled_axis(&self) -> Vector3<N> {
//         if let Some(axis) = self.axis() {
//             axis.into_inner() * self.angle()
//         } else {
//             Vector3::zero()
//         }
//     }

//     /// The rotation axis and angle in ]0, pi] of this unit quaternion.
//     ///
//     /// Returns `None` if the angle is zero.
//     ///
//     /// # Example
//     /// ```
//     /// # use nalgebra::{UnitDualQuaternion, Vector3, Unit};
//     /// let axis = Unit::new_normalize(Vector3::new(1.0, 2.0, 3.0));
//     /// let angle = 1.2;
//     /// let rot = UnitDualQuaternion::from_axis_angle(&axis, angle);
//     /// assert_eq!(rot.axis_angle(), Some((axis, angle)));
//     ///
//     /// // Case with a zero angle.
//     /// let rot = UnitDualQuaternion::from_axis_angle(&axis, 0.0);
//     /// assert!(rot.axis_angle().is_none());
//     /// ```
//     #[inline]
//     pub fn axis_angle(&self) -> Option<(Unit<Vector3<N>>, N)> {
//         if let Some(axis) = self.axis() {
//             Some((axis, self.angle()))
//         } else {
//             None
//         }
//     }

//     /// Compute the exponential of a quaternion.
//     ///
//     /// Note that this function yields a `DualQuaternion<N>` because it looses the unit property.
//     #[inline]
//     pub fn exp(&self) -> DualQuaternion<N> {
//         self.as_ref().exp()
//     }

//     /// Compute the natural logarithm of a quaternion.
//     ///
//     /// Note that this function yields a `DualQuaternion<N>` because it looses the unit property.
//     /// The vector part of the return value corresponds to the axis-angle representation (divided
//     /// by 2.0) of this unit quaternion.
//     ///
//     /// # Example
//     /// ```
//     /// # #[macro_use] extern crate approx;
//     /// # use nalgebra::{Vector3, UnitDualQuaternion};
//     /// let axisangle = Vector3::new(0.1, 0.2, 0.3);
//     /// let q = UnitDualQuaternion::new(axisangle);
//     /// assert_relative_eq!(q.ln().vector().into_owned(), axisangle, epsilon = 1.0e-6);
//     /// ```
//     #[inline]
//     pub fn ln(&self) -> DualQuaternion<N> {
//         if let Some(v) = self.axis() {
//             DualQuaternion::from_parts(N::zero(), v.into_inner() * self.angle())
//         } else {
//             DualQuaternion::zero()
//         }
//     }

//     /// Raise the quaternion to a given floating power.
//     ///
//     /// This returns the unit quaternion that identifies a rotation with axis `self.axis()` and
//     /// angle `self.angle() × n`.
//     ///
//     /// # Example
//     /// ```
//     /// # #[macro_use] extern crate approx;
//     /// # use nalgebra::{UnitDualQuaternion, Vector3, Unit};
//     /// let axis = Unit::new_normalize(Vector3::new(1.0, 2.0, 3.0));
//     /// let angle = 1.2;
//     /// let rot = UnitDualQuaternion::from_axis_angle(&axis, angle);
//     /// let pow = rot.powf(2.0);
//     /// assert_relative_eq!(pow.axis().unwrap(), axis, epsilon = 1.0e-6);
//     /// assert_eq!(pow.angle(), 2.4);
//     /// ```
//     #[inline]
//     pub fn powf(&self, n: N) -> UnitDualQuaternion<N> {
//         if let Some(v) = self.axis() {
//             UnitDualQuaternion::from_axis_angle(&v, self.angle() * n)
//         } else {
//             UnitDualQuaternion::identity()
//         }
//     }

//     /// Builds a rotation matrix from this unit quaternion.
//     ///
//     /// # Example
//     ///
//     /// ```
//     /// # #[macro_use] extern crate approx;
//     /// # use std::f32;
//     /// # use nalgebra::{UnitDualQuaternion, Vector3, Matrix3};
//     /// let q = UnitDualQuaternion::from_axis_angle(&Vector3::z_axis(), f32::consts::FRAC_PI_6);
//     /// let rot = q.to_rotation_matrix();
//     /// let expected = Matrix3::new(0.8660254, -0.5,      0.0,
//     ///                             0.5,       0.8660254, 0.0,
//     ///                             0.0,       0.0,       1.0);
//     ///
//     /// assert_relative_eq!(*rot.matrix(), expected, epsilon = 1.0e-6);
//     /// ```
//     #[inline]
//     pub fn to_rotation_matrix(&self) -> Rotation<N, U3> {
//         let i = self.as_ref()[0];
//         let j = self.as_ref()[1];
//         let k = self.as_ref()[2];
//         let w = self.as_ref()[3];

//         let ww = w * w;
//         let ii = i * i;
//         let jj = j * j;
//         let kk = k * k;
//         let ij = i * j * ::convert(2.0f64);
//         let wk = w * k * ::convert(2.0f64);
//         let wj = w * j * ::convert(2.0f64);
//         let ik = i * k * ::convert(2.0f64);
//         let jk = j * k * ::convert(2.0f64);
//         let wi = w * i * ::convert(2.0f64);

//         Rotation::from_matrix_unchecked(Matrix3::new(
//             ww + ii - jj - kk,
//             ij - wk,
//             wj + ik,
//             wk + ij,
//             ww - ii + jj - kk,
//             jk - wi,
//             ik - wj,
//             wi + jk,
//             ww - ii - jj + kk,
//         ))
//     }

//     /// Converts this unit quaternion into its equivalent Euler angles.
//     ///
//     /// The angles are produced in the form (roll, pitch, yaw).
//     #[inline]
//     #[deprecated(note = "This is renamed to use `.euler_angles()`.")]
//     pub fn to_euler_angles(&self) -> (N, N, N) {
//         self.euler_angles()
//     }

//     /// Retrieves the euler angles corresponding to this unit quaternion.
//     ///
//     /// The angles are produced in the form (roll, pitch, yaw).
//     ///
//     /// # Example
//     /// ```
//     /// # #[macro_use] extern crate approx;
//     /// # use nalgebra::UnitDualQuaternion;
//     /// let rot = UnitDualQuaternion::from_euler_angles(0.1, 0.2, 0.3);
//     /// let euler = rot.euler_angles();
//     /// assert_relative_eq!(euler.0, 0.1, epsilon = 1.0e-6);
//     /// assert_relative_eq!(euler.1, 0.2, epsilon = 1.0e-6);
//     /// assert_relative_eq!(euler.2, 0.3, epsilon = 1.0e-6);
//     /// ```
//     #[inline]
//     pub fn euler_angles(&self) -> (N, N, N) {
//         self.to_rotation_matrix().euler_angles()
//     }

//     /// Converts this unit quaternion into its equivalent homogeneous transformation matrix.
//     ///
//     /// # Example
//     ///
//     /// ```
//     /// # #[macro_use] extern crate approx;
//     /// # use std::f32;
//     /// # use nalgebra::{UnitDualQuaternion, Vector3, Matrix4};
//     /// let rot = UnitDualQuaternion::from_axis_angle(&Vector3::z_axis(), f32::consts::FRAC_PI_6);
//     /// let expected = Matrix4::new(0.8660254, -0.5,      0.0, 0.0,
//     ///                             0.5,       0.8660254, 0.0, 0.0,
//     ///                             0.0,       0.0,       1.0, 0.0,
//     ///                             0.0,       0.0,       0.0, 1.0);
//     ///
//     /// assert_relative_eq!(rot.to_homogeneous(), expected, epsilon = 1.0e-6);
//     /// ```
//     #[inline]
//     pub fn to_homogeneous(&self) -> MatrixN<N, U4> {
//         self.to_rotation_matrix().to_homogeneous()
//     }
// }

// impl<N: Real + fmt::Display> fmt::Display for UnitDualQuaternion<N> {
//     fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
//         if let Some(axis) = self.axis() {
//             let axis = axis.into_inner();
//             write!(
//                 f,
//                 "UnitDualQuaternion angle: {} − axis: ({}, {}, {})",
//                 self.angle(),
//                 axis[0],
//                 axis[1],
//                 axis[2]
//             )
//         } else {
//             write!(
//                 f,
//                 "UnitDualQuaternion angle: {} − axis: (undefined)",
//                 self.angle()
//             )
//         }
//     }
// }

// impl<N: Real + AbsDiffEq<Epsilon = N>> AbsDiffEq for UnitDualQuaternion<N> {
//     type Epsilon = N;

//     #[inline]
//     fn default_epsilon() -> Self::Epsilon {
//         N::default_epsilon()
//     }

//     #[inline]
//     fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
//         self.as_ref().abs_diff_eq(other.as_ref(), epsilon)
//     }
// }

// impl<N: Real + RelativeEq<Epsilon = N>> RelativeEq for UnitDualQuaternion<N> {
//     #[inline]
//     fn default_max_relative() -> Self::Epsilon {
//         N::default_max_relative()
//     }

//     #[inline]
//     fn relative_eq(
//         &self,
//         other: &Self,
//         epsilon: Self::Epsilon,
//         max_relative: Self::Epsilon,
//     ) -> bool
//     {
//         self.as_ref()
//             .relative_eq(other.as_ref(), epsilon, max_relative)
//     }
// }

// impl<N: Real + UlpsEq<Epsilon = N>> UlpsEq for UnitDualQuaternion<N> {
//     #[inline]
//     fn default_max_ulps() -> u32 {
//         N::default_max_ulps()
//     }

//     #[inline]
//     fn ulps_eq(&self, other: &Self, epsilon: Self::Epsilon, max_ulps: u32) -> bool {
//         self.as_ref().ulps_eq(other.as_ref(), epsilon, max_ulps)
//     }
// }
