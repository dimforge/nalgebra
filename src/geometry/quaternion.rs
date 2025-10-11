// Needed otherwise the rkyv macros generate code incompatible with rust-2024
#![cfg_attr(feature = "rkyv-serialize", allow(unsafe_op_in_unsafe_fn))]

use approx::{AbsDiffEq, RelativeEq, UlpsEq};
use num::Zero;
use std::fmt;
use std::hash::{Hash, Hasher};

#[cfg(feature = "serde-serialize-no-std")]
use crate::base::storage::Owned;
#[cfg(feature = "serde-serialize-no-std")]
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use simba::scalar::{ClosedNeg, RealField};
use simba::simd::{SimdBool, SimdOption, SimdRealField};

use crate::base::dimension::{U1, U3, U4};
use crate::base::storage::{CStride, RStride};
use crate::base::{
    Matrix3, Matrix4, MatrixView, MatrixViewMut, Normed, Scalar, Unit, Vector3, Vector4,
};

use crate::geometry::{Point3, Rotation};

#[cfg(feature = "rkyv-serialize")]
use rkyv::bytecheck;

/// A quaternion. See the type alias `UnitQuaternion = Unit<Quaternion>` for a quaternion
/// that may be used as a rotation.
#[repr(C)]
#[derive(Copy, Clone)]
#[cfg_attr(
    feature = "rkyv-serialize-no-std",
    derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize),
    archive(
        as = "Quaternion<T::Archived>",
        bound(archive = "
            T: rkyv::Archive,
            Vector4<T>: rkyv::Archive<Archived = Vector4<T::Archived>>
        ")
    )
)]
#[cfg_attr(feature = "rkyv-serialize", derive(bytecheck::CheckBytes))]
#[cfg_attr(feature = "defmt", derive(defmt::Format))]
pub struct Quaternion<T> {
    /// This quaternion as a 4D vector of coordinates in the `[ x, y, z, w ]` storage order.
    pub coords: Vector4<T>,
}

impl<T: fmt::Debug> fmt::Debug for Quaternion<T> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        self.coords.as_slice().fmt(formatter)
    }
}

impl<T: Scalar + Hash> Hash for Quaternion<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.coords.hash(state)
    }
}

impl<T: Scalar + Eq> Eq for Quaternion<T> {}

impl<T: Scalar> PartialEq for Quaternion<T> {
    #[inline]
    fn eq(&self, right: &Self) -> bool {
        self.coords == right.coords
    }
}

impl<T: Scalar + Zero> Default for Quaternion<T> {
    fn default() -> Self {
        Quaternion {
            coords: Vector4::zeros(),
        }
    }
}

#[cfg(feature = "bytemuck")]
unsafe impl<T: Scalar> bytemuck::Zeroable for Quaternion<T> where Vector4<T>: bytemuck::Zeroable {}

#[cfg(feature = "bytemuck")]
unsafe impl<T: Scalar> bytemuck::Pod for Quaternion<T>
where
    Vector4<T>: bytemuck::Pod,
    T: Copy,
{
}

#[cfg(feature = "serde-serialize-no-std")]
impl<T: Scalar> Serialize for Quaternion<T>
where
    Owned<T, U4>: Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.coords.serialize(serializer)
    }
}

#[cfg(feature = "serde-serialize-no-std")]
impl<'a, T: Scalar> Deserialize<'a> for Quaternion<T>
where
    Owned<T, U4>: Deserialize<'a>,
{
    fn deserialize<Des>(deserializer: Des) -> Result<Self, Des::Error>
    where
        Des: Deserializer<'a>,
    {
        let coords = Vector4::<T>::deserialize(deserializer)?;

        Ok(Self::from(coords))
    }
}

impl<T: SimdRealField> Quaternion<T>
where
    T::Element: SimdRealField,
{
    /// Moves this unit quaternion into one that owns its data.
    #[inline]
    #[deprecated(note = "This method is a no-op and will be removed in a future release.")]
    pub const fn into_owned(self) -> Self {
        self
    }

    /// Clones this unit quaternion into one that owns its data.
    #[inline]
    #[deprecated(note = "This method is a no-op and will be removed in a future release.")]
    pub fn clone_owned(&self) -> Self {
        Self::from(self.coords.clone_owned())
    }

    /// Normalizes this quaternion to have unit length.
    ///
    /// This operation converts a general quaternion into a unit quaternion by dividing
    /// all components by the quaternion's magnitude (norm). Unit quaternions are essential
    /// for representing rotations in 3D space, as they avoid gimbal lock and provide
    /// smooth interpolation between orientations.
    ///
    /// # Returns
    /// A new quaternion with the same direction but unit length (magnitude = 1.0).
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::Quaternion;
    /// // Create a non-unit quaternion
    /// let q = Quaternion::new(1.0, 2.0, 3.0, 4.0);
    /// let q_normalized = q.normalize();
    ///
    /// // The normalized quaternion has magnitude 1
    /// assert_relative_eq!(q_normalized.norm(), 1.0);
    ///
    /// // Game development: normalize a quaternion after accumulating rotations
    /// let mut rotation = Quaternion::new(2.0, 1.0, 0.5, 0.3);
    /// rotation = rotation.normalize();  // Now safe to use as a rotation
    /// ```
    ///
    /// # See Also
    /// - [`normalize_mut`](Self::normalize_mut) - In-place version of this method
    /// - [`norm`](Self::norm) - Get the magnitude of this quaternion
    /// - [`UnitQuaternion`] - Type representing normalized quaternions for rotations
    #[inline]
    #[must_use = "Did you mean to use normalize_mut()?"]
    pub fn normalize(&self) -> Self {
        Self::from(self.coords.normalize())
    }

    /// Returns the imaginary part of this quaternion as a 3D vector.
    ///
    /// A quaternion can be represented as `q = w + xi + yj + zk`, where `w` is the scalar
    /// (real) part, and `(x, y, z)` is the imaginary (vector) part. This method returns
    /// the vector `(x, y, z)`, which corresponds to the `i`, `j`, and `k` components.
    ///
    /// In the context of rotations, the imaginary part relates to the rotation axis
    /// scaled by the sine of half the rotation angle.
    ///
    /// # Returns
    /// A `Vector3<T>` containing the `(i, j, k)` components of the quaternion.
    ///
    /// # Example
    /// ```
    /// # use nalgebra::{Quaternion, Vector3};
    /// // Create quaternion: q = 1 + 2i + 3j + 4k
    /// let q = Quaternion::new(1.0, 2.0, 3.0, 4.0);
    /// let imag = q.imag();
    ///
    /// assert_eq!(imag, Vector3::new(2.0, 3.0, 4.0));
    ///
    /// // Aerospace: extract rotation axis information
    /// let rotation_quat = Quaternion::new(0.707, 0.707, 0.0, 0.0);
    /// let axis_scaled = rotation_quat.imag();
    /// // axis_scaled contains the rotation axis scaled by sin(angle/2)
    /// ```
    ///
    /// # See Also
    /// - [`scalar`](Self::scalar) - Get the real (scalar) part
    /// - [`vector`](Self::vector) - Get a view of the vector part
    /// - [`from_imag`](Self::from_imag) - Create a pure quaternion from a vector
    #[inline]
    #[must_use]
    pub fn imag(&self) -> Vector3<T> {
        self.coords.xyz()
    }

    /// Computes the conjugate of this quaternion.
    ///
    /// The conjugate of a quaternion `q = w + xi + yj + zk` is `q* = w - xi - yj - zk`.
    /// This operation negates the imaginary part while keeping the scalar part unchanged.
    ///
    /// For unit quaternions representing rotations, the conjugate represents the inverse
    /// rotation (rotation in the opposite direction). The conjugate is computationally
    /// cheaper than computing a full inverse.
    ///
    /// # Mathematical Properties
    /// - `(q*)* = q` (conjugate of conjugate is the original)
    /// - `(pq)* = q*p*` (conjugate of product is reverse product of conjugates)
    /// - For unit quaternions: `q* = q^(-1)` (conjugate equals inverse)
    ///
    /// # Returns
    /// A new quaternion with the imaginary components negated.
    ///
    /// # Example
    /// ```
    /// # use nalgebra::Quaternion;
    /// // Basic conjugate operation
    /// let q = Quaternion::new(1.0, 2.0, 3.0, 4.0);
    /// let conj = q.conjugate();
    /// assert!(conj.i == -2.0 && conj.j == -3.0 && conj.k == -4.0 && conj.w == 1.0);
    ///
    /// // Game development: reverse a rotation
    /// use nalgebra::UnitQuaternion;
    /// let forward_rotation = UnitQuaternion::from_euler_angles(0.5, 0.0, 0.0);
    /// let backward_rotation = forward_rotation.conjugate();
    /// // backward_rotation represents the opposite rotation
    ///
    /// // Aerospace: conjugate equals inverse for unit quaternions
    /// let q_unit = UnitQuaternion::from_euler_angles(0.3, 0.2, 0.1);
    /// let q_conj = q_unit.conjugate();
    /// let q_inv = q_unit.inverse();
    /// assert_eq!(q_conj, q_inv);
    /// ```
    ///
    /// # See Also
    /// - [`conjugate_mut`](Self::conjugate_mut) - In-place version
    /// - [`UnitQuaternion::inverse`] - Quaternion inverse (same as conjugate for unit quaternions)
    /// - [`try_inverse`](Self::try_inverse) - Safe inverse that handles zero quaternions
    #[inline]
    #[must_use = "Did you mean to use conjugate_mut()?"]
    pub fn conjugate(&self) -> Self {
        Self::from_parts(self.w.clone(), -self.imag())
    }

    /// Performs linear interpolation between two quaternions.
    ///
    /// Computes the weighted blend `self * (1 - t) + other * t`, where `t` is the
    /// interpolation parameter. When `t = 0`, returns `self`; when `t = 1`, returns `other`.
    ///
    /// **Note:** Linear interpolation (LERP) does not preserve quaternion magnitude. For
    /// rotation quaternions, you typically want to use [`UnitQuaternion::nlerp`] (normalized lerp)
    /// or [`UnitQuaternion::slerp`] (spherical lerp) instead, which maintain unit length.
    ///
    /// LERP is the fastest interpolation method but provides non-uniform angular velocity,
    /// meaning the rotation speed varies during interpolation.
    ///
    /// # Parameters
    /// - `other`: The target quaternion to interpolate toward
    /// - `t`: Interpolation parameter, typically in range [0, 1]
    ///
    /// # Returns
    /// A new quaternion that is the linear blend of `self` and `other`.
    ///
    /// # Example
    /// ```
    /// # use nalgebra::Quaternion;
    /// let q1 = Quaternion::new(1.0, 2.0, 3.0, 4.0);
    /// let q2 = Quaternion::new(10.0, 20.0, 30.0, 40.0);
    ///
    /// // Interpolate 10% of the way from q1 to q2
    /// assert_eq!(q1.lerp(&q2, 0.1), Quaternion::new(1.9, 3.8, 5.7, 7.6));
    ///
    /// // At t=0, returns self
    /// assert_eq!(q1.lerp(&q2, 0.0), q1);
    ///
    /// // At t=1, returns other
    /// assert_eq!(q1.lerp(&q2, 1.0), q2);
    ///
    /// // Game development: blend animation poses (remember to normalize!)
    /// use nalgebra::UnitQuaternion;
    /// let idle_rotation = UnitQuaternion::identity();
    /// let jump_rotation = UnitQuaternion::from_euler_angles(0.5, 0.0, 0.0);
    /// let blended = idle_rotation.lerp(&jump_rotation, 0.5);
    /// // Note: need to normalize before using as rotation
    /// let normalized_blend = blended.normalize();
    /// ```
    ///
    /// # See Also
    /// - [`nlerp`](UnitQuaternion::nlerp) - Normalized linear interpolation (recommended for rotations)
    /// - [`slerp`](UnitQuaternion::slerp) - Spherical linear interpolation (best quality for rotations)
    /// - [`try_slerp`](UnitQuaternion::try_slerp) - Safe spherical interpolation
    #[inline]
    #[must_use]
    pub fn lerp(&self, other: &Self, t: T) -> Self {
        self * (T::one() - t.clone()) + other * t
    }

    /// Returns a view of the vector part `(i, j, k)` of this quaternion.
    ///
    /// This method provides a read-only view (slice) of the imaginary components without
    /// copying data. A quaternion `q = w + xi + yj + zk` has a vector part `(x, y, z)`.
    ///
    /// This is similar to [`imag`](Self::imag) but returns a matrix view instead of an
    /// owned vector, which can be more efficient when you don't need ownership.
    ///
    /// # Returns
    /// A matrix view of the 3D vector part of the quaternion.
    ///
    /// # Example
    /// ```
    /// # use nalgebra::Quaternion;
    /// let q = Quaternion::new(1.0, 2.0, 3.0, 4.0);
    /// let vec = q.vector();
    ///
    /// assert_eq!(vec[0], 2.0);  // i component
    /// assert_eq!(vec[1], 3.0);  // j component
    /// assert_eq!(vec[2], 4.0);  // k component
    ///
    /// // Use for read-only access without copying
    /// let norm_squared = vec.norm_squared();
    /// ```
    ///
    /// # See Also
    /// - [`imag`](Self::imag) - Get an owned copy of the vector part
    /// - [`vector_mut`](Self::vector_mut) - Get a mutable view
    /// - [`scalar`](Self::scalar) - Get the scalar (real) part
    #[inline]
    #[must_use]
    pub fn vector(&self) -> MatrixView<'_, T, U3, U1, RStride<T, U4, U1>, CStride<T, U4, U1>> {
        self.coords.fixed_rows::<3>(0)
    }

    /// Returns the scalar (real) part `w` of this quaternion.
    ///
    /// A quaternion `q = w + xi + yj + zk` has a scalar part `w` (also called the real part)
    /// and a vector part `(x, y, z)` (the imaginary part). This method returns the `w` component.
    ///
    /// In the context of rotations, the scalar part equals `cos(angle/2)`, where `angle` is
    /// the rotation angle represented by a unit quaternion.
    ///
    /// # Returns
    /// The scalar component of the quaternion.
    ///
    /// # Example
    /// ```
    /// # use nalgebra::Quaternion;
    /// // Create quaternion q = 1 + 2i + 3j + 4k
    /// let q = Quaternion::new(1.0, 2.0, 3.0, 4.0);
    /// assert_eq!(q.scalar(), 1.0);
    ///
    /// // Aerospace: extract rotation angle information
    /// use nalgebra::UnitQuaternion;
    /// let rotation = UnitQuaternion::from_axis_angle(&nalgebra::Vector3::x_axis(), 1.0);
    /// let w = rotation.scalar();
    /// // w = cos(0.5) for a 1 radian rotation
    /// ```
    ///
    /// # See Also
    /// - [`imag`](Self::imag) - Get the imaginary (vector) part
    /// - [`vector`](Self::vector) - Get a view of the vector part
    /// - [`as_vector`](Self::as_vector) - Get the full quaternion as a 4D vector
    #[inline]
    #[must_use]
    pub fn scalar(&self) -> T {
        self.coords[3].clone()
    }

    /// Reinterprets this quaternion as a 4D vector reference.
    ///
    /// Returns a reference to the internal 4D vector representation of the quaternion.
    /// The storage order is `[i, j, k, w]` (imaginary components first, then scalar).
    ///
    /// **Important:** Note the storage order differs from the constructor argument order.
    /// The [`new`](Self::new) constructor takes arguments as `(w, i, j, k)`, but the
    /// internal storage is `[i, j, k, w]`.
    ///
    /// # Returns
    /// A reference to the underlying `Vector4<T>` with layout `[i, j, k, w]`.
    ///
    /// # Example
    /// ```
    /// # use nalgebra::{Vector4, Quaternion};
    /// // Constructor order is (w, i, j, k)
    /// let q = Quaternion::new(1.0, 2.0, 3.0, 4.0);
    ///
    /// // But storage order is [i, j, k, w]
    /// assert_eq!(*q.as_vector(), Vector4::new(2.0, 3.0, 4.0, 1.0));
    ///
    /// // Useful for direct component access
    /// let vec = q.as_vector();
    /// assert_eq!(vec[0], 2.0);  // i
    /// assert_eq!(vec[1], 3.0);  // j
    /// assert_eq!(vec[2], 4.0);  // k
    /// assert_eq!(vec[3], 1.0);  // w
    ///
    /// // Game development: efficient serialization
    /// let bytes = q.as_vector().as_slice();
    /// ```
    ///
    /// # See Also
    /// - [`as_vector_mut`](Self::as_vector_mut) - Get a mutable reference
    /// - [`coords`](Self::coords) - The public field holding the vector
    /// - [`into_inner`](crate::UnitQuaternion::into_inner) - Convert UnitQuaternion to Quaternion
    #[inline]
    #[must_use]
    pub const fn as_vector(&self) -> &Vector4<T> {
        &self.coords
    }

    /// Computes the norm (magnitude/length) of this quaternion.
    ///
    /// The norm is computed as `sqrt(w² + x² + y² + z²)`, representing the length
    /// of the quaternion when viewed as a 4D vector. For unit quaternions (rotations),
    /// the norm is always 1.0.
    ///
    /// # Returns
    /// The Euclidean norm of the quaternion.
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::Quaternion;
    /// let q = Quaternion::new(1.0, 2.0, 3.0, 4.0);
    /// // norm = sqrt(1² + 2² + 3² + 4²) = sqrt(30)
    /// assert_relative_eq!(q.norm(), 5.47722557, epsilon = 1.0e-6);
    ///
    /// // Unit quaternions always have norm 1
    /// use nalgebra::UnitQuaternion;
    /// let unit_q = UnitQuaternion::from_euler_angles(1.0, 0.5, 0.3);
    /// assert_relative_eq!(unit_q.norm(), 1.0);
    /// ```
    ///
    /// # See Also
    /// - [`norm_squared`](Self::norm_squared) - More efficient when you only need the squared norm
    /// - [`magnitude`](Self::magnitude) - Synonym for this method
    /// - [`normalize`](Self::normalize) - Create a unit quaternion
    #[inline]
    #[must_use]
    pub fn norm(&self) -> T {
        self.coords.norm()
    }

    /// Computes the magnitude of this quaternion (synonym for [`norm`](Self::norm)).
    ///
    /// This method is provided for convenience and is identical to [`norm`](Self::norm).
    /// The magnitude represents the length of the quaternion in 4D space.
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::Quaternion;
    /// let q = Quaternion::new(1.0, 2.0, 3.0, 4.0);
    /// assert_relative_eq!(q.magnitude(), 5.47722557, epsilon = 1.0e-6);
    /// assert_eq!(q.magnitude(), q.norm());  // Same result
    /// ```
    ///
    /// # See Also
    /// - [`norm`](Self::norm) - The primary method name
    /// - [`magnitude_squared`](Self::magnitude_squared) - Squared magnitude
    #[inline]
    #[must_use]
    pub fn magnitude(&self) -> T {
        self.norm()
    }

    /// Computes the squared norm of this quaternion.
    ///
    /// Returns `w² + x² + y² + z²` without taking the square root. This is more
    /// computationally efficient than [`norm`](Self::norm) when you only need to
    /// compare magnitudes or check if a quaternion is normalized.
    ///
    /// # Returns
    /// The squared Euclidean norm of the quaternion.
    ///
    /// # Example
    /// ```
    /// # use nalgebra::Quaternion;
    /// let q = Quaternion::new(1.0, 2.0, 3.0, 4.0);
    /// // norm_squared = 1² + 2² + 3² + 4² = 30
    /// assert_eq!(q.norm_squared(), 30.0);
    ///
    /// // Game development: efficient comparison without sqrt
    /// let q1 = Quaternion::new(1.0, 0.0, 0.0, 0.0);
    /// let q2 = Quaternion::new(0.5, 0.5, 0.5, 0.5);
    /// if q1.norm_squared() > q2.norm_squared() {
    ///     // q1 has larger magnitude
    /// }
    ///
    /// // Check if quaternion is approximately normalized
    /// use nalgebra::UnitQuaternion;
    /// let unit_q = UnitQuaternion::identity();
    /// assert!((unit_q.norm_squared() - 1.0).abs() < 1e-6);
    /// ```
    ///
    /// # See Also
    /// - [`norm`](Self::norm) - Takes the square root
    /// - [`magnitude_squared`](Self::magnitude_squared) - Synonym for this method
    #[inline]
    #[must_use]
    pub fn norm_squared(&self) -> T {
        self.coords.norm_squared()
    }

    /// Computes the squared magnitude (synonym for [`norm_squared`](Self::norm_squared)).
    ///
    /// This method is provided for convenience and is identical to
    /// [`norm_squared`](Self::norm_squared).
    ///
    /// # Example
    /// ```
    /// # use nalgebra::Quaternion;
    /// let q = Quaternion::new(1.0, 2.0, 3.0, 4.0);
    /// assert_eq!(q.magnitude_squared(), 30.0);
    /// assert_eq!(q.magnitude_squared(), q.norm_squared());
    /// ```
    ///
    /// # See Also
    /// - [`norm_squared`](Self::norm_squared) - The primary method name
    /// - [`magnitude`](Self::magnitude) - Non-squared magnitude
    #[inline]
    #[must_use]
    pub fn magnitude_squared(&self) -> T {
        self.norm_squared()
    }

    /// Computes the dot product (inner product) of two quaternions.
    ///
    /// The dot product is calculated as `w₁w₂ + x₁x₂ + y₁y₂ + z₁z₂`, treating
    /// quaternions as 4D vectors. For unit quaternions, the dot product indicates
    /// how similar two rotations are:
    /// - `1.0` means identical rotations
    /// - `0.0` means perpendicular (90° apart in 4D space)
    /// - `-1.0` means opposite rotations
    ///
    /// # Parameters
    /// - `rhs`: The other quaternion
    ///
    /// # Returns
    /// The scalar dot product value.
    ///
    /// # Example
    /// ```
    /// # use nalgebra::Quaternion;
    /// let q1 = Quaternion::new(1.0, 2.0, 3.0, 4.0);
    /// let q2 = Quaternion::new(5.0, 6.0, 7.0, 8.0);
    /// // dot = 1*5 + 2*6 + 3*7 + 4*8 = 5 + 12 + 21 + 32 = 70
    /// assert_eq!(q1.dot(&q2), 70.0);
    ///
    /// // Game development: measure rotation similarity
    /// use nalgebra::UnitQuaternion;
    /// let rotation1 = UnitQuaternion::from_euler_angles(0.1, 0.0, 0.0);
    /// let rotation2 = UnitQuaternion::from_euler_angles(0.2, 0.0, 0.0);
    /// let similarity = rotation1.dot(&rotation2);
    /// // similarity close to 1.0 means rotations are similar
    ///
    /// // Aerospace: determine if quaternions are in same hemisphere
    /// let q3 = UnitQuaternion::identity();
    /// let q4 = UnitQuaternion::from_euler_angles(3.0, 0.0, 0.0);
    /// if q3.dot(&q4) < 0.0 {
    ///     // Use negation to ensure shortest path interpolation
    /// }
    /// ```
    ///
    /// # See Also
    /// - [`inner`](Self::inner) - Quaternion inner product (different from dot product)
    /// - [`norm_squared`](Self::norm_squared) - Square of the norm equals self.dot(self)
    #[inline]
    #[must_use]
    pub fn dot(&self, rhs: &Self) -> T {
        self.coords.dot(&rhs.coords)
    }
}

impl<T: SimdRealField> Quaternion<T>
where
    T::Element: SimdRealField,
{
    /// Inverts this quaternion if it is not zero.
    ///
    /// This method also does not works with SIMD components (see `simd_try_inverse` instead).
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
    #[must_use = "Did you mean to use try_inverse_mut()?"]
    pub fn try_inverse(&self) -> Option<Self>
    where
        T: RealField,
    {
        let mut res = self.clone();

        if res.try_inverse_mut() {
            Some(res)
        } else {
            None
        }
    }

    /// Attempt to inverse this quaternion.
    ///
    /// This method also works with SIMD components.
    #[inline]
    #[must_use = "Did you mean to use try_inverse_mut()?"]
    pub fn simd_try_inverse(&self) -> SimdOption<Self> {
        let norm_squared = self.norm_squared();
        let ge = norm_squared.clone().simd_ge(T::simd_default_epsilon());
        SimdOption::new(self.conjugate() / norm_squared, ge)
    }

    /// Calculates the inner product (also known as the dot product).
    /// See "Foundations of Game Engine Development, Volume 1: Mathematics" by Lengyel
    /// Formula 4.89.
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::Quaternion;
    /// let a = Quaternion::new(0.0, 2.0, 3.0, 4.0);
    /// let b = Quaternion::new(0.0, 5.0, 2.0, 1.0);
    /// let expected = Quaternion::new(-20.0, 0.0, 0.0, 0.0);
    /// let result = a.inner(&b);
    /// assert_relative_eq!(expected, result, epsilon = 1.0e-5);
    /// ```
    #[inline]
    #[must_use]
    pub fn inner(&self, other: &Self) -> Self {
        (self * other + other * self).half()
    }

    /// Calculates the outer product (also known as the wedge product).
    /// See "Foundations of Game Engine Development, Volume 1: Mathematics" by Lengyel
    /// Formula 4.89.
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::Quaternion;
    /// let a = Quaternion::new(0.0, 2.0, 3.0, 4.0);
    /// let b = Quaternion::new(0.0, 5.0, 2.0, 1.0);
    /// let expected = Quaternion::new(0.0, -5.0, 18.0, -11.0);
    /// let result = a.outer(&b);
    /// assert_relative_eq!(expected, result, epsilon = 1.0e-5);
    /// ```
    #[inline]
    #[must_use]
    pub fn outer(&self, other: &Self) -> Self {
        #[allow(clippy::eq_op)]
        (self * other - other * self).half()
    }

    /// Calculates the projection of `self` onto `other` (also known as the parallel).
    /// See "Foundations of Game Engine Development, Volume 1: Mathematics" by Lengyel
    /// Formula 4.94.
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::Quaternion;
    /// let a = Quaternion::new(0.0, 2.0, 3.0, 4.0);
    /// let b = Quaternion::new(0.0, 5.0, 2.0, 1.0);
    /// let expected = Quaternion::new(0.0, 3.333333333333333, 1.3333333333333333, 0.6666666666666666);
    /// let result = a.project(&b).unwrap();
    /// assert_relative_eq!(expected, result, epsilon = 1.0e-5);
    /// ```
    #[inline]
    #[must_use]
    pub fn project(&self, other: &Self) -> Option<Self>
    where
        T: RealField,
    {
        self.inner(other).right_div(other)
    }

    /// Calculates the rejection of `self` from `other` (also known as the perpendicular).
    /// See "Foundations of Game Engine Development, Volume 1: Mathematics" by Lengyel
    /// Formula 4.94.
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::Quaternion;
    /// let a = Quaternion::new(0.0, 2.0, 3.0, 4.0);
    /// let b = Quaternion::new(0.0, 5.0, 2.0, 1.0);
    /// let expected = Quaternion::new(0.0, -1.3333333333333333, 1.6666666666666665, 3.3333333333333335);
    /// let result = a.reject(&b).unwrap();
    /// assert_relative_eq!(expected, result, epsilon = 1.0e-5);
    /// ```
    #[inline]
    #[must_use]
    pub fn reject(&self, other: &Self) -> Option<Self>
    where
        T: RealField,
    {
        self.outer(other).right_div(other)
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
    #[must_use]
    pub fn polar_decomposition(&self) -> (T, T, Option<Unit<Vector3<T>>>)
    where
        T: RealField,
    {
        match Unit::try_new_and_get(self.clone(), T::zero()) {
            Some((q, n)) => match Unit::try_new(self.vector().clone_owned(), T::zero()) {
                Some(axis) => {
                    let angle = q.angle() / crate::convert(2.0f64);

                    (n, angle, Some(axis))
                }
                None => (n, T::zero(), None),
            },
            None => (T::zero(), T::zero(), None),
        }
    }

    /// Computes the natural logarithm of a quaternion.
    ///
    /// The quaternion logarithm extends the complex logarithm to four dimensions.
    /// For a quaternion `q = ||q|| * e^(v*θ)` where `v` is a unit vector, this returns
    /// `ln(q) = ln(||q||) + v*θ`, effectively converting from exponential to logarithmic form.
    ///
    /// This is particularly useful for:
    /// - Converting rotations to axis-angle representation
    /// - Interpolating rotations in log space
    /// - Computing quaternion powers and roots
    ///
    /// # Mathematical Background
    /// For a unit quaternion representing a rotation by angle `θ` around axis `v`:
    /// `ln(q) = 0 + v*(θ/2)` (the scalar part becomes 0)
    ///
    /// # Returns
    /// The natural logarithm of the quaternion.
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::Quaternion;
    /// let q = Quaternion::new(2.0, 5.0, 0.0, 0.0);
    /// let ln_q = q.ln();
    /// assert_relative_eq!(ln_q, Quaternion::new(1.683647, 1.190289, 0.0, 0.0), epsilon = 1.0e-6);
    ///
    /// // Verify with exponential: exp(ln(q)) = q
    /// assert_relative_eq!(ln_q.exp(), q, epsilon = 1.0e-5);
    ///
    /// // Aerospace: extract rotation information
    /// use nalgebra::UnitQuaternion;
    /// let rotation = UnitQuaternion::from_euler_angles(0.5, 0.3, 0.2);
    /// let log_rot = rotation.ln();
    /// // The vector part contains the scaled rotation axis
    /// ```
    ///
    /// # See Also
    /// - [`exp`](Self::exp) - Quaternion exponential (inverse operation)
    /// - [`powf`](Self::powf) - Raise quaternion to a power (uses ln internally)
    /// - [`UnitQuaternion::ln`] - Specialized version for unit quaternions
    #[inline]
    #[must_use]
    pub fn ln(&self) -> Self {
        let n = self.norm();
        let v = self.vector();
        let s = self.scalar();

        Self::from_parts(n.clone().simd_ln(), v.normalize() * (s / n).simd_acos())
    }

    /// Computes the exponential of a quaternion.
    ///
    /// The quaternion exponential extends the complex exponential to four dimensions.
    /// For a quaternion `q = w + v` (where `v` is the vector part), this computes
    /// `exp(q) = e^w * (cos(||v||) + (v/||v||)*sin(||v||))`.
    ///
    /// This is the fundamental operation for:
    /// - Converting axis-angle to quaternion rotation
    /// - Implementing quaternion interpolation
    /// - Solving differential equations involving rotations
    ///
    /// # Mathematical Properties
    /// - `exp(0) = 1` (identity quaternion)
    /// - `exp(ln(q)) = q` (inverse of logarithm)
    /// - `exp(q)` is always non-zero
    ///
    /// # Returns
    /// The exponential of the quaternion.
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::Quaternion;
    /// let q = Quaternion::new(1.683647, 1.190289, 0.0, 0.0);
    /// let exp_q = q.exp();
    /// assert_relative_eq!(exp_q, Quaternion::new(2.0, 5.0, 0.0, 0.0), epsilon = 1.0e-5);
    ///
    /// // Verify with logarithm: ln(exp(q)) = q
    /// assert_relative_eq!(exp_q.ln(), q, epsilon = 1.0e-5);
    ///
    /// // Game development: convert axis-angle to rotation
    /// use nalgebra::{UnitQuaternion, Vector3};
    /// let axis_angle = Vector3::new(0.1, 0.2, 0.3);
    /// let rotation = UnitQuaternion::new(axis_angle);
    /// // Internally uses exp: rotation = exp(axis_angle/2)
    /// ```
    ///
    /// # See Also
    /// - [`ln`](Self::ln) - Quaternion logarithm (inverse operation)
    /// - [`exp_eps`](Self::exp_eps) - Version with epsilon parameter for edge cases
    /// - [`powf`](Self::powf) - Raise to a power (uses exp and ln)
    #[inline]
    #[must_use]
    pub fn exp(&self) -> Self {
        self.exp_eps(T::simd_default_epsilon())
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
    #[must_use]
    pub fn exp_eps(&self, eps: T) -> Self {
        let v = self.vector();
        let nn = v.norm_squared();
        let le = nn.clone().simd_le(eps.clone() * eps);
        le.if_else(Self::identity, || {
            let w_exp = self.scalar().simd_exp();
            let n = nn.simd_sqrt();
            let nv = v * (w_exp.clone() * n.clone().simd_sin() / n.clone());

            Self::from_parts(w_exp * n.simd_cos(), nv)
        })
    }

    /// Raises the quaternion to a given floating-point power.
    ///
    /// Computes `q^n` where `q` is this quaternion and `n` is a scalar power.
    /// This is calculated as `exp(n * ln(q))`, using the quaternion logarithm
    /// and exponential functions.
    ///
    /// For unit quaternions representing rotations, raising to power `n` effectively
    /// scales the rotation angle by `n`. For example, `q^0.5` gives half the rotation,
    /// and `q^2.0` gives double the rotation.
    ///
    /// # Parameters
    /// - `n`: The power to raise the quaternion to
    ///
    /// # Returns
    /// The quaternion raised to the power `n`.
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::Quaternion;
    /// let q = Quaternion::new(1.0, 2.0, 3.0, 4.0);
    /// let q_pow = q.powf(1.5);
    /// assert_relative_eq!(q_pow, Quaternion::new(-6.2576659, 4.1549037, 6.2323556, 8.3098075), epsilon = 1.0e-6);
    ///
    /// // Game development: scale rotation angles
    /// use nalgebra::UnitQuaternion;
    /// let rotation = UnitQuaternion::from_euler_angles(1.0, 0.0, 0.0);
    /// let half_rotation = rotation.powf(0.5);
    /// // half_rotation represents half the rotation angle
    ///
    /// // Aerospace: double a rotation
    /// let turn = UnitQuaternion::from_euler_angles(0.5, 0.3, 0.2);
    /// let double_turn = turn.powf(2.0);
    /// // Equivalent to applying the rotation twice
    ///
    /// // Square root of a quaternion
    /// let q_sqrt = q.powf(0.5);
    /// assert_relative_eq!(q_sqrt.squared(), q, epsilon = 1.0e-5);
    /// ```
    ///
    /// # See Also
    /// - [`ln`](Self::ln) - Natural logarithm (used internally)
    /// - [`exp`](Self::exp) - Exponential function (used internally)
    /// - [`UnitQuaternion::powf`] - Optimized version for unit quaternions
    /// - [`sqrt`](Self::sqrt) - Square root (equivalent to powf(0.5))
    #[inline]
    #[must_use]
    pub fn powf(&self, n: T) -> Self {
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
    pub const fn as_vector_mut(&mut self) -> &mut Vector4<T> {
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
    ) -> MatrixViewMut<'_, T, U3, U1, RStride<T, U4, U1>, CStride<T, U4, U1>> {
        self.coords.fixed_rows_mut::<3>(0)
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
        self.coords[0] = -self.coords[0].clone();
        self.coords[1] = -self.coords[1].clone();
        self.coords[2] = -self.coords[2].clone();
    }

    /// Inverts this quaternion in-place if it is not zero.
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::Quaternion;
    /// let mut q = Quaternion::new(1.0f32, 2.0, 3.0, 4.0);
    ///
    /// assert!(q.try_inverse_mut());
    /// assert_relative_eq!(q * Quaternion::new(1.0, 2.0, 3.0, 4.0), Quaternion::identity());
    ///
    /// //Non-invertible case
    /// let mut q = Quaternion::new(0.0f32, 0.0, 0.0, 0.0);
    /// assert!(!q.try_inverse_mut());
    /// ```
    #[inline]
    pub fn try_inverse_mut(&mut self) -> T::SimdBool {
        let norm_squared = self.norm_squared();
        let ge = norm_squared.clone().simd_ge(T::simd_default_epsilon());
        *self = ge.if_else(|| self.conjugate() / norm_squared, || self.clone());
        ge
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
    pub fn normalize_mut(&mut self) -> T {
        self.coords.normalize_mut()
    }

    /// Calculates square of a quaternion.
    #[inline]
    #[must_use]
    pub fn squared(&self) -> Self {
        self * self
    }

    /// Divides quaternion into two.
    #[inline]
    #[must_use]
    pub fn half(&self) -> Self {
        self / crate::convert(2.0f64)
    }

    /// Calculates square root.
    #[inline]
    #[must_use]
    pub fn sqrt(&self) -> Self {
        self.powf(crate::convert(0.5))
    }

    /// Check if the quaternion is pure.
    ///
    /// A quaternion is pure if it has no real part (`self.w == 0.0`).
    #[inline]
    #[must_use]
    pub fn is_pure(&self) -> bool {
        self.w.is_zero()
    }

    /// Convert quaternion to pure quaternion.
    #[inline]
    #[must_use]
    pub fn pure(&self) -> Self {
        Self::from_imag(self.imag())
    }

    /// Left quaternionic division.
    ///
    /// Calculates B<sup>-1</sup> * A where A = self, B = other.
    #[inline]
    #[must_use]
    pub fn left_div(&self, other: &Self) -> Option<Self>
    where
        T: RealField,
    {
        other.try_inverse().map(|inv| inv * self)
    }

    /// Right quaternionic division.
    ///
    /// Calculates A * B<sup>-1</sup> where A = self, B = other.
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::Quaternion;
    /// let a = Quaternion::new(0.0, 1.0, 2.0, 3.0);
    /// let b = Quaternion::new(0.0, 5.0, 2.0, 1.0);
    /// let result = a.right_div(&b).unwrap();
    /// let expected = Quaternion::new(0.4, 0.13333333333333336, -0.4666666666666667, 0.26666666666666666);
    /// assert_relative_eq!(expected, result, epsilon = 1.0e-7);
    /// ```
    #[inline]
    #[must_use]
    pub fn right_div(&self, other: &Self) -> Option<Self>
    where
        T: RealField,
    {
        other.try_inverse().map(|inv| self * inv)
    }

    /// Calculates the quaternionic cosinus.
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::Quaternion;
    /// let input = Quaternion::new(1.0, 2.0, 3.0, 4.0);
    /// let expected = Quaternion::new(58.93364616794395, -34.086183690465596, -51.1292755356984, -68.17236738093119);
    /// let result = input.cos();
    /// assert_relative_eq!(expected, result, epsilon = 1.0e-7);
    /// ```
    #[inline]
    #[must_use]
    pub fn cos(&self) -> Self {
        let z = self.imag().magnitude();
        let w = -self.w.clone().simd_sin() * z.clone().simd_sinhc();
        Self::from_parts(self.w.clone().simd_cos() * z.simd_cosh(), self.imag() * w)
    }

    /// Calculates the quaternionic arccosinus.
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::Quaternion;
    /// let input = Quaternion::new(1.0, 2.0, 3.0, 4.0);
    /// let result = input.cos().acos();
    /// assert_relative_eq!(input, result, epsilon = 1.0e-7);
    /// ```
    #[inline]
    #[must_use]
    pub fn acos(&self) -> Self {
        let u = Self::from_imag(self.imag().normalize());
        let identity = Self::identity();

        let z = (self + (self.squared() - identity).sqrt()).ln();

        -(u * z)
    }

    /// Calculates the quaternionic sinus.
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::Quaternion;
    /// let input = Quaternion::new(1.0, 2.0, 3.0, 4.0);
    /// let expected = Quaternion::new(91.78371578403467, 21.886486853029176, 32.82973027954377, 43.77297370605835);
    /// let result = input.sin();
    /// assert_relative_eq!(expected, result, epsilon = 1.0e-7);
    /// ```
    #[inline]
    #[must_use]
    pub fn sin(&self) -> Self {
        let z = self.imag().magnitude();
        let w = self.w.clone().simd_cos() * z.clone().simd_sinhc();
        Self::from_parts(self.w.clone().simd_sin() * z.simd_cosh(), self.imag() * w)
    }

    /// Calculates the quaternionic arcsinus.
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::Quaternion;
    /// let input = Quaternion::new(1.0, 2.0, 3.0, 4.0);
    /// let result = input.sin().asin();
    /// assert_relative_eq!(input, result, epsilon = 1.0e-7);
    /// ```
    #[inline]
    #[must_use]
    pub fn asin(&self) -> Self {
        let u = Self::from_imag(self.imag().normalize());
        let identity = Self::identity();

        let z = ((u.clone() * self) + (identity - self.squared()).sqrt()).ln();

        -(u * z)
    }

    /// Calculates the quaternionic tangent.
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::Quaternion;
    /// let input = Quaternion::new(1.0, 2.0, 3.0, 4.0);
    /// let expected = Quaternion::new(0.00003821631725009489, 0.3713971716439371, 0.5570957574659058, 0.7427943432878743);
    /// let result = input.tan();
    /// assert_relative_eq!(expected, result, epsilon = 1.0e-7);
    /// ```
    #[inline]
    #[must_use]
    pub fn tan(&self) -> Self
    where
        T: RealField,
    {
        self.sin().right_div(&self.cos()).unwrap()
    }

    /// Calculates the quaternionic arctangent.
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::Quaternion;
    /// let input = Quaternion::new(1.0, 2.0, 3.0, 4.0);
    /// let result = input.tan().atan();
    /// assert_relative_eq!(input, result, epsilon = 1.0e-7);
    /// ```
    #[inline]
    #[must_use]
    pub fn atan(&self) -> Self
    where
        T: RealField,
    {
        let u = Self::from_imag(self.imag().normalize());
        let num = u.clone() + self;
        let den = u.clone() - self;
        let fr = num.right_div(&den).unwrap();
        let ln = fr.ln();
        (u.half()) * ln
    }

    /// Calculates the hyperbolic quaternionic sinus.
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::Quaternion;
    /// let input = Quaternion::new(1.0, 2.0, 3.0, 4.0);
    /// let expected = Quaternion::new(0.7323376060463428, -0.4482074499805421, -0.6723111749708133, -0.8964148999610843);
    /// let result = input.sinh();
    /// assert_relative_eq!(expected, result, epsilon = 1.0e-7);
    /// ```
    #[inline]
    #[must_use]
    pub fn sinh(&self) -> Self {
        (self.exp() - (-self).exp()).half()
    }

    /// Calculates the hyperbolic quaternionic arcsinus.
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::Quaternion;
    /// let input = Quaternion::new(1.0, 2.0, 3.0, 4.0);
    /// let expected = Quaternion::new(2.385889902585242, 0.514052600662788, 0.7710789009941821, 1.028105201325576);
    /// let result = input.asinh();
    /// assert_relative_eq!(expected, result, epsilon = 1.0e-7);
    /// ```
    #[inline]
    #[must_use]
    pub fn asinh(&self) -> Self {
        let identity = Self::identity();
        (self + (identity + self.squared()).sqrt()).ln()
    }

    /// Calculates the hyperbolic quaternionic cosinus.
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::Quaternion;
    /// let input = Quaternion::new(1.0, 2.0, 3.0, 4.0);
    /// let expected = Quaternion::new(0.9615851176369566, -0.3413521745610167, -0.5120282618415251, -0.6827043491220334);
    /// let result = input.cosh();
    /// assert_relative_eq!(expected, result, epsilon = 1.0e-7);
    /// ```
    #[inline]
    #[must_use]
    pub fn cosh(&self) -> Self {
        (self.exp() + (-self).exp()).half()
    }

    /// Calculates the hyperbolic quaternionic arccosinus.
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::Quaternion;
    /// let input = Quaternion::new(1.0, 2.0, 3.0, 4.0);
    /// let expected = Quaternion::new(2.4014472020074007, 0.5162761016176176, 0.7744141524264264, 1.0325522032352352);
    /// let result = input.acosh();
    /// assert_relative_eq!(expected, result, epsilon = 1.0e-7);
    /// ```
    #[inline]
    #[must_use]
    pub fn acosh(&self) -> Self {
        let identity = Self::identity();
        (self + (self + identity.clone()).sqrt() * (self - identity).sqrt()).ln()
    }

    /// Calculates the hyperbolic quaternionic tangent.
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::Quaternion;
    /// let input = Quaternion::new(1.0, 2.0, 3.0, 4.0);
    /// let expected = Quaternion::new(1.0248695360556623, -0.10229568178876419, -0.1534435226831464, -0.20459136357752844);
    /// let result = input.tanh();
    /// assert_relative_eq!(expected, result, epsilon = 1.0e-7);
    /// ```
    #[inline]
    #[must_use]
    pub fn tanh(&self) -> Self
    where
        T: RealField,
    {
        self.sinh().right_div(&self.cosh()).unwrap()
    }

    /// Calculates the hyperbolic quaternionic arctangent.
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::Quaternion;
    /// let input = Quaternion::new(1.0, 2.0, 3.0, 4.0);
    /// let expected = Quaternion::new(0.03230293287000163, 0.5173453683196951, 0.7760180524795426, 1.0346907366393903);
    /// let result = input.atanh();
    /// assert_relative_eq!(expected, result, epsilon = 1.0e-7);
    /// ```
    #[inline]
    #[must_use]
    pub fn atanh(&self) -> Self {
        let identity = Self::identity();
        ((identity.clone() + self).ln() - (identity - self).ln()).half()
    }
}

impl<T: RealField + AbsDiffEq<Epsilon = T>> AbsDiffEq for Quaternion<T> {
    type Epsilon = T;

    #[inline]
    fn default_epsilon() -> Self::Epsilon {
        T::default_epsilon()
    }

    #[inline]
    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        self.as_vector().abs_diff_eq(other.as_vector(), epsilon.clone()) ||
        // Account for the double-covering of S², i.e. q = -q
        self.as_vector().iter().zip(other.as_vector().iter()).all(|(a, b)| a.abs_diff_eq(&-b.clone(), epsilon.clone()))
    }
}

impl<T: RealField + RelativeEq<Epsilon = T>> RelativeEq for Quaternion<T> {
    #[inline]
    fn default_max_relative() -> Self::Epsilon {
        T::default_max_relative()
    }

    #[inline]
    fn relative_eq(
        &self,
        other: &Self,
        epsilon: Self::Epsilon,
        max_relative: Self::Epsilon,
    ) -> bool {
        self.as_vector().relative_eq(other.as_vector(), epsilon.clone(), max_relative.clone()) ||
        // Account for the double-covering of S², i.e. q = -q
        self.as_vector().iter().zip(other.as_vector().iter()).all(|(a, b)| a.relative_eq(&-b.clone(), epsilon.clone(), max_relative.clone()))
    }
}

impl<T: RealField + UlpsEq<Epsilon = T>> UlpsEq for Quaternion<T> {
    #[inline]
    fn default_max_ulps() -> u32 {
        T::default_max_ulps()
    }

    #[inline]
    fn ulps_eq(&self, other: &Self, epsilon: Self::Epsilon, max_ulps: u32) -> bool {
        self.as_vector().ulps_eq(other.as_vector(), epsilon.clone(), max_ulps) ||
        // Account for the double-covering of S², i.e. q = -q.
        self.as_vector().iter().zip(other.as_vector().iter()).all(|(a, b)| a.ulps_eq(&-b.clone(), epsilon.clone(), max_ulps))
    }
}

impl<T: RealField + fmt::Display> fmt::Display for Quaternion<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Quaternion {} − ({}, {}, {})",
            self[3], self[0], self[1], self[2]
        )
    }
}

/// A unit quaternions. May be used to represent a rotation.
pub type UnitQuaternion<T> = Unit<Quaternion<T>>;

impl<T: Scalar + ClosedNeg + PartialEq> PartialEq for UnitQuaternion<T> {
    #[inline]
    fn eq(&self, rhs: &Self) -> bool {
        self.coords == rhs.coords ||
        // Account for the double-covering of S², i.e. q = -q
        self.coords.iter().zip(rhs.coords.iter()).all(|(a, b)| *a == -b.clone())
    }
}

impl<T: Scalar + ClosedNeg + Eq> Eq for UnitQuaternion<T> {}

impl<T: SimdRealField> Normed for Quaternion<T> {
    type Norm = T::SimdRealField;

    #[inline]
    fn norm(&self) -> T::SimdRealField {
        self.coords.norm()
    }

    #[inline]
    fn norm_squared(&self) -> T::SimdRealField {
        self.coords.norm_squared()
    }

    #[inline]
    fn scale_mut(&mut self, n: Self::Norm) {
        self.coords.scale_mut(n)
    }

    #[inline]
    fn unscale_mut(&mut self, n: Self::Norm) {
        self.coords.unscale_mut(n)
    }
}

impl<T: SimdRealField> UnitQuaternion<T>
where
    T::Element: SimdRealField,
{
    /// Returns the rotation angle represented by this unit quaternion, in radians.
    ///
    /// The angle is always in the range `[0, π]` (0 to 180 degrees). This is because
    /// quaternions exhibit double-coverage: `q` and `-q` represent the same rotation,
    /// and we always choose the representation with the smaller angle.
    ///
    /// For a unit quaternion representing a rotation, the relationship between the
    /// quaternion components and the rotation angle `θ` is:
    /// - `w = cos(θ/2)` (scalar part)
    /// - `||xyz|| = sin(θ/2)` (magnitude of vector part)
    ///
    /// # Returns
    /// The rotation angle in radians, in the range `[0, π]`.
    ///
    /// # Example
    /// ```
    /// # use nalgebra::{Unit, UnitQuaternion, Vector3};
    /// let axis = Unit::new_normalize(Vector3::new(1.0, 2.0, 3.0));
    /// let rot = UnitQuaternion::from_axis_angle(&axis, 1.78);
    /// assert_eq!(rot.angle(), 1.78);
    ///
    /// // Game development: check if rotation is small
    /// let small_rotation = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), 0.1);
    /// if small_rotation.angle() < 0.2 {
    ///     println!("Almost no rotation");
    /// }
    ///
    /// // Aerospace: compute rotation magnitude
    /// use std::f32::consts::PI;
    /// let rotation = UnitQuaternion::from_euler_angles(PI / 4.0, 0.0, 0.0);
    /// let angle_degrees = rotation.angle().to_degrees();
    /// // angle_degrees ≈ 45 degrees
    ///
    /// // Identity rotation has angle 0
    /// let identity = UnitQuaternion::identity();
    /// assert_eq!(identity.angle(), 0.0);
    /// ```
    ///
    /// # See Also
    /// - [`axis`](Self::axis) - Get the rotation axis
    /// - [`axis_angle`](Self::axis_angle) - Get both axis and angle together
    /// - [`scaled_axis`](Self::scaled_axis) - Get axis scaled by angle
    /// - [`euler_angles`](Self::euler_angles) - Convert to Euler angles
    #[inline]
    #[must_use]
    pub fn angle(&self) -> T {
        let w = self.quaternion().scalar().simd_abs();
        self.quaternion().imag().norm().simd_atan2(w) * crate::convert(2.0f64)
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
    #[must_use]
    pub fn quaternion(&self) -> &Quaternion<T> {
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
    #[must_use = "Did you mean to use conjugate_mut()?"]
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
    #[must_use = "Did you mean to use inverse_mut()?"]
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
    #[must_use]
    pub fn angle_to(&self, other: &Self) -> T {
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
    #[must_use]
    pub fn rotation_to(&self, other: &Self) -> Self {
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
    #[must_use]
    pub fn lerp(&self, other: &Self, t: T) -> Quaternion<T> {
        self.as_ref().lerp(other.as_ref(), t)
    }

    /// Performs normalized linear interpolation (NLERP) between two unit quaternions.
    ///
    /// NLERP is a fast approximation of spherical linear interpolation (SLERP) that
    /// linearly interpolates between two quaternions and then normalizes the result.
    /// This is the recommended interpolation method for most real-time applications
    /// because it:
    /// - Is much faster than SLERP (no trigonometry)
    /// - Maintains unit length (unlike raw LERP)
    /// - Produces visually acceptable results for small angles
    /// - Never fails (unlike SLERP at 180° angles)
    ///
    /// The main drawback is that NLERP doesn't preserve constant angular velocity,
    /// meaning the rotation speed varies slightly during interpolation. For most
    /// game and graphics applications, this is imperceptible.
    ///
    /// # Parameters
    /// - `other`: The target quaternion to interpolate toward
    /// - `t`: Interpolation parameter in range [0, 1]
    ///   - `t = 0` returns `self`
    ///   - `t = 1` returns `other`
    ///   - `t = 0.5` returns halfway between
    ///
    /// # Returns
    /// A new unit quaternion that is the normalized linear interpolation.
    ///
    /// # Example
    /// ```
    /// # use nalgebra::{UnitQuaternion, Quaternion};
    /// let q1 = UnitQuaternion::new_normalize(Quaternion::new(1.0, 0.0, 0.0, 0.0));
    /// let q2 = UnitQuaternion::new_normalize(Quaternion::new(0.0, 1.0, 0.0, 0.0));
    /// let interpolated = q1.nlerp(&q2, 0.1);
    /// assert_eq!(interpolated, UnitQuaternion::new_normalize(Quaternion::new(0.9, 0.1, 0.0, 0.0)));
    ///
    /// // Game development: smooth camera rotation (recommended)
    /// let current_rotation = UnitQuaternion::identity();
    /// let target_rotation = UnitQuaternion::from_euler_angles(0.0, 1.57, 0.0);
    /// let smooth_rotation = current_rotation.nlerp(&target_rotation, 0.1);
    /// // Apply smooth_rotation to camera each frame for smooth transition
    ///
    /// // Aerospace: attitude interpolation for control systems
    /// let current_attitude = UnitQuaternion::from_euler_angles(0.1, 0.2, 0.3);
    /// let desired_attitude = UnitQuaternion::from_euler_angles(0.15, 0.25, 0.35);
    /// let dt = 0.016; // ~60 FPS
    /// let new_attitude = current_attitude.nlerp(&desired_attitude, dt * 5.0);
    ///
    /// // Animation: blend between keyframes
    /// let pose1 = UnitQuaternion::from_axis_angle(&nalgebra::Vector3::y_axis(), 0.0);
    /// let pose2 = UnitQuaternion::from_axis_angle(&nalgebra::Vector3::y_axis(), 1.57);
    /// let blended = pose1.nlerp(&pose2, 0.5); // 50% between poses
    /// ```
    ///
    /// # Performance Comparison
    /// ```text
    /// NLERP: ~2-3x faster than SLERP
    /// LERP:  Fastest but doesn't preserve unit length
    /// SLERP: Slowest but constant angular velocity
    /// ```
    ///
    /// # See Also
    /// - [`slerp`](Self::slerp) - Spherical interpolation (slower, constant velocity)
    /// - [`lerp`](Self::lerp) - Raw linear interpolation (doesn't normalize)
    /// - [`try_slerp`](Self::try_slerp) - Safe SLERP with fallback
    #[inline]
    #[must_use]
    pub fn nlerp(&self, other: &Self, t: T) -> Self {
        let mut res = self.lerp(other, t);
        let _ = res.normalize_mut();

        Self::new_unchecked(res)
    }

    /// Performs spherical linear interpolation (SLERP) between two unit quaternions.
    ///
    /// SLERP provides the smoothest possible interpolation between two rotations,
    /// maintaining constant angular velocity throughout the interpolation. This is
    /// the gold standard for rotation interpolation when quality is paramount.
    ///
    /// Unlike NLERP, SLERP travels along the shortest great circle arc on the
    /// 4D unit sphere, ensuring uniform rotation speed. However, it's more
    /// computationally expensive due to trigonometric calculations.
    ///
    /// **Important:** This function panics if the quaternions are approximately
    /// 180° apart, as the interpolation becomes ambiguous (infinitely many shortest
    /// paths). Use [`try_slerp`](Self::try_slerp) to handle this case safely.
    ///
    /// # Parameters
    /// - `other`: The target quaternion to interpolate toward
    /// - `t`: Interpolation parameter in range [0, 1]
    ///   - `t = 0` returns `self`
    ///   - `t = 1` returns `other`
    ///   - `t = 0.5` returns halfway between (on the great circle)
    ///
    /// # Returns
    /// A new unit quaternion that is the spherical linear interpolation.
    ///
    /// # Panics
    /// Panics if the angle between the quaternions is approximately 180°.
    ///
    /// # Example
    /// ```
    /// # use nalgebra::geometry::UnitQuaternion;
    /// let q1 = UnitQuaternion::from_euler_angles(std::f32::consts::FRAC_PI_4, 0.0, 0.0);
    /// let q2 = UnitQuaternion::from_euler_angles(-std::f32::consts::PI, 0.0, 0.0);
    ///
    /// let q = q1.slerp(&q2, 1.0 / 3.0);
    /// assert_eq!(q.euler_angles(), (std::f32::consts::FRAC_PI_2, 0.0, 0.0));
    ///
    /// // Game development: cinematic camera interpolation (high quality)
    /// let camera_start = UnitQuaternion::from_euler_angles(0.0, 0.0, 0.0);
    /// let camera_end = UnitQuaternion::from_euler_angles(0.0, 1.57, 0.0);
    /// let time = 0.5; // Halfway through animation
    /// let camera_rotation = camera_start.slerp(&camera_end, time);
    /// // Provides smooth, constant-speed rotation
    ///
    /// // Aerospace: high-precision attitude interpolation
    /// let attitude_t0 = UnitQuaternion::from_euler_angles(0.1, 0.2, 0.3);
    /// let attitude_t1 = UnitQuaternion::from_euler_angles(0.15, 0.25, 0.35);
    /// let interpolation_factor = 0.6;
    /// let current_attitude = attitude_t0.slerp(&attitude_t1, interpolation_factor);
    /// // Exact interpolation for navigation systems
    ///
    /// // Animation: keyframe interpolation for character joints
    /// let shoulder_rotation_start = UnitQuaternion::from_axis_angle(&nalgebra::Vector3::x_axis(), 0.0);
    /// let shoulder_rotation_end = UnitQuaternion::from_axis_angle(&nalgebra::Vector3::x_axis(), 1.0);
    /// let anim_time = 0.25;
    /// let current_rotation = shoulder_rotation_start.slerp(&shoulder_rotation_end, anim_time);
    /// ```
    ///
    /// # When to Use SLERP vs NLERP
    /// - **Use SLERP when:**
    ///   - Constant angular velocity is required
    ///   - High precision is needed (aerospace, robotics)
    ///   - Cinematic animations where quality matters
    ///   - Large rotation angles (>30°)
    ///
    /// - **Use NLERP when:**
    ///   - Real-time performance is critical
    ///   - Small rotation angles (<30°)
    ///   - Game character/camera interpolation
    ///   - Visual quality difference is imperceptible
    ///
    /// # See Also
    /// - [`try_slerp`](Self::try_slerp) - Safe version that doesn't panic
    /// - [`nlerp`](Self::nlerp) - Faster approximation (recommended for games)
    /// - [`lerp`](Self::lerp) - Fastest but doesn't maintain unit length
    #[inline]
    #[must_use]
    pub fn slerp(&self, other: &Self, t: T) -> Self
    where
        T: RealField,
    {
        self.try_slerp(other, t, T::default_epsilon())
            .expect("Quaternion slerp: ambiguous configuration.")
    }

    /// Safely computes spherical linear interpolation, returning `None` if undefined.
    ///
    /// This is the safe version of [`slerp`](Self::slerp) that handles the edge case
    /// where the quaternions are approximately 180° apart. When quaternions are near
    /// opposite orientations, there are infinitely many shortest paths between them,
    /// making SLERP undefined.
    ///
    /// This function returns `None` when the sine of the angle between the quaternions
    /// is smaller than `epsilon`, indicating they're too close to opposite. In practice,
    /// you should use [`nlerp`](Self::nlerp) as a fallback in this case.
    ///
    /// # Parameters
    /// - `other`: The target quaternion to interpolate toward
    /// - `t`: Interpolation parameter in range [0, 1]
    /// - `epsilon`: Threshold below which interpolation is considered undefined.
    ///   The sine of the angle between quaternions must be greater than this value.
    ///   Use `T::default_epsilon()` for a reasonable default.
    ///
    /// # Returns
    /// - `Some(quaternion)`: The interpolated quaternion
    /// - `None`: If the quaternions are approximately 180° apart
    ///
    /// # Example
    /// ```
    /// # use nalgebra::UnitQuaternion;
    /// let q1 = UnitQuaternion::from_euler_angles(0.0, 0.0, 0.0);
    /// let q2 = UnitQuaternion::from_euler_angles(0.5, 0.0, 0.0);
    ///
    /// // Normal case: interpolation succeeds
    /// let result = q1.try_slerp(&q2, 0.5, 1.0e-6);
    /// assert!(result.is_some());
    ///
    /// // Edge case: opposite quaternions (uncommon in practice)
    /// // This returns None, and you should fall back to nlerp
    ///
    /// // Game development: safe interpolation with fallback
    /// let current = UnitQuaternion::identity();
    /// let target = UnitQuaternion::from_euler_angles(0.0, 1.57, 0.0);
    /// let epsilon = 1.0e-6;
    ///
    /// let interpolated = current.try_slerp(&target, 0.1, epsilon)
    ///     .unwrap_or_else(|| current.nlerp(&target, 0.1));
    /// // Falls back to nlerp if slerp is undefined
    ///
    /// // Aerospace: robust attitude interpolation
    /// let att1 = UnitQuaternion::from_euler_angles(0.1, 0.2, 0.3);
    /// let att2 = UnitQuaternion::from_euler_angles(0.15, 0.25, 0.35);
    ///
    /// match att1.try_slerp(&att2, 0.5, 1.0e-10) {
    ///     Some(interpolated) => {
    ///         // Use high-precision SLERP result
    ///     }
    ///     None => {
    ///         // Handle ambiguous case - use alternative method
    ///         let fallback = att1.nlerp(&att2, 0.5);
    ///     }
    /// }
    /// ```
    ///
    /// # Best Practices
    /// ```rust,ignore
    /// // Recommended pattern for production code:
    /// fn interpolate_rotation(from: &UnitQuaternion<f32>,
    ///                        to: &UnitQuaternion<f32>,
    ///                        t: f32) -> UnitQuaternion<f32> {
    ///     from.try_slerp(to, t, 1.0e-6)
    ///         .unwrap_or_else(|| from.nlerp(to, t))
    /// }
    /// ```
    ///
    /// # See Also
    /// - [`slerp`](Self::slerp) - Panicking version (use when quaternions are known to be valid)
    /// - [`nlerp`](Self::nlerp) - Recommended fallback when try_slerp returns None
    /// - [`lerp`](Self::lerp) - Fastest interpolation (no normalization)
    #[inline]
    #[must_use]
    pub fn try_slerp(&self, other: &Self, t: T, epsilon: T) -> Option<Self>
    where
        T: RealField,
    {
        let coords = if self.coords.dot(&other.coords) < T::zero() {
            Unit::new_unchecked(self.coords.clone()).try_slerp(
                &Unit::new_unchecked(-other.coords.clone()),
                t,
                epsilon,
            )
        } else {
            Unit::new_unchecked(self.coords.clone()).try_slerp(
                &Unit::new_unchecked(other.coords.clone()),
                t,
                epsilon,
            )
        };

        coords.map(|q| Unit::new_unchecked(Quaternion::from(q.into_inner())))
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

    /// Returns the rotation axis of this unit quaternion, or `None` if the rotation angle is zero.
    ///
    /// A unit quaternion represents a rotation around an axis by a certain angle.
    /// This method extracts that axis as a unit vector. If the rotation angle is zero
    /// (identity rotation), there is no meaningful axis, so `None` is returned.
    ///
    /// The axis is normalized (has length 1) and represents the direction around which
    /// the rotation occurs, following the right-hand rule: if your thumb points along
    /// the axis, your fingers curl in the direction of rotation.
    ///
    /// # Returns
    /// - `Some(axis)`: A unit vector representing the rotation axis
    /// - `None`: If this is the identity rotation (angle = 0)
    ///
    /// # Example
    /// ```
    /// # use nalgebra::{UnitQuaternion, Vector3, Unit};
    /// let axis = Unit::new_normalize(Vector3::new(1.0, 2.0, 3.0));
    /// let angle = 1.2;
    /// let rot = UnitQuaternion::from_axis_angle(&axis, angle);
    /// assert_eq!(rot.axis(), Some(axis));
    ///
    /// // Identity rotation has no axis
    /// let identity = UnitQuaternion::identity();
    /// assert!(identity.axis().is_none());
    ///
    /// // Case with a zero angle
    /// let rot_zero = UnitQuaternion::from_axis_angle(&axis, 0.0);
    /// assert!(rot_zero.axis().is_none());
    ///
    /// // Game development: extract rotation information
    /// let player_rotation = UnitQuaternion::from_euler_angles(0.0, 1.57, 0.0);
    /// if let Some(axis) = player_rotation.axis() {
    ///     println!("Rotating around axis: {:?}", axis);
    /// }
    ///
    /// // Aerospace: get rotation axis for attitude control
    /// let spacecraft_rot = UnitQuaternion::from_axis_angle(&Vector3::z_axis(), 0.5);
    /// let control_axis = spacecraft_rot.axis().unwrap();
    /// // Use control_axis for thrust vectoring
    /// ```
    ///
    /// # See Also
    /// - [`angle`](Self::angle) - Get the rotation angle
    /// - [`axis_angle`](Self::axis_angle) - Get both axis and angle together
    /// - [`scaled_axis`](Self::scaled_axis) - Get axis scaled by the angle
    #[inline]
    #[must_use]
    pub fn axis(&self) -> Option<Unit<Vector3<T>>>
    where
        T: RealField,
    {
        let v = if self.quaternion().scalar() >= T::zero() {
            self.as_ref().vector().clone_owned()
        } else {
            -self.as_ref().vector()
        };

        Unit::try_new(v, T::zero())
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
    #[must_use]
    pub fn scaled_axis(&self) -> Vector3<T>
    where
        T: RealField,
    {
        match self.axis() {
            Some(axis) => axis.into_inner() * self.angle(),
            None => Vector3::zero(),
        }
    }

    /// Returns both the rotation axis and angle of this unit quaternion.
    ///
    /// This is a convenience method that combines [`axis`](Self::axis) and
    /// [`angle`](Self::angle) into a single call. It returns the axis-angle
    /// representation of the rotation, which is one of the most intuitive ways
    /// to think about 3D rotations.
    ///
    /// The angle is in radians and in the range `(0, π]`. If the rotation angle
    /// is zero (identity rotation), returns `None` since there's no meaningful axis.
    ///
    /// # Returns
    /// - `Some((axis, angle))`: The unit axis vector and rotation angle in radians
    /// - `None`: If this is the identity rotation
    ///
    /// # Example
    /// ```
    /// # use nalgebra::{UnitQuaternion, Vector3, Unit};
    /// let axis = Unit::new_normalize(Vector3::new(1.0, 2.0, 3.0));
    /// let angle = 1.2;
    /// let rot = UnitQuaternion::from_axis_angle(&axis, angle);
    /// assert_eq!(rot.axis_angle(), Some((axis, angle)));
    ///
    /// // Identity rotation returns None
    /// let identity = UnitQuaternion::identity();
    /// assert!(identity.axis_angle().is_none());
    ///
    /// // Case with a zero angle
    /// let rot_zero = UnitQuaternion::from_axis_angle(&axis, 0.0);
    /// assert!(rot_zero.axis_angle().is_none());
    ///
    /// // Game development: convert quaternion to axis-angle for debugging
    /// let camera_rotation = UnitQuaternion::from_euler_angles(0.5, 1.0, 0.0);
    /// if let Some((axis, angle)) = camera_rotation.axis_angle() {
    ///     println!("Camera rotated {:.2} radians around {:?}", angle, axis);
    /// }
    ///
    /// // Aerospace: extract rotation for control systems
    /// let attitude = UnitQuaternion::from_euler_angles(0.1, 0.2, 0.3);
    /// if let Some((correction_axis, error_angle)) = attitude.axis_angle() {
    ///     if error_angle > 0.05 {
    ///         // Apply corrective torque along correction_axis
    ///     }
    /// }
    /// ```
    ///
    /// # See Also
    /// - [`axis`](Self::axis) - Get only the rotation axis
    /// - [`angle`](Self::angle) - Get only the rotation angle
    /// - [`scaled_axis`](Self::scaled_axis) - Get axis pre-multiplied by angle
    /// - [`from_axis_angle`](Self::from_axis_angle) - Create quaternion from axis-angle
    #[inline]
    #[must_use]
    pub fn axis_angle(&self) -> Option<(Unit<Vector3<T>>, T)>
    where
        T: RealField,
    {
        self.axis().map(|axis| (axis, self.angle()))
    }

    /// Compute the exponential of a quaternion.
    ///
    /// Note that this function yields a `Quaternion<T>` because it loses the unit property.
    #[inline]
    #[must_use]
    pub fn exp(&self) -> Quaternion<T> {
        self.as_ref().exp()
    }

    /// Compute the natural logarithm of a quaternion.
    ///
    /// Note that this function yields a `Quaternion<T>` because it loses the unit property.
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
    #[must_use]
    pub fn ln(&self) -> Quaternion<T>
    where
        T: RealField,
    {
        match self.axis() {
            Some(v) => Quaternion::from_imag(v.into_inner() * self.angle()),
            None => Quaternion::zero(),
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
    #[must_use]
    pub fn powf(&self, n: T) -> Self
    where
        T: RealField,
    {
        match self.axis() {
            Some(v) => Self::from_axis_angle(&v, self.angle() * n),
            None => Self::identity(),
        }
    }

    /// Builds a rotation matrix from this unit quaternion.
    ///
    /// # Example
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
    #[must_use]
    pub fn to_rotation_matrix(self) -> Rotation<T, 3> {
        let i = self.as_ref()[0].clone();
        let j = self.as_ref()[1].clone();
        let k = self.as_ref()[2].clone();
        let w = self.as_ref()[3].clone();

        let ww = w.clone() * w.clone();
        let ii = i.clone() * i.clone();
        let jj = j.clone() * j.clone();
        let kk = k.clone() * k.clone();
        let ij = i.clone() * j.clone() * crate::convert(2.0f64);
        let wk = w.clone() * k.clone() * crate::convert(2.0f64);
        let wj = w.clone() * j.clone() * crate::convert(2.0f64);
        let ik = i.clone() * k.clone() * crate::convert(2.0f64);
        let jk = j * k * crate::convert(2.0f64);
        let wi = w * i * crate::convert(2.0f64);

        Rotation::from_matrix_unchecked(Matrix3::new(
            ww.clone() + ii.clone() - jj.clone() - kk.clone(),
            ij.clone() - wk.clone(),
            wj.clone() + ik.clone(),
            wk + ij,
            ww.clone() - ii.clone() + jj.clone() - kk.clone(),
            jk.clone() - wi.clone(),
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
    pub fn to_euler_angles(self) -> (T, T, T)
    where
        T: RealField,
    {
        self.euler_angles()
    }

    /// Extracts the Euler angles corresponding to this unit quaternion.
    ///
    /// Converts the quaternion rotation into three sequential rotations around the
    /// coordinate axes. The angles are returned in the order (roll, pitch, yaw), which
    /// corresponds to rotations around the X, Y, and Z axes respectively:
    /// 1. **Roll** (φ): Rotation around X-axis (affects tilting left/right)
    /// 2. **Pitch** (θ): Rotation around Y-axis (affects tilting up/down)
    /// 3. **Yaw** (ψ): Rotation around Z-axis (affects turning left/right)
    ///
    /// All angles are in radians. The rotation order is: first roll, then pitch, then yaw.
    ///
    /// **Warning:** Euler angles suffer from gimbal lock when pitch is ±π/2 (±90°),
    /// where roll and yaw become ambiguous. For smooth interpolation and to avoid
    /// gimbal lock, use quaternions directly instead of converting to Euler angles.
    ///
    /// # Returns
    /// A tuple `(roll, pitch, yaw)` in radians.
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::UnitQuaternion;
    /// let rot = UnitQuaternion::from_euler_angles(0.1, 0.2, 0.3);
    /// let (roll, pitch, yaw) = rot.euler_angles();
    /// assert_relative_eq!(roll, 0.1, epsilon = 1.0e-6);
    /// assert_relative_eq!(pitch, 0.2, epsilon = 1.0e-6);
    /// assert_relative_eq!(yaw, 0.3, epsilon = 1.0e-6);
    ///
    /// // Game development: extract camera angles
    /// let camera_quat = UnitQuaternion::from_euler_angles(0.5, 1.0, 0.0);
    /// let (roll, pitch, yaw) = camera_quat.euler_angles();
    /// println!("Camera: roll={:.2}, pitch={:.2}, yaw={:.2}", roll, pitch, yaw);
    ///
    /// // Aerospace: convert spacecraft attitude to intuitive angles
    /// use std::f32::consts::PI;
    /// let attitude = UnitQuaternion::from_axis_angle(&nalgebra::Vector3::y_axis(), PI/4.0);
    /// let (roll, pitch, yaw) = attitude.euler_angles();
    /// let yaw_degrees = yaw.to_degrees();
    /// // yaw_degrees ≈ 45 degrees (rotation around Y-axis)
    /// ```
    ///
    /// # Gimbal Lock Example
    /// ```
    /// # use nalgebra::UnitQuaternion;
    /// # use std::f32::consts::PI;
    /// // When pitch is ±90°, gimbal lock occurs
    /// let gimbal_lock = UnitQuaternion::from_euler_angles(0.5, PI/2.0, 0.3);
    /// let (roll, pitch, yaw) = gimbal_lock.euler_angles();
    /// // roll and yaw may not match input due to ambiguity
    /// ```
    ///
    /// # See Also
    /// - [`from_euler_angles`](Self::from_euler_angles) - Create quaternion from Euler angles
    /// - [`to_rotation_matrix`](Self::to_rotation_matrix) - Convert to rotation matrix
    /// - [`axis_angle`](Self::axis_angle) - Alternative rotation representation
    #[inline]
    #[must_use]
    pub fn euler_angles(&self) -> (T, T, T)
    where
        T: RealField,
    {
        self.clone().to_rotation_matrix().euler_angles()
    }

    /// Converts this unit quaternion into its equivalent homogeneous transformation matrix.
    ///
    /// # Example
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
    #[must_use]
    pub fn to_homogeneous(self) -> Matrix4<T> {
        self.to_rotation_matrix().to_homogeneous()
    }

    /// Rotates a point in 3D space by this unit quaternion.
    ///
    /// Applies the rotation represented by this quaternion to a point, returning
    /// the rotated point. This is the fundamental operation for transforming
    /// coordinates between different reference frames.
    ///
    /// This method is equivalent to the multiplication `self * pt` but is provided
    /// for clarity and semantic meaning. The rotation is performed around the origin.
    ///
    /// # Parameters
    /// - `pt`: The point to rotate
    ///
    /// # Returns
    /// The rotated point.
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use std::f32;
    /// # use nalgebra::{UnitQuaternion, Vector3, Point3};
    /// // 90-degree rotation around Y-axis
    /// let rot = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), f32::consts::FRAC_PI_2);
    /// let point = Point3::new(1.0, 2.0, 3.0);
    /// let rotated = rot.transform_point(&point);
    ///
    /// // X and Z coordinates swap and Z negates
    /// assert_relative_eq!(rotated, Point3::new(3.0, 2.0, -1.0), epsilon = 1.0e-6);
    ///
    /// // Game development: rotate object positions
    /// let player_pos = Point3::new(10.0, 0.0, 0.0);
    /// let camera_rotation = UnitQuaternion::from_euler_angles(0.0, 1.57, 0.0);
    /// let world_pos = camera_rotation.transform_point(&player_pos);
    ///
    /// // Aerospace: transform coordinates between reference frames
    /// let body_point = Point3::new(1.0, 0.0, 0.0);
    /// let attitude = UnitQuaternion::from_euler_angles(0.1, 0.2, 0.3);
    /// let inertial_point = attitude.transform_point(&body_point);
    /// ```
    ///
    /// # See Also
    /// - [`transform_vector`](Self::transform_vector) - Rotate a vector (direction)
    /// - [`inverse_transform_point`](Self::inverse_transform_point) - Apply inverse rotation
    /// - [`Mul<Point3>`](std::ops::Mul) - Operator form: `rotation * point`
    #[inline]
    #[must_use]
    pub fn transform_point(&self, pt: &Point3<T>) -> Point3<T> {
        self * pt
    }

    /// Rotates a vector by this unit quaternion.
    ///
    /// Applies the rotation represented by this quaternion to a vector, returning
    /// the rotated vector. Unlike points, vectors represent directions or displacements
    /// and are not affected by translation (though quaternions don't include translation).
    ///
    /// This method is equivalent to the multiplication `self * v` but is provided
    /// for clarity. It's commonly used to rotate directions like forward vectors,
    /// velocity, or force vectors.
    ///
    /// # Parameters
    /// - `v`: The vector to rotate
    ///
    /// # Returns
    /// The rotated vector.
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use std::f32;
    /// # use nalgebra::{UnitQuaternion, Vector3};
    /// // 90-degree rotation around Y-axis
    /// let rot = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), f32::consts::FRAC_PI_2);
    /// let vector = Vector3::new(1.0, 2.0, 3.0);
    /// let rotated = rot.transform_vector(&vector);
    ///
    /// assert_relative_eq!(rotated, Vector3::new(3.0, 2.0, -1.0), epsilon = 1.0e-6);
    ///
    /// // Game development: rotate forward vector
    /// let forward = Vector3::new(0.0, 0.0, 1.0);
    /// let player_rotation = UnitQuaternion::from_euler_angles(0.0, 1.57, 0.0);
    /// let actual_forward = player_rotation.transform_vector(&forward);
    /// // Player is now facing 90° to the left
    ///
    /// // Aerospace: rotate velocity vector
    /// let local_velocity = Vector3::new(100.0, 0.0, 0.0);
    /// let spacecraft_attitude = UnitQuaternion::from_axis_angle(&Vector3::z_axis(), 0.5);
    /// let world_velocity = spacecraft_attitude.transform_vector(&local_velocity);
    ///
    /// // Physics: rotate force application
    /// let local_thrust = Vector3::new(0.0, 10.0, 0.0);
    /// let orientation = UnitQuaternion::from_euler_angles(0.3, 0.0, 0.0);
    /// let world_thrust = orientation.transform_vector(&local_thrust);
    /// ```
    ///
    /// # See Also
    /// - [`transform_point`](Self::transform_point) - Rotate a point (position)
    /// - [`inverse_transform_vector`](Self::inverse_transform_vector) - Apply inverse rotation
    /// - [`Mul<Vector3>`](std::ops::Mul) - Operator form: `rotation * vector`
    #[inline]
    #[must_use]
    pub fn transform_vector(&self, v: &Vector3<T>) -> Vector3<T> {
        self * v
    }

    /// Rotate a point by the inverse of this unit quaternion. This may be
    /// cheaper than inverting the unit quaternion and transforming the
    /// point.
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use std::f32;
    /// # use nalgebra::{UnitQuaternion, Vector3, Point3};
    /// let rot = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), f32::consts::FRAC_PI_2);
    /// let transformed_point = rot.inverse_transform_point(&Point3::new(1.0, 2.0, 3.0));
    ///
    /// assert_relative_eq!(transformed_point, Point3::new(-3.0, 2.0, 1.0), epsilon = 1.0e-6);
    /// ```
    #[inline]
    #[must_use]
    pub fn inverse_transform_point(&self, pt: &Point3<T>) -> Point3<T> {
        // TODO: would it be useful performance-wise not to call inverse explicitly (i-e. implement
        // the inverse transformation explicitly here) ?
        self.inverse() * pt
    }

    /// Rotate a vector by the inverse of this unit quaternion. This may be
    /// cheaper than inverting the unit quaternion and transforming the
    /// vector.
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use std::f32;
    /// # use nalgebra::{UnitQuaternion, Vector3};
    /// let rot = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), f32::consts::FRAC_PI_2);
    /// let transformed_vector = rot.inverse_transform_vector(&Vector3::new(1.0, 2.0, 3.0));
    ///
    /// assert_relative_eq!(transformed_vector, Vector3::new(-3.0, 2.0, 1.0), epsilon = 1.0e-6);
    /// ```
    #[inline]
    #[must_use]
    pub fn inverse_transform_vector(&self, v: &Vector3<T>) -> Vector3<T> {
        self.inverse() * v
    }

    /// Rotate a vector by the inverse of this unit quaternion. This may be
    /// cheaper than inverting the unit quaternion and transforming the
    /// vector.
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use std::f32;
    /// # use nalgebra::{UnitQuaternion, Vector3};
    /// let rot = UnitQuaternion::from_axis_angle(&Vector3::z_axis(), f32::consts::FRAC_PI_2);
    /// let transformed_vector = rot.inverse_transform_unit_vector(&Vector3::x_axis());
    ///
    /// assert_relative_eq!(transformed_vector, -Vector3::y_axis(), epsilon = 1.0e-6);
    /// ```
    #[inline]
    #[must_use]
    pub fn inverse_transform_unit_vector(&self, v: &Unit<Vector3<T>>) -> Unit<Vector3<T>> {
        self.inverse() * v
    }

    /// Appends to `self` a rotation given in the axis-angle form, using a linearized formulation.
    ///
    /// This is faster, but approximate, way to compute `UnitQuaternion::new(axisangle) * self`.
    #[inline]
    #[must_use]
    pub fn append_axisangle_linearized(&self, axisangle: &Vector3<T>) -> Self {
        let half: T = crate::convert(0.5);
        let q1 = self.clone().into_inner();
        let q2 = Quaternion::from_imag(axisangle * half);
        Unit::new_normalize(&q1 + q2 * &q1)
    }
}

impl<T: RealField> Default for UnitQuaternion<T> {
    fn default() -> Self {
        Self::identity()
    }
}

impl<T: RealField + fmt::Display> fmt::Display for UnitQuaternion<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.axis() {
            Some(axis) => {
                let axis = axis.into_inner();
                write!(
                    f,
                    "UnitQuaternion angle: {} − axis: ({}, {}, {})",
                    self.angle(),
                    axis[0],
                    axis[1],
                    axis[2]
                )
            }
            None => {
                write!(
                    f,
                    "UnitQuaternion angle: {} − axis: (undefined)",
                    self.angle()
                )
            }
        }
    }
}

impl<T: RealField + AbsDiffEq<Epsilon = T>> AbsDiffEq for UnitQuaternion<T> {
    type Epsilon = T;

    #[inline]
    fn default_epsilon() -> Self::Epsilon {
        T::default_epsilon()
    }

    #[inline]
    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        self.as_ref().abs_diff_eq(other.as_ref(), epsilon)
    }
}

impl<T: RealField + RelativeEq<Epsilon = T>> RelativeEq for UnitQuaternion<T> {
    #[inline]
    fn default_max_relative() -> Self::Epsilon {
        T::default_max_relative()
    }

    #[inline]
    fn relative_eq(
        &self,
        other: &Self,
        epsilon: Self::Epsilon,
        max_relative: Self::Epsilon,
    ) -> bool {
        self.as_ref()
            .relative_eq(other.as_ref(), epsilon, max_relative)
    }
}

impl<T: RealField + UlpsEq<Epsilon = T>> UlpsEq for UnitQuaternion<T> {
    #[inline]
    fn default_max_ulps() -> u32 {
        T::default_max_ulps()
    }

    #[inline]
    fn ulps_eq(&self, other: &Self, epsilon: Self::Epsilon, max_ulps: u32) -> bool {
        self.as_ref().ulps_eq(other.as_ref(), epsilon, max_ulps)
    }
}
