// Needed otherwise the rkyv macros generate code incompatible with rust-2024
#![cfg_attr(feature = "rkyv-serialize", allow(unsafe_op_in_unsafe_fn))]
// The macros break if the references are taken out, for some reason.
#![allow(clippy::op_ref)]

#[cfg(feature = "rkyv-serialize")]
use rkyv::bytecheck;

use crate::{
    Isometry3, Matrix4, Normed, OVector, Point3, Quaternion, Scalar, SimdRealField, Translation3,
    U8, Unit, UnitQuaternion, Vector3, Zero,
};
use approx::{AbsDiffEq, RelativeEq, UlpsEq};
#[cfg(feature = "serde-serialize-no-std")]
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::fmt;

use simba::scalar::{ClosedNeg, RealField};

/// A dual quaternion.
///
/// # What are Dual Quaternions?
///
/// Dual quaternions are a mathematical representation that can efficiently encode both
/// rotation and translation in 3D space. They are particularly useful in computer graphics,
/// robotics, and physics simulations because they:
///
/// - Represent rigid body transformations (rotation + translation) compactly
/// - Allow smooth interpolation between transformations (important for animation)
/// - Avoid gimbal lock issues present in Euler angles
/// - Are more efficient than 4x4 transformation matrices for certain operations
///
/// A dual quaternion consists of two quaternions: a "real" part and a "dual" part.
/// For representing rigid body transformations, the real part encodes the rotation,
/// while the dual part encodes the translation.
///
/// # Common Use Cases
///
/// - **Character Skinning**: Smooth blending between bone transformations in skeletal animation
/// - **Rigid Body Physics**: Representing the position and orientation of objects
/// - **Robotics**: Describing robot joint configurations and end-effector poses
/// - **Camera Animation**: Smoothly interpolating between camera positions
///
/// # Indexing
///
/// `DualQuaternions` are stored as \[..real, ..dual\].
/// Both of the quaternion components are laid out in `i, j, k, w` order.
///
/// # Example
/// ```
/// # use nalgebra::{DualQuaternion, Quaternion};
///
/// let real = Quaternion::new(1.0, 2.0, 3.0, 4.0);
/// let dual = Quaternion::new(5.0, 6.0, 7.0, 8.0);
///
/// let dq = DualQuaternion::from_real_and_dual(real, dual);
/// assert_eq!(dq[0], 2.0);
/// assert_eq!(dq[1], 3.0);
///
/// assert_eq!(dq[4], 6.0);
/// assert_eq!(dq[7], 5.0);
/// ```
///
/// NOTE:
///  As of December 2020, dual quaternion support is a work in progress.
///  If a feature that you need is missing, feel free to open an issue or a PR.
///  See <https://github.com/dimforge/nalgebra/issues/487>
#[repr(C)]
#[derive(Debug, Copy, Clone)]
#[cfg_attr(
    feature = "rkyv-serialize-no-std",
    derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize),
    archive(
        as = "DualQuaternion<T::Archived>",
        bound(archive = "
        T: rkyv::Archive,
        Quaternion<T>: rkyv::Archive<Archived = Quaternion<T::Archived>>
    ")
    )
)]
#[cfg_attr(feature = "rkyv-serialize", derive(bytecheck::CheckBytes))]
#[cfg_attr(feature = "defmt", derive(defmt::Format))]
pub struct DualQuaternion<T> {
    /// The real component of the quaternion
    pub real: Quaternion<T>,
    /// The dual component of the quaternion
    pub dual: Quaternion<T>,
}

impl<T: Scalar + Eq> Eq for DualQuaternion<T> {}

impl<T: Scalar> PartialEq for DualQuaternion<T> {
    #[inline]
    fn eq(&self, right: &Self) -> bool {
        self.real == right.real && self.dual == right.dual
    }
}

impl<T: Scalar + Zero> Default for DualQuaternion<T> {
    fn default() -> Self {
        Self {
            real: Quaternion::default(),
            dual: Quaternion::default(),
        }
    }
}

impl<T: SimdRealField> DualQuaternion<T>
where
    T::Element: SimdRealField,
{
    /// Normalizes this dual quaternion.
    ///
    /// Normalization ensures that the real part of the dual quaternion has unit length.
    /// This is important for dual quaternions representing rigid body transformations,
    /// as only normalized dual quaternions correctly represent rotations and translations.
    ///
    /// The normalization divides both the real and dual parts by the norm of the real part.
    /// After normalization, the real part will have a norm of 1.0.
    ///
    /// # Returns
    ///
    /// A new normalized dual quaternion. This method does not modify the original.
    ///
    /// # Example
    ///
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{DualQuaternion, Quaternion};
    /// // Create a dual quaternion with non-unit real part
    /// let real = Quaternion::new(1.0, 2.0, 3.0, 4.0);
    /// let dual = Quaternion::new(5.0, 6.0, 7.0, 8.0);
    /// let dq = DualQuaternion::from_real_and_dual(real, dual);
    ///
    /// // Normalize it
    /// let dq_normalized = dq.normalize();
    ///
    /// // The real part now has unit length
    /// assert_relative_eq!(dq_normalized.real.norm(), 1.0);
    /// ```
    ///
    /// # Use Case: Character Animation
    ///
    /// ```
    /// # use nalgebra::{DualQuaternion, UnitDualQuaternion, UnitQuaternion, Vector3};
    /// // When blending between bone transformations in character skinning,
    /// // the result needs to be normalized
    /// let bone1 = UnitDualQuaternion::from_parts(
    ///     Vector3::new(1.0, 0.0, 0.0).into(),
    ///     UnitQuaternion::from_euler_angles(0.5, 0.0, 0.0)
    /// );
    /// let bone2 = UnitDualQuaternion::from_parts(
    ///     Vector3::new(2.0, 0.0, 0.0).into(),
    ///     UnitQuaternion::from_euler_angles(1.0, 0.0, 0.0)
    /// );
    ///
    /// // Blend between transformations (linear interpolation)
    /// let weight = 0.5;
    /// let blended = bone1.lerp(&bone2, weight);
    ///
    /// // Normalize the result for use
    /// let blended_normalized = blended.normalize();
    /// ```
    ///
    /// # See Also
    ///
    /// - [`normalize_mut`](Self::normalize_mut) - In-place version of this method
    /// - [`UnitDualQuaternion::new_normalize`] - Creates a unit dual quaternion by normalizing
    #[inline]
    #[must_use = "Did you mean to use normalize_mut()?"]
    pub fn normalize(&self) -> Self {
        let real_norm = self.real.norm();

        Self::from_real_and_dual(
            self.real.clone() / real_norm.clone(),
            self.dual.clone() / real_norm,
        )
    }

    /// Normalizes this dual quaternion in-place.
    ///
    /// This method modifies the dual quaternion directly, dividing both the real and dual
    /// parts by the norm of the real part. After normalization, the real part will have
    /// a norm of 1.0.
    ///
    /// This is more efficient than [`normalize`](Self::normalize) when you don't need
    /// to keep the original unnormalized value.
    ///
    /// # Returns
    ///
    /// The norm of the real part before normalization.
    ///
    /// # Example
    ///
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{DualQuaternion, Quaternion};
    /// let real = Quaternion::new(1.0, 2.0, 3.0, 4.0);
    /// let dual = Quaternion::new(5.0, 6.0, 7.0, 8.0);
    /// let mut dq = DualQuaternion::from_real_and_dual(real, dual);
    ///
    /// let old_norm = dq.normalize_mut();
    ///
    /// assert_relative_eq!(dq.real.norm(), 1.0);
    /// assert_relative_eq!(old_norm, 5.477226); // sqrt(1^2 + 2^2 + 3^2 + 4^2)
    /// ```
    ///
    /// # Use Case: Accumulating Transformations
    ///
    /// ```
    /// # use nalgebra::{UnitDualQuaternion, UnitQuaternion, Vector3, DualQuaternion};
    /// // In a physics simulation, you might accumulate small transformations
    /// // and periodically normalize to prevent numerical drift
    /// let mut current_transform = UnitDualQuaternion::identity();
    ///
    /// for _ in 0..100 {
    ///     // Apply small incremental transformation
    ///     let delta = UnitDualQuaternion::from_parts(
    ///         Vector3::new(0.01, 0.0, 0.0).into(),
    ///         UnitQuaternion::from_euler_angles(0.01, 0.0, 0.0)
    ///     );
    ///
    ///     // Multiply transformations
    ///     let mut new_transform = (current_transform * delta).into_inner();
    ///
    ///     // Normalize in-place to prevent drift
    ///     new_transform.normalize_mut();
    ///     current_transform = UnitDualQuaternion::new_unchecked(new_transform);
    /// }
    /// ```
    ///
    /// # See Also
    ///
    /// - [`normalize`](Self::normalize) - Non-mutating version that returns a new dual quaternion
    /// - [`UnitDualQuaternion::new_normalize`] - Creates a unit dual quaternion by normalizing
    #[inline]
    pub fn normalize_mut(&mut self) -> T {
        let real_norm = self.real.norm();
        self.real /= real_norm.clone();
        self.dual /= real_norm.clone();
        real_norm
    }

    /// The conjugate of this dual quaternion.
    ///
    /// The conjugate negates the vector (imaginary) parts of both the real and dual
    /// quaternions while keeping the scalar (real) parts unchanged. For a quaternion
    /// q = w + xi + yj + zk, its conjugate is q* = w - xi - yj - zk.
    ///
    /// For a unit dual quaternion representing a rigid body transformation, the conjugate
    /// represents the inverse transformation (rotation and translation reversed).
    ///
    /// # Returns
    ///
    /// A new dual quaternion with conjugated real and dual parts.
    ///
    /// # Example
    ///
    /// ```
    /// # use nalgebra::{DualQuaternion, Quaternion};
    /// let real = Quaternion::new(1.0, 2.0, 3.0, 4.0);
    /// let dual = Quaternion::new(5.0, 6.0, 7.0, 8.0);
    /// let dq = DualQuaternion::from_real_and_dual(real, dual);
    ///
    /// let conj = dq.conjugate();
    ///
    /// // Vector parts are negated
    /// assert!(conj.real.i == -2.0 && conj.real.j == -3.0 && conj.real.k == -4.0);
    /// // Scalar parts remain the same
    /// assert!(conj.real.w == 1.0);
    /// assert!(conj.dual.i == -6.0 && conj.dual.j == -7.0 && conj.dual.k == -8.0);
    /// assert!(conj.dual.w == 5.0);
    /// ```
    ///
    /// # Use Case: Inverse Transformation
    ///
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{UnitDualQuaternion, UnitQuaternion, Vector3, Point3};
    /// // Create a transformation
    /// let transform = UnitDualQuaternion::from_parts(
    ///     Vector3::new(1.0, 2.0, 3.0).into(),
    ///     UnitQuaternion::from_euler_angles(0.5, 0.0, 0.0)
    /// );
    ///
    /// // Apply transformation to a point
    /// let point = Point3::new(1.0, 0.0, 0.0);
    /// let transformed = transform * point;
    ///
    /// // Use conjugate to reverse the transformation
    /// let conjugated = transform.conjugate();
    /// let reversed = conjugated * transformed;
    ///
    /// // We get back the original point
    /// assert_relative_eq!(reversed, point, epsilon = 1.0e-6);
    /// ```
    ///
    /// # See Also
    ///
    /// - [`conjugate_mut`](Self::conjugate_mut) - In-place version of this method
    /// - [`inverse`](UnitDualQuaternion::inverse) - For unit dual quaternions, provides the proper inverse
    /// - [`try_inverse`](Self::try_inverse) - Safe inverse that handles non-unit dual quaternions
    #[inline]
    #[must_use = "Did you mean to use conjugate_mut()?"]
    pub fn conjugate(&self) -> Self {
        Self::from_real_and_dual(self.real.conjugate(), self.dual.conjugate())
    }

    /// Replaces this dual quaternion by its conjugate in-place.
    ///
    /// This method modifies the dual quaternion directly, negating the vector (imaginary)
    /// parts of both the real and dual quaternions while keeping the scalar parts unchanged.
    ///
    /// This is more efficient than [`conjugate`](Self::conjugate) when you don't need
    /// to keep the original value.
    ///
    /// # Example
    ///
    /// ```
    /// # use nalgebra::{DualQuaternion, Quaternion};
    /// let real = Quaternion::new(1.0, 2.0, 3.0, 4.0);
    /// let dual = Quaternion::new(5.0, 6.0, 7.0, 8.0);
    /// let mut dq = DualQuaternion::from_real_and_dual(real, dual);
    ///
    /// dq.conjugate_mut();
    ///
    /// // Vector parts are negated
    /// assert!(dq.real.i == -2.0 && dq.real.j == -3.0 && dq.real.k == -4.0);
    /// // Scalar parts remain the same
    /// assert!(dq.real.w == 1.0);
    /// assert!(dq.dual.i == -6.0 && dq.dual.j == -7.0 && dq.dual.k == -8.0);
    /// assert!(dq.dual.w == 5.0);
    /// ```
    ///
    /// # See Also
    ///
    /// - [`conjugate`](Self::conjugate) - Non-mutating version that returns a new dual quaternion
    /// - [`inverse_mut`](UnitDualQuaternion::inverse_mut) - In-place inverse for unit dual quaternions
    #[inline]
    pub fn conjugate_mut(&mut self) {
        self.real.conjugate_mut();
        self.dual.conjugate_mut();
    }

    /// Inverts this dual quaternion if it is not zero.
    ///
    /// The inverse of a dual quaternion dq is computed such that `dq * dq.inverse() = identity`.
    /// For a dual quaternion representing a rigid body transformation, the inverse represents
    /// the opposite transformation (moving back to the original position and orientation).
    ///
    /// A dual quaternion can be inverted only if its real part is non-zero. If the real part
    /// is zero (or very close to zero), the dual quaternion cannot be inverted and this
    /// method returns `None`.
    ///
    /// # Returns
    ///
    /// - `Some(inverse)` if the dual quaternion can be inverted
    /// - `None` if the real part is zero (non-invertible case)
    ///
    /// # Example
    ///
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{DualQuaternion, Quaternion};
    /// // Invertible dual quaternion
    /// let real = Quaternion::new(1.0, 2.0, 3.0, 4.0);
    /// let dual = Quaternion::new(5.0, 6.0, 7.0, 8.0);
    /// let dq = DualQuaternion::from_real_and_dual(real, dual);
    /// let inverse = dq.try_inverse();
    ///
    /// assert!(inverse.is_some());
    /// assert_relative_eq!(inverse.unwrap() * dq, DualQuaternion::identity());
    ///
    /// // Non-invertible case: zero dual quaternion
    /// let zero = Quaternion::new(0.0, 0.0, 0.0, 0.0);
    /// let dq = DualQuaternion::from_real_and_dual(zero, zero);
    /// let inverse = dq.try_inverse();
    ///
    /// assert!(inverse.is_none());
    /// ```
    ///
    /// # Use Case: Robotic Arm Inverse Kinematics
    ///
    /// ```
    /// # use nalgebra::{UnitDualQuaternion, UnitQuaternion, Vector3, Point3};
    /// // In robotics, you might need to find the inverse of a joint transformation
    /// // to convert from end-effector pose back to joint configuration
    /// let joint_transform = UnitDualQuaternion::from_parts(
    ///     Vector3::new(0.0, 0.0, 1.0).into(), // Joint position
    ///     UnitQuaternion::from_euler_angles(0.0, 0.0, 1.57) // Joint rotation
    /// );
    ///
    /// // Compute the inverse to transform world coordinates to joint-local coordinates
    /// let inverse_transform = joint_transform.inverse();
    ///
    /// // Transform a point from world space to joint-local space
    /// let world_point = Point3::new(1.0, 1.0, 1.0);
    /// let local_point = inverse_transform * world_point;
    /// ```
    ///
    /// # See Also
    ///
    /// - [`try_inverse_mut`](Self::try_inverse_mut) - In-place version of this method
    /// - [`inverse`](UnitDualQuaternion::inverse) - Infallible inverse for unit dual quaternions
    /// - [`conjugate`](Self::conjugate) - For unit dual quaternions, conjugate equals inverse
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

    /// Inverts this dual quaternion in-place if it is not zero.
    ///
    /// This method modifies the dual quaternion directly, computing its inverse.
    /// This is more efficient than [`try_inverse`](Self::try_inverse) when you don't
    /// need to keep the original value.
    ///
    /// # Returns
    ///
    /// - `true` if the inversion was successful
    /// - `false` if the dual quaternion cannot be inverted (real part is zero)
    ///
    /// # Example
    ///
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{DualQuaternion, Quaternion};
    /// // Invertible case
    /// let real = Quaternion::new(1.0, 2.0, 3.0, 4.0);
    /// let dual = Quaternion::new(5.0, 6.0, 7.0, 8.0);
    /// let dq = DualQuaternion::from_real_and_dual(real, dual);
    /// let mut dq_inverse = dq;
    /// let success = dq_inverse.try_inverse_mut();
    ///
    /// assert!(success);
    /// assert_relative_eq!(dq_inverse * dq, DualQuaternion::identity());
    ///
    /// // Non-invertible case
    /// let zero = Quaternion::new(0.0, 0.0, 0.0, 0.0);
    /// let mut dq = DualQuaternion::from_real_and_dual(zero, zero);
    /// let success = dq.try_inverse_mut();
    /// assert!(!success);
    /// ```
    ///
    /// # See Also
    ///
    /// - [`try_inverse`](Self::try_inverse) - Non-mutating version that returns an Option
    /// - [`inverse_mut`](UnitDualQuaternion::inverse_mut) - In-place inverse for unit dual quaternions
    #[inline]
    pub fn try_inverse_mut(&mut self) -> bool
    where
        T: RealField,
    {
        let inverted = self.real.try_inverse_mut();
        if inverted {
            self.dual = -self.real.clone() * self.dual.clone() * self.real.clone();
            true
        } else {
            false
        }
    }

    /// Linear interpolation between two dual quaternions.
    ///
    /// Computes `self * (1 - t) + other * t`, performing component-wise linear interpolation
    /// between the two dual quaternions. The parameter `t` controls the interpolation:
    /// - When `t = 0.0`, returns a value equal to `self`
    /// - When `t = 1.0`, returns a value equal to `other`
    /// - When `t = 0.5`, returns the midpoint between `self` and `other`
    ///
    /// **Note**: The result is NOT normalized. For unit dual quaternions representing
    /// transformations, you should use [`nlerp`](UnitDualQuaternion::nlerp) or
    /// [`sclerp`](UnitDualQuaternion::sclerp) instead.
    ///
    /// # Arguments
    ///
    /// - `other`: The target dual quaternion to interpolate toward
    /// - `t`: The interpolation parameter, typically between 0.0 and 1.0
    ///
    /// # Example
    ///
    /// ```
    /// # use nalgebra::{DualQuaternion, Quaternion};
    /// let dq1 = DualQuaternion::from_real_and_dual(
    ///     Quaternion::new(1.0, 0.0, 0.0, 4.0),
    ///     Quaternion::new(0.0, 2.0, 0.0, 0.0)
    /// );
    /// let dq2 = DualQuaternion::from_real_and_dual(
    ///     Quaternion::new(2.0, 0.0, 1.0, 0.0),
    ///     Quaternion::new(0.0, 2.0, 0.0, 0.0)
    /// );
    ///
    /// // Interpolate 25% of the way from dq1 to dq2
    /// let result = dq1.lerp(&dq2, 0.25);
    ///
    /// assert_eq!(result, DualQuaternion::from_real_and_dual(
    ///     Quaternion::new(1.25, 0.0, 0.25, 3.0),
    ///     Quaternion::new(0.0, 2.0, 0.0, 0.0)
    /// ));
    /// ```
    ///
    /// # See Also
    ///
    /// - [`nlerp`](UnitDualQuaternion::nlerp) - Normalized linear interpolation for unit dual quaternions
    /// - [`sclerp`](UnitDualQuaternion::sclerp) - Screw linear interpolation, providing smooth constant-velocity motion
    #[inline]
    #[must_use]
    pub fn lerp(&self, other: &Self, t: T) -> Self {
        self * (T::one() - t.clone()) + other * t
    }
}

#[cfg(feature = "bytemuck")]
unsafe impl<T> bytemuck::Zeroable for DualQuaternion<T>
where
    T: Scalar + bytemuck::Zeroable,
    Quaternion<T>: bytemuck::Zeroable,
{
}

#[cfg(feature = "bytemuck")]
unsafe impl<T> bytemuck::Pod for DualQuaternion<T>
where
    T: Scalar + bytemuck::Pod,
    Quaternion<T>: bytemuck::Pod,
{
}

#[cfg(feature = "serde-serialize-no-std")]
impl<T: SimdRealField> Serialize for DualQuaternion<T>
where
    T: Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<<S as Serializer>::Ok, <S as Serializer>::Error>
    where
        S: Serializer,
    {
        self.as_ref().serialize(serializer)
    }
}

#[cfg(feature = "serde-serialize-no-std")]
impl<'a, T: SimdRealField> Deserialize<'a> for DualQuaternion<T>
where
    T: Deserialize<'a>,
{
    fn deserialize<Des>(deserializer: Des) -> Result<Self, Des::Error>
    where
        Des: Deserializer<'a>,
    {
        type Dq<T> = [T; 8];

        let dq: Dq<T> = Dq::<T>::deserialize(deserializer)?;

        Ok(Self {
            real: Quaternion::new(dq[3].clone(), dq[0].clone(), dq[1].clone(), dq[2].clone()),
            dual: Quaternion::new(dq[7].clone(), dq[4].clone(), dq[5].clone(), dq[6].clone()),
        })
    }
}

impl<T: RealField> DualQuaternion<T> {
    #[allow(clippy::wrong_self_convention)]
    fn to_vector(&self) -> OVector<T, U8> {
        self.as_ref().clone().into()
    }
}

impl<T: RealField + AbsDiffEq<Epsilon = T>> AbsDiffEq for DualQuaternion<T> {
    type Epsilon = T;

    #[inline]
    fn default_epsilon() -> Self::Epsilon {
        T::default_epsilon()
    }

    #[inline]
    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        self.to_vector().abs_diff_eq(&other.to_vector(), epsilon.clone()) ||
        // Account for the double-covering of S², i.e. q = -q
        self.to_vector().iter().zip(other.to_vector().iter()).all(|(a, b)| a.abs_diff_eq(&-b.clone(), epsilon.clone()))
    }
}

impl<T: RealField + RelativeEq<Epsilon = T>> RelativeEq for DualQuaternion<T> {
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
        self.to_vector().relative_eq(&other.to_vector(), epsilon.clone(), max_relative.clone()) ||
        // Account for the double-covering of S², i.e. q = -q
        self.to_vector().iter().zip(other.to_vector().iter()).all(|(a, b)| a.relative_eq(&-b.clone(), epsilon.clone(), max_relative.clone()))
    }
}

impl<T: RealField + UlpsEq<Epsilon = T>> UlpsEq for DualQuaternion<T> {
    #[inline]
    fn default_max_ulps() -> u32 {
        T::default_max_ulps()
    }

    #[inline]
    fn ulps_eq(&self, other: &Self, epsilon: Self::Epsilon, max_ulps: u32) -> bool {
        self.to_vector().ulps_eq(&other.to_vector(), epsilon.clone(), max_ulps) ||
        // Account for the double-covering of S², i.e. q = -q.
        self.to_vector().iter().zip(other.to_vector().iter()).all(|(a, b)| a.ulps_eq(&-b.clone(), epsilon.clone(), max_ulps))
    }
}

/// A unit dual quaternion representing a rigid body transformation.
///
/// A unit dual quaternion is a dual quaternion whose real part has unit length (norm = 1).
/// This type is specifically designed to represent 3D rigid body transformations, which
/// consist of a rotation followed by a translation (also known as an isometry or SE(3)
/// transformation).
///
/// # Why Use Unit Dual Quaternions?
///
/// Unit dual quaternions offer several advantages for representing rigid transformations:
///
/// - **Compact**: Only 8 numbers (compared to 16 for a 4x4 matrix)
/// - **Smooth Interpolation**: Enable natural blending between poses (crucial for animation)
/// - **No Gimbal Lock**: Avoid singularities present in Euler angle representations
/// - **Efficient**: Faster than matrices for composing and applying transformations
/// - **Numerically Stable**: Less prone to drift than matrices when accumulated
///
/// # Relationship to Screw Theory
///
/// Unit dual quaternions naturally represent "screw motions" - simultaneous rotation
/// around and translation along an axis. This makes them particularly useful in robotics
/// where joint motions often follow screw patterns.
///
/// # Common Applications
///
/// - **Character Animation**: Smooth bone interpolation in skeletal animation (skinning)
/// - **Robotics**: Representing robot poses and end-effector transformations
/// - **Physics Simulation**: Tracking rigid body positions and orientations
/// - **Camera Control**: Smooth camera movements and transitions
/// - **Motion Planning**: Interpolating between waypoints in 3D space
///
/// # Example: Basic Transformation
///
/// ```
/// # use nalgebra::{UnitDualQuaternion, UnitQuaternion, Vector3, Point3};
/// // Create a transformation: rotate 90° around Z-axis, then translate
/// let transform = UnitDualQuaternion::from_parts(
///     Vector3::new(1.0, 2.0, 0.0).into(),
///     UnitQuaternion::from_euler_angles(0.0, 0.0, std::f32::consts::FRAC_PI_2)
/// );
///
/// // Apply to a point
/// let point = Point3::new(1.0, 0.0, 0.0);
/// let transformed = transform * point;
/// ```
///
/// # See Also
///
/// - [`DualQuaternion`] - The underlying non-unit dual quaternion type
/// - [`Isometry3`] - Alternative representation using matrix + translation vector
/// - [`UnitQuaternion`] - Represents rotation only (no translation)
pub type UnitDualQuaternion<T> = Unit<DualQuaternion<T>>;

impl<T: Scalar + ClosedNeg + PartialEq + SimdRealField> PartialEq for UnitDualQuaternion<T> {
    #[inline]
    fn eq(&self, rhs: &Self) -> bool {
        self.as_ref().eq(rhs.as_ref())
    }
}

impl<T: Scalar + ClosedNeg + Eq + SimdRealField> Eq for UnitDualQuaternion<T> {}

impl<T: SimdRealField> Normed for DualQuaternion<T> {
    type Norm = T::SimdRealField;

    #[inline]
    fn norm(&self) -> T::SimdRealField {
        self.real.norm()
    }

    #[inline]
    fn norm_squared(&self) -> T::SimdRealField {
        self.real.norm_squared()
    }

    #[inline]
    fn scale_mut(&mut self, n: Self::Norm) {
        self.real.scale_mut(n.clone());
        self.dual.scale_mut(n);
    }

    #[inline]
    fn unscale_mut(&mut self, n: Self::Norm) {
        self.real.unscale_mut(n.clone());
        self.dual.unscale_mut(n);
    }
}

impl<T: SimdRealField> UnitDualQuaternion<T>
where
    T::Element: SimdRealField,
{
    /// Returns a reference to the underlying dual quaternion.
    ///
    /// This provides access to the raw dual quaternion data without consuming the
    /// `UnitDualQuaternion` wrapper. This is equivalent to calling `self.as_ref()`.
    ///
    /// # Returns
    ///
    /// A reference to the inner `DualQuaternion<T>`.
    ///
    /// # Example
    ///
    /// ```
    /// # use nalgebra::{DualQuaternion, UnitDualQuaternion, Quaternion};
    /// let id = UnitDualQuaternion::identity();
    /// let inner = id.dual_quaternion();
    ///
    /// assert_eq!(*inner, DualQuaternion::from_real_and_dual(
    ///     Quaternion::new(1.0, 0.0, 0.0, 0.0),
    ///     Quaternion::new(0.0, 0.0, 0.0, 0.0)
    /// ));
    /// ```
    ///
    /// # See Also
    ///
    /// - [`into_inner`](Unit::into_inner) - Consumes the wrapper and returns the inner value
    #[inline]
    #[must_use]
    pub fn dual_quaternion(&self) -> &DualQuaternion<T> {
        self.as_ref()
    }

    /// Compute the conjugate of this unit quaternion.
    ///
    /// # Example
    /// ```
    /// # use nalgebra::{UnitDualQuaternion, DualQuaternion, Quaternion};
    /// let qr = Quaternion::new(1.0, 2.0, 3.0, 4.0);
    /// let qd = Quaternion::new(5.0, 6.0, 7.0, 8.0);
    /// let unit = UnitDualQuaternion::new_normalize(
    ///     DualQuaternion::from_real_and_dual(qr, qd)
    /// );
    /// let conj = unit.conjugate();
    /// assert_eq!(conj.real, unit.real.conjugate());
    /// assert_eq!(conj.dual, unit.dual.conjugate());
    /// ```
    #[inline]
    #[must_use = "Did you mean to use conjugate_mut()?"]
    pub fn conjugate(&self) -> Self {
        Self::new_unchecked(self.as_ref().conjugate())
    }

    /// Compute the conjugate of this unit quaternion in-place.
    ///
    /// # Example
    /// ```
    /// # use nalgebra::{UnitDualQuaternion, DualQuaternion, Quaternion};
    /// let qr = Quaternion::new(1.0, 2.0, 3.0, 4.0);
    /// let qd = Quaternion::new(5.0, 6.0, 7.0, 8.0);
    /// let unit = UnitDualQuaternion::new_normalize(
    ///     DualQuaternion::from_real_and_dual(qr, qd)
    /// );
    /// let mut conj = unit.clone();
    /// conj.conjugate_mut();
    /// assert_eq!(conj.as_ref().real, unit.as_ref().real.conjugate());
    /// assert_eq!(conj.as_ref().dual, unit.as_ref().dual.conjugate());
    /// ```
    #[inline]
    pub fn conjugate_mut(&mut self) {
        self.as_mut_unchecked().conjugate_mut()
    }

    /// Inverts this dual quaternion if it is not zero.
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{UnitDualQuaternion, Quaternion, DualQuaternion};
    /// let qr = Quaternion::new(1.0, 2.0, 3.0, 4.0);
    /// let qd = Quaternion::new(5.0, 6.0, 7.0, 8.0);
    /// let unit = UnitDualQuaternion::new_normalize(DualQuaternion::from_real_and_dual(qr, qd));
    /// let inv = unit.inverse();
    /// assert_relative_eq!(unit * inv, UnitDualQuaternion::identity(), epsilon = 1.0e-6);
    /// assert_relative_eq!(inv * unit, UnitDualQuaternion::identity(), epsilon = 1.0e-6);
    /// ```
    #[inline]
    #[must_use = "Did you mean to use inverse_mut()?"]
    pub fn inverse(&self) -> Self {
        let real = Unit::new_unchecked(self.as_ref().real.clone())
            .inverse()
            .into_inner();
        let dual = -real.clone() * self.as_ref().dual.clone() * real.clone();
        UnitDualQuaternion::new_unchecked(DualQuaternion { real, dual })
    }

    /// Inverts this dual quaternion in place if it is not zero.
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{UnitDualQuaternion, Quaternion, DualQuaternion};
    /// let qr = Quaternion::new(1.0, 2.0, 3.0, 4.0);
    /// let qd = Quaternion::new(5.0, 6.0, 7.0, 8.0);
    /// let unit = UnitDualQuaternion::new_normalize(DualQuaternion::from_real_and_dual(qr, qd));
    /// let mut inv = unit.clone();
    /// inv.inverse_mut();
    /// assert_relative_eq!(unit * inv, UnitDualQuaternion::identity(), epsilon = 1.0e-6);
    /// assert_relative_eq!(inv * unit, UnitDualQuaternion::identity(), epsilon = 1.0e-6);
    /// ```
    #[inline]
    pub fn inverse_mut(&mut self) {
        let quat = self.as_mut_unchecked();
        quat.real = Unit::new_unchecked(quat.real.clone())
            .inverse()
            .into_inner();
        quat.dual = -quat.real.clone() * quat.dual.clone() * quat.real.clone();
    }

    /// Computes the transformation needed to transform `self` into `other`.
    ///
    /// This returns the relative transformation (isometry) that, when applied to `self`,
    /// produces `other`. In other words: `self.isometry_to(other) * self == other`.
    ///
    /// This is useful for computing the difference between two poses or for finding
    /// the transformation needed to move from one configuration to another.
    ///
    /// # Arguments
    ///
    /// - `other`: The target transformation we want to reach
    ///
    /// # Returns
    ///
    /// The transformation delta from `self` to `other`.
    ///
    /// # Example
    ///
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{UnitDualQuaternion, DualQuaternion, Quaternion};
    /// let qr = Quaternion::new(1.0, 2.0, 3.0, 4.0);
    /// let qd = Quaternion::new(5.0, 6.0, 7.0, 8.0);
    /// let dq1 = UnitDualQuaternion::new_normalize(DualQuaternion::from_real_and_dual(qr, qd));
    /// let dq2 = UnitDualQuaternion::new_normalize(DualQuaternion::from_real_and_dual(qd, qr));
    ///
    /// // Compute the transformation from dq1 to dq2
    /// let delta = dq1.isometry_to(&dq2);
    ///
    /// // Verify that applying delta to dq1 gives us dq2
    /// assert_relative_eq!(delta * dq1, dq2, epsilon = 1.0e-6);
    /// ```
    ///
    /// # Use Case: Motion Planning
    ///
    /// ```
    /// # use nalgebra::{UnitDualQuaternion, UnitQuaternion, Vector3};
    /// // Current robot end-effector pose
    /// let current_pose = UnitDualQuaternion::from_parts(
    ///     Vector3::new(0.0, 0.0, 0.5).into(),
    ///     UnitQuaternion::identity()
    /// );
    ///
    /// // Desired target pose
    /// let target_pose = UnitDualQuaternion::from_parts(
    ///     Vector3::new(0.5, 0.5, 0.5).into(),
    ///     UnitQuaternion::from_euler_angles(0.0, 0.0, 1.57)
    /// );
    ///
    /// // Compute the required motion
    /// let required_motion = current_pose.isometry_to(&target_pose);
    ///
    /// // Can now plan a trajectory to execute this motion
    /// ```
    ///
    /// # See Also
    ///
    /// - [`inverse`](Self::inverse) - Computes the inverse transformation
    /// - [`sclerp`](Self::sclerp) - Smooth interpolation between transformations
    #[inline]
    #[must_use]
    pub fn isometry_to(&self, other: &Self) -> Self {
        other / self
    }

    /// Linear interpolation between two unit dual quaternions.
    ///
    /// The result is not normalized.
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{UnitDualQuaternion, DualQuaternion, Quaternion};
    /// let dq1 = UnitDualQuaternion::new_normalize(DualQuaternion::from_real_and_dual(
    ///     Quaternion::new(0.5, 0.0, 0.5, 0.0),
    ///     Quaternion::new(0.0, 0.5, 0.0, 0.5)
    /// ));
    /// let dq2 = UnitDualQuaternion::new_normalize(DualQuaternion::from_real_and_dual(
    ///     Quaternion::new(0.5, 0.0, 0.0, 0.5),
    ///     Quaternion::new(0.5, 0.0, 0.5, 0.0)
    /// ));
    /// assert_relative_eq!(
    ///     UnitDualQuaternion::new_normalize(dq1.lerp(&dq2, 0.5)),
    ///     UnitDualQuaternion::new_normalize(
    ///         DualQuaternion::from_real_and_dual(
    ///             Quaternion::new(0.5, 0.0, 0.25, 0.25),
    ///             Quaternion::new(0.25, 0.25, 0.25, 0.25)
    ///         )
    ///     ),
    ///     epsilon = 1.0e-6
    /// );
    /// ```
    #[inline]
    #[must_use]
    pub fn lerp(&self, other: &Self, t: T) -> DualQuaternion<T> {
        self.as_ref().lerp(other.as_ref(), t)
    }

    /// Normalized linear interpolation between two unit dual quaternions.
    ///
    /// NLERP (Normalized Linear Interpolation) performs linear interpolation and then
    /// normalizes the result. It's faster than [`sclerp`](Self::sclerp) but doesn't
    /// provide constant angular velocity. This is often good enough for many applications,
    /// especially when interpolating between nearby poses.
    ///
    /// The parameter `t` controls the interpolation:
    /// - When `t = 0.0`, returns `self`
    /// - When `t = 1.0`, returns `other`
    /// - When `t = 0.5`, returns approximately halfway between
    ///
    /// # Arguments
    ///
    /// - `other`: The target transformation to interpolate toward
    /// - `t`: The interpolation parameter, typically between 0.0 and 1.0
    ///
    /// # Example
    ///
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{UnitDualQuaternion, DualQuaternion, Quaternion};
    /// let dq1 = UnitDualQuaternion::new_normalize(DualQuaternion::from_real_and_dual(
    ///     Quaternion::new(0.5, 0.0, 0.5, 0.0),
    ///     Quaternion::new(0.0, 0.5, 0.0, 0.5)
    /// ));
    /// let dq2 = UnitDualQuaternion::new_normalize(DualQuaternion::from_real_and_dual(
    ///     Quaternion::new(0.5, 0.0, 0.0, 0.5),
    ///     Quaternion::new(0.5, 0.0, 0.5, 0.0)
    /// ));
    ///
    /// // Interpolate 20% of the way from dq1 to dq2
    /// let result = dq1.nlerp(&dq2, 0.2);
    ///
    /// assert_relative_eq!(result, UnitDualQuaternion::new_normalize(
    ///     DualQuaternion::from_real_and_dual(
    ///         Quaternion::new(0.5, 0.0, 0.4, 0.1),
    ///         Quaternion::new(0.1, 0.4, 0.1, 0.4)
    ///     )
    /// ), epsilon = 1.0e-6);
    /// ```
    ///
    /// # Use Case: Character Animation
    ///
    /// ```
    /// # use nalgebra::{UnitDualQuaternion, UnitQuaternion, Vector3};
    /// // Blend between two bone transformations for character skinning
    /// let bone_pose_a = UnitDualQuaternion::from_parts(
    ///     Vector3::new(1.0, 0.0, 0.0).into(),
    ///     UnitQuaternion::from_euler_angles(0.0, 0.0, 0.0)
    /// );
    /// let bone_pose_b = UnitDualQuaternion::from_parts(
    ///     Vector3::new(1.5, 0.5, 0.0).into(),
    ///     UnitQuaternion::from_euler_angles(0.0, 0.0, 0.5)
    /// );
    ///
    /// // Blend based on animation time (e.g., 0.0 to 1.0)
    /// let animation_time = 0.3;
    /// let blended_pose = bone_pose_a.nlerp(&bone_pose_b, animation_time);
    /// ```
    ///
    /// # Performance Note
    ///
    /// NLERP is faster than SCLERP but doesn't guarantee constant angular velocity.
    /// Use NLERP when:
    /// - Performance is critical
    /// - Interpolating between nearby poses
    /// - Constant velocity is not required
    ///
    /// Use SCLERP when:
    /// - Smooth, constant-velocity motion is needed
    /// - Interpolating across large angular distances
    ///
    /// # See Also
    ///
    /// - [`sclerp`](Self::sclerp) - Screw linear interpolation with constant velocity
    /// - [`lerp`](Self::lerp) - Linear interpolation without normalization
    /// - [`try_sclerp`](Self::try_sclerp) - Safe version of sclerp that handles edge cases
    #[inline]
    #[must_use]
    pub fn nlerp(&self, other: &Self, t: T) -> Self {
        let mut res = self.lerp(other, t);
        let _ = res.normalize_mut();

        Self::new_unchecked(res)
    }

    /// Screw linear interpolation between two unit dual quaternions.
    ///
    /// SCLERP (Screw Linear Interpolation) creates a smooth arc from one transformation
    /// to another with constant velocity. This is based on screw theory, where any rigid
    /// body motion can be represented as a rotation around and translation along a single
    /// axis (called a screw axis).
    ///
    /// SCLERP provides the most natural and smooth interpolation for rigid body motion,
    /// making it ideal for:
    /// - Camera animations requiring smooth motion
    /// - Robot trajectory planning with constant velocity
    /// - High-quality character animation
    ///
    /// The parameter `t` controls the interpolation:
    /// - When `t = 0.0`, returns `self`
    /// - When `t = 1.0`, returns `other`
    /// - Values between create smooth, constant-velocity motion
    ///
    /// # Arguments
    ///
    /// - `other`: The target transformation to interpolate toward
    /// - `t`: The interpolation parameter, typically between 0.0 and 1.0
    ///
    /// # Panics
    ///
    /// Panics if the angle between both quaternions is 180 degrees (in which case the
    /// interpolation is not well-defined). Use [`try_sclerp`](Self::try_sclerp) to
    /// handle this case gracefully.
    ///
    /// # Example
    ///
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{UnitDualQuaternion, UnitQuaternion, Vector3};
    /// let dq1 = UnitDualQuaternion::from_parts(
    ///     Vector3::new(0.0, 3.0, 0.0).into(),
    ///     UnitQuaternion::from_euler_angles(std::f32::consts::FRAC_PI_4, 0.0, 0.0),
    /// );
    ///
    /// let dq2 = UnitDualQuaternion::from_parts(
    ///     Vector3::new(0.0, 0.0, 3.0).into(),
    ///     UnitQuaternion::from_euler_angles(-std::f32::consts::PI, 0.0, 0.0),
    /// );
    ///
    /// // Interpolate 1/3 of the way with constant velocity
    /// let dq = dq1.sclerp(&dq2, 1.0 / 3.0);
    ///
    /// assert_relative_eq!(
    ///     dq.rotation().euler_angles().0, std::f32::consts::FRAC_PI_2, epsilon = 1.0e-6
    /// );
    /// assert_relative_eq!(dq.translation().vector.y, 3.0, epsilon = 1.0e-6);
    /// ```
    ///
    /// # Use Case: Smooth Camera Movement
    ///
    /// ```
    /// # use nalgebra::{UnitDualQuaternion, UnitQuaternion, Vector3, Point3};
    /// // Camera at starting position
    /// let camera_start = UnitDualQuaternion::from_parts(
    ///     Vector3::new(0.0, 0.0, 5.0).into(),
    ///     UnitQuaternion::from_euler_angles(0.0, 0.0, 0.0)
    /// );
    ///
    /// // Camera at ending position (moved and rotated)
    /// let camera_end = UnitDualQuaternion::from_parts(
    ///     Vector3::new(5.0, 0.0, 0.0).into(),
    ///     UnitQuaternion::from_euler_angles(0.0, std::f32::consts::FRAC_PI_2, 0.0)
    /// );
    ///
    /// // Animate camera with smooth constant velocity
    /// let num_frames = 60;
    /// for frame in 0..=num_frames {
    ///     let t = frame as f32 / num_frames as f32;
    ///     let camera_pose = camera_start.sclerp(&camera_end, t);
    ///     // Use camera_pose to render frame...
    /// }
    /// ```
    ///
    /// # See Also
    ///
    /// - [`try_sclerp`](Self::try_sclerp) - Safe version that returns `None` for ambiguous cases
    /// - [`nlerp`](Self::nlerp) - Faster but non-constant velocity interpolation
    /// - [`lerp`](Self::lerp) - Basic linear interpolation without normalization
    #[inline]
    #[must_use]
    pub fn sclerp(&self, other: &Self, t: T) -> Self
    where
        T: RealField,
    {
        self.try_sclerp(other, t, T::default_epsilon())
            .expect("DualQuaternion sclerp: ambiguous configuration.")
    }

    /// Computes the screw-linear interpolation between two unit dual quaternions or
    /// returns `None` if both quaternions are approximately 180 degrees apart.
    ///
    /// This is the safe version of [`sclerp`](Self::sclerp) that handles the ambiguous
    /// case where the two transformations are exactly opposite (180 degrees apart).
    /// In such cases, there are infinitely many valid interpolation paths, so this
    /// method returns `None`.
    ///
    /// SCLERP (Screw Linear Interpolation) provides smooth, constant-velocity motion
    /// based on screw theory. It's the highest quality interpolation method for rigid
    /// body transformations.
    ///
    /// # Arguments
    ///
    /// * `self`: The starting transformation to interpolate from
    /// * `other`: The target transformation to interpolate toward
    /// * `t`: The interpolation parameter, typically between 0.0 and 1.0
    /// * `epsilon`: The tolerance below which the sine of the angle separating both
    ///   quaternions must be to return `None` (handles the 180-degree case)
    ///
    /// # Returns
    ///
    /// - `Some(interpolated)` if the interpolation is well-defined
    /// - `None` if the quaternions are approximately 180 degrees apart
    ///
    /// # Example
    ///
    /// ```
    /// # use nalgebra::{UnitDualQuaternion, UnitQuaternion, Vector3};
    /// let dq1 = UnitDualQuaternion::from_parts(
    ///     Vector3::new(0.0, 0.0, 0.0).into(),
    ///     UnitQuaternion::from_euler_angles(0.0, 0.0, 0.0)
    /// );
    /// let dq2 = UnitDualQuaternion::from_parts(
    ///     Vector3::new(1.0, 0.0, 0.0).into(),
    ///     UnitQuaternion::from_euler_angles(0.0, 0.0, 1.0)
    /// );
    ///
    /// let result = dq1.try_sclerp(&dq2, 0.5, 1.0e-6);
    /// assert!(result.is_some());
    /// ```
    ///
    /// # Use Case: Robust Animation System
    ///
    /// ```
    /// # use nalgebra::{UnitDualQuaternion, UnitQuaternion, Vector3};
    /// fn interpolate_poses(
    ///     start: &UnitDualQuaternion<f32>,
    ///     end: &UnitDualQuaternion<f32>,
    ///     t: f32
    /// ) -> UnitDualQuaternion<f32> {
    ///     // Try sclerp first (best quality)
    ///     if let Some(result) = start.try_sclerp(end, t, 1.0e-6) {
    ///         result
    ///     } else {
    ///         // Fall back to nlerp if sclerp is ambiguous
    ///         start.nlerp(end, t)
    ///     }
    /// }
    /// ```
    ///
    /// # See Also
    ///
    /// - [`sclerp`](Self::sclerp) - Panicking version for when you know the interpolation is valid
    /// - [`nlerp`](Self::nlerp) - Faster alternative that works in all cases
    /// - [`lerp`](Self::lerp) - Basic linear interpolation
    #[inline]
    #[must_use]
    pub fn try_sclerp(&self, other: &Self, t: T, epsilon: T) -> Option<Self>
    where
        T: RealField,
    {
        let two = T::one() + T::one();
        let half = T::one() / two.clone();

        // Invert one of the quaternions if we've got a longest-path
        // interpolation.
        let other = {
            let dot_product = self.as_ref().real.coords.dot(&other.as_ref().real.coords);
            if relative_eq!(dot_product, T::zero(), epsilon = epsilon.clone()) {
                return None;
            }

            if dot_product < T::zero() {
                -other.clone()
            } else {
                other.clone()
            }
        };

        let difference = self.as_ref().conjugate() * other.as_ref();
        let norm_squared = difference.real.vector().norm_squared();
        if relative_eq!(norm_squared, T::zero(), epsilon = epsilon) {
            return Some(Self::from_parts(
                self.translation()
                    .vector
                    .lerp(&other.translation().vector, t)
                    .into(),
                self.rotation(),
            ));
        }

        let scalar: T = difference.real.scalar();
        let mut angle = two.clone() * scalar.acos();

        let inverse_norm_squared: T = T::one() / norm_squared;
        let inverse_norm = inverse_norm_squared.sqrt();

        let mut pitch = -two * difference.dual.scalar() * inverse_norm.clone();
        let direction = difference.real.vector() * inverse_norm.clone();
        let moment = (difference.dual.vector()
            - direction.clone() * (pitch.clone() * difference.real.scalar() * half.clone()))
            * inverse_norm;

        angle *= t.clone();
        pitch *= t;

        let sin = (half.clone() * angle.clone()).sin();
        let cos = (half.clone() * angle).cos();

        let real = Quaternion::from_parts(cos.clone(), direction.clone() * sin.clone());
        let dual = Quaternion::from_parts(
            -pitch.clone() * half.clone() * sin.clone(),
            moment * sin + direction * (pitch * half * cos),
        );

        Some(
            self * UnitDualQuaternion::new_unchecked(DualQuaternion::from_real_and_dual(
                real, dual,
            )),
        )
    }

    /// Extracts the rotation component of this unit dual quaternion.
    ///
    /// A unit dual quaternion represents a rigid body transformation consisting of
    /// a rotation followed by a translation. This method returns only the rotation
    /// part as a `UnitQuaternion`, discarding the translation.
    ///
    /// # Returns
    ///
    /// The rotation component as a `UnitQuaternion<T>`.
    ///
    /// # Example
    ///
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{UnitDualQuaternion, UnitQuaternion, Vector3};
    /// let dq = UnitDualQuaternion::from_parts(
    ///     Vector3::new(0.0, 3.0, 0.0).into(),
    ///     UnitQuaternion::from_euler_angles(std::f32::consts::FRAC_PI_4, 0.0, 0.0)
    /// );
    ///
    /// let rotation = dq.rotation();
    /// assert_relative_eq!(
    ///     rotation.angle(), std::f32::consts::FRAC_PI_4, epsilon = 1.0e-6
    /// );
    /// ```
    ///
    /// # Use Case: Extracting Orientation from Robot Pose
    ///
    /// ```
    /// # use nalgebra::{UnitDualQuaternion, UnitQuaternion, Vector3};
    /// // Robot end-effector pose (position + orientation)
    /// let end_effector_pose = UnitDualQuaternion::from_parts(
    ///     Vector3::new(1.0, 2.0, 0.5).into(),
    ///     UnitQuaternion::from_euler_angles(0.0, 0.0, 1.57)
    /// );
    ///
    /// // Extract just the orientation for a gripper
    /// let gripper_orientation = end_effector_pose.rotation();
    ///
    /// // Can now use this to control gripper rotation independently
    /// ```
    ///
    /// # See Also
    ///
    /// - [`translation`](Self::translation) - Extracts the translation component
    /// - [`to_isometry`](Self::to_isometry) - Converts to an `Isometry3` with both components
    /// - [`from_rotation`](Self::from_rotation) - Creates a dual quaternion from just a rotation
    #[inline]
    #[must_use]
    pub fn rotation(&self) -> UnitQuaternion<T> {
        Unit::new_unchecked(self.as_ref().real.clone())
    }

    /// Extracts the translation component of this unit dual quaternion.
    ///
    /// A unit dual quaternion represents a rigid body transformation consisting of
    /// a rotation followed by a translation. This method returns only the translation
    /// part as a `Translation3`, discarding the rotation.
    ///
    /// # Returns
    ///
    /// The translation component as a `Translation3<T>`.
    ///
    /// # Example
    ///
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{UnitDualQuaternion, UnitQuaternion, Vector3};
    /// let dq = UnitDualQuaternion::from_parts(
    ///     Vector3::new(0.0, 3.0, 0.0).into(),
    ///     UnitQuaternion::from_euler_angles(std::f32::consts::FRAC_PI_4, 0.0, 0.0)
    /// );
    ///
    /// let translation = dq.translation();
    /// assert_relative_eq!(
    ///     translation.vector, Vector3::new(0.0, 3.0, 0.0), epsilon = 1.0e-6
    /// );
    /// ```
    ///
    /// # Use Case: Extracting Position from Camera Pose
    ///
    /// ```
    /// # use nalgebra::{UnitDualQuaternion, UnitQuaternion, Vector3};
    /// // Camera pose (position + orientation)
    /// let camera_pose = UnitDualQuaternion::from_parts(
    ///     Vector3::new(10.0, 5.0, 2.0).into(),
    ///     UnitQuaternion::from_euler_angles(0.0, 0.5, 0.0)
    /// );
    ///
    /// // Extract just the position
    /// let camera_position = camera_pose.translation();
    /// println!("Camera is at: {:?}", camera_position.vector);
    /// ```
    ///
    /// # See Also
    ///
    /// - [`rotation`](Self::rotation) - Extracts the rotation component
    /// - [`to_isometry`](Self::to_isometry) - Converts to an `Isometry3` with both components
    /// - [`from_parts`](Self::from_parts) - Creates a dual quaternion from translation and rotation
    #[inline]
    #[must_use]
    pub fn translation(&self) -> Translation3<T> {
        let two = T::one() + T::one();
        Translation3::from(
            ((self.as_ref().dual.clone() * self.as_ref().real.clone().conjugate()) * two)
                .vector()
                .into_owned(),
        )
    }

    /// Converts this unit dual quaternion into an isometry.
    ///
    /// An `Isometry3` represents a 3D rigid body transformation using a separate
    /// rotation quaternion and translation vector. This method extracts these components
    /// from the dual quaternion representation.
    ///
    /// This is useful when you need to interface with code that works with isometries
    /// rather than dual quaternions, or when you want to separately access and manipulate
    /// the rotation and translation components.
    ///
    /// # Returns
    ///
    /// An `Isometry3<T>` with the same transformation.
    ///
    /// # Example
    ///
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{UnitDualQuaternion, UnitQuaternion, Vector3};
    /// let rotation = UnitQuaternion::from_euler_angles(std::f32::consts::PI, 0.0, 0.0);
    /// let translation = Vector3::new(1.0, 3.0, 2.5);
    /// let dq = UnitDualQuaternion::from_parts(
    ///     translation.into(),
    ///     rotation
    /// );
    ///
    /// // Convert to isometry
    /// let iso = dq.to_isometry();
    ///
    /// assert_relative_eq!(iso.rotation.angle(), std::f32::consts::PI, epsilon = 1.0e-6);
    /// assert_relative_eq!(iso.translation.vector, translation, epsilon = 1.0e-6);
    /// ```
    ///
    /// # Use Case: Interfacing with Physics Engines
    ///
    /// ```
    /// # use nalgebra::{UnitDualQuaternion, UnitQuaternion, Vector3, Isometry3};
    /// // Animation system uses dual quaternions for smooth interpolation
    /// let start = UnitDualQuaternion::from_parts(
    ///     Vector3::new(0.0, 0.0, 0.0).into(),
    ///     UnitQuaternion::identity()
    /// );
    /// let end = UnitDualQuaternion::from_parts(
    ///     Vector3::new(1.0, 0.0, 0.0).into(),
    ///     UnitQuaternion::from_euler_angles(0.0, 0.0, 1.0)
    /// );
    /// let interpolated = start.sclerp(&end, 0.5);
    ///
    /// // Convert to isometry for physics engine
    /// let physics_transform: Isometry3<f32> = interpolated.to_isometry();
    /// // Set physics body transform...
    /// ```
    ///
    /// # See Also
    ///
    /// - [`from_isometry`](Self::from_isometry) - Creates from an isometry
    /// - [`to_homogeneous`](Self::to_homogeneous) - Converts to a 4x4 matrix
    /// - `From<UnitDualQuaternion<T>> for Isometry3<T>` - Conversion trait implementation
    #[inline]
    #[must_use]
    pub fn to_isometry(self) -> Isometry3<T> {
        Isometry3::from_parts(self.translation(), self.rotation())
    }

    /// Transforms a point by this unit dual quaternion.
    ///
    /// This applies the full rigid body transformation (rotation + translation) to the point.
    /// The transformation is applied in the order: first rotate, then translate.
    ///
    /// This is equivalent to the multiplication `self * pt` but may be more readable.
    ///
    /// # Arguments
    ///
    /// - `pt`: The point to transform
    ///
    /// # Returns
    ///
    /// The transformed point in 3D space.
    ///
    /// # Example
    ///
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{UnitDualQuaternion, UnitQuaternion, Vector3, Point3};
    /// let dq = UnitDualQuaternion::from_parts(
    ///     Vector3::new(0.0, 3.0, 0.0).into(),
    ///     UnitQuaternion::from_euler_angles(std::f32::consts::FRAC_PI_2, 0.0, 0.0)
    /// );
    /// let point = Point3::new(1.0, 2.0, 3.0);
    ///
    /// let transformed = dq.transform_point(&point);
    /// assert_relative_eq!(
    ///     transformed, Point3::new(1.0, 0.0, 2.0), epsilon = 1.0e-6
    /// );
    /// ```
    ///
    /// # Use Case: Applying Robot Transformation
    ///
    /// ```
    /// # use nalgebra::{UnitDualQuaternion, UnitQuaternion, Vector3, Point3};
    /// // Robot base transformation
    /// let robot_pose = UnitDualQuaternion::from_parts(
    ///     Vector3::new(1.0, 0.0, 0.0).into(),
    ///     UnitQuaternion::from_euler_angles(0.0, 0.0, std::f32::consts::FRAC_PI_2)
    /// );
    ///
    /// // Point in robot's local coordinate system
    /// let local_point = Point3::new(0.5, 0.0, 0.0);
    ///
    /// // Transform to world coordinates
    /// let world_point = robot_pose.transform_point(&local_point);
    /// ```
    ///
    /// # See Also
    ///
    /// - [`inverse_transform_point`](Self::inverse_transform_point) - Applies the inverse transformation
    /// - [`transform_vector`](Self::transform_vector) - Transforms a vector (no translation)
    /// - Operator `*` - Alternative syntax: `transform * point`
    #[inline]
    #[must_use]
    pub fn transform_point(&self, pt: &Point3<T>) -> Point3<T> {
        self * pt
    }

    /// Transforms a vector by this unit dual quaternion, ignoring translation.
    ///
    /// This applies only the rotation part of the transformation to the vector.
    /// Unlike [`transform_point`](Self::transform_point), the translation component
    /// is ignored because vectors represent directions, not positions.
    ///
    /// This is useful for transforming directions like surface normals or velocity vectors.
    ///
    /// This is equivalent to the multiplication `self * v` but may be more readable.
    ///
    /// # Arguments
    ///
    /// - `v`: The vector to rotate
    ///
    /// # Returns
    ///
    /// The rotated vector (translation is not applied).
    ///
    /// # Example
    ///
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{UnitDualQuaternion, UnitQuaternion, Vector3};
    /// let dq = UnitDualQuaternion::from_parts(
    ///     Vector3::new(0.0, 3.0, 0.0).into(),  // Translation (ignored for vectors)
    ///     UnitQuaternion::from_euler_angles(std::f32::consts::FRAC_PI_2, 0.0, 0.0)
    /// );
    /// let vector = Vector3::new(1.0, 2.0, 3.0);
    ///
    /// let rotated = dq.transform_vector(&vector);
    /// assert_relative_eq!(
    ///     rotated, Vector3::new(1.0, -3.0, 2.0), epsilon = 1.0e-6
    /// );
    /// ```
    ///
    /// # Use Case: Transforming Surface Normals
    ///
    /// ```
    /// # use nalgebra::{UnitDualQuaternion, UnitQuaternion, Vector3};
    /// // Object transformation
    /// let object_transform = UnitDualQuaternion::from_parts(
    ///     Vector3::new(5.0, 0.0, 0.0).into(),
    ///     UnitQuaternion::from_euler_angles(0.0, std::f32::consts::FRAC_PI_4, 0.0)
    /// );
    ///
    /// // Surface normal in object's local space
    /// let local_normal = Vector3::new(0.0, 1.0, 0.0);
    ///
    /// // Transform to world space (only rotation applied, translation ignored)
    /// let world_normal = object_transform.transform_vector(&local_normal);
    /// ```
    ///
    /// # See Also
    ///
    /// - [`transform_point`](Self::transform_point) - Transforms a point (includes translation)
    /// - [`inverse_transform_vector`](Self::inverse_transform_vector) - Applies the inverse rotation
    /// - [`rotation`](Self::rotation) - Extracts just the rotation component
    #[inline]
    #[must_use]
    pub fn transform_vector(&self, v: &Vector3<T>) -> Vector3<T> {
        self * v
    }

    /// Transforms a point by the inverse of this unit dual quaternion.
    ///
    /// This applies the inverse transformation (reverse rotation and translation) to the point.
    /// It's equivalent to computing `self.inverse() * pt` but may be more efficient
    /// as it doesn't require explicitly computing the inverse first.
    ///
    /// This is useful when you need to transform from world space to local space.
    ///
    /// # Arguments
    ///
    /// - `pt`: The point to transform
    ///
    /// # Returns
    ///
    /// The point transformed by the inverse transformation.
    ///
    /// # Example
    ///
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{UnitDualQuaternion, UnitQuaternion, Vector3, Point3};
    /// let dq = UnitDualQuaternion::from_parts(
    ///     Vector3::new(0.0, 3.0, 0.0).into(),
    ///     UnitQuaternion::from_euler_angles(std::f32::consts::FRAC_PI_2, 0.0, 0.0)
    /// );
    /// let point = Point3::new(1.0, 2.0, 3.0);
    ///
    /// let inverse_transformed = dq.inverse_transform_point(&point);
    /// assert_relative_eq!(
    ///     inverse_transformed, Point3::new(1.0, 3.0, 1.0), epsilon = 1.0e-6
    /// );
    /// ```
    ///
    /// # Use Case: World to Local Space Conversion
    ///
    /// ```
    /// # use nalgebra::{UnitDualQuaternion, UnitQuaternion, Vector3, Point3};
    /// // Camera transformation in world space
    /// let camera_transform = UnitDualQuaternion::from_parts(
    ///     Vector3::new(0.0, 5.0, 10.0).into(),
    ///     UnitQuaternion::from_euler_angles(0.0, 0.0, 0.0)
    /// );
    ///
    /// // World space point
    /// let world_point = Point3::new(5.0, 5.0, 5.0);
    ///
    /// // Convert to camera-local space
    /// let camera_local = camera_transform.inverse_transform_point(&world_point);
    /// ```
    ///
    /// # See Also
    ///
    /// - [`transform_point`](Self::transform_point) - Applies the forward transformation
    /// - [`inverse`](Self::inverse) - Computes the inverse transformation
    /// - [`inverse_transform_vector`](Self::inverse_transform_vector) - For vectors (no translation)
    #[inline]
    #[must_use]
    pub fn inverse_transform_point(&self, pt: &Point3<T>) -> Point3<T> {
        self.inverse() * pt
    }

    /// Transforms a vector by the inverse rotation, ignoring translation.
    ///
    /// This applies only the inverse rotation to the vector, with the translation
    /// component ignored. It's equivalent to computing `self.inverse() * v` but
    /// may be more efficient as it doesn't require explicitly computing the full inverse.
    ///
    /// This is useful for transforming directions (like normals or velocities) from
    /// world space to local space.
    ///
    /// # Arguments
    ///
    /// - `v`: The vector to transform
    ///
    /// # Returns
    ///
    /// The vector transformed by the inverse rotation (translation not applied).
    ///
    /// # Example
    ///
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{UnitDualQuaternion, UnitQuaternion, Vector3};
    /// let dq = UnitDualQuaternion::from_parts(
    ///     Vector3::new(0.0, 3.0, 0.0).into(),
    ///     UnitQuaternion::from_euler_angles(std::f32::consts::FRAC_PI_2, 0.0, 0.0)
    /// );
    /// let vector = Vector3::new(1.0, 2.0, 3.0);
    ///
    /// let inverse_rotated = dq.inverse_transform_vector(&vector);
    /// assert_relative_eq!(
    ///     inverse_rotated, Vector3::new(1.0, 3.0, -2.0), epsilon = 1.0e-6
    /// );
    /// ```
    ///
    /// # Use Case: Converting Normals to Local Space
    ///
    /// ```
    /// # use nalgebra::{UnitDualQuaternion, UnitQuaternion, Vector3};
    /// // Object transformation in world space
    /// let object_transform = UnitDualQuaternion::from_parts(
    ///     Vector3::new(5.0, 0.0, 0.0).into(),
    ///     UnitQuaternion::from_euler_angles(0.0, 0.0, std::f32::consts::FRAC_PI_2)
    /// );
    ///
    /// // Surface normal in world space
    /// let world_normal = Vector3::new(0.0, 1.0, 0.0);
    ///
    /// // Convert to object-local space (only rotation matters)
    /// let local_normal = object_transform.inverse_transform_vector(&world_normal);
    /// ```
    ///
    /// # See Also
    ///
    /// - [`transform_vector`](Self::transform_vector) - Applies the forward rotation
    /// - [`inverse_transform_point`](Self::inverse_transform_point) - For points (includes translation)
    /// - [`inverse_transform_unit_vector`](Self::inverse_transform_unit_vector) - For unit vectors
    #[inline]
    #[must_use]
    pub fn inverse_transform_vector(&self, v: &Vector3<T>) -> Vector3<T> {
        self.inverse() * v
    }

    /// Transforms a unit vector by the inverse rotation, ignoring translation.
    ///
    /// This is a specialized version of [`inverse_transform_vector`](Self::inverse_transform_vector)
    /// for unit-length vectors. Since rotation preserves length, the result is guaranteed
    /// to remain a unit vector, so this returns a `Unit<Vector3<T>>` instead of a plain vector.
    ///
    /// This may be more efficient than computing the full inverse transformation and is
    /// particularly useful for transforming normalized directions like axis vectors or normals.
    ///
    /// # Arguments
    ///
    /// - `v`: The unit vector to transform
    ///
    /// # Returns
    ///
    /// The unit vector transformed by the inverse rotation (translation not applied).
    ///
    /// # Example
    ///
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{UnitDualQuaternion, UnitQuaternion, Unit, Vector3};
    /// let dq = UnitDualQuaternion::from_parts(
    ///     Vector3::new(0.0, 3.0, 0.0).into(),
    ///     UnitQuaternion::from_euler_angles(std::f32::consts::FRAC_PI_2, 0.0, 0.0)
    /// );
    /// let unit_vector = Unit::new_unchecked(Vector3::new(0.0, 1.0, 0.0));
    ///
    /// let inverse_rotated = dq.inverse_transform_unit_vector(&unit_vector);
    /// assert_relative_eq!(
    ///     inverse_rotated,
    ///     Unit::new_unchecked(Vector3::new(0.0, 0.0, -1.0)),
    ///     epsilon = 1.0e-6
    /// );
    /// ```
    ///
    /// # Use Case: Coordinate System Conversion
    ///
    /// ```
    /// # use nalgebra::{UnitDualQuaternion, UnitQuaternion, Unit, Vector3};
    /// // Object's coordinate system in world space
    /// let object_transform = UnitDualQuaternion::from_parts(
    ///     Vector3::new(5.0, 0.0, 0.0).into(),
    ///     UnitQuaternion::from_euler_angles(0.0, 1.57, 0.0)
    /// );
    ///
    /// // World-space up vector
    /// let world_up = Vector3::y_axis();
    ///
    /// // What direction is "up" in the object's local space?
    /// let local_up = object_transform.inverse_transform_unit_vector(&world_up);
    /// ```
    ///
    /// # See Also
    ///
    /// - [`inverse_transform_vector`](Self::inverse_transform_vector) - For non-unit vectors
    /// - [`transform_vector`](Self::transform_vector) - Applies the forward rotation
    /// - [`inverse_transform_point`](Self::inverse_transform_point) - For points (includes translation)
    #[inline]
    #[must_use]
    pub fn inverse_transform_unit_vector(&self, v: &Unit<Vector3<T>>) -> Unit<Vector3<T>> {
        self.inverse() * v
    }
}

impl<T: SimdRealField + RealField> UnitDualQuaternion<T>
where
    T::Element: SimdRealField,
{
    /// Converts this unit dual quaternion into a 4x4 homogeneous transformation matrix.
    ///
    /// A homogeneous transformation matrix is a 4x4 matrix commonly used in computer graphics
    /// and robotics to represent rigid body transformations. The resulting matrix combines
    /// the rotation and translation into a single matrix that can be used with homogeneous
    /// coordinates.
    ///
    /// The matrix has the form:
    /// ```text
    /// [ R  R  R  tx ]
    /// [ R  R  R  ty ]
    /// [ R  R  R  tz ]
    /// [ 0  0  0  1  ]
    /// ```
    /// where R is the 3x3 rotation matrix and (tx, ty, tz) is the translation vector.
    ///
    /// # Returns
    ///
    /// A 4x4 homogeneous transformation matrix.
    ///
    /// # Example
    ///
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{Matrix4, UnitDualQuaternion, UnitQuaternion, Vector3};
    /// let dq = UnitDualQuaternion::from_parts(
    ///     Vector3::new(1.0, 3.0, 2.0).into(),
    ///     UnitQuaternion::from_axis_angle(&Vector3::z_axis(), std::f32::consts::FRAC_PI_6)
    /// );
    ///
    /// let matrix = dq.to_homogeneous();
    ///
    /// let expected = Matrix4::new(
    ///     0.8660254, -0.5,      0.0, 1.0,
    ///     0.5,       0.8660254, 0.0, 3.0,
    ///     0.0,       0.0,       1.0, 2.0,
    ///     0.0,       0.0,       0.0, 1.0
    /// );
    ///
    /// assert_relative_eq!(matrix, expected, epsilon = 1.0e-6);
    /// ```
    ///
    /// # Use Case: Integration with Graphics Pipeline
    ///
    /// ```
    /// # use nalgebra::{UnitDualQuaternion, UnitQuaternion, Vector3, Matrix4, Point3};
    /// // Object transformation
    /// let object_transform = UnitDualQuaternion::from_parts(
    ///     Vector3::new(2.0, 0.0, 1.0).into(),
    ///     UnitQuaternion::from_euler_angles(0.0, 0.0, 0.5)
    /// );
    ///
    /// // Convert to matrix for use with graphics APIs (OpenGL, WebGPU, etc.)
    /// let model_matrix: Matrix4<f32> = object_transform.to_homogeneous();
    ///
    /// // Can now send model_matrix to GPU as a uniform
    /// // uniform mat4 model_matrix;
    /// ```
    ///
    /// # See Also
    ///
    /// - [`to_isometry`](Self::to_isometry) - Converts to an `Isometry3` (more efficient representation)
    /// - [`from_isometry`](Self::from_isometry) - Creates from an `Isometry3`
    /// - `From<UnitDualQuaternion<T>> for Matrix4<T>` - Conversion trait implementation
    #[inline]
    #[must_use]
    pub fn to_homogeneous(self) -> Matrix4<T> {
        self.to_isometry().to_homogeneous()
    }
}

impl<T: RealField> Default for UnitDualQuaternion<T> {
    fn default() -> Self {
        Self::identity()
    }
}

impl<T: RealField + fmt::Display> fmt::Display for UnitDualQuaternion<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.rotation().axis() {
            Some(axis) => {
                let axis = axis.into_inner();
                write!(
                    f,
                    "UnitDualQuaternion translation: {} − angle: {} − axis: ({}, {}, {})",
                    self.translation().vector,
                    self.rotation().angle(),
                    axis[0],
                    axis[1],
                    axis[2]
                )
            }
            None => {
                write!(
                    f,
                    "UnitDualQuaternion translation: {} − angle: {} − axis: (undefined)",
                    self.translation().vector,
                    self.rotation().angle()
                )
            }
        }
    }
}

impl<T: RealField + AbsDiffEq<Epsilon = T>> AbsDiffEq for UnitDualQuaternion<T> {
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

impl<T: RealField + RelativeEq<Epsilon = T>> RelativeEq for UnitDualQuaternion<T> {
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

impl<T: RealField + UlpsEq<Epsilon = T>> UlpsEq for UnitDualQuaternion<T> {
    #[inline]
    fn default_max_ulps() -> u32 {
        T::default_max_ulps()
    }

    #[inline]
    fn ulps_eq(&self, other: &Self, epsilon: Self::Epsilon, max_ulps: u32) -> bool {
        self.as_ref().ulps_eq(other.as_ref(), epsilon, max_ulps)
    }
}
