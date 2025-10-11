// Needed otherwise the rkyv macros generate code incompatible with rust-2024
#![cfg_attr(feature = "rkyv-serialize", allow(unsafe_op_in_unsafe_fn))]

use std::fmt;
use std::ops::Deref;

#[cfg(feature = "serde-serialize-no-std")]
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::allocator::Allocator;
use crate::base::DefaultAllocator;
use crate::storage::RawStorage;
use crate::{Dim, Matrix, OMatrix, RealField, Scalar, SimdComplexField, SimdRealField};

#[cfg(feature = "rkyv-serialize")]
use rkyv::bytecheck;

/// A wrapper that ensures the underlying algebraic entity has a unit norm.
///
/// **It is likely that the only piece of documentation that you need in this page are:**
/// - **[The construction with normalization](#construction-with-normalization)**
/// - **[Data extraction and construction without normalization](#data-extraction-and-construction-without-normalization)**
/// - **[Interpolation between two unit vectors](#interpolation-between-two-unit-vectors)**
///
/// All the other impl blocks you will see in this page are about [`UnitComplex`](crate::UnitComplex)
/// and [`UnitQuaternion`](crate::UnitQuaternion); both built on top of `Unit`.  If you are interested
/// in their documentation, read their dedicated pages directly.
#[repr(transparent)]
#[derive(Clone, Hash, Copy)]
#[cfg_attr(
    feature = "rkyv-serialize-no-std",
    derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize),
    archive(
        as = "Unit<T::Archived>",
        bound(archive = "
        T: rkyv::Archive,
    ")
    )
)]
#[cfg_attr(feature = "rkyv-serialize", derive(bytecheck::CheckBytes))]
#[cfg_attr(feature = "defmt", derive(defmt::Format))]
pub struct Unit<T> {
    pub(crate) value: T,
}

impl<T: fmt::Debug> fmt::Debug for Unit<T> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        self.value.fmt(formatter)
    }
}

#[cfg(feature = "bytemuck")]
unsafe impl<T> bytemuck::Zeroable for Unit<T> where T: bytemuck::Zeroable {}

#[cfg(feature = "bytemuck")]
unsafe impl<T> bytemuck::Pod for Unit<T> where T: bytemuck::Pod {}

#[cfg(feature = "serde-serialize-no-std")]
impl<T: Serialize> Serialize for Unit<T> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.value.serialize(serializer)
    }
}

#[cfg(feature = "serde-serialize-no-std")]
impl<'de, T: Deserialize<'de>> Deserialize<'de> for Unit<T> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        T::deserialize(deserializer).map(|x| Unit { value: x })
    }
}

impl<T, R, C, S> PartialEq for Unit<Matrix<T, R, C, S>>
where
    T: Scalar + PartialEq,
    R: Dim,
    C: Dim,
    S: RawStorage<T, R, C>,
{
    #[inline]
    fn eq(&self, rhs: &Self) -> bool {
        self.value.eq(&rhs.value)
    }
}

impl<T, R, C, S> Eq for Unit<Matrix<T, R, C, S>>
where
    T: Scalar + Eq,
    R: Dim,
    C: Dim,
    S: RawStorage<T, R, C>,
{
}

/// Trait implemented by entities scan be be normalized and put in an `Unit` struct.
pub trait Normed {
    /// The type of the norm.
    type Norm: SimdRealField;
    /// Computes the norm.
    fn norm(&self) -> Self::Norm;
    /// Computes the squared norm.
    fn norm_squared(&self) -> Self::Norm;
    /// Multiply `self` by n.
    fn scale_mut(&mut self, n: Self::Norm);
    /// Divides `self` by n.
    fn unscale_mut(&mut self, n: Self::Norm);
}

/// # Construction with normalization
impl<T: Normed> Unit<T> {
    /// Normalizes the given value and wraps it in a `Unit` structure.
    ///
    /// A **unit vector** is a vector with a length (norm) of exactly 1. Unit vectors are commonly
    /// used to represent directions in space, surface normals in 3D graphics, or any other
    /// scenario where only the direction matters, not the magnitude.
    ///
    /// This function takes any non-zero value (typically a vector), calculates its norm,
    /// divides the value by that norm to make its length equal to 1, and wraps the result
    /// in a `Unit` wrapper that guarantees it remains normalized.
    ///
    /// # Examples
    ///
    /// Creating a unit vector from a 2D vector:
    ///
    /// ```
    /// use nalgebra::{Unit, Vector2};
    ///
    /// let v = Vector2::new(3.0, 4.0);
    /// let unit_v = Unit::new_normalize(v);
    ///
    /// // The resulting vector has length 1.0
    /// assert!((unit_v.norm() - 1.0).abs() < 1e-10);
    ///
    /// // The direction is preserved (3:4 ratio becomes 0.6:0.8)
    /// assert!((unit_v.x - 0.6).abs() < 1e-10);
    /// assert!((unit_v.y - 0.8).abs() < 1e-10);
    /// ```
    ///
    /// Creating a unit vector representing a direction in 3D space:
    ///
    /// ```
    /// use nalgebra::{Unit, Vector3};
    ///
    /// // Create a direction vector pointing northeast and up
    /// let direction = Vector3::new(1.0, 1.0, 1.0);
    /// let unit_direction = Unit::new_normalize(direction);
    ///
    /// // Now we have a proper unit direction vector
    /// assert!((unit_direction.norm() - 1.0).abs() < 1e-10);
    /// ```
    ///
    /// # Panics
    ///
    /// This function will panic (or produce `NaN` values) if the input has a norm of zero
    /// or very close to zero. For safer handling of potentially zero-length vectors,
    /// use [`try_new`](Self::try_new) instead.
    ///
    /// # See Also
    ///
    /// - [`try_new`](Self::try_new) - Safe version that returns `None` for zero-length inputs
    /// - [`new_and_get`](Self::new_and_get) - Returns both the unit value and the original norm
    /// - [`new_unchecked`](Self::new_unchecked) - Creates a `Unit` without normalization (unsafe)
    #[inline]
    pub fn new_normalize(value: T) -> Self {
        Self::new_and_get(value).0
    }

    /// Attempts to normalize the given value and wrap it in a `Unit` structure.
    ///
    /// This is the **safe version** of normalization. Unlike [`new_normalize`](Self::new_normalize),
    /// this function will not panic or produce `NaN` values when given a zero-length or very
    /// small vector. Instead, it returns `None` if the norm of the input is less than or equal
    /// to the specified `min_norm` threshold.
    ///
    /// The `Unit` wrapper guarantees that the wrapped value has a norm of exactly 1. This is
    /// essential for representing directions, rotations, or any mathematical operation that
    /// requires normalized values.
    ///
    /// # Arguments
    ///
    /// * `value` - The value to normalize (typically a vector)
    /// * `min_norm` - The minimum acceptable norm. If the input's norm is less than or equal
    ///   to this value, the function returns `None`
    ///
    /// # Returns
    ///
    /// - `Some(Unit<T>)` if normalization succeeds (norm > min_norm)
    /// - `None` if the norm is too small (norm <= min_norm)
    ///
    /// # Examples
    ///
    /// Successfully normalizing a valid vector:
    ///
    /// ```
    /// use nalgebra::{Unit, Vector2};
    ///
    /// let v = Vector2::new(3.0, 4.0);
    /// let unit_v = Unit::try_new(v, 1e-10);
    ///
    /// assert!(unit_v.is_some());
    /// let unit_v = unit_v.unwrap();
    /// assert!((unit_v.norm() - 1.0).abs() < 1e-10);
    /// ```
    ///
    /// Handling a near-zero vector safely:
    ///
    /// ```
    /// use nalgebra::{Unit, Vector3};
    ///
    /// // A vector that's too small to normalize reliably
    /// let tiny = Vector3::new(1e-15, 1e-15, 1e-15);
    /// let result = Unit::try_new(tiny, 1e-10);
    ///
    /// // Returns None instead of panicking or producing NaN
    /// assert!(result.is_none());
    /// ```
    ///
    /// Practical use case - validating user input:
    ///
    /// ```
    /// use nalgebra::{Unit, Vector3};
    ///
    /// fn create_camera_direction(x: f64, y: f64, z: f64) -> Option<Unit<Vector3<f64>>> {
    ///     let direction = Vector3::new(x, y, z);
    ///     // Use a small epsilon to catch zero or near-zero vectors
    ///     Unit::try_new(direction, 1e-6)
    /// }
    ///
    /// // Valid direction
    /// assert!(create_camera_direction(1.0, 0.0, 0.0).is_some());
    ///
    /// // Invalid (zero) direction
    /// assert!(create_camera_direction(0.0, 0.0, 0.0).is_none());
    /// ```
    ///
    /// # See Also
    ///
    /// - [`new_normalize`](Self::new_normalize) - Normalizes without checking (may panic)
    /// - [`try_new_and_get`](Self::try_new_and_get) - Also returns the original norm
    /// - [`new_unchecked`](Self::new_unchecked) - Creates a `Unit` without normalization
    #[inline]
    pub fn try_new(value: T, min_norm: T::Norm) -> Option<Self>
    where
        T::Norm: RealField,
    {
        Self::try_new_and_get(value, min_norm).map(|res| res.0)
    }

    /// Normalizes the given value and returns both the `Unit` wrapper and the original norm.
    ///
    /// This function is similar to [`new_normalize`](Self::new_normalize), but it also returns
    /// the norm (length) of the original value before normalization. This is useful when you
    /// need both the direction (unit vector) and the magnitude of the original vector.
    ///
    /// The `Unit` wrapper ensures the returned value has a norm of exactly 1, while the second
    /// element of the tuple contains the original norm. This is commonly used in physics
    /// simulations where you need to separate direction from magnitude, or in graphics
    /// applications where you need to normalize vectors while preserving their original lengths.
    ///
    /// # Returns
    ///
    /// A tuple `(Unit<T>, T::Norm)` where:
    /// - The first element is the normalized value wrapped in `Unit`
    /// - The second element is the norm of the original value
    ///
    /// # Examples
    ///
    /// Getting both direction and magnitude:
    ///
    /// ```
    /// use nalgebra::{Unit, Vector2};
    ///
    /// let v = Vector2::new(3.0, 4.0);
    /// let (unit_v, original_length) = Unit::new_and_get(v);
    ///
    /// // The unit vector has length 1.0
    /// assert!((unit_v.norm() - 1.0).abs() < 1e-10);
    ///
    /// // The original length was 5.0 (from 3-4-5 triangle)
    /// assert!((original_length - 5.0).abs() < 1e-10);
    ///
    /// // We can reconstruct the original vector if needed
    /// let reconstructed = unit_v.into_inner() * original_length;
    /// assert!((reconstructed.x - 3.0).abs() < 1e-10);
    /// assert!((reconstructed.y - 4.0).abs() < 1e-10);
    /// ```
    ///
    /// Practical use case - velocity in physics:
    ///
    /// ```
    /// use nalgebra::{Unit, Vector3};
    ///
    /// // An object moving with velocity
    /// let velocity = Vector3::new(3.0, 4.0, 0.0);
    ///
    /// // Separate direction from speed
    /// let (direction, speed) = Unit::new_and_get(velocity);
    ///
    /// println!("Speed: {}", speed);  // 5.0 units per second
    /// println!("Direction: {:?}", direction.as_ref());  // Normalized vector
    ///
    /// // Later, we can apply a different speed to the same direction
    /// let new_velocity = direction.into_inner() * (speed * 2.0);  // Double the speed
    /// ```
    ///
    /// Working with surface normals and area:
    ///
    /// ```
    /// use nalgebra::{Unit, Vector3};
    ///
    /// // A surface normal that also encodes area information
    /// let surface_vector = Vector3::new(0.0, 0.0, 10.0);
    /// let (normal, area) = Unit::new_and_get(surface_vector);
    ///
    /// // Normal is the direction perpendicular to the surface
    /// assert!((normal.z - 1.0).abs() < 1e-10);
    ///
    /// // Area is encoded in the magnitude
    /// assert!((area - 10.0).abs() < 1e-10);
    /// ```
    ///
    /// # See Also
    ///
    /// - [`new_normalize`](Self::new_normalize) - Normalizes without returning the original norm
    /// - [`try_new_and_get`](Self::try_new_and_get) - Safe version that handles zero-length inputs
    /// - [`into_inner`](Self::into_inner) - Extracts the wrapped value from a `Unit`
    #[inline]
    pub fn new_and_get(mut value: T) -> (Self, T::Norm) {
        let n = value.norm();
        value.unscale_mut(n.clone());
        (Unit { value }, n)
    }

    /// Attempts to normalize the given value and returns both the `Unit` wrapper and original norm.
    ///
    /// This is the **safe version** of [`new_and_get`](Self::new_and_get). It combines the
    /// safety of [`try_new`](Self::try_new) with the additional information returned by
    /// [`new_and_get`](Self::new_and_get), making it ideal for scenarios where you need both
    /// the normalized direction and the original magnitude, but want to handle zero-length
    /// or very small inputs gracefully.
    ///
    /// The `Unit` wrapper guarantees that the wrapped value has a norm of exactly 1, while
    /// preserving the original norm information. This function will not panic or produce `NaN`
    /// values when given degenerate inputs.
    ///
    /// # Arguments
    ///
    /// * `value` - The value to normalize (typically a vector)
    /// * `min_norm` - The minimum acceptable norm. If the input's norm is less than or equal
    ///   to this value, the function returns `None`
    ///
    /// # Returns
    ///
    /// - `Some((Unit<T>, T::Norm))` if normalization succeeds, where the first element is the
    ///   normalized value and the second is the original norm
    /// - `None` if the norm is too small (norm <= min_norm)
    ///
    /// # Examples
    ///
    /// Successfully normalizing and getting the magnitude:
    ///
    /// ```
    /// use nalgebra::{Unit, Vector3};
    ///
    /// let v = Vector3::new(1.0, 2.0, 2.0);
    /// let result = Unit::try_new_and_get(v, 1e-10);
    ///
    /// assert!(result.is_some());
    /// let (unit_v, magnitude) = result.unwrap();
    ///
    /// // The vector is now normalized
    /// assert!((unit_v.norm() - 1.0).abs() < 1e-10);
    ///
    /// // Original length was 3.0
    /// assert!((magnitude - 3.0).abs() < 1e-10);
    /// ```
    ///
    /// Safely handling zero or near-zero vectors:
    ///
    /// ```
    /// use nalgebra::{Unit, Vector2};
    ///
    /// let tiny = Vector2::new(1e-12, 1e-12);
    /// let result = Unit::try_new_and_get(tiny, 1e-6);
    ///
    /// // Returns None for vectors that are too small
    /// assert!(result.is_none());
    /// ```
    ///
    /// Practical use case - processing user input for physics:
    ///
    /// ```
    /// use nalgebra::{Unit, Vector3};
    ///
    /// fn apply_force(force_vector: Vector3<f64>) -> Option<(Unit<Vector3<f64>>, f64)> {
    ///     // Only apply forces above a minimum threshold
    ///     Unit::try_new_and_get(force_vector, 0.001)
    /// }
    ///
    /// // Strong force - processes successfully
    /// let strong_force = Vector3::new(10.0, 0.0, 0.0);
    /// if let Some((direction, magnitude)) = apply_force(strong_force) {
    ///     println!("Applying force of {} in direction {:?}", magnitude, direction.as_ref());
    /// }
    ///
    /// // Weak force - ignored
    /// let weak_force = Vector3::new(0.0001, 0.0, 0.0);
    /// assert!(apply_force(weak_force).is_none());
    /// ```
    ///
    /// Separating velocity into speed and direction safely:
    ///
    /// ```
    /// use nalgebra::{Unit, Vector3};
    ///
    /// let velocity = Vector3::new(3.0, 4.0, 0.0);
    ///
    /// match Unit::try_new_and_get(velocity, 1e-6) {
    ///     Some((direction, speed)) => {
    ///         println!("Moving at {} m/s in direction {:?}", speed, direction.as_ref());
    ///         // Can now modify speed independently of direction
    ///         let new_velocity = direction.into_inner() * speed * 0.5;  // Half speed
    ///     }
    ///     None => {
    ///         println!("Object is stationary");
    ///     }
    /// }
    /// ```
    ///
    /// # See Also
    ///
    /// - [`new_and_get`](Self::new_and_get) - Unsafe version that doesn't check for zero-length
    /// - [`try_new`](Self::try_new) - Safe normalization without returning the original norm
    /// - [`new_normalize`](Self::new_normalize) - Simple normalization without safety checks
    #[inline]
    pub fn try_new_and_get(mut value: T, min_norm: T::Norm) -> Option<(Self, T::Norm)>
    where
        T::Norm: RealField,
    {
        let sq_norm = value.norm_squared();

        if sq_norm > min_norm.clone() * min_norm {
            let n = sq_norm.simd_sqrt();
            value.unscale_mut(n.clone());
            Some((Unit { value }, n))
        } else {
            None
        }
    }

    /// Re-normalizes this unit value to ensure it still has a norm of exactly 1.
    ///
    /// **Why is this needed?** Due to floating-point arithmetic inaccuracies, repeated
    /// mathematical operations on a `Unit` value can cause its norm to drift slightly away
    /// from 1.0. For example, after many rotations or transformations, a unit vector might
    /// have a norm of 1.0001 or 0.9999 instead of exactly 1.0.
    ///
    /// This function recalculates the norm and normalizes the value again, correcting any
    /// accumulated drift. It returns the norm before re-normalization, which tells you how
    /// much drift had occurred.
    ///
    /// The `Unit` wrapper is designed to maintain the guarantee that the wrapped value has
    /// a unit norm, but floating-point errors can accumulate over many operations. Use this
    /// function periodically when precision is critical.
    ///
    /// # Returns
    ///
    /// The norm of the value before re-normalization. If this is close to 1.0, there was
    /// little drift. If it's significantly different from 1.0, the value had drifted and
    /// needed correction.
    ///
    /// # Examples
    ///
    /// Detecting and correcting drift:
    ///
    /// ```
    /// use nalgebra::{Unit, Vector3};
    ///
    /// let mut unit_v = Unit::new_normalize(Vector3::new(1.0, 0.0, 0.0));
    ///
    /// // Simulate many operations that might cause drift
    /// for _ in 0..1000 {
    ///     let temp = unit_v.into_inner() * 1.00001;  // Slight scaling
    ///     unit_v = Unit::new_unchecked(temp);  // Bypass normalization
    /// }
    ///
    /// // Check the norm before renormalization
    /// let old_norm = unit_v.renormalize();
    /// println!("Norm before correction: {}", old_norm);
    ///
    /// // Now it's back to exactly 1.0
    /// assert!((unit_v.norm() - 1.0).abs() < 1e-10);
    /// ```
    ///
    /// Periodic renormalization in a simulation loop:
    ///
    /// ```
    /// use nalgebra::{Unit, Vector3};
    ///
    /// let mut direction = Unit::new_normalize(Vector3::new(1.0, 1.0, 0.0));
    ///
    /// // In a game loop or physics simulation
    /// for frame in 0..100 {
    ///     // ... perform various operations on direction ...
    ///
    ///     // Renormalize every 10 frames to prevent drift accumulation
    ///     if frame % 10 == 0 {
    ///         let drift = direction.renormalize();
    ///         if (drift - 1.0).abs() > 0.01 {
    ///             println!("Warning: significant drift detected: {}", drift);
    ///         }
    ///     }
    /// }
    /// ```
    ///
    /// Comparing drift with different operations:
    ///
    /// ```
    /// use nalgebra::{Unit, Vector2};
    ///
    /// let mut v1 = Unit::new_normalize(Vector2::new(1.0, 1.0));
    ///
    /// // Perform operations that might cause drift
    /// for _ in 0..100 {
    ///     let temp = v1.into_inner();
    ///     let rotated = Vector2::new(-temp.y, temp.x);  // 90 degree rotation
    ///     v1 = Unit::new_unchecked(rotated);
    /// }
    ///
    /// let norm_before = v1.renormalize();
    /// println!("Accumulated error: {}", (norm_before - 1.0).abs());
    /// ```
    ///
    /// # See Also
    ///
    /// - [`renormalize_fast`](Self::renormalize_fast) - Faster approximation using Taylor series
    /// - [`new_normalize`](Self::new_normalize) - Initial normalization when creating a `Unit`
    /// - [`try_new`](Self::try_new) - Safe normalization that checks for zero-length inputs
    #[inline]
    pub fn renormalize(&mut self) -> T::Norm {
        let n = self.norm();
        self.value.unscale_mut(n.clone());
        n
    }

    /// Re-normalizes this unit value using a fast approximation (first-order Taylor series).
    ///
    /// This function serves the same purpose as [`renormalize`](Self::renormalize) - correcting
    /// floating-point drift that can accumulate over many operations. However, it uses a
    /// **faster mathematical approximation** based on a first-order Taylor expansion instead
    /// of computing the full norm.
    ///
    /// **The `Unit` wrapper** guarantees that wrapped values have a norm of 1, but numerical
    /// errors can cause drift. This fast method is ideal when you need to renormalize frequently
    /// (e.g., every frame in a game) and the drift is expected to be small.
    ///
    /// # When to Use This
    ///
    /// - **Use `renormalize_fast`** when:
    ///   - You renormalize frequently (e.g., every frame)
    ///   - The drift from 1.0 is expected to be very small (< 0.1)
    ///   - Performance is critical
    ///
    /// - **Use `renormalize`** when:
    ///   - The drift might be significant
    ///   - You need maximum precision
    ///   - You renormalize infrequently
    ///
    /// # Mathematical Details
    ///
    /// This uses the approximation: `normalized ≈ v * 0.5 * (3 - ||v||²)`
    ///
    /// This is accurate when ||v|| is close to 1, but becomes less accurate as the norm
    /// drifts further from 1.
    ///
    /// # Examples
    ///
    /// Basic usage with small drift:
    ///
    /// ```
    /// use nalgebra::{Unit, Vector3};
    ///
    /// let mut direction = Unit::new_normalize(Vector3::new(1.0, 0.0, 0.0));
    ///
    /// // Simulate a small drift (norm becomes 1.001)
    /// let temp = direction.into_inner() * 1.001;
    /// direction = Unit::new_unchecked(temp);
    ///
    /// // Fast renormalization
    /// direction.renormalize_fast();
    ///
    /// // Back to approximately 1.0
    /// assert!((direction.norm() - 1.0).abs() < 1e-6);
    /// ```
    ///
    /// Performance-critical game loop:
    ///
    /// ```
    /// use nalgebra::{Unit, Vector3};
    ///
    /// let mut camera_direction = Unit::new_normalize(Vector3::new(0.0, 0.0, -1.0));
    /// let mut up_vector = Unit::new_normalize(Vector3::new(0.0, 1.0, 0.0));
    ///
    /// // In a 60 FPS game loop
    /// for _frame in 0..60 {
    ///     // ... perform camera rotations and movements ...
    ///
    ///     // Quick renormalization each frame to prevent drift
    ///     camera_direction.renormalize_fast();
    ///     up_vector.renormalize_fast();
    /// }
    /// ```
    ///
    /// Comparing fast vs. accurate renormalization:
    ///
    /// ```
    /// use nalgebra::{Unit, Vector2};
    ///
    /// // Create two identical vectors with some drift
    /// let mut v1 = Unit::new_normalize(Vector2::new(1.0, 1.0));
    /// let mut v2 = v1.clone();
    ///
    /// // Introduce drift
    /// v1 = Unit::new_unchecked(v1.into_inner() * 1.01);
    /// v2 = Unit::new_unchecked(v2.into_inner() * 1.01);
    ///
    /// // Compare methods
    /// v1.renormalize_fast();  // Fast approximation
    /// v2.renormalize();       // Accurate method
    ///
    /// // Both should be close to 1.0, but v2 might be slightly more accurate
    /// println!("Fast: {}", v1.norm());
    /// println!("Accurate: {}", v2.norm());
    /// ```
    ///
    /// # See Also
    ///
    /// - [`renormalize`](Self::renormalize) - More accurate but slower renormalization
    /// - [`new_normalize`](Self::new_normalize) - Initial normalization
    /// - [`new_unchecked`](Self::new_unchecked) - Create a `Unit` without normalization
    #[inline]
    pub fn renormalize_fast(&mut self) {
        let sq_norm = self.value.norm_squared();
        let three: T::Norm = crate::convert(3.0);
        let half: T::Norm = crate::convert(0.5);
        self.value.scale_mut(half * (three - sq_norm));
    }
}

/// # Data extraction and construction without normalization
impl<T> Unit<T> {
    /// Wraps the given value in a `Unit` without normalizing it.
    ///
    /// **⚠️ Important: This is an "unchecked" function.** Unlike [`new_normalize`](Self::new_normalize)
    /// or [`try_new`](Self::try_new), this function does **not** verify that the input has a
    /// norm of 1.0. You are promising that the value is already normalized.
    ///
    /// The **`Unit` wrapper** is a type-level guarantee that a value (typically a vector) has
    /// a norm of exactly 1. This is crucial for many geometric and mathematical operations
    /// that require normalized vectors (like representing directions, rotations, or surface normals).
    ///
    /// # When to Use This
    ///
    /// Use `new_unchecked` when:
    /// - You **know for certain** the value is already normalized (e.g., from a lookup table
    ///   of pre-computed unit vectors)
    /// - You're constructing from components that are mathematically guaranteed to be normalized
    /// - You need maximum performance and can guarantee normalization through other means
    ///
    /// # Safety Considerations
    ///
    /// While this function is not marked as `unsafe` in the Rust sense (it won't cause memory
    /// unsafety), using it with a non-normalized value **violates the contract** of the `Unit`
    /// type. This can lead to:
    /// - Incorrect mathematical results in calculations
    /// - Assertion failures in debug builds of some operations
    /// - Subtle bugs that are hard to track down
    ///
    /// If you're not absolutely certain the value is normalized, use [`new_normalize`](Self::new_normalize)
    /// or [`try_new`](Self::try_new) instead.
    ///
    /// # Examples
    ///
    /// Creating from mathematically guaranteed unit vectors:
    ///
    /// ```
    /// use nalgebra::{Unit, Vector3};
    ///
    /// // These are the standard basis vectors - mathematically guaranteed to be unit vectors
    /// let x_axis = Unit::new_unchecked(Vector3::new(1.0, 0.0, 0.0));
    /// let y_axis = Unit::new_unchecked(Vector3::new(0.0, 1.0, 0.0));
    /// let z_axis = Unit::new_unchecked(Vector3::new(0.0, 0.0, 1.0));
    ///
    /// assert!((x_axis.norm() - 1.0).abs() < 1e-10);
    /// assert!((y_axis.norm() - 1.0).abs() < 1e-10);
    /// assert!((z_axis.norm() - 1.0).abs() < 1e-10);
    /// ```
    ///
    /// Using pre-computed lookup table values:
    ///
    /// ```
    /// use nalgebra::{Unit, Vector2};
    ///
    /// // Lookup table of pre-computed unit vectors for common angles
    /// const NORTH: Vector2<f64> = Vector2::new(0.0, 1.0);
    /// const EAST: Vector2<f64> = Vector2::new(1.0, 0.0);
    /// const SOUTH: Vector2<f64> = Vector2::new(0.0, -1.0);
    /// const WEST: Vector2<f64> = Vector2::new(-1.0, 0.0);
    ///
    /// // Since these are pre-computed and verified, we can use new_unchecked
    /// let direction = Unit::new_unchecked(NORTH);
    /// ```
    ///
    /// Converting from spherical coordinates (mathematically guaranteed):
    ///
    /// ```
    /// use nalgebra::{Unit, Vector3};
    ///
    /// fn direction_from_angles(theta: f64, phi: f64) -> Unit<Vector3<f64>> {
    ///     // Spherical to Cartesian conversion always produces a unit vector
    ///     let x = theta.sin() * phi.cos();
    ///     let y = theta.sin() * phi.sin();
    ///     let z = theta.cos();
    ///
    ///     // We know this is normalized by construction (x² + y² + z² = 1)
    ///     Unit::new_unchecked(Vector3::new(x, y, z))
    /// }
    ///
    /// let dir = direction_from_angles(std::f64::consts::PI / 4.0, std::f64::consts::PI / 6.0);
    /// assert!((dir.norm() - 1.0).abs() < 1e-10);
    /// ```
    ///
    /// **Wrong usage** (will cause bugs):
    ///
    /// ```no_run
    /// use nalgebra::{Unit, Vector3};
    ///
    /// // DON'T DO THIS! This vector is not normalized.
    /// let bad = Unit::new_unchecked(Vector3::new(3.0, 4.0, 0.0));
    ///
    /// // This will not be 1.0, violating the Unit contract
    /// println!("Norm: {}", bad.norm());  // Prints 5.0, not 1.0!
    ///
    /// // Instead, use:
    /// let good = Unit::new_normalize(Vector3::new(3.0, 4.0, 0.0));
    /// ```
    ///
    /// # See Also
    ///
    /// - [`new_normalize`](Self::new_normalize) - Safe normalization that always works
    /// - [`try_new`](Self::try_new) - Safe normalization with zero-check
    /// - [`into_inner`](Self::into_inner) - Extracts the wrapped value
    #[inline]
    pub const fn new_unchecked(value: T) -> Self {
        Unit { value }
    }

    /// Wraps the given reference, assuming it is already normalized.
    #[inline]
    pub const fn from_ref_unchecked(value: &T) -> &Self {
        unsafe { &*(value as *const T as *const Self) }
    }

    /// Consumes the `Unit` wrapper and returns the underlying normalized value.
    ///
    /// This function extracts the wrapped value from a `Unit`, giving you back the raw
    /// vector (or other type) that was being protected by the `Unit` wrapper. The extracted
    /// value will still be normalized (have a norm of 1.0), but you lose the type-level
    /// guarantee that comes with the `Unit` wrapper.
    ///
    /// **Understanding `Unit`**: The `Unit` wrapper is a type-level guarantee that a value
    /// (typically a vector) has a norm of exactly 1. This is essential for many geometric
    /// operations. When you call `into_inner`, you're saying "I need the raw value and I'll
    /// take responsibility for maintaining its normalized state."
    ///
    /// # When to Use This
    ///
    /// Use `into_inner` when:
    /// - You need to perform operations that aren't available on `Unit<T>` but are on `T`
    /// - You're passing the value to an API that expects a regular vector, not a `Unit`
    /// - You need to scale the direction by a magnitude (converting direction back to velocity, etc.)
    /// - You want to store the value in a format that doesn't use `Unit`
    ///
    /// # Returns
    ///
    /// The underlying value of type `T` (typically a vector). This value will be normalized
    /// (norm = 1.0), but the type system no longer enforces this guarantee.
    ///
    /// # Examples
    ///
    /// Basic usage - extracting the value:
    ///
    /// ```
    /// use nalgebra::{Unit, Vector3};
    ///
    /// let unit_v = Unit::new_normalize(Vector3::new(3.0, 4.0, 0.0));
    ///
    /// // Extract the underlying normalized vector
    /// let normalized = unit_v.into_inner();
    ///
    /// // It's still normalized, just not wrapped in Unit anymore
    /// assert!((normalized.norm() - 1.0).abs() < 1e-10);
    /// assert!((normalized.x - 0.6).abs() < 1e-10);
    /// assert!((normalized.y - 0.8).abs() < 1e-10);
    /// ```
    ///
    /// Converting direction and speed back to velocity:
    ///
    /// ```
    /// use nalgebra::{Unit, Vector3};
    ///
    /// // We have a direction and a speed
    /// let direction = Unit::new_normalize(Vector3::new(1.0, 1.0, 0.0));
    /// let speed = 10.0;
    ///
    /// // Convert back to a velocity vector
    /// let velocity = direction.into_inner() * speed;
    ///
    /// // The velocity now has the correct magnitude
    /// assert!((velocity.norm() - speed).abs() < 1e-10);
    /// ```
    ///
    /// Working with APIs that don't use `Unit`:
    ///
    /// ```
    /// use nalgebra::{Unit, Vector3};
    ///
    /// fn some_graphics_api(normal: Vector3<f32>) {
    ///     // This function expects a regular Vector3, not Unit<Vector3>
    ///     // ...
    /// }
    ///
    /// let surface_normal = Unit::new_normalize(Vector3::new(0.0, 1.0, 0.0));
    ///
    /// // Extract the value to pass to the API
    /// some_graphics_api(surface_normal.into_inner());
    /// ```
    ///
    /// Combining direction with magnitude from different sources:
    ///
    /// ```
    /// use nalgebra::{Unit, Vector2};
    ///
    /// // Get direction from one source
    /// let move_direction = Unit::new_normalize(Vector2::new(1.0, 1.0));
    ///
    /// // Get magnitude from another source (e.g., player input)
    /// let player_speed = 5.0;
    ///
    /// // Combine them
    /// let movement_vector = move_direction.into_inner() * player_speed;
    ///
    /// println!("Moving at {:?} with speed {}", movement_vector, player_speed);
    /// ```
    ///
    /// Reconstructing original vector from normalized version and magnitude:
    ///
    /// ```
    /// use nalgebra::{Unit, Vector3};
    ///
    /// let original = Vector3::new(3.0, 4.0, 0.0);
    /// let (unit_v, magnitude) = Unit::new_and_get(original);
    ///
    /// // Later, reconstruct the original
    /// let reconstructed = unit_v.into_inner() * magnitude;
    ///
    /// assert!((reconstructed.x - 3.0).abs() < 1e-10);
    /// assert!((reconstructed.y - 4.0).abs() < 1e-10);
    /// ```
    ///
    /// # Note on Ownership
    ///
    /// This function consumes the `Unit` wrapper (takes ownership). If you need to keep
    /// the `Unit` and just access the value, use `.as_ref()` instead:
    ///
    /// ```
    /// use nalgebra::{Unit, Vector2};
    ///
    /// let unit_v = Unit::new_normalize(Vector2::new(1.0, 1.0));
    ///
    /// // Borrow the inner value without consuming the Unit
    /// let inner_ref = unit_v.as_ref();
    /// println!("x = {}, y = {}", inner_ref.x, inner_ref.y);
    ///
    /// // unit_v is still usable here
    /// assert!((unit_v.norm() - 1.0).abs() < 1e-10);
    /// ```
    ///
    /// # See Also
    ///
    /// - [`new_normalize`](Self::new_normalize) - Create a `Unit` by normalizing a value
    /// - [`new_unchecked`](Self::new_unchecked) - Wrap a value without normalization
    /// - [`new_and_get`](Self::new_and_get) - Get both the unit value and original magnitude
    #[inline]
    pub fn into_inner(self) -> T {
        self.value
    }

    /// Retrieves the underlying value.
    /// Deprecated: use [`Unit::into_inner`] instead.
    #[deprecated(note = "use `.into_inner()` instead")]
    #[inline]
    pub fn unwrap(self) -> T {
        self.value
    }

    /// Returns a mutable reference to the underlying value. This is `_unchecked` because modifying
    /// the underlying value in such a way that it no longer has unit length may lead to unexpected
    /// results.
    #[inline]
    pub const fn as_mut_unchecked(&mut self) -> &mut T {
        &mut self.value
    }
}

impl<T> AsRef<T> for Unit<T> {
    #[inline]
    fn as_ref(&self) -> &T {
        &self.value
    }
}

/*
/*
 *
 * Conversions.
 *
 */
impl<T: NormedSpace> SubsetOf<T> for Unit<T>
where T::RealField: RelativeEq
{
    #[inline]
    fn to_superset(&self) -> T {
        self.clone().into_inner()
    }

    #[inline]
    fn is_in_subset(value: &T) -> bool {
        relative_eq!(value.norm_squared(), crate::one())
    }

    #[inline]
    fn from_superset_unchecked(value: &T) -> Self {
        Unit::new_normalize(value.clone()) // We still need to re-normalize because the condition is inexact.
    }
}

// impl<T: RelativeEq> RelativeEq for Unit<T> {
//     type Epsilon = T::Epsilon;
//
//     #[inline]
//     fn default_epsilon() -> Self::Epsilon {
//         T::default_epsilon()
//     }
//
//     #[inline]
//     fn default_max_relative() -> Self::Epsilon {
//         T::default_max_relative()
//     }
//
//     #[inline]
//     fn default_max_ulps() -> u32 {
//         T::default_max_ulps()
//     }
//
//     #[inline]
//     fn relative_eq(&self, other: &Self, epsilon: Self::Epsilon, max_relative: Self::Epsilon) -> bool {
//         self.value.relative_eq(&other.value, epsilon, max_relative)
//     }
//
//     #[inline]
//     fn ulps_eq(&self, other: &Self, epsilon: Self::Epsilon, max_ulps: u32) -> bool {
//         self.value.ulps_eq(&other.value, epsilon, max_ulps)
//     }
// }
*/
// TODO:re-enable this impl when specialization is possible.
// Currently, it is disabled so that we can have a nice output for the `UnitQuaternion` display.
/*
impl<T: fmt::Display> fmt::Display for Unit<T> {
    // XXX: will not always work correctly due to rounding errors.
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.value.fmt(f)
    }
}
*/

impl<T> Deref for Unit<T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &T {
        unsafe { &*(self as *const Self as *const T) }
    }
}

// NOTE: we can't use a generic implementation for `Unit<T>` because
// num_complex::Complex does not implement `From[Complex<...>...]` (and can't
// because of the orphan rules).
impl<T: Scalar + simba::simd::PrimitiveSimdValue, R: Dim, C: Dim>
    From<[Unit<OMatrix<T::Element, R, C>>; 2]> for Unit<OMatrix<T, R, C>>
where
    T: From<[<T as simba::simd::SimdValue>::Element; 2]>,
    T::Element: Scalar,
    DefaultAllocator: Allocator<R, C>,
{
    #[inline]
    fn from(arr: [Unit<OMatrix<T::Element, R, C>>; 2]) -> Self {
        Self::new_unchecked(OMatrix::from([
            arr[0].clone().into_inner(),
            arr[1].clone().into_inner(),
        ]))
    }
}

impl<T: Scalar + simba::simd::PrimitiveSimdValue, R: Dim, C: Dim>
    From<[Unit<OMatrix<T::Element, R, C>>; 4]> for Unit<OMatrix<T, R, C>>
where
    T: From<[<T as simba::simd::SimdValue>::Element; 4]>,
    T::Element: Scalar,
    DefaultAllocator: Allocator<R, C>,
{
    #[inline]
    fn from(arr: [Unit<OMatrix<T::Element, R, C>>; 4]) -> Self {
        Self::new_unchecked(OMatrix::from([
            arr[0].clone().into_inner(),
            arr[1].clone().into_inner(),
            arr[2].clone().into_inner(),
            arr[3].clone().into_inner(),
        ]))
    }
}

impl<T: Scalar + simba::simd::PrimitiveSimdValue, R: Dim, C: Dim>
    From<[Unit<OMatrix<T::Element, R, C>>; 8]> for Unit<OMatrix<T, R, C>>
where
    T: From<[<T as simba::simd::SimdValue>::Element; 8]>,
    T::Element: Scalar,
    DefaultAllocator: Allocator<R, C>,
{
    #[inline]
    fn from(arr: [Unit<OMatrix<T::Element, R, C>>; 8]) -> Self {
        Self::new_unchecked(OMatrix::from([
            arr[0].clone().into_inner(),
            arr[1].clone().into_inner(),
            arr[2].clone().into_inner(),
            arr[3].clone().into_inner(),
            arr[4].clone().into_inner(),
            arr[5].clone().into_inner(),
            arr[6].clone().into_inner(),
            arr[7].clone().into_inner(),
        ]))
    }
}

impl<T: Scalar + simba::simd::PrimitiveSimdValue, R: Dim, C: Dim>
    From<[Unit<OMatrix<T::Element, R, C>>; 16]> for Unit<OMatrix<T, R, C>>
where
    T: From<[<T as simba::simd::SimdValue>::Element; 16]>,
    T::Element: Scalar,
    DefaultAllocator: Allocator<R, C>,
{
    #[inline]
    fn from(arr: [Unit<OMatrix<T::Element, R, C>>; 16]) -> Self {
        Self::new_unchecked(OMatrix::from([
            arr[0].clone().into_inner(),
            arr[1].clone().into_inner(),
            arr[2].clone().into_inner(),
            arr[3].clone().into_inner(),
            arr[4].clone().into_inner(),
            arr[5].clone().into_inner(),
            arr[6].clone().into_inner(),
            arr[7].clone().into_inner(),
            arr[8].clone().into_inner(),
            arr[9].clone().into_inner(),
            arr[10].clone().into_inner(),
            arr[11].clone().into_inner(),
            arr[12].clone().into_inner(),
            arr[13].clone().into_inner(),
            arr[14].clone().into_inner(),
            arr[15].clone().into_inner(),
        ]))
    }
}
