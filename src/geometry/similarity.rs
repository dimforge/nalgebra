// Needed otherwise the rkyv macros generate code incompatible with rust-2024
#![cfg_attr(feature = "rkyv-serialize", allow(unsafe_op_in_unsafe_fn))]

use approx::{AbsDiffEq, RelativeEq, UlpsEq};
use num::Zero;
use std::fmt;
use std::hash;

#[cfg(feature = "serde-serialize-no-std")]
use serde::{Deserialize, Serialize};

use simba::scalar::{RealField, SubsetOf};
use simba::simd::SimdRealField;

use crate::base::allocator::Allocator;
use crate::base::dimension::{DimNameAdd, DimNameSum, U1};
use crate::base::storage::Owned;
use crate::base::{Const, DefaultAllocator, OMatrix, SVector, Scalar};
use crate::geometry::{AbstractRotation, Isometry, Point, Translation};

#[cfg(feature = "rkyv-serialize")]
use rkyv::bytecheck;

/// A similarity transformation in N-dimensional space.
///
/// A similarity transformation combines three operations applied in this order:
/// 1. **Uniform scaling** - Resizes objects proportionally in all directions
/// 2. **Rotation** - Rotates objects around the origin
/// 3. **Translation** - Moves objects to a new position
///
/// # What is a Similarity Transformation?
///
/// Similarity transformations preserve the *shape* of objects while allowing their size,
/// orientation, and position to change. Unlike general transformations, similarities maintain
/// angles between lines and the proportions between distances. This makes them ideal for:
///
/// - **Game development**: Resizing game objects while preserving their proportions
/// - **Computer graphics**: Implementing camera zoom without distortion
/// - **Computer vision**: Matching objects at different scales and orientations
/// - **Animation**: Smoothly scaling and moving characters
///
/// # Example: Basic Usage
///
/// ```
/// # use nalgebra::{Similarity2, Vector2, Point2};
/// # use std::f32;
/// // Create a similarity: translate by (1, 2), rotate by 90°, scale by 2.0
/// let sim = Similarity2::new(
///     Vector2::new(1.0, 2.0),  // translation
///     f32::consts::FRAC_PI_2,   // rotation (90 degrees)
///     2.0                        // uniform scaling
/// );
///
/// // Transform a point
/// let point = Point2::new(1.0, 0.0);
/// let transformed = sim * point;
///
/// // The point is first scaled, then rotated, then translated
/// ```
///
/// # Example: Game Object Resizing
///
/// ```
/// # use nalgebra::{Similarity3, Vector3, Point3};
/// // Make a game object 1.5x larger while keeping it at position (10, 5, 0)
/// let sim = Similarity3::new(
///     Vector3::new(10.0, 5.0, 0.0),  // position
///     Vector3::zeros(),               // no rotation
///     1.5                             // 50% larger
/// );
///
/// // Transform the object's vertices
/// let vertex = Point3::new(1.0, 1.0, 1.0);
/// let scaled_vertex = sim * vertex;
/// ```
#[repr(C)]
#[derive(Debug, Copy, Clone)]
#[cfg_attr(feature = "serde-serialize-no-std", derive(Serialize, Deserialize))]
#[cfg_attr(
    feature = "serde-serialize-no-std",
    serde(bound(serialize = "T: Scalar + Serialize,
                     R: Serialize,
                     DefaultAllocator: Allocator<Const<D>>,
                     Owned<T, Const<D>>: Serialize"))
)]
#[cfg_attr(
    feature = "serde-serialize-no-std",
    serde(bound(deserialize = "T: Scalar + Deserialize<'de>,
                       R: Deserialize<'de>,
                       DefaultAllocator: Allocator<Const<D>>,
                       Owned<T, Const<D>>: Deserialize<'de>"))
)]
#[cfg_attr(feature = "rkyv-serialize", derive(bytecheck::CheckBytes))]
#[cfg_attr(
    feature = "rkyv-serialize-no-std",
    derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize),
    archive(
        as = "Similarity<T::Archived, R::Archived, D>",
        bound(archive = "
        T: rkyv::Archive,
        R: rkyv::Archive,
        Isometry<T, R, D>: rkyv::Archive<Archived = Isometry<T::Archived, R::Archived, D>>
    ")
    )
)]
#[cfg_attr(feature = "defmt", derive(defmt::Format))]
pub struct Similarity<T, R, const D: usize> {
    /// The part of this similarity that does not include the scaling factor.
    pub isometry: Isometry<T, R, D>,
    scaling: T,
}

impl<T: Scalar + hash::Hash, R: hash::Hash, const D: usize> hash::Hash for Similarity<T, R, D>
where
    Owned<T, Const<D>>: hash::Hash,
{
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        self.isometry.hash(state);
        self.scaling.hash(state);
    }
}

#[cfg(feature = "bytemuck")]
unsafe impl<T: Scalar, R, const D: usize> bytemuck::Zeroable for Similarity<T, R, D> where
    Isometry<T, R, D>: bytemuck::Zeroable
{
}

#[cfg(feature = "bytemuck")]
unsafe impl<T: Scalar, R, const D: usize> bytemuck::Pod for Similarity<T, R, D>
where
    Isometry<T, R, D>: bytemuck::Pod,
    R: Copy,
    T: Copy,
{
}

impl<T: Scalar + Zero, R, const D: usize> Similarity<T, R, D>
where
    R: AbstractRotation<T, D>,
{
    /// Creates a new similarity transformation from its component parts.
    ///
    /// A similarity transformation combines scaling, rotation, and translation. This constructor
    /// lets you specify each component separately, giving you full control over the transformation.
    ///
    /// # Parameters
    ///
    /// * `translation` - The translation component (movement in space)
    /// * `rotation` - The rotation component (orientation change)
    /// * `scaling` - The uniform scaling factor (must be non-zero)
    ///
    /// # Panics
    ///
    /// Panics if `scaling` is zero, as this would create a degenerate transformation.
    ///
    /// # Example: 2D Similarity
    ///
    /// ```
    /// # use nalgebra::{Similarity2, Translation2, UnitComplex, Point2};
    /// # use std::f32;
    /// // Create components separately
    /// let translation = Translation2::new(5.0, 10.0);
    /// let rotation = UnitComplex::new(f32::consts::FRAC_PI_4); // 45 degrees
    /// let scaling = 2.0;
    ///
    /// // Combine into a similarity transformation
    /// let sim = Similarity2::from_parts(translation, rotation, scaling);
    ///
    /// // Apply to a point
    /// let point = Point2::new(1.0, 0.0);
    /// let result = sim * point;
    /// ```
    ///
    /// # Example: 3D Similarity
    ///
    /// ```
    /// # use nalgebra::{Similarity3, Translation3, UnitQuaternion, Vector3, Point3};
    /// # use std::f32;
    /// // Create a 3D similarity transformation
    /// let translation = Translation3::new(1.0, 2.0, 3.0);
    /// let axis = Vector3::y_axis();
    /// let rotation = UnitQuaternion::from_axis_angle(&axis, f32::consts::FRAC_PI_2);
    /// let scaling = 0.5; // Half size
    ///
    /// let sim = Similarity3::from_parts(translation, rotation, scaling);
    ///
    /// let point = Point3::new(2.0, 0.0, 0.0);
    /// let transformed = sim * point;
    /// ```
    ///
    /// # See Also
    ///
    /// * [`from_isometry`](Self::from_isometry) - Create from an isometry and scaling
    /// * [`new`](Self::new) - Convenient constructor for common cases
    #[inline]
    pub fn from_parts(translation: Translation<T, D>, rotation: R, scaling: T) -> Self {
        Self::from_isometry(Isometry::from_parts(translation, rotation), scaling)
    }

    /// Creates a new similarity from an isometry and a scaling factor.
    ///
    /// An isometry is a transformation that preserves distances (rotation + translation).
    /// This method adds a uniform scaling component to an existing isometry, converting
    /// it into a similarity transformation.
    ///
    /// This is particularly useful when you already have an isometry (e.g., from a camera
    /// or object pose) and want to add scaling without decomposing and reconstructing
    /// the entire transformation.
    ///
    /// # Parameters
    ///
    /// * `isometry` - The isometry (rotation + translation) component
    /// * `scaling` - The uniform scaling factor (must be non-zero)
    ///
    /// # Panics
    ///
    /// Panics if `scaling` is zero, as this would create a degenerate transformation.
    ///
    /// # Example: Adding Scale to an Isometry
    ///
    /// ```
    /// # use nalgebra::{Similarity2, Isometry2, Vector2, Point2};
    /// # use std::f32;
    /// // Start with an isometry (rotation + translation, no scaling)
    /// let iso = Isometry2::new(
    ///     Vector2::new(1.0, 2.0),
    ///     f32::consts::FRAC_PI_4
    /// );
    ///
    /// // Add uniform scaling to create a similarity
    /// let sim = Similarity2::from_isometry(iso, 2.5);
    ///
    /// // The similarity now scales points by 2.5x
    /// let point = Point2::new(1.0, 0.0);
    /// let result = sim * point;
    /// ```
    ///
    /// # Example: Camera Zoom
    ///
    /// ```
    /// # use nalgebra::{Similarity3, Isometry3, Vector3, Point3};
    /// // Camera position and orientation (isometry)
    /// let camera_pose = Isometry3::new(
    ///     Vector3::new(0.0, 5.0, 10.0),
    ///     Vector3::zeros()
    /// );
    ///
    /// // Add 2x zoom by scaling
    /// let zoomed_camera = Similarity3::from_isometry(camera_pose, 2.0);
    ///
    /// // Objects viewed through this camera appear 2x larger
    /// ```
    ///
    /// # See Also
    ///
    /// * [`from_parts`](Self::from_parts) - Create from translation, rotation, and scaling
    /// * [`from_scaling`](Self::from_scaling) - Create a pure scaling transformation
    #[inline]
    pub fn from_isometry(isometry: Isometry<T, R, D>, scaling: T) -> Self {
        assert!(!scaling.is_zero(), "The scaling factor must not be zero.");

        Self { isometry, scaling }
    }

    /// Changes the scaling factor of this similarity transformation.
    ///
    /// This modifies the similarity in-place, updating only the scaling component while
    /// keeping the rotation and translation unchanged. This is useful for animations or
    /// dynamic resizing effects.
    ///
    /// # Parameters
    ///
    /// * `scaling` - The new uniform scaling factor (must be non-zero)
    ///
    /// # Panics
    ///
    /// Panics if `scaling` is zero.
    ///
    /// # Example: Animating Object Size
    ///
    /// ```
    /// # use nalgebra::{Similarity2, Vector2};
    /// # use std::f32;
    /// let mut sim = Similarity2::new(
    ///     Vector2::new(10.0, 5.0),
    ///     0.0,
    ///     1.0  // Start at normal size
    /// );
    ///
    /// // Animate growth over time
    /// for i in 1..=5 {
    ///     let scale = 1.0 + (i as f32) * 0.2;
    ///     sim.set_scaling(scale);
    ///     // Object grows by 20% each step: 1.2x, 1.4x, 1.6x, 1.8x, 2.0x
    /// }
    /// ```
    ///
    /// # Example: Power-up Effect
    ///
    /// ```
    /// # use nalgebra::{Similarity3, Vector3};
    /// // Game object transformation
    /// let mut player_transform = Similarity3::new(
    ///     Vector3::new(0.0, 0.0, 0.0),
    ///     Vector3::zeros(),
    ///     1.0
    /// );
    ///
    /// // Player gets a power-up that doubles their size
    /// player_transform.set_scaling(2.0);
    /// ```
    ///
    /// # See Also
    ///
    /// * [`scaling`](Self::scaling) - Get the current scaling factor
    /// * [`append_scaling`](Self::append_scaling) - Add additional scaling
    /// * [`prepend_scaling`](Self::prepend_scaling) - Add scaling before current transformation
    #[inline]
    pub fn set_scaling(&mut self, scaling: T) {
        assert!(
            !scaling.is_zero(),
            "The similarity scaling factor must not be zero."
        );

        self.scaling = scaling;
    }
}

impl<T: Scalar, R, const D: usize> Similarity<T, R, D> {
    /// Returns the uniform scaling factor of this similarity transformation.
    ///
    /// The scaling factor determines how much objects are resized during the transformation.
    /// A scaling factor of 1.0 means no size change, values > 1.0 make objects larger,
    /// and values < 1.0 (but > 0.0) make objects smaller.
    ///
    /// # Returns
    ///
    /// The uniform scaling factor applied by this similarity.
    ///
    /// # Example: Reading the Scale
    ///
    /// ```
    /// # use nalgebra::{Similarity2, Vector2};
    /// let sim = Similarity2::new(Vector2::new(1.0, 2.0), 0.0, 3.5);
    /// assert_eq!(sim.scaling(), 3.5);
    /// ```
    ///
    /// # Example: Checking Object Size
    ///
    /// ```
    /// # use nalgebra::{Similarity3, Vector3};
    /// let transform = Similarity3::new(
    ///     Vector3::zeros(),
    ///     Vector3::zeros(),
    ///     0.5
    /// );
    ///
    /// // Check if object is scaled down
    /// if transform.scaling() < 1.0 {
    ///     println!("Object is smaller than original");
    /// }
    /// ```
    ///
    /// # Example: Proportional Resizing
    ///
    /// ```
    /// # use nalgebra::{Similarity2, Vector2};
    /// let original = Similarity2::new(Vector2::zeros(), 0.0, 2.0);
    ///
    /// // Create a new similarity with doubled scaling
    /// let doubled = Similarity2::new(
    ///     Vector2::zeros(),
    ///     0.0,
    ///     original.scaling() * 2.0  // Now 4.0x
    /// );
    /// ```
    ///
    /// # See Also
    ///
    /// * [`set_scaling`](Self::set_scaling) - Change the scaling factor
    /// * [`from_scaling`](Self::from_scaling) - Create a pure scaling transformation
    #[inline]
    #[must_use]
    pub fn scaling(&self) -> T {
        self.scaling.clone()
    }
}

impl<T: SimdRealField, R, const D: usize> Similarity<T, R, D>
where
    T::Element: SimdRealField,
    R: AbstractRotation<T, D>,
{
    /// Creates a similarity transformation that only applies uniform scaling.
    ///
    /// This creates a similarity with no rotation or translation - only scaling. The resulting
    /// transformation will resize objects uniformly in all directions around the origin.
    /// This is useful for simple zoom effects or resizing operations.
    ///
    /// # Parameters
    ///
    /// * `scaling` - The uniform scaling factor (must be non-zero)
    ///
    /// # Example: Simple Scaling
    ///
    /// ```
    /// # use nalgebra::{Similarity2, Point2};
    /// # use approx::assert_relative_eq;
    /// // Create a pure 2x scaling transformation
    /// let sim = Similarity2::from_scaling(2.0);
    ///
    /// let point = Point2::new(3.0, 4.0);
    /// let scaled = sim * point;
    ///
    /// assert_relative_eq!(scaled, Point2::new(6.0, 8.0));
    /// ```
    ///
    /// # Example: Zoom Effect
    ///
    /// ```
    /// # use nalgebra::{Similarity3, Point3};
    /// # use approx::assert_relative_eq;
    /// // Create a 1.5x zoom (no movement or rotation)
    /// let zoom = Similarity3::from_scaling(1.5);
    ///
    /// let vertex = Point3::new(2.0, 2.0, 2.0);
    /// let zoomed = zoom * vertex;
    ///
    /// assert_relative_eq!(zoomed, Point3::new(3.0, 3.0, 3.0));
    /// ```
    ///
    /// # Example: Shrinking Objects
    ///
    /// ```
    /// # use nalgebra::{Similarity2, Point2};
    /// # use approx::assert_relative_eq;
    /// // Create a 50% reduction (0.5x scale)
    /// let shrink = Similarity2::from_scaling(0.5);
    ///
    /// let point = Point2::new(10.0, 20.0);
    /// let small = shrink * point;
    ///
    /// assert_relative_eq!(small, Point2::new(5.0, 10.0));
    /// ```
    ///
    /// # See Also
    ///
    /// * [`identity`](Self::identity) - Create an identity transformation (no change)
    /// * [`from_isometry`](Self::from_isometry) - Add scaling to a rotation+translation
    #[inline]
    pub fn from_scaling(scaling: T) -> Self {
        Self::from_isometry(Isometry::identity(), scaling)
    }

    /// Computes the inverse of this similarity transformation.
    ///
    /// The inverse transformation "undoes" the effect of the original similarity. If you
    /// apply a similarity and then its inverse, you get back to where you started.
    ///
    /// For a similarity transformation with scaling `s`, rotation `R`, and translation `t`,
    /// the inverse has:
    /// - Scaling: `1/s`
    /// - Rotation: `R^-1` (inverse rotation)
    /// - Translation adjusted accordingly
    ///
    /// # Returns
    ///
    /// A new similarity that is the inverse of this one.
    ///
    /// # Example: Undoing a Transformation
    ///
    /// ```
    /// # use nalgebra::{Similarity2, Vector2, Point2};
    /// # use approx::assert_relative_eq;
    /// # use std::f32;
    /// let sim = Similarity2::new(
    ///     Vector2::new(5.0, 3.0),
    ///     f32::consts::FRAC_PI_4,
    ///     2.0
    /// );
    ///
    /// let point = Point2::new(1.0, 2.0);
    ///
    /// // Apply transformation and then its inverse
    /// let transformed = sim * point;
    /// let back = sim.inverse() * transformed;
    ///
    /// // We're back to the original point
    /// assert_relative_eq!(back, point, epsilon = 1.0e-5);
    /// ```
    ///
    /// # Example: Camera Transform Inversion
    ///
    /// ```
    /// # use nalgebra::{Similarity3, Vector3, Point3};
    /// # use approx::assert_relative_eq;
    /// // World-to-camera transformation
    /// let world_to_cam = Similarity3::new(
    ///     Vector3::new(0.0, 0.0, -10.0),
    ///     Vector3::zeros(),
    ///     1.0
    /// );
    ///
    /// // Camera-to-world is the inverse
    /// let cam_to_world = world_to_cam.inverse();
    ///
    /// // Verify they cancel out
    /// let point = Point3::new(1.0, 2.0, 3.0);
    /// let result = cam_to_world * (world_to_cam * point);
    /// assert_relative_eq!(result, point, epsilon = 1.0e-5);
    /// ```
    ///
    /// # See Also
    ///
    /// * [`inverse_mut`](Self::inverse_mut) - Inverts in-place (more efficient)
    /// * [`inverse_transform_point`](Self::inverse_transform_point) - Transform by inverse without computing it
    #[inline]
    #[must_use = "Did you mean to use inverse_mut()?"]
    pub fn inverse(&self) -> Self {
        let mut res = self.clone();
        res.inverse_mut();
        res
    }

    /// Inverts this similarity transformation in-place.
    ///
    /// This is more efficient than [`inverse`](Self::inverse) because it doesn't allocate
    /// a new similarity - it modifies the existing one. Use this when you don't need to
    /// keep the original transformation.
    ///
    /// # Example: In-Place Inversion
    ///
    /// ```
    /// # use nalgebra::{Similarity2, Vector2, Point2};
    /// # use approx::assert_relative_eq;
    /// # use std::f32;
    /// let mut sim = Similarity2::new(
    ///     Vector2::new(3.0, 4.0),
    ///     f32::consts::FRAC_PI_2,
    ///     2.0
    /// );
    ///
    /// let point = Point2::new(1.0, 0.0);
    /// let transformed = sim * point;
    ///
    /// // Invert the transformation
    /// sim.inverse_mut();
    ///
    /// // Now it undoes the original transformation
    /// let back = sim * transformed;
    /// assert_relative_eq!(back, point, epsilon = 1.0e-5);
    /// ```
    ///
    /// # Example: Toggle Between Two States
    ///
    /// ```
    /// # use nalgebra::{Similarity3, Vector3};
    /// let mut transform = Similarity3::new(
    ///     Vector3::new(5.0, 0.0, 0.0),
    ///     Vector3::zeros(),
    ///     2.0
    /// );
    ///
    /// // First state: object moved right and scaled up
    /// // ... use transform ...
    ///
    /// // Second state: invert to move left and scale down
    /// transform.inverse_mut();
    /// // ... use inverted transform ...
    ///
    /// // Can invert again to get back to first state
    /// transform.inverse_mut();
    /// ```
    ///
    /// # See Also
    ///
    /// * [`inverse`](Self::inverse) - Returns a new inverse (doesn't modify self)
    /// * [`inverse_transform_point`](Self::inverse_transform_point) - Apply inverse to a point
    #[inline]
    pub fn inverse_mut(&mut self) {
        self.scaling = T::one() / self.scaling.clone();
        self.isometry.inverse_mut();
        self.isometry.translation.vector *= self.scaling.clone();
    }

    /// Creates a new similarity with additional scaling applied before this transformation.
    ///
    /// "Prepending" means the new scaling happens first, then the existing transformation.
    /// This multiplies the scaling factors together. The rotation and translation remain unchanged.
    ///
    /// Think of it as: `new_sim = self ∘ scale(scaling)` where ∘ means composition.
    ///
    /// # Parameters
    ///
    /// * `scaling` - Additional scaling factor to apply first (must be non-zero)
    ///
    /// # Returns
    ///
    /// A new similarity with the combined scaling.
    ///
    /// # Panics
    ///
    /// Panics if `scaling` is zero.
    ///
    /// # Example: Doubling the Scale
    ///
    /// ```
    /// # use nalgebra::{Similarity2, Vector2, Point2};
    /// # use approx::assert_relative_eq;
    /// // Original: 2x scaling
    /// let sim = Similarity2::new(Vector2::new(1.0, 2.0), 0.0, 2.0);
    ///
    /// // Prepend additional 3x scaling = total 6x
    /// let bigger = sim.prepend_scaling(3.0);
    ///
    /// assert_eq!(bigger.scaling(), 6.0);
    ///
    /// let point = Point2::new(1.0, 0.0);
    /// let result = bigger * point;
    /// // Point is scaled 6x, then rotated (none), then translated
    /// ```
    ///
    /// # Example: Progressive Scaling
    ///
    /// ```
    /// # use nalgebra::{Similarity3, Vector3};
    /// let mut transform = Similarity3::new(
    ///     Vector3::new(10.0, 0.0, 0.0),
    ///     Vector3::zeros(),
    ///     1.0
    /// );
    ///
    /// // Each frame, objects grow by 10%
    /// for _ in 0..5 {
    ///     transform = transform.prepend_scaling(1.1);
    /// }
    /// // After 5 frames: scaling ≈ 1.61
    /// ```
    ///
    /// # See Also
    ///
    /// * [`prepend_scaling_mut`](Self::prepend_scaling_mut) - Same but modifies in-place
    /// * [`append_scaling`](Self::append_scaling) - Apply scaling after this transformation
    #[inline]
    #[must_use = "Did you mean to use prepend_scaling_mut()?"]
    pub fn prepend_scaling(&self, scaling: T) -> Self {
        assert!(
            !scaling.is_zero(),
            "The similarity scaling factor must not be zero."
        );

        Self::from_isometry(self.isometry.clone(), self.scaling.clone() * scaling)
    }

    /// Creates a new similarity with additional scaling applied after this transformation.
    ///
    /// "Appending" means the existing transformation happens first, then the new scaling.
    /// This scales both the scaling factor AND the translation component, effectively
    /// scaling the entire transformed space.
    ///
    /// Think of it as: `new_sim = scale(scaling) ∘ self` where ∘ means composition.
    ///
    /// # Parameters
    ///
    /// * `scaling` - Additional scaling factor to apply after (must be non-zero)
    ///
    /// # Returns
    ///
    /// A new similarity with the appended scaling.
    ///
    /// # Panics
    ///
    /// Panics if `scaling` is zero.
    ///
    /// # Example: Appending vs Prepending
    ///
    /// ```
    /// # use nalgebra::{Similarity2, Vector2, Point2};
    /// # use approx::assert_relative_eq;
    /// // Similarity with translation (5, 0) and 2x scale
    /// let sim = Similarity2::new(Vector2::new(5.0, 0.0), 0.0, 2.0);
    ///
    /// // Prepend 3x: only affects object scaling
    /// let prepended = sim.prepend_scaling(3.0);
    /// assert_eq!(prepended.scaling(), 6.0);
    /// // Translation is still (5, 0)
    ///
    /// // Append 3x: affects BOTH scaling and translation
    /// let appended = sim.append_scaling(3.0);
    /// assert_eq!(appended.scaling(), 6.0);
    /// // Translation is now (15, 0) - also scaled by 3x!
    /// ```
    ///
    /// # Example: Hierarchical Transformations
    ///
    /// ```
    /// # use nalgebra::{Similarity3, Vector3, Point3};
    /// # use approx::assert_relative_eq;
    /// // Parent object: at position (10, 0, 0), scale 2x
    /// let parent = Similarity3::new(
    ///     Vector3::new(10.0, 0.0, 0.0),
    ///     Vector3::zeros(),
    ///     2.0
    /// );
    ///
    /// // Child offset: (0, 5, 0) relative to parent
    /// let child_offset = Similarity3::new(
    ///     Vector3::new(0.0, 5.0, 0.0),
    ///     Vector3::zeros(),
    ///     1.0
    /// );
    ///
    /// // Apply parent's scale to child's position
    /// let child_world = child_offset.append_scaling(parent.scaling());
    /// // Child is now at (0, 10, 0) in parent's scaled space
    /// ```
    ///
    /// # See Also
    ///
    /// * [`append_scaling_mut`](Self::append_scaling_mut) - Same but modifies in-place
    /// * [`prepend_scaling`](Self::prepend_scaling) - Apply scaling before this transformation
    #[inline]
    #[must_use = "Did you mean to use append_scaling_mut()?"]
    pub fn append_scaling(&self, scaling: T) -> Self {
        assert!(
            !scaling.is_zero(),
            "The similarity scaling factor must not be zero."
        );

        Self::from_parts(
            Translation::from(&self.isometry.translation.vector * scaling.clone()),
            self.isometry.rotation.clone(),
            self.scaling.clone() * scaling,
        )
    }

    /// Applies additional scaling before this transformation, modifying it in-place.
    ///
    /// This is the in-place version of [`prepend_scaling`](Self::prepend_scaling).
    /// It multiplies the current scaling factor by the new one without creating a new
    /// similarity instance.
    ///
    /// # Parameters
    ///
    /// * `scaling` - Additional scaling factor to apply first (must be non-zero)
    ///
    /// # Panics
    ///
    /// Panics if `scaling` is zero.
    ///
    /// # Example: Incremental Scaling
    ///
    /// ```
    /// # use nalgebra::{Similarity2, Vector2};
    /// let mut sim = Similarity2::new(Vector2::zeros(), 0.0, 1.0);
    ///
    /// // Gradually increase scale
    /// sim.prepend_scaling_mut(1.5);
    /// assert_eq!(sim.scaling(), 1.5);
    ///
    /// sim.prepend_scaling_mut(2.0);
    /// assert_eq!(sim.scaling(), 3.0);  // 1.5 * 2.0
    /// ```
    ///
    /// # Example: Animation Loop
    ///
    /// ```
    /// # use nalgebra::{Similarity3, Vector3};
    /// let mut object = Similarity3::new(
    ///     Vector3::new(5.0, 0.0, 0.0),
    ///     Vector3::zeros(),
    ///     1.0
    /// );
    ///
    /// // Grow by 5% each frame
    /// for _ in 0..10 {
    ///     object.prepend_scaling_mut(1.05);
    /// }
    /// // Object is now ~1.63x its original size
    /// ```
    ///
    /// # See Also
    ///
    /// * [`prepend_scaling`](Self::prepend_scaling) - Returns new similarity (doesn't modify self)
    /// * [`append_scaling_mut`](Self::append_scaling_mut) - Apply scaling after this transformation
    #[inline]
    pub fn prepend_scaling_mut(&mut self, scaling: T) {
        assert!(
            !scaling.is_zero(),
            "The similarity scaling factor must not be zero."
        );

        self.scaling *= scaling
    }

    /// Applies additional scaling after this transformation, modifying it in-place.
    ///
    /// This is the in-place version of [`append_scaling`](Self::append_scaling).
    /// It scales both the scaling factor AND the translation, affecting the entire
    /// transformed space.
    ///
    /// # Parameters
    ///
    /// * `scaling` - Additional scaling factor to apply after (must be non-zero)
    ///
    /// # Panics
    ///
    /// Panics if `scaling` is zero.
    ///
    /// # Example: Scaling Everything
    ///
    /// ```
    /// # use nalgebra::{Similarity2, Vector2, Point2};
    /// # use approx::assert_relative_eq;
    /// let mut sim = Similarity2::new(
    ///     Vector2::new(10.0, 5.0),  // Translation
    ///     0.0,
    ///     2.0                        // Scale
    /// );
    ///
    /// // Append 3x scaling
    /// sim.append_scaling_mut(3.0);
    ///
    /// // Both scale and translation are affected
    /// assert_eq!(sim.scaling(), 6.0);         // 2.0 * 3.0
    /// // Translation is also tripled to (30.0, 15.0)
    /// ```
    ///
    /// # Example: World Scale Change
    ///
    /// ```
    /// # use nalgebra::{Similarity3, Vector3};
    /// // Object at position (100, 50, 0) with 2x scale
    /// let mut object = Similarity3::new(
    ///     Vector3::new(100.0, 50.0, 0.0),
    ///     Vector3::zeros(),
    ///     2.0
    /// );
    ///
    /// // Change world units: 1 old unit = 0.5 new units
    /// object.append_scaling_mut(0.5);
    /// // Position is now (50, 25, 0), scale is 1.0
    /// ```
    ///
    /// # See Also
    ///
    /// * [`append_scaling`](Self::append_scaling) - Returns new similarity (doesn't modify self)
    /// * [`prepend_scaling_mut`](Self::prepend_scaling_mut) - Apply scaling before this transformation
    #[inline]
    pub fn append_scaling_mut(&mut self, scaling: T) {
        assert!(
            !scaling.is_zero(),
            "The similarity scaling factor must not be zero."
        );

        self.isometry.translation.vector *= scaling.clone();
        self.scaling *= scaling;
    }

    /// Appends a translation to this similarity transformation in-place.
    ///
    /// The translation is applied after the similarity's scale and rotation, moving the
    /// final result in space. This is useful for repositioning objects while keeping their
    /// scale and orientation.
    ///
    /// # Parameters
    ///
    /// * `t` - The translation to append
    ///
    /// # Example: Moving an Object
    ///
    /// ```
    /// # use nalgebra::{Similarity2, Translation2, Vector2, Point2};
    /// # use approx::assert_relative_eq;
    /// let mut sim = Similarity2::new(
    ///     Vector2::new(1.0, 2.0),
    ///     0.0,
    ///     2.0
    /// );
    ///
    /// // Move the object by (3, 4)
    /// sim.append_translation_mut(&Translation2::new(3.0, 4.0));
    ///
    /// // The new translation is (1+3, 2+4) = (4, 6)
    /// ```
    ///
    /// # See Also
    ///
    /// * [`append_rotation_mut`](Self::append_rotation_mut) - Append a rotation
    /// * [`append_scaling_mut`](Self::append_scaling_mut) - Append scaling
    #[inline]
    pub fn append_translation_mut(&mut self, t: &Translation<T, D>) {
        self.isometry.append_translation_mut(t)
    }

    /// Appends a rotation to this similarity transformation in-place.
    ///
    /// The rotation is applied after the similarity's scale, rotating the scaled object
    /// and then translating it. This changes the object's orientation.
    ///
    /// # Parameters
    ///
    /// * `r` - The rotation to append
    ///
    /// # Example: Rotating an Object
    ///
    /// ```
    /// # use nalgebra::{Similarity2, UnitComplex, Vector2};
    /// # use std::f32;
    /// let mut sim = Similarity2::new(Vector2::zeros(), 0.0, 2.0);
    ///
    /// // Add a 45-degree rotation
    /// let rotation = UnitComplex::new(f32::consts::FRAC_PI_4);
    /// sim.append_rotation_mut(&rotation);
    /// ```
    ///
    /// # See Also
    ///
    /// * [`append_translation_mut`](Self::append_translation_mut) - Append a translation
    /// * [`append_rotation_wrt_point_mut`](Self::append_rotation_wrt_point_mut) - Rotate around a point
    #[inline]
    pub fn append_rotation_mut(&mut self, r: &R) {
        self.isometry.append_rotation_mut(r)
    }

    /// Appends a rotation around a specific point to this similarity in-place.
    ///
    /// Unlike [`append_rotation_mut`](Self::append_rotation_mut), which rotates around the origin,
    /// this rotates around point `p`. The point `p` stays fixed during the rotation - imagine
    /// rotating an object while keeping one of its corners pinned in place.
    ///
    /// # Parameters
    ///
    /// * `r` - The rotation to append
    /// * `p` - The point to rotate around (this point remains invariant)
    ///
    /// # Example: Rotating Around a Pivot
    ///
    /// ```
    /// # use nalgebra::{Similarity2, UnitComplex, Point2, Vector2};
    /// # use approx::assert_relative_eq;
    /// # use std::f32;
    /// let mut sim = Similarity2::new(Vector2::zeros(), 0.0, 1.0);
    ///
    /// let pivot = Point2::new(5.0, 0.0);
    /// let rotation = UnitComplex::new(f32::consts::PI); // 180 degrees
    ///
    /// sim.append_rotation_wrt_point_mut(&rotation, &pivot);
    ///
    /// // The pivot point stays at (5, 0), but the transformation rotates around it
    /// assert_relative_eq!(sim * pivot, pivot, epsilon = 1.0e-5);
    /// ```
    ///
    /// # See Also
    ///
    /// * [`append_rotation_mut`](Self::append_rotation_mut) - Rotate around origin
    /// * [`append_rotation_wrt_center_mut`](Self::append_rotation_wrt_center_mut) - Rotate around the translation
    #[inline]
    pub fn append_rotation_wrt_point_mut(&mut self, r: &R, p: &Point<T, D>) {
        self.isometry.append_rotation_wrt_point_mut(r, p)
    }

    /// Appends a rotation around this similarity's translation point in-place.
    ///
    /// This rotates the transformation around the point at `self.translation`. This is
    /// particularly useful for rotating an object around its own position, like spinning
    /// a game character in place.
    ///
    /// # Parameters
    ///
    /// * `r` - The rotation to append
    ///
    /// # Example: Spinning in Place
    ///
    /// ```
    /// # use nalgebra::{Similarity3, UnitQuaternion, Vector3};
    /// # use std::f32;
    /// // Object at position (10, 5, 0)
    /// let mut sim = Similarity3::new(
    ///     Vector3::new(10.0, 5.0, 0.0),
    ///     Vector3::zeros(),
    ///     1.0
    /// );
    ///
    /// // Rotate 90 degrees around Y axis, centered at the object's position
    /// let rotation = UnitQuaternion::from_axis_angle(
    ///     &Vector3::y_axis(),
    ///     f32::consts::FRAC_PI_2
    /// );
    /// sim.append_rotation_wrt_center_mut(&rotation);
    ///
    /// // Object rotated in place - position unchanged, orientation changed
    /// ```
    ///
    /// # See Also
    ///
    /// * [`append_rotation_mut`](Self::append_rotation_mut) - Rotate around origin
    /// * [`append_rotation_wrt_point_mut`](Self::append_rotation_wrt_point_mut) - Rotate around any point
    #[inline]
    pub fn append_rotation_wrt_center_mut(&mut self, r: &R) {
        self.isometry.append_rotation_wrt_center_mut(r)
    }

    /// Transforms a point using this similarity transformation.
    ///
    /// This applies the complete similarity transformation to a point: first scaling,
    /// then rotating, then translating. This is the main way to transform geometric
    /// points in space.
    ///
    /// This method is equivalent to using the `*` operator: `sim * point`.
    ///
    /// # Parameters
    ///
    /// * `pt` - The point to transform
    ///
    /// # Returns
    ///
    /// The transformed point
    ///
    /// # Example: 2D Point Transformation
    ///
    /// ```
    /// # use nalgebra::{Similarity2, Vector2, Point2};
    /// # use approx::assert_relative_eq;
    /// # use std::f32;
    /// // Create a similarity: move to (5, 3), rotate 90°, scale 2x
    /// let sim = Similarity2::new(
    ///     Vector2::new(5.0, 3.0),
    ///     f32::consts::FRAC_PI_2,
    ///     2.0
    /// );
    ///
    /// let point = Point2::new(1.0, 0.0);
    /// let transformed = sim.transform_point(&point);
    ///
    /// // Point is: scaled to (2, 0), rotated to (0, 2), translated to (5, 5)
    /// assert_relative_eq!(transformed, Point2::new(5.0, 5.0), epsilon = 1.0e-5);
    /// ```
    ///
    /// # Example: 3D Point Transformation
    ///
    /// ```
    /// # use nalgebra::{Similarity3, Vector3, Point3};
    /// # use approx::assert_relative_eq;
    /// # use std::f32;
    /// let axisangle = Vector3::y() * f32::consts::FRAC_PI_2; // 90° around Y
    /// let translation = Vector3::new(1.0, 2.0, 3.0);
    /// let sim = Similarity3::new(translation, axisangle, 3.0);
    ///
    /// let point = Point3::new(4.0, 5.0, 6.0);
    /// let transformed = sim.transform_point(&point);
    ///
    /// assert_relative_eq!(transformed, Point3::new(19.0, 17.0, -9.0), epsilon = 1.0e-5);
    /// ```
    ///
    /// # Example: Game Object Vertex Transformation
    ///
    /// ```
    /// # use nalgebra::{Similarity3, Vector3, Point3};
    /// // Object transform: at (10, 0, 5), no rotation, 2x scale
    /// let object_transform = Similarity3::new(
    ///     Vector3::new(10.0, 0.0, 5.0),
    ///     Vector3::zeros(),
    ///     2.0
    /// );
    ///
    /// // Local vertex at (1, 1, 1)
    /// let local_vertex = Point3::new(1.0, 1.0, 1.0);
    ///
    /// // Transform to world space
    /// let world_vertex = object_transform.transform_point(&local_vertex);
    /// // Result: (12, 2, 7) - scaled 2x and moved to object position
    /// ```
    ///
    /// # See Also
    ///
    /// * [`transform_vector`](Self::transform_vector) - Transform a vector (ignores translation)
    /// * [`inverse_transform_point`](Self::inverse_transform_point) - Apply the inverse transformation
    #[inline]
    #[must_use]
    pub fn transform_point(&self, pt: &Point<T, D>) -> Point<T, D> {
        self * pt
    }

    /// Transforms a vector using this similarity, ignoring translation.
    ///
    /// Vectors represent directions or displacements, not positions, so they are not
    /// affected by translation. This method applies only the scaling and rotation
    /// components of the similarity transformation.
    ///
    /// This is useful for transforming directions (like normals, velocities, or offsets)
    /// where position doesn't matter, only magnitude and direction.
    ///
    /// This method is equivalent to using the `*` operator: `sim * vector`.
    ///
    /// # Parameters
    ///
    /// * `v` - The vector to transform
    ///
    /// # Returns
    ///
    /// The transformed vector (scaled and rotated, but not translated)
    ///
    /// # Example: 2D Vector Transformation
    ///
    /// ```
    /// # use nalgebra::{Similarity2, Vector2};
    /// # use approx::assert_relative_eq;
    /// # use std::f32;
    /// // Similarity with translation, rotation, and scaling
    /// let sim = Similarity2::new(
    ///     Vector2::new(100.0, 200.0),  // Translation (ignored for vectors!)
    ///     f32::consts::FRAC_PI_2,       // 90° rotation
    ///     2.0                            // 2x scale
    /// );
    ///
    /// let vec = Vector2::new(1.0, 0.0);
    /// let transformed = sim.transform_vector(&vec);
    ///
    /// // Vector is scaled to (2, 0), then rotated to (0, 2)
    /// // Translation is NOT applied
    /// assert_relative_eq!(transformed, Vector2::new(0.0, 2.0), epsilon = 1.0e-5);
    /// ```
    ///
    /// # Example: 3D Direction Transformation
    ///
    /// ```
    /// # use nalgebra::{Similarity3, Vector3};
    /// # use approx::assert_relative_eq;
    /// # use std::f32;
    /// let axisangle = Vector3::y() * f32::consts::FRAC_PI_2; // 90° around Y
    /// let translation = Vector3::new(1.0, 2.0, 3.0);          // Ignored!
    /// let sim = Similarity3::new(translation, axisangle, 3.0);
    ///
    /// let vector = Vector3::new(4.0, 5.0, 6.0);
    /// let transformed = sim.transform_vector(&vector);
    ///
    /// // Only scale and rotation applied
    /// assert_relative_eq!(transformed, Vector3::new(18.0, 15.0, -12.0), epsilon = 1.0e-5);
    /// ```
    ///
    /// # Example: Scaling Object Dimensions
    ///
    /// ```
    /// # use nalgebra::{Similarity3, Vector3};
    /// # use approx::assert_relative_eq;
    /// // Object is scaled 2x and positioned somewhere (position doesn't matter here)
    /// let object_transform = Similarity3::new(
    ///     Vector3::new(10.0, 20.0, 30.0),
    ///     Vector3::zeros(),
    ///     2.0
    /// );
    ///
    /// // Object's local size/dimensions
    /// let local_size = Vector3::new(1.0, 2.0, 3.0);
    ///
    /// // Get world-space size (scaled but not translated)
    /// let world_size = object_transform.transform_vector(&local_size);
    ///
    /// assert_relative_eq!(world_size, Vector3::new(2.0, 4.0, 6.0), epsilon = 1.0e-5);
    /// ```
    ///
    /// # See Also
    ///
    /// * [`transform_point`](Self::transform_point) - Transform a point (includes translation)
    /// * [`inverse_transform_vector`](Self::inverse_transform_vector) - Apply the inverse transformation
    #[inline]
    #[must_use]
    pub fn transform_vector(&self, v: &SVector<T, D>) -> SVector<T, D> {
        self * v
    }

    /// Transforms a point by the inverse of this similarity transformation.
    ///
    /// This is more efficient than calling `sim.inverse() * point` because it doesn't
    /// need to construct the full inverse similarity - it just applies the inverse
    /// operations directly.
    ///
    /// This is useful for converting from world space back to local space, such as
    /// transforming mouse coordinates into an object's coordinate system.
    ///
    /// # Parameters
    ///
    /// * `pt` - The point to transform
    ///
    /// # Returns
    ///
    /// The point transformed by the inverse similarity
    ///
    /// # Example: Reversing a Transformation
    ///
    /// ```
    /// # use nalgebra::{Similarity2, Vector2, Point2};
    /// # use approx::assert_relative_eq;
    /// # use std::f32;
    /// let sim = Similarity2::new(
    ///     Vector2::new(3.0, 4.0),
    ///     f32::consts::FRAC_PI_4,
    ///     2.0
    /// );
    ///
    /// let original = Point2::new(1.0, 1.0);
    /// let transformed = sim.transform_point(&original);
    ///
    /// // Get back to the original point
    /// let back = sim.inverse_transform_point(&transformed);
    /// assert_relative_eq!(back, original, epsilon = 1.0e-5);
    /// ```
    ///
    /// # Example: 3D Inverse Transform
    ///
    /// ```
    /// # use nalgebra::{Similarity3, Vector3, Point3};
    /// # use approx::assert_relative_eq;
    /// # use std::f32;
    /// let axisangle = Vector3::y() * f32::consts::FRAC_PI_2;
    /// let translation = Vector3::new(1.0, 2.0, 3.0);
    /// let sim = Similarity3::new(translation, axisangle, 2.0);
    ///
    /// let point = Point3::new(4.0, 5.0, 6.0);
    /// let transformed = sim.inverse_transform_point(&point);
    ///
    /// assert_relative_eq!(transformed, Point3::new(-1.5, 1.5, 1.5), epsilon = 1.0e-5);
    /// ```
    ///
    /// # Example: Mouse to Object Coordinates
    ///
    /// ```
    /// # use nalgebra::{Similarity2, Vector2, Point2};
    /// // Object transform in world space
    /// let object_transform = Similarity2::new(
    ///     Vector2::new(100.0, 100.0),  // Object center on screen
    ///     0.0,                          // No rotation
    ///     2.0                            // Object is 2x larger than its local size
    /// );
    ///
    /// // Mouse clicked at screen position (120, 140)
    /// let mouse_pos = Point2::new(120.0, 140.0);
    ///
    /// // Convert to object's local coordinates
    /// let local_pos = object_transform.inverse_transform_point(&mouse_pos);
    /// // Result: (10, 20) in object's local space
    /// ```
    ///
    /// # See Also
    ///
    /// * [`transform_point`](Self::transform_point) - Apply the forward transformation
    /// * [`inverse_transform_vector`](Self::inverse_transform_vector) - Transform a vector by the inverse
    /// * [`inverse`](Self::inverse) - Get the full inverse similarity
    #[inline]
    #[must_use]
    pub fn inverse_transform_point(&self, pt: &Point<T, D>) -> Point<T, D> {
        self.isometry.inverse_transform_point(pt) / self.scaling()
    }

    /// Transforms a vector by the inverse of this similarity, ignoring translation.
    ///
    /// This is more efficient than calling `sim.inverse() * vector` because it doesn't
    /// need to construct the full inverse similarity. Since vectors aren't affected by
    /// translation anyway, this applies only the inverse scaling and rotation.
    ///
    /// This is useful for converting directions or displacements from world space back
    /// to local space.
    ///
    /// # Parameters
    ///
    /// * `v` - The vector to transform
    ///
    /// # Returns
    ///
    /// The vector transformed by the inverse similarity (inverse scale and rotation only)
    ///
    /// # Example: Reversing a Vector Transformation
    ///
    /// ```
    /// # use nalgebra::{Similarity2, Vector2};
    /// # use approx::assert_relative_eq;
    /// # use std::f32;
    /// let sim = Similarity2::new(
    ///     Vector2::new(10.0, 20.0),    // Translation (ignored for vectors)
    ///     f32::consts::FRAC_PI_2,       // 90° rotation
    ///     3.0                            // 3x scale
    /// );
    ///
    /// let original = Vector2::new(1.0, 0.0);
    /// let transformed = sim.transform_vector(&original);
    ///
    /// // Get back to the original vector
    /// let back = sim.inverse_transform_vector(&transformed);
    /// assert_relative_eq!(back, original, epsilon = 1.0e-5);
    /// ```
    ///
    /// # Example: 3D Inverse Vector Transform
    ///
    /// ```
    /// # use nalgebra::{Similarity3, Vector3};
    /// # use approx::assert_relative_eq;
    /// # use std::f32;
    /// let axisangle = Vector3::y() * f32::consts::FRAC_PI_2;
    /// let translation = Vector3::new(1.0, 2.0, 3.0);
    /// let sim = Similarity3::new(translation, axisangle, 2.0);
    ///
    /// let vector = Vector3::new(4.0, 5.0, 6.0);
    /// let transformed = sim.inverse_transform_vector(&vector);
    ///
    /// assert_relative_eq!(transformed, Vector3::new(-3.0, 2.5, 2.0), epsilon = 1.0e-5);
    /// ```
    ///
    /// # Example: World Velocity to Local Velocity
    ///
    /// ```
    /// # use nalgebra::{Similarity3, Vector3};
    /// # use approx::assert_relative_eq;
    /// // Object transform: 2x scale, some rotation
    /// let object_transform = Similarity3::new(
    ///     Vector3::new(10.0, 5.0, 0.0),
    ///     Vector3::zeros(),
    ///     2.0
    /// );
    ///
    /// // Velocity in world space
    /// let world_velocity = Vector3::new(4.0, 6.0, 8.0);
    ///
    /// // Convert to object's local space
    /// let local_velocity = object_transform.inverse_transform_vector(&world_velocity);
    ///
    /// // Velocity is divided by scale (2x) to get local velocity
    /// assert_relative_eq!(local_velocity, Vector3::new(2.0, 3.0, 4.0), epsilon = 1.0e-5);
    /// ```
    ///
    /// # See Also
    ///
    /// * [`transform_vector`](Self::transform_vector) - Apply the forward transformation
    /// * [`inverse_transform_point`](Self::inverse_transform_point) - Transform a point by the inverse
    /// * [`inverse`](Self::inverse) - Get the full inverse similarity
    #[inline]
    #[must_use]
    pub fn inverse_transform_vector(&self, v: &SVector<T, D>) -> SVector<T, D> {
        self.isometry.inverse_transform_vector(v) / self.scaling()
    }
}

// NOTE: we don't require `R: Rotation<...>` here because this is not useful for the implementation
// and makes it harder to use it, e.g., for Transform × Isometry implementation.
// This is OK since all constructors of the isometry enforce the Rotation bound already (and
// explicit struct construction is prevented by the private scaling factor).
impl<T: SimdRealField, R, const D: usize> Similarity<T, R, D> {
    /// Converts this similarity into its equivalent homogeneous transformation matrix.
    ///
    /// A homogeneous matrix is a standard representation used in computer graphics and
    /// robotics. For a similarity in N dimensions, this produces an (N+1)×(N+1) matrix
    /// that can be used with homogeneous coordinates.
    ///
    /// The resulting matrix can be multiplied with other transformation matrices or used
    /// with graphics APIs that expect matrix representations.
    ///
    /// # Returns
    ///
    /// A homogeneous transformation matrix representing this similarity
    ///
    /// # Example: 2D to 3×3 Matrix
    ///
    /// ```
    /// # use nalgebra::{Similarity2, Vector2, Matrix3};
    /// # use approx::assert_relative_eq;
    /// let sim = Similarity2::new(
    ///     Vector2::new(5.0, 10.0),  // Translation
    ///     0.0,                       // No rotation
    ///     2.0                        // 2x scale
    /// );
    ///
    /// let matrix: Matrix3<f32> = sim.to_homogeneous();
    ///
    /// // The matrix encodes scale, rotation, and translation
    /// // Format: [[scale*R, translation],
    /// //          [0,       1          ]]
    /// ```
    ///
    /// # Example: 3D to 4×4 Matrix
    ///
    /// ```
    /// # use nalgebra::{Similarity3, Vector3, Matrix4};
    /// let sim = Similarity3::new(
    ///     Vector3::new(1.0, 2.0, 3.0),
    ///     Vector3::zeros(),
    ///     1.5
    /// );
    ///
    /// let matrix: Matrix4<f32> = sim.to_homogeneous();
    /// // This 4×4 matrix can be used with OpenGL, DirectX, etc.
    /// ```
    ///
    /// # Example: Composing with Other Matrices
    ///
    /// ```
    /// # use nalgebra::{Similarity2, Vector2, Matrix3};
    /// # use approx::assert_relative_eq;
    /// let sim1 = Similarity2::new(Vector2::new(1.0, 0.0), 0.0, 2.0);
    /// let sim2 = Similarity2::new(Vector2::new(0.0, 1.0), 0.0, 3.0);
    ///
    /// // Convert to matrices and compose
    /// let mat1 = sim1.to_homogeneous();
    /// let mat2 = sim2.to_homogeneous();
    /// let combined_mat = mat2 * mat1;
    ///
    /// // Can also compose similarities directly
    /// let combined_sim = sim2 * sim1;
    /// let combined_mat2 = combined_sim.to_homogeneous();
    ///
    /// assert_relative_eq!(combined_mat, combined_mat2, epsilon = 1.0e-5);
    /// ```
    ///
    /// # See Also
    ///
    /// * [`from_parts`](Self::from_parts) - Create similarity from components
    #[inline]
    #[must_use]
    pub fn to_homogeneous(&self) -> OMatrix<T, DimNameSum<Const<D>, U1>, DimNameSum<Const<D>, U1>>
    where
        Const<D>: DimNameAdd<U1>,
        R: SubsetOf<OMatrix<T, DimNameSum<Const<D>, U1>, DimNameSum<Const<D>, U1>>>,
        DefaultAllocator: Allocator<DimNameSum<Const<D>, U1>, DimNameSum<Const<D>, U1>>,
    {
        let mut res = self.isometry.to_homogeneous();

        for e in res.fixed_view_mut::<D, D>(0, 0).iter_mut() {
            *e *= self.scaling.clone()
        }

        res
    }
}

impl<T: SimdRealField, R, const D: usize> Eq for Similarity<T, R, D> where
    R: AbstractRotation<T, D> + Eq
{
}

impl<T: SimdRealField, R, const D: usize> PartialEq for Similarity<T, R, D>
where
    R: AbstractRotation<T, D> + PartialEq,
{
    #[inline]
    fn eq(&self, right: &Self) -> bool {
        self.isometry == right.isometry && self.scaling == right.scaling
    }
}

impl<T: RealField, R, const D: usize> AbsDiffEq for Similarity<T, R, D>
where
    R: AbstractRotation<T, D> + AbsDiffEq<Epsilon = T::Epsilon>,
    T::Epsilon: Clone,
{
    type Epsilon = T::Epsilon;

    #[inline]
    fn default_epsilon() -> Self::Epsilon {
        T::default_epsilon()
    }

    #[inline]
    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        self.isometry.abs_diff_eq(&other.isometry, epsilon.clone())
            && self.scaling.abs_diff_eq(&other.scaling, epsilon)
    }
}

impl<T: RealField, R, const D: usize> RelativeEq for Similarity<T, R, D>
where
    R: AbstractRotation<T, D> + RelativeEq<Epsilon = T::Epsilon>,
    T::Epsilon: Clone,
{
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
        self.isometry
            .relative_eq(&other.isometry, epsilon.clone(), max_relative.clone())
            && self
                .scaling
                .relative_eq(&other.scaling, epsilon, max_relative)
    }
}

impl<T: RealField, R, const D: usize> UlpsEq for Similarity<T, R, D>
where
    R: AbstractRotation<T, D> + UlpsEq<Epsilon = T::Epsilon>,
    T::Epsilon: Clone,
{
    #[inline]
    fn default_max_ulps() -> u32 {
        T::default_max_ulps()
    }

    #[inline]
    fn ulps_eq(&self, other: &Self, epsilon: Self::Epsilon, max_ulps: u32) -> bool {
        self.isometry
            .ulps_eq(&other.isometry, epsilon.clone(), max_ulps)
            && self.scaling.ulps_eq(&other.scaling, epsilon, max_ulps)
    }
}

/*
 *
 * Display
 *
 */
impl<T, R, const D: usize> fmt::Display for Similarity<T, R, D>
where
    T: RealField + fmt::Display,
    R: AbstractRotation<T, D> + fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let precision = f.precision().unwrap_or(3);

        writeln!(f, "Similarity {{")?;
        write!(f, "{:.*}", precision, self.isometry)?;
        write!(f, "Scaling: {:.*}", precision, self.scaling)?;
        writeln!(f, "}}")
    }
}
