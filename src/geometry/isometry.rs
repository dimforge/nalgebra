// Needed otherwise the rkyv macros generate code incompatible with rust-2024
#![cfg_attr(feature = "rkyv-serialize", allow(unsafe_op_in_unsafe_fn))]

use approx::{AbsDiffEq, RelativeEq, UlpsEq};
use std::fmt;
use std::hash;

#[cfg(feature = "serde-serialize-no-std")]
use serde::{Deserialize, Serialize};

use simba::scalar::{RealField, SubsetOf};
use simba::simd::SimdRealField;

use crate::base::allocator::Allocator;
use crate::base::dimension::{DimNameAdd, DimNameSum, U1};
use crate::base::storage::Owned;
use crate::base::{Const, DefaultAllocator, OMatrix, SVector, Scalar, Unit};
use crate::geometry::{AbstractRotation, Point, Translation};

#[cfg(doc)]
use crate::{Isometry3, Quaternion, Vector3, Vector4};

#[cfg(feature = "rkyv-serialize")]
use rkyv::bytecheck;

/// A direct isometry, i.e., a rotation followed by a translation (aka. a rigid-body motion).
///
/// This is also known as an element of a Special Euclidean (SE) group.
/// The `Isometry` type can either represent a 2D or 3D isometry.
/// A 2D isometry is composed of:
/// - A translation part of type [`Translation2`](crate::Translation2)
/// - A rotation part which can either be a [`UnitComplex`](crate::UnitComplex) or a [`Rotation2`](crate::Rotation2).
///
/// A 3D isometry is composed of:
/// - A translation part of type [`Translation3`](crate::Translation3)
/// - A rotation part which can either be a [`UnitQuaternion`](crate::UnitQuaternion) or a [`Rotation3`](crate::Rotation3).
///
/// Note that instead of using the [`Isometry`] type in your code directly, you should use one
/// of its aliases: [`Isometry2`](crate::Isometry2), [`Isometry3`],
/// [`IsometryMatrix2`](crate::IsometryMatrix2), [`IsometryMatrix3`](crate::IsometryMatrix3). Though
/// keep in mind that all the documentation of all the methods of these aliases will also appears on
/// this page.
///
/// # Construction
/// * [From a 2D vector and/or an angle <span style="float:right;">`new`, `translation`, `rotation`…</span>](#construction-from-a-2d-vector-andor-a-rotation-angle)
/// * [From a 3D vector and/or an axis-angle <span style="float:right;">`new`, `translation`, `rotation`…</span>](#construction-from-a-3d-vector-andor-an-axis-angle)
/// * [From a 3D eye position and target point <span style="float:right;">`look_at`, `look_at_lh`, `face_towards`…</span>](#construction-from-a-3d-eye-position-and-target-point)
/// * [From the translation and rotation parts <span style="float:right;">`from_parts`…</span>](#from-the-translation-and-rotation-parts)
///
/// # Transformation and composition
/// Note that transforming vectors and points can be done by multiplication, e.g., `isometry * point`.
/// Composing an isometry with another transformation can also be done by multiplication or division.
///
/// * [Transformation of a vector or a point <span style="float:right;">`transform_vector`, `inverse_transform_point`…</span>](#transformation-of-a-vector-or-a-point)
/// * [Inversion and in-place composition <span style="float:right;">`inverse`, `append_rotation_wrt_point_mut`…</span>](#inversion-and-in-place-composition)
/// * [Interpolation <span style="float:right;">`lerp_slerp`…</span>](#interpolation)
///
/// # Conversion to a matrix
/// * [Conversion to a matrix <span style="float:right;">`to_matrix`…</span>](#conversion-to-a-matrix)
///
#[repr(C)]
#[derive(Debug, Copy, Clone)]
#[cfg_attr(feature = "serde-serialize-no-std", derive(Serialize, Deserialize))]
#[cfg_attr(
    feature = "serde-serialize-no-std",
    serde(bound(serialize = "R: Serialize,
                     DefaultAllocator: Allocator<Const<D>>,
                     Owned<T, Const<D>>: Serialize,
                     T: Scalar"))
)]
#[cfg_attr(
    feature = "serde-serialize-no-std",
    serde(bound(deserialize = "R: Deserialize<'de>,
                       DefaultAllocator: Allocator<Const<D>>,
                       Owned<T, Const<D>>: Deserialize<'de>,
                       T: Scalar"))
)]
#[cfg_attr(feature = "rkyv-serialize", derive(bytecheck::CheckBytes))]
#[cfg_attr(
    feature = "rkyv-serialize-no-std",
    derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize),
    archive(
        as = "Isometry<T::Archived, R::Archived, D>",
        bound(archive = "
        T: rkyv::Archive,
        R: rkyv::Archive,
        Translation<T, D>: rkyv::Archive<Archived = Translation<T::Archived, D>>
    ")
    )
)]
#[cfg_attr(feature = "defmt", derive(defmt::Format))]
pub struct Isometry<T, R, const D: usize> {
    /// The pure rotational part of this isometry.
    pub rotation: R,
    /// The pure translational part of this isometry.
    pub translation: Translation<T, D>,
}

impl<T: Scalar + hash::Hash, R: hash::Hash, const D: usize> hash::Hash for Isometry<T, R, D>
where
    Owned<T, Const<D>>: hash::Hash,
{
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        self.translation.hash(state);
        self.rotation.hash(state);
    }
}

#[cfg(feature = "bytemuck")]
unsafe impl<T: Scalar, R, const D: usize> bytemuck::Zeroable for Isometry<T, R, D>
where
    SVector<T, D>: bytemuck::Zeroable,
    R: bytemuck::Zeroable,
{
}

#[cfg(feature = "bytemuck")]
unsafe impl<T: Scalar, R, const D: usize> bytemuck::Pod for Isometry<T, R, D>
where
    SVector<T, D>: bytemuck::Pod,
    R: bytemuck::Pod,
    T: Copy,
{
}

/// # From the translation and rotation parts
impl<T: Scalar, R: AbstractRotation<T, D>, const D: usize> Isometry<T, R, D> {
    /// Creates a new isometry from its rotational and translational parts.
    ///
    /// An isometry represents a rigid-body transformation that preserves distances and angles.
    /// It combines a rotation and a translation, and is commonly used in game development,
    /// robotics, and physics simulations to represent the position and orientation of objects.
    ///
    /// This constructor allows you to build an isometry by separately specifying:
    /// - A `Translation` component that moves points in space
    /// - A rotation component (like `UnitQuaternion` or `Rotation3`) that rotates points
    ///
    /// The transformation is applied in the order: rotation first, then translation.
    ///
    /// # Parameters
    /// - `translation`: The translational component (how far to move)
    /// - `rotation`: The rotational component (how much to rotate)
    ///
    /// # Examples
    ///
    /// ## Basic 3D Example
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use std::f32;
    /// # use nalgebra::{Isometry3, Translation3, UnitQuaternion, Vector3, Point3};
    /// // Create a translation that moves 3 units along the Z axis
    /// let tra = Translation3::new(0.0, 0.0, 3.0);
    ///
    /// // Create a 180-degree rotation around the Y axis
    /// let rot = UnitQuaternion::from_scaled_axis(Vector3::y() * f32::consts::PI);
    ///
    /// // Combine them into an isometry
    /// let iso = Isometry3::from_parts(tra, rot);
    ///
    /// // Apply the transformation to a point
    /// assert_relative_eq!(iso * Point3::new(1.0, 2.0, 3.0), Point3::new(-1.0, 2.0, 0.0), epsilon = 1.0e-6);
    /// ```
    ///
    /// ## Game Object Positioning
    /// ```
    /// # use nalgebra::{Isometry3, Translation3, UnitQuaternion, Vector3, Point3};
    /// // Position a game character at (10, 0, 5) facing east (90° rotation around Y)
    /// let position = Translation3::new(10.0, 0.0, 5.0);
    /// let orientation = UnitQuaternion::from_axis_angle(
    ///     &Vector3::y_axis(),
    ///     std::f32::consts::FRAC_PI_2
    /// );
    ///
    /// let character_transform = Isometry3::from_parts(position, orientation);
    ///
    /// // The character's local "forward" direction (0, 0, -1) becomes world direction
    /// let local_forward = Vector3::new(0.0, 0.0, -1.0);
    /// let world_forward = character_transform.rotation * local_forward;
    /// ```
    ///
    /// ## 2D Robot Pose
    /// ```
    /// # use nalgebra::{Isometry2, Translation2, UnitComplex, Point2};
    /// // A robot at position (5, 3) rotated 45 degrees
    /// let position = Translation2::new(5.0, 3.0);
    /// let heading = UnitComplex::new(std::f32::consts::FRAC_PI_4);
    ///
    /// let robot_pose = Isometry2::from_parts(position, heading);
    ///
    /// // Sensor offset: 1 meter in front of robot center
    /// let sensor_local = Point2::new(1.0, 0.0);
    /// let sensor_world = robot_pose * sensor_local;
    /// ```
    ///
    /// # See Also
    /// - [`Isometry::inverse`] - Compute the inverse transformation
    /// - [`Isometry::transform_point`] - Apply the transformation to a point
    /// - [`Translation`] - The translation component
    /// - For construction with vectors: `Isometry2::new` and `Isometry3::new`
    #[inline]
    pub const fn from_parts(translation: Translation<T, D>, rotation: R) -> Self {
        Self {
            rotation,
            translation,
        }
    }
}

/// # Inversion and in-place composition
impl<T: SimdRealField, R: AbstractRotation<T, D>, const D: usize> Isometry<T, R, D>
where
    T::Element: SimdRealField,
{
    /// Computes the inverse of this isometry.
    ///
    /// The inverse isometry "undoes" the transformation. If you first apply an isometry
    /// to a point and then apply its inverse, you get back the original point.
    ///
    /// For an isometry composed of rotation R and translation T, the inverse isometry
    /// has rotation R⁻¹ and translation -(R⁻¹ * T). This effectively reverses both
    /// the rotation and translation components.
    ///
    /// This method creates a new isometry without modifying the original. If you want
    /// to modify in place, use [`inverse_mut`](Self::inverse_mut) instead.
    ///
    /// # Returns
    /// A new isometry that is the inverse of `self`.
    ///
    /// # Examples
    ///
    /// ## Basic Inversion
    /// ```
    /// # use std::f32;
    /// # use nalgebra::{Isometry2, Point2, Vector2};
    /// let iso = Isometry2::new(Vector2::new(1.0, 2.0), f32::consts::FRAC_PI_2);
    /// let inv = iso.inverse();
    /// let pt = Point2::new(1.0, 2.0);
    ///
    /// // Applying isometry then its inverse returns the original point
    /// assert_eq!(inv * (iso * pt), pt);
    /// ```
    ///
    /// ## Undo a Rigid Body Transformation
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{Isometry3, Vector3, Point3};
    /// // A rigid body moved and rotated in 3D space
    /// let transform = Isometry3::new(
    ///     Vector3::new(5.0, 0.0, 0.0),
    ///     Vector3::z() * std::f32::consts::FRAC_PI_4
    /// );
    ///
    /// // Get the inverse to transform from world space back to local space
    /// let inverse_transform = transform.inverse();
    ///
    /// let world_point = Point3::new(10.0, 5.0, 0.0);
    /// let local_point = inverse_transform * world_point;
    ///
    /// // Converting back to world space gives the original point
    /// assert_relative_eq!(transform * local_point, world_point, epsilon = 1.0e-6);
    /// ```
    ///
    /// ## Converting Between Coordinate Frames
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{Isometry2, Vector2, Point2};
    /// // Transform from local to world coordinates
    /// let local_to_world = Isometry2::new(
    ///     Vector2::new(10.0, 20.0),
    ///     std::f32::consts::FRAC_PI_2
    /// );
    ///
    /// // Inverse transforms from world to local coordinates
    /// let world_to_local = local_to_world.inverse();
    ///
    /// let point_in_local = Point2::new(1.0, 0.0);
    /// let point_in_world = local_to_world * point_in_local;
    /// let back_to_local = world_to_local * point_in_world;
    ///
    /// assert_relative_eq!(back_to_local, point_in_local, epsilon = 1.0e-6);
    /// ```
    ///
    /// # See Also
    /// - [`inverse_mut`](Self::inverse_mut) - In-place version that modifies `self`
    /// - [`inv_mul`](Self::inv_mul) - Efficiently compute `self.inverse() * other`
    /// - [`inverse_transform_point`](Self::inverse_transform_point) - Transform a point by the inverse
    /// - [`inverse_transform_vector`](Self::inverse_transform_vector) - Transform a vector by the inverse
    #[inline]
    #[must_use = "Did you mean to use inverse_mut()?"]
    pub fn inverse(&self) -> Self {
        let mut res = self.clone();
        res.inverse_mut();
        res
    }

    /// Inverts this isometry in-place.
    ///
    /// This method modifies the current isometry to become its inverse, without allocating
    /// a new isometry. After calling this method, the isometry will "undo" what it
    /// previously did.
    ///
    /// This is more efficient than calling [`inverse`](Self::inverse) when you don't need
    /// to keep the original isometry, as it avoids creating a new object.
    ///
    /// # Examples
    ///
    /// ## Basic In-Place Inversion
    /// ```
    /// # use std::f32;
    /// # use nalgebra::{Isometry2, Point2, Vector2};
    /// let mut iso = Isometry2::new(Vector2::new(1.0, 2.0), f32::consts::FRAC_PI_2);
    /// let pt = Point2::new(1.0, 2.0);
    /// let transformed_pt = iso * pt;
    ///
    /// // Invert the isometry in-place
    /// iso.inverse_mut();
    ///
    /// // Now it undoes the previous transformation
    /// assert_eq!(iso * transformed_pt, pt);
    /// ```
    ///
    /// ## Efficiently Toggle Between Frames
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{Isometry3, Vector3, Point3};
    /// // Start with camera-to-world transformation
    /// let mut transform = Isometry3::new(
    ///     Vector3::new(0.0, 5.0, 10.0),
    ///     Vector3::x() * std::f32::consts::FRAC_PI_6
    /// );
    ///
    /// let world_point = Point3::new(1.0, 2.0, 3.0);
    /// let camera_point = transform.inverse() * world_point;
    ///
    /// // Flip to world-to-camera without allocating
    /// transform.inverse_mut();
    /// assert_relative_eq!(transform * world_point, camera_point, epsilon = 1.0e-6);
    /// ```
    ///
    /// ## Physics Simulation: Relative Motion
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{Isometry2, Vector2, Point2};
    /// // Object A's pose
    /// let pose_a = Isometry2::new(Vector2::new(5.0, 3.0), 0.5);
    ///
    /// // Object B's pose
    /// let pose_b = Isometry2::new(Vector2::new(8.0, 7.0), 1.0);
    ///
    /// // Compute B's position relative to A
    /// let mut a_to_world = pose_a;
    /// a_to_world.inverse_mut(); // Now it's world_to_a
    ///
    /// let b_relative_to_a = a_to_world * pose_b;
    /// // Now b_relative_to_a represents B's pose in A's local frame
    /// ```
    ///
    /// # See Also
    /// - [`inverse`](Self::inverse) - Returns a new inverse without modifying `self`
    /// - [`inv_mul`](Self::inv_mul) - Efficiently compute `self.inverse() * other`
    /// - [`inverse_transform_point`](Self::inverse_transform_point) - Transform a point by the inverse
    #[inline]
    pub fn inverse_mut(&mut self) {
        self.rotation.inverse_mut();
        self.translation.inverse_mut();
        self.translation.vector = self.rotation.transform_vector(&self.translation.vector);
    }

    /// Efficiently computes `self.inverse() * rhs` without creating a temporary inverse.
    ///
    /// This method computes the relative transformation from `self`'s coordinate frame to
    /// `rhs`'s coordinate frame. It's more efficient than computing the inverse separately
    /// and then multiplying, as it avoids intermediate allocations.
    ///
    /// This is particularly useful when you want to express one object's pose relative to
    /// another, which is common in robotics (sensor frames relative to robot base),
    /// graphics (object space to parent space), and physics (collision detection).
    ///
    /// # Parameters
    /// - `rhs`: The isometry to transform by the inverse of `self`
    ///
    /// # Returns
    /// The composition `self.inverse() * rhs`, computed efficiently.
    ///
    /// # Examples
    ///
    /// ## Basic Usage
    /// ```
    /// # use std::f32;
    /// # use nalgebra::{Isometry2, Vector2};
    /// let iso1 = Isometry2::new(Vector2::new(1.0, 2.0), f32::consts::FRAC_PI_2);
    /// let iso2 = Isometry2::new(Vector2::new(10.0, 20.0), f32::consts::FRAC_PI_4);
    ///
    /// // These are equivalent, but inv_mul is more efficient
    /// assert_eq!(iso1.inverse() * iso2, iso1.inv_mul(&iso2));
    /// ```
    ///
    /// ## Relative Pose Between Objects
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{Isometry3, Vector3, Point3};
    /// // Robot base pose in world frame
    /// let robot_pose = Isometry3::new(
    ///     Vector3::new(1.0, 0.0, 0.0),
    ///     Vector3::z() * std::f32::consts::FRAC_PI_4
    /// );
    ///
    /// // Camera pose in world frame
    /// let camera_pose = Isometry3::new(
    ///     Vector3::new(1.5, 0.5, 1.0),
    ///     Vector3::z() * std::f32::consts::FRAC_PI_4
    /// );
    ///
    /// // Camera pose relative to robot base (in robot's local frame)
    /// let camera_in_robot_frame = robot_pose.inv_mul(&camera_pose);
    ///
    /// // Can now express camera measurements in robot's coordinate system
    /// ```
    ///
    /// ## Computing Relative Motion
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{Isometry2, Vector2};
    /// // Object position at time t1
    /// let pose_t1 = Isometry2::new(Vector2::new(0.0, 0.0), 0.0);
    ///
    /// // Object position at time t2
    /// let pose_t2 = Isometry2::new(Vector2::new(3.0, 4.0), 0.5);
    ///
    /// // Compute the motion that occurred between t1 and t2
    /// let relative_motion = pose_t1.inv_mul(&pose_t2);
    ///
    /// // This tells us how the object moved in its own local frame
    /// assert_relative_eq!(
    ///     relative_motion.translation.vector,
    ///     Vector2::new(3.0, 4.0),
    ///     epsilon = 1.0e-6
    /// );
    /// ```
    ///
    /// ## Hierarchical Transformations
    /// ```
    /// # use nalgebra::{Isometry3, Vector3};
    /// // Parent object (e.g., a car)
    /// let parent = Isometry3::new(Vector3::new(10.0, 0.0, 0.0), Vector3::zeros());
    ///
    /// // Child object in world frame (e.g., a wheel)
    /// let child_world = Isometry3::new(Vector3::new(11.0, 1.0, 0.0), Vector3::zeros());
    ///
    /// // Get child's pose relative to parent
    /// let child_local = parent.inv_mul(&child_world);
    ///
    /// // Now child_local represents the wheel's offset from the car center
    /// ```
    ///
    /// # See Also
    /// - [`inverse`](Self::inverse) - Compute the full inverse transformation
    /// - [`inverse_mut`](Self::inverse_mut) - In-place inversion
    /// - Multiplication operator `*` - Compose isometries directly
    #[inline]
    #[must_use]
    pub fn inv_mul(&self, rhs: &Isometry<T, R, D>) -> Self {
        let inv_rot1 = self.rotation.inverse();
        let tr_12 = &rhs.translation.vector - &self.translation.vector;
        Isometry::from_parts(
            inv_rot1.transform_vector(&tr_12).into(),
            inv_rot1 * rhs.rotation.clone(),
        )
    }

    /// Appends a translation to this isometry by pre-multiplying it (applies in world frame).
    ///
    /// This method modifies the isometry in-place by adding a translation that is applied
    /// **before** the existing rotation. The translation is specified in the world/global
    /// coordinate frame, not the local frame of the isometry.
    ///
    /// Mathematically: `self = Translation * self`, or equivalently the final translation
    /// becomes `self.translation + t`.
    ///
    /// This is useful when you want to move an object in world coordinates while preserving
    /// its orientation.
    ///
    /// # Parameters
    /// - `t`: The translation to append, specified in world coordinates
    ///
    /// # Examples
    ///
    /// ## Basic Usage
    /// ```
    /// # use std::f32;
    /// # use nalgebra::{Isometry2, Translation2, Vector2};
    /// let mut iso = Isometry2::new(Vector2::new(1.0, 2.0), f32::consts::FRAC_PI_2);
    /// let tra = Translation2::new(3.0, 4.0);
    ///
    /// // This is the same as: iso = tra * iso
    /// iso.append_translation_mut(&tra);
    ///
    /// assert_eq!(iso.translation, Translation2::new(4.0, 6.0));
    /// ```
    ///
    /// ## Moving an Object in World Space
    /// ```
    /// # use nalgebra::{Isometry3, Translation3, Vector3, Point3};
    /// // A rotated object at the origin
    /// let mut object_pose = Isometry3::new(
    ///     Vector3::zeros(),
    ///     Vector3::z() * std::f32::consts::FRAC_PI_2
    /// );
    ///
    /// // Move it 5 units along the world X axis (regardless of rotation)
    /// let world_translation = Translation3::new(5.0, 0.0, 0.0);
    /// object_pose.append_translation_mut(&world_translation);
    ///
    /// // The object is now at (5, 0, 0) with its rotation unchanged
    /// assert_eq!(object_pose.translation.vector, Vector3::new(5.0, 0.0, 0.0));
    /// ```
    ///
    /// ## Physics: Applying External Forces
    /// ```
    /// # use nalgebra::{Isometry2, Translation2, Vector2};
    /// // A game character at position (10, 5)
    /// let mut character = Isometry2::new(
    ///     Vector2::new(10.0, 5.0),
    ///     std::f32::consts::FRAC_PI_4
    /// );
    ///
    /// // Apply wind force moving character 2 units west (world -X direction)
    /// let wind_displacement = Translation2::new(-2.0, 0.0);
    /// character.append_translation_mut(&wind_displacement);
    ///
    /// // Character moved in world space, rotation unchanged
    /// assert_eq!(character.translation.vector, Vector2::new(8.0, 5.0));
    /// ```
    ///
    /// # See Also
    /// - [`append_rotation_mut`](Self::append_rotation_mut) - Append a rotation in world frame
    /// - [`append_rotation_wrt_center_mut`](Self::append_rotation_wrt_center_mut) - Rotate around object's center
    #[inline]
    pub fn append_translation_mut(&mut self, t: &Translation<T, D>) {
        self.translation.vector += &t.vector
    }

    /// Appends a rotation to this isometry by pre-multiplying it (applies in world frame).
    ///
    /// This method modifies the isometry in-place by applying a rotation in the world
    /// coordinate frame. The rotation is applied **before** the existing transformation,
    /// which affects both the orientation and the position of the isometry.
    ///
    /// Mathematically: `self = Rotation * self`
    ///
    /// This means:
    /// - The rotation component becomes `r * self.rotation`
    /// - The translation is rotated: `self.translation` becomes `r * self.translation`
    ///
    /// Use this when you want to rotate an object around the world origin, which will
    /// both change its orientation and move it if it's not at the origin.
    ///
    /// # Parameters
    /// - `r`: The rotation to append, applied in world coordinates
    ///
    /// # Examples
    ///
    /// ## Basic Usage
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use std::f32;
    /// # use nalgebra::{Isometry2, UnitComplex, Vector2};
    /// let mut iso = Isometry2::new(Vector2::new(1.0, 2.0), f32::consts::PI / 6.0);
    /// let rot = UnitComplex::new(f32::consts::PI / 2.0);
    ///
    /// // Same as: iso = rot * iso
    /// iso.append_rotation_mut(&rot);
    ///
    /// // Both position and orientation changed
    /// assert_relative_eq!(
    ///     iso,
    ///     Isometry2::new(Vector2::new(-2.0, 1.0), f32::consts::PI * 2.0 / 3.0),
    ///     epsilon = 1.0e-6
    /// );
    /// ```
    ///
    /// ## Rotating Around World Origin
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{Isometry3, UnitQuaternion, Vector3, Point3};
    /// // Object at (5, 0, 0) with no rotation
    /// let mut object = Isometry3::new(Vector3::new(5.0, 0.0, 0.0), Vector3::zeros());
    ///
    /// // Rotate 90 degrees around world Z axis
    /// let rotation = UnitQuaternion::from_axis_angle(
    ///     &Vector3::z_axis(),
    ///     std::f32::consts::FRAC_PI_2
    /// );
    /// object.append_rotation_mut(&rotation);
    ///
    /// // Object moved to (0, 5, 0) due to rotation around world origin
    /// assert_relative_eq!(
    ///     object.translation.vector,
    ///     Vector3::new(0.0, 5.0, 0.0),
    ///     epsilon = 1.0e-6
    /// );
    /// ```
    ///
    /// ## Orbit Camera Around Target
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{Isometry3, UnitQuaternion, Vector3};
    /// // Camera looking at origin from (10, 0, 0)
    /// let mut camera = Isometry3::new(Vector3::new(10.0, 0.0, 0.0), Vector3::zeros());
    ///
    /// // Rotate camera around world Y axis (orbit around target at origin)
    /// let orbit_rotation = UnitQuaternion::from_axis_angle(
    ///     &Vector3::y_axis(),
    ///     std::f32::consts::FRAC_PI_4
    /// );
    /// camera.append_rotation_mut(&orbit_rotation);
    ///
    /// // Camera orbited 45 degrees, maintaining distance from origin
    /// ```
    ///
    /// ## Planetary Motion Simulation
    /// ```
    /// # use nalgebra::{Isometry2, UnitComplex, Vector2};
    /// // Planet at distance 10 from sun (at origin)
    /// let mut planet = Isometry2::new(Vector2::new(10.0, 0.0), 0.0);
    ///
    /// // Simulate orbital motion: rotate around sun
    /// let orbital_step = UnitComplex::new(0.1); // Small rotation per timestep
    /// planet.append_rotation_mut(&orbital_step);
    ///
    /// // Planet moved along its orbit
    /// ```
    ///
    /// # See Also
    /// - [`append_rotation_wrt_point_mut`](Self::append_rotation_wrt_point_mut) - Rotate around a specific point
    /// - [`append_rotation_wrt_center_mut`](Self::append_rotation_wrt_center_mut) - Rotate around object's center
    /// - [`append_translation_mut`](Self::append_translation_mut) - Append a translation
    #[inline]
    pub fn append_rotation_mut(&mut self, r: &R) {
        self.rotation = r.clone() * self.rotation.clone();
        self.translation.vector = r.transform_vector(&self.translation.vector);
    }

    /// Appends a rotation centered at a specific point, leaving that point invariant.
    ///
    /// This method modifies the isometry in-place by applying a rotation around a specified
    /// pivot point. The pivot point remains fixed (invariant) during the rotation, while
    /// the isometry's position and orientation change as if it were rotating around that point.
    ///
    /// This is extremely useful for:
    /// - Rotating objects around a pivot point (like a door rotating around its hinge)
    /// - Orbiting a camera around a target point
    /// - Rotating mechanical parts around joints
    /// - Implementing turret rotation on a vehicle
    ///
    /// Mathematically, this performs: `self = translate(p) * rotation * translate(-p) * self`
    ///
    /// # Parameters
    /// - `r`: The rotation to apply
    /// - `p`: The pivot point (center of rotation) in world coordinates
    ///
    /// # Examples
    ///
    /// ## Basic Usage
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use std::f32;
    /// # use nalgebra::{Isometry2, UnitComplex, Vector2, Point2};
    /// let mut iso = Isometry2::new(Vector2::new(1.0, 2.0), f32::consts::FRAC_PI_2);
    /// let rot = UnitComplex::new(f32::consts::FRAC_PI_2);
    /// let pivot = Point2::new(1.0, 0.0);
    ///
    /// iso.append_rotation_wrt_point_mut(&rot, &pivot);
    ///
    /// // The pivot point itself remains at the same location after transformation
    /// assert_relative_eq!(iso * pivot, Point2::new(-2.0, 0.0), epsilon = 1.0e-6);
    /// ```
    ///
    /// ## Door Hinge Rotation
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{Isometry2, UnitComplex, Vector2, Point2};
    /// // Door with its center at (5, 0), hinge at left edge (3, 0)
    /// let mut door = Isometry2::new(Vector2::new(5.0, 0.0), 0.0);
    /// let hinge_point = Point2::new(3.0, 0.0);
    ///
    /// // Open door by 90 degrees around the hinge
    /// let rotation = UnitComplex::new(std::f32::consts::FRAC_PI_2);
    /// door.append_rotation_wrt_point_mut(&rotation, &hinge_point);
    ///
    /// // Door center moved due to rotation around hinge
    /// // The hinge point remains fixed
    /// ```
    ///
    /// ## Camera Orbiting Around Target
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{Isometry3, UnitQuaternion, Vector3, Point3};
    /// // Camera at (10, 0, 0) looking at target (0, 0, 0)
    /// let mut camera = Isometry3::new(
    ///     Vector3::new(10.0, 0.0, 0.0),
    ///     Vector3::zeros()
    /// );
    ///
    /// let target = Point3::new(0.0, 0.0, 0.0);
    /// let orbit_rotation = UnitQuaternion::from_axis_angle(
    ///     &Vector3::y_axis(),
    ///     std::f32::consts::FRAC_PI_4
    /// );
    ///
    /// // Orbit camera 45 degrees around the target
    /// camera.append_rotation_wrt_point_mut(&orbit_rotation, &target);
    ///
    /// // Camera maintains distance from target but changes position
    /// ```
    ///
    /// ## Turret on a Tank
    /// ```
    /// # use nalgebra::{Isometry2, UnitComplex, Vector2, Point2};
    /// // Tank body at (0, 0)
    /// let tank_position = Point2::new(0.0, 0.0);
    ///
    /// // Turret offset from tank center
    /// let mut turret = Isometry2::new(Vector2::new(0.0, 0.5), 0.0);
    ///
    /// // Rotate turret around tank center (not turret's own center)
    /// let turret_rotation = UnitComplex::new(std::f32::consts::FRAC_PI_4);
    /// turret.append_rotation_wrt_point_mut(&turret_rotation, &tank_position);
    ///
    /// // Turret rotated 45 degrees around tank center
    /// ```
    ///
    /// ## Robotic Arm Joint
    /// ```
    /// # use nalgebra::{Isometry3, UnitQuaternion, Vector3, Point3};
    /// // Forearm segment with elbow at (0, 0, 5)
    /// let mut forearm = Isometry3::new(Vector3::new(0.0, 0.0, 7.0), Vector3::zeros());
    /// let elbow_joint = Point3::new(0.0, 0.0, 5.0);
    ///
    /// // Bend elbow by 30 degrees
    /// let bend_rotation = UnitQuaternion::from_axis_angle(
    ///     &Vector3::x_axis(),
    ///     std::f32::consts::PI / 6.0
    /// );
    /// forearm.append_rotation_wrt_point_mut(&bend_rotation, &elbow_joint);
    ///
    /// // Forearm rotated around elbow joint
    /// ```
    ///
    /// # See Also
    /// - [`append_rotation_mut`](Self::append_rotation_mut) - Rotate around world origin
    /// - [`append_rotation_wrt_center_mut`](Self::append_rotation_wrt_center_mut) - Rotate around object's own center
    /// - [`append_translation_mut`](Self::append_translation_mut) - Add a translation
    #[inline]
    pub fn append_rotation_wrt_point_mut(&mut self, r: &R, p: &Point<T, D>) {
        self.translation.vector -= &p.coords;
        self.append_rotation_mut(r);
        self.translation.vector += &p.coords;
    }

    /// Appends a rotation around the object's own center (its translation point).
    ///
    /// This method rotates the isometry around its own position, changing only its
    /// orientation while keeping its position (translation) fixed. This is the most
    /// common type of rotation for objects in games and simulations - rotating "in place."
    ///
    /// Unlike [`append_rotation_mut`](Self::append_rotation_mut) which rotates around the
    /// world origin (potentially moving the object), or [`append_rotation_wrt_point_mut`](Self::append_rotation_wrt_point_mut) which rotates around an arbitrary point, this
    /// method ensures the object stays at its current position.
    ///
    /// Mathematically: Only `self.rotation` changes to `r * self.rotation`, while
    /// `self.translation` remains unchanged.
    ///
    /// # Parameters
    /// - `r`: The rotation to apply around the object's center
    ///
    /// # Examples
    ///
    /// ## Basic Usage
    /// ```
    /// # use std::f32;
    /// # use nalgebra::{Isometry2, UnitComplex, Vector2};
    /// let mut iso = Isometry2::new(Vector2::new(1.0, 2.0), f32::consts::FRAC_PI_2);
    /// let rot = UnitComplex::new(f32::consts::FRAC_PI_2);
    ///
    /// iso.append_rotation_wrt_center_mut(&rot);
    ///
    /// // Position unchanged, but orientation increased by 90 degrees
    /// assert_eq!(iso.translation.vector, Vector2::new(1.0, 2.0));
    /// assert_eq!(iso.rotation, UnitComplex::new(f32::consts::PI));
    /// ```
    ///
    /// ## Game Character Turning
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{Isometry2, UnitComplex, Vector2};
    /// // Character at position (10, 5) facing east (0 radians)
    /// let mut character = Isometry2::new(Vector2::new(10.0, 5.0), 0.0);
    ///
    /// // Turn character 45 degrees counter-clockwise (stays in place)
    /// let turn = UnitComplex::new(std::f32::consts::FRAC_PI_4);
    /// character.append_rotation_wrt_center_mut(&turn);
    ///
    /// // Character still at (10, 5), but now facing northeast
    /// assert_eq!(character.translation.vector, Vector2::new(10.0, 5.0));
    /// assert_relative_eq!(character.rotation.angle(), std::f32::consts::FRAC_PI_4, epsilon = 1.0e-6);
    /// ```
    ///
    /// ## Rotating Spaceship in Place
    /// ```
    /// # use nalgebra::{Isometry3, UnitQuaternion, Vector3};
    /// // Spaceship at (100, 50, 25) with some orientation
    /// let mut spaceship = Isometry3::new(
    ///     Vector3::new(100.0, 50.0, 25.0),
    ///     Vector3::y() * 0.5
    /// );
    ///
    /// // Apply yaw rotation (turn left/right) without moving position
    /// let yaw = UnitQuaternion::from_axis_angle(
    ///     &Vector3::y_axis(),
    ///     std::f32::consts::FRAC_PI_6
    /// );
    /// spaceship.append_rotation_wrt_center_mut(&yaw);
    ///
    /// // Spaceship rotated but position unchanged
    /// assert_eq!(spaceship.translation.vector, Vector3::new(100.0, 50.0, 25.0));
    /// ```
    ///
    /// ## FPS Camera Look Rotation
    /// ```
    /// # use nalgebra::{Isometry3, UnitQuaternion, Vector3};
    /// // Camera at eye position with current orientation
    /// let mut camera = Isometry3::new(Vector3::new(0.0, 1.8, 0.0), Vector3::zeros());
    ///
    /// // Look up 15 degrees (pitch) without moving the camera position
    /// let pitch = UnitQuaternion::from_axis_angle(
    ///     &Vector3::x_axis(),
    ///     std::f32::consts::PI / 12.0
    /// );
    /// camera.append_rotation_wrt_center_mut(&pitch);
    ///
    /// // Camera still at same position, just looking in different direction
    /// assert_eq!(camera.translation.vector, Vector3::new(0.0, 1.8, 0.0));
    /// ```
    ///
    /// ## Physics: Angular Velocity Integration
    /// ```
    /// # use nalgebra::{Isometry3, UnitQuaternion, Vector3};
    /// // Rigid body at some position
    /// let mut rigid_body = Isometry3::new(Vector3::new(5.0, 3.0, 2.0), Vector3::zeros());
    ///
    /// // Angular velocity: spinning around local Z axis
    /// let angular_velocity = Vector3::z() * 2.0; // rad/s
    /// let dt = 0.016; // 60 FPS timestep
    ///
    /// // Apply rotation from angular velocity (rotate in place)
    /// let delta_rotation = UnitQuaternion::from_scaled_axis(angular_velocity * dt);
    /// rigid_body.append_rotation_wrt_center_mut(&delta_rotation);
    ///
    /// // Body spun but didn't move
    /// assert_eq!(rigid_body.translation.vector, Vector3::new(5.0, 3.0, 2.0));
    /// ```
    ///
    /// # See Also
    /// - [`append_rotation_mut`](Self::append_rotation_mut) - Rotate around world origin
    /// - [`append_rotation_wrt_point_mut`](Self::append_rotation_wrt_point_mut) - Rotate around arbitrary point
    /// - [`append_translation_mut`](Self::append_translation_mut) - Change position
    #[inline]
    pub fn append_rotation_wrt_center_mut(&mut self, r: &R) {
        self.rotation = r.clone() * self.rotation.clone();
    }
}

/// # Transformation of a vector or a point
impl<T: SimdRealField, R: AbstractRotation<T, D>, const D: usize> Isometry<T, R, D>
where
    T::Element: SimdRealField,
{
    /// Transforms a point by this isometry.
    ///
    /// This applies both the rotation and translation components of the isometry to the point.
    /// The rotation is applied first, then the translation is added. This is the standard
    /// way to transform points in 3D graphics, game engines, and robotics.
    ///
    /// This method is equivalent to the multiplication `self * pt`, but can be more explicit
    /// about the intent of the operation.
    ///
    /// # Parameters
    /// - `pt`: The point to transform
    ///
    /// # Returns
    /// The transformed point
    ///
    /// # Examples
    ///
    /// ## Basic Usage
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use std::f32;
    /// # use nalgebra::{Isometry3, Translation3, UnitQuaternion, Vector3, Point3};
    /// let tra = Translation3::new(0.0, 0.0, 3.0);
    /// let rot = UnitQuaternion::from_scaled_axis(Vector3::y() * f32::consts::FRAC_PI_2);
    /// let iso = Isometry3::from_parts(tra, rot);
    ///
    /// let transformed_point = iso.transform_point(&Point3::new(1.0, 2.0, 3.0));
    /// assert_relative_eq!(transformed_point, Point3::new(3.0, 2.0, 2.0), epsilon = 1.0e-6);
    /// ```
    ///
    /// ## Transform Local to World Space
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{Isometry3, Vector3, Point3};
    /// // Object at position (10, 5, 0) rotated 90° around Z
    /// let object_pose = Isometry3::new(
    ///     Vector3::new(10.0, 5.0, 0.0),
    ///     Vector3::z() * std::f32::consts::FRAC_PI_2
    /// );
    ///
    /// // Point in object's local coordinates
    /// let local_point = Point3::new(1.0, 0.0, 0.0);
    ///
    /// // Transform to world coordinates
    /// let world_point = object_pose.transform_point(&local_point);
    ///
    /// // Local X (1,0,0) rotated 90° becomes (0,1,0), then translated
    /// assert_relative_eq!(world_point, Point3::new(10.0, 6.0, 0.0), epsilon = 1.0e-6);
    /// ```
    ///
    /// ## Game: Projectile Spawn Position
    /// ```
    /// # use nalgebra::{Isometry3, Vector3, Point3};
    /// // Player character position and orientation
    /// let player = Isometry3::new(
    ///     Vector3::new(0.0, 1.0, 0.0),  // Standing position
    ///     Vector3::y() * 0.5             // Facing direction
    /// );
    ///
    /// // Spawn point for bullet: 0.5 units in front, 1.5 units up (local coords)
    /// let gun_offset_local = Point3::new(0.0, 0.5, -0.5);
    ///
    /// // Calculate world position where bullet spawns
    /// let bullet_spawn_world = player.transform_point(&gun_offset_local);
    /// // Bullet spawns at the gun's world position
    /// ```
    ///
    /// ## Robotics: Sensor Reading to World Frame
    /// ```
    /// # use nalgebra::{Isometry2, Vector2, Point2};
    /// // Robot at (3, 4) facing 45 degrees
    /// let robot_pose = Isometry2::new(
    ///     Vector2::new(3.0, 4.0),
    ///     std::f32::consts::FRAC_PI_4
    /// );
    ///
    /// // LIDAR detected obstacle 2 meters in front (local coordinates)
    /// let obstacle_local = Point2::new(2.0, 0.0);
    ///
    /// // Convert to world coordinates for mapping
    /// let obstacle_world = robot_pose.transform_point(&obstacle_local);
    /// // Now we know the obstacle's position in the map
    /// ```
    ///
    /// ## Physics: Collision Point Calculation
    /// ```
    /// # use nalgebra::{Isometry3, Vector3, Point3};
    /// // Rigid body position and orientation
    /// let body = Isometry3::new(
    ///     Vector3::new(5.0, 2.0, 1.0),
    ///     Vector3::zeros()
    /// );
    ///
    /// // Contact point on body surface (in body's local frame)
    /// let contact_local = Point3::new(0.5, -0.5, 0.0);
    ///
    /// // Get contact point in world frame for force application
    /// let contact_world = body.transform_point(&contact_local);
    /// ```
    ///
    /// # See Also
    /// - [`transform_vector`](Self::transform_vector) - Transform a vector (no translation)
    /// - [`inverse_transform_point`](Self::inverse_transform_point) - Transform by inverse
    /// - Multiplication operator `*` - Alternative syntax for `transform_point`
    #[inline]
    #[must_use]
    pub fn transform_point(&self, pt: &Point<T, D>) -> Point<T, D> {
        self * pt
    }

    /// Transforms a vector by this isometry, ignoring the translation component.
    ///
    /// This applies only the rotation part of the isometry to the vector, completely ignoring
    /// the translation. This is the correct way to transform directions, velocities, forces,
    /// and other vector quantities (as opposed to positions/points).
    ///
    /// Vectors represent directions and magnitudes without a specific location, so they should
    /// not be affected by the translation component of a transformation - only by rotation.
    ///
    /// This method is equivalent to the multiplication `self * v`.
    ///
    /// # Parameters
    /// - `v`: The vector to transform
    ///
    /// # Returns
    /// The rotated vector
    ///
    /// # Examples
    ///
    /// ## Basic Usage
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use std::f32;
    /// # use nalgebra::{Isometry3, Translation3, UnitQuaternion, Vector3};
    /// let tra = Translation3::new(0.0, 0.0, 3.0);
    /// let rot = UnitQuaternion::from_scaled_axis(Vector3::y() * f32::consts::FRAC_PI_2);
    /// let iso = Isometry3::from_parts(tra, rot);
    ///
    /// let transformed_vector = iso.transform_vector(&Vector3::new(1.0, 2.0, 3.0));
    /// // Only rotated, not translated
    /// assert_relative_eq!(transformed_vector, Vector3::new(3.0, 2.0, -1.0), epsilon = 1.0e-6);
    /// ```
    ///
    /// ## Transform Direction Vectors
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{Isometry3, Vector3};
    /// // Object at (100, 200, 300) rotated 90° around Z
    /// let object = Isometry3::new(
    ///     Vector3::new(100.0, 200.0, 300.0),
    ///     Vector3::z() * std::f32::consts::FRAC_PI_2
    /// );
    ///
    /// // Local "forward" direction
    /// let forward_local = Vector3::new(1.0, 0.0, 0.0);
    ///
    /// // Get world-space forward direction
    /// let forward_world = object.transform_vector(&forward_local);
    ///
    /// // Translation doesn't affect direction vectors
    /// assert_relative_eq!(forward_world, Vector3::new(0.0, 1.0, 0.0), epsilon = 1.0e-6);
    /// ```
    ///
    /// ## Game: Character Looking Direction
    /// ```
    /// # use nalgebra::{Isometry3, Vector3};
    /// // Character at some position with some rotation
    /// let character = Isometry3::new(
    ///     Vector3::new(10.0, 0.0, 5.0),
    ///     Vector3::y() * std::f32::consts::FRAC_PI_4
    /// );
    ///
    /// // Standard "forward" in local space
    /// let local_forward = Vector3::new(0.0, 0.0, -1.0);
    ///
    /// // Which direction is character facing in world space?
    /// let world_forward = character.transform_vector(&local_forward);
    /// // Use this for movement, aiming, etc.
    /// ```
    ///
    /// ## Physics: Transform Velocity
    /// ```
    /// # use nalgebra::{Isometry2, Vector2};
    /// // Object in 2D with position and orientation
    /// let object = Isometry2::new(
    ///     Vector2::new(5.0, 10.0),
    ///     std::f32::consts::FRAC_PI_6
    /// );
    ///
    /// // Velocity in object's local frame (moving forward at 10 units/s)
    /// let velocity_local = Vector2::new(10.0, 0.0);
    ///
    /// // Transform to world velocity
    /// let velocity_world = object.transform_vector(&velocity_local);
    /// // This is the velocity vector in world coordinates
    /// ```
    ///
    /// ## Robotics: Transform Force Vector
    /// ```
    /// # use nalgebra::{Isometry3, Vector3};
    /// // Robot arm segment pose
    /// let arm_segment = Isometry3::new(
    ///     Vector3::new(0.5, 0.0, 1.0),
    ///     Vector3::z() * 0.3
    /// );
    ///
    /// // Force applied in arm's local frame
    /// let force_local = Vector3::new(0.0, 10.0, 0.0); // 10N upward
    ///
    /// // Transform to world frame for dynamics computation
    /// let force_world = arm_segment.transform_vector(&force_local);
    /// // Now we can sum forces in world coordinates
    /// ```
    ///
    /// # See Also
    /// - [`transform_point`](Self::transform_point) - Transform a point (includes translation)
    /// - [`inverse_transform_vector`](Self::inverse_transform_vector) - Transform by inverse rotation
    /// - Multiplication operator `*` - Alternative syntax for vector transformation
    #[inline]
    #[must_use]
    pub fn transform_vector(&self, v: &SVector<T, D>) -> SVector<T, D> {
        self * v
    }

    /// Transforms a point by the inverse of this isometry.
    ///
    /// This applies the inverse transformation without explicitly computing the full inverse
    /// isometry, making it more efficient than calling `self.inverse() * pt`.
    ///
    /// This is commonly used to transform from world space to local space, such as:
    /// - Converting world coordinates to camera/view space
    /// - Testing if a point is inside an object's local bounds
    /// - Finding where a collision occurred relative to an object
    ///
    /// The operation first subtracts the translation, then applies the inverse rotation.
    ///
    /// # Parameters
    /// - `pt`: The point to transform
    ///
    /// # Returns
    /// The point transformed by the inverse isometry
    ///
    /// # Examples
    ///
    /// ## Basic Usage
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use std::f32;
    /// # use nalgebra::{Isometry3, Translation3, UnitQuaternion, Vector3, Point3};
    /// let tra = Translation3::new(0.0, 0.0, 3.0);
    /// let rot = UnitQuaternion::from_scaled_axis(Vector3::y() * f32::consts::FRAC_PI_2);
    /// let iso = Isometry3::from_parts(tra, rot);
    ///
    /// let transformed_point = iso.inverse_transform_point(&Point3::new(1.0, 2.0, 3.0));
    /// assert_relative_eq!(transformed_point, Point3::new(0.0, 2.0, 1.0), epsilon = 1.0e-6);
    /// ```
    ///
    /// ## Transform World to Local Space
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{Isometry3, Vector3, Point3};
    /// // Object at (10, 0, 0) with no rotation
    /// let object = Isometry3::new(Vector3::new(10.0, 0.0, 0.0), Vector3::zeros());
    ///
    /// // World space point
    /// let world_point = Point3::new(15.0, 5.0, 3.0);
    ///
    /// // Convert to object's local coordinates
    /// let local_point = object.inverse_transform_point(&world_point);
    ///
    /// // In local space, the point is relative to object's center
    /// assert_relative_eq!(local_point, Point3::new(5.0, 5.0, 3.0), epsilon = 1.0e-6);
    /// ```
    ///
    /// ## Collision Detection: Local Bounds Test
    /// ```
    /// # use nalgebra::{Isometry3, Vector3, Point3};
    /// // Box at position (5, 3, 2) with rotation
    /// let box_transform = Isometry3::new(
    ///     Vector3::new(5.0, 3.0, 2.0),
    ///     Vector3::z() * 0.5
    /// );
    ///
    /// // Test if world point is inside box's local bounds [-1, 1] in each axis
    /// let world_point = Point3::new(6.0, 4.0, 2.0);
    /// let local_point = box_transform.inverse_transform_point(&world_point);
    ///
    /// let inside = local_point.x.abs() <= 1.0
    ///     && local_point.y.abs() <= 1.0
    ///     && local_point.z.abs() <= 1.0;
    /// ```
    ///
    /// ## Camera View Space Transformation
    /// ```
    /// # use nalgebra::{Isometry3, Vector3, Point3};
    /// // Camera position and orientation
    /// let camera = Isometry3::new(
    ///     Vector3::new(0.0, 2.0, 5.0),
    ///     Vector3::y() * std::f32::consts::PI
    /// );
    ///
    /// // Object in world space
    /// let world_object = Point3::new(1.0, 2.0, 3.0);
    ///
    /// // Transform to camera/view space for rendering
    /// let view_space_object = camera.inverse_transform_point(&world_object);
    /// // Now in camera coordinates for projection
    /// ```
    ///
    /// ## Mouse Picking: Screen to Object Space
    /// ```
    /// # use nalgebra::{Isometry2, Vector2, Point2};
    /// // UI element at (100, 50) rotated 30 degrees
    /// let ui_element = Isometry2::new(
    ///     Vector2::new(100.0, 50.0),
    ///     std::f32::consts::PI / 6.0
    /// );
    ///
    /// // Mouse click position in screen space
    /// let mouse_pos = Point2::new(110.0, 60.0);
    ///
    /// // Convert to UI element's local space
    /// let local_mouse = ui_element.inverse_transform_point(&mouse_pos);
    ///
    /// // Now can test against element's local bounds
    /// ```
    ///
    /// # See Also
    /// - [`transform_point`](Self::transform_point) - Transform by the forward isometry
    /// - [`inverse_transform_vector`](Self::inverse_transform_vector) - Transform vector by inverse
    /// - [`inverse`](Self::inverse) - Compute the full inverse isometry
    #[inline]
    #[must_use]
    pub fn inverse_transform_point(&self, pt: &Point<T, D>) -> Point<T, D> {
        self.rotation
            .inverse_transform_point(&(pt - &self.translation.vector))
    }

    /// Transforms a vector by the inverse rotation of this isometry.
    ///
    /// This applies only the inverse rotation to the vector, completely ignoring the
    /// translation component. This is more efficient than computing the full inverse
    /// isometry and then transforming the vector.
    ///
    /// Use this to transform direction vectors, velocities, or forces from world space
    /// back to local space. Since vectors don't have a position, the translation is
    /// irrelevant and only the rotation matters.
    ///
    /// # Parameters
    /// - `v`: The vector to transform
    ///
    /// # Returns
    /// The vector transformed by the inverse rotation
    ///
    /// # Examples
    ///
    /// ## Basic Usage
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use std::f32;
    /// # use nalgebra::{Isometry3, Translation3, UnitQuaternion, Vector3};
    /// let tra = Translation3::new(0.0, 0.0, 3.0);
    /// let rot = UnitQuaternion::from_scaled_axis(Vector3::y() * f32::consts::FRAC_PI_2);
    /// let iso = Isometry3::from_parts(tra, rot);
    ///
    /// let transformed_vector = iso.inverse_transform_vector(&Vector3::new(1.0, 2.0, 3.0));
    /// assert_relative_eq!(transformed_vector, Vector3::new(-3.0, 2.0, 1.0), epsilon = 1.0e-6);
    /// ```
    ///
    /// ## Convert World Direction to Local
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{Isometry3, Vector3};
    /// // Object rotated 90° around Z axis
    /// let object = Isometry3::new(
    ///     Vector3::new(5.0, 10.0, 0.0),
    ///     Vector3::z() * std::f32::consts::FRAC_PI_2
    /// );
    ///
    /// // Vector pointing north in world space
    /// let world_north = Vector3::new(0.0, 1.0, 0.0);
    ///
    /// // What direction is "north" in the object's local frame?
    /// let local_direction = object.inverse_transform_vector(&world_north);
    ///
    /// // Due to 90° rotation, world +Y becomes local -X
    /// assert_relative_eq!(local_direction, Vector3::new(-1.0, 0.0, 0.0), epsilon = 1.0e-6);
    /// ```
    ///
    /// ## Game: Relative Velocity
    /// ```
    /// # use nalgebra::{Isometry2, Vector2};
    /// // Vehicle at some position with heading
    /// let vehicle = Isometry2::new(
    ///     Vector2::new(100.0, 50.0),
    ///     std::f32::consts::FRAC_PI_4
    /// );
    ///
    /// // World velocity (maybe from physics engine)
    /// let world_velocity = Vector2::new(10.0, 10.0);
    ///
    /// // Convert to vehicle's local frame to determine forward/sideways motion
    /// let local_velocity = vehicle.inverse_transform_vector(&world_velocity);
    ///
    /// // Now can determine if vehicle is moving forward (local +X)
    /// // or sliding sideways (local +Y)
    /// ```
    ///
    /// ## Physics: Force in Local Coordinates
    /// ```
    /// # use nalgebra::{Isometry3, Vector3};
    /// // Rigid body with some orientation
    /// let body = Isometry3::new(
    ///     Vector3::new(0.0, 0.0, 0.0),
    ///     Vector3::y() * 0.5
    /// );
    ///
    /// // Applied force in world space (e.g., gravity, wind)
    /// let world_force = Vector3::new(0.0, -9.81, 0.0);
    ///
    /// // Transform to body's local frame for torque calculation
    /// let local_force = body.inverse_transform_vector(&world_force);
    ///
    /// // Now can compute torques in body frame
    /// ```
    ///
    /// ## Camera: Screen to View Direction
    /// ```
    /// # use nalgebra::{Isometry3, Vector3};
    /// // Camera transform
    /// let camera = Isometry3::new(
    ///     Vector3::new(0.0, 5.0, 10.0),
    ///     Vector3::x() * (-std::f32::consts::FRAC_PI_6)
    /// );
    ///
    /// // Ray direction in world space (from screen picking)
    /// let world_ray = Vector3::new(0.1, 0.2, -1.0).normalize();
    ///
    /// // Convert to camera's local view space
    /// let view_ray = camera.inverse_transform_vector(&world_ray);
    ///
    /// // Now in camera's coordinate system for frustum tests
    /// ```
    ///
    /// ## Robotics: Sensor Reading to Body Frame
    /// ```
    /// # use nalgebra::{Isometry2, Vector2};
    /// // Robot pose
    /// let robot = Isometry2::new(
    ///     Vector2::new(5.0, 3.0),
    ///     std::f32::consts::PI / 3.0
    /// );
    ///
    /// // Detected velocity of nearby object in world frame
    /// let object_velocity_world = Vector2::new(-2.0, 3.0);
    ///
    /// // Transform to robot's frame for collision avoidance
    /// let object_velocity_local = robot.inverse_transform_vector(&object_velocity_world);
    ///
    /// // Now robot can reason about object motion relative to itself
    /// ```
    ///
    /// # See Also
    /// - [`transform_vector`](Self::transform_vector) - Transform by forward rotation
    /// - [`inverse_transform_point`](Self::inverse_transform_point) - Transform point by inverse
    /// - [`inverse_transform_unit_vector`](Self::inverse_transform_unit_vector) - Specialized for unit vectors
    #[inline]
    #[must_use]
    pub fn inverse_transform_vector(&self, v: &SVector<T, D>) -> SVector<T, D> {
        self.rotation.inverse_transform_vector(v)
    }

    /// Transforms a unit vector by the inverse rotation of this isometry.
    ///
    /// This is a specialized version of [`inverse_transform_vector`](Self::inverse_transform_vector)
    /// optimized for unit vectors (vectors with length 1). The result is guaranteed to remain
    /// a unit vector.
    ///
    /// Use this for transforming normalized directions like surface normals, axis vectors,
    /// or any other direction that must remain normalized. It's more efficient than
    /// [`inverse_transform_vector`](Self::inverse_transform_vector) followed by normalization.
    ///
    /// The translation component is ignored (as with all vector transformations).
    ///
    /// # Parameters
    /// - `v`: The unit vector to transform
    ///
    /// # Returns
    /// The transformed unit vector
    ///
    /// # Examples
    ///
    /// ## Basic Usage
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use std::f32;
    /// # use nalgebra::{Isometry3, Translation3, UnitQuaternion, Vector3};
    /// let tra = Translation3::new(0.0, 0.0, 3.0);
    /// let rot = UnitQuaternion::from_scaled_axis(Vector3::z() * f32::consts::FRAC_PI_2);
    /// let iso = Isometry3::from_parts(tra, rot);
    ///
    /// let transformed = iso.inverse_transform_unit_vector(&Vector3::x_axis());
    /// assert_relative_eq!(transformed, -Vector3::y_axis(), epsilon = 1.0e-6);
    /// ```
    ///
    /// ## Transform Surface Normals
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{Isometry3, Vector3};
    /// // Object rotated 45° around Y axis
    /// let object = Isometry3::new(
    ///     Vector3::new(10.0, 5.0, 0.0),
    ///     Vector3::y() * std::f32::consts::FRAC_PI_4
    /// );
    ///
    /// // Surface normal in world space (pointing up-ish)
    /// let world_normal = Vector3::y_axis();
    ///
    /// // Convert to object's local frame for lighting calculations
    /// let local_normal = object.inverse_transform_unit_vector(&world_normal);
    ///
    /// // Still a unit vector after transformation
    /// assert_relative_eq!(local_normal.norm(), 1.0, epsilon = 1.0e-6);
    /// ```
    ///
    /// ## Graphics: Light Direction in Object Space
    /// ```
    /// # use nalgebra::{Isometry3, Vector3};
    /// // Rotated object
    /// let object = Isometry3::new(
    ///     Vector3::new(0.0, 0.0, 0.0),
    ///     Vector3::z() * 0.5
    /// );
    ///
    /// // Light direction in world space (sun direction)
    /// let world_light_dir = Vector3::new(1.0, -1.0, 0.0).normalize();
    /// let world_light_dir = Unit::new_unchecked(world_light_dir);
    ///
    /// // Transform to object space for shader calculations
    /// let local_light_dir = object.inverse_transform_unit_vector(&world_light_dir);
    ///
    /// // Can now compute per-vertex lighting in object space
    /// ```
    ///
    /// ## Physics: Contact Normal in Body Frame
    /// ```
    /// # use nalgebra::{Isometry3, Vector3, Unit};
    /// // Rigid body with some orientation
    /// let body = Isometry3::new(
    ///     Vector3::new(5.0, 2.0, 1.0),
    ///     Vector3::x() * 0.3
    /// );
    ///
    /// // Contact normal pointing away from collision (world space)
    /// let contact_normal_world = Unit::new_normalize(Vector3::new(0.0, 1.0, 0.1));
    ///
    /// // Convert to body's local frame for impulse calculation
    /// let contact_normal_local = body.inverse_transform_unit_vector(&contact_normal_world);
    ///
    /// // Now can compute collision response in body coordinates
    /// ```
    ///
    /// ## Camera: View Direction
    /// ```
    /// # use nalgebra::{Isometry3, Vector3};
    /// // Camera transform
    /// let camera = Isometry3::new(
    ///     Vector3::new(0.0, 10.0, 20.0),
    ///     Vector3::x() * (-0.5)
    /// );
    ///
    /// // Ray direction in world space (from raycast)
    /// let world_ray_dir = Vector3::new(0.1, -0.2, -1.0).normalize();
    /// let world_ray_dir = Unit::new_unchecked(world_ray_dir);
    ///
    /// // Convert to camera/view space for clipping tests
    /// let view_ray_dir = camera.inverse_transform_unit_vector(&world_ray_dir);
    /// ```
    ///
    /// ## Robotics: Desired Heading in Body Frame
    /// ```
    /// # use nalgebra::{Isometry2, Vector2, Unit};
    /// // Robot at position with heading
    /// let robot = Isometry2::new(
    ///     Vector2::new(10.0, 5.0),
    ///     std::f32::consts::PI / 3.0
    /// );
    ///
    /// // Desired movement direction in world frame (to waypoint)
    /// let desired_dir_world = Unit::new_normalize(Vector2::new(1.0, 1.0));
    ///
    /// // Convert to robot's frame for steering control
    /// let desired_dir_local = robot.inverse_transform_unit_vector(&desired_dir_world);
    ///
    /// // Now can compute steering angle from local direction
    /// ```
    ///
    /// # See Also
    /// - [`inverse_transform_vector`](Self::inverse_transform_vector) - Transform non-unit vectors
    /// - [`inverse_transform_point`](Self::inverse_transform_point) - Transform points
    /// - [`transform_vector`](Self::transform_vector) - Forward transformation
    #[inline]
    #[must_use]
    pub fn inverse_transform_unit_vector(&self, v: &Unit<SVector<T, D>>) -> Unit<SVector<T, D>> {
        self.rotation.inverse_transform_unit_vector(v)
    }
}

// NOTE: we don't require `R: Rotation<...>` here because this is not useful for the implementation
// and makes it hard to use it, e.g., for Transform × Isometry implementation.
// This is OK since all constructors of the isometry enforce the Rotation bound already (and
// explicit struct construction is prevented by the dummy ZST field).
/// # Conversion to a matrix
impl<T: SimdRealField, R, const D: usize> Isometry<T, R, D> {
    /// Converts this isometry into its equivalent homogeneous transformation matrix.
    ///
    /// A homogeneous transformation matrix is a square matrix that combines rotation and
    /// translation into a single matrix representation. This is widely used in computer
    /// graphics pipelines, robotics kinematics, and physics engines because it allows
    /// chaining multiple transformations through matrix multiplication.
    ///
    /// For 2D isometries, this produces a 3×3 matrix. For 3D isometries, a 4×4 matrix.
    /// The last row is always [0, 0, ..., 0, 1].
    ///
    /// This method is identical to [`to_matrix`](Self::to_matrix).
    ///
    /// # Returns
    /// A homogeneous transformation matrix representing this isometry
    ///
    /// # Examples
    ///
    /// ## Basic 2D Conversion
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use std::f32;
    /// # use nalgebra::{Isometry2, Vector2, Matrix3};
    /// let iso = Isometry2::new(Vector2::new(10.0, 20.0), f32::consts::FRAC_PI_6);
    ///
    /// // Convert to 3×3 homogeneous matrix
    /// let matrix = iso.to_homogeneous();
    ///
    /// let expected = Matrix3::new(0.8660254, -0.5,      10.0,
    ///                             0.5,       0.8660254, 20.0,
    ///                             0.0,       0.0,       1.0);
    ///
    /// assert_relative_eq!(matrix, expected, epsilon = 1.0e-6);
    /// ```
    ///
    /// ## 3D Graphics Pipeline
    /// ```
    /// # use nalgebra::{Isometry3, Vector3, Matrix4};
    /// // Object transformation
    /// let object = Isometry3::new(
    ///     Vector3::new(5.0, 2.0, 1.0),
    ///     Vector3::y() * std::f32::consts::FRAC_PI_4
    /// );
    ///
    /// // Convert to matrix for shader uniform
    /// let model_matrix: Matrix4<f32> = object.to_homogeneous();
    ///
    /// // Can now multiply with view and projection matrices
    /// // let mvp = projection * view * model_matrix;
    /// ```
    ///
    /// ## Chaining Transformations
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{Isometry3, Vector3, Point3};
    /// // Parent object (e.g., character body)
    /// let parent = Isometry3::new(Vector3::new(10.0, 0.0, 0.0), Vector3::zeros());
    ///
    /// // Child object offset (e.g., weapon)
    /// let child_offset = Isometry3::new(Vector3::new(1.0, 0.5, 0.0), Vector3::zeros());
    ///
    /// // Combine using matrix multiplication
    /// let parent_matrix = parent.to_homogeneous();
    /// let child_matrix = child_offset.to_homogeneous();
    /// let world_matrix = parent_matrix * child_matrix;
    ///
    /// // Now world_matrix represents child in world space
    /// ```
    ///
    /// ## Robotics: Forward Kinematics
    /// ```
    /// # use nalgebra::{Isometry3, Vector3, Matrix4};
    /// // Joint transforms in a robot arm
    /// let joint1 = Isometry3::new(Vector3::new(0.0, 0.0, 1.0), Vector3::z() * 0.5);
    /// let joint2 = Isometry3::new(Vector3::new(0.0, 0.0, 1.0), Vector3::z() * 0.3);
    /// let joint3 = Isometry3::new(Vector3::new(0.0, 0.0, 0.5), Vector3::z() * 0.2);
    ///
    /// // Compute end-effector pose by chaining matrices
    /// let m1 = joint1.to_homogeneous();
    /// let m2 = joint2.to_homogeneous();
    /// let m3 = joint3.to_homogeneous();
    ///
    /// let end_effector_matrix = m1 * m2 * m3;
    /// // This gives the position and orientation of the robot's end effector
    /// ```
    ///
    /// ## Interoperability with External Libraries
    /// ```
    /// # use nalgebra::{Isometry3, Vector3};
    /// // Create isometry
    /// let transform = Isometry3::new(Vector3::new(1.0, 2.0, 3.0), Vector3::zeros());
    ///
    /// // Convert to matrix for passing to graphics API
    /// let matrix = transform.to_homogeneous();
    ///
    /// // Extract raw data for OpenGL/WebGL/etc
    /// let matrix_data: &[f32; 16] = matrix.as_slice().try_into().unwrap();
    /// // Can now pass matrix_data to glUniformMatrix4fv or similar
    /// ```
    ///
    /// # See Also
    /// - [`to_matrix`](Self::to_matrix) - Identical method with different name
    /// - [`from_parts`](Self::from_parts) - Create isometry from components
    /// - Matrix multiplication for combining transformations
    #[inline]
    #[must_use]
    pub fn to_homogeneous(&self) -> OMatrix<T, DimNameSum<Const<D>, U1>, DimNameSum<Const<D>, U1>>
    where
        Const<D>: DimNameAdd<U1>,
        R: SubsetOf<OMatrix<T, DimNameSum<Const<D>, U1>, DimNameSum<Const<D>, U1>>>,
        DefaultAllocator: Allocator<DimNameSum<Const<D>, U1>, DimNameSum<Const<D>, U1>>,
    {
        let mut res: OMatrix<T, _, _> = crate::convert_ref(&self.rotation);
        res.fixed_view_mut::<D, 1>(0, D)
            .copy_from(&self.translation.vector);

        res
    }

    /// Converts this isometry into its equivalent homogeneous transformation matrix.
    ///
    /// A homogeneous transformation matrix is a square matrix representation that combines
    /// both rotation and translation. This format is essential for computer graphics and
    /// robotics because it enables efficient composition of multiple transformations through
    /// standard matrix multiplication.
    ///
    /// The resulting matrix has the rotation in the upper-left block, translation in the
    /// rightmost column (or last column for 2D), and [0, 0, ..., 0, 1] in the bottom row.
    ///
    /// This method is identical to [`to_homogeneous`](Self::to_homogeneous).
    ///
    /// # Returns
    /// A homogeneous transformation matrix (3×3 for 2D, 4×4 for 3D)
    ///
    /// # Examples
    ///
    /// ## Basic 2D Usage
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use std::f32;
    /// # use nalgebra::{Isometry2, Vector2, Matrix3};
    /// let iso = Isometry2::new(Vector2::new(10.0, 20.0), f32::consts::FRAC_PI_6);
    ///
    /// // Convert to matrix form
    /// let matrix = iso.to_matrix();
    ///
    /// let expected = Matrix3::new(0.8660254, -0.5,      10.0,
    ///                             0.5,       0.8660254, 20.0,
    ///                             0.0,       0.0,       1.0);
    ///
    /// assert_relative_eq!(matrix, expected, epsilon = 1.0e-6);
    /// ```
    ///
    /// ## Building Model-View-Projection Matrix
    /// ```
    /// # use nalgebra::{Isometry3, Perspective3, Vector3, Matrix4};
    /// // Camera view transform
    /// let view = Isometry3::new(
    ///     Vector3::new(0.0, 2.0, 5.0),
    ///     Vector3::zeros()
    /// ).inverse();
    ///
    /// // Object model transform
    /// let model = Isometry3::new(
    ///     Vector3::new(1.0, 0.0, 0.0),
    ///     Vector3::y() * 0.5
    /// );
    ///
    /// // Projection matrix
    /// let projection = Perspective3::new(16.0 / 9.0, 3.14 / 4.0, 0.1, 100.0);
    ///
    /// // Combine all transformations
    /// let view_matrix = view.to_matrix();
    /// let model_matrix = model.to_matrix();
    /// let mvp = projection.to_homogeneous() * view_matrix * model_matrix;
    /// ```
    ///
    /// ## Skeletal Animation Bone Transform
    /// ```
    /// # use nalgebra::{Isometry3, Vector3, Matrix4, Point3};
    /// // Bone rest pose
    /// let rest_pose = Isometry3::new(Vector3::new(0.0, 1.0, 0.0), Vector3::zeros());
    ///
    /// // Animated pose
    /// let animated_pose = Isometry3::new(
    ///     Vector3::new(0.0, 1.0, 0.0),
    ///     Vector3::z() * 0.3
    /// );
    ///
    /// // Compute skinning matrix: animated * rest_inverse
    /// let rest_inverse = rest_pose.inverse().to_matrix();
    /// let animated_matrix = animated_pose.to_matrix();
    /// let skinning_matrix = animated_matrix * rest_inverse;
    ///
    /// // Apply to vertex positions for skeletal animation
    /// ```
    ///
    /// ## Physics: Inertia Tensor Transformation
    /// ```
    /// # use nalgebra::{Isometry3, Vector3, Matrix3};
    /// // Body transform
    /// let body = Isometry3::new(
    ///     Vector3::new(5.0, 2.0, 1.0),
    ///     Vector3::y() * 0.5
    /// );
    ///
    /// // Extract rotation matrix for transforming inertia tensor
    /// let transform = body.to_matrix();
    ///
    /// // Inertia tensor in body frame
    /// let inertia_local = Matrix3::from_diagonal(&Vector3::new(1.0, 2.0, 1.0));
    ///
    /// // Transform to world frame: R * I * R^T
    /// // (where R is the rotation part of the transform)
    /// ```
    ///
    /// ## Scene Graph Hierarchy
    /// ```
    /// # use nalgebra::{Isometry3, Vector3};
    /// // Parent transform (e.g., vehicle)
    /// let vehicle = Isometry3::new(Vector3::new(10.0, 0.0, 5.0), Vector3::y() * 1.5);
    ///
    /// // Child transform (e.g., wheel, relative to vehicle)
    /// let wheel_offset = Isometry3::new(Vector3::new(1.5, -0.5, 1.0), Vector3::zeros());
    ///
    /// // Compute wheel's world transform
    /// let vehicle_matrix = vehicle.to_matrix();
    /// let wheel_matrix = wheel_offset.to_matrix();
    /// let wheel_world = vehicle_matrix * wheel_matrix;
    ///
    /// // wheel_world now contains the wheel's position in world space
    /// ```
    ///
    /// # See Also
    /// - [`to_homogeneous`](Self::to_homogeneous) - Identical method with different name
    /// - [`from_parts`](Self::from_parts) - Construct isometry from components
    /// - Matrix multiplication operators for composing transformations
    #[inline]
    #[must_use]
    pub fn to_matrix(&self) -> OMatrix<T, DimNameSum<Const<D>, U1>, DimNameSum<Const<D>, U1>>
    where
        Const<D>: DimNameAdd<U1>,
        R: SubsetOf<OMatrix<T, DimNameSum<Const<D>, U1>, DimNameSum<Const<D>, U1>>>,
        DefaultAllocator: Allocator<DimNameSum<Const<D>, U1>, DimNameSum<Const<D>, U1>>,
    {
        self.to_homogeneous()
    }
}

impl<T: SimdRealField, R, const D: usize> Eq for Isometry<T, R, D> where
    R: AbstractRotation<T, D> + Eq
{
}

impl<T: SimdRealField, R, const D: usize> PartialEq for Isometry<T, R, D>
where
    R: AbstractRotation<T, D> + PartialEq,
{
    #[inline]
    fn eq(&self, right: &Self) -> bool {
        self.translation == right.translation && self.rotation == right.rotation
    }
}

impl<T: RealField, R, const D: usize> AbsDiffEq for Isometry<T, R, D>
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
        self.translation
            .abs_diff_eq(&other.translation, epsilon.clone())
            && self.rotation.abs_diff_eq(&other.rotation, epsilon)
    }
}

impl<T: RealField, R, const D: usize> RelativeEq for Isometry<T, R, D>
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
        self.translation
            .relative_eq(&other.translation, epsilon.clone(), max_relative.clone())
            && self
                .rotation
                .relative_eq(&other.rotation, epsilon, max_relative)
    }
}

impl<T: RealField, R, const D: usize> UlpsEq for Isometry<T, R, D>
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
        self.translation
            .ulps_eq(&other.translation, epsilon.clone(), max_ulps)
            && self.rotation.ulps_eq(&other.rotation, epsilon, max_ulps)
    }
}

/*
 *
 * Display
 *
 */
impl<T: RealField + fmt::Display, R, const D: usize> fmt::Display for Isometry<T, R, D>
where
    R: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let precision = f.precision().unwrap_or(3);

        writeln!(f, "Isometry {{")?;
        write!(f, "{:.*}", precision, self.translation)?;
        write!(f, "{:.*}", precision, self.rotation)?;
        writeln!(f, "}}")
    }
}
