#[cfg(feature = "arbitrary")]
use crate::base::storage::Owned;
#[cfg(feature = "arbitrary")]
use quickcheck::{Arbitrary, Gen};

use num::One;
#[cfg(feature = "rand-no-std")]
use rand::{
    Rng,
    distr::{Distribution, StandardUniform},
};

use simba::scalar::SupersetOf;
use simba::simd::SimdRealField;

use crate::base::{Vector2, Vector3};

use crate::{
    AbstractRotation, Isometry, Isometry2, Isometry3, IsometryMatrix2, IsometryMatrix3, Point,
    Point3, Rotation, Rotation3, Scalar, Translation, Translation2, Translation3, UnitComplex,
    UnitQuaternion,
};

impl<T: SimdRealField, R: AbstractRotation<T, D>, const D: usize> Default for Isometry<T, R, D>
where
    T::Element: SimdRealField,
{
    fn default() -> Self {
        Self::identity()
    }
}

impl<T: SimdRealField, R: AbstractRotation<T, D>, const D: usize> Isometry<T, R, D>
where
    T::Element: SimdRealField,
{
    /// Creates a new identity isometry.
    ///
    /// An **isometry** is a rigid body transformation that preserves distances and angles.
    /// It combines a rotation and a translation, making it perfect for representing the
    /// position and orientation of objects in games, robotics, and physics simulations.
    ///
    /// The identity isometry performs no transformation at all - it's like having an object
    /// at the origin with no rotation. When you apply it to any point or vector, you get
    /// back the exact same point or vector unchanged.
    ///
    /// # Use Cases
    /// - Initialize objects at their default position and orientation
    /// - Reset transformations to a neutral state
    /// - Use as a starting point before applying transformations
    /// - Represent "no transformation" in physics engines
    ///
    /// # Examples
    ///
    /// ```
    /// # use nalgebra::{Isometry2, Point2, Isometry3, Point3};
    ///
    /// // 2D identity isometry - leaves all points unchanged
    /// let iso = Isometry2::identity();
    /// let pt = Point2::new(1.0, 2.0);
    /// assert_eq!(iso * pt, pt);
    ///
    /// // 3D identity isometry - same behavior in 3D
    /// let iso = Isometry3::identity();
    /// let pt = Point3::new(1.0, 2.0, 3.0);
    /// assert_eq!(iso * pt, pt);
    /// ```
    ///
    /// ## Practical Example: Game Object Initialization
    ///
    /// ```
    /// # use nalgebra::{Isometry3, Point3};
    /// // Initialize a game object at the origin with no rotation
    /// struct GameObject {
    ///     transform: Isometry3<f32>,
    /// }
    ///
    /// let player = GameObject {
    ///     transform: Isometry3::identity(),
    /// };
    ///
    /// // The player starts at the origin
    /// let spawn_point = Point3::origin();
    /// assert_eq!(player.transform * spawn_point, spawn_point);
    /// ```
    ///
    /// # See Also
    /// - [`Isometry::new`] - Create an isometry with specific translation and rotation
    /// - [`Isometry::translation`] - Create an isometry with only translation
    /// - [`Isometry::rotation`] - Create an isometry with only rotation
    #[inline]
    pub fn identity() -> Self {
        Self::from_parts(Translation::identity(), R::identity())
    }

    /// Creates an isometry that rotates around a specific point.
    ///
    /// This function creates a **rotation around a point** transformation. Unlike a simple rotation
    /// (which rotates around the origin), this keeps the specified point `p` fixed in place while
    /// rotating everything else around it. Think of it like spinning a door around its hinges -
    /// the hinge point stays fixed while the rest of the door moves.
    ///
    /// This is incredibly useful for rotating objects around their center, rotating a camera around
    /// a target, or any situation where you want to rotate around a point other than the origin.
    ///
    /// # Arguments
    /// - `r` - The rotation to apply
    /// - `p` - The center point for the rotation (this point will remain unchanged)
    ///
    /// # Use Cases
    /// - Rotate a game object around its center instead of the world origin
    /// - Implement orbital camera movement around a target point
    /// - Rotate a robot arm around a joint
    /// - Spin objects around a pivot point
    ///
    /// # Examples
    ///
    /// ## Basic 2D Rotation Around a Point
    ///
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use std::f32;
    /// # use nalgebra::{Isometry2, Point2, UnitComplex};
    ///
    /// // Rotate 180 degrees around the point (1.0, 0.0)
    /// let rot = UnitComplex::new(f32::consts::PI);
    /// let center = Point2::new(1.0, 0.0);
    /// let iso = Isometry2::rotation_wrt_point(rot, center);
    ///
    /// // The center point stays exactly where it is
    /// assert_eq!(iso * center, center);
    ///
    /// // A point above the center gets flipped to below it
    /// assert_relative_eq!(iso * Point2::new(1.0, 2.0), Point2::new(1.0, -2.0), epsilon = 1.0e-6);
    /// ```
    ///
    /// ## Practical Example: Rotating a Game Object Around Its Center
    ///
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use std::f32;
    /// # use nalgebra::{Isometry2, Point2, UnitComplex, Vector2};
    ///
    /// // A rectangular game object with corners relative to world origin
    /// let object_center = Point2::new(5.0, 5.0);
    /// let corner = Point2::new(6.0, 6.0);  // 1 unit away from center diagonally
    ///
    /// // Rotate the object 90 degrees counter-clockwise around its center
    /// let rotation = UnitComplex::new(f32::consts::FRAC_PI_2);
    /// let transform = Isometry2::rotation_wrt_point(rotation, object_center);
    ///
    /// // The center doesn't move
    /// assert_relative_eq!(transform * object_center, object_center, epsilon = 1.0e-6);
    ///
    /// // The corner rotates around the center
    /// let new_corner = transform * corner;
    /// assert_relative_eq!(new_corner, Point2::new(4.0, 6.0), epsilon = 1.0e-6);
    /// ```
    ///
    /// ## Practical Example: Camera Orbiting Around Target
    ///
    /// ```
    /// # use std::f32;
    /// # use nalgebra::{Isometry2, Point2, UnitComplex};
    ///
    /// // Camera orbiting around a target at 45 degrees
    /// let target = Point2::new(10.0, 10.0);
    /// let camera_pos = Point2::new(15.0, 10.0);  // 5 units to the right of target
    ///
    /// let orbit_angle = f32::consts::FRAC_PI_4;  // 45 degrees
    /// let orbit_transform = Isometry2::rotation_wrt_point(
    ///     UnitComplex::new(orbit_angle),
    ///     target
    /// );
    ///
    /// let new_camera_pos = orbit_transform * camera_pos;
    /// // Camera has moved to a new position around the target
    /// // Target remains at the same location
    /// assert_eq!(orbit_transform * target, target);
    /// ```
    ///
    /// # See Also
    /// - [`Isometry::rotation`] - Create a rotation around the origin
    /// - [`Isometry::new`] - Create an isometry from translation and rotation components
    /// - [`UnitComplex::new`] - Create a 2D rotation (for use with Isometry2)
    /// - [`UnitQuaternion::from_axis_angle`] - Create a 3D rotation (for use with Isometry3)
    #[inline]
    pub fn rotation_wrt_point(r: R, p: Point<T, D>) -> Self {
        let shift = r.transform_vector(&-&p.coords);
        Self::from_parts(Translation::from(shift + p.coords), r)
    }
}

impl<T: SimdRealField, R: AbstractRotation<T, D>, const D: usize> One for Isometry<T, R, D>
where
    T::Element: SimdRealField,
{
    /// Creates a new identity isometry.
    #[inline]
    fn one() -> Self {
        Self::identity()
    }
}

#[cfg(feature = "rand-no-std")]
impl<T: crate::RealField, R, const D: usize> Distribution<Isometry<T, R, D>> for StandardUniform
where
    R: AbstractRotation<T, D>,
    StandardUniform: Distribution<T> + Distribution<R>,
{
    #[inline]
    fn sample<G: Rng + ?Sized>(&self, rng: &mut G) -> Isometry<T, R, D> {
        Isometry::from_parts(rng.random(), rng.random())
    }
}

#[cfg(feature = "arbitrary")]
impl<T, R, const D: usize> Arbitrary for Isometry<T, R, D>
where
    T: SimdRealField + Arbitrary + Send,
    T::Element: SimdRealField,
    R: AbstractRotation<T, D> + Arbitrary + Send,
    Owned<T, crate::Const<D>>: Send,
{
    #[inline]
    fn arbitrary(rng: &mut Gen) -> Self {
        Self::from_parts(Arbitrary::arbitrary(rng), Arbitrary::arbitrary(rng))
    }
}

/*
 *
 * Constructors for various static dimensions.
 *
 */

/// # Construction from a 2D vector and/or a rotation angle
impl<T: SimdRealField> IsometryMatrix2<T>
where
    T::Element: SimdRealField,
{
    /// Creates a new 2D isometry from a translation and a rotation angle.
    ///
    /// An **isometry** is a rigid body transformation that combines translation (moving) and
    /// rotation (spinning) while preserving distances and angles. This is the fundamental
    /// transformation for 2D games, robotics, and physics simulations.
    ///
    /// The rotational part is represented as a 2x2 rotation matrix internally, making it
    /// efficient for matrix operations. This is an `IsometryMatrix2`, which uses matrix
    /// representation for rotations.
    ///
    /// # Arguments
    /// - `translation` - A 2D vector specifying how far to move in x and y directions
    /// - `angle` - The rotation angle in radians (counter-clockwise, positive)
    ///
    /// # Use Cases
    /// - Position and orient 2D game objects (sprites, characters, obstacles)
    /// - Represent robot poses in 2D navigation
    /// - Transform coordinate frames in 2D physics
    /// - Camera transformations in 2D games
    ///
    /// # Examples
    ///
    /// ## Basic Usage
    ///
    /// ```
    /// # use std::f32;
    /// # use nalgebra::{IsometryMatrix2, Vector2, Point2};
    ///
    /// // Move 1 unit right, 2 units up, and rotate 90 degrees counter-clockwise
    /// let iso = IsometryMatrix2::new(Vector2::new(1.0, 2.0), f32::consts::FRAC_PI_2);
    ///
    /// // Transform a point: first rotate, then translate
    /// let pt = Point2::new(3.0, 4.0);
    /// let result = iso * pt;
    /// assert_eq!(result, Point2::new(-3.0, 5.0));
    /// ```
    ///
    /// ## Practical Example: Positioning a 2D Game Object
    ///
    /// ```
    /// # use std::f32;
    /// # use nalgebra::{IsometryMatrix2, Vector2, Point2};
    ///
    /// // Create a player at position (100, 50) facing right (0 degrees)
    /// let player_transform = IsometryMatrix2::new(
    ///     Vector2::new(100.0, 50.0),
    ///     0.0  // No rotation, facing right
    /// );
    ///
    /// // The player's local "front" point (1, 0) in world space
    /// let local_front = Point2::new(1.0, 0.0);
    /// let world_front = player_transform * local_front;
    /// assert_eq!(world_front, Point2::new(101.0, 50.0));
    /// ```
    ///
    /// ## Practical Example: Rotating and Moving a Car
    ///
    /// ```
    /// # use std::f32;
    /// # use nalgebra::{IsometryMatrix2, Vector2, Point2};
    ///
    /// // Car at position (10, 20), rotated 45 degrees
    /// let car_position = Vector2::new(10.0, 20.0);
    /// let car_angle = f32::consts::FRAC_PI_4;  // 45 degrees
    /// let car_transform = IsometryMatrix2::new(car_position, car_angle);
    ///
    /// // Transform the car's front bumper from local to world coordinates
    /// // In car's local space, front bumper is at (2, 0)
    /// let local_bumper = Point2::new(2.0, 0.0);
    /// let world_bumper = car_transform * local_bumper;
    ///
    /// // The bumper is now rotated and translated in world space
    /// let expected_x = 10.0 + 2.0 * f32::consts::FRAC_PI_4.cos();
    /// let expected_y = 20.0 + 2.0 * f32::consts::FRAC_PI_4.sin();
    /// assert!((world_bumper.x - expected_x).abs() < 1e-6);
    /// assert!((world_bumper.y - expected_y).abs() < 1e-6);
    /// ```
    ///
    /// ## Understanding Transformation Order
    ///
    /// ```
    /// # use std::f32;
    /// # use nalgebra::{IsometryMatrix2, Vector2, Point2};
    ///
    /// // Important: Isometry applies rotation FIRST, then translation
    /// let iso = IsometryMatrix2::new(Vector2::new(5.0, 0.0), f32::consts::FRAC_PI_2);
    ///
    /// // Point at (1, 0) is rotated 90° to (0, 1), then translated by (5, 0)
    /// let pt = Point2::new(1.0, 0.0);
    /// let result = iso * pt;
    /// assert_eq!(result, Point2::new(5.0, 1.0));
    /// ```
    ///
    /// # See Also
    /// - [`IsometryMatrix2::translation`] - Create an isometry with only translation
    /// - [`IsometryMatrix2::rotation`] - Create an isometry with only rotation
    /// - [`IsometryMatrix2::identity`] - Create an identity isometry (no transformation)
    /// - [`Isometry2::new`] - Similar but uses UnitComplex for rotation (more compact)
    #[inline]
    pub fn new(translation: Vector2<T>, angle: T) -> Self {
        Self::from_parts(Translation::from(translation), Rotation::<T, 2>::new(angle))
    }

    /// Creates a new 2D isometry with only translation (no rotation).
    ///
    /// This creates a pure translation transformation - it moves points by the specified
    /// amounts in the x and y directions without any rotation. The resulting isometry
    /// behaves like shifting everything on a 2D plane.
    ///
    /// This is useful when you need to move objects without changing their orientation,
    /// such as sliding a platform, moving a character in a fixed direction, or
    /// translating a camera view.
    ///
    /// # Arguments
    /// - `x` - Distance to move along the x-axis
    /// - `y` - Distance to move along the y-axis
    ///
    /// # Use Cases
    /// - Move objects without rotating them
    /// - Implement smooth linear movement in games
    /// - Offset coordinate systems
    /// - Camera panning without rotation
    ///
    /// # Examples
    ///
    /// ## Basic Translation
    ///
    /// ```
    /// # use nalgebra::{IsometryMatrix2, Point2};
    ///
    /// // Create a translation that moves everything 3 units right and 4 units up
    /// let iso = IsometryMatrix2::translation(3.0, 4.0);
    ///
    /// // Transform a point
    /// let pt = Point2::new(1.0, 2.0);
    /// let result = iso * pt;
    /// assert_eq!(result, Point2::new(4.0, 6.0));
    /// ```
    ///
    /// ## Practical Example: Moving a Platform
    ///
    /// ```
    /// # use nalgebra::{IsometryMatrix2, Point2};
    ///
    /// // A moving platform that slides horizontally
    /// let platform_offset = 10.0;
    /// let platform_transform = IsometryMatrix2::translation(platform_offset, 0.0);
    ///
    /// // Transform the platform's corner points
    /// let corner = Point2::new(0.0, 0.0);
    /// let moved_corner = platform_transform * corner;
    /// assert_eq!(moved_corner, Point2::new(10.0, 0.0));
    /// ```
    ///
    /// ## Practical Example: Camera Panning
    ///
    /// ```
    /// # use nalgebra::{IsometryMatrix2, Point2};
    ///
    /// // Pan the camera 50 pixels right and 30 pixels down
    /// let camera_pan = IsometryMatrix2::translation(50.0, -30.0);
    ///
    /// // Transform world coordinates to camera view coordinates
    /// let world_point = Point2::new(100.0, 100.0);
    /// let view_point = camera_pan * world_point;
    /// assert_eq!(view_point, Point2::new(150.0, 70.0));
    /// ```
    ///
    /// # See Also
    /// - [`IsometryMatrix2::new`] - Create an isometry with both translation and rotation
    /// - [`IsometryMatrix2::rotation`] - Create an isometry with only rotation
    /// - [`IsometryMatrix2::identity`] - Create an identity transformation
    #[inline]
    pub fn translation(x: T, y: T) -> Self {
        Self::new(Vector2::new(x, y), T::zero())
    }

    /// Creates a new 2D isometry with only rotation (no translation).
    ///
    /// This creates a pure rotation transformation around the origin (0, 0). The rotation
    /// is applied counter-clockwise (when y-axis points up) by the specified angle in radians.
    /// No translation is performed - the origin stays at the origin.
    ///
    /// This is useful for rotating objects that are already positioned at the origin, or
    /// for creating rotation components that will be combined with other transformations.
    ///
    /// # Arguments
    /// - `angle` - The rotation angle in radians (positive = counter-clockwise)
    ///
    /// # Use Cases
    /// - Rotate objects positioned at the origin
    /// - Create rotation components for composite transformations
    /// - Implement spinning objects (combine with time)
    /// - Rotate coordinate frames without translation
    ///
    /// # Examples
    ///
    /// ## Basic Rotation
    ///
    /// ```
    /// # use std::f32;
    /// # use nalgebra::{IsometryMatrix2, Point2};
    ///
    /// // Rotate 90 degrees counter-clockwise around the origin
    /// let iso = IsometryMatrix2::rotation(f32::consts::FRAC_PI_2);
    ///
    /// // Point on the positive x-axis rotates to positive y-axis
    /// let pt = Point2::new(1.0, 0.0);
    /// let result = iso * pt;
    /// assert!((result.x - 0.0).abs() < 1e-6);
    /// assert!((result.y - 1.0).abs() < 1e-6);
    /// ```
    ///
    /// ## Practical Example: Spinning Object
    ///
    /// ```
    /// # use std::f32;
    /// # use nalgebra::{IsometryMatrix2, Point2};
    ///
    /// // Rotate a point on an object by 45 degrees (like a spinning wheel)
    /// let angle = f32::consts::FRAC_PI_4;  // 45 degrees
    /// let rotation = IsometryMatrix2::rotation(angle);
    ///
    /// // A point on the wheel's edge
    /// let wheel_point = Point2::new(5.0, 0.0);
    /// let rotated_point = rotation * wheel_point;
    ///
    /// // After 45 degree rotation
    /// let expected_x = 5.0 * angle.cos();
    /// let expected_y = 5.0 * angle.sin();
    /// assert!((rotated_point.x - expected_x).abs() < 1e-6);
    /// assert!((rotated_point.y - expected_y).abs() < 1e-6);
    /// ```
    ///
    /// ## Practical Example: Rotating a Direction Vector
    ///
    /// ```
    /// # use std::f32;
    /// # use nalgebra::{IsometryMatrix2, Vector2};
    ///
    /// // Rotate a forward direction by 30 degrees to get a new heading
    /// let initial_direction = Vector2::new(1.0, 0.0);
    /// let turn_angle = f32::consts::PI / 6.0;  // 30 degrees
    /// let rotation = IsometryMatrix2::rotation(turn_angle);
    ///
    /// let new_direction = rotation * initial_direction;
    /// assert!((new_direction.x - turn_angle.cos()).abs() < 1e-6);
    /// assert!((new_direction.y - turn_angle.sin()).abs() < 1e-6);
    /// ```
    ///
    /// ## Understanding Rotation Direction
    ///
    /// ```
    /// # use std::f32;
    /// # use nalgebra::{IsometryMatrix2, Point2};
    ///
    /// // Positive angles rotate counter-clockwise
    /// let ccw = IsometryMatrix2::rotation(f32::consts::FRAC_PI_2);  // +90°
    /// assert_eq!(ccw * Point2::new(1.0, 0.0), Point2::new(0.0, 1.0));
    ///
    /// // Negative angles rotate clockwise
    /// let cw = IsometryMatrix2::rotation(-f32::consts::FRAC_PI_2);  // -90°
    /// assert_eq!(cw * Point2::new(1.0, 0.0), Point2::new(0.0, -1.0));
    /// ```
    ///
    /// # See Also
    /// - [`IsometryMatrix2::new`] - Create an isometry with both translation and rotation
    /// - [`IsometryMatrix2::translation`] - Create an isometry with only translation
    /// - [`IsometryMatrix2::rotation_wrt_point`] - Rotate around a point other than origin
    #[inline]
    pub fn rotation(angle: T) -> Self {
        Self::new(Vector2::zeros(), angle)
    }

    /// Converts the isometry to use a different numeric type.
    ///
    /// This method allows you to convert between different numeric types (e.g., from `f64`
    /// to `f32`, or between custom numeric types). This is useful when interfacing with
    /// libraries that require specific numeric types, or when optimizing for memory/performance.
    ///
    /// # Type Parameters
    /// - `To` - The target numeric type to convert to
    ///
    /// # Use Cases
    /// - Convert between single and double precision for performance/accuracy tradeoffs
    /// - Interface with graphics APIs that require specific float types
    /// - Convert to custom numeric types for specialized computations
    ///
    /// # Examples
    ///
    /// ## Basic Type Conversion
    ///
    /// ```
    /// # use nalgebra::IsometryMatrix2;
    ///
    /// // Create an isometry with f64 precision
    /// let iso = IsometryMatrix2::<f64>::identity();
    ///
    /// // Convert to f32 for a graphics API
    /// let iso2 = iso.cast::<f32>();
    /// assert_eq!(iso2, IsometryMatrix2::<f32>::identity());
    /// ```
    ///
    /// ## Practical Example: Converting for Graphics API
    ///
    /// ```
    /// # use std::f64;
    /// # use nalgebra::{IsometryMatrix2, Vector2};
    ///
    /// // Physics simulation uses f64 for accuracy
    /// let physics_transform = IsometryMatrix2::<f64>::new(
    ///     Vector2::new(100.0, 200.0),
    ///     f64::consts::FRAC_PI_4
    /// );
    ///
    /// // Graphics rendering uses f32
    /// let render_transform = physics_transform.cast::<f32>();
    ///
    /// // Both represent the same transformation
    /// assert_eq!(render_transform.translation.x, 100.0f32);
    /// assert_eq!(render_transform.translation.y, 200.0f32);
    /// ```
    ///
    /// # See Also
    /// - [`Isometry2::cast`] - Cast for Isometry2 (with UnitComplex rotation)
    /// - [`IsometryMatrix3::cast`] - Cast for 3D isometry matrices
    pub fn cast<To: Scalar>(self) -> IsometryMatrix2<To>
    where
        IsometryMatrix2<To>: SupersetOf<Self>,
    {
        crate::convert(self)
    }
}

impl<T: SimdRealField> Isometry2<T>
where
    T::Element: SimdRealField,
{
    /// Creates a new 2D isometry from a translation and a rotation angle.
    ///
    /// An **isometry** is a rigid body transformation that combines translation (moving) and
    /// rotation (spinning) while preserving distances and angles. This is the fundamental
    /// transformation for 2D games, robotics, and physics simulations.
    ///
    /// The rotational part is represented internally as a **unit complex number**, which is
    /// a compact and efficient way to store 2D rotations (similar to quaternions in 3D).
    /// This representation is more memory-efficient than the 2x2 matrix used by `IsometryMatrix2`.
    ///
    /// # Arguments
    /// - `translation` - A 2D vector specifying how far to move in x and y directions
    /// - `angle` - The rotation angle in radians (counter-clockwise, positive)
    ///
    /// # Use Cases
    /// - Position and orient 2D game objects (sprites, characters, obstacles)
    /// - Represent robot poses in 2D navigation
    /// - Transform coordinate frames in 2D physics
    /// - Camera transformations in 2D games
    ///
    /// # Examples
    ///
    /// ## Basic Usage
    ///
    /// ```
    /// # use std::f32;
    /// # use nalgebra::{Isometry2, Point2, Vector2};
    ///
    /// // Move 1 unit right, 2 units up, and rotate 90 degrees counter-clockwise
    /// let iso = Isometry2::new(Vector2::new(1.0, 2.0), f32::consts::FRAC_PI_2);
    ///
    /// // Transform a point: first rotate, then translate
    /// let pt = Point2::new(3.0, 4.0);
    /// let result = iso * pt;
    /// assert_eq!(result, Point2::new(-3.0, 5.0));
    /// ```
    ///
    /// ## Practical Example: 2D Spaceship
    ///
    /// ```
    /// # use std::f32;
    /// # use nalgebra::{Isometry2, Point2, Vector2};
    ///
    /// // Create a spaceship at position (50, 100), facing up (90 degrees)
    /// let spaceship_pos = Vector2::new(50.0, 100.0);
    /// let spaceship_angle = f32::consts::FRAC_PI_2;  // 90° = facing up
    /// let spaceship_transform = Isometry2::new(spaceship_pos, spaceship_angle);
    ///
    /// // The spaceship's nose is at local position (2, 0) in its own coordinate system
    /// let local_nose = Point2::new(2.0, 0.0);
    /// let world_nose = spaceship_transform * local_nose;
    ///
    /// // After transformation, the nose points up in world space
    /// assert!((world_nose.x - 50.0).abs() < 1e-5);
    /// assert!((world_nose.y - 102.0).abs() < 1e-5);
    /// ```
    ///
    /// ## Practical Example: Top-Down Vehicle
    ///
    /// ```
    /// # use std::f32;
    /// # use nalgebra::{Isometry2, Point2, Vector2};
    ///
    /// // Car at position (10, 20), heading 45 degrees from east
    /// let car_transform = Isometry2::new(
    ///     Vector2::new(10.0, 20.0),
    ///     f32::consts::FRAC_PI_4  // 45 degrees
    /// );
    ///
    /// // Front-left wheel in car's local coordinates
    /// let local_wheel = Point2::new(1.5, 0.8);
    /// let world_wheel = car_transform * local_wheel;
    ///
    /// // Wheel is now properly positioned and oriented in world space
    /// ```
    ///
    /// ## Comparing with IsometryMatrix2
    ///
    /// ```
    /// # use std::f32;
    /// # use nalgebra::{Isometry2, IsometryMatrix2, Point2, Vector2};
    ///
    /// let translation = Vector2::new(5.0, 10.0);
    /// let angle = f32::consts::FRAC_PI_4;
    ///
    /// // Isometry2 uses unit complex (more compact)
    /// let iso_complex = Isometry2::new(translation, angle);
    ///
    /// // IsometryMatrix2 uses 2x2 matrix (easier for some operations)
    /// let iso_matrix = IsometryMatrix2::new(translation, angle);
    ///
    /// // Both produce the same transformations
    /// let pt = Point2::new(1.0, 0.0);
    /// let result1 = iso_complex * pt;
    /// let result2 = iso_matrix * pt;
    /// assert!((result1.x - result2.x).abs() < 1e-6);
    /// assert!((result1.y - result2.y).abs() < 1e-6);
    /// ```
    ///
    /// # See Also
    /// - [`Isometry2::translation`] - Create an isometry with only translation
    /// - [`Isometry2::rotation`] - Create an isometry with only rotation
    /// - [`Isometry2::identity`] - Create an identity isometry (no transformation)
    /// - [`IsometryMatrix2::new`] - Similar but uses matrix representation for rotation
    #[inline]
    pub fn new(translation: Vector2<T>, angle: T) -> Self {
        Self::from_parts(
            Translation::from(translation),
            UnitComplex::from_angle(angle),
        )
    }

    /// Creates a new 2D isometry with only translation (no rotation).
    ///
    /// This creates a pure translation transformation - it moves points by the specified
    /// amounts in the x and y directions without any rotation. The resulting isometry
    /// behaves like shifting everything on a 2D plane.
    ///
    /// This is useful when you need to move objects without changing their orientation,
    /// such as sliding a platform, moving a character in a fixed direction, or
    /// translating a camera view.
    ///
    /// # Arguments
    /// - `x` - Distance to move along the x-axis
    /// - `y` - Distance to move along the y-axis
    ///
    /// # Use Cases
    /// - Move objects without rotating them
    /// - Implement smooth linear movement in games
    /// - Offset coordinate systems
    /// - Camera panning without rotation
    ///
    /// # Examples
    ///
    /// ## Basic Translation
    ///
    /// ```
    /// # use nalgebra::{Isometry2, Point2};
    ///
    /// // Create a translation that moves everything 3 units right and 4 units up
    /// let iso = Isometry2::translation(3.0, 4.0);
    ///
    /// // Transform a point
    /// let pt = Point2::new(1.0, 2.0);
    /// let result = iso * pt;
    /// assert_eq!(result, Point2::new(4.0, 6.0));
    /// ```
    ///
    /// ## Practical Example: Scrolling Background
    ///
    /// ```
    /// # use nalgebra::{Isometry2, Point2};
    ///
    /// // Scroll a parallax background layer horizontally
    /// let scroll_offset = -25.0;  // Negative = scrolling left
    /// let background_transform = Isometry2::translation(scroll_offset, 0.0);
    ///
    /// // Transform a tile position
    /// let tile_pos = Point2::new(100.0, 50.0);
    /// let screen_pos = background_transform * tile_pos;
    /// assert_eq!(screen_pos, Point2::new(75.0, 50.0));
    /// ```
    ///
    /// # See Also
    /// - [`Isometry2::new`] - Create an isometry with both translation and rotation
    /// - [`Isometry2::rotation`] - Create an isometry with only rotation
    /// - [`Isometry2::identity`] - Create an identity transformation
    #[inline]
    pub fn translation(x: T, y: T) -> Self {
        Self::from_parts(Translation2::new(x, y), UnitComplex::identity())
    }

    /// Creates a new 2D isometry with only rotation (no translation).
    ///
    /// This creates a pure rotation transformation around the origin (0, 0). The rotation
    /// is applied counter-clockwise (when y-axis points up) by the specified angle in radians.
    /// No translation is performed - the origin stays at the origin.
    ///
    /// This is useful for rotating objects that are already positioned at the origin, or
    /// for creating rotation components that will be combined with other transformations.
    ///
    /// # Arguments
    /// - `angle` - The rotation angle in radians (positive = counter-clockwise)
    ///
    /// # Use Cases
    /// - Rotate objects positioned at the origin
    /// - Create rotation components for composite transformations
    /// - Implement spinning objects (combine with time)
    /// - Rotate coordinate frames without translation
    ///
    /// # Examples
    ///
    /// ## Basic Rotation
    ///
    /// ```
    /// # use std::f32;
    /// # use nalgebra::{Isometry2, Point2};
    ///
    /// // Rotate 90 degrees counter-clockwise around the origin
    /// let iso = Isometry2::rotation(f32::consts::FRAC_PI_2);
    ///
    /// // Point on the positive x-axis rotates to positive y-axis
    /// let pt = Point2::new(1.0, 0.0);
    /// let result = iso * pt;
    /// assert!((result.x - 0.0).abs() < 1e-6);
    /// assert!((result.y - 1.0).abs() < 1e-6);
    /// ```
    ///
    /// ## Practical Example: Rotating a Turret
    ///
    /// ```
    /// # use std::f32;
    /// # use nalgebra::{Isometry2, Vector2};
    ///
    /// // Turret aiming direction, rotating 30 degrees from initial heading
    /// let turret_rotation = Isometry2::rotation(f32::consts::PI / 6.0);
    ///
    /// // Initial forward direction
    /// let forward = Vector2::new(1.0, 0.0);
    /// let aim_direction = turret_rotation * forward;
    ///
    /// // Now pointing 30 degrees counter-clockwise from right
    /// assert!((aim_direction.x - (f32::consts::PI / 6.0).cos()).abs() < 1e-6);
    /// ```
    ///
    /// # See Also
    /// - [`Isometry2::new`] - Create an isometry with both translation and rotation
    /// - [`Isometry2::translation`] - Create an isometry with only translation
    /// - [`Isometry2::rotation_wrt_point`] - Rotate around a point other than origin
    #[inline]
    pub fn rotation(angle: T) -> Self {
        Self::new(Vector2::zeros(), angle)
    }

    /// Converts the isometry to use a different numeric type.
    ///
    /// This method allows you to convert between different numeric types (e.g., from `f64`
    /// to `f32`, or between custom numeric types). This is useful when interfacing with
    /// libraries that require specific numeric types, or when optimizing for memory/performance.
    ///
    /// # Type Parameters
    /// - `To` - The target numeric type to convert to
    ///
    /// # Use Cases
    /// - Convert between single and double precision for performance/accuracy tradeoffs
    /// - Interface with graphics APIs that require specific float types
    /// - Convert to custom numeric types for specialized computations
    ///
    /// # Examples
    ///
    /// ## Basic Type Conversion
    ///
    /// ```
    /// # use nalgebra::Isometry2;
    ///
    /// // Create an isometry with f64 precision
    /// let iso = Isometry2::<f64>::identity();
    ///
    /// // Convert to f32 for a graphics API
    /// let iso2 = iso.cast::<f32>();
    /// assert_eq!(iso2, Isometry2::<f32>::identity());
    /// ```
    ///
    /// ## Practical Example: Physics to Rendering Pipeline
    ///
    /// ```
    /// # use std::f64;
    /// # use nalgebra::{Isometry2, Vector2};
    ///
    /// // Physics simulation uses f64 for accuracy
    /// let physics_transform = Isometry2::<f64>::new(
    ///     Vector2::new(123.456789, 987.654321),
    ///     f64::consts::FRAC_PI_3
    /// );
    ///
    /// // Graphics rendering uses f32 for efficiency
    /// let render_transform = physics_transform.cast::<f32>();
    ///
    /// // Values are converted (with some precision loss)
    /// assert!((render_transform.translation.x - 123.456789f32).abs() < 1e-5);
    /// ```
    ///
    /// # See Also
    /// - [`IsometryMatrix2::cast`] - Cast for IsometryMatrix2 (with matrix rotation)
    /// - [`Isometry3::cast`] - Cast for 3D isometries
    pub fn cast<To: Scalar>(self) -> Isometry2<To>
    where
        Isometry2<To>: SupersetOf<Self>,
    {
        crate::convert(self)
    }
}

// 3D rotation.
macro_rules! basic_isometry_construction_impl(
    ($RotId: ident < $($RotParams: ident),*>) => {
        /// Creates a new 3D isometry from a translation and a rotation.
        ///
        /// An **isometry** is a rigid body transformation that preserves distances and angles.
        /// In 3D, this combines translation (moving in x, y, z) and rotation (orientation in 3D space).
        /// Isometries are fundamental for representing the position and orientation of objects in
        /// 3D games, robotics, physics simulations, and computer graphics.
        ///
        /// The rotation is specified using an **axis-angle** representation: a 3D vector whose
        /// direction indicates the rotation axis and whose magnitude indicates the rotation angle
        /// in radians. For example, `Vector3::new(0.0, 3.14, 0.0)` means "rotate 3.14 radians
        /// (180 degrees) around the y-axis".
        ///
        /// # Arguments
        /// - `translation` - A 3D vector specifying how far to move in x, y, and z directions
        /// - `axisangle` - A 3D vector representing rotation: direction = axis, magnitude = angle in radians
        ///
        /// # Use Cases
        /// - Position and orient 3D game objects (characters, props, cameras)
        /// - Represent robot poses in 3D space (position + orientation)
        /// - Transform coordinate frames in physics simulations
        /// - Camera transformations in 3D graphics
        /// - Rigid body dynamics in physics engines
        ///
        /// # Examples
        ///
        /// ## Basic Usage
        ///
        /// ```
        /// # #[macro_use] extern crate approx;
        /// # use std::f32;
        /// # use nalgebra::{Isometry3, IsometryMatrix3, Point3, Vector3};
        ///
        /// // Rotate 90° around y-axis, then translate by (1, 2, 3)
        /// let axisangle = Vector3::y() * f32::consts::FRAC_PI_2;
        /// let translation = Vector3::new(1.0, 2.0, 3.0);
        ///
        /// // Point and vector to transform
        /// let pt = Point3::new(4.0, 5.0, 6.0);
        /// let vec = Vector3::new(4.0, 5.0, 6.0);
        ///
        /// // Isometry3 uses UnitQuaternion for rotation (compact, efficient)
        /// let iso = Isometry3::new(translation, axisangle);
        /// assert_relative_eq!(iso * pt, Point3::new(7.0, 7.0, -1.0), epsilon = 1.0e-6);
        /// assert_relative_eq!(iso * vec, Vector3::new(6.0, 5.0, -4.0), epsilon = 1.0e-6);
        ///
        /// // IsometryMatrix3 uses Rotation3 (3x3 matrix) for rotation
        /// let iso = IsometryMatrix3::new(translation, axisangle);
        /// assert_relative_eq!(iso * pt, Point3::new(7.0, 7.0, -1.0), epsilon = 1.0e-6);
        /// assert_relative_eq!(iso * vec, Vector3::new(6.0, 5.0, -4.0), epsilon = 1.0e-6);
        /// ```
        ///
        /// ## Practical Example: 3D Game Character
        ///
        /// ```
        /// # use std::f32;
        /// # use nalgebra::{Isometry3, Point3, Vector3};
        ///
        /// // Character at position (10, 0, 5), facing 45° from forward direction
        /// let character_pos = Vector3::new(10.0, 0.0, 5.0);
        /// let facing_angle = f32::consts::FRAC_PI_4;  // 45 degrees
        /// let rotation_axis = Vector3::y();  // Rotate around vertical axis
        /// let character_transform = Isometry3::new(
        ///     character_pos,
        ///     rotation_axis * facing_angle
        /// );
        ///
        /// // Transform a point on the character's sword (in local coordinates)
        /// let sword_tip_local = Point3::new(2.0, 1.0, 0.0);
        /// let sword_tip_world = character_transform * sword_tip_local;
        ///
        /// // The sword tip is now in world coordinates
        /// ```
        ///
        /// ## Practical Example: Robot Arm Joint
        ///
        /// ```
        /// # use std::f32;
        /// # use nalgebra::{Isometry3, Vector3};
        ///
        /// // Robot arm segment: 5 units long, rotated 30° around z-axis
        /// let segment_length = Vector3::new(5.0, 0.0, 0.0);
        /// let joint_angle = f32::consts::PI / 6.0;  // 30 degrees
        /// let joint_axis = Vector3::z();
        ///
        /// let arm_segment = Isometry3::new(
        ///     segment_length,
        ///     joint_axis * joint_angle
        /// );
        /// ```
        ///
        /// ## Understanding Axis-Angle Rotation
        ///
        /// ```
        /// # use std::f32;
        /// # use nalgebra::{Isometry3, Point3, Vector3};
        ///
        /// // Rotate 180° around the x-axis
        /// let rotation = Vector3::x() * f32::consts::PI;
        /// let iso = Isometry3::new(Vector3::zeros(), rotation);
        ///
        /// // Point above the x-axis gets flipped below
        /// let pt = Point3::new(1.0, 1.0, 0.0);
        /// let result = iso * pt;
        /// assert!((result.x - 1.0).abs() < 1e-6);
        /// assert!((result.y + 1.0).abs() < 1e-6);  // Flipped
        /// assert!((result.z - 0.0).abs() < 1e-6);
        /// ```
        ///
        /// ## Practical Example: Camera Looking at Scene
        ///
        /// ```
        /// # use std::f32;
        /// # use nalgebra::{Isometry3, Vector3};
        ///
        /// // Position camera 10 units back from origin, tilted down 15°
        /// let camera_position = Vector3::new(0.0, 2.0, 10.0);
        /// let tilt_angle = -f32::consts::PI / 12.0;  // -15° (looking down)
        /// let tilt_axis = Vector3::x();
        ///
        /// let camera_transform = Isometry3::new(
        ///     camera_position,
        ///     tilt_axis * tilt_angle
        /// );
        /// ```
        ///
        /// # See Also
        /// - [`Isometry3::translation`] - Create an isometry with only translation
        /// - [`Isometry3::rotation`] - Create an isometry with only rotation
        /// - [`Isometry3::identity`] - Create an identity isometry (no transformation)
        /// - [`Isometry3::face_towards`] - Create an isometry that faces a target point
        /// - [`UnitQuaternion::from_axis_angle`] - Alternative way to create rotations
        #[inline]
        pub fn new(translation: Vector3<T>, axisangle: Vector3<T>) -> Self {
            Self::from_parts(
                Translation::from(translation),
                $RotId::<$($RotParams),*>::from_scaled_axis(axisangle))
        }

        /// Creates a new 3D isometry with only translation (no rotation).
        ///
        /// This creates a pure translation transformation - it moves points by the specified
        /// amounts in the x, y, and z directions without any rotation. The resulting isometry
        /// preserves orientation while changing position.
        ///
        /// This is useful when you need to move 3D objects without changing their orientation,
        /// such as moving a camera along a track, translating a physics object, or offsetting
        /// coordinate systems.
        ///
        /// # Arguments
        /// - `x` - Distance to move along the x-axis
        /// - `y` - Distance to move along the y-axis
        /// - `z` - Distance to move along the z-axis
        ///
        /// # Use Cases
        /// - Move 3D objects without rotating them
        /// - Implement linear motion in physics simulations
        /// - Camera dolly movements (forward/back, left/right, up/down)
        /// - Offset coordinate frames while preserving orientation
        ///
        /// # Examples
        ///
        /// ## Basic Translation
        ///
        /// ```
        /// # use nalgebra::{Isometry3, Point3};
        ///
        /// // Move 3 units right, 4 units up, 5 units forward
        /// let iso = Isometry3::translation(3.0, 4.0, 5.0);
        ///
        /// // Transform a point
        /// let pt = Point3::new(1.0, 2.0, 3.0);
        /// let result = iso * pt;
        /// assert_eq!(result, Point3::new(4.0, 6.0, 8.0));
        /// ```
        ///
        /// ## Practical Example: Moving Platform
        ///
        /// ```
        /// # use nalgebra::{Isometry3, Point3};
        ///
        /// // Platform moving upward at y = 10
        /// let platform_transform = Isometry3::translation(0.0, 10.0, 0.0);
        ///
        /// // Object standing on the platform (at platform's origin)
        /// let object_pos = Point3::new(2.0, 0.0, 3.0);
        /// let world_pos = platform_transform * object_pos;
        ///
        /// // Object moved up with the platform
        /// assert_eq!(world_pos, Point3::new(2.0, 10.0, 3.0));
        /// ```
        ///
        /// ## Practical Example: Camera Dolly Movement
        ///
        /// ```
        /// # use nalgebra::{Isometry3, Point3};
        ///
        /// // Camera moves 5 units back (positive z) without rotating
        /// let camera_dolly = Isometry3::translation(0.0, 0.0, 5.0);
        ///
        /// // Transform a point in camera space to world space
        /// let camera_space = Point3::new(1.0, 1.0, 0.0);
        /// let world_space = camera_dolly * camera_space;
        /// assert_eq!(world_space, Point3::new(1.0, 1.0, 5.0));
        /// ```
        ///
        /// # See Also
        /// - [`Isometry3::new`] - Create an isometry with both translation and rotation
        /// - [`Isometry3::rotation`] - Create an isometry with only rotation
        /// - [`Isometry3::identity`] - Create an identity transformation
        #[inline]
        pub fn translation(x: T, y: T, z: T) -> Self {
            Self::from_parts(Translation3::new(x, y, z), $RotId::identity())
        }

        /// Creates a new 3D isometry with only rotation (no translation).
        ///
        /// This creates a pure rotation transformation around the origin. The rotation is
        /// specified using an **axis-angle** representation: a 3D vector whose direction
        /// indicates the rotation axis and whose magnitude indicates the rotation angle in radians.
        ///
        /// This is useful for rotating objects positioned at the origin, creating rotation
        /// components for composite transformations, or rotating coordinate frames without
        /// translating them.
        ///
        /// # Arguments
        /// - `axisangle` - A 3D vector: direction = rotation axis, magnitude = angle in radians
        ///
        /// # Use Cases
        /// - Rotate objects positioned at the origin
        /// - Create rotation components for composite transformations
        /// - Implement spinning/rotating objects (combine with time)
        /// - Rotate coordinate frames without translation
        /// - Represent pure orientation changes
        ///
        /// # Examples
        ///
        /// ## Basic Rotation
        ///
        /// ```
        /// # #[macro_use] extern crate approx;
        /// # use std::f32;
        /// # use nalgebra::{Isometry3, Point3, Vector3};
        ///
        /// // Rotate 90° around the y-axis (vertical)
        /// let rotation = Vector3::y() * f32::consts::FRAC_PI_2;
        /// let iso = Isometry3::rotation(rotation);
        ///
        /// // Point on positive x-axis rotates to negative z-axis
        /// let pt = Point3::new(1.0, 0.0, 0.0);
        /// let result = iso * pt;
        /// assert_relative_eq!(result, Point3::new(0.0, 0.0, -1.0), epsilon = 1e-6);
        /// ```
        ///
        /// ## Practical Example: Spinning Propeller
        ///
        /// ```
        /// # use std::f32;
        /// # use nalgebra::{Isometry3, Point3, Vector3};
        ///
        /// // Propeller blade rotating around z-axis
        /// let spin_angle = f32::consts::FRAC_PI_4;  // 45 degrees
        /// let spin_axis = Vector3::z();
        /// let propeller_rotation = Isometry3::rotation(spin_axis * spin_angle);
        ///
        /// // Transform a point on the blade tip
        /// let blade_tip = Point3::new(2.0, 0.0, 0.0);
        /// let rotated_tip = propeller_rotation * blade_tip;
        /// // Blade tip has rotated 45° around the hub
        /// ```
        ///
        /// ## Practical Example: Rotating Turret
        ///
        /// ```
        /// # use std::f32;
        /// # use nalgebra::{Isometry3, Vector3};
        ///
        /// // Turret rotating horizontally (yaw)
        /// let turret_angle = f32::consts::PI / 3.0;  // 60 degrees
        /// let yaw_axis = Vector3::y();
        /// let turret_rotation = Isometry3::rotation(yaw_axis * turret_angle);
        ///
        /// // Barrel direction after rotation
        /// let forward = Vector3::x();
        /// let barrel_direction = turret_rotation * forward;
        /// ```
        ///
        /// ## Understanding Axis-Angle
        ///
        /// ```
        /// # #[macro_use] extern crate approx;
        /// # use std::f32;
        /// # use nalgebra::{Isometry3, Point3, Vector3};
        ///
        /// // Rotate 180° around x-axis (flips y and z)
        /// let flip = Isometry3::rotation(Vector3::x() * f32::consts::PI);
        /// let pt = Point3::new(0.0, 1.0, 2.0);
        /// let result = flip * pt;
        /// assert_relative_eq!(result, Point3::new(0.0, -1.0, -2.0), epsilon = 1e-6);
        /// ```
        ///
        /// # See Also
        /// - [`Isometry3::new`] - Create an isometry with both translation and rotation
        /// - [`Isometry3::translation`] - Create an isometry with only translation
        /// - [`Isometry3::rotation_wrt_point`] - Rotate around a point other than origin
        /// - [`Isometry3::face_towards`] - Create rotation facing a specific direction
        #[inline]
        pub fn rotation(axisangle: Vector3<T>) -> Self {
            Self::new(Vector3::zeros(), axisangle)
        }
    }
);

macro_rules! look_at_isometry_construction_impl(
    ($RotId: ident < $($RotParams: ident),*>) => {
        /// Creates an isometry representing an observer at `eye` looking toward `target`.
        ///
        /// This creates a **local coordinate frame** for an observer (like a camera or character)
        /// positioned at the `eye` point and looking toward the `target` point. The resulting
        /// isometry transforms points from the observer's local space to world space.
        ///
        /// In the local frame:
        /// - The origin (0, 0, 0) is at the `eye` position
        /// - The positive z-axis points toward the `target`
        /// - The y-axis approximately aligns with the `up` direction
        /// - The x-axis is perpendicular to both (forming a right-handed coordinate system)
        ///
        /// This is commonly used for:
        /// - Setting up coordinate frames for 3D objects that should face a target
        /// - Creating "billboard" effects where objects always face the camera
        /// - Setting up character orientation to face a point of interest
        ///
        /// **Note**: This is NOT a view matrix. For camera view matrices, use [`look_at_rh`] or
        /// [`look_at_lh`] instead.
        ///
        /// # Arguments
        /// - `eye` - The observer's position in world space
        /// - `target` - The point to look toward in world space
        /// - `up` - Approximate "up" direction (must not be parallel to view direction)
        ///
        /// # Use Cases
        /// - Objects that need to face toward a point (billboards, sprites, enemies)
        /// - Character orientation toward a target
        /// - Setting up local coordinate frames for objects
        /// - Aligning objects with a specific direction
        ///
        /// # Examples
        ///
        /// ## Basic Usage
        ///
        /// ```
        /// # #[macro_use] extern crate approx;
        /// # use nalgebra::{Isometry3, IsometryMatrix3, Point3, Vector3};
        ///
        /// let eye = Point3::new(1.0, 2.0, 3.0);
        /// let target = Point3::new(2.0, 2.0, 3.0);
        /// let up = Vector3::y();
        ///
        /// // Create isometry (works with both Isometry3 and IsometryMatrix3)
        /// let iso = Isometry3::face_towards(&eye, &target, &up);
        ///
        /// // The origin in local space maps to eye position in world space
        /// assert_eq!(iso * Point3::origin(), eye);
        ///
        /// // The local z-axis points toward the target
        /// assert_relative_eq!(iso * Vector3::z(), Vector3::x());
        /// ```
        ///
        /// ## Practical Example: Enemy Facing Player
        ///
        /// ```
        /// # use nalgebra::{Isometry3, Point3, Vector3};
        ///
        /// // Enemy position and player position
        /// let enemy_pos = Point3::new(10.0, 0.0, 5.0);
        /// let player_pos = Point3::new(15.0, 0.0, 8.0);
        /// let up = Vector3::y();
        ///
        /// // Create enemy transformation facing the player
        /// let enemy_transform = Isometry3::face_towards(&enemy_pos, &player_pos, &up);
        ///
        /// // Now the enemy's local forward direction (z-axis) points toward player
        /// // Transform enemy's weapon position from local to world space
        /// let weapon_offset = Point3::new(0.0, 0.5, 1.0);  // In front of enemy
        /// let weapon_world_pos = enemy_transform * weapon_offset;
        /// ```
        ///
        /// ## Practical Example: Billboard Sprite
        ///
        /// ```
        /// # use nalgebra::{Isometry3, Point3, Vector3};
        ///
        /// // Billboard position and camera position
        /// let billboard_pos = Point3::new(0.0, 2.0, 0.0);
        /// let camera_pos = Point3::new(5.0, 3.0, 5.0);
        /// let up = Vector3::y();
        ///
        /// // Make billboard face the camera
        /// let billboard_transform = Isometry3::face_towards(
        ///     &billboard_pos,
        ///     &camera_pos,
        ///     &up
        /// );
        ///
        /// // Transform billboard corners from local to world space
        /// let corner = Point3::new(0.5, 0.5, 0.0);
        /// let world_corner = billboard_transform * corner;
        /// ```
        ///
        /// ## Practical Example: Turret Tracking Target
        ///
        /// ```
        /// # use nalgebra::{Isometry3, Point3, Vector3};
        ///
        /// // Turret on the ground tracking an aerial target
        /// let turret_pos = Point3::new(0.0, 0.0, 0.0);
        /// let target_aircraft = Point3::new(10.0, 15.0, 8.0);
        /// let up = Vector3::y();
        ///
        /// // Orient turret to face aircraft
        /// let turret_orientation = Isometry3::face_towards(
        ///     &turret_pos,
        ///     &target_aircraft,
        ///     &up
        /// );
        ///
        /// // Barrel direction in world space (local z-axis)
        /// let barrel_direction = turret_orientation * Vector3::z();
        /// ```
        ///
        /// # See Also
        /// - [`Isometry3::look_at_rh`] - Create a right-handed view matrix (for cameras)
        /// - [`Isometry3::look_at_lh`] - Create a left-handed view matrix (for cameras)
        /// - [`Isometry3::new`] - Create an isometry from explicit translation and rotation
        #[inline]
        pub fn face_towards(eye:    &Point3<T>,
                            target: &Point3<T>,
                            up:     &Vector3<T>)
                            -> Self {
            Self::from_parts(
                Translation::from(eye.coords.clone()),
                $RotId::face_towards(&(target - eye), up))
        }

        /// Deprecated: Use [`Isometry::face_towards`] instead.
        #[deprecated(note="renamed to `face_towards`")]
        pub fn new_observer_frame(eye:    &Point3<T>,
                                  target: &Point3<T>,
                                  up:     &Vector3<T>)
                                  -> Self {
            Self::face_towards(eye, target, up)
        }

        /// Creates a right-handed look-at view matrix (for cameras).
        ///
        /// This creates a **view matrix** for a right-handed camera system - the standard
        /// convention in OpenGL and most 3D graphics. The camera is positioned at `eye`,
        /// looks toward `target`, and the resulting transformation converts world coordinates
        /// to camera/view coordinates.
        ///
        /// In the resulting view space:
        /// - The camera position (`eye`) maps to the origin
        /// - The view direction `target - eye` maps to the **negative z-axis** (-z)
        /// - The camera looks "into" the screen along its local -z direction
        /// - The y-axis approximately aligns with the `up` direction
        /// - The x-axis points to the right (forming a right-handed system)
        ///
        /// This is the inverse of a "model" transformation - it transforms FROM world space
        /// TO view/camera space, which is what you need for rendering.
        ///
        /// # Arguments
        /// - `eye` - The camera's position in world space
        /// - `target` - The point the camera looks at in world space
        /// - `up` - Approximate "up" direction (usually world up vector like `Vector3::y()`)
        ///
        /// # Use Cases
        /// - Creating camera view matrices for OpenGL-style rendering
        /// - First-person camera systems
        /// - Third-person camera following a character
        /// - Cinematic camera movements
        ///
        /// # Examples
        ///
        /// ## Basic Usage
        ///
        /// ```
        /// # #[macro_use] extern crate approx;
        /// # use nalgebra::{Isometry3, IsometryMatrix3, Point3, Vector3};
        ///
        /// let eye = Point3::new(1.0, 2.0, 3.0);
        /// let target = Point3::new(2.0, 2.0, 3.0);
        /// let up = Vector3::y();
        ///
        /// // Create view matrix (works with both Isometry3 and IsometryMatrix3)
        /// let view = Isometry3::look_at_rh(&eye, &target, &up);
        ///
        /// // Camera position in world maps to origin in view space
        /// assert_eq!(view * eye, Point3::origin());
        ///
        /// // World x-axis direction maps to negative z in view space
        /// assert_relative_eq!(view * Vector3::x(), -Vector3::z());
        /// ```
        ///
        /// ## Practical Example: First-Person Camera
        ///
        /// ```
        /// # use nalgebra::{Isometry3, Point3, Vector3};
        ///
        /// // Player standing at (10, 1.7, 10), looking north (toward negative z)
        /// let player_pos = Point3::new(10.0, 1.7, 10.0);
        /// let look_target = Point3::new(10.0, 1.7, 5.0);  // 5 units ahead
        /// let up = Vector3::y();
        ///
        /// // Create first-person camera view matrix
        /// let view_matrix = Isometry3::look_at_rh(&player_pos, &look_target, &up);
        ///
        /// // Transform world objects to view space for rendering
        /// let world_object = Point3::new(10.0, 2.0, 5.0);  // Object in front of player
        /// let view_space_pos = view_matrix * world_object;
        /// ```
        ///
        /// ## Practical Example: Third-Person Camera
        ///
        /// ```
        /// # use nalgebra::{Isometry3, Point3, Vector3};
        ///
        /// // Character position
        /// let character_pos = Point3::new(0.0, 0.0, 0.0);
        ///
        /// // Camera behind and above the character
        /// let camera_offset = Vector3::new(0.0, 3.0, 5.0);
        /// let camera_pos = character_pos + camera_offset;
        ///
        /// // Camera looks at the character
        /// let view_matrix = Isometry3::look_at_rh(
        ///     &camera_pos.into(),
        ///     &character_pos,
        ///     &Vector3::y()
        /// );
        ///
        /// // Use view_matrix for rendering the scene
        /// ```
        ///
        /// ## Practical Example: Orbiting Camera
        ///
        /// ```
        /// # use std::f32;
        /// # use nalgebra::{Isometry3, Point3, Vector3};
        ///
        /// // Camera orbits around origin at distance 10
        /// let angle = f32::consts::FRAC_PI_4;  // 45 degrees
        /// let distance = 10.0;
        /// let camera_pos = Point3::new(
        ///     distance * angle.cos(),
        ///     5.0,
        ///     distance * angle.sin()
        /// );
        ///
        /// // Always look at origin
        /// let view_matrix = Isometry3::look_at_rh(
        ///     &camera_pos,
        ///     &Point3::origin(),
        ///     &Vector3::y()
        /// );
        /// ```
        ///
        /// ## Understanding View Space Transformation
        ///
        /// ```
        /// # use nalgebra::{Isometry3, Point3, Vector3};
        ///
        /// // Camera at origin looking along positive x-axis
        /// let camera_pos = Point3::origin();
        /// let target = Point3::new(1.0, 0.0, 0.0);
        /// let view = Isometry3::look_at_rh(&camera_pos, &target, &Vector3::y());
        ///
        /// // In view space, things in front of camera have negative z
        /// let point_ahead = Point3::new(5.0, 0.0, 0.0);
        /// let view_space = view * point_ahead;
        /// assert!(view_space.z < 0.0);  // Negative z = in front of camera
        /// ```
        ///
        /// # See Also
        /// - [`Isometry3::look_at_lh`] - Left-handed look-at (for DirectX-style systems)
        /// - [`Isometry3::face_towards`] - Create object-to-world transform (not a view matrix)
        /// - [`Perspective3::new`] - Create perspective projection (often combined with view matrix)
        #[inline]
        pub fn look_at_rh(eye:    &Point3<T>,
                          target: &Point3<T>,
                          up:     &Vector3<T>)
                          -> Self {
            let rotation = $RotId::look_at_rh(&(target - eye), up);
            let trans    = &rotation * (-eye);

            Self::from_parts(Translation::from(trans.coords), rotation)
        }

        /// Creates a left-handed look-at view matrix (for cameras).
        ///
        /// This creates a **view matrix** for a left-handed camera system - the convention
        /// used in DirectX and some other 3D systems. The camera is positioned at `eye`,
        /// looks toward `target`, and the resulting transformation converts world coordinates
        /// to camera/view coordinates.
        ///
        /// In the resulting view space:
        /// - The camera position (`eye`) maps to the origin
        /// - The view direction `target - eye` maps to the **positive z-axis** (+z)
        /// - The camera looks "into" the screen along its local +z direction
        /// - The y-axis approximately aligns with the `up` direction
        /// - The x-axis points to the right (forming a left-handed system)
        ///
        /// The key difference from [`look_at_rh`] is the z-axis direction: in left-handed
        /// systems, positive z goes "into" the screen (forward), while in right-handed systems,
        /// negative z goes into the screen.
        ///
        /// # Arguments
        /// - `eye` - The camera's position in world space
        /// - `target` - The point the camera looks at in world space
        /// - `up` - Approximate "up" direction (usually world up vector like `Vector3::y()`)
        ///
        /// # Use Cases
        /// - Creating camera view matrices for DirectX-style rendering
        /// - Left-handed coordinate system engines
        /// - Systems where positive z points forward
        ///
        /// # Examples
        ///
        /// ## Basic Usage
        ///
        /// ```
        /// # #[macro_use] extern crate approx;
        /// # use nalgebra::{Isometry3, IsometryMatrix3, Point3, Vector3};
        ///
        /// let eye = Point3::new(1.0, 2.0, 3.0);
        /// let target = Point3::new(2.0, 2.0, 3.0);
        /// let up = Vector3::y();
        ///
        /// // Create left-handed view matrix
        /// let view = Isometry3::look_at_lh(&eye, &target, &up);
        ///
        /// // Camera position in world maps to origin in view space
        /// assert_eq!(view * eye, Point3::origin());
        ///
        /// // World x-axis direction maps to positive z in view space
        /// assert_relative_eq!(view * Vector3::x(), Vector3::z());
        /// ```
        ///
        /// ## Practical Example: DirectX-Style Camera
        ///
        /// ```
        /// # use nalgebra::{Isometry3, Point3, Vector3};
        ///
        /// // Camera for a DirectX application
        /// let camera_pos = Point3::new(0.0, 5.0, -10.0);
        /// let look_at = Point3::origin();  // Looking at world origin
        /// let up = Vector3::y();
        ///
        /// // Create left-handed view matrix
        /// let view_matrix = Isometry3::look_at_lh(&camera_pos, &look_at, &up);
        ///
        /// // Transform world points to view space for rendering
        /// let world_point = Point3::new(1.0, 0.0, 0.0);
        /// let view_space = view_matrix * world_point;
        /// ```
        ///
        /// ## Practical Example: Flight Simulator Camera
        ///
        /// ```
        /// # use nalgebra::{Isometry3, Point3, Vector3};
        ///
        /// // Aircraft cockpit camera
        /// let cockpit_pos = Point3::new(0.0, 100.0, 0.0);
        /// let nose_direction = Point3::new(100.0, 100.0, 100.0);  // Looking ahead
        /// let up = Vector3::y();
        ///
        /// // In left-handed system, forward is +z
        /// let view_matrix = Isometry3::look_at_lh(&cockpit_pos, &nose_direction, &up);
        /// ```
        ///
        /// ## Understanding Left-Handed vs Right-Handed
        ///
        /// ```
        /// # use nalgebra::{Isometry3, Point3, Vector3};
        ///
        /// let eye = Point3::new(0.0, 0.0, -5.0);
        /// let target = Point3::origin();
        /// let up = Vector3::y();
        ///
        /// // Left-handed: looking toward +z
        /// let view_lh = Isometry3::look_at_lh(&eye, &target, &up);
        /// let point_ahead = target;
        /// let view_space_lh = view_lh * point_ahead;
        /// assert!(view_space_lh.z > 0.0);  // Positive z = in front
        ///
        /// // Right-handed: looking toward -z
        /// let view_rh = Isometry3::look_at_rh(&eye, &target, &up);
        /// let view_space_rh = view_rh * point_ahead;
        /// assert!(view_space_rh.z < 0.0);  // Negative z = in front
        /// ```
        ///
        /// ## Practical Example: RTS Game Camera
        ///
        /// ```
        /// # use nalgebra::{Isometry3, Point3, Vector3};
        ///
        /// // Top-down RTS camera looking at the battlefield
        /// let camera_height = 50.0;
        /// let camera_pos = Point3::new(25.0, camera_height, 25.0);
        /// let battlefield_center = Point3::new(25.0, 0.0, 25.0);
        ///
        /// // Using left-handed system
        /// let view_matrix = Isometry3::look_at_lh(
        ///     &camera_pos,
        ///     &battlefield_center,
        ///     &Vector3::y()
        /// );
        ///
        /// // Transform unit positions to view space
        /// let unit_pos = Point3::new(30.0, 0.0, 20.0);
        /// let view_space_pos = view_matrix * unit_pos;
        /// ```
        ///
        /// # See Also
        /// - [`Isometry3::look_at_rh`] - Right-handed look-at (for OpenGL-style systems)
        /// - [`Isometry3::face_towards`] - Create object-to-world transform (not a view matrix)
        /// - [`Perspective3::new`] - Create perspective projection (often combined with view matrix)
        #[inline]
        pub fn look_at_lh(eye:    &Point3<T>,
                          target: &Point3<T>,
                          up:     &Vector3<T>)
                          -> Self {
            let rotation = $RotId::look_at_lh(&(target - eye), up);
            let trans    = &rotation * (-eye);

            Self::from_parts(Translation::from(trans.coords), rotation)
        }
    }
);

/// # Construction from a 3D vector and/or an axis-angle
impl<T: SimdRealField> Isometry3<T>
where
    T::Element: SimdRealField,
{
    basic_isometry_construction_impl!(UnitQuaternion<T>);

    /// Converts the isometry to use a different numeric type.
    ///
    /// This method allows you to convert between different numeric types (e.g., from `f64`
    /// to `f32`, or between custom numeric types). This is useful when interfacing with
    /// libraries that require specific numeric types, or when optimizing for memory/performance.
    ///
    /// The conversion applies to both the translation and rotation components of the isometry.
    ///
    /// # Type Parameters
    /// - `To` - The target numeric type to convert to
    ///
    /// # Use Cases
    /// - Convert between single and double precision for performance/accuracy tradeoffs
    /// - Interface with graphics APIs that require specific float types
    /// - Convert physics simulation results to rendering precision
    /// - Convert to custom numeric types for specialized computations
    ///
    /// # Examples
    ///
    /// ## Basic Type Conversion
    ///
    /// ```
    /// # use nalgebra::Isometry3;
    ///
    /// // Create an isometry with f64 precision
    /// let iso = Isometry3::<f64>::identity();
    ///
    /// // Convert to f32 for a graphics API
    /// let iso2 = iso.cast::<f32>();
    /// assert_eq!(iso2, Isometry3::<f32>::identity());
    /// ```
    ///
    /// ## Practical Example: Physics to Rendering Pipeline
    ///
    /// ```
    /// # use std::f64;
    /// # use nalgebra::{Isometry3, Vector3};
    ///
    /// // Physics simulation uses f64 for accuracy
    /// let physics_transform = Isometry3::<f64>::new(
    ///     Vector3::new(100.0, 50.0, 75.0),
    ///     Vector3::y() * f64::consts::FRAC_PI_4
    /// );
    ///
    /// // Graphics rendering uses f32 for efficiency
    /// let render_transform = physics_transform.cast::<f32>();
    ///
    /// // Both represent the same transformation (within precision limits)
    /// assert!((render_transform.translation.x - 100.0f32).abs() < 1e-5);
    /// ```
    ///
    /// ## Practical Example: Multi-Precision Simulation
    ///
    /// ```
    /// # use std::f32;
    /// # use nalgebra::{Isometry3, Vector3};
    ///
    /// // Game object transform in f32
    /// let game_object = Isometry3::<f32>::new(
    ///     Vector3::new(10.0, 0.0, 5.0),
    ///     Vector3::y() * f32::consts::FRAC_PI_2
    /// );
    ///
    /// // Convert to f64 for high-precision collision detection
    /// let precise_transform = game_object.cast::<f64>();
    ///
    /// // Perform precise calculations...
    /// // Then convert back to f32 for rendering
    /// let render_transform = precise_transform.cast::<f32>();
    /// ```
    ///
    /// # See Also
    /// - [`Isometry2::cast`] - Cast for 2D isometries
    /// - [`IsometryMatrix3::cast`] - Cast for IsometryMatrix3 (with matrix rotation)
    pub fn cast<To: Scalar>(self) -> Isometry3<To>
    where
        Isometry3<To>: SupersetOf<Self>,
    {
        crate::convert(self)
    }
}

impl<T: SimdRealField> IsometryMatrix3<T>
where
    T::Element: SimdRealField,
{
    basic_isometry_construction_impl!(Rotation3<T>);

    /// Converts the isometry to use a different numeric type.
    ///
    /// This method allows you to convert between different numeric types (e.g., from `f64`
    /// to `f32`, or between custom numeric types). This is useful when interfacing with
    /// libraries that require specific numeric types, or when optimizing for memory/performance.
    ///
    /// The conversion applies to both the translation and the 3x3 rotation matrix components.
    ///
    /// # Type Parameters
    /// - `To` - The target numeric type to convert to
    ///
    /// # Use Cases
    /// - Convert between single and double precision for performance/accuracy tradeoffs
    /// - Interface with graphics APIs that require specific float types
    /// - Convert physics simulation results to rendering precision
    /// - Convert to custom numeric types for specialized computations
    ///
    /// # Examples
    ///
    /// ## Basic Type Conversion
    ///
    /// ```
    /// # use nalgebra::IsometryMatrix3;
    ///
    /// // Create an isometry with f64 precision
    /// let iso = IsometryMatrix3::<f64>::identity();
    ///
    /// // Convert to f32 for a graphics API
    /// let iso2 = iso.cast::<f32>();
    /// assert_eq!(iso2, IsometryMatrix3::<f32>::identity());
    /// ```
    ///
    /// ## Practical Example: High-Precision to GPU Rendering
    ///
    /// ```
    /// # use std::f64;
    /// # use nalgebra::{IsometryMatrix3, Vector3};
    ///
    /// // Simulation uses f64 and matrix rotation for accuracy
    /// let simulation_transform = IsometryMatrix3::<f64>::new(
    ///     Vector3::new(1000.123456, 2000.654321, 3000.987654),
    ///     Vector3::z() * f64::consts::FRAC_PI_3
    /// );
    ///
    /// // GPU shader uses f32
    /// let gpu_transform = simulation_transform.cast::<f32>();
    ///
    /// // Values are converted (with some precision loss)
    /// assert!((gpu_transform.translation.x - 1000.123456f32).abs() < 1e-3);
    /// ```
    ///
    /// # See Also
    /// - [`Isometry3::cast`] - Cast for Isometry3 (with quaternion rotation)
    /// - [`IsometryMatrix2::cast`] - Cast for 2D isometry matrices
    pub fn cast<To: Scalar>(self) -> IsometryMatrix3<To>
    where
        IsometryMatrix3<To>: SupersetOf<Self>,
    {
        crate::convert(self)
    }
}

/// # Construction from a 3D eye position and target point
impl<T: SimdRealField> Isometry3<T>
where
    T::Element: SimdRealField,
{
    look_at_isometry_construction_impl!(UnitQuaternion<T>);
}

impl<T: SimdRealField> IsometryMatrix3<T>
where
    T::Element: SimdRealField,
{
    look_at_isometry_construction_impl!(Rotation3<T>);
}
