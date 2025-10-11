/*
 *
 * Computer-graphics specific implementations.
 * Currently, it is mostly implemented for homogeneous matrices in 2- and 3-space.
 *
 */

use num::{One, Zero};

use crate::base::allocator::Allocator;
use crate::base::dimension::{DimName, DimNameDiff, DimNameSub, U1};
use crate::base::storage::{Storage, StorageMut};
use crate::base::{
    Const, DefaultAllocator, Matrix3, Matrix4, OMatrix, OVector, Scalar, SquareMatrix, Unit,
    Vector, Vector2, Vector3,
};
use crate::geometry::{
    Isometry, IsometryMatrix3, Orthographic3, Perspective3, Point, Point2, Point3, Rotation2,
    Rotation3,
};

use simba::scalar::{ClosedAddAssign, ClosedMulAssign, RealField};

/// # Translation and scaling in any dimension
impl<T, D: DimName> OMatrix<T, D, D>
where
    T: Scalar + Zero + One,
    DefaultAllocator: Allocator<D, D>,
{
    /// Creates a new homogeneous matrix that applies the same scaling factor on each dimension.
    ///
    /// This function creates a **uniform scaling transformation** in homogeneous coordinates.
    /// Uniform scaling means all dimensions (x, y, z, etc.) are scaled by the same factor.
    ///
    /// In computer graphics, scaling transformations change the size of objects. A scaling
    /// factor of 2.0 makes objects twice as large, while 0.5 makes them half the size.
    /// The transformation is represented as a homogeneous matrix, which allows combining
    /// multiple transformations (scaling, rotation, translation) into a single matrix.
    ///
    /// # Arguments
    ///
    /// * `scaling` - The uniform scaling factor to apply to all dimensions
    ///
    /// # Examples
    ///
    /// ## 2D Uniform Scaling (3x3 matrix)
    ///
    /// ```
    /// use nalgebra::Matrix3;
    ///
    /// // Create a 2D scaling matrix that doubles the size of objects
    /// let scale_matrix = Matrix3::new_scaling(2.0);
    ///
    /// // The resulting matrix looks like:
    /// // [2.0, 0.0, 0.0]
    /// // [0.0, 2.0, 0.0]
    /// // [0.0, 0.0, 1.0]
    /// assert_eq!(scale_matrix.m11, 2.0);
    /// assert_eq!(scale_matrix.m22, 2.0);
    /// assert_eq!(scale_matrix.m33, 1.0);
    /// ```
    ///
    /// ## 3D Uniform Scaling (4x4 matrix)
    ///
    /// ```
    /// use nalgebra::{Matrix4, Point3};
    ///
    /// // Create a 3D scaling matrix that halves the size of objects
    /// let scale_matrix = Matrix4::new_scaling(0.5);
    ///
    /// // Apply the scaling to a point
    /// let point = Point3::new(4.0, 6.0, 8.0);
    /// let scaled_point = scale_matrix.transform_point(&point);
    ///
    /// assert_eq!(scaled_point, Point3::new(2.0, 3.0, 4.0));
    /// ```
    ///
    /// ## Using in an MVP (Model-View-Projection) matrix
    ///
    /// ```
    /// use nalgebra::Matrix4;
    ///
    /// // Create a model matrix that scales an object by 1.5x
    /// let model = Matrix4::new_scaling(1.5_f32);
    ///
    /// // Create a simple view matrix (camera at origin looking down -Z)
    /// let view = Matrix4::<f32>::identity();
    ///
    /// // Combine transformations (in graphics, order matters!)
    /// let model_view = view * model;
    /// ```
    ///
    /// # See Also
    ///
    /// * [`new_nonuniform_scaling`](Self::new_nonuniform_scaling) - For different scaling factors per dimension
    /// * [`append_scaling`](Self::append_scaling) - To append scaling to an existing transformation
    /// * [`prepend_scaling`](Self::prepend_scaling) - To prepend scaling to an existing transformation
    #[inline]
    pub fn new_scaling(scaling: T) -> Self {
        let mut res = Self::from_diagonal_element(scaling);
        res[(D::DIM - 1, D::DIM - 1)] = T::one();

        res
    }

    /// Creates a new homogeneous matrix that applies a distinct scaling factor for each dimension.
    ///
    /// This function creates a **non-uniform scaling transformation** in homogeneous coordinates.
    /// Unlike uniform scaling, each dimension (x, y, z, etc.) can be scaled by a different factor.
    /// This is useful for stretching or squashing objects along specific axes.
    ///
    /// Non-uniform scaling is commonly used in computer graphics for:
    /// - Stretching UI elements to fit different screen ratios
    /// - Creating ellipsoids from spheres
    /// - Adjusting object proportions without maintaining aspect ratio
    ///
    /// # Arguments
    ///
    /// * `scaling` - A vector containing the scaling factor for each dimension
    ///
    /// # Examples
    ///
    /// ## 2D Non-uniform Scaling
    ///
    /// ```
    /// use nalgebra::{Matrix3, Vector2, Point2};
    ///
    /// // Scale x by 3.0 and y by 0.5 (stretch horizontally, compress vertically)
    /// let scaling = Vector2::new(3.0, 0.5);
    /// let scale_matrix = Matrix3::new_nonuniform_scaling(&scaling);
    ///
    /// // Transform a point
    /// let point = Point2::new(2.0, 4.0);
    /// let scaled_point = scale_matrix.transform_point(&point);
    ///
    /// assert_eq!(scaled_point, Point2::new(6.0, 2.0));
    /// ```
    ///
    /// ## 3D Non-uniform Scaling
    ///
    /// ```
    /// use nalgebra::{Matrix4, Vector3, Point3};
    ///
    /// // Create a scaling that stretches along X and Z, but compresses Y
    /// let scaling = Vector3::new(2.0, 0.5, 2.0);
    /// let scale_matrix = Matrix4::new_nonuniform_scaling(&scaling);
    ///
    /// // Transform a cube corner point
    /// let point = Point3::new(1.0, 1.0, 1.0);
    /// let scaled_point = scale_matrix.transform_point(&point);
    ///
    /// assert_eq!(scaled_point, Point3::new(2.0, 0.5, 2.0));
    /// ```
    ///
    /// ## Creating an ellipsoid from a sphere
    ///
    /// ```
    /// use nalgebra::{Matrix4, Vector3};
    ///
    /// // Scale a unit sphere to create an ellipsoid
    /// // Semi-axes: a=2.0 (x), b=1.0 (y), c=3.0 (z)
    /// let ellipsoid_scale = Vector3::new(2.0, 1.0, 3.0);
    /// let transform = Matrix4::new_nonuniform_scaling(&ellipsoid_scale);
    ///
    /// // This matrix can now transform sphere vertices to ellipsoid vertices
    /// ```
    ///
    /// ## Aspect ratio correction for UI
    ///
    /// ```
    /// use nalgebra::{Matrix3, Vector2};
    ///
    /// // Correct aspect ratio for a 16:9 screen
    /// let aspect_ratio = 16.0 / 9.0;
    /// let correction = Vector2::new(1.0, aspect_ratio);
    /// let aspect_matrix = Matrix3::new_nonuniform_scaling(&correction);
    /// ```
    ///
    /// # See Also
    ///
    /// * [`new_scaling`](Self::new_scaling) - For uniform scaling (same factor for all dimensions)
    /// * [`append_nonuniform_scaling`](Self::append_nonuniform_scaling) - To append non-uniform scaling to an existing transformation
    /// * [`prepend_nonuniform_scaling`](Self::prepend_nonuniform_scaling) - To prepend non-uniform scaling to an existing transformation
    #[inline]
    pub fn new_nonuniform_scaling<SB>(scaling: &Vector<T, DimNameDiff<D, U1>, SB>) -> Self
    where
        D: DimNameSub<U1>,
        SB: Storage<T, DimNameDiff<D, U1>>,
    {
        let mut res = Self::identity();
        for i in 0..scaling.len() {
            res[(i, i)] = scaling[i].clone();
        }

        res
    }

    /// Creates a new homogeneous matrix that applies a pure translation.
    ///
    /// This function creates a **translation transformation** in homogeneous coordinates.
    /// Translation moves objects from one position to another without rotating or scaling them.
    ///
    /// In computer graphics, translation is one of the fundamental transformations. It shifts
    /// all points of an object by the same amount in each dimension. For example, adding
    /// `(2, 3)` to every point moves an object 2 units right and 3 units up in 2D.
    ///
    /// The translation is stored in the last column of the homogeneous matrix, which allows
    /// it to be combined with rotations and scaling into a single transformation matrix.
    ///
    /// # Arguments
    ///
    /// * `translation` - A vector specifying how far to move in each dimension
    ///
    /// # Examples
    ///
    /// ## 2D Translation
    ///
    /// ```
    /// use nalgebra::{Matrix3, Vector2, Point2};
    ///
    /// // Move objects 3 units right and 2 units up
    /// let translation = Vector2::new(3.0, 2.0);
    /// let trans_matrix = Matrix3::new_translation(&translation);
    ///
    /// // Apply translation to a point
    /// let point = Point2::new(1.0, 1.0);
    /// let translated_point = trans_matrix.transform_point(&point);
    ///
    /// assert_eq!(translated_point, Point2::new(4.0, 3.0));
    /// ```
    ///
    /// ## 3D Translation
    ///
    /// ```
    /// use nalgebra::{Matrix4, Vector3, Point3};
    ///
    /// // Move objects in 3D space
    /// let translation = Vector3::new(5.0, -2.0, 3.0);
    /// let trans_matrix = Matrix4::new_translation(&translation);
    ///
    /// // Apply to a point
    /// let point = Point3::new(0.0, 0.0, 0.0);
    /// let translated_point = trans_matrix.transform_point(&point);
    ///
    /// assert_eq!(translated_point, Point3::new(5.0, -2.0, 3.0));
    /// ```
    ///
    /// ## Positioning objects in a scene
    ///
    /// ```
    /// use nalgebra::{Matrix4, Vector3};
    ///
    /// // Position three objects at different locations
    /// let object1_pos = Matrix4::new_translation(&Vector3::new(0.0, 0.0, -5.0));
    /// let object2_pos = Matrix4::new_translation(&Vector3::new(3.0, 1.0, -5.0));
    /// let object3_pos = Matrix4::new_translation(&Vector3::new(-3.0, -1.0, -5.0));
    /// ```
    ///
    /// ## Combining with other transformations
    ///
    /// ```
    /// use nalgebra::{Matrix4, Vector3};
    ///
    /// // Create a model matrix: first rotate, then translate
    /// // (Note: matrix multiplication order is right-to-left)
    /// let rotation = Matrix4::from_euler_angles(0.0, 0.0, std::f32::consts::PI / 4.0);
    /// let translation = Matrix4::new_translation(&Vector3::new(10.0, 0.0, 0.0));
    /// let model = translation * rotation;
    /// ```
    ///
    /// # See Also
    ///
    /// * [`append_translation`](Self::append_translation) - To append translation to an existing transformation
    /// * [`prepend_translation`](Self::prepend_translation) - To prepend translation to an existing transformation
    /// * [`new_scaling`](Self::new_scaling) - For scaling transformations
    /// * [`Matrix4::new_rotation`] - For rotation transformations
    #[inline]
    pub fn new_translation<SB>(translation: &Vector<T, DimNameDiff<D, U1>, SB>) -> Self
    where
        D: DimNameSub<U1>,
        SB: Storage<T, DimNameDiff<D, U1>>,
    {
        let mut res = Self::identity();
        res.generic_view_mut((0, D::DIM - 1), (DimNameDiff::<D, U1>::name(), Const::<1>))
            .copy_from(translation);

        res
    }
}

/// # 2D transformations as a Matrix3
impl<T: RealField> Matrix3<T> {
    /// Builds a 2 dimensional homogeneous rotation matrix from an angle in radian.
    ///
    /// This function creates a **2D rotation transformation** around the origin (0, 0).
    /// In 2D, rotation happens in the plane, spinning objects around a central point.
    ///
    /// The angle is measured in **radians**, not degrees. Positive angles rotate
    /// counter-clockwise (CCW), and negative angles rotate clockwise (CW).
    /// A full rotation is 2π radians (approximately 6.28), which equals 360 degrees.
    ///
    /// # Arguments
    ///
    /// * `angle` - The rotation angle in radians. Use positive for counter-clockwise rotation
    ///
    /// # Examples
    ///
    /// ## Basic 2D rotation
    ///
    /// ```
    /// use nalgebra::{Matrix3, Point2};
    /// use std::f32::consts::PI;
    ///
    /// // Rotate 90 degrees counter-clockwise (π/2 radians)
    /// let rotation = Matrix3::new_rotation(PI / 2.0);
    ///
    /// // Point on the positive x-axis
    /// let point = Point2::new(1.0, 0.0);
    /// let rotated = rotation.transform_point(&point);
    ///
    /// // After 90° CCW rotation, point is on positive y-axis
    /// assert!((rotated.x - 0.0).abs() < 1e-6);
    /// assert!((rotated.y - 1.0).abs() < 1e-6);
    /// ```
    ///
    /// ## Converting degrees to radians
    ///
    /// ```
    /// use nalgebra::Matrix3;
    ///
    /// // Helper function to convert degrees to radians
    /// fn deg_to_rad(degrees: f32) -> f32 {
    ///     degrees * std::f32::consts::PI / 180.0
    /// }
    ///
    /// // Rotate 45 degrees counter-clockwise
    /// let rotation = Matrix3::new_rotation(deg_to_rad(45.0));
    /// ```
    ///
    /// ## Rotating game objects
    ///
    /// ```
    /// use nalgebra::{Matrix3, Point2};
    ///
    /// // Rotate a player character by 30 degrees
    /// let angle_radians = 30.0_f32.to_radians();
    /// let rotation = Matrix3::new_rotation(angle_radians);
    ///
    /// let player_pos = Point2::new(5.0, 3.0);
    /// let rotated_pos = rotation.transform_point(&player_pos);
    /// ```
    ///
    /// ## Clockwise rotation
    ///
    /// ```
    /// use nalgebra::{Matrix3, Point2};
    /// use std::f32::consts::PI;
    ///
    /// // Rotate 90 degrees clockwise (negative angle)
    /// let rotation = Matrix3::new_rotation(-PI / 2.0);
    ///
    /// let point = Point2::new(1.0, 0.0);
    /// let rotated = rotation.transform_point(&point);
    ///
    /// // Point is now on negative y-axis
    /// assert!((rotated.x - 0.0).abs() < 1e-6);
    /// assert!((rotated.y - (-1.0)).abs() < 1e-6);
    /// ```
    ///
    /// # See Also
    ///
    /// * [`Matrix4::new_rotation`] - For 3D rotation using axis-angle representation
    /// * [`new_translation`](Self::new_translation) - For translation transformations
    /// * [`new_scaling`](Self::new_scaling) - For scaling transformations
    #[inline]
    pub fn new_rotation(angle: T) -> Self {
        Rotation2::new(angle).to_homogeneous()
    }

    /// Creates a new homogeneous matrix that applies a scaling factor for each dimension with respect to point.
    ///
    /// This function creates a **non-uniform scaling transformation with a fixed pivot point**.
    /// Unlike regular scaling which always scales around the origin (0, 0), this function
    /// scales around an arbitrary point. The specified point remains stationary while
    /// everything else scales relative to it.
    ///
    /// This is extremely useful for implementing "zoom to cursor" functionality in 2D
    /// applications, where you want to zoom in/out while keeping the mouse cursor position
    /// fixed on the screen.
    ///
    /// # Arguments
    ///
    /// * `scaling` - The scaling factor for each dimension (x, y)
    /// * `pt` - The pivot point that remains fixed during scaling
    ///
    /// # Examples
    ///
    /// ## Basic scaling with respect to a point
    ///
    /// ```
    /// use nalgebra::{Matrix3, Vector2, Point2};
    ///
    /// // Scale by 2x around point (5, 5)
    /// let scaling = Vector2::new(2.0_f32, 2.0);
    /// let pivot = Point2::new(5.0_f32, 5.0);
    /// let transform = Matrix3::new_nonuniform_scaling_wrt_point(&scaling, &pivot);
    ///
    /// // The pivot point stays in the same place
    /// let transformed_pivot = transform.transform_point(&pivot);
    /// assert!((transformed_pivot.x - 5.0).abs() < 1e-6);
    /// assert!((transformed_pivot.y - 5.0).abs() < 1e-6);
    ///
    /// // Other points move away from the pivot
    /// let point = Point2::new(7.0_f32, 5.0);
    /// let transformed = transform.transform_point(&point);
    /// assert!((transformed.x - 9.0).abs() < 1e-6); // Moved 2 units away
    /// ```
    ///
    /// ## Zoom to cursor functionality
    ///
    /// ```
    /// use nalgebra::{Matrix3, Vector2, Point2};
    ///
    /// // User's mouse cursor position
    /// let cursor_pos = Point2::new(100.0_f32, 150.0);
    ///
    /// // Zoom in by 1.5x around the cursor
    /// let zoom_factor = 1.5_f32;
    /// let scaling = Vector2::new(zoom_factor, zoom_factor);
    /// let zoom_transform = Matrix3::new_nonuniform_scaling_wrt_point(&scaling, &cursor_pos);
    ///
    /// // The cursor position stays fixed after the zoom
    /// let transformed_cursor = zoom_transform.transform_point(&cursor_pos);
    /// assert!((transformed_cursor.x - 100.0).abs() < 1e-5);
    /// assert!((transformed_cursor.y - 150.0).abs() < 1e-5);
    /// ```
    ///
    /// ## Non-uniform scaling around a point
    ///
    /// ```
    /// use nalgebra::{Matrix3, Vector2, Point2};
    ///
    /// // Stretch horizontally, compress vertically, around center point
    /// let center = Point2::new(10.0_f32, 10.0);
    /// let scaling = Vector2::new(2.0_f32, 0.5);
    /// let transform = Matrix3::new_nonuniform_scaling_wrt_point(&scaling, &center);
    ///
    /// // Point above center moves closer (compressed in Y)
    /// let point = Point2::new(10.0_f32, 14.0);
    /// let transformed = transform.transform_point(&point);
    /// assert!((transformed.x - 10.0).abs() < 1e-6);
    /// assert!((transformed.y - 12.0).abs() < 1e-6); // Only 2 units above now
    /// ```
    ///
    /// # See Also
    ///
    /// * [`new_nonuniform_scaling`](Self::new_nonuniform_scaling) - For scaling around the origin
    /// * [`Matrix4::new_nonuniform_scaling_wrt_point`] - For 3D version
    /// * [`new_scaling`](Self::new_scaling) - For uniform scaling
    #[inline]
    pub fn new_nonuniform_scaling_wrt_point(scaling: &Vector2<T>, pt: &Point2<T>) -> Self {
        let zero = T::zero();
        let one = T::one();
        Matrix3::new(
            scaling.x.clone(),
            zero.clone(),
            pt.x.clone() - pt.x.clone() * scaling.x.clone(),
            zero.clone(),
            scaling.y.clone(),
            pt.y.clone() - pt.y.clone() * scaling.y.clone(),
            zero.clone(),
            zero,
            one,
        )
    }
}

/// # 3D transformations as a Matrix4
impl<T: RealField> Matrix4<T> {
    /// Builds a 3D homogeneous rotation matrix from an axis and an angle (multiplied together).
    ///
    /// This function creates a **3D rotation transformation** using the **axis-angle representation**.
    /// In this representation, the rotation is specified by a vector whose direction indicates
    /// the axis of rotation, and whose magnitude (length) indicates the angle in radians.
    ///
    /// In 3D graphics, rotations happen around an axis (an imaginary line through space).
    /// For example, rotating around the Y-axis spins an object like a spinning top,
    /// while rotating around the X-axis tilts it forward or backward.
    ///
    /// Returns the identity matrix (no rotation) if the given vector is zero.
    ///
    /// # Arguments
    ///
    /// * `axisangle` - A vector whose **direction** is the rotation axis and **magnitude** is the angle in radians
    ///
    /// # Examples
    ///
    /// ## Basic rotation around Y-axis
    ///
    /// ```
    /// use nalgebra::{Matrix4, Vector3, Point3};
    /// use std::f32::consts::PI;
    ///
    /// // Rotate 90 degrees around Y-axis (up direction)
    /// // Y-axis is (0, 1, 0), and angle is PI/2
    /// let axis_angle = Vector3::new(0.0, PI / 2.0, 0.0);
    /// let rotation = Matrix4::new_rotation(axis_angle);
    ///
    /// // Point on positive X-axis
    /// let point = Point3::new(1.0, 0.0, 0.0);
    /// let rotated = rotation.transform_point(&point);
    ///
    /// // After rotation, point is on negative Z-axis
    /// assert!((rotated.x - 0.0).abs() < 1e-6);
    /// assert!((rotated.z - (-1.0)).abs() < 1e-6);
    /// ```
    ///
    /// ## Arbitrary axis rotation
    ///
    /// ```
    /// use nalgebra::{Matrix4, Vector3, Point3};
    ///
    /// // Rotate around a diagonal axis
    /// let axis = Vector3::new(1.0, 1.0, 0.0).normalize();
    /// let angle = std::f32::consts::PI / 4.0; // 45 degrees
    /// let axis_angle = axis * angle;
    /// let rotation = Matrix4::new_rotation(axis_angle);
    /// ```
    ///
    /// ## Zero vector returns identity
    ///
    /// ```
    /// use nalgebra::{Matrix4, Vector3, Point3};
    ///
    /// // Zero rotation
    /// let no_rotation = Matrix4::new_rotation(Vector3::zeros());
    ///
    /// // Points don't move
    /// let point = Point3::new(1.0, 2.0, 3.0);
    /// let result = no_rotation.transform_point(&point);
    /// assert_eq!(result, point);
    /// ```
    ///
    /// ## Combining rotations for character orientation
    ///
    /// ```
    /// use nalgebra::{Matrix4, Vector3};
    ///
    /// // Character facing direction (rotate around Y)
    /// let facing = Matrix4::new_rotation(Vector3::new(0.0, 1.57, 0.0));
    ///
    /// // Camera pitch (rotate around X)
    /// let pitch = Matrix4::new_rotation(Vector3::new(0.3, 0.0, 0.0));
    ///
    /// // Combined transformation
    /// let view_rotation = pitch * facing;
    /// ```
    ///
    /// # See Also
    ///
    /// * [`from_axis_angle`](Self::from_axis_angle) - Alternative with separate axis and angle parameters
    /// * [`from_euler_angles`](Self::from_euler_angles) - For Euler angle rotations (roll, pitch, yaw)
    /// * [`from_scaled_axis`](Self::from_scaled_axis) - Identical to this function
    /// * [`new_rotation_wrt_point`](Self::new_rotation_wrt_point) - To rotate around a specific point
    #[inline]
    pub fn new_rotation(axisangle: Vector3<T>) -> Self {
        Rotation3::new(axisangle).to_homogeneous()
    }

    /// Builds a 3D homogeneous rotation matrix from an axis and an angle (multiplied together).
    ///
    /// This function creates a **3D rotation around an arbitrary point** (pivot point).
    /// Instead of rotating around the origin like `new_rotation`, this rotates around
    /// a specified point in 3D space. The pivot point stays fixed while everything else
    /// rotates around it.
    ///
    /// This is useful for:
    /// - Rotating an object around a specific joint or hinge point
    /// - Rotating a door around its hinges
    /// - Spinning a wheel around its center (which may not be at the origin)
    /// - Rotating a camera around a look-at point
    ///
    /// Returns the identity matrix if the given axis-angle vector is zero.
    ///
    /// # Arguments
    ///
    /// * `axisangle` - A vector whose direction is the rotation axis and magnitude is the angle in radians
    /// * `pt` - The pivot point that remains fixed during rotation
    ///
    /// # Examples
    ///
    /// ## Rotating around a specific point
    ///
    /// ```
    /// use nalgebra::{Matrix4, Vector3, Point3};
    /// use std::f32::consts::PI;
    ///
    /// // Rotate 180 degrees around Y-axis, but centered at (5, 0, 0)
    /// let axis_angle = Vector3::new(0.0, PI, 0.0);
    /// let pivot = Point3::new(5.0, 0.0, 0.0);
    /// let rotation = Matrix4::new_rotation_wrt_point(axis_angle, pivot);
    ///
    /// // The pivot point stays in place
    /// let transformed_pivot = rotation.transform_point(&pivot);
    /// assert!((transformed_pivot.x - 5.0).abs() < 1e-5);
    /// assert!((transformed_pivot.y - 0.0).abs() < 1e-5);
    /// assert!((transformed_pivot.z - 0.0).abs() < 1e-5);
    /// ```
    ///
    /// ## Door hinge simulation
    ///
    /// ```
    /// use nalgebra::{Matrix4, Vector3, Point3};
    ///
    /// // Door hinge at position (0, 0, 0), door opens around Y-axis
    /// let hinge_pos = Point3::new(0.0, 0.0, 0.0);
    /// let open_angle = 90.0_f32.to_radians();
    /// let axis_angle = Vector3::new(0.0, open_angle, 0.0);
    ///
    /// let door_transform = Matrix4::new_rotation_wrt_point(axis_angle, hinge_pos);
    ///
    /// // Door handle position before opening
    /// let handle = Point3::new(1.0, 1.0, 0.0);
    /// let handle_after = door_transform.transform_point(&handle);
    /// ```
    ///
    /// ## Orbital rotation around a planet
    ///
    /// ```
    /// use nalgebra::{Matrix4, Vector3, Point3};
    ///
    /// // Planet at (0, 0, 0), moon orbits around it
    /// let planet_pos = Point3::new(0.0, 0.0, 0.0);
    /// let orbit_angle = 0.1; // radians per frame
    /// let orbit_axis = Vector3::new(0.0, orbit_angle, 0.0);
    ///
    /// let orbit_transform = Matrix4::new_rotation_wrt_point(orbit_axis, planet_pos);
    ///
    /// // Moon position
    /// let moon_pos = Point3::new(10.0, 0.0, 0.0);
    /// let new_moon_pos = orbit_transform.transform_point(&moon_pos);
    /// ```
    ///
    /// # See Also
    ///
    /// * [`new_rotation`](Self::new_rotation) - For rotation around the origin
    /// * [`from_axis_angle`](Self::from_axis_angle) - For separate axis and angle parameters
    /// * [`Matrix3::new_nonuniform_scaling_wrt_point`] - For 2D scaling with a pivot point
    #[inline]
    pub fn new_rotation_wrt_point(axisangle: Vector3<T>, pt: Point3<T>) -> Self {
        let rot = Rotation3::from_scaled_axis(axisangle);
        Isometry::rotation_wrt_point(rot, pt).to_homogeneous()
    }

    /// Creates a new homogeneous matrix that applies a scaling factor for each dimension with respect to point.
    ///
    /// This function creates a **3D non-uniform scaling transformation with a fixed pivot point**.
    /// The specified point remains stationary while everything else scales relative to it.
    /// This is the 3D version of the 2D scaling with respect to a point.
    ///
    /// This is useful for:
    /// - Implementing "zoom to point" in 3D viewers
    /// - Scaling objects around their center rather than the world origin
    /// - Creating size animations that grow/shrink from a specific point
    ///
    /// # Arguments
    ///
    /// * `scaling` - The scaling factor for each dimension (x, y, z)
    /// * `pt` - The pivot point that remains fixed during scaling
    ///
    /// # Examples
    ///
    /// ## Basic 3D scaling with respect to a point
    ///
    /// ```
    /// use nalgebra::{Matrix4, Vector3, Point3};
    ///
    /// // Scale by 2x around point (5, 5, 5)
    /// let scaling = Vector3::new(2.0_f32, 2.0, 2.0);
    /// let pivot = Point3::new(5.0_f32, 5.0, 5.0);
    /// let transform = Matrix4::new_nonuniform_scaling_wrt_point(&scaling, &pivot);
    ///
    /// // The pivot point stays in the same place
    /// let transformed_pivot = transform.transform_point(&pivot);
    /// assert!((transformed_pivot.x - 5.0).abs() < 1e-6);
    /// assert!((transformed_pivot.y - 5.0).abs() < 1e-6);
    /// assert!((transformed_pivot.z - 5.0).abs() < 1e-6);
    /// ```
    ///
    /// ## Scaling an object around its center
    ///
    /// ```
    /// use nalgebra::{Matrix4, Vector3, Point3};
    ///
    /// // Object center is at (10, 5, -3)
    /// let object_center = Point3::new(10.0, 5.0, -3.0);
    ///
    /// // Make object 1.5x larger in all dimensions
    /// let scaling = Vector3::new(1.5, 1.5, 1.5);
    /// let scale_transform = Matrix4::new_nonuniform_scaling_wrt_point(&scaling, &object_center);
    ///
    /// // The center stays at the same position
    /// let new_center = scale_transform.transform_point(&object_center);
    /// assert_eq!(new_center, object_center);
    /// ```
    ///
    /// ## Non-uniform 3D scaling for special effects
    ///
    /// ```
    /// use nalgebra::{Matrix4, Vector3, Point3};
    ///
    /// // Create a "squash and stretch" effect
    /// let center = Point3::new(0.0, 0.0, 0.0);
    /// let squash = Vector3::new(1.5, 0.7, 1.5); // Wider but flatter
    /// let transform = Matrix4::new_nonuniform_scaling_wrt_point(&squash, &center);
    /// ```
    ///
    /// # See Also
    ///
    /// * [`new_nonuniform_scaling`](Self::new_nonuniform_scaling) - For scaling around the origin
    /// * [`Matrix3::new_nonuniform_scaling_wrt_point`] - For 2D version
    /// * [`new_scaling`](Self::new_scaling) - For uniform scaling
    #[inline]
    pub fn new_nonuniform_scaling_wrt_point(scaling: &Vector3<T>, pt: &Point3<T>) -> Self {
        let zero = T::zero();
        let one = T::one();
        Matrix4::new(
            scaling.x.clone(),
            zero.clone(),
            zero.clone(),
            pt.x.clone() - pt.x.clone() * scaling.x.clone(),
            zero.clone(),
            scaling.y.clone(),
            zero.clone(),
            pt.y.clone() - pt.y.clone() * scaling.y.clone(),
            zero.clone(),
            zero.clone(),
            scaling.z.clone(),
            pt.z.clone() - pt.z.clone() * scaling.z.clone(),
            zero.clone(),
            zero.clone(),
            zero,
            one,
        )
    }

    /// Builds a 3D homogeneous rotation matrix from an axis and an angle (multiplied together).
    ///
    /// Returns the identity matrix if the given argument is zero.
    /// This is identical to [`Self::new_rotation`].
    ///
    /// This function is provided as an alternative name for `new_rotation`. Both functions
    /// do exactly the same thing - they create a rotation matrix from a scaled axis vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use nalgebra::{Matrix4, Vector3};
    ///
    /// // These two calls are equivalent
    /// let rotation1 = Matrix4::from_scaled_axis(Vector3::new(0.0, 1.0, 0.0));
    /// let rotation2 = Matrix4::new_rotation(Vector3::new(0.0, 1.0, 0.0));
    ///
    /// assert_eq!(rotation1, rotation2);
    /// ```
    ///
    /// # See Also
    ///
    /// * [`new_rotation`](Self::new_rotation) - Identical function with different name
    /// * [`from_axis_angle`](Self::from_axis_angle) - For separate axis and angle parameters
    #[inline]
    pub fn from_scaled_axis(axisangle: Vector3<T>) -> Self {
        Rotation3::from_scaled_axis(axisangle).to_homogeneous()
    }

    /// Creates a new rotation from Euler angles.
    ///
    /// **Euler angles** are a way to represent 3D rotations using three sequential rotations
    /// around different axes. This function uses the **roll-pitch-yaw** convention, which is
    /// commonly used in aerospace and robotics.
    ///
    /// The rotations are applied in order:
    /// 1. **Roll** - Rotation around the X-axis (tilting left/right)
    /// 2. **Pitch** - Rotation around the Y-axis (tilting up/down)
    /// 3. **Yaw** - Rotation around the Z-axis (turning left/right)
    ///
    /// Think of an airplane:
    /// - Roll makes the plane bank left or right
    /// - Pitch makes the nose go up or down
    /// - Yaw makes the plane turn left or right
    ///
    /// # Arguments
    ///
    /// * `roll` - Rotation around X-axis in radians
    /// * `pitch` - Rotation around Y-axis in radians
    /// * `yaw` - Rotation around Z-axis in radians
    ///
    /// # Examples
    ///
    /// ## Basic Euler angle rotation
    ///
    /// ```
    /// use nalgebra::{Matrix4, Point3};
    /// use std::f32::consts::PI;
    ///
    /// // Create a rotation: no roll, 90° pitch, no yaw
    /// let rotation = Matrix4::from_euler_angles(0.0, PI / 2.0, 0.0);
    ///
    /// let point = Point3::new(1.0, 0.0, 0.0);
    /// let rotated = rotation.transform_point(&point);
    ///
    /// // Point rotated 90° around Y-axis
    /// assert!((rotated.x - 0.0).abs() < 1e-6);
    /// assert!((rotated.z - (-1.0)).abs() < 1e-6);
    /// ```
    ///
    /// ## Camera orientation
    ///
    /// ```
    /// use nalgebra::Matrix4;
    ///
    /// // Camera looking slightly down and to the right
    /// let roll = 0.0;
    /// let pitch = -0.3;  // Looking down 0.3 radians
    /// let yaw = 0.5;     // Turned right 0.5 radians
    ///
    /// let camera_rotation = Matrix4::from_euler_angles(roll, pitch, yaw);
    /// ```
    ///
    /// ## Character controller rotation
    ///
    /// ```
    /// use nalgebra::Matrix4;
    ///
    /// // Character facing north (0° yaw) looking straight ahead
    /// let character_rotation = Matrix4::from_euler_angles(0.0, 0.0, 0.0);
    ///
    /// // Character facing east (90° yaw)
    /// let facing_east = Matrix4::from_euler_angles(
    ///     0.0,
    ///     0.0,
    ///     90.0_f32.to_radians()
    /// );
    /// ```
    ///
    /// ## Combining roll, pitch, and yaw
    ///
    /// ```
    /// use nalgebra::Matrix4;
    /// use std::f32::consts::PI;
    ///
    /// // All three rotations at once
    /// let roll = PI / 6.0;   // 30° roll
    /// let pitch = PI / 4.0;  // 45° pitch
    /// let yaw = PI / 3.0;    // 60° yaw
    ///
    /// let combined = Matrix4::from_euler_angles(roll, pitch, yaw);
    /// ```
    ///
    /// # See Also
    ///
    /// * [`new_rotation`](Self::new_rotation) - For axis-angle rotation
    /// * [`from_axis_angle`](Self::from_axis_angle) - For rotation around a specific axis
    /// * [`look_at_rh`](Self::look_at_rh) - For camera view matrices
    pub fn from_euler_angles(roll: T, pitch: T, yaw: T) -> Self {
        Rotation3::from_euler_angles(roll, pitch, yaw).to_homogeneous()
    }

    /// Builds a 3D homogeneous rotation matrix from an axis and a rotation angle.
    ///
    /// This function creates a **3D rotation** using the **axis-angle representation** with
    /// separate parameters. Unlike `new_rotation` where the axis and angle are combined
    /// into one vector, this function takes them separately.
    ///
    /// The axis must be a **unit vector** (normalized to length 1). The rotation follows
    /// the right-hand rule: if your right thumb points along the axis, your fingers curl
    /// in the direction of positive rotation.
    ///
    /// # Arguments
    ///
    /// * `axis` - A unit vector representing the axis of rotation
    /// * `angle` - The rotation angle in radians
    ///
    /// # Examples
    ///
    /// ## Basic rotation around Y-axis
    ///
    /// ```
    /// use nalgebra::{Matrix4, Vector3, Unit, Point3};
    /// use std::f32::consts::PI;
    ///
    /// // Rotate 90° around the Y-axis (up direction)
    /// let y_axis = Unit::new_normalize(Vector3::new(0.0, 1.0, 0.0));
    /// let rotation = Matrix4::from_axis_angle(&y_axis, PI / 2.0);
    ///
    /// let point = Point3::new(1.0, 0.0, 0.0);
    /// let rotated = rotation.transform_point(&point);
    ///
    /// assert!((rotated.x - 0.0).abs() < 1e-6);
    /// assert!((rotated.z - (-1.0)).abs() < 1e-6);
    /// ```
    ///
    /// ## Rotation around arbitrary axis
    ///
    /// ```
    /// use nalgebra::{Matrix4, Vector3, Unit};
    ///
    /// // Rotate around a diagonal axis
    /// let axis = Unit::new_normalize(Vector3::new(1.0, 1.0, 0.0));
    /// let angle = std::f32::consts::PI / 4.0;
    /// let rotation = Matrix4::from_axis_angle(&axis, angle);
    /// ```
    ///
    /// ## Wheel rotation animation
    ///
    /// ```
    /// use nalgebra::{Matrix4, Vector3, Unit};
    ///
    /// // Wheel rotates around X-axis
    /// let wheel_axis = Unit::new_normalize(Vector3::new(1.0, 0.0, 0.0));
    ///
    /// // Rotate a bit each frame
    /// let rotation_per_frame = 0.1; // radians
    /// let wheel_rotation = Matrix4::from_axis_angle(&wheel_axis, rotation_per_frame);
    /// ```
    ///
    /// ## Comparison with new_rotation
    ///
    /// ```
    /// use nalgebra::{Matrix4, Vector3, Unit};
    /// use std::f32::consts::PI;
    ///
    /// let axis = Unit::new_normalize(Vector3::new(0.0, 1.0, 0.0));
    /// let angle = PI / 2.0;
    ///
    /// // These two are equivalent:
    /// let rotation1 = Matrix4::from_axis_angle(&axis, angle);
    /// let rotation2 = Matrix4::new_rotation(axis.into_inner() * angle);
    ///
    /// // Both produce the same result
    /// ```
    ///
    /// # See Also
    ///
    /// * [`new_rotation`](Self::new_rotation) - For axis-angle as a single scaled vector
    /// * [`from_euler_angles`](Self::from_euler_angles) - For roll-pitch-yaw rotations
    /// * [`from_scaled_axis`](Self::from_scaled_axis) - Identical to `new_rotation`
    pub fn from_axis_angle(axis: &Unit<Vector3<T>>, angle: T) -> Self {
        Rotation3::from_axis_angle(axis, angle).to_homogeneous()
    }

    /// Creates a new homogeneous matrix for an orthographic projection.
    ///
    /// An **orthographic projection** (also called parallel projection) maps 3D coordinates
    /// to 2D screen space without perspective distortion. Objects appear the same size
    /// regardless of their distance from the camera - parallel lines stay parallel.
    ///
    /// This is commonly used for:
    /// - 2D games and UI rendering
    /// - CAD applications and architectural diagrams
    /// - Isometric games (like classic strategy games)
    /// - Technical drawings where precise measurements matter
    ///
    /// The projection is defined by a rectangular viewing box (frustum) in 3D space.
    /// Anything inside this box is visible; everything outside is clipped.
    ///
    /// # Arguments
    ///
    /// * `left` - Left edge of the viewing box
    /// * `right` - Right edge of the viewing box
    /// * `bottom` - Bottom edge of the viewing box
    /// * `top` - Top edge of the viewing box
    /// * `znear` - Near clipping plane (closest visible distance)
    /// * `zfar` - Far clipping plane (farthest visible distance)
    ///
    /// # Examples
    ///
    /// ## Basic orthographic projection
    ///
    /// ```
    /// use nalgebra::{Matrix4, Point3};
    ///
    /// // Create an orthographic projection for a 800x600 window
    /// // Viewing volume from -10 to 10 in depth
    /// let projection = Matrix4::new_orthographic(
    ///     0.0, 800.0,    // left, right
    ///     0.0, 600.0,    // bottom, top
    ///     -10.0, 10.0    // near, far
    /// );
    /// ```
    ///
    /// ## Centered orthographic projection
    ///
    /// ```
    /// use nalgebra::Matrix4;
    ///
    /// // Center the view at origin, 20 units wide and 15 units tall
    /// let projection = Matrix4::new_orthographic(
    ///     -10.0, 10.0,   // left to right: 20 units
    ///     -7.5, 7.5,     // bottom to top: 15 units
    ///     1.0, 100.0     // near to far
    /// );
    /// ```
    ///
    /// ## 2D game projection
    ///
    /// ```
    /// use nalgebra::Matrix4;
    ///
    /// // For a 2D game with 1920x1080 resolution
    /// let width = 1920.0;
    /// let height = 1080.0;
    /// let projection = Matrix4::new_orthographic(
    ///     0.0, width,
    ///     0.0, height,
    ///     -1.0, 1.0      // Minimal depth range for 2D
    /// );
    /// ```
    ///
    /// ## Isometric game projection
    ///
    /// ```
    /// use nalgebra::Matrix4;
    ///
    /// // Orthographic projection for isometric view
    /// let view_size = 20.0;
    /// let aspect_ratio = 16.0 / 9.0;
    /// let projection = Matrix4::new_orthographic(
    ///     -view_size * aspect_ratio, view_size * aspect_ratio,
    ///     -view_size, view_size,
    ///     0.1, 1000.0
    /// );
    /// ```
    ///
    /// ## Complete MVP matrix for 2D rendering
    ///
    /// ```
    /// use nalgebra::{Matrix4, Vector3};
    ///
    /// // Projection matrix
    /// let projection = Matrix4::new_orthographic(0.0, 800.0, 0.0, 600.0, -1.0, 1.0);
    ///
    /// // View matrix (camera position)
    /// let view = Matrix4::new_translation(&Vector3::new(0.0, 0.0, 0.0));
    ///
    /// // Model matrix (object position)
    /// let model = Matrix4::new_translation(&Vector3::new(400.0, 300.0, 0.0));
    ///
    /// // Combine into MVP matrix
    /// let mvp = projection * view * model;
    /// ```
    ///
    /// # See Also
    ///
    /// * [`new_perspective`](Self::new_perspective) - For perspective projection with depth
    /// * [`look_at_rh`](Self::look_at_rh) - For creating view matrices
    /// * [`Orthographic3`](crate::geometry::Orthographic3) - The underlying orthographic projection type
    #[inline]
    pub fn new_orthographic(left: T, right: T, bottom: T, top: T, znear: T, zfar: T) -> Self {
        Orthographic3::new(left, right, bottom, top, znear, zfar).into_inner()
    }

    /// Creates a new homogeneous matrix for a perspective projection.
    ///
    /// A **perspective projection** simulates how the human eye sees the world: objects
    /// farther away appear smaller, and parallel lines converge toward a vanishing point
    /// on the horizon. This creates realistic depth perception in 3D scenes.
    ///
    /// This is the standard projection for:
    /// - 3D games and simulations
    /// - Virtual reality applications
    /// - Realistic 3D rendering
    /// - Any application requiring depth perception
    ///
    /// The projection is defined by a viewing frustum - a pyramid shape with the tip at
    /// the camera position. The `fovy` parameter controls the "field of view" (how much
    /// you can see), similar to a camera's zoom lens.
    ///
    /// # Arguments
    ///
    /// * `aspect` - Aspect ratio (width / height) of the viewport
    /// * `fovy` - Vertical field of view angle in radians (larger = wider view)
    /// * `znear` - Near clipping plane distance (must be > 0)
    /// * `zfar` - Far clipping plane distance (must be > znear)
    ///
    /// # Examples
    ///
    /// ## Basic perspective projection
    ///
    /// ```
    /// use nalgebra::Matrix4;
    /// use std::f32::consts::PI;
    ///
    /// // Create a perspective projection for a 16:9 display
    /// let aspect_ratio = 16.0 / 9.0;
    /// let fov = PI / 4.0;  // 45 degrees vertical FOV
    /// let projection = Matrix4::new_perspective(
    ///     aspect_ratio,
    ///     fov,
    ///     0.1,    // near plane at 0.1 units
    ///     100.0   // far plane at 100 units
    /// );
    /// ```
    ///
    /// ## Wide field of view (for a dramatic effect)
    ///
    /// ```
    /// use nalgebra::Matrix4;
    /// use std::f32::consts::PI;
    ///
    /// // Wide FOV for a "fish-eye" effect
    /// let projection = Matrix4::new_perspective(
    ///     16.0 / 9.0,
    ///     PI / 2.0,  // 90 degrees - very wide view
    ///     0.1,
    ///     1000.0
    /// );
    /// ```
    ///
    /// ## First-person camera projection
    ///
    /// ```
    /// use nalgebra::Matrix4;
    ///
    /// // Typical first-person game camera
    /// let aspect = 1920.0 / 1080.0;
    /// let fov = 75.0_f32.to_radians();  // 75 degrees is common for FPS games
    /// let projection = Matrix4::new_perspective(aspect, fov, 0.1, 500.0);
    /// ```
    ///
    /// ## Adjusting FOV for zoom effect
    ///
    /// ```
    /// use nalgebra::Matrix4;
    ///
    /// let aspect = 16.0 / 9.0;
    ///
    /// // Normal view
    /// let normal_fov = 60.0_f32.to_radians();
    /// let normal_view = Matrix4::new_perspective(aspect, normal_fov, 0.1, 100.0);
    ///
    /// // Zoomed in (narrower FOV = telephoto lens)
    /// let zoomed_fov = 30.0_f32.to_radians();
    /// let zoomed_view = Matrix4::new_perspective(aspect, zoomed_fov, 0.1, 100.0);
    /// ```
    ///
    /// ## Complete 3D scene MVP matrix
    ///
    /// ```
    /// use nalgebra::{Matrix4, Vector3, Point3};
    /// use std::f32::consts::PI;
    ///
    /// // Projection matrix (P)
    /// let projection = Matrix4::new_perspective(16.0 / 9.0, PI / 4.0, 0.1, 100.0);
    ///
    /// // View matrix (V) - camera looking at origin from position
    /// let eye = Point3::new(5.0, 5.0, 5.0);
    /// let target = Point3::new(0.0, 0.0, 0.0);
    /// let up = Vector3::new(0.0, 1.0, 0.0);
    /// let view = Matrix4::look_at_rh(&eye, &target, &up);
    ///
    /// // Model matrix (M) - object at origin
    /// let model = Matrix4::identity();
    ///
    /// // Combine into MVP matrix for rendering
    /// let mvp = projection * view * model;
    /// ```
    ///
    /// # See Also
    ///
    /// * [`new_orthographic`](Self::new_orthographic) - For parallel projection without perspective
    /// * [`look_at_rh`](Self::look_at_rh) - For creating view matrices
    /// * [`Perspective3`](crate::geometry::Perspective3) - The underlying perspective projection type
    #[inline]
    pub fn new_perspective(aspect: T, fovy: T, znear: T, zfar: T) -> Self {
        Perspective3::new(aspect, fovy, znear, zfar).into_inner()
    }

    /// Creates an isometry that corresponds to the local frame of an observer standing at the
    /// point `eye` and looking toward `target`.
    ///
    /// This function creates a **local coordinate frame** (or observer frame) for a viewer.
    /// It's useful when you want to orient an object to face toward a specific point,
    /// rather than creating a camera view matrix.
    ///
    /// The key difference from `look_at_rh`:
    /// - `face_towards` maps the view direction to the **positive Z-axis** (object's forward)
    /// - `look_at_rh` maps the view direction to the **negative Z-axis** (camera looks down -Z)
    ///
    /// Use this for:
    /// - Making characters or objects face a target
    /// - Orienting billboards toward the camera
    /// - Setting up directional lights to point at a target
    ///
    /// # Arguments
    ///
    /// * `eye` - The observer's position (where they are standing)
    /// * `target` - The point to look toward
    /// * `up` - The "up" direction (usually the world's Y-axis: `Vector3::y()`)
    ///
    /// # Examples
    ///
    /// ## Making a character face a target
    ///
    /// ```
    /// use nalgebra::{Matrix4, Point3, Vector3};
    ///
    /// // Character at origin looking at an enemy
    /// let character_pos = Point3::new(0.0, 0.0, 0.0);
    /// let enemy_pos = Point3::new(5.0, 0.0, 3.0);
    /// let up = Vector3::new(0.0, 1.0, 0.0);
    ///
    /// let character_transform = Matrix4::face_towards(&character_pos, &enemy_pos, &up);
    /// ```
    ///
    /// ## Billboard always facing camera
    ///
    /// ```
    /// use nalgebra::{Matrix4, Point3, Vector3};
    ///
    /// // Billboard position
    /// let billboard_pos = Point3::new(10.0, 2.0, 5.0);
    ///
    /// // Camera position
    /// let camera_pos = Point3::new(0.0, 1.0, 0.0);
    ///
    /// // Make billboard face the camera
    /// let up = Vector3::new(0.0, 1.0, 0.0);
    /// let billboard_transform = Matrix4::face_towards(&billboard_pos, &camera_pos, &up);
    /// ```
    ///
    /// ## Turret aiming at a target
    ///
    /// ```
    /// use nalgebra::{Matrix4, Point3, Vector3};
    ///
    /// // Turret base position
    /// let turret_pos = Point3::new(0.0, 1.0, 0.0);
    ///
    /// // Target to aim at
    /// let target_pos = Point3::new(15.0, 3.0, -8.0);
    ///
    /// // Orient turret toward target
    /// let up = Vector3::new(0.0, 1.0, 0.0);
    /// let turret_orientation = Matrix4::face_towards(&turret_pos, &target_pos, &up);
    /// ```
    ///
    /// # See Also
    ///
    /// * [`look_at_rh`](Self::look_at_rh) - For camera view matrices (inverted orientation)
    /// * [`look_at_lh`](Self::look_at_lh) - For left-handed camera view matrices
    #[inline]
    pub fn face_towards(eye: &Point3<T>, target: &Point3<T>, up: &Vector3<T>) -> Self {
        IsometryMatrix3::face_towards(eye, target, up).to_homogeneous()
    }

    /// Deprecated: Use [`Matrix4::face_towards`] instead.
    #[deprecated(note = "renamed to `face_towards`")]
    pub fn new_observer_frame(eye: &Point3<T>, target: &Point3<T>, up: &Vector3<T>) -> Self {
        Matrix4::face_towards(eye, target, up)
    }

    /// Builds a right-handed look-at view matrix.
    ///
    /// This function creates a **view matrix** for a camera positioned at `eye` looking toward
    /// `target`. This is one of the most important functions for setting up 3D cameras.
    ///
    /// **Right-handed** means the camera looks down the **negative Z-axis** (into the screen),
    /// with X pointing right and Y pointing up. This is the standard for OpenGL and most
    /// 3D graphics applications.
    ///
    /// A view matrix transforms world coordinates into camera/eye space, where the camera
    /// is at the origin looking down -Z. This is the "V" in the MVP (Model-View-Projection)
    /// matrix chain used in 3D rendering.
    ///
    /// # Arguments
    ///
    /// * `eye` - Camera position in world space
    /// * `target` - Point the camera is looking at
    /// * `up` - The "up" direction (usually world Y-axis: `Vector3::y()`)
    ///
    /// # Examples
    ///
    /// ## Basic camera setup
    ///
    /// ```
    /// use nalgebra::{Matrix4, Point3, Vector3};
    ///
    /// // Camera at (5, 5, 5) looking at the origin
    /// let eye = Point3::new(5.0, 5.0, 5.0);
    /// let target = Point3::new(0.0, 0.0, 0.0);
    /// let up = Vector3::new(0.0, 1.0, 0.0);
    ///
    /// let view_matrix = Matrix4::look_at_rh(&eye, &target, &up);
    /// ```
    ///
    /// ## First-person camera
    ///
    /// ```
    /// use nalgebra::{Matrix4, Point3, Vector3};
    ///
    /// // Player position and look-at point
    /// let player_pos = Point3::new(0.0, 1.8, 0.0);  // Eye height ~1.8m
    /// let look_target = Point3::new(5.0, 1.8, -10.0);  // Looking forward
    /// let up = Vector3::new(0.0, 1.0, 0.0);
    ///
    /// let view = Matrix4::look_at_rh(&player_pos, &look_target, &up);
    /// ```
    ///
    /// ## Orbital camera around an object
    ///
    /// ```
    /// use nalgebra::{Matrix4, Point3, Vector3};
    ///
    /// // Object at origin, camera orbiting around it
    /// let distance = 10.0;
    /// let angle = 0.5_f32;  // Orbit angle
    ///
    /// let camera_pos = Point3::new(
    ///     angle.cos() * distance,
    ///     5.0,
    ///     angle.sin() * distance
    /// );
    /// let object_pos = Point3::new(0.0, 0.0, 0.0);
    /// let up = Vector3::new(0.0, 1.0, 0.0);
    ///
    /// let view = Matrix4::look_at_rh(&camera_pos, &object_pos, &up);
    /// ```
    ///
    /// ## Complete MVP matrix setup
    ///
    /// ```
    /// use nalgebra::{Matrix4, Point3, Vector3};
    /// use std::f32::consts::PI;
    ///
    /// // Model matrix - object at origin
    /// let model = Matrix4::identity();
    ///
    /// // View matrix - camera setup
    /// let eye = Point3::new(0.0, 2.0, 5.0);
    /// let target = Point3::new(0.0, 0.0, 0.0);
    /// let up = Vector3::new(0.0, 1.0, 0.0);
    /// let view = Matrix4::look_at_rh(&eye, &target, &up);
    ///
    /// // Projection matrix
    /// let projection = Matrix4::new_perspective(16.0 / 9.0, PI / 4.0, 0.1, 100.0);
    ///
    /// // Final MVP matrix
    /// let mvp = projection * view * model;
    /// ```
    ///
    /// ## Following camera for a game
    ///
    /// ```
    /// use nalgebra::{Matrix4, Point3, Vector3};
    ///
    /// // Player position
    /// let player_pos = Point3::new(10.0, 0.0, 5.0);
    ///
    /// // Camera offset behind and above player
    /// let camera_offset = Vector3::new(0.0, 3.0, 5.0);
    /// let camera_pos = player_pos + camera_offset;
    ///
    /// let up = Vector3::new(0.0, 1.0, 0.0);
    /// let view = Matrix4::look_at_rh(&camera_pos, &player_pos, &up);
    /// ```
    ///
    /// # See Also
    ///
    /// * [`look_at_lh`](Self::look_at_lh) - For left-handed coordinate systems
    /// * [`face_towards`](Self::face_towards) - For orienting objects (not cameras)
    /// * [`new_perspective`](Self::new_perspective) - For projection matrices
    /// * [`new_orthographic`](Self::new_orthographic) - For orthographic projection matrices
    #[inline]
    pub fn look_at_rh(eye: &Point3<T>, target: &Point3<T>, up: &Vector3<T>) -> Self {
        IsometryMatrix3::look_at_rh(eye, target, up).to_homogeneous()
    }

    /// Builds a left-handed look-at view matrix.
    ///
    /// This function creates a **view matrix** for a left-handed coordinate system.
    /// In left-handed systems, the camera looks down the **positive Z-axis** (into the screen),
    /// which is used in some graphics APIs like DirectX (though modern DirectX also supports
    /// right-handed).
    ///
    /// Most applications should use `look_at_rh` (right-handed) instead, which is the
    /// OpenGL standard. Only use this if you specifically need left-handed coordinates.
    ///
    /// # Arguments
    ///
    /// * `eye` - Camera position in world space
    /// * `target` - Point the camera is looking at
    /// * `up` - The "up" direction (usually world Y-axis: `Vector3::y()`)
    ///
    /// # Examples
    ///
    /// ## Basic left-handed camera
    ///
    /// ```
    /// use nalgebra::{Matrix4, Point3, Vector3};
    ///
    /// // Camera at (0, 0, -5) looking at origin in left-handed space
    /// let eye = Point3::new(0.0, 0.0, -5.0);
    /// let target = Point3::new(0.0, 0.0, 0.0);
    /// let up = Vector3::new(0.0, 1.0, 0.0);
    ///
    /// let view_matrix = Matrix4::look_at_lh(&eye, &target, &up);
    /// ```
    ///
    /// ## DirectX-style camera setup
    ///
    /// ```
    /// use nalgebra::{Matrix4, Point3, Vector3};
    ///
    /// // For left-handed coordinate system (some DirectX applications)
    /// let camera_pos = Point3::new(0.0, 5.0, -10.0);
    /// let look_at = Point3::new(0.0, 0.0, 0.0);
    /// let up = Vector3::new(0.0, 1.0, 0.0);
    ///
    /// let view = Matrix4::look_at_lh(&camera_pos, &look_at, &up);
    /// ```
    ///
    /// # See Also
    ///
    /// * [`look_at_rh`](Self::look_at_rh) - For right-handed systems (OpenGL standard, recommended)
    /// * [`face_towards`](Self::face_towards) - For orienting objects toward a target
    /// * [`new_perspective`](Self::new_perspective) - For projection matrices
    #[inline]
    pub fn look_at_lh(eye: &Point3<T>, target: &Point3<T>, up: &Vector3<T>) -> Self {
        IsometryMatrix3::look_at_lh(eye, target, up).to_homogeneous()
    }
}

/// # Append/prepend translation and scaling
impl<T: Scalar + Zero + One + ClosedMulAssign + ClosedAddAssign, D: DimName, S: Storage<T, D, D>>
    SquareMatrix<T, D, S>
{
    /// Computes the transformation equal to `self` followed by an uniform scaling factor.
    ///
    /// This function **appends** (adds to the end) a uniform scaling transformation to an
    /// existing transformation matrix. The result applies your original transformation first,
    /// then applies the scaling.
    ///
    /// **Important**: In computer graphics, transformation order matters! `T * S` (transform then
    /// scale) produces different results than `S * T` (scale then transform).
    ///
    /// This creates a new matrix without modifying the original. For in-place modification,
    /// use [`append_scaling_mut`](Self::append_scaling_mut).
    ///
    /// # Arguments
    ///
    /// * `scaling` - The uniform scaling factor to append
    ///
    /// # Examples
    ///
    /// ## Basic append scaling
    ///
    /// ```
    /// use nalgebra::{Matrix4, Vector3};
    ///
    /// // Start with a translation
    /// let translation = Matrix4::new_translation(&Vector3::new(5.0, 0.0, 0.0));
    ///
    /// // Append a scaling
    /// let transform = translation.append_scaling(2.0);
    ///
    /// // This is equivalent to: translate first, then scale
    /// ```
    ///
    /// ## Combining multiple transformations
    ///
    /// ```
    /// use nalgebra::{Matrix4, Vector3};
    ///
    /// // Create a complex transformation chain
    /// let transform = Matrix4::new_rotation(Vector3::new(0.0, 1.0, 0.0))
    ///     .append_scaling(2.0)
    ///     .append_translation(&Vector3::new(10.0, 0.0, 0.0));
    /// ```
    ///
    /// # See Also
    ///
    /// * [`prepend_scaling`](Self::prepend_scaling) - To add scaling before the transformation
    /// * [`append_scaling_mut`](Self::append_scaling_mut) - For in-place modification
    /// * [`append_nonuniform_scaling`](Self::append_nonuniform_scaling) - For non-uniform scaling
    #[inline]
    #[must_use = "Did you mean to use append_scaling_mut()?"]
    pub fn append_scaling(&self, scaling: T) -> OMatrix<T, D, D>
    where
        D: DimNameSub<U1>,
        DefaultAllocator: Allocator<D, D>,
    {
        let mut res = self.clone_owned();
        res.append_scaling_mut(scaling);
        res
    }

    /// Computes the transformation equal to an uniform scaling factor followed by `self`.
    ///
    /// This function **prepends** (adds to the beginning) a uniform scaling transformation
    /// before an existing transformation matrix. The result applies the scaling first,
    /// then applies your original transformation.
    ///
    /// **Important**: The order matters! Scaling before rotation gives different results
    /// than rotating before scaling.
    ///
    /// This creates a new matrix without modifying the original. For in-place modification,
    /// use [`prepend_scaling_mut`](Self::prepend_scaling_mut).
    ///
    /// # Arguments
    ///
    /// * `scaling` - The uniform scaling factor to prepend
    ///
    /// # Examples
    ///
    /// ## Basic prepend scaling
    ///
    /// ```
    /// use nalgebra::{Matrix4, Vector3};
    ///
    /// // Start with a translation
    /// let translation = Matrix4::new_translation(&Vector3::new(5.0, 0.0, 0.0));
    ///
    /// // Prepend a scaling (scale first, then translate)
    /// let transform = translation.prepend_scaling(2.0);
    /// ```
    ///
    /// # See Also
    ///
    /// * [`append_scaling`](Self::append_scaling) - To add scaling after the transformation
    /// * [`prepend_scaling_mut`](Self::prepend_scaling_mut) - For in-place modification
    /// * [`prepend_nonuniform_scaling`](Self::prepend_nonuniform_scaling) - For non-uniform scaling
    #[inline]
    #[must_use = "Did you mean to use prepend_scaling_mut()?"]
    pub fn prepend_scaling(&self, scaling: T) -> OMatrix<T, D, D>
    where
        D: DimNameSub<U1>,
        DefaultAllocator: Allocator<D, D>,
    {
        let mut res = self.clone_owned();
        res.prepend_scaling_mut(scaling);
        res
    }

    /// Computes the transformation equal to `self` followed by a non-uniform scaling factor.
    ///
    /// Appends a non-uniform scaling (different scale factors per axis) to an existing
    /// transformation. The original transformation is applied first, then the scaling.
    ///
    /// # See Also
    ///
    /// * [`append_scaling`](Self::append_scaling) - For uniform scaling
    /// * [`prepend_nonuniform_scaling`](Self::prepend_nonuniform_scaling) - To scale before the transformation
    #[inline]
    #[must_use = "Did you mean to use append_nonuniform_scaling_mut()?"]
    pub fn append_nonuniform_scaling<SB>(
        &self,
        scaling: &Vector<T, DimNameDiff<D, U1>, SB>,
    ) -> OMatrix<T, D, D>
    where
        D: DimNameSub<U1>,
        SB: Storage<T, DimNameDiff<D, U1>>,
        DefaultAllocator: Allocator<D, D>,
    {
        let mut res = self.clone_owned();
        res.append_nonuniform_scaling_mut(scaling);
        res
    }

    /// Computes the transformation equal to a non-uniform scaling factor followed by `self`.
    ///
    /// Prepends a non-uniform scaling (different scale factors per axis) before an existing
    /// transformation. The scaling is applied first, then the original transformation.
    ///
    /// # See Also
    ///
    /// * [`prepend_scaling`](Self::prepend_scaling) - For uniform scaling
    /// * [`append_nonuniform_scaling`](Self::append_nonuniform_scaling) - To scale after the transformation
    #[inline]
    #[must_use = "Did you mean to use prepend_nonuniform_scaling_mut()?"]
    pub fn prepend_nonuniform_scaling<SB>(
        &self,
        scaling: &Vector<T, DimNameDiff<D, U1>, SB>,
    ) -> OMatrix<T, D, D>
    where
        D: DimNameSub<U1>,
        SB: Storage<T, DimNameDiff<D, U1>>,
        DefaultAllocator: Allocator<D, D>,
    {
        let mut res = self.clone_owned();
        res.prepend_nonuniform_scaling_mut(scaling);
        res
    }

    /// Computes the transformation equal to `self` followed by a translation.
    ///
    /// Appends a translation to an existing transformation. The original transformation
    /// is applied first, then the translation.
    ///
    /// # Examples
    ///
    /// ```
    /// use nalgebra::{Matrix4, Vector3};
    ///
    /// // Rotate then translate
    /// let rotation = Matrix4::new_rotation(Vector3::new(0.0, 1.0, 0.0));
    /// let transform = rotation.append_translation(&Vector3::new(5.0, 0.0, 0.0));
    /// ```
    ///
    /// # See Also
    ///
    /// * [`prepend_translation`](Self::prepend_translation) - To translate before the transformation
    /// * [`append_translation_mut`](Self::append_translation_mut) - For in-place modification
    #[inline]
    #[must_use = "Did you mean to use append_translation_mut()?"]
    pub fn append_translation<SB>(
        &self,
        shift: &Vector<T, DimNameDiff<D, U1>, SB>,
    ) -> OMatrix<T, D, D>
    where
        D: DimNameSub<U1>,
        SB: Storage<T, DimNameDiff<D, U1>>,
        DefaultAllocator: Allocator<D, D>,
    {
        let mut res = self.clone_owned();
        res.append_translation_mut(shift);
        res
    }

    /// Computes the transformation equal to a translation followed by `self`.
    ///
    /// Prepends a translation before an existing transformation. The translation
    /// is applied first, then the original transformation.
    ///
    /// # Examples
    ///
    /// ```
    /// use nalgebra::{Matrix4, Vector3};
    ///
    /// // Translate then rotate
    /// let rotation = Matrix4::new_rotation(Vector3::new(0.0, 1.0, 0.0));
    /// let transform = rotation.prepend_translation(&Vector3::new(5.0, 0.0, 0.0));
    /// ```
    ///
    /// # See Also
    ///
    /// * [`append_translation`](Self::append_translation) - To translate after the transformation
    /// * [`prepend_translation_mut`](Self::prepend_translation_mut) - For in-place modification
    #[inline]
    #[must_use = "Did you mean to use prepend_translation_mut()?"]
    pub fn prepend_translation<SB>(
        &self,
        shift: &Vector<T, DimNameDiff<D, U1>, SB>,
    ) -> OMatrix<T, D, D>
    where
        D: DimNameSub<U1>,
        SB: Storage<T, DimNameDiff<D, U1>>,
        DefaultAllocator: Allocator<D, D> + Allocator<DimNameDiff<D, U1>>,
    {
        let mut res = self.clone_owned();
        res.prepend_translation_mut(shift);
        res
    }

    /// Computes in-place the transformation equal to `self` followed by an uniform scaling factor.
    ///
    /// Modifies this matrix in place by appending a uniform scaling transformation.
    ///
    /// # See Also
    ///
    /// * [`append_scaling`](Self::append_scaling) - Non-mutating version
    #[inline]
    pub fn append_scaling_mut(&mut self, scaling: T)
    where
        S: StorageMut<T, D, D>,
        D: DimNameSub<U1>,
    {
        let mut to_scale = self.rows_generic_mut(0, DimNameDiff::<D, U1>::name());
        to_scale *= scaling;
    }

    /// Computes in-place the transformation equal to an uniform scaling factor followed by `self`.
    ///
    /// Modifies this matrix in place by prepending a uniform scaling transformation.
    ///
    /// # See Also
    ///
    /// * [`prepend_scaling`](Self::prepend_scaling) - Non-mutating version
    #[inline]
    pub fn prepend_scaling_mut(&mut self, scaling: T)
    where
        S: StorageMut<T, D, D>,
        D: DimNameSub<U1>,
    {
        let mut to_scale = self.columns_generic_mut(0, DimNameDiff::<D, U1>::name());
        to_scale *= scaling;
    }

    /// Computes in-place the transformation equal to `self` followed by a non-uniform scaling factor.
    ///
    /// Modifies this matrix in place by appending a non-uniform scaling transformation.
    ///
    /// # See Also
    ///
    /// * [`append_nonuniform_scaling`](Self::append_nonuniform_scaling) - Non-mutating version
    #[inline]
    pub fn append_nonuniform_scaling_mut<SB>(&mut self, scaling: &Vector<T, DimNameDiff<D, U1>, SB>)
    where
        S: StorageMut<T, D, D>,
        D: DimNameSub<U1>,
        SB: Storage<T, DimNameDiff<D, U1>>,
    {
        for i in 0..scaling.len() {
            let mut to_scale = self.fixed_rows_mut::<1>(i);
            to_scale *= scaling[i].clone();
        }
    }

    /// Computes in-place the transformation equal to a non-uniform scaling factor followed by `self`.
    ///
    /// Modifies this matrix in place by prepending a non-uniform scaling transformation.
    ///
    /// # See Also
    ///
    /// * [`prepend_nonuniform_scaling`](Self::prepend_nonuniform_scaling) - Non-mutating version
    #[inline]
    pub fn prepend_nonuniform_scaling_mut<SB>(
        &mut self,
        scaling: &Vector<T, DimNameDiff<D, U1>, SB>,
    ) where
        S: StorageMut<T, D, D>,
        D: DimNameSub<U1>,
        SB: Storage<T, DimNameDiff<D, U1>>,
    {
        for i in 0..scaling.len() {
            let mut to_scale = self.fixed_columns_mut::<1>(i);
            to_scale *= scaling[i].clone();
        }
    }

    /// Computes the transformation equal to `self` followed by a translation.
    ///
    /// Modifies this matrix in place by appending a translation transformation.
    ///
    /// # See Also
    ///
    /// * [`append_translation`](Self::append_translation) - Non-mutating version
    #[inline]
    pub fn append_translation_mut<SB>(&mut self, shift: &Vector<T, DimNameDiff<D, U1>, SB>)
    where
        S: StorageMut<T, D, D>,
        D: DimNameSub<U1>,
        SB: Storage<T, DimNameDiff<D, U1>>,
    {
        for i in 0..D::DIM {
            for j in 0..D::DIM - 1 {
                let add = shift[j].clone() * self[(D::DIM - 1, i)].clone();
                self[(j, i)] += add;
            }
        }
    }

    /// Computes the transformation equal to a translation followed by `self`.
    ///
    /// Modifies this matrix in place by prepending a translation transformation.
    ///
    /// # See Also
    ///
    /// * [`prepend_translation`](Self::prepend_translation) - Non-mutating version
    #[inline]
    pub fn prepend_translation_mut<SB>(&mut self, shift: &Vector<T, DimNameDiff<D, U1>, SB>)
    where
        D: DimNameSub<U1>,
        S: StorageMut<T, D, D>,
        SB: Storage<T, DimNameDiff<D, U1>>,
        DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
    {
        let scale = self
            .generic_view((D::DIM - 1, 0), (Const::<1>, DimNameDiff::<D, U1>::name()))
            .tr_dot(shift);
        let post_translation = self.generic_view(
            (0, 0),
            (DimNameDiff::<D, U1>::name(), DimNameDiff::<D, U1>::name()),
        ) * shift;

        self[(D::DIM - 1, D::DIM - 1)] += scale;

        let mut translation =
            self.generic_view_mut((0, D::DIM - 1), (DimNameDiff::<D, U1>::name(), Const::<1>));
        translation += post_translation;
    }
}

/// # Transformation of vectors and points
impl<T: RealField, D: DimNameSub<U1>, S: Storage<T, D, D>> SquareMatrix<T, D, S>
where
    DefaultAllocator: Allocator<D, D>
        + Allocator<DimNameDiff<D, U1>>
        + Allocator<DimNameDiff<D, U1>, DimNameDiff<D, U1>>,
{
    /// Transforms the given vector, assuming the matrix `self` uses homogeneous coordinates.
    ///
    /// This function applies the transformation matrix to a **vector** (as opposed to a point).
    /// In computer graphics, vectors and points are treated differently:
    /// - **Vectors** represent directions and magnitudes (like velocity, normals)
    /// - **Points** represent positions in space
    ///
    /// When transforming a vector with a homogeneous matrix, the translation component
    /// is ignored - only rotation and scaling are applied. This is because vectors don't
    /// have a position, only a direction.
    ///
    /// The function handles perspective division if needed (for projective transformations).
    ///
    /// # Examples
    ///
    /// ## Transforming a direction vector
    ///
    /// ```
    /// use nalgebra::{Matrix4, Vector3};
    ///
    /// // Rotation matrix
    /// let rotation = Matrix4::new_rotation(Vector3::new(0.0, std::f32::consts::FRAC_PI_2, 0.0));
    ///
    /// // Direction vector pointing right
    /// let direction = Vector3::new(1.0, 0.0, 0.0);
    ///
    /// // Rotate the direction - translation is ignored for vectors
    /// let rotated_dir = rotation.transform_vector(&direction);
    ///
    /// // Direction now points backward
    /// assert!((rotated_dir.z - (-1.0)).abs() < 1e-6);
    /// ```
    ///
    /// ## Transforming a normal vector
    ///
    /// ```
    /// use nalgebra::{Matrix4, Vector3};
    ///
    /// // Scaling transformation
    /// let scale = Matrix4::new_scaling(2.0);
    ///
    /// // Surface normal (unit vector)
    /// let normal = Vector3::new(0.0, 1.0, 0.0);
    ///
    /// // Transform the normal
    /// let transformed_normal = scale.transform_vector(&normal);
    /// ```
    ///
    /// # See Also
    ///
    /// * [`transform_point`](Self::transform_point) - For transforming points (includes translation)
    #[inline]
    pub fn transform_vector(
        &self,
        v: &OVector<T, DimNameDiff<D, U1>>,
    ) -> OVector<T, DimNameDiff<D, U1>> {
        let transform = self.generic_view(
            (0, 0),
            (DimNameDiff::<D, U1>::name(), DimNameDiff::<D, U1>::name()),
        );
        let normalizer =
            self.generic_view((D::DIM - 1, 0), (Const::<1>, DimNameDiff::<D, U1>::name()));
        let n = normalizer.tr_dot(v);

        if !n.is_zero() {
            return transform * (v / n);
        }

        transform * v
    }
}

impl<T: RealField, S: Storage<T, Const<3>, Const<3>>> SquareMatrix<T, Const<3>, S> {
    /// Transforms the given point, assuming the matrix `self` uses homogeneous coordinates.
    ///
    /// This function applies a 2D transformation matrix (3x3 homogeneous) to a **point**.
    /// Unlike `transform_vector`, this includes the translation component of the transformation.
    ///
    /// Points represent positions in space, so they are affected by all transformations:
    /// translation, rotation, and scaling. This is the function you use to move objects
    /// around in 2D space.
    ///
    /// The function automatically handles perspective division for projective transformations.
    ///
    /// # Arguments
    ///
    /// * `pt` - The 2D point to transform
    ///
    /// # Examples
    ///
    /// ## Basic point transformation
    ///
    /// ```
    /// use nalgebra::{Matrix3, Point2, Vector2};
    ///
    /// // Translation matrix
    /// let translation = Matrix3::new_translation(&Vector2::new(5.0, 3.0));
    ///
    /// // Transform a point
    /// let point = Point2::new(1.0, 2.0);
    /// let transformed = translation.transform_point(&point);
    ///
    /// assert_eq!(transformed, Point2::new(6.0, 5.0));
    /// ```
    ///
    /// ## Combined transformation
    ///
    /// ```
    /// use nalgebra::{Matrix3, Point2, Vector2};
    ///
    /// // Rotate then translate
    /// let rotation = Matrix3::new_rotation(std::f32::consts::FRAC_PI_2);
    /// let translation = Matrix3::new_translation(&Vector2::new(10.0, 0.0));
    /// let transform = translation * rotation;
    ///
    /// let point = Point2::new(1.0, 0.0);
    /// let transformed = transform.transform_point(&point);
    /// ```
    ///
    /// # See Also
    ///
    /// * [`transform_vector`](Self::transform_vector) - For transforming vectors (ignores translation)
    /// * [`Matrix4::transform_point`] - For 3D point transformation
    #[inline]
    pub fn transform_point(&self, pt: &Point<T, 2>) -> Point<T, 2> {
        let transform = self.fixed_view::<2, 2>(0, 0);
        let translation = self.fixed_view::<2, 1>(0, 2);
        let normalizer = self.fixed_view::<1, 2>(2, 0);
        let n = normalizer.tr_dot(&pt.coords) + unsafe { self.get_unchecked((2, 2)).clone() };

        if !n.is_zero() {
            (transform * pt + translation) / n
        } else {
            transform * pt + translation
        }
    }
}

impl<T: RealField, S: Storage<T, Const<4>, Const<4>>> SquareMatrix<T, Const<4>, S> {
    /// Transforms the given point, assuming the matrix `self` uses homogeneous coordinates.
    ///
    /// This function applies a 3D transformation matrix (4x4 homogeneous) to a **point**.
    /// This is one of the most frequently used functions in 3D graphics programming.
    ///
    /// Points represent positions in 3D space, so they are affected by all transformations:
    /// translation, rotation, and scaling. This is what you use to position objects, vertices,
    /// and cameras in your 3D scene.
    ///
    /// The function automatically handles perspective division, which is essential for
    /// projecting 3D points onto a 2D screen using perspective projection matrices.
    ///
    /// # Arguments
    ///
    /// * `pt` - The 3D point to transform
    ///
    /// # Examples
    ///
    /// ## Basic point transformation
    ///
    /// ```
    /// use nalgebra::{Matrix4, Point3, Vector3};
    ///
    /// // Translation matrix
    /// let translation = Matrix4::new_translation(&Vector3::new(5.0, 3.0, -2.0));
    ///
    /// // Transform a point
    /// let point = Point3::new(1.0, 2.0, 3.0);
    /// let transformed = translation.transform_point(&point);
    ///
    /// assert_eq!(transformed, Point3::new(6.0, 5.0, 1.0));
    /// ```
    ///
    /// ## MVP transformation for rendering
    ///
    /// ```
    /// use nalgebra::{Matrix4, Point3, Vector3};
    /// use std::f32::consts::PI;
    ///
    /// // Build the MVP matrix
    /// let model = Matrix4::new_translation(&Vector3::new(0.0, 0.0, -5.0));
    /// let view = Matrix4::look_at_rh(
    ///     &Point3::new(0.0, 2.0, 5.0),
    ///     &Point3::new(0.0, 0.0, 0.0),
    ///     &Vector3::new(0.0, 1.0, 0.0)
    /// );
    /// let projection = Matrix4::new_perspective(16.0 / 9.0, PI / 4.0, 0.1, 100.0);
    /// let mvp = projection * view * model;
    ///
    /// // Transform a 3D vertex to screen space
    /// let vertex = Point3::new(1.0, 1.0, 0.0);
    /// let screen_pos = mvp.transform_point(&vertex);
    /// ```
    ///
    /// ## Rotating an object around a point
    ///
    /// ```
    /// use nalgebra::{Matrix4, Point3, Vector3};
    ///
    /// // Rotate around Y-axis
    /// let rotation = Matrix4::new_rotation(Vector3::new(0.0, 1.0, 0.0));
    ///
    /// // Transform multiple vertices of an object
    /// let vertices = vec![
    ///     Point3::new(1.0, 0.0, 0.0),
    ///     Point3::new(0.0, 1.0, 0.0),
    ///     Point3::new(0.0, 0.0, 1.0),
    /// ];
    ///
    /// let rotated_vertices: Vec<_> = vertices
    ///     .iter()
    ///     .map(|v| rotation.transform_point(v))
    ///     .collect();
    /// ```
    ///
    /// # See Also
    ///
    /// * [`transform_vector`](Self::transform_vector) - For transforming vectors (ignores translation)
    /// * [`Matrix3::transform_point`] - For 2D point transformation
    /// * [`new_perspective`](Self::new_perspective) - For creating projection matrices
    /// * [`look_at_rh`](Self::look_at_rh) - For creating view matrices
    #[inline]
    pub fn transform_point(&self, pt: &Point<T, 3>) -> Point<T, 3> {
        let transform = self.fixed_view::<3, 3>(0, 0);
        let translation = self.fixed_view::<3, 1>(0, 3);
        let normalizer = self.fixed_view::<1, 3>(3, 0);
        let n = normalizer.tr_dot(&pt.coords) + unsafe { self.get_unchecked((3, 3)).clone() };

        if !n.is_zero() {
            (transform * pt + translation) / n
        } else {
            transform * pt + translation
        }
    }
}
