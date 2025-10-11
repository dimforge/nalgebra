use crate::{Isometry2, Isometry3, IsometryMatrix2, IsometryMatrix3, RealField, SimdRealField};

/// # Interpolation
impl<T: SimdRealField> Isometry3<T> {
    /// Interpolates between two 3D isometries using linear interpolation for translation
    /// and spherical linear interpolation for rotation.
    ///
    /// An isometry represents a rigid body transformation combining both translation and rotation.
    /// This function smoothly interpolates both components simultaneously:
    /// - **Translation**: Uses LERP (Linear Interpolation) to move in a straight line
    /// - **Rotation**: Uses SLERP (Spherical Linear Interpolation) to rotate smoothly
    ///
    /// This combined interpolation is essential for:
    /// - Animating moving and rotating objects (characters, vehicles, projectiles)
    /// - Smooth camera movements that both travel and change viewing direction
    /// - Physics simulations with rigid body motion
    /// - Robotic arm and mechanism control
    /// - Any scenario where an object needs to smoothly transition between two poses
    ///
    /// The interpolation maintains the rigid body property throughout - no scaling,
    /// shearing, or other deformations occur, only smooth motion and rotation.
    ///
    /// # Parameters
    ///
    /// * `self`: The starting isometry (pose) returned when `t = 0.0`
    /// * `other`: The target isometry (pose) returned when `t = 1.0`
    /// * `t`: The interpolation parameter, typically in the range `[0.0, 1.0]`
    ///   - `t = 0.0` returns `self`
    ///   - `t = 1.0` returns `other`
    ///   - `t = 0.5` returns the pose halfway between `self` and `other`
    ///   - Values outside `[0.0, 1.0]` will extrapolate the motion
    ///
    /// # Panics
    ///
    /// Panics if the angle between both rotations is approximately 180 degrees, because
    /// the rotation interpolation becomes undefined. Use [`Isometry3::try_lerp_slerp`]
    /// to handle this case safely.
    ///
    /// # Examples
    ///
    /// Basic interpolation of position and orientation:
    ///
    /// ```
    /// # use nalgebra::{Vector3, Translation3, Isometry3, UnitQuaternion};
    /// use std::f32::consts::PI;
    ///
    /// // Start: at position (1,2,3), rotated 45 degrees around X
    /// let start = Isometry3::from_parts(
    ///     Translation3::new(1.0, 2.0, 3.0),
    ///     UnitQuaternion::from_euler_angles(PI / 4.0, 0.0, 0.0)
    /// );
    ///
    /// // End: at position (4,8,12), rotated -180 degrees around X
    /// let end = Isometry3::from_parts(
    ///     Translation3::new(4.0, 8.0, 12.0),
    ///     UnitQuaternion::from_euler_angles(-PI, 0.0, 0.0)
    /// );
    ///
    /// // Interpolate one-third of the way
    /// let mid = start.lerp_slerp(&end, 1.0 / 3.0);
    ///
    /// // Position moves 1/3 of the way: (1,2,3) + 1/3 * (3,6,9) = (2,4,6)
    /// assert_eq!(mid.translation.vector, Vector3::new(2.0, 4.0, 6.0));
    /// // Rotation interpolates to 90 degrees
    /// assert_eq!(mid.rotation.euler_angles(), (PI / 2.0, 0.0, 0.0));
    /// ```
    ///
    /// Animating a flying projectile:
    ///
    /// ```
    /// # use nalgebra::{Isometry3, Translation3, UnitQuaternion, Vector3};
    /// use std::f32::consts::PI;
    ///
    /// // Projectile starts at origin, pointing forward
    /// let launch_pose = Isometry3::from_parts(
    ///     Translation3::new(0.0, 0.0, 0.0),
    ///     UnitQuaternion::identity()
    /// );
    ///
    /// // Projectile ends 10 units away, rotated to point downward
    /// let impact_pose = Isometry3::from_parts(
    ///     Translation3::new(10.0, 0.0, 5.0),
    ///     UnitQuaternion::from_euler_angles(PI / 4.0, 0.0, 0.0)
    /// );
    ///
    /// // Calculate 10 frames of animation
    /// for frame in 0..=10 {
    ///     let t = frame as f32 / 10.0;
    ///     let current_pose = launch_pose.lerp_slerp(&impact_pose, t);
    ///
    ///     // Use current_pose to position and orient the projectile model
    ///     let position = current_pose.translation.vector;
    ///     let forward = current_pose.rotation * Vector3::z();
    ///     println!("Frame {}: pos={:?}, forward={:?}", frame, position, forward);
    /// }
    /// ```
    ///
    /// Smooth camera movement between two viewpoints:
    ///
    /// ```
    /// # use nalgebra::{Isometry3, Translation3, UnitQuaternion, Vector3, Point3};
    /// use std::f32::consts::PI;
    ///
    /// // Camera starts at (0,5,10) looking at origin
    /// let start_camera = Isometry3::face_towards(
    ///     &Point3::new(0.0, 5.0, 10.0),
    ///     &Point3::origin(),
    ///     &Vector3::y_axis()
    /// );
    ///
    /// // Camera ends at (10,5,0) looking at origin from the side
    /// let end_camera = Isometry3::face_towards(
    ///     &Point3::new(10.0, 5.0, 0.0),
    ///     &Point3::origin(),
    ///     &Vector3::y_axis()
    /// );
    ///
    /// // Smoothly transition camera over 2 seconds at 60 FPS
    /// let total_frames = 120;
    /// for frame in 0..=total_frames {
    ///     let t = frame as f32 / total_frames as f32;
    ///     let camera_pose = start_camera.lerp_slerp(&end_camera, t);
    ///
    ///     // Use camera_pose to update your view matrix
    ///     // let view_matrix = camera_pose.inverse().to_homogeneous();
    /// }
    /// ```
    ///
    /// Interpolating a character's pose in an animation:
    ///
    /// ```
    /// # use nalgebra::{Isometry3, Translation3, UnitQuaternion, Vector3};
    /// use std::f32::consts::PI;
    ///
    /// // Character standing upright
    /// let standing = Isometry3::from_parts(
    ///     Translation3::new(0.0, 1.0, 0.0),
    ///     UnitQuaternion::identity()
    /// );
    ///
    /// // Character crouched and moved forward
    /// let crouched = Isometry3::from_parts(
    ///     Translation3::new(0.5, 0.5, 0.0),
    ///     UnitQuaternion::from_euler_angles(PI / 6.0, 0.0, 0.0)
    /// );
    ///
    /// // Smooth transition for crouch animation
    /// let halfway = standing.lerp_slerp(&crouched, 0.5);
    /// ```
    ///
    /// # See Also
    ///
    /// * [`Isometry3::try_lerp_slerp`] - Safe version that handles 180-degree rotations
    /// * [`Isometry2::lerp_slerp`] - 2D version for planar motion
    /// * [`IsometryMatrix3::lerp_slerp`] - Matrix-based isometry interpolation
    /// * [`Rotation3::slerp`] - Rotation-only interpolation
    #[inline]
    #[must_use]
    pub fn lerp_slerp(&self, other: &Self, t: T) -> Self
    where
        T: RealField,
    {
        let tr = self
            .translation
            .vector
            .lerp(&other.translation.vector, t.clone());
        let rot = self.rotation.slerp(&other.rotation, t);
        Self::from_parts(tr.into(), rot)
    }

    /// Attempts to interpolate between two 3D isometries, returning `None` if rotations are opposite.
    ///
    /// This is the safe version of [`Isometry3::lerp_slerp`] that handles the edge case where
    /// the two rotations are approximately 180 degrees apart. In this case, the rotation
    /// interpolation is undefined (there are infinitely many shortest paths), so this function
    /// returns `None` to let you handle it appropriately.
    ///
    /// The translation component is always interpolated linearly (LERP), but the rotation
    /// component uses spherical linear interpolation (SLERP) which can fail for opposite
    /// orientations.
    ///
    /// # Parameters
    ///
    /// * `self`: The starting isometry (position and orientation)
    /// * `other`: The target isometry (position and orientation)
    /// * `t`: The interpolation parameter, typically in `[0.0, 1.0]`
    /// * `epsilon`: Threshold for detecting opposite rotations (e.g., `1.0e-6`)
    ///
    /// # Returns
    ///
    /// * `Some(isometry)` - The interpolated pose if rotation is well-defined
    /// * `None` - If the rotations are approximately 180 degrees apart
    ///
    /// # Examples
    ///
    /// Basic usage with successful interpolation:
    ///
    /// ```
    /// # use nalgebra::{Vector3, Translation3, Isometry3, UnitQuaternion};
    /// use std::f32::consts::PI;
    ///
    /// let start = Isometry3::from_parts(
    ///     Translation3::new(1.0, 2.0, 3.0),
    ///     UnitQuaternion::from_euler_angles(PI / 4.0, 0.0, 0.0)
    /// );
    ///
    /// let end = Isometry3::from_parts(
    ///     Translation3::new(4.0, 8.0, 12.0),
    ///     UnitQuaternion::from_euler_angles(-PI, 0.0, 0.0)
    /// );
    ///
    /// // Try to interpolate
    /// let result = start.try_lerp_slerp(&end, 1.0 / 3.0, 1.0e-6);
    ///
    /// if let Some(mid) = result {
    ///     assert_eq!(mid.translation.vector, Vector3::new(2.0, 4.0, 6.0));
    ///     assert_eq!(mid.rotation.euler_angles(), (PI / 2.0, 0.0, 0.0));
    /// }
    /// ```
    ///
    /// Handling the failure case gracefully:
    ///
    /// ```
    /// # use nalgebra::{Isometry3, Translation3, UnitQuaternion, Vector3};
    /// use std::f32::consts::PI;
    ///
    /// let pose1 = Isometry3::from_parts(
    ///     Translation3::new(0.0, 0.0, 0.0),
    ///     UnitQuaternion::identity()
    /// );
    ///
    /// // Opposite rotation (180 degrees)
    /// let pose2 = Isometry3::from_parts(
    ///     Translation3::new(10.0, 0.0, 0.0),
    ///     UnitQuaternion::from_axis_angle(&Vector3::y_axis(), PI)
    /// );
    ///
    /// match pose1.try_lerp_slerp(&pose2, 0.5, 1.0e-6) {
    ///     Some(interpolated) => {
    ///         println!("Successfully interpolated to {:?}", interpolated);
    ///     }
    ///     None => {
    ///         println!("Cannot interpolate - using fallback strategy");
    ///         // Could choose an arbitrary rotation path, or use the endpoint
    ///         let fallback = pose2.clone();
    ///     }
    /// }
    /// ```
    ///
    /// Robust animation system with error handling:
    ///
    /// ```
    /// # use nalgebra::{Isometry3, Translation3, UnitQuaternion, Vector3};
    /// use std::f32::consts::PI;
    ///
    /// fn safe_animate(
    ///     start: &Isometry3<f32>,
    ///     end: &Isometry3<f32>,
    ///     t: f32
    /// ) -> Isometry3<f32> {
    ///     start.try_lerp_slerp(end, t, 1.0e-6)
    ///         .unwrap_or_else(|| {
    ///             // Fallback: interpolate translation only, use end rotation
    ///             let position = start.translation.vector.lerp(&end.translation.vector, t);
    ///             Isometry3::from_parts(position.into(), end.rotation)
    ///         })
    /// }
    ///
    /// let start = Isometry3::identity();
    /// let end = Isometry3::translation(5.0, 0.0, 0.0);
    /// let animated = safe_animate(&start, &end, 0.5);
    /// ```
    ///
    /// Using in a game engine with proper error propagation:
    ///
    /// ```
    /// # use nalgebra::{Isometry3, Translation3, UnitQuaternion};
    ///
    /// fn interpolate_entity_pose(
    ///     current: &Isometry3<f32>,
    ///     target: &Isometry3<f32>,
    ///     speed: f32,
    ///     delta_time: f32,
    /// ) -> Result<Isometry3<f32>, &'static str> {
    ///     let t = (speed * delta_time).min(1.0);
    ///
    ///     current.try_lerp_slerp(target, t, 1.0e-6)
    ///         .ok_or("Entity orientations are opposite - cannot interpolate")
    /// }
    /// ```
    ///
    /// Smooth vehicle movement with safety checks:
    ///
    /// ```
    /// # use nalgebra::{Isometry3, Translation3, UnitQuaternion, Vector3, Point3};
    /// use std::f32::consts::PI;
    ///
    /// struct Vehicle {
    ///     pose: Isometry3<f32>,
    /// }
    ///
    /// impl Vehicle {
    ///     fn move_towards(&mut self, target: &Isometry3<f32>, t: f32) -> bool {
    ///         if let Some(new_pose) = self.pose.try_lerp_slerp(target, t, 1.0e-6) {
    ///             self.pose = new_pose;
    ///             true
    ///         } else {
    ///             // Handle degenerate case - snap to target
    ///             self.pose = *target;
    ///             false
    ///         }
    ///     }
    /// }
    ///
    /// let mut vehicle = Vehicle {
    ///     pose: Isometry3::identity()
    /// };
    /// let target = Isometry3::translation(10.0, 0.0, 0.0);
    /// vehicle.move_towards(&target, 0.1);
    /// ```
    ///
    /// # See Also
    ///
    /// * [`Isometry3::lerp_slerp`] - Panicking version for when rotations are known to be valid
    /// * [`Rotation3::try_slerp`] - Safe rotation-only interpolation
    /// * [`Isometry2::lerp_slerp`] - 2D isometry interpolation (no degenerate case)
    /// * [`IsometryMatrix3::try_lerp_slerp`] - Matrix-based variant
    #[inline]
    #[must_use]
    pub fn try_lerp_slerp(&self, other: &Self, t: T, epsilon: T) -> Option<Self>
    where
        T: RealField,
    {
        let tr = self
            .translation
            .vector
            .lerp(&other.translation.vector, t.clone());
        let rot = self.rotation.try_slerp(&other.rotation, t, epsilon)?;
        Some(Self::from_parts(tr.into(), rot))
    }
}

impl<T: SimdRealField> IsometryMatrix3<T> {
    /// Interpolates between two 3D isometries (matrix form) using LERP for translation and SLERP for rotation.
    ///
    /// This is the matrix-based variant of isometry interpolation, which uses `Rotation3` matrices
    /// instead of `UnitQuaternion` for rotations. The behavior is identical to [`Isometry3::lerp_slerp`],
    /// but works with the matrix representation of rotations.
    ///
    /// An `IsometryMatrix3` combines:
    /// - **Translation**: A 3D position vector, interpolated linearly
    /// - **Rotation**: A 3×3 rotation matrix, interpolated using SLERP
    ///
    /// This representation is useful when you need direct access to the rotation matrix
    /// for transformations or when interfacing with graphics APIs that expect matrix form.
    ///
    /// # Parameters
    ///
    /// * `self`: The starting isometry (position and orientation) at `t = 0.0`
    /// * `other`: The target isometry (position and orientation) at `t = 1.0`
    /// * `t`: The interpolation parameter, typically in `[0.0, 1.0]`
    ///
    /// # Panics
    ///
    /// Panics if the rotations are approximately 180 degrees apart. Use
    /// [`IsometryMatrix3::try_lerp_slerp`] to handle this case safely.
    ///
    /// # Examples
    ///
    /// Basic interpolation with rotation matrices:
    ///
    /// ```
    /// # use nalgebra::{Vector3, Translation3, Rotation3, IsometryMatrix3};
    /// use std::f32::consts::PI;
    ///
    /// let start = IsometryMatrix3::from_parts(
    ///     Translation3::new(1.0, 2.0, 3.0),
    ///     Rotation3::from_euler_angles(PI / 4.0, 0.0, 0.0)
    /// );
    ///
    /// let end = IsometryMatrix3::from_parts(
    ///     Translation3::new(4.0, 8.0, 12.0),
    ///     Rotation3::from_euler_angles(-PI, 0.0, 0.0)
    /// );
    ///
    /// let mid = start.lerp_slerp(&end, 1.0 / 3.0);
    ///
    /// assert_eq!(mid.translation.vector, Vector3::new(2.0, 4.0, 6.0));
    /// assert_eq!(mid.rotation.euler_angles(), (PI / 2.0, 0.0, 0.0));
    /// ```
    ///
    /// Animating an object with matrix-based rotations:
    ///
    /// ```
    /// # use nalgebra::{IsometryMatrix3, Translation3, Rotation3, Vector3};
    /// use std::f32::consts::PI;
    ///
    /// // Object starts at origin with no rotation
    /// let start = IsometryMatrix3::identity();
    ///
    /// // Object ends translated and rotated
    /// let end = IsometryMatrix3::from_parts(
    ///     Translation3::new(5.0, 0.0, 0.0),
    ///     Rotation3::from_axis_angle(&Vector3::y_axis(), PI / 2.0)
    /// );
    ///
    /// // Generate animation frames
    /// for frame in 0..=20 {
    ///     let t = frame as f32 / 20.0;
    ///     let pose = start.lerp_slerp(&end, t);
    ///
    ///     // The rotation matrix can be directly used for rendering
    ///     let matrix = pose.to_homogeneous();
    ///     println!("Frame {}: transformation matrix = {:?}", frame, matrix);
    /// }
    /// ```
    ///
    /// Interpolating a robot arm joint:
    ///
    /// ```
    /// # use nalgebra::{IsometryMatrix3, Translation3, Rotation3, Vector3};
    /// use std::f32::consts::PI;
    ///
    /// // Joint at rest position
    /// let rest_pose = IsometryMatrix3::from_parts(
    ///     Translation3::new(0.0, 0.0, 0.0),
    ///     Rotation3::identity()
    /// );
    ///
    /// // Joint at extended position
    /// let extended_pose = IsometryMatrix3::from_parts(
    ///     Translation3::new(0.0, 0.0, 2.0),
    ///     Rotation3::from_axis_angle(&Vector3::x_axis(), PI / 3.0)
    /// );
    ///
    /// // Smoothly move from rest to extended
    /// let current_pose = rest_pose.lerp_slerp(&extended_pose, 0.6);
    /// ```
    ///
    /// Smooth path following with waypoints:
    ///
    /// ```
    /// # use nalgebra::{IsometryMatrix3, Translation3, Rotation3, Vector3};
    /// use std::f32::consts::PI;
    ///
    /// let waypoint1 = IsometryMatrix3::translation(0.0, 0.0, 0.0);
    /// let waypoint2 = IsometryMatrix3::translation(5.0, 0.0, 0.0);
    /// let waypoint3 = IsometryMatrix3::from_parts(
    ///     Translation3::new(5.0, 5.0, 0.0),
    ///     Rotation3::from_axis_angle(&Vector3::z_axis(), PI / 2.0)
    /// );
    ///
    /// // Interpolate between waypoints
    /// let segment1_mid = waypoint1.lerp_slerp(&waypoint2, 0.5);
    /// let segment2_mid = waypoint2.lerp_slerp(&waypoint3, 0.5);
    /// ```
    ///
    /// # See Also
    ///
    /// * [`IsometryMatrix3::try_lerp_slerp`] - Safe version handling opposite rotations
    /// * [`Isometry3::lerp_slerp`] - Quaternion-based variant
    /// * [`Rotation3::slerp`] - Rotation matrix interpolation
    /// * [`IsometryMatrix2::lerp_slerp`] - 2D version
    #[inline]
    #[must_use]
    pub fn lerp_slerp(&self, other: &Self, t: T) -> Self
    where
        T: RealField,
    {
        let tr = self
            .translation
            .vector
            .lerp(&other.translation.vector, t.clone());
        let rot = self.rotation.slerp(&other.rotation, t);
        Self::from_parts(tr.into(), rot)
    }

    /// Attempts to interpolate between two 3D isometries (matrix form), handling opposite rotations.
    ///
    /// This is the safe, non-panicking version of [`IsometryMatrix3::lerp_slerp`] that returns
    /// `None` when the rotation matrices are approximately 180 degrees apart, making the
    /// interpolation mathematically undefined.
    ///
    /// This function is the matrix-based equivalent of [`Isometry3::try_lerp_slerp`], using
    /// `Rotation3` matrices instead of `UnitQuaternion` for the rotation component.
    ///
    /// # Parameters
    ///
    /// * `self`: Starting isometry (position and orientation matrix)
    /// * `other`: Target isometry (position and orientation matrix)
    /// * `t`: Interpolation parameter in `[0.0, 1.0]`
    /// * `epsilon`: Threshold for detecting opposite rotations (e.g., `1.0e-6`)
    ///
    /// # Returns
    ///
    /// * `Some(isometry)` - Interpolated pose if well-defined
    /// * `None` - If rotations are approximately opposite
    ///
    /// # Examples
    ///
    /// Basic usage with successful interpolation:
    ///
    /// ```
    /// # use nalgebra::{Vector3, Translation3, Rotation3, IsometryMatrix3};
    /// use std::f32::consts::PI;
    ///
    /// let start = IsometryMatrix3::from_parts(
    ///     Translation3::new(1.0, 2.0, 3.0),
    ///     Rotation3::from_euler_angles(PI / 4.0, 0.0, 0.0)
    /// );
    ///
    /// let end = IsometryMatrix3::from_parts(
    ///     Translation3::new(4.0, 8.0, 12.0),
    ///     Rotation3::from_euler_angles(-PI, 0.0, 0.0)
    /// );
    ///
    /// if let Some(mid) = start.try_lerp_slerp(&end, 1.0 / 3.0, 1.0e-6) {
    ///     assert_eq!(mid.translation.vector, Vector3::new(2.0, 4.0, 6.0));
    ///     assert_eq!(mid.rotation.euler_angles(), (PI / 2.0, 0.0, 0.0));
    /// }
    /// ```
    ///
    /// Handling degenerate rotation cases:
    ///
    /// ```
    /// # use nalgebra::{IsometryMatrix3, Translation3, Rotation3, Vector3};
    /// use std::f32::consts::PI;
    ///
    /// let pose1 = IsometryMatrix3::identity();
    /// let pose2 = IsometryMatrix3::from_parts(
    ///     Translation3::new(10.0, 0.0, 0.0),
    ///     Rotation3::from_axis_angle(&Vector3::x_axis(), PI)  // 180 degrees
    /// );
    ///
    /// match pose1.try_lerp_slerp(&pose2, 0.5, 1.0e-6) {
    ///     Some(interpolated) => println!("Successful interpolation"),
    ///     None => println!("Opposite rotations - choosing alternative path")
    /// }
    /// ```
    ///
    /// Robust robot control with fallback:
    ///
    /// ```
    /// # use nalgebra::{IsometryMatrix3, Translation3, Rotation3, Vector3};
    /// use std::f32::consts::PI;
    ///
    /// fn move_robot_arm(
    ///     current: &IsometryMatrix3<f32>,
    ///     target: &IsometryMatrix3<f32>,
    ///     step: f32
    /// ) -> IsometryMatrix3<f32> {
    ///     current.try_lerp_slerp(target, step, 1.0e-6)
    ///         .unwrap_or_else(|| {
    ///             // Fallback: break the motion into smaller steps via an intermediate pose
    ///             let mid_rotation = Rotation3::from_axis_angle(&Vector3::z_axis(), PI / 4.0);
    ///             let mid_pos = current.translation.vector.lerp(&target.translation.vector, 0.5);
    ///             IsometryMatrix3::from_parts(mid_pos.into(), mid_rotation)
    ///         })
    /// }
    ///
    /// let current = IsometryMatrix3::identity();
    /// let target = IsometryMatrix3::translation(5.0, 5.0, 0.0);
    /// let next = move_robot_arm(&current, &target, 0.1);
    /// ```
    ///
    /// Animating with error recovery:
    ///
    /// ```
    /// # use nalgebra::{IsometryMatrix3, Translation3, Rotation3, Vector3};
    /// use std::f32::consts::PI;
    ///
    /// struct AnimatedObject {
    ///     pose: IsometryMatrix3<f32>,
    /// }
    ///
    /// impl AnimatedObject {
    ///     fn interpolate_to(&mut self, target: &IsometryMatrix3<f32>, t: f32) -> Result<(), String> {
    ///         match self.pose.try_lerp_slerp(target, t, 1.0e-6) {
    ///             Some(new_pose) => {
    ///                 self.pose = new_pose;
    ///                 Ok(())
    ///             }
    ///             None => {
    ///                 Err("Cannot interpolate: rotations are opposite".to_string())
    ///             }
    ///         }
    ///     }
    /// }
    ///
    /// let mut obj = AnimatedObject {
    ///     pose: IsometryMatrix3::identity()
    /// };
    /// let target = IsometryMatrix3::translation(1.0, 2.0, 3.0);
    /// let _ = obj.interpolate_to(&target, 0.5);
    /// ```
    ///
    /// Physics simulation with safe interpolation:
    ///
    /// ```
    /// # use nalgebra::{IsometryMatrix3, Translation3, Rotation3, Vector3};
    ///
    /// fn integrate_rigid_body(
    ///     current_pose: &IsometryMatrix3<f32>,
    ///     target_pose: &IsometryMatrix3<f32>,
    ///     dt: f32,
    ///     velocity: f32,
    /// ) -> Option<IsometryMatrix3<f32>> {
    ///     let t = (velocity * dt).min(1.0);
    ///     current_pose.try_lerp_slerp(target_pose, t, 1.0e-6)
    /// }
    /// ```
    ///
    /// # See Also
    ///
    /// * [`IsometryMatrix3::lerp_slerp`] - Panicking version for known-safe cases
    /// * [`Isometry3::try_lerp_slerp`] - Quaternion-based variant
    /// * [`Rotation3::try_slerp`] - Safe rotation matrix interpolation
    /// * [`IsometryMatrix2::lerp_slerp`] - 2D version (no degenerate case)
    #[inline]
    #[must_use]
    pub fn try_lerp_slerp(&self, other: &Self, t: T, epsilon: T) -> Option<Self>
    where
        T: RealField,
    {
        let tr = self
            .translation
            .vector
            .lerp(&other.translation.vector, t.clone());
        let rot = self.rotation.try_slerp(&other.rotation, t, epsilon)?;
        Some(Self::from_parts(tr.into(), rot))
    }
}

impl<T: SimdRealField> Isometry2<T> {
    /// Interpolates between two 2D isometries using LERP for translation and SLERP for rotation.
    ///
    /// An `Isometry2` represents a rigid body transformation in 2D space, combining both
    /// translation (position) and rotation (orientation). This function smoothly interpolates
    /// both components:
    /// - **Translation**: Uses LERP (Linear Interpolation) to move in a straight line
    /// - **Rotation**: Uses SLERP (Spherical Linear Interpolation) for smooth angular motion
    ///
    /// In 2D, SLERP simplifies to interpolating along a circular arc between two angles,
    /// maintaining constant angular velocity throughout the transition.
    ///
    /// Common applications include:
    /// - 2D game object animations (characters, enemies, projectiles)
    /// - Top-down or side-scrolling camera movements
    /// - UI element transitions with rotation
    /// - 2D physics simulations
    /// - Sprite positioning and orientation in games
    ///
    /// # Parameters
    ///
    /// * `self`: The starting isometry (2D pose) at `t = 0.0`
    /// * `other`: The target isometry (2D pose) at `t = 1.0`
    /// * `t`: The interpolation parameter, typically in `[0.0, 1.0]`
    ///   - `t = 0.0` returns `self`
    ///   - `t = 1.0` returns `other`
    ///   - `t = 0.5` returns the pose halfway between
    ///
    /// # Panics
    ///
    /// Panics if the rotations are approximately 180 degrees apart. While rare in 2D,
    /// this can still occur and should be handled with care in production code.
    ///
    /// # Examples
    ///
    /// Basic 2D interpolation:
    ///
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{Vector2, Translation2, UnitComplex, Isometry2};
    /// use std::f32::consts::PI;
    ///
    /// // Start at (1, 2) rotated 45 degrees
    /// let start = Isometry2::from_parts(
    ///     Translation2::new(1.0, 2.0),
    ///     UnitComplex::new(PI / 4.0)
    /// );
    ///
    /// // End at (4, 8) rotated -180 degrees
    /// let end = Isometry2::from_parts(
    ///     Translation2::new(4.0, 8.0),
    ///     UnitComplex::new(-PI)
    /// );
    ///
    /// // Interpolate one-third of the way
    /// let mid = start.lerp_slerp(&end, 1.0 / 3.0);
    ///
    /// assert_eq!(mid.translation.vector, Vector2::new(2.0, 4.0));
    /// assert_relative_eq!(mid.rotation.angle(), PI / 2.0);
    /// ```
    ///
    /// Animating a 2D game character:
    ///
    /// ```
    /// # use nalgebra::{Isometry2, Translation2, UnitComplex, Vector2};
    /// use std::f32::consts::PI;
    ///
    /// // Character at origin, facing right (0 degrees)
    /// let start_pose = Isometry2::from_parts(
    ///     Translation2::new(0.0, 0.0),
    ///     UnitComplex::new(0.0)
    /// );
    ///
    /// // Character moved to (10, 5), facing up (90 degrees)
    /// let end_pose = Isometry2::from_parts(
    ///     Translation2::new(10.0, 5.0),
    ///     UnitComplex::new(PI / 2.0)
    /// );
    ///
    /// // Generate 30 frames of animation
    /// for frame in 0..=30 {
    ///     let t = frame as f32 / 30.0;
    ///     let current_pose = start_pose.lerp_slerp(&end_pose, t);
    ///
    ///     // Use current_pose to render the character sprite
    ///     let position = current_pose.translation.vector;
    ///     let angle = current_pose.rotation.angle();
    ///     // draw_sprite(position, angle);
    /// }
    /// ```
    ///
    /// Smooth camera panning and rotation in 2D game:
    ///
    /// ```
    /// # use nalgebra::{Isometry2, Translation2, UnitComplex, Point2};
    /// use std::f32::consts::PI;
    ///
    /// // Camera starts centered at origin, no rotation
    /// let camera_start = Isometry2::identity();
    ///
    /// // Camera moves to follow player at (20, 10), tilted slightly
    /// let camera_target = Isometry2::from_parts(
    ///     Translation2::new(20.0, 10.0),
    ///     UnitComplex::new(PI / 16.0)  // 11.25 degrees tilt
    /// );
    ///
    /// // Smoothly interpolate camera over time
    /// let camera_speed = 0.05;  // 5% per frame
    /// let camera_current = camera_start.lerp_slerp(&camera_target, camera_speed);
    /// ```
    ///
    /// Moving and rotating a projectile:
    ///
    /// ```
    /// # use nalgebra::{Isometry2, Translation2, UnitComplex, Vector2};
    /// use std::f32::consts::PI;
    ///
    /// // Bullet fired from (0, 0) pointing right
    /// let bullet_start = Isometry2::from_parts(
    ///     Translation2::new(0.0, 0.0),
    ///     UnitComplex::new(0.0)
    /// );
    ///
    /// // Bullet trajectory ends at (100, 20) after curving upward
    /// let bullet_end = Isometry2::from_parts(
    ///     Translation2::new(100.0, 20.0),
    ///     UnitComplex::new(PI / 8.0)  // Rotated 22.5 degrees up
    /// );
    ///
    /// // Calculate bullet position at 70% through trajectory
    /// let bullet_pos = bullet_start.lerp_slerp(&bullet_end, 0.7);
    /// ```
    ///
    /// Animating a rotating platform:
    ///
    /// ```
    /// # use nalgebra::{Isometry2, Translation2, UnitComplex};
    /// use std::f32::consts::PI;
    ///
    /// let platform_rest = Isometry2::translation(10.0, 5.0);
    /// let platform_rotated = Isometry2::from_parts(
    ///     Translation2::new(10.0, 5.0),
    ///     UnitComplex::new(PI)  // Rotated 180 degrees
    /// );
    ///
    /// // Animate platform rotating in place
    /// let platform_halfway = platform_rest.lerp_slerp(&platform_rotated, 0.5);
    /// ```
    ///
    /// # See Also
    ///
    /// * [`Isometry3::lerp_slerp`] - 3D version with position and orientation
    /// * [`IsometryMatrix2::lerp_slerp`] - Matrix-based 2D variant
    /// * [`UnitComplex::slerp`] - 2D rotation interpolation
    /// * [`Rotation2::slerp`] - Alternative rotation representation
    #[inline]
    #[must_use]
    pub fn lerp_slerp(&self, other: &Self, t: T) -> Self
    where
        T: RealField,
    {
        let tr = self
            .translation
            .vector
            .lerp(&other.translation.vector, t.clone());
        let rot = self.rotation.slerp(&other.rotation, t);
        Self::from_parts(tr.into(), rot)
    }
}

impl<T: SimdRealField> IsometryMatrix2<T> {
    /// Interpolates between two 2D isometries (matrix form) using LERP and SLERP.
    ///
    /// This is the matrix-based variant of 2D isometry interpolation, using `Rotation2` matrices
    /// instead of `UnitComplex` for rotations. The behavior is identical to [`Isometry2::lerp_slerp`],
    /// but works with the 2×2 rotation matrix representation.
    ///
    /// An `IsometryMatrix2` combines:
    /// - **Translation**: A 2D position vector, interpolated linearly
    /// - **Rotation**: A 2×2 rotation matrix, interpolated using SLERP
    ///
    /// This representation is useful when you need direct matrix access for transformations
    /// or when interfacing with graphics systems that work with matrix forms.
    ///
    /// # Parameters
    ///
    /// * `self`: The starting isometry (2D pose) at `t = 0.0`
    /// * `other`: The target isometry (2D pose) at `t = 1.0`
    /// * `t`: The interpolation parameter, typically in `[0.0, 1.0]`
    ///
    /// # Panics
    ///
    /// Panics if the rotations are approximately 180 degrees apart. This is rare in 2D
    /// but can occur in edge cases.
    ///
    /// # Examples
    ///
    /// Basic 2D matrix-based interpolation:
    ///
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{Vector2, Translation2, Rotation2, IsometryMatrix2};
    /// use std::f32::consts::PI;
    ///
    /// // Start at (1, 2) rotated 45 degrees
    /// let start = IsometryMatrix2::from_parts(
    ///     Translation2::new(1.0, 2.0),
    ///     Rotation2::new(PI / 4.0)
    /// );
    ///
    /// // End at (4, 8) rotated -180 degrees
    /// let end = IsometryMatrix2::from_parts(
    ///     Translation2::new(4.0, 8.0),
    ///     Rotation2::new(-PI)
    /// );
    ///
    /// // Interpolate one-third of the way
    /// let mid = start.lerp_slerp(&end, 1.0 / 3.0);
    ///
    /// assert_eq!(mid.translation.vector, Vector2::new(2.0, 4.0));
    /// assert_relative_eq!(mid.rotation.angle(), PI / 2.0);
    /// ```
    ///
    /// Animating 2D sprites with matrix transformations:
    ///
    /// ```
    /// # use nalgebra::{IsometryMatrix2, Translation2, Rotation2, Vector2};
    /// use std::f32::consts::PI;
    ///
    /// // Sprite starts at origin, no rotation
    /// let sprite_start = IsometryMatrix2::identity();
    ///
    /// // Sprite ends at (50, 30), rotated 45 degrees
    /// let sprite_end = IsometryMatrix2::from_parts(
    ///     Translation2::new(50.0, 30.0),
    ///     Rotation2::new(PI / 4.0)
    /// );
    ///
    /// // Generate frames for smooth animation
    /// for frame in 0..=60 {
    ///     let t = frame as f32 / 60.0;
    ///     let sprite_pose = sprite_start.lerp_slerp(&sprite_end, t);
    ///
    ///     // Convert to 3x3 homogeneous matrix for rendering
    ///     let transform_matrix = sprite_pose.to_homogeneous();
    ///     // Use transform_matrix for GPU rendering
    /// }
    /// ```
    ///
    /// Top-down vehicle steering animation:
    ///
    /// ```
    /// # use nalgebra::{IsometryMatrix2, Translation2, Rotation2};
    /// use std::f32::consts::PI;
    ///
    /// // Vehicle at start of turn
    /// let vehicle_start = IsometryMatrix2::from_parts(
    ///     Translation2::new(0.0, 0.0),
    ///     Rotation2::new(0.0)  // Facing right
    /// );
    ///
    /// // Vehicle at end of turn
    /// let vehicle_end = IsometryMatrix2::from_parts(
    ///     Translation2::new(10.0, 5.0),
    ///     Rotation2::new(PI / 3.0)  // Turned 60 degrees
    /// );
    ///
    /// // Smooth steering transition
    /// let vehicle_mid = vehicle_start.lerp_slerp(&vehicle_end, 0.5);
    /// ```
    ///
    /// Animating UI elements with rotation:
    ///
    /// ```
    /// # use nalgebra::{IsometryMatrix2, Translation2, Rotation2};
    /// use std::f32::consts::PI;
    ///
    /// // Menu item in closed state
    /// let menu_closed = IsometryMatrix2::from_parts(
    ///     Translation2::new(-100.0, 0.0),  // Off-screen
    ///     Rotation2::new(-PI / 2.0)  // Rotated away
    /// );
    ///
    /// // Menu item in open state
    /// let menu_open = IsometryMatrix2::from_parts(
    ///     Translation2::new(0.0, 0.0),  // On-screen
    ///     Rotation2::new(0.0)  // Upright
    /// );
    ///
    /// // Animate menu sliding in and rotating
    /// let menu_current = menu_closed.lerp_slerp(&menu_open, 0.3);
    /// ```
    ///
    /// 2D robotic arm simulation:
    ///
    /// ```
    /// # use nalgebra::{IsometryMatrix2, Translation2, Rotation2, Point2};
    /// use std::f32::consts::PI;
    ///
    /// // Arm segment at rest
    /// let arm_rest = IsometryMatrix2::from_parts(
    ///     Translation2::new(0.0, 0.0),
    ///     Rotation2::new(0.0)
    /// );
    ///
    /// // Arm segment extended and rotated
    /// let arm_extended = IsometryMatrix2::from_parts(
    ///     Translation2::new(5.0, 0.0),
    ///     Rotation2::new(PI / 4.0)
    /// );
    ///
    /// // Interpolate arm motion
    /// let arm_position = arm_rest.lerp_slerp(&arm_extended, 0.75);
    ///
    /// // Apply transformation to end effector
    /// let end_point = Point2::new(1.0, 0.0);
    /// let transformed = arm_position * end_point;
    /// ```
    ///
    /// # See Also
    ///
    /// * [`Isometry2::lerp_slerp`] - Complex number-based variant
    /// * [`IsometryMatrix3::lerp_slerp`] - 3D matrix-based version
    /// * [`Rotation2::slerp`] - Rotation matrix interpolation
    /// * [`Isometry3::lerp_slerp`] - 3D quaternion-based version
    #[inline]
    #[must_use]
    pub fn lerp_slerp(&self, other: &Self, t: T) -> Self
    where
        T: RealField,
    {
        let tr = self
            .translation
            .vector
            .lerp(&other.translation.vector, t.clone());
        let rot = self.rotation.slerp(&other.rotation, t);
        Self::from_parts(tr.into(), rot)
    }
}
