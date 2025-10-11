use crate::{RealField, Rotation2, Rotation3, SimdRealField, UnitComplex, UnitQuaternion};

/// # Interpolation
impl<T: SimdRealField> Rotation2<T> {
    /// Spherical linear interpolation between two 2D rotation matrices.
    ///
    /// This performs a smooth interpolation between `self` and `other` rotation matrices,
    /// taking the shortest angular path between them. Unlike simple linear interpolation,
    /// SLERP (Spherical Linear Interpolation) maintains constant angular velocity and
    /// produces rotations that are evenly distributed along the arc between the two
    /// input rotations.
    ///
    /// This is particularly useful for:
    /// - Animating smooth rotations between two orientations
    /// - Creating camera transitions
    /// - Interpolating object orientations in physics simulations
    /// - Any scenario requiring smooth, natural-looking rotation transitions
    ///
    /// # Parameters
    ///
    /// * `self`: The starting rotation (returned when `t = 0.0`)
    /// * `other`: The target rotation (returned when `t = 1.0`)
    /// * `t`: The interpolation parameter, typically in the range `[0.0, 1.0]`
    ///   - `t = 0.0` returns `self`
    ///   - `t = 1.0` returns `other`
    ///   - `t = 0.5` returns the rotation halfway between `self` and `other`
    ///   - Values outside `[0.0, 1.0]` will extrapolate beyond the two rotations
    ///
    /// # Examples
    ///
    /// Basic interpolation between two rotations:
    ///
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::geometry::Rotation2;
    /// use std::f32::consts::PI;
    ///
    /// // Start at 45 degrees
    /// let rot1 = Rotation2::new(PI / 4.0);
    /// // End at -180 degrees
    /// let rot2 = Rotation2::new(-PI);
    ///
    /// // Interpolate one-third of the way
    /// let rot = rot1.slerp(&rot2, 1.0 / 3.0);
    ///
    /// // Result is at 90 degrees
    /// assert_relative_eq!(rot.angle(), PI / 2.0);
    /// ```
    ///
    /// Animating a rotating object over time:
    ///
    /// ```
    /// # use nalgebra::{Rotation2, Vector2};
    /// use std::f32::consts::PI;
    ///
    /// // Initial rotation: facing right (0 degrees)
    /// let start_rotation = Rotation2::new(0.0);
    /// // Target rotation: facing up (90 degrees)
    /// let end_rotation = Rotation2::new(PI / 2.0);
    ///
    /// // Simulate 5 frames of animation
    /// for frame in 0..=5 {
    ///     let t = frame as f32 / 5.0;  // 0.0, 0.2, 0.4, 0.6, 0.8, 1.0
    ///     let current_rotation = start_rotation.slerp(&end_rotation, t);
    ///
    ///     // Apply rotation to a point
    ///     let point = Vector2::new(1.0, 0.0);
    ///     let rotated = current_rotation * point;
    ///     println!("Frame {}: {:?}", frame, rotated);
    /// }
    /// ```
    ///
    /// Creating a smooth spinning animation:
    ///
    /// ```
    /// # use nalgebra::Rotation2;
    /// use std::f32::consts::PI;
    ///
    /// let no_rotation = Rotation2::identity();
    /// let full_spin = Rotation2::new(2.0 * PI);
    ///
    /// // Get rotation at 25% through a full spin
    /// let quarter_spin = no_rotation.slerp(&full_spin, 0.25);
    /// assert!((quarter_spin.angle() - PI / 2.0).abs() < 1e-5);
    /// ```
    ///
    /// # See Also
    ///
    /// * [`Rotation3::slerp`] - 3D rotation interpolation
    /// * [`UnitComplex::slerp`] - Underlying complex number interpolation used by this method
    /// * [`Isometry2::lerp_slerp`] - Combined translation and rotation interpolation in 2D
    #[inline]
    #[must_use]
    pub fn slerp(&self, other: &Self, t: T) -> Self
    where
        T::Element: SimdRealField,
    {
        let c1 = UnitComplex::from(self.clone());
        let c2 = UnitComplex::from(other.clone());
        c1.slerp(&c2, t).into()
    }
}

impl<T: SimdRealField> Rotation3<T> {
    /// Spherical linear interpolation between two 3D rotation matrices.
    ///
    /// This performs a smooth interpolation between `self` and `other` rotation matrices in 3D space,
    /// taking the shortest angular path between them. SLERP (Spherical Linear Interpolation) maintains
    /// constant angular velocity and produces rotations that are evenly distributed along the great
    /// circle arc connecting the two orientations.
    ///
    /// This is the standard method for interpolating 3D rotations because:
    /// - It produces smooth, natural-looking rotations
    /// - It maintains constant angular velocity (no speed-up or slow-down)
    /// - It takes the shortest path between orientations
    /// - It avoids gimbal lock issues that plague Euler angle interpolation
    ///
    /// Common use cases include:
    /// - Character animation and skeletal rigging
    /// - Camera movements and viewpoint transitions
    /// - Spacecraft and vehicle orientation control
    /// - Smooth object rotation in 3D graphics
    /// - Physics simulations requiring orientation interpolation
    ///
    /// # Parameters
    ///
    /// * `self`: The starting rotation (returned when `t = 0.0`)
    /// * `other`: The target rotation (returned when `t = 1.0`)
    /// * `t`: The interpolation parameter, typically in the range `[0.0, 1.0]`
    ///   - `t = 0.0` returns `self`
    ///   - `t = 1.0` returns `other`
    ///   - `t = 0.5` returns the rotation halfway between `self` and `other`
    ///   - Values outside `[0.0, 1.0]` will extrapolate beyond the two rotations
    ///
    /// # Panics
    ///
    /// Panics if the angle between both rotations is approximately 180 degrees, because
    /// there are infinitely many shortest paths between opposite orientations (the
    /// interpolation is mathematically undefined). Use [`Rotation3::try_slerp`] instead
    /// if you need to handle this case gracefully.
    ///
    /// # Examples
    ///
    /// Basic interpolation between two rotations:
    ///
    /// ```
    /// # use nalgebra::geometry::Rotation3;
    /// use std::f32::consts::PI;
    ///
    /// // Start rotated 45 degrees around X axis
    /// let start = Rotation3::from_euler_angles(PI / 4.0, 0.0, 0.0);
    /// // End rotated -180 degrees around X axis
    /// let end = Rotation3::from_euler_angles(-PI, 0.0, 0.0);
    ///
    /// // Interpolate one-third of the way
    /// let mid = start.slerp(&end, 1.0 / 3.0);
    ///
    /// // Result is rotated 90 degrees around X axis
    /// assert_eq!(mid.euler_angles(), (PI / 2.0, 0.0, 0.0));
    /// ```
    ///
    /// Animating a smooth camera rotation:
    ///
    /// ```
    /// # use nalgebra::{Rotation3, Vector3, Point3};
    /// use std::f32::consts::PI;
    ///
    /// // Camera starts looking down the Z axis
    /// let start_rotation = Rotation3::identity();
    /// // Camera ends looking at 45 degrees to the side
    /// let end_rotation = Rotation3::from_axis_angle(
    ///     &Vector3::y_axis(),
    ///     PI / 4.0
    /// );
    ///
    /// // Simulate 10 frames of smooth camera rotation
    /// for frame in 0..=10 {
    ///     let t = frame as f32 / 10.0;
    ///     let camera_rotation = start_rotation.slerp(&end_rotation, t);
    ///
    ///     // Get the camera's forward direction
    ///     let forward = camera_rotation * Vector3::z();
    ///     println!("Frame {}: camera looking at {:?}", frame, forward);
    /// }
    /// ```
    ///
    /// Interpolating a spacecraft's orientation:
    ///
    /// ```
    /// # use nalgebra::{Rotation3, Vector3};
    /// use std::f32::consts::PI;
    ///
    /// // Spacecraft starts upright
    /// let current_orientation = Rotation3::identity();
    ///
    /// // Target orientation: rolled 90 degrees
    /// let target_orientation = Rotation3::from_axis_angle(
    ///     &Vector3::z_axis(),
    ///     PI / 2.0
    /// );
    ///
    /// // Smoothly interpolate over time (e.g., 30% complete)
    /// let new_orientation = current_orientation.slerp(&target_orientation, 0.3);
    ///
    /// // Apply to spacecraft model vertices
    /// let vertex = Vector3::new(1.0, 0.0, 0.0);
    /// let rotated_vertex = new_orientation * vertex;
    /// ```
    ///
    /// Creating a rotating object animation:
    ///
    /// ```
    /// # use nalgebra::{Rotation3, Vector3};
    /// use std::f32::consts::PI;
    ///
    /// let no_rotation = Rotation3::identity();
    /// let half_rotation = Rotation3::from_axis_angle(&Vector3::y_axis(), PI);
    ///
    /// // Get rotation at various points along the path
    /// let at_25_percent = no_rotation.slerp(&half_rotation, 0.25);
    /// let at_50_percent = no_rotation.slerp(&half_rotation, 0.50);
    /// let at_75_percent = no_rotation.slerp(&half_rotation, 0.75);
    /// ```
    ///
    /// # See Also
    ///
    /// * [`Rotation3::try_slerp`] - Safe version that handles 180-degree rotations
    /// * [`Rotation2::slerp`] - 2D rotation interpolation
    /// * [`UnitQuaternion::slerp`] - Underlying quaternion interpolation used by this method
    /// * [`Isometry3::lerp_slerp`] - Combined translation and rotation interpolation in 3D
    #[inline]
    #[must_use]
    pub fn slerp(&self, other: &Self, t: T) -> Self
    where
        T: RealField,
    {
        let q1 = UnitQuaternion::from(self.clone());
        let q2 = UnitQuaternion::from(other.clone());
        q1.slerp(&q2, t).into()
    }

    /// Attempts to perform spherical linear interpolation between two 3D rotation matrices.
    ///
    /// This is the safe version of [`Rotation3::slerp`] that handles the edge case where
    /// the two rotations are approximately 180 degrees apart. When rotations are opposite,
    /// there are infinitely many shortest paths between them (any great circle through the
    /// antipodal points), making the interpolation mathematically undefined.
    ///
    /// This function returns `None` when the rotations are too close to being opposite,
    /// allowing you to handle this case gracefully (e.g., by choosing an alternative
    /// interpolation path or fallback rotation).
    ///
    /// # Parameters
    ///
    /// * `self`: The starting rotation to interpolate from
    /// * `other`: The target rotation to interpolate toward
    /// * `t`: The interpolation parameter, typically in the range `[0.0, 1.0]`
    ///   - `t = 0.0` returns `self`
    ///   - `t = 1.0` returns `other`
    ///   - `t = 0.5` returns the rotation halfway between `self` and `other`
    /// * `epsilon`: Threshold for detecting opposite rotations. Returns `None` if the
    ///   sine of the angle between rotations is below this value. Typical values are
    ///   `1.0e-6` for `f32` or `1.0e-12` for `f64`.
    ///
    /// # Returns
    ///
    /// * `Some(rotation)` - The interpolated rotation if well-defined
    /// * `None` - If the rotations are approximately 180 degrees apart (within `epsilon`)
    ///
    /// # Examples
    ///
    /// Basic usage with successful interpolation:
    ///
    /// ```
    /// # use nalgebra::{Rotation3, Vector3};
    /// use std::f32::consts::PI;
    ///
    /// let start = Rotation3::from_axis_angle(&Vector3::x_axis(), PI / 4.0);
    /// let end = Rotation3::from_axis_angle(&Vector3::x_axis(), 3.0 * PI / 4.0);
    ///
    /// // These rotations are only 90 degrees apart, so interpolation succeeds
    /// let result = start.try_slerp(&end, 0.5, 1.0e-6);
    /// assert!(result.is_some());
    /// ```
    ///
    /// Handling the opposite rotation case:
    ///
    /// ```
    /// # use nalgebra::{Rotation3, Vector3};
    /// use std::f32::consts::PI;
    ///
    /// let forward = Rotation3::identity();
    /// let backward = Rotation3::from_axis_angle(&Vector3::y_axis(), PI);
    ///
    /// // These rotations are 180 degrees apart - interpolation is undefined
    /// let result = forward.try_slerp(&backward, 0.5, 1.0e-6);
    ///
    /// match result {
    ///     Some(rotation) => {
    ///         // Safe to use the interpolated rotation
    ///         println!("Interpolation succeeded");
    ///     }
    ///     None => {
    ///         // Handle the degenerate case - perhaps choose a specific path
    ///         // or use a different interpolation method
    ///         println!("Rotations are opposite - using fallback");
    ///         // For example, could rotate around a chosen axis
    ///         let fallback = Rotation3::from_axis_angle(&Vector3::z_axis(), PI / 2.0);
    ///     }
    /// }
    /// ```
    ///
    /// Robust animation system that handles all cases:
    ///
    /// ```
    /// # use nalgebra::{Rotation3, Vector3};
    /// use std::f32::consts::PI;
    ///
    /// fn safe_interpolate(start: &Rotation3<f32>, end: &Rotation3<f32>, t: f32) -> Rotation3<f32> {
    ///     match start.try_slerp(end, t, 1.0e-6) {
    ///         Some(rotation) => rotation,
    ///         None => {
    ///             // Fallback: rotate around the up axis
    ///             // This provides a predictable path when rotations are opposite
    ///             start.clone() * Rotation3::from_axis_angle(&Vector3::y_axis(), PI * t)
    ///         }
    ///     }
    /// }
    ///
    /// let rot1 = Rotation3::identity();
    /// let rot2 = Rotation3::from_axis_angle(&Vector3::x_axis(), 0.1);
    /// let interpolated = safe_interpolate(&rot1, &rot2, 0.5);
    /// ```
    ///
    /// Using in a game engine with error handling:
    ///
    /// ```
    /// # use nalgebra::{Rotation3, Vector3};
    ///
    /// fn animate_character_rotation(
    ///     current: &Rotation3<f32>,
    ///     target: &Rotation3<f32>,
    ///     delta_time: f32,
    ///     rotation_speed: f32,
    /// ) -> Result<Rotation3<f32>, &'static str> {
    ///     let t = (delta_time * rotation_speed).min(1.0);
    ///
    ///     current.try_slerp(target, t, 1.0e-6)
    ///         .ok_or("Cannot interpolate: rotations are opposite")
    /// }
    /// ```
    ///
    /// # See Also
    ///
    /// * [`Rotation3::slerp`] - Panicking version for when you know rotations aren't opposite
    /// * [`Rotation2::slerp`] - 2D rotation interpolation (no opposite rotation issue)
    /// * [`UnitQuaternion::try_slerp`] - Underlying quaternion interpolation
    /// * [`Isometry3::try_lerp_slerp`] - Safe combined translation and rotation interpolation
    #[inline]
    #[must_use]
    pub fn try_slerp(&self, other: &Self, t: T, epsilon: T) -> Option<Self>
    where
        T: RealField,
    {
        let q1 = UnitQuaternion::from(self.clone());
        let q2 = UnitQuaternion::from(other.clone());
        q1.try_slerp(&q2, t, epsilon).map(|q| q.into())
    }
}
