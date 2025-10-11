use crate::storage::Storage;
use crate::{
    Allocator, DefaultAllocator, Dim, OVector, One, RealField, Scalar, Unit, Vector, Zero,
};
use simba::scalar::{ClosedAddAssign, ClosedMulAssign, ClosedSubAssign};

/// # Interpolation
impl<
    T: Scalar + Zero + One + ClosedAddAssign + ClosedSubAssign + ClosedMulAssign,
    D: Dim,
    S: Storage<T, D>,
> Vector<T, D, S>
{
    /// Performs linear interpolation (lerp) between two vectors.
    ///
    /// Linear interpolation smoothly transitions from one vector to another based on a
    /// parameter `t`. When `t = 0.0`, the result equals `self`. When `t = 1.0`, the result
    /// equals `rhs`. Values between 0.0 and 1.0 produce intermediate vectors along the
    /// straight line connecting `self` and `rhs`.
    ///
    /// The mathematical formula is: `self * (1 - t) + rhs * t`
    ///
    /// # Parameters
    ///
    /// * `rhs` - The target vector to interpolate towards
    /// * `t` - The interpolation parameter (not restricted to [0, 1])
    ///
    /// # Interpolation Parameter
    ///
    /// While `t` is typically in the range [0, 1] for interpolation:
    /// - `t < 0.0` extrapolates beyond `self` in the opposite direction of `rhs`
    /// - `t > 1.0` extrapolates beyond `rhs`
    ///
    /// # Examples
    ///
    /// Basic interpolation between two 3D vectors:
    ///
    /// ```
    /// # use nalgebra::Vector3;
    /// let start = Vector3::new(0.0, 0.0, 0.0);
    /// let end = Vector3::new(10.0, 20.0, 30.0);
    ///
    /// // At t=0.0, result equals start
    /// assert_eq!(start.lerp(&end, 0.0), start);
    ///
    /// // At t=1.0, result equals end
    /// assert_eq!(start.lerp(&end, 1.0), end);
    ///
    /// // At t=0.5, result is midpoint
    /// assert_eq!(start.lerp(&end, 0.5), Vector3::new(5.0, 10.0, 15.0));
    /// ```
    ///
    /// Interpolation with non-zero start vector:
    ///
    /// ```
    /// # use nalgebra::Vector3;
    /// let x = Vector3::new(1.0, 2.0, 3.0);
    /// let y = Vector3::new(10.0, 20.0, 30.0);
    ///
    /// // 10% of the way from x to y
    /// assert_eq!(x.lerp(&y, 0.1), Vector3::new(1.9, 3.8, 5.7));
    /// ```
    ///
    /// Extrapolation using values outside [0, 1]:
    ///
    /// ```
    /// # use nalgebra::Vector2;
    /// let a = Vector2::new(0.0, 0.0);
    /// let b = Vector2::new(10.0, 10.0);
    ///
    /// // Extrapolate beyond b (t > 1.0)
    /// assert_eq!(a.lerp(&b, 2.0), Vector2::new(20.0, 20.0));
    ///
    /// // Extrapolate before a (t < 0.0)
    /// assert_eq!(a.lerp(&b, -1.0), Vector2::new(-10.0, -10.0));
    /// ```
    ///
    /// Practical example - animating a position:
    ///
    /// ```
    /// # use nalgebra::Vector3;
    /// // Animate an object from start to end position over time
    /// let start_pos = Vector3::new(0.0, 0.0, 0.0);
    /// let end_pos = Vector3::new(100.0, 50.0, 0.0);
    ///
    /// // time_elapsed is between 0.0 and animation_duration
    /// let animation_duration = 2.0; // seconds
    /// let time_elapsed = 0.5; // seconds
    /// let t = time_elapsed / animation_duration;
    ///
    /// let current_pos = start_pos.lerp(&end_pos, t);
    /// assert_eq!(current_pos, Vector3::new(25.0, 12.5, 0.0));
    /// ```
    ///
    /// # See Also
    ///
    /// * [`slerp`](Self::slerp) - Spherical linear interpolation for smoother rotation-like interpolation
    #[must_use]
    pub fn lerp<S2: Storage<T, D>>(&self, rhs: &Vector<T, D, S2>, t: T) -> OVector<T, D>
    where
        DefaultAllocator: Allocator<D>,
    {
        let mut res = self.clone_owned();
        res.axpy(t.clone(), rhs, T::one() - t);
        res
    }

    /// Performs spherical linear interpolation (slerp) between two non-zero vectors.
    ///
    /// Spherical linear interpolation (slerp) provides smooth interpolation between vectors
    /// while maintaining constant angular velocity. Unlike linear interpolation (`lerp`),
    /// which creates a straight-line path, slerp follows the arc of a great circle on the
    /// unit sphere. This makes it particularly useful for interpolating rotations and
    /// directions where maintaining constant speed is important.
    ///
    /// The input vectors are automatically normalized before interpolation, and the result
    /// is returned as a normalized (unit length) vector.
    ///
    /// # Parameters
    ///
    /// * `rhs` - The target vector to interpolate towards
    /// * `t` - The interpolation parameter (typically in [0, 1])
    ///   - `t = 0.0` returns the normalized `self`
    ///   - `t = 1.0` returns the normalized `rhs`
    ///   - Values between 0.0 and 1.0 interpolate along the spherical arc
    ///
    /// # Why Use Slerp?
    ///
    /// Slerp is preferred over lerp when:
    /// - You need constant angular velocity during interpolation
    /// - You're interpolating directions or orientations
    /// - You want the interpolated path to follow a circular arc
    ///
    /// For simple position interpolation where straight-line paths are acceptable,
    /// `lerp` is more efficient.
    ///
    /// # Examples
    ///
    /// Basic spherical interpolation between two 2D vectors:
    ///
    /// ```
    /// # use nalgebra::Vector2;
    /// let v1 = Vector2::new(1.0, 0.0);  // Points right
    /// let v2 = Vector2::new(0.0, 1.0);  // Points up
    ///
    /// // At t=0.0, result points in v1's direction
    /// let result = v1.slerp(&v2, 0.0);
    /// assert!((result - v1.normalize()).norm() < 1e-7);
    ///
    /// // At t=1.0, result points in v2's direction
    /// let result = v1.slerp(&v2, 1.0);
    /// assert!((result - v2.normalize()).norm() < 1e-7);
    ///
    /// // At t=0.5, result is halfway along the circular arc
    /// let result = v1.slerp(&v2, 0.5);
    /// let expected = Vector2::new(1.0, 1.0).normalize(); // 45 degrees
    /// assert!((result - expected).norm() < 1e-7);
    /// ```
    ///
    /// Interpolating with non-unit vectors:
    ///
    /// ```
    /// # use nalgebra::Vector2;
    /// let v1 = Vector2::new(1.0_f64, 2.0);
    /// let v2 = Vector2::new(2.0_f64, -3.0);
    ///
    /// // Vectors are automatically normalized
    /// let v = v1.slerp(&v2, 1.0);
    /// assert_eq!(v, v2.normalize());
    ///
    /// // Result is always a unit vector
    /// let v_mid = v1.slerp(&v2, 0.5);
    /// assert!((v_mid.norm() - 1.0).abs() < 1e-7);
    /// ```
    ///
    /// Comparing slerp vs lerp for direction interpolation:
    ///
    /// ```
    /// # use nalgebra::Vector2;
    /// let v1 = Vector2::new(1.0_f64, 0.0);
    /// let v2 = Vector2::new(0.0_f64, 1.0);
    ///
    /// // Slerp maintains unit length throughout interpolation
    /// let slerp_mid = v1.slerp(&v2, 0.5);
    /// assert!((slerp_mid.norm() - 1.0).abs() < 1e-7);
    ///
    /// // Lerp does not maintain unit length (gets shorter in the middle)
    /// let lerp_mid = v1.lerp(&v2, 0.5);
    /// assert!(lerp_mid.norm() < 1.0);  // Length is approximately 0.707
    /// ```
    ///
    /// Practical example - smoothly rotating a camera direction:
    ///
    /// ```
    /// # use nalgebra::Vector3;
    /// // Camera starts looking forward, rotates to look right
    /// let look_forward = Vector3::new(0.0_f64, 0.0, -1.0);
    /// let look_right = Vector3::new(1.0_f64, 0.0, 0.0);
    ///
    /// // Smoothly interpolate camera direction over time
    /// let t = 0.25;  // 25% through the rotation
    /// let current_direction = look_forward.slerp(&look_right, t);
    ///
    /// // Direction is always normalized and follows a smooth arc
    /// assert!((current_direction.norm() - 1.0).abs() < 1e-7);
    /// ```
    ///
    /// # See Also
    ///
    /// * [`lerp`](Self::lerp) - Linear interpolation (faster but doesn't maintain constant angular velocity)
    /// * [`Unit::slerp`] - Spherical interpolation for unit vectors (more efficient when inputs are already normalized)
    #[must_use]
    pub fn slerp<S2: Storage<T, D>>(&self, rhs: &Vector<T, D, S2>, t: T) -> OVector<T, D>
    where
        T: RealField,
        DefaultAllocator: Allocator<D>,
    {
        let me = Unit::new_normalize(self.clone_owned());
        let rhs = Unit::new_normalize(rhs.clone_owned());
        me.slerp(&rhs, t).into_inner()
    }
}

/// # Interpolation between two unit vectors
impl<T: RealField, D: Dim, S: Storage<T, D>> Unit<Vector<T, D, S>> {
    /// Performs spherical linear interpolation (slerp) between two unit vectors.
    ///
    /// This is a more efficient version of `Vector::slerp` for when your vectors are
    /// already normalized (unit length). Since the inputs are guaranteed to be unit vectors,
    /// this method skips the normalization step, making it faster.
    ///
    /// Slerp provides smooth interpolation along the surface of a sphere, maintaining
    /// constant angular velocity. This makes it ideal for interpolating rotations,
    /// orientations, and directions in 3D graphics, robotics, and animation.
    ///
    /// # Parameters
    ///
    /// * `rhs` - The target unit vector to interpolate towards
    /// * `t` - The interpolation parameter (typically in [0, 1])
    ///   - `t = 0.0` returns `self`
    ///   - `t = 1.0` returns `rhs`
    ///   - Values between 0.0 and 1.0 interpolate along the great circle arc
    ///
    /// # Behavior with Opposite Vectors
    ///
    /// When the two vectors are nearly opposite (pointing in opposite directions), there
    /// are infinitely many shortest paths between them on the sphere. In this case, the
    /// method returns `self` unchanged. For more control over this edge case, use
    /// [`try_slerp`](Self::try_slerp) which returns `None` in such situations.
    ///
    /// # Examples
    ///
    /// Basic usage with unit vectors:
    ///
    /// ```
    /// # use nalgebra::{Unit, Vector2};
    /// let v1 = Unit::new_normalize(Vector2::new(1.0, 0.0));
    /// let v2 = Unit::new_normalize(Vector2::new(0.0, 1.0));
    ///
    /// // At t=0.0, returns v1
    /// let result = v1.slerp(&v2, 0.0);
    /// assert_eq!(result, v1);
    ///
    /// // At t=1.0, returns v2
    /// let result = v1.slerp(&v2, 1.0);
    /// assert_eq!(result, v2);
    ///
    /// // At t=0.5, returns halfway point along the arc
    /// let result = v1.slerp(&v2, 0.5);
    /// let expected = Unit::new_normalize(Vector2::new(1.0, 1.0));
    /// assert!((result.into_inner() - expected.into_inner()).norm() < 1e-7);
    /// ```
    ///
    /// Interpolating normalized vectors:
    ///
    /// ```
    /// # use nalgebra::{Unit, Vector2};
    /// let v1 = Unit::new_normalize(Vector2::new(1.0_f64, 2.0));
    /// let v2 = Unit::new_normalize(Vector2::new(2.0_f64, -3.0));
    ///
    /// let v = v1.slerp(&v2, 1.0);
    /// assert_eq!(v, v2);
    ///
    /// // Result is always a unit vector
    /// let v_mid = v1.slerp(&v2, 0.5);
    /// assert!((v_mid.norm() - 1.0).abs() < 1e-7);
    /// ```
    ///
    /// Interpolating 3D directions (e.g., for camera orientation):
    ///
    /// ```
    /// # use nalgebra::{Unit, Vector3};
    /// let forward = Unit::new_normalize(Vector3::new(0.0, 0.0, -1.0));
    /// let right = Unit::new_normalize(Vector3::new(1.0, 0.0, 0.0));
    ///
    /// // Smoothly rotate from forward to right
    /// for i in 0..=10 {
    ///     let t = i as f64 / 10.0;
    ///     let direction = forward.slerp(&right, t);
    ///     // Each direction is guaranteed to be unit length
    ///     assert!((direction.norm() - 1.0).abs() < 1e-7);
    /// }
    /// ```
    ///
    /// Comparing with Vector::slerp:
    ///
    /// ```
    /// # use nalgebra::{Unit, Vector2};
    /// let v1 = Vector2::new(1.0, 0.0);
    /// let v2 = Vector2::new(0.0, 1.0);
    ///
    /// // Using Vector::slerp (normalizes internally)
    /// let result1 = v1.slerp(&v2, 0.5);
    ///
    /// // Using Unit::slerp (more efficient for pre-normalized vectors)
    /// let u1 = Unit::new_normalize(v1);
    /// let u2 = Unit::new_normalize(v2);
    /// let result2 = u1.slerp(&u2, 0.5);
    ///
    /// // Results are equivalent
    /// assert!((result1 - result2.into_inner()).norm() < 1e-7);
    /// ```
    ///
    /// # See Also
    ///
    /// * [`try_slerp`](Self::try_slerp) - Returns `None` for opposite vectors instead of falling back to `self`
    /// * [`Vector::slerp`] - Slerp for non-unit vectors (includes automatic normalization)
    #[must_use]
    pub fn slerp<S2: Storage<T, D>>(
        &self,
        rhs: &Unit<Vector<T, D, S2>>,
        t: T,
    ) -> Unit<OVector<T, D>>
    where
        DefaultAllocator: Allocator<D>,
    {
        // TODO: the result is wrong when self and rhs are collinear with opposite direction.
        self.try_slerp(rhs, t, T::default_epsilon())
            .unwrap_or_else(|| Unit::new_unchecked(self.clone_owned()))
    }

    /// Attempts spherical linear interpolation (slerp) between two unit vectors.
    ///
    /// This is a fallible version of [`slerp`](Self::slerp) that returns `None` when the
    /// interpolation is ambiguous or undefined. This occurs when the two vectors are nearly
    /// opposite (collinear with opposite directions), where there are infinitely many
    /// valid shortest paths between them on the sphere.
    ///
    /// Use this method when you need to explicitly handle the edge case of opposite vectors,
    /// rather than silently falling back to returning `self` as the regular `slerp` does.
    ///
    /// # Parameters
    ///
    /// * `rhs` - The target unit vector to interpolate towards
    /// * `t` - The interpolation parameter (typically in [0, 1])
    /// * `epsilon` - The tolerance for detecting collinear opposite vectors
    ///
    /// # Returns
    ///
    /// * `Some(Unit<Vector>)` - The interpolated unit vector if interpolation is well-defined
    /// * `None` - If the vectors are nearly opposite (within `epsilon` tolerance)
    ///
    /// # When Does This Return None?
    ///
    /// The function returns `None` when the dot product of the two vectors indicates they
    /// are approximately opposite. This happens when the angle between them is close to 180
    /// degrees (π radians). In this case, there's no unique "shortest path" on the sphere.
    ///
    /// # Examples
    ///
    /// Successful interpolation:
    ///
    /// ```
    /// # use nalgebra::{Unit, Vector2};
    /// let v1 = Unit::new_normalize(Vector2::new(1.0_f64, 0.0));
    /// let v2 = Unit::new_normalize(Vector2::new(0.0_f64, 1.0));
    ///
    /// // These vectors are perpendicular, so interpolation succeeds
    /// let result = v1.try_slerp(&v2, 0.5, 1e-7);
    /// assert!(result.is_some());
    ///
    /// let interpolated = result.unwrap();
    /// assert!((interpolated.norm() - 1.0).abs() < 1e-7);
    /// ```
    ///
    /// Handling opposite vectors:
    ///
    /// ```
    /// # use nalgebra::{Unit, Vector3};
    /// let v1 = Unit::new_normalize(Vector3::new(1.0, 0.0, 0.0));
    /// let v2 = Unit::new_normalize(Vector3::new(-1.0, 0.0, 0.0));
    ///
    /// // These vectors are opposite, so interpolation is undefined
    /// let result = v1.try_slerp(&v2, 0.5, 1e-7);
    /// assert!(result.is_none());
    /// ```
    ///
    /// Using different epsilon values:
    ///
    /// ```
    /// # use nalgebra::{Unit, Vector2};
    /// let v1 = Unit::new_normalize(Vector2::new(1.0, 0.0));
    /// let v2 = Unit::new_normalize(Vector2::new(-0.999, -0.001));
    ///
    /// // With a small epsilon, nearly-opposite vectors might succeed
    /// let result1 = v1.try_slerp(&v2, 0.5, 1e-10);
    /// assert!(result1.is_some());
    ///
    /// // With a larger epsilon, they're considered opposite
    /// let result2 = v1.try_slerp(&v2, 0.5, 1e-1);
    /// assert!(result2.is_none());
    /// ```
    ///
    /// Robust interpolation with fallback:
    ///
    /// ```
    /// # use nalgebra::{Unit, Vector3};
    /// fn safe_slerp(
    ///     v1: &Unit<Vector3<f64>>,
    ///     v2: &Unit<Vector3<f64>>,
    ///     t: f64
    /// ) -> Unit<Vector3<f64>> {
    ///     v1.try_slerp(v2, t, 1e-7).unwrap_or_else(|| {
    ///         // Handle opposite vectors by choosing an arbitrary perpendicular path
    ///         // In practice, you might want a more sophisticated fallback
    ///         *v1
    ///     })
    /// }
    ///
    /// let forward = Unit::new_normalize(Vector3::new(0.0, 0.0, 1.0));
    /// let backward = Unit::new_normalize(Vector3::new(0.0, 0.0, -1.0));
    ///
    /// // Function handles opposite vectors gracefully
    /// let result = safe_slerp(&forward, &backward, 0.5);
    /// assert_eq!(result, forward);
    /// ```
    ///
    /// Practical example - camera interpolation with error handling:
    ///
    /// ```
    /// # use nalgebra::{Unit, Vector3};
    /// fn interpolate_camera_direction(
    ///     from: Unit<Vector3<f64>>,
    ///     to: Unit<Vector3<f64>>,
    ///     progress: f64,
    /// ) -> Option<Unit<Vector3<f64>>> {
    ///     // Try to interpolate, return None if directions are opposite
    ///     from.try_slerp(&to, progress, 1e-6)
    /// }
    ///
    /// let look_forward = Unit::new_normalize(Vector3::new(0.0, 0.0, -1.0));
    /// let look_right = Unit::new_normalize(Vector3::new(1.0, 0.0, 0.0));
    ///
    /// match interpolate_camera_direction(look_forward, look_right, 0.5) {
    ///     Some(direction) => println!("Camera direction interpolated successfully"),
    ///     None => println!("Cannot interpolate opposite directions"),
    /// }
    /// ```
    ///
    /// # See Also
    ///
    /// * [`slerp`](Self::slerp) - Always returns a result, falling back to `self` for opposite vectors
    /// * [`Vector::slerp`] - Slerp for non-unit vectors (includes automatic normalization)
    #[must_use]
    pub fn try_slerp<S2: Storage<T, D>>(
        &self,
        rhs: &Unit<Vector<T, D, S2>>,
        t: T,
        epsilon: T,
    ) -> Option<Unit<OVector<T, D>>>
    where
        DefaultAllocator: Allocator<D>,
    {
        let c_hang = self.dot(rhs);

        // self == other
        if c_hang >= T::one() {
            return Some(Unit::new_unchecked(self.clone_owned()));
        }

        let hang = c_hang.clone().acos();
        let s_hang = (T::one() - c_hang.clone() * c_hang).sqrt();

        // TODO: what if s_hang is 0.0 ? The result is not well-defined.
        if relative_eq!(s_hang, T::zero(), epsilon = epsilon) {
            None
        } else {
            let ta = ((T::one() - t.clone()) * hang.clone()).sin() / s_hang.clone();
            let tb = (t * hang).sin() / s_hang;
            let mut res = self.scale(ta);
            res.axpy(tb, &**rhs, T::one());

            Some(Unit::new_unchecked(res))
        }
    }
}
