#[cfg(feature = "arbitrary")]
use crate::base::storage::Owned;
#[cfg(feature = "arbitrary")]
use quickcheck::{Arbitrary, Gen};

use num::Zero;

#[cfg(feature = "rand-no-std")]
use rand::{
    Rng,
    distr::{Distribution, OpenClosed01, StandardUniform, Uniform, uniform::SampleUniform},
};

use simba::scalar::RealField;
use simba::simd::{SimdBool, SimdRealField};
use std::ops::Neg;

use crate::base::dimension::{U1, U2, U3};
use crate::base::storage::Storage;
use crate::base::{
    Matrix2, Matrix3, SMatrix, SVector, Unit, UnitVector3, Vector, Vector1, Vector2, Vector3,
};

use crate::geometry::{Rotation2, Rotation3, UnitComplex, UnitQuaternion};

/*
 *
 * 2D Rotation matrix.
 *
 */
/// # Construction from a 2D rotation angle
impl<T: SimdRealField> Rotation2<T> {
    /// Creates a 2D rotation matrix from an angle in radians.
    ///
    /// This is the primary way to create a 2D rotation. The rotation is counterclockwise
    /// when viewed from above (following the right-hand rule with the Z axis pointing up).
    /// A positive angle rotates from the positive X axis toward the positive Y axis.
    ///
    /// # Parameters
    ///
    /// * `angle` - The rotation angle in radians. Positive values rotate counterclockwise.
    ///
    /// # Returns
    ///
    /// A 2×2 rotation matrix representing the rotation
    ///
    /// # Mathematical Representation
    ///
    /// The resulting matrix is:
    /// ```text
    /// [ cos(θ)  -sin(θ) ]
    /// [ sin(θ)   cos(θ) ]
    /// ```
    ///
    /// # Examples
    ///
    /// ## Basic 2D rotation
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use std::f32;
    /// # use nalgebra::{Rotation2, Point2};
    /// // Rotate 90 degrees counterclockwise
    /// let rot = Rotation2::new(f32::consts::FRAC_PI_2);
    ///
    /// // Point (3, 4) rotates to approximately (-4, 3)
    /// let point = Point2::new(3.0, 4.0);
    /// let rotated = rot * point;
    /// assert_relative_eq!(rotated, Point2::new(-4.0, 3.0), epsilon = 1.0e-6);
    /// ```
    ///
    /// ## Game sprite rotation
    /// ```
    /// # use std::f32;
    /// # use nalgebra::{Rotation2, Vector2};
    /// // Rotate sprite 45 degrees (π/4 radians)
    /// let angle = f32::consts::FRAC_PI_4;
    /// let rotation = Rotation2::new(angle);
    ///
    /// // Rotate the sprite's forward direction
    /// let forward = Vector2::new(1.0, 0.0);
    /// let rotated_forward = rotation * forward;
    ///
    /// // Now pointing at 45 degrees
    /// assert!((rotated_forward.x - 0.7071068).abs() < 1e-6);
    /// assert!((rotated_forward.y - 0.7071068).abs() < 1e-6);
    /// ```
    ///
    /// ## Physics simulation: Angular motion
    /// ```
    /// # use nalgebra::{Rotation2, Point2};
    /// // Simulate rotation over time
    /// let angular_velocity = 1.0; // radians per second
    /// let delta_time = 0.1; // seconds
    /// let angle_change = angular_velocity * delta_time;
    ///
    /// let rotation = Rotation2::new(angle_change);
    ///
    /// // Update object position
    /// let mut position = Point2::new(5.0, 0.0);
    /// position = rotation * position;
    ///
    /// // Position has rotated slightly counterclockwise
    /// assert!((position.x - 4.975021).abs() < 1e-5);
    /// assert!((position.y - 0.499167).abs() < 1e-5);
    /// ```
    ///
    /// ## Converting degrees to radians
    /// ```
    /// # use nalgebra::Rotation2;
    /// // Most game engines use degrees, but nalgebra uses radians
    /// let degrees = 30.0;
    /// let radians = degrees.to_radians();
    /// let rotation = Rotation2::new(radians);
    ///
    /// // Can also use the conversion directly
    /// let rotation2 = Rotation2::new(30.0_f32.to_radians());
    /// ```
    ///
    /// # See Also
    /// - [`angle`](Self::angle) - Extract the angle from a rotation
    /// - [`rotation_between`](Self::rotation_between) - Create rotation between two vectors
    pub fn new(angle: T) -> Self {
        let (sia, coa) = angle.simd_sin_cos();
        Self::from_matrix_unchecked(Matrix2::new(coa.clone(), -sia.clone(), sia, coa))
    }

    /// Builds a 2D rotation from an angle wrapped in a 1-dimensional vector.
    ///
    /// This is equivalent to calling [`new(angle)`](Self::new) but takes the angle as a 1D vector
    /// instead of a scalar. This is primarily useful for generic programming where you want
    /// consistent behavior between 2D and 3D rotations.
    ///
    /// For 2D rotations, there's only one axis (perpendicular to the plane), so the "axis-angle"
    /// representation simplifies to just an angle. The 1D vector holds this single value.
    ///
    /// # Parameters
    ///
    /// * `axisangle` - A 1D vector containing the rotation angle in radians
    ///
    /// # Examples
    ///
    /// ## Basic usage
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{Rotation2, Vector1};
    /// // Create rotation from 1D vector
    /// let angle_vec = Vector1::new(1.5);
    /// let rot = Rotation2::from_scaled_axis(angle_vec);
    ///
    /// // This is the same as using new()
    /// let rot2 = Rotation2::new(1.5);
    /// assert_eq!(rot, rot2);
    /// ```
    ///
    /// ## Generic programming example
    /// ```
    /// # use nalgebra::{Rotation2, Rotation3, Vector1, Vector3};
    /// // Function that works with both 2D and 3D rotations
    /// fn create_rotation_2d() -> Rotation2<f64> {
    ///     // In generic code, you might receive axis-angle vectors
    ///     let axis_angle = Vector1::new(0.5);
    ///     Rotation2::from_scaled_axis(axis_angle)
    /// }
    ///
    /// fn create_rotation_3d() -> Rotation3<f64> {
    ///     // 3D version uses a 3D vector
    ///     let axis_angle = Vector3::new(0.0, 0.5, 0.0);
    ///     Rotation3::from_scaled_axis(axis_angle)
    /// }
    /// ```
    ///
    /// ## Extracting and recreating rotations
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::Rotation2;
    /// let original = Rotation2::new(2.0);
    ///
    /// // Extract axis-angle representation
    /// let axis_angle = original.scaled_axis();
    ///
    /// // Recreate the rotation
    /// let recreated = Rotation2::from_scaled_axis(axis_angle);
    ///
    /// assert_relative_eq!(original, recreated, epsilon = 1.0e-6);
    /// ```
    ///
    /// # See Also
    /// - [`new`](Self::new) - Create rotation from angle directly (recommended for most cases)
    /// - [`scaled_axis`](Self::scaled_axis) - Extract angle as 1D vector
    #[inline]
    pub fn from_scaled_axis<SB: Storage<T, U1>>(axisangle: Vector<T, U1, SB>) -> Self {
        Self::new(axisangle[0].clone())
    }
}

/// # Construction from an existing 2D matrix or rotations
impl<T: SimdRealField> Rotation2<T> {
    /// Builds a 2D rotation from two basis vectors without checking orthonormality.
    ///
    /// This constructs a rotation matrix directly from two 2D vectors that represent the
    /// rotated X and Y axes. For this to produce a valid rotation, the basis vectors **must**
    /// be orthonormal (perpendicular to each other and unit length).
    ///
    /// **Warning:** This method does **not** verify the orthonormality constraints. Using
    /// non-orthonormal vectors will create an invalid rotation matrix that may cause
    /// unexpected behavior in your code.
    ///
    /// # Parameters
    ///
    /// * `basis` - An array of two 2D vectors representing the rotated coordinate axes.
    ///   - `basis[0]` is the rotated X axis
    ///   - `basis[1]` is the rotated Y axis
    ///
    /// # Safety Requirements (unchecked)
    ///
    /// The caller must ensure:
    /// 1. Both vectors have unit length (magnitude = 1)
    /// 2. The vectors are perpendicular (dot product = 0)
    /// 3. The vectors form a right-handed coordinate system
    ///
    /// # Examples
    ///
    /// ## Valid orthonormal basis
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use std::f32;
    /// # use nalgebra::{Rotation2, Vector2};
    /// // Create basis vectors for 45-degree rotation
    /// let cos_45 = f32::consts::FRAC_1_SQRT_2;
    /// let sin_45 = f32::consts::FRAC_1_SQRT_2;
    ///
    /// let x_axis = Vector2::new(cos_45, sin_45);   // Rotated X axis
    /// let y_axis = Vector2::new(-sin_45, cos_45);  // Rotated Y axis
    ///
    /// let rot = Rotation2::from_basis_unchecked(&[x_axis, y_axis]);
    ///
    /// // Verify it matches the expected rotation
    /// let expected = Rotation2::new(f32::consts::FRAC_PI_4);
    /// assert_relative_eq!(rot, expected, epsilon = 1.0e-6);
    /// ```
    ///
    /// ## Using rotated coordinate axes
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{Rotation2, Vector2};
    /// // Start with standard axes
    /// let original_x = Vector2::x();
    /// let original_y = Vector2::y();
    ///
    /// // Rotate them by some amount
    /// let rotation_angle = 1.2;
    /// let rot = Rotation2::new(rotation_angle);
    /// let rotated_x = rot * original_x;
    /// let rotated_y = rot * original_y;
    ///
    /// // Reconstruct the rotation from the rotated axes
    /// let reconstructed = Rotation2::from_basis_unchecked(&[rotated_x, rotated_y]);
    ///
    /// assert_relative_eq!(rot, reconstructed, epsilon = 1.0e-6);
    /// ```
    ///
    /// ## When to use from_matrix instead
    /// ```
    /// # use nalgebra::{Rotation2, Vector2, Matrix2};
    /// // If you have vectors that might not be orthonormal
    /// let v1 = Vector2::new(1.0, 0.1);  // Not unit length
    /// let v2 = Vector2::new(-0.1, 1.0); // Not exactly perpendicular
    ///
    /// // Don't use from_basis_unchecked - use from_matrix instead
    /// let mat = Matrix2::from_columns(&[v1, v2]);
    /// let rot = Rotation2::from_matrix(&mat);
    /// // from_matrix will find the closest valid rotation
    /// ```
    ///
    /// # See Also
    /// - [`from_matrix`](Self::from_matrix) - Extract rotation from any matrix (checks validity)
    /// - [`new`](Self::new) - Create rotation from angle (recommended for most cases)
    pub fn from_basis_unchecked(basis: &[Vector2<T>; 2]) -> Self {
        let mat = Matrix2::from_columns(&basis[..]);
        Self::from_matrix_unchecked(mat)
    }

    /// Extracts the rotation component from any 2D transformation matrix.
    ///
    /// This method takes any 2×2 matrix (which might include scaling, shearing, or other
    /// transformations) and finds the closest pure rotation matrix. This is useful when you
    /// have a matrix from an external source or from accumulated transformations that may
    /// have drifted from being a pure rotation.
    ///
    /// The method uses an iterative algorithm based on "A Robust Method to Extract the
    /// Rotational Part of Deformations" by Müller et al. It converges to the rotation
    /// matrix that best approximates the input.
    ///
    /// # Parameters
    ///
    /// * `m` - Any 2×2 matrix from which to extract the rotation
    ///
    /// # Examples
    ///
    /// ## Extracting rotation from a scaled rotation
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use std::f32;
    /// # use nalgebra::{Rotation2, Matrix2};
    /// // Create a matrix that combines rotation and scaling
    /// let angle = f32::consts::FRAC_PI_4; // 45 degrees
    /// let scale = 2.0;
    /// let cos = angle.cos() * scale;
    /// let sin = angle.sin() * scale;
    /// let scaled_rotation = Matrix2::new(cos, -sin, sin, cos);
    ///
    /// // Extract just the rotation part
    /// let rotation = Rotation2::from_matrix(&scaled_rotation);
    ///
    /// // Result is a pure rotation (no scaling)
    /// assert_relative_eq!(rotation.angle(), f32::consts::FRAC_PI_4, epsilon = 1.0e-6);
    /// ```
    ///
    /// ## Cleaning up numerical drift
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::Rotation2;
    /// // After many operations, a rotation matrix might drift slightly
    /// let mut rot = Rotation2::new(1.5);
    ///
    /// // Simulate many operations that introduce small errors
    /// for _ in 0..1000 {
    ///     rot = rot * Rotation2::new(0.001);
    /// }
    ///
    /// // Extract a clean rotation matrix
    /// let cleaned = Rotation2::from_matrix(rot.matrix());
    ///
    /// // Now we have a proper rotation again
    /// assert_relative_eq!(cleaned.angle(), rot.angle(), epsilon = 1.0e-4);
    /// ```
    ///
    /// ## Rotation from arbitrary transformation
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{Rotation2, Matrix2};
    /// // Matrix with rotation, scale, and slight shear
    /// let matrix = Matrix2::new(
    ///     1.5, -0.3,
    ///     0.2,  1.4
    /// );
    ///
    /// // Extract the rotational component
    /// let rotation = Rotation2::from_matrix(&matrix);
    ///
    /// // The result is a pure rotation (determinant = 1, orthogonal)
    /// assert_relative_eq!(rotation.matrix().determinant(), 1.0, epsilon = 1.0e-6);
    /// ```
    ///
    /// # Performance Notes
    ///
    /// This method uses an iterative algorithm that typically converges quickly but may
    /// take longer for matrices that are far from being rotations. For fine-grained control
    /// over convergence, use [`from_matrix_eps`](Self::from_matrix_eps).
    ///
    /// # See Also
    /// - [`from_matrix_eps`](Self::from_matrix_eps) - Same but with custom convergence parameters
    /// - [`renormalize`](Self::renormalize) - Clean up an existing rotation matrix
    /// - [`from_basis_unchecked`](Self::from_basis_unchecked) - Fast construction if you know the matrix is already a rotation
    pub fn from_matrix(m: &Matrix2<T>) -> Self
    where
        T: RealField,
    {
        Self::from_matrix_eps(m, T::default_epsilon(), 0, Self::identity())
    }

    /// Extracts the rotation from a matrix with custom convergence parameters.
    ///
    /// This is the advanced version of [`from_matrix`](Self::from_matrix) that gives you control
    /// over the iterative algorithm's convergence criteria and starting point. This is useful
    /// when you need specific accuracy guarantees or when you have a good initial guess that
    /// can speed up convergence.
    ///
    /// The algorithm iteratively refines a rotation estimate until it's within the specified
    /// tolerance or reaches the maximum iteration count. It implements "A Robust Method to
    /// Extract the Rotational Part of Deformations" by Müller et al.
    ///
    /// # Parameters
    ///
    /// * `m` - The matrix from which to extract the rotation
    /// * `eps` - Convergence threshold: iteration stops when angular error is below this (in radians)
    /// * `max_iter` - Maximum number of iterations (use `0` for unlimited iterations until convergence)
    /// * `guess` - Initial estimate of the rotation (use `Rotation2::identity()` if unknown)
    ///
    /// # Examples
    ///
    /// ## Basic usage with custom tolerance
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{Rotation2, Matrix2};
    /// let matrix = Matrix2::new(
    ///     1.5, -0.5,
    ///     0.5,  1.5
    /// );
    ///
    /// // Extract rotation with tight tolerance
    /// let rotation = Rotation2::from_matrix_eps(
    ///     &matrix,
    ///     1.0e-8,  // Very tight tolerance
    ///     100,     // Up to 100 iterations
    ///     Rotation2::identity()  // Start from identity
    /// );
    ///
    /// // Result is a pure rotation
    /// assert_relative_eq!(rotation.matrix().determinant(), 1.0, epsilon = 1.0e-6);
    /// ```
    ///
    /// ## Using an initial guess for faster convergence
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use std::f32;
    /// # use nalgebra::{Rotation2, Matrix2};
    /// // Matrix is approximately a 45-degree rotation with some noise
    /// let angle = f32::consts::FRAC_PI_4;
    /// let noise = 0.1;
    /// let matrix = Matrix2::new(
    ///     angle.cos() + noise, -angle.sin(),
    ///     angle.sin(), angle.cos() - noise
    /// );
    ///
    /// // Provide a good initial guess
    /// let guess = Rotation2::new(f32::consts::FRAC_PI_4);
    ///
    /// // This will converge much faster than starting from identity
    /// let rotation = Rotation2::from_matrix_eps(
    ///     &matrix,
    ///     1.0e-6,
    ///     50,
    ///     guess
    /// );
    ///
    /// assert_relative_eq!(rotation.angle(), f32::consts::FRAC_PI_4, epsilon = 0.15);
    /// ```
    ///
    /// ## Controlling iteration count
    /// ```
    /// # use nalgebra::{Rotation2, Matrix2};
    /// let matrix = Matrix2::new(2.0, 0.0, 0.0, 2.0);  // Pure scaling
    ///
    /// // Limit iterations for real-time applications
    /// let rotation = Rotation2::from_matrix_eps(
    ///     &matrix,
    ///     1.0e-6,
    ///     10,  // Maximum 10 iterations
    ///     Rotation2::identity()
    /// );
    ///
    /// // Will give best result within iteration budget
    /// ```
    ///
    /// ## Unlimited iterations until convergence
    /// ```
    /// # use nalgebra::{Rotation2, Matrix2};
    /// let matrix = Matrix2::new(1.1, -0.2, 0.2, 1.1);
    ///
    /// // Use 0 for max_iter to run until convergence
    /// let rotation = Rotation2::from_matrix_eps(
    ///     &matrix,
    ///     1.0e-10,  // Very tight tolerance
    ///     0,        // No iteration limit
    ///     Rotation2::identity()
    /// );
    /// ```
    ///
    /// # Performance Tips
    ///
    /// - Provide a good initial `guess` if you have one - it dramatically speeds up convergence
    /// - Use a looser `eps` (e.g., 1.0e-6) for real-time applications
    /// - Set `max_iter` to a reasonable limit in performance-critical code
    ///
    /// # See Also
    /// - [`from_matrix`](Self::from_matrix) - Simple version with default parameters
    /// - [`renormalize`](Self::renormalize) - Orthonormalize an existing rotation
    pub fn from_matrix_eps(m: &Matrix2<T>, eps: T, mut max_iter: usize, guess: Self) -> Self
    where
        T: RealField,
    {
        if max_iter == 0 {
            max_iter = usize::MAX;
        }

        let mut rot = guess.into_inner();

        for _ in 0..max_iter {
            let axis = rot.column(0).perp(&m.column(0)) + rot.column(1).perp(&m.column(1));
            let denom = rot.column(0).dot(&m.column(0)) + rot.column(1).dot(&m.column(1));

            let angle = axis / (denom.abs() + T::default_epsilon());
            if angle.clone().abs() > eps {
                rot = Self::new(angle) * rot;
            } else {
                break;
            }
        }

        Self::from_matrix_unchecked(rot)
    }

    /// Creates the rotation that aligns vector `a` with vector `b`.
    ///
    /// This computes the rotation matrix R such that `R * a` is aligned with `b` (pointing in
    /// the same direction). Both vectors can have any non-zero length - they don't need to be
    /// unit vectors. This is one of the most useful functions for orienting objects toward
    /// targets or aligning coordinate systems.
    ///
    /// The resulting rotation is the **shortest** rotation (smallest angle) that aligns the
    /// vectors. In 2D, this is always unique.
    ///
    /// # Parameters
    ///
    /// * `a` - The source vector (does not need to be normalized)
    /// * `b` - The target vector (does not need to be normalized)
    ///
    /// # Returns
    ///
    /// The rotation R such that `R * a` is parallel to `b` with positive dot product
    ///
    /// # Examples
    ///
    /// ## Basic vector alignment
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{Vector2, Rotation2};
    /// let a = Vector2::new(1.0, 2.0);
    /// let b = Vector2::new(2.0, 1.0);
    ///
    /// let rot = Rotation2::rotation_between(&a, &b);
    ///
    /// // After rotation, a is aligned with b
    /// assert_relative_eq!(rot * a, b, epsilon = 1.0e-6);
    ///
    /// // Inverse rotation aligns b with a
    /// assert_relative_eq!(rot.inverse() * b, a, epsilon = 1.0e-6);
    /// ```
    ///
    /// ## Game: Point sprite toward target
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{Vector2, Rotation2, Point2};
    /// // Sprite at origin, initially facing right
    /// let sprite_forward = Vector2::new(1.0, 0.0);
    ///
    /// // Target is at some position
    /// let sprite_pos = Point2::new(5.0, 5.0);
    /// let target_pos = Point2::new(8.0, 9.0);
    ///
    /// // Calculate direction to target
    /// let to_target = target_pos - sprite_pos;
    ///
    /// // Rotate sprite to face target
    /// let rotation = Rotation2::rotation_between(&sprite_forward, &to_target);
    ///
    /// // Now sprite faces the target
    /// let new_forward = rotation * sprite_forward;
    /// assert_relative_eq!(new_forward.normalize(), to_target.normalize(), epsilon = 1.0e-6);
    /// ```
    ///
    /// ## Robotics: Align robot heading
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{Vector2, Rotation2};
    /// // Robot's current heading
    /// let current_heading = Vector2::new(1.0, 0.0);
    ///
    /// // Desired heading (northeast)
    /// let target_heading = Vector2::new(1.0, 1.0);
    ///
    /// // Calculate required rotation
    /// let rotation = Rotation2::rotation_between(&current_heading, &target_heading);
    ///
    /// // Apply rotation to current heading
    /// let new_heading = rotation * current_heading;
    /// assert_relative_eq!(new_heading.normalize(), target_heading.normalize(), epsilon = 1.0e-6);
    ///
    /// // Get the rotation angle for motor control
    /// let turn_angle = rotation.angle();
    /// ```
    ///
    /// ## Coordinate system alignment
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{Vector2, Rotation2};
    /// // Align one coordinate system's X axis with another's
    /// let system_a_x = Vector2::new(3.0, 4.0);  // Can be any length
    /// let system_b_x = Vector2::new(-1.0, 1.0);
    ///
    /// let alignment = Rotation2::rotation_between(&system_a_x, &system_b_x);
    ///
    /// // Verify alignment
    /// let aligned = alignment * system_a_x;
    /// assert_relative_eq!(aligned.normalize(), system_b_x.normalize(), epsilon = 1.0e-6);
    /// ```
    ///
    /// # See Also
    /// - [`scaled_rotation_between`](Self::scaled_rotation_between) - Partial rotation (for smooth animation)
    /// - [`rotation_to`](Self::rotation_to) - Rotation between two rotation matrices
    /// - [`angle_to`](Self::angle_to) - Just the angle without the rotation matrix
    #[inline]
    pub fn rotation_between<SB, SC>(a: &Vector<T, U2, SB>, b: &Vector<T, U2, SC>) -> Self
    where
        T: RealField,
        SB: Storage<T, U2>,
        SC: Storage<T, U2>,
    {
        crate::convert(UnitComplex::rotation_between(a, b).to_rotation_matrix())
    }

    /// Creates a partial rotation that aligns vector `a` with vector `b`, scaled by factor `s`.
    ///
    /// This is like [`rotation_between`](Self::rotation_between) but applies only a fraction
    /// of the rotation. The scaling factor `s` controls how much of the full rotation to apply:
    /// - `s = 0.0` → no rotation (identity)
    /// - `s = 0.5` → halfway rotation
    /// - `s = 1.0` → full rotation (same as `rotation_between`)
    /// - `s = 2.0` → double the rotation
    ///
    /// This is perfect for smooth animations and gradual orientation changes. Think of it as
    /// rotation interpolation where `s` is the interpolation parameter.
    ///
    /// # Parameters
    ///
    /// * `a` - The source vector (does not need to be normalized)
    /// * `b` - The target vector (does not need to be normalized)
    /// * `s` - Scaling factor for the rotation (typically 0.0 to 1.0 for interpolation)
    ///
    /// # Examples
    ///
    /// ## Smooth rotation animation
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{Vector2, Rotation2};
    /// let start = Vector2::new(1.0, 0.0);
    /// let target = Vector2::new(0.0, 1.0);  // 90 degrees away
    ///
    /// // Animate from start to target over multiple frames
    /// let frame_count = 5;
    /// for i in 0..=frame_count {
    ///     let t = i as f32 / frame_count as f32;  // 0.0 to 1.0
    ///     let rotation = Rotation2::scaled_rotation_between(&start, &target, t);
    ///     let current = rotation * start;
    ///
    ///     // At t=0, current equals start
    ///     // At t=1, current equals target
    ///     // In between, current smoothly interpolates
    /// }
    /// ```
    ///
    /// ## Verification example from docs
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{Vector2, Rotation2};
    /// let a = Vector2::new(1.0, 2.0);
    /// let b = Vector2::new(2.0, 1.0);
    ///
    /// // 20% of the rotation (1/5th)
    /// let rot2 = Rotation2::scaled_rotation_between(&a, &b, 0.2);
    ///
    /// // Applying it 5 times gives the full rotation
    /// assert_relative_eq!(rot2 * rot2 * rot2 * rot2 * rot2 * a, b, epsilon = 1.0e-6);
    ///
    /// // 50% of the rotation (half)
    /// let rot5 = Rotation2::scaled_rotation_between(&a, &b, 0.5);
    ///
    /// // Applying it twice gives the full rotation
    /// assert_relative_eq!(rot5 * rot5 * a, b, epsilon = 1.0e-6);
    /// ```
    ///
    /// ## Game: Smooth enemy turret tracking
    /// ```
    /// # use nalgebra::{Vector2, Rotation2};
    /// // Turret's current aim direction
    /// let current_aim = Vector2::new(1.0, 0.0);
    ///
    /// // Player position relative to turret
    /// let player_direction = Vector2::new(1.0, 1.0);
    ///
    /// // Rotate gradually (20% per frame for smooth tracking)
    /// let turn_speed = 0.2;
    /// let rotation = Rotation2::scaled_rotation_between(
    ///     &current_aim,
    ///     &player_direction,
    ///     turn_speed
    /// );
    ///
    /// let new_aim = rotation * current_aim;
    /// // Turret smoothly tracks toward player instead of snapping instantly
    /// ```
    ///
    /// ## Robotics: Controlled servo movement
    /// ```
    /// # use nalgebra::{Vector2, Rotation2};
    /// let current_orientation = Vector2::new(1.0, 0.0);
    /// let target_orientation = Vector2::new(0.0, 1.0);
    ///
    /// // Move 30% of the way each control loop
    /// let move_fraction = 0.3;
    ///
    /// let mut orientation = current_orientation;
    /// for _ in 0..10 {  // 10 control steps
    ///     let step_rotation = Rotation2::scaled_rotation_between(
    ///         &orientation,
    ///         &target_orientation,
    ///         move_fraction
    ///     );
    ///     orientation = step_rotation * orientation;
    ///     // Servo gradually approaches target orientation
    /// }
    /// ```
    ///
    /// ## Animation easing with custom scaling
    /// ```
    /// # use nalgebra::{Vector2, Rotation2};
    /// let start = Vector2::new(1.0, 0.0);
    /// let end = Vector2::new(0.0, 1.0);
    ///
    /// // Use easing functions for non-linear interpolation
    /// let t = 0.3; // Animation progress 0-1
    /// let eased_t = t * t; // Ease-in (quadratic)
    ///
    /// let rotation = Rotation2::scaled_rotation_between(&start, &end, eased_t);
    /// let current = rotation * start;
    /// // Rotation starts slow and speeds up
    /// ```
    ///
    /// # See Also
    /// - [`rotation_between`](Self::rotation_between) - Full rotation (same as s=1.0)
    /// - [`powf`](Self::powf) - Scale an existing rotation's angle
    #[inline]
    pub fn scaled_rotation_between<SB, SC>(
        a: &Vector<T, U2, SB>,
        b: &Vector<T, U2, SC>,
        s: T,
    ) -> Self
    where
        T: RealField,
        SB: Storage<T, U2>,
        SC: Storage<T, U2>,
    {
        crate::convert(UnitComplex::scaled_rotation_between(a, b, s).to_rotation_matrix())
    }

    /// Computes the rotation needed to transform this rotation into another.
    ///
    /// This calculates the **delta rotation** - the rotation you need to apply to `self` to
    /// get `other`. This is extremely useful for:
    /// - Computing how much something has rotated
    /// - Interpolating between two rotations
    /// - Calculating relative orientations
    ///
    /// The mathematical relationship is: `self.rotation_to(other) * self == other`
    ///
    /// # Parameters
    ///
    /// * `other` - The target rotation
    ///
    /// # Returns
    ///
    /// The delta rotation R such that R * self = other
    ///
    /// # Examples
    ///
    /// ## Basic delta rotation
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::Rotation2;
    /// let rot1 = Rotation2::new(0.1);  // ~5.7 degrees
    /// let rot2 = Rotation2::new(1.7);  // ~97.4 degrees
    ///
    /// // What rotation transforms rot1 into rot2?
    /// let rot_to = rot1.rotation_to(&rot2);
    ///
    /// // Verify the relationship
    /// assert_relative_eq!(rot_to * rot1, rot2);
    /// assert_relative_eq!(rot_to.inverse() * rot2, rot1);
    ///
    /// // The delta is about 91.7 degrees
    /// assert_relative_eq!(rot_to.angle(), 1.6, epsilon = 1.0e-6);
    /// ```
    ///
    /// ## Game: Calculate rotation change
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use std::f32;
    /// # use nalgebra::Rotation2;
    /// // Player's rotation at time t0 and t1
    /// let rotation_t0 = Rotation2::new(0.2);
    /// let rotation_t1 = Rotation2::new(0.8);
    ///
    /// // How much did the player rotate?
    /// let delta = rotation_t0.rotation_to(&rotation_t1);
    ///
    /// assert_relative_eq!(delta.angle(), 0.6, epsilon = 1.0e-6);
    ///
    /// // Apply the delta to verify
    /// assert_relative_eq!(delta * rotation_t0, rotation_t1, epsilon = 1.0e-6);
    /// ```
    ///
    /// ## Animation: Interpolate between rotations
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::Rotation2;
    /// let start = Rotation2::new(0.0);
    /// let end = Rotation2::new(1.5);
    ///
    /// // Get the rotation from start to end
    /// let delta = start.rotation_to(&end);
    ///
    /// // Interpolate: 50% of the way from start to end
    /// let halfway = delta.powf(0.5) * start;
    ///
    /// assert_relative_eq!(halfway.angle(), 0.75, epsilon = 1.0e-6);
    /// ```
    ///
    /// ## Robotics: Incremental rotation
    /// ```
    /// # use nalgebra::Rotation2;
    /// // Robot arm joint positions
    /// let current_joint_angle = Rotation2::new(0.5);
    /// let target_joint_angle = Rotation2::new(1.2);
    ///
    /// // Calculate how much more to rotate
    /// let remaining_rotation = current_joint_angle.rotation_to(&target_joint_angle);
    ///
    /// // Could apply a fraction of this each control cycle
    /// let step = remaining_rotation.powf(0.1);  // 10% of remaining
    /// let next_angle = step * current_joint_angle;
    /// ```
    ///
    /// ## Relative orientation between objects
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::Rotation2;
    /// // Two objects with different orientations
    /// let object_a_rotation = Rotation2::new(0.3);
    /// let object_b_rotation = Rotation2::new(1.1);
    ///
    /// // What's B's orientation relative to A?
    /// let relative = object_a_rotation.rotation_to(&object_b_rotation);
    ///
    /// // In A's coordinate system, B is rotated by this amount
    /// assert_relative_eq!(relative.angle(), 0.8, epsilon = 1.0e-6);
    /// ```
    ///
    /// # See Also
    /// - [`angle_to`](Self::angle_to) - Just get the angle difference
    /// - [`inverse`](Self::inverse) - Reverse a rotation
    /// - [`powf`](Self::powf) - Scale a rotation by a factor
    #[inline]
    #[must_use]
    pub fn rotation_to(&self, other: &Self) -> Self {
        other * self.inverse()
    }

    /// Ensures this rotation matrix is orthonormal by correcting numerical errors.
    ///
    /// Over many operations, floating-point errors can cause a rotation matrix to drift from
    /// being perfectly orthonormal (perpendicular axes with unit length). This method corrects
    /// those errors and restores the matrix to a valid rotation.
    ///
    /// You typically need this when:
    /// - Accumulating many small rotations
    /// - After long-running simulations
    /// - When loading rotation data from files
    /// - After interpolating or blending rotations
    ///
    /// # Examples
    ///
    /// ## Cleaning up accumulated errors
    /// ```
    /// # use nalgebra::Rotation2;
    /// let mut rotation = Rotation2::new(0.1);
    ///
    /// // Simulate many small operations that accumulate errors
    /// for _ in 0..10000 {
    ///     rotation = rotation * Rotation2::new(0.0001);
    /// }
    ///
    /// // The matrix might have drifted slightly from being orthonormal
    /// rotation.renormalize();
    ///
    /// // Now it's a perfect rotation again
    /// ```
    ///
    /// ## Periodic renormalization in game loop
    /// ```
    /// # use nalgebra::Rotation2;
    /// struct GameObject {
    ///     rotation: Rotation2<f32>,
    ///     frames_since_renormalize: u32,
    /// }
    ///
    /// impl GameObject {
    ///     fn update(&mut self, delta_rotation: f32) {
    ///         // Apply rotation
    ///         self.rotation = self.rotation * Rotation2::new(delta_rotation);
    ///         self.frames_since_renormalize += 1;
    ///
    ///         // Renormalize every 100 frames to prevent drift
    ///         if self.frames_since_renormalize >= 100 {
    ///             self.rotation.renormalize();
    ///             self.frames_since_renormalize = 0;
    ///         }
    ///     }
    /// }
    /// ```
    ///
    /// ## After loading from file
    /// ```
    /// # use nalgebra::{Rotation2, Matrix2};
    /// // Rotation data loaded from file (might have small errors)
    /// let loaded_matrix = Matrix2::new(
    ///     0.707, -0.707,
    ///     0.707,  0.708  // Slight error in last value
    /// );
    ///
    /// let mut rotation = Rotation2::from_matrix(&loaded_matrix);
    /// rotation.renormalize();
    ///
    /// // Now it's a proper rotation
    /// ```
    ///
    /// # Performance Notes
    ///
    /// This operation is not free - it uses an iterative algorithm. Only call it when needed:
    /// - Don't call it every frame unless you're accumulating many operations
    /// - Call it periodically (e.g., every 100 frames) in long-running simulations
    /// - Call it once after loading data from external sources
    ///
    /// # See Also
    /// - [`from_matrix`](Self::from_matrix) - Extract clean rotation from any matrix
    /// - [`from_matrix_eps`](Self::from_matrix_eps) - Same with custom tolerance
    #[inline]
    pub fn renormalize(&mut self)
    where
        T: RealField,
    {
        let mut c = UnitComplex::from(self.clone());
        let _ = c.renormalize();

        *self = Self::from_matrix_eps(self.matrix(), T::default_epsilon(), 0, c.into())
    }

    /// Raises the rotation to a power, multiplying its angle by the given factor.
    ///
    /// This scales the rotation angle by a factor `n`. It's equivalent to rotating by
    /// the same amount multiple times, but more efficient. This is perfect for:
    /// - Animation and interpolation
    /// - Scaling rotations
    /// - Reversing rotations (use negative power)
    ///
    /// # Parameters
    ///
    /// * `n` - The power to raise the rotation to (angle multiplier)
    ///   - `n = 2.0` → double the rotation angle
    ///   - `n = 0.5` → half the rotation angle
    ///   - `n = -1.0` → reverse the rotation (same as inverse)
    ///   - `n = 0.0` → identity (no rotation)
    ///
    /// # Examples
    ///
    /// ## Basic power operation
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::Rotation2;
    /// let rot = Rotation2::new(0.78);
    ///
    /// // Double the rotation
    /// let pow = rot.powf(2.0);
    /// assert_relative_eq!(pow.angle(), 2.0 * 0.78);
    ///
    /// // Half the rotation
    /// let half = rot.powf(0.5);
    /// assert_relative_eq!(half.angle(), 0.5 * 0.78);
    /// ```
    ///
    /// ## Animation interpolation
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use std::f32;
    /// # use nalgebra::Rotation2;
    /// let full_rotation = Rotation2::new(f32::consts::PI); // 180 degrees
    ///
    /// // Animate from 0 to full rotation
    /// let t = 0.3; // 30% through animation
    /// let current = full_rotation.powf(t);
    ///
    /// // At 30%, we're at 54 degrees
    /// assert_relative_eq!(current.angle(), f32::consts::PI * 0.3, epsilon = 1.0e-6);
    /// ```
    ///
    /// ## Reverse a rotation
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{Rotation2, Vector2};
    /// let rotation = Rotation2::new(1.5);
    /// let point = Vector2::new(1.0, 0.0);
    ///
    /// // Rotate forward
    /// let rotated = rotation * point;
    ///
    /// // Rotate backward (same as inverse)
    /// let reversed = rotation.powf(-1.0);
    /// let back = reversed * rotated;
    ///
    /// assert_relative_eq!(back, point, epsilon = 1.0e-6);
    /// ```
    ///
    /// ## Game: Smooth door opening
    /// ```
    /// # use std::f32;
    /// # use nalgebra::Rotation2;
    /// // Door opens 90 degrees
    /// let fully_open = Rotation2::new(f32::consts::FRAC_PI_2);
    ///
    /// // Current opening progress (0 = closed, 1 = fully open)
    /// let open_amount = 0.6; // 60% open
    ///
    /// let current_rotation = fully_open.powf(open_amount);
    /// // Door is currently at 54 degrees
    /// ```
    ///
    /// ## Robotics: Fractional rotation
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::Rotation2;
    /// let target_rotation = Rotation2::new(2.0);
    ///
    /// // Apply 1/5th of the rotation per control step
    /// let step_rotation = target_rotation.powf(0.2);
    ///
    /// // After 5 steps, reach target
    /// let result = step_rotation * step_rotation * step_rotation * step_rotation * step_rotation;
    /// assert_relative_eq!(result, target_rotation, epsilon = 1.0e-6);
    /// ```
    ///
    /// ## Multiple rotations in one
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{Rotation2, Vector2};
    /// let rotation = Rotation2::new(0.5);
    ///
    /// // Rotate 3 times (same as rotation * rotation * rotation)
    /// let triple = rotation.powf(3.0);
    ///
    /// let point = Vector2::new(1.0, 0.0);
    /// let manual = rotation * rotation * rotation * point;
    /// let power = triple * point;
    ///
    /// assert_relative_eq!(manual, power, epsilon = 1.0e-6);
    /// ```
    ///
    /// # See Also
    /// - [`scaled_rotation_between`](Self::scaled_rotation_between) - Partial rotation between vectors
    /// - [`inverse`](Self::inverse) - Reverse rotation (same as powf(-1.0))
    #[inline]
    #[must_use]
    pub fn powf(&self, n: T) -> Self {
        Self::new(self.angle() * n)
    }
}

/// # 2D angle extraction
impl<T: SimdRealField> Rotation2<T> {
    /// Returns the rotation angle in radians.
    ///
    /// This extracts the angle of rotation from the rotation matrix. The angle is always
    /// in the range [-π, π], where positive angles represent counterclockwise rotations
    /// and negative angles represent clockwise rotations.
    ///
    /// This is the inverse operation of [`new`](Self::new): if you create a rotation with
    /// an angle θ, calling `angle()` on it will return θ (or an equivalent angle).
    ///
    /// # Returns
    ///
    /// The rotation angle in radians, in the range [-π, π]
    ///
    /// # Examples
    ///
    /// ## Basic angle extraction
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::Rotation2;
    /// // Create a rotation with a known angle
    /// let rot = Rotation2::new(1.78);
    ///
    /// // Extract the angle back
    /// let angle = rot.angle();
    /// assert_relative_eq!(angle, 1.78, epsilon = 1.0e-6);
    /// ```
    ///
    /// ## Game object orientation tracking
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use std::f32;
    /// # use nalgebra::Rotation2;
    /// // Sprite facing 45 degrees
    /// let sprite_rotation = Rotation2::new(f32::consts::FRAC_PI_4);
    ///
    /// // Get current angle for UI display or game logic
    /// let current_angle = sprite_rotation.angle();
    ///
    /// // Convert to degrees for display
    /// let degrees = current_angle.to_degrees();
    /// assert_relative_eq!(degrees, 45.0, epsilon = 1.0e-5);
    /// ```
    ///
    /// ## Physics: Angular velocity calculation
    /// ```
    /// # use std::f32;
    /// # use nalgebra::Rotation2;
    /// // Object rotations at two time steps
    /// let rotation_t0 = Rotation2::new(0.5);
    /// let rotation_t1 = Rotation2::new(0.8);
    /// let delta_time = 0.1; // seconds
    ///
    /// // Calculate angular velocity
    /// let angle_change = rotation_t1.angle() - rotation_t0.angle();
    /// let angular_velocity = angle_change / delta_time;
    ///
    /// // 3 radians per second
    /// assert!((angular_velocity - 3.0).abs() < 1e-6);
    /// ```
    ///
    /// ## Combining rotations
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use std::f32;
    /// # use nalgebra::Rotation2;
    /// let rot1 = Rotation2::new(f32::consts::FRAC_PI_4); // 45 degrees
    /// let rot2 = Rotation2::new(f32::consts::FRAC_PI_4); // 45 degrees
    ///
    /// // Combine rotations
    /// let combined = rot1 * rot2;
    ///
    /// // Result is 90 degrees
    /// assert_relative_eq!(combined.angle(), f32::consts::FRAC_PI_2, epsilon = 1.0e-6);
    /// ```
    ///
    /// # See Also
    /// - [`new`](Self::new) - Create a rotation from an angle
    /// - [`angle_to`](Self::angle_to) - Get the angle between two rotations
    /// - [`scaled_axis`](Self::scaled_axis) - Get angle as a 1D vector (for generic code)
    #[inline]
    #[must_use]
    pub fn angle(&self) -> T {
        self.matrix()[(1, 0)]
            .clone()
            .simd_atan2(self.matrix()[(0, 0)].clone())
    }

    /// Returns the rotation angle needed to transform this rotation into another.
    ///
    /// This computes the angular difference between two rotations. The result is the angle
    /// you would need to rotate by to go from `self` to `other`. This is equivalent to
    /// computing `(other * self.inverse()).angle()`.
    ///
    /// The returned angle is in the range [-π, π], where:
    /// - Positive values mean `other` is rotated counterclockwise relative to `self`
    /// - Negative values mean `other` is rotated clockwise relative to `self`
    ///
    /// # Parameters
    ///
    /// * `other` - The target rotation
    ///
    /// # Returns
    ///
    /// The angle difference in radians
    ///
    /// # Examples
    ///
    /// ## Basic angle difference
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::Rotation2;
    /// let rot1 = Rotation2::new(0.1);  // ~5.7 degrees
    /// let rot2 = Rotation2::new(1.7);  // ~97.4 degrees
    ///
    /// // How much to rotate from rot1 to reach rot2
    /// let angle_diff = rot1.angle_to(&rot2);
    /// assert_relative_eq!(angle_diff, 1.6, epsilon = 1.0e-6);
    /// ```
    ///
    /// ## Game AI: Turn towards target
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use std::f32;
    /// # use nalgebra::Rotation2;
    /// // Enemy facing east
    /// let enemy_rotation = Rotation2::new(0.0);
    ///
    /// // Target is 90 degrees counterclockwise (north)
    /// let target_rotation = Rotation2::new(f32::consts::FRAC_PI_2);
    ///
    /// // Calculate how much to turn
    /// let turn_angle = enemy_rotation.angle_to(&target_rotation);
    ///
    /// assert_relative_eq!(turn_angle, f32::consts::FRAC_PI_2, epsilon = 1.0e-6);
    ///
    /// // For smooth rotation, you might limit the turn rate
    /// let max_turn_per_frame = 0.1;
    /// let actual_turn = turn_angle.clamp(-max_turn_per_frame, max_turn_per_frame);
    /// ```
    ///
    /// ## Robotics: Servo control
    /// ```
    /// # use nalgebra::Rotation2;
    /// // Current servo orientation
    /// let current_angle = 0.5;
    /// let current_rotation = Rotation2::new(current_angle);
    ///
    /// // Desired servo orientation
    /// let target_angle = 1.2;
    /// let target_rotation = Rotation2::new(target_angle);
    ///
    /// // Calculate control signal (how much to rotate)
    /// let control_angle = current_rotation.angle_to(&target_rotation);
    ///
    /// assert!((control_angle - 0.7).abs() < 1e-6);
    /// ```
    ///
    /// # See Also
    /// - [`angle`](Self::angle) - Get the absolute angle of a rotation
    /// - [`rotation_to`](Self::rotation_to) - Get the rotation (not just angle) between rotations
    #[inline]
    #[must_use]
    pub fn angle_to(&self, other: &Self) -> T {
        self.rotation_to(other).angle()
    }

    /// Returns the rotation angle as a 1-dimensional vector.
    ///
    /// This is equivalent to calling [`angle()`](Self::angle) and wrapping the result in a
    /// 1D vector. It's primarily used in generic programming contexts where you need consistent
    /// dimensionality between 2D and 3D rotations.
    ///
    /// For 3D rotations, the scaled axis is a 3D vector representing both the rotation axis
    /// and angle. To maintain API consistency, 2D rotations return the angle as a 1D vector,
    /// since 2D rotations only have one degree of freedom.
    ///
    /// # Returns
    ///
    /// A 1-dimensional vector containing the rotation angle in radians
    ///
    /// # Examples
    ///
    /// ## Generic rotation handling
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::Rotation2;
    /// let rot = Rotation2::new(1.5);
    ///
    /// // Get angle as a vector for generic code
    /// let axis_angle_vec = rot.scaled_axis();
    ///
    /// assert_eq!(axis_angle_vec.len(), 1);
    /// assert_relative_eq!(axis_angle_vec[0], 1.5, epsilon = 1.0e-6);
    ///
    /// // This is the same as just calling angle()
    /// assert_relative_eq!(axis_angle_vec[0], rot.angle(), epsilon = 1.0e-6);
    /// ```
    ///
    /// ## For most use cases, prefer angle()
    /// ```
    /// # use nalgebra::Rotation2;
    /// let rot = Rotation2::new(2.0);
    ///
    /// // Instead of this:
    /// let angle_vec = rot.scaled_axis();
    /// let angle = angle_vec[0];
    ///
    /// // Just do this:
    /// let angle = rot.angle();
    /// ```
    ///
    /// # See Also
    /// - [`angle`](Self::angle) - Get the angle directly (recommended for most use cases)
    /// - [`from_scaled_axis`](Self::from_scaled_axis) - Create rotation from a 1D vector
    #[inline]
    #[must_use]
    pub fn scaled_axis(&self) -> SVector<T, 1> {
        Vector1::new(self.angle())
    }
}

#[cfg(feature = "rand-no-std")]
impl<T: SimdRealField> Distribution<Rotation2<T>> for StandardUniform
where
    T::Element: SimdRealField,
    T: SampleUniform,
{
    /// Generate a uniformly distributed random rotation.
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Rotation2<T> {
        let twopi = Uniform::new(T::zero(), T::simd_two_pi())
            .expect("Failed to construct `Uniform`, should be unreachable");

        Rotation2::new(rng.sample(twopi))
    }
}

#[cfg(feature = "arbitrary")]
impl<T: SimdRealField + Arbitrary> Arbitrary for Rotation2<T>
where
    T::Element: SimdRealField,
    Owned<T, U2, U2>: Send,
{
    #[inline]
    fn arbitrary(g: &mut Gen) -> Self {
        Self::new(T::arbitrary(g))
    }
}

/*
 *
 * 3D Rotation matrix.
 *
 */
/// # Construction from a 3D axis and/or angles
impl<T: SimdRealField> Rotation3<T>
where
    T::Element: SimdRealField,
{
    /// Creates a 3D rotation from an axis-angle representation.
    ///
    /// This is the primary way to create a 3D rotation. The input is a 3D vector where:
    /// - The **direction** of the vector is the axis of rotation
    /// - The **magnitude** (length) of the vector is the angle of rotation in radians
    ///
    /// This compact representation is also called "rotation vector" or "exponential coordinates".
    /// It's very convenient because you can scale the vector to change the rotation amount while
    /// keeping the same axis.
    ///
    /// The rotation follows the right-hand rule: if your right thumb points along the axis,
    /// your fingers curl in the direction of positive rotation.
    ///
    /// # Parameters
    ///
    /// * `axisangle` - A 3D vector where direction = axis, magnitude = angle in radians
    ///
    /// # Special Cases
    ///
    /// - Zero vector (or very small magnitude) results in identity (no rotation)
    /// - The vector is automatically normalized to extract the axis
    ///
    /// # Examples
    ///
    /// ## Basic 3D rotation
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use std::f32;
    /// # use nalgebra::{Rotation3, Point3, Vector3};
    /// // Rotate 90 degrees (π/2) around Y axis
    /// let axisangle = Vector3::y() * f32::consts::FRAC_PI_2;
    /// let rot = Rotation3::new(axisangle);
    ///
    /// // Transform a point
    /// let pt = Point3::new(4.0, 5.0, 6.0);
    /// let transformed = rot * pt;
    /// assert_relative_eq!(transformed, Point3::new(6.0, 5.0, -4.0), epsilon = 1.0e-6);
    ///
    /// // Also works for vectors
    /// let vec = Vector3::new(4.0, 5.0, 6.0);
    /// assert_relative_eq!(rot * vec, Vector3::new(6.0, 5.0, -4.0), epsilon = 1.0e-6);
    /// ```
    ///
    /// ## Identity rotation
    /// ```
    /// # use nalgebra::{Rotation3, Vector3};
    /// // Zero vector creates identity (no rotation)
    /// let identity = Rotation3::new(Vector3::<f32>::zeros());
    /// assert_eq!(identity, Rotation3::identity());
    /// ```
    ///
    /// ## Game camera rotation
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use std::f32;
    /// # use nalgebra::{Rotation3, Vector3, Point3};
    /// // Camera looks down 30 degrees (rotate around X axis)
    /// let look_down_angle = -30.0_f32.to_radians();
    /// let axis = Vector3::x_axis().into_inner();
    /// let camera_rot = Rotation3::new(axis * look_down_angle);
    ///
    /// // Apply to camera forward vector
    /// let forward = Vector3::new(0.0, 0.0, -1.0);
    /// let rotated_forward = camera_rot * forward;
    ///
    /// // Forward now points slightly downward
    /// assert!(rotated_forward.y < 0.0); // Negative Y means down
    /// ```
    ///
    /// ## Robotics: Joint rotation
    /// ```
    /// # use std::f32;
    /// # use nalgebra::{Rotation3, Vector3};
    /// // Robot joint rotating around custom axis
    /// let joint_axis = Vector3::new(1.0, 1.0, 0.0).normalize();
    /// let joint_angle = f32::consts::FRAC_PI_4; // 45 degrees
    ///
    /// let joint_rotation = Rotation3::new(joint_axis * joint_angle);
    ///
    /// // The magnitude encodes the angle
    /// let axis_angle_vec = joint_axis * joint_angle;
    /// assert!((axis_angle_vec.magnitude() - joint_angle).abs() < 1e-6);
    /// ```
    ///
    /// ## Interpolation-friendly representation
    /// ```
    /// # use std::f32;
    /// # use nalgebra::{Rotation3, Vector3};
    /// // Axis-angle is great for interpolation
    /// let start_rotation = Rotation3::identity();
    /// let end_axisangle = Vector3::z() * f32::consts::PI; // 180 degrees
    /// let end_rotation = Rotation3::new(end_axisangle);
    ///
    /// // Interpolate by scaling the axis-angle vector
    /// let t = 0.5; // 50% of the way
    /// let interpolated = Rotation3::new(end_axisangle * t);
    ///
    /// // This is now rotated 90 degrees (halfway to 180)
    /// assert!((interpolated.angle() - f32::consts::FRAC_PI_2).abs() < 1e-6);
    /// ```
    ///
    /// # See Also
    /// - [`from_axis_angle`](Self::from_axis_angle) - Separate axis and angle parameters
    /// - [`from_scaled_axis`](Self::from_scaled_axis) - Alias for this method
    /// - [`scaled_axis`](Self::scaled_axis) - Extract the axis-angle representation
    pub fn new<SB: Storage<T, U3>>(axisangle: Vector<T, U3, SB>) -> Self {
        let axisangle = axisangle.into_owned();
        let (axis, angle) = Unit::new_and_get(axisangle);
        Self::from_axis_angle(&axis, angle)
    }

    /// Creates a 3D rotation from an axis-angle vector (alias for `new`).
    ///
    /// This is an alias for [`new`](Self::new) provided for API consistency and clarity.
    /// The input is a 3D vector where the direction is the rotation axis and the magnitude
    /// is the rotation angle in radians.
    ///
    /// The name "scaled axis" emphasizes that the vector encodes both axis (via direction)
    /// and angle (via magnitude), making it clear this is an axis-angle representation.
    ///
    /// # Parameters
    ///
    /// * `axisangle` - A 3D vector: direction = axis, magnitude = angle in radians
    ///
    /// # Examples
    ///
    /// ## Basic usage
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use std::f32;
    /// # use nalgebra::{Rotation3, Point3, Vector3};
    /// // Rotate 90 degrees around Y axis
    /// let axisangle = Vector3::y() * f32::consts::FRAC_PI_2;
    /// let rot = Rotation3::from_scaled_axis(axisangle);
    ///
    /// // Transform points and vectors
    /// let pt = Point3::new(4.0, 5.0, 6.0);
    /// let vec = Vector3::new(4.0, 5.0, 6.0);
    ///
    /// assert_relative_eq!(rot * pt, Point3::new(6.0, 5.0, -4.0), epsilon = 1.0e-6);
    /// assert_relative_eq!(rot * vec, Vector3::new(6.0, 5.0, -4.0), epsilon = 1.0e-6);
    /// ```
    ///
    /// ## Identity from zero vector
    /// ```
    /// # use nalgebra::{Rotation3, Vector3};
    /// // A zero vector creates an identity rotation (no rotation)
    /// let identity = Rotation3::from_scaled_axis(Vector3::<f32>::zeros());
    /// assert_eq!(identity, Rotation3::identity());
    /// ```
    ///
    /// ## Equivalent to new()
    /// ```
    /// # use nalgebra::{Rotation3, Vector3};
    /// let axis_angle = Vector3::new(0.1, 0.2, 0.3);
    ///
    /// // These are identical
    /// let rot1 = Rotation3::new(axis_angle);
    /// let rot2 = Rotation3::from_scaled_axis(axis_angle);
    ///
    /// assert_eq!(rot1, rot2);
    /// ```
    ///
    /// ## Extracting and recreating rotations
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{Rotation3, Vector3};
    /// let original = Rotation3::from_axis_angle(&Vector3::z_axis(), 1.5);
    ///
    /// // Extract the scaled axis representation
    /// let scaled_axis = original.scaled_axis();
    ///
    /// // Recreate the rotation
    /// let recreated = Rotation3::from_scaled_axis(scaled_axis);
    ///
    /// assert_relative_eq!(original, recreated, epsilon = 1.0e-6);
    /// ```
    ///
    /// # See Also
    /// - [`new`](Self::new) - The primary constructor (this is an alias for it)
    /// - [`from_axis_angle`](Self::from_axis_angle) - Create from separate axis and angle
    /// - [`scaled_axis`](Self::scaled_axis) - Extract the axis-angle vector
    pub fn from_scaled_axis<SB: Storage<T, U3>>(axisangle: Vector<T, U3, SB>) -> Self {
        Self::new(axisangle)
    }

    /// Creates a 3D rotation from a unit axis and an angle.
    ///
    /// This is an alternative to [`new`](Self::new) where the axis and angle are provided
    /// separately. The axis must be a **unit vector** (length 1), and the angle is in radians.
    ///
    /// This form is often clearer when you already have a normalized axis direction and want
    /// to specify the rotation amount independently.
    ///
    /// The rotation follows the right-hand rule around the given axis.
    ///
    /// # Parameters
    ///
    /// * `axis` - A unit vector representing the rotation axis (must have length 1)
    /// * `angle` - The rotation angle in radians (positive = right-hand rule direction)
    ///
    /// # Examples
    ///
    /// ## Basic axis-angle rotation
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use std::f32;
    /// # use nalgebra::{Rotation3, Point3, Vector3};
    /// // Rotate 90 degrees around the Y axis
    /// let axis = Vector3::y_axis();
    /// let angle = f32::consts::FRAC_PI_2;
    /// let rot = Rotation3::from_axis_angle(&axis, angle);
    ///
    /// // Verify the rotation was constructed correctly
    /// assert_eq!(rot.axis().unwrap(), axis);
    /// assert_eq!(rot.angle(), angle);
    ///
    /// // Transform a point
    /// let pt = Point3::new(4.0, 5.0, 6.0);
    /// let transformed = rot * pt;
    /// assert_relative_eq!(transformed, Point3::new(6.0, 5.0, -4.0), epsilon = 1.0e-6);
    ///
    /// // Also works for vectors
    /// let vec = Vector3::new(4.0, 5.0, 6.0);
    /// assert_relative_eq!(rot * vec, Vector3::new(6.0, 5.0, -4.0), epsilon = 1.0e-6);
    /// ```
    ///
    /// ## Game character rotation
    /// ```
    /// # use std::f32;
    /// # use nalgebra::{Rotation3, Vector3, Point3};
    /// // Character turns around the up axis (Y)
    /// let up_axis = Vector3::y_axis();
    /// let turn_angle = 90.0_f32.to_radians();
    ///
    /// let rotation = Rotation3::from_axis_angle(&up_axis, turn_angle);
    ///
    /// // Character's forward direction rotates
    /// let forward = Vector3::new(0.0, 0.0, -1.0);
    /// let new_forward = rotation * forward;
    ///
    /// // Now facing to the side
    /// assert!((new_forward.x - 1.0).abs() < 1e-6);
    /// assert!((new_forward.z).abs() < 1e-6);
    /// ```
    ///
    /// ## Robotics: Revolute joint
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use std::f32;
    /// # use nalgebra::{Rotation3, Vector3, Unit};
    /// // Robot arm joint with custom rotation axis
    /// let joint_axis = Unit::new_normalize(Vector3::new(1.0, 1.0, 0.0));
    /// let joint_angle = f32::consts::FRAC_PI_4; // 45 degrees
    ///
    /// let joint_rotation = Rotation3::from_axis_angle(&joint_axis, joint_angle);
    ///
    /// // Verify parameters are preserved
    /// assert_relative_eq!(joint_rotation.axis().unwrap(), joint_axis, epsilon = 1.0e-6);
    /// assert_relative_eq!(joint_rotation.angle(), joint_angle, epsilon = 1.0e-6);
    /// ```
    ///
    /// ## Flight dynamics: Pitch, yaw, roll axes
    /// ```
    /// # use std::f32;
    /// # use nalgebra::{Rotation3, Vector3};
    /// // Aircraft coordinate system rotations
    /// let pitch_axis = Vector3::x_axis(); // Nose up/down
    /// let yaw_axis = Vector3::y_axis();   // Nose left/right
    /// let roll_axis = Vector3::z_axis();  // Barrel roll
    ///
    /// // Apply 15 degrees of pitch (nose up)
    /// let pitch = Rotation3::from_axis_angle(&pitch_axis, 15.0_f32.to_radians());
    ///
    /// // Apply 30 degrees of yaw (turn right)
    /// let yaw = Rotation3::from_axis_angle(&yaw_axis, 30.0_f32.to_radians());
    ///
    /// // Combine rotations (order matters!)
    /// let combined = yaw * pitch;
    /// ```
    ///
    /// ## Animation: Smooth rotation
    /// ```
    /// # use std::f32;
    /// # use nalgebra::{Rotation3, Vector3};
    /// // Animate a rotation over time
    /// let axis = Vector3::z_axis();
    /// let target_angle = f32::consts::PI; // 180 degrees total
    /// let duration = 2.0; // seconds
    ///
    /// // At time t in the animation:
    /// let t = 0.5; // seconds elapsed
    /// let current_angle = (t / duration) * target_angle;
    ///
    /// let current_rotation = Rotation3::from_axis_angle(&axis, current_angle);
    /// ```
    ///
    /// # See Also
    /// - [`new`](Self::new) - Create from combined axis-angle vector
    /// - [`axis`](Self::axis) - Extract the rotation axis
    /// - [`angle`](Self::angle) - Extract the rotation angle
    /// - [`axis_angle`](Self::axis_angle) - Extract both axis and angle together
    pub fn from_axis_angle<SB>(axis: &Unit<Vector<T, U3, SB>>, angle: T) -> Self
    where
        SB: Storage<T, U3>,
    {
        angle.clone().simd_ne(T::zero()).if_else(
            || {
                let ux = axis.as_ref()[0].clone();
                let uy = axis.as_ref()[1].clone();
                let uz = axis.as_ref()[2].clone();
                let sqx = ux.clone() * ux.clone();
                let sqy = uy.clone() * uy.clone();
                let sqz = uz.clone() * uz.clone();
                let (sin, cos) = angle.simd_sin_cos();
                let one_m_cos = T::one() - cos.clone();

                Self::from_matrix_unchecked(SMatrix::<T, 3, 3>::new(
                    sqx.clone() + (T::one() - sqx) * cos.clone(),
                    ux.clone() * uy.clone() * one_m_cos.clone() - uz.clone() * sin.clone(),
                    ux.clone() * uz.clone() * one_m_cos.clone() + uy.clone() * sin.clone(),
                    ux.clone() * uy.clone() * one_m_cos.clone() + uz.clone() * sin.clone(),
                    sqy.clone() + (T::one() - sqy) * cos.clone(),
                    uy.clone() * uz.clone() * one_m_cos.clone() - ux.clone() * sin.clone(),
                    ux.clone() * uz.clone() * one_m_cos.clone() - uy.clone() * sin.clone(),
                    uy * uz * one_m_cos + ux * sin,
                    sqz.clone() + (T::one() - sqz) * cos,
                ))
            },
            Self::identity,
        )
    }

    /// Creates a 3D rotation from Euler angles (roll, pitch, yaw).
    ///
    /// Euler angles provide an intuitive way to describe 3D rotations using three sequential
    /// rotations around coordinate axes. This function uses the **roll-pitch-yaw (XYZ)** convention,
    /// which is commonly used in aerospace and robotics.
    ///
    /// The rotations are applied in this specific order:
    /// 1. **Roll** - Rotation around the X axis (banking/tilting left-right)
    /// 2. **Pitch** - Rotation around the Y axis (nose up/down)
    /// 3. **Yaw** - Rotation around the Z axis (turning left/right)
    ///
    /// # Parameters
    ///
    /// * `roll` - Rotation around X axis in radians (positive = right wing down)
    /// * `pitch` - Rotation around Y axis in radians (positive = nose up)
    /// * `yaw` - Rotation around Z axis in radians (positive = turn left)
    ///
    /// # Convention Details
    ///
    /// The resulting rotation matrix is: R = Rz(yaw) * Ry(pitch) * Rx(roll)
    ///
    /// Note: Different fields use different Euler angle conventions. This is the **extrinsic XYZ**
    /// convention (also called Tait-Bryan angles). If you need a different convention, use
    /// [`euler_angles_ordered`](Self::euler_angles_ordered).
    ///
    /// # Examples
    ///
    /// ## Basic Euler angle construction
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::Rotation3;
    /// // Create rotation from roll, pitch, yaw
    /// let rot = Rotation3::from_euler_angles(0.1, 0.2, 0.3);
    ///
    /// // Extract Euler angles back
    /// let (roll, pitch, yaw) = rot.euler_angles();
    /// assert_relative_eq!(roll, 0.1, epsilon = 1.0e-6);
    /// assert_relative_eq!(pitch, 0.2, epsilon = 1.0e-6);
    /// assert_relative_eq!(yaw, 0.3, epsilon = 1.0e-6);
    /// ```
    ///
    /// ## Aircraft orientation
    /// ```
    /// # use std::f32;
    /// # use nalgebra::{Rotation3, Vector3};
    /// // Aircraft with 10° bank right, 15° nose up, 30° heading
    /// let roll = -10.0_f32.to_radians();  // Negative = right wing down
    /// let pitch = 15.0_f32.to_radians();  // Positive = nose up
    /// let yaw = 30.0_f32.to_radians();    // Heading 30° from north
    ///
    /// let aircraft_orientation = Rotation3::from_euler_angles(roll, pitch, yaw);
    ///
    /// // Transform aircraft's forward direction
    /// let forward_in_aircraft = Vector3::new(1.0, 0.0, 0.0);
    /// let forward_in_world = aircraft_orientation * forward_in_aircraft;
    /// ```
    ///
    /// ## Camera orientation in games
    /// ```
    /// # use std::f32;
    /// # use nalgebra::{Rotation3, Vector3};
    /// // First-person camera: no roll, look up 20°, face 90° right
    /// let camera_roll = 0.0;
    /// let camera_pitch = 20.0_f32.to_radians();
    /// let camera_yaw = 90.0_f32.to_radians();
    ///
    /// let camera_rotation = Rotation3::from_euler_angles(
    ///     camera_roll,
    ///     camera_pitch,
    ///     camera_yaw
    /// );
    ///
    /// // Get camera's view direction
    /// let forward = Vector3::new(0.0, 0.0, -1.0); // Camera looks down -Z
    /// let view_direction = camera_rotation * forward;
    /// ```
    ///
    /// ## Robotics: End-effector orientation
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use std::f32;
    /// # use nalgebra::Rotation3;
    /// // Robot gripper orientation
    /// let gripper_roll = 0.0;
    /// let gripper_pitch = f32::consts::FRAC_PI_2;  // 90° pitch
    /// let gripper_yaw = 0.0;
    ///
    /// let gripper_orientation = Rotation3::from_euler_angles(
    ///     gripper_roll,
    ///     gripper_pitch,
    ///     gripper_yaw
    /// );
    ///
    /// // Verify the orientation
    /// let extracted = gripper_orientation.euler_angles();
    /// assert_relative_eq!(extracted.1, f32::consts::FRAC_PI_2, epsilon = 1.0e-6);
    /// ```
    ///
    /// ## Gimbal lock awareness
    /// ```
    /// # use std::f32;
    /// # use nalgebra::Rotation3;
    /// // When pitch is exactly ±90°, roll and yaw become dependent (gimbal lock)
    /// let problematic = Rotation3::from_euler_angles(
    ///     0.1,
    ///     f32::consts::FRAC_PI_2,  // 90° pitch causes gimbal lock
    ///     0.2
    /// );
    ///
    /// // Extraction may not return the same values due to gimbal lock
    /// // Consider using quaternions for orientations near gimbal lock
    /// ```
    ///
    /// # Notes on Gimbal Lock
    ///
    /// Euler angles suffer from gimbal lock when pitch approaches ±90°. In these cases:
    /// - The representation becomes ambiguous
    /// - Interpolation behaves poorly
    /// - Consider using [`UnitQuaternion`](crate::UnitQuaternion) instead
    ///
    /// # See Also
    /// - [`euler_angles`](Self::euler_angles) - Extract Euler angles from a rotation
    /// - [`euler_angles_ordered`](Self::euler_angles_ordered) - Use custom Euler angle conventions
    /// - [`from_axis_angle`](Self::from_axis_angle) - Alternative rotation representation
    pub fn from_euler_angles(roll: T, pitch: T, yaw: T) -> Self {
        let (sr, cr) = roll.simd_sin_cos();
        let (sp, cp) = pitch.simd_sin_cos();
        let (sy, cy) = yaw.simd_sin_cos();

        Self::from_matrix_unchecked(SMatrix::<T, 3, 3>::new(
            cy.clone() * cp.clone(),
            cy.clone() * sp.clone() * sr.clone() - sy.clone() * cr.clone(),
            cy.clone() * sp.clone() * cr.clone() + sy.clone() * sr.clone(),
            sy.clone() * cp.clone(),
            sy.clone() * sp.clone() * sr.clone() + cy.clone() * cr.clone(),
            sy * sp.clone() * cr.clone() - cy * sr.clone(),
            -sp,
            cp.clone() * sr,
            cp * cr,
        ))
    }
}

/// # Construction from a 3D eye position and target point
impl<T: SimdRealField> Rotation3<T>
where
    T::Element: SimdRealField,
{
    /// Creates a rotation that orients an object to face a given direction.
    ///
    /// This builds a rotation matrix representing a local coordinate frame where the **Z axis**
    /// points in the direction of `dir`. This is perfect for orienting objects, characters, or
    /// cameras toward specific directions while controlling their up orientation.
    ///
    /// The rotation is constructed such that:
    /// - The **+Z axis** aligns with `dir` (forward direction)
    /// - The **+Y axis** is as close as possible to `up` (vertical direction)
    /// - The **+X axis** is perpendicular to both (right direction)
    ///
    /// This uses the Gram-Schmidt orthogonalization process to build an orthonormal frame.
    ///
    /// # Parameters
    ///
    /// * `dir` - The forward direction to face (does not need to be normalized)
    /// * `up` - The approximate up direction (does not need to be normalized)
    ///   - Must not be parallel to `dir` (this is not checked)
    ///
    /// # Examples
    ///
    /// ## Basic object orientation
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{Rotation3, Vector3};
    /// let dir = Vector3::new(1.0, 2.0, 3.0);
    /// let up = Vector3::y();
    ///
    /// let rot = Rotation3::face_towards(&dir, &up);
    ///
    /// // The Z axis now points in the direction of dir
    /// assert_relative_eq!(rot * Vector3::z(), dir.normalize());
    /// ```
    ///
    /// ## Game character facing a target
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{Rotation3, Vector3, Point3};
    /// // Character at origin wants to face a target
    /// let character_pos = Point3::new(0.0, 0.0, 0.0);
    /// let target_pos = Point3::new(5.0, 0.0, 3.0);
    ///
    /// // Calculate direction to target
    /// let face_direction = target_pos - character_pos;
    /// let up = Vector3::y(); // Keep character upright
    ///
    /// let rotation = Rotation3::face_towards(&face_direction, &up);
    ///
    /// // Character's forward (Z) now points toward target
    /// let forward = rotation * Vector3::z();
    /// assert_relative_eq!(forward, face_direction.normalize(), epsilon = 1.0e-6);
    /// ```
    ///
    /// ## Billboard sprite facing camera
    /// ```
    /// # use nalgebra::{Rotation3, Vector3, Point3};
    /// // Sprite position and camera position
    /// let sprite_pos = Point3::new(10.0, 5.0, 10.0);
    /// let camera_pos = Point3::new(0.0, 8.0, 0.0);
    ///
    /// // Make sprite face camera
    /// let to_camera = camera_pos - sprite_pos;
    /// let world_up = Vector3::y();
    ///
    /// let sprite_rotation = Rotation3::face_towards(&to_camera, &world_up);
    /// // Sprite now faces the camera
    /// ```
    ///
    /// ## Spaceship orienting to travel direction
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{Rotation3, Vector3};
    /// // Spaceship velocity defines forward direction
    /// let velocity = Vector3::new(100.0, 20.0, 50.0);
    ///
    /// // Use ship's current up direction (or world up)
    /// let ship_up = Vector3::new(0.0, 1.0, 0.1).normalize();
    ///
    /// let rotation = Rotation3::face_towards(&velocity, &ship_up);
    ///
    /// // Ship's Z axis aligned with velocity
    /// let ship_forward = rotation * Vector3::z();
    /// assert_relative_eq!(ship_forward, velocity.normalize(), epsilon = 1.0e-6);
    /// ```
    ///
    /// ## Turret aiming
    /// ```
    /// # use nalgebra::{Rotation3, Vector3, Point3};
    /// // Turret aims at target
    /// let turret_pos = Point3::new(0.0, 0.0, 0.0);
    /// let enemy_pos = Point3::new(10.0, 2.0, 15.0);
    ///
    /// let aim_direction = enemy_pos - turret_pos;
    /// let turret_rotation = Rotation3::face_towards(&aim_direction, &Vector3::y());
    ///
    /// // Turret's barrel (Z axis) now points at enemy
    /// ```
    ///
    /// # Important Notes
    ///
    /// - The `dir` and `up` vectors must **not be parallel**. If they are, the result is undefined.
    /// - The actual up direction might differ slightly from the `up` parameter to ensure orthogonality
    /// - Both input vectors can have any non-zero length
    ///
    /// # See Also
    /// - [`look_at_rh`](Self::look_at_rh) - Camera-style look-at (Z points toward camera, not target)
    /// - [`look_at_lh`](Self::look_at_lh) - Left-handed version
    /// - [`new_observer_frames`](Self::new_observer_frames) - Deprecated alias for this function
    #[inline]
    pub fn face_towards<SB, SC>(dir: &Vector<T, U3, SB>, up: &Vector<T, U3, SC>) -> Self
    where
        SB: Storage<T, U3>,
        SC: Storage<T, U3>,
    {
        // Gram–Schmidt process
        let zaxis = dir.normalize();
        let xaxis = up.cross(&zaxis).normalize();
        let yaxis = zaxis.cross(&xaxis);

        Self::from_matrix_unchecked(SMatrix::<T, 3, 3>::new(
            xaxis.x.clone(),
            yaxis.x.clone(),
            zaxis.x.clone(),
            xaxis.y.clone(),
            yaxis.y.clone(),
            zaxis.y.clone(),
            xaxis.z.clone(),
            yaxis.z.clone(),
            zaxis.z.clone(),
        ))
    }

    /// Deprecated: Use [`Rotation3::face_towards`] instead.
    #[deprecated(note = "renamed to `face_towards`")]
    pub fn new_observer_frames<SB, SC>(dir: &Vector<T, U3, SB>, up: &Vector<T, U3, SC>) -> Self
    where
        SB: Storage<T, U3>,
        SC: Storage<T, U3>,
    {
        Self::face_towards(dir, up)
    }

    /// Creates a right-handed camera rotation looking toward a direction.
    ///
    /// This builds a **camera-oriented** rotation matrix following the **right-handed** convention
    /// used in OpenGL and most 3D graphics. The key difference from `face_towards` is that the
    /// camera looks **down the negative Z axis**, so this function maps the view direction to **-Z**.
    ///
    /// Right-handed camera convention:
    /// - **+X** points right
    /// - **+Y** points up
    /// - **+Z** points toward the camera (out of screen)
    /// - Camera looks down **-Z** (into the screen)
    ///
    /// # Parameters
    ///
    /// * `dir` - The direction the camera looks toward (does not need to be normalized)
    /// * `up` - Approximate up direction (does not need to be normalized, must not be parallel to `dir`)
    ///
    /// # Examples
    ///
    /// ## Basic camera setup
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{Rotation3, Vector3};
    /// let dir = Vector3::new(1.0, 2.0, 3.0);
    /// let up = Vector3::y();
    ///
    /// let rot = Rotation3::look_at_rh(&dir, &up);
    ///
    /// // The camera looks toward dir, which maps to -Z axis
    /// assert_relative_eq!(rot * dir.normalize(), -Vector3::z());
    /// ```
    ///
    /// ## OpenGL-style camera
    /// ```
    /// # use nalgebra::{Rotation3, Vector3, Point3};
    /// // Camera at origin looking at a target
    /// let camera_pos = Point3::new(0.0, 5.0, 10.0);
    /// let target = Point3::new(0.0, 0.0, 0.0);
    ///
    /// let view_direction = target - camera_pos;
    /// let camera_rotation = Rotation3::look_at_rh(&view_direction, &Vector3::y());
    ///
    /// // Use this rotation for the camera's orientation in a view matrix
    /// ```
    ///
    /// ## First-person camera
    /// ```
    /// # use nalgebra::{Rotation3, Vector3};
    /// // Player looks forward and slightly down
    /// let look_direction = Vector3::new(0.0, -0.3, -1.0);
    /// let up = Vector3::y();
    ///
    /// let camera_rot = Rotation3::look_at_rh(&look_direction, &up);
    /// // Camera is oriented for first-person view
    /// ```
    ///
    /// # See Also
    /// - [`look_at_lh`](Self::look_at_lh) - Left-handed version (DirectX style)
    /// - [`face_towards`](Self::face_towards) - Object-oriented (Z points forward, not backward)
    #[inline]
    pub fn look_at_rh<SB, SC>(dir: &Vector<T, U3, SB>, up: &Vector<T, U3, SC>) -> Self
    where
        SB: Storage<T, U3>,
        SC: Storage<T, U3>,
    {
        Self::face_towards(&dir.neg(), up).inverse()
    }

    /// Creates a left-handed camera rotation looking toward a direction.
    ///
    /// This builds a **camera-oriented** rotation matrix following the **left-handed** convention
    /// used in DirectX and some game engines. In this convention, the camera looks **down the
    /// positive Z axis**, so this function maps the view direction to **+Z**.
    ///
    /// Left-handed camera convention:
    /// - **+X** points right
    /// - **+Y** points up
    /// - **+Z** points into the screen (forward)
    /// - Camera looks down **+Z**
    ///
    /// # Parameters
    ///
    /// * `dir` - The direction the camera looks toward (does not need to be normalized)
    /// * `up` - Approximate up direction (does not need to be normalized, must not be parallel to `dir`)
    ///
    /// # Examples
    ///
    /// ## Basic camera setup
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{Rotation3, Vector3};
    /// let dir = Vector3::new(1.0, 2.0, 3.0);
    /// let up = Vector3::y();
    ///
    /// let rot = Rotation3::look_at_lh(&dir, &up);
    /// assert_relative_eq!(rot * dir.normalize(), Vector3::z());
    /// ```
    ///
    /// ## DirectX-style camera
    /// ```
    /// # use nalgebra::{Rotation3, Vector3, Point3};
    /// // Camera looking toward a target
    /// let camera_pos = Point3::new(0.0, 5.0, 10.0);
    /// let target = Point3::new(0.0, 0.0, 0.0);
    ///
    /// let view_direction = target - camera_pos;
    /// let camera_rotation = Rotation3::look_at_lh(&view_direction, &Vector3::y());
    ///
    /// // Use with DirectX-style projection matrices
    /// ```
    ///
    /// # See Also
    /// - [`look_at_rh`](Self::look_at_rh) - Right-handed version (OpenGL style)
    /// - [`face_towards`](Self::face_towards) - Object-oriented (Z points forward)
    #[inline]
    pub fn look_at_lh<SB, SC>(dir: &Vector<T, U3, SB>, up: &Vector<T, U3, SC>) -> Self
    where
        SB: Storage<T, U3>,
        SC: Storage<T, U3>,
    {
        Self::face_towards(dir, up).inverse()
    }
}

/// # Construction from an existing 3D matrix or rotations
impl<T: SimdRealField> Rotation3<T>
where
    T::Element: SimdRealField,
{
    /// Creates the rotation that aligns vector `a` with vector `b` in 3D.
    ///
    /// This computes the shortest rotation that transforms vector `a` to point in the same
    /// direction as vector `b`. This is essential for:
    /// - Orienting objects toward targets
    /// - Aligning coordinate frames
    /// - Character and camera controllers
    /// - Physics simulations
    ///
    /// The rotation is around an axis perpendicular to both `a` and `b`, by the angle between them.
    ///
    /// # Parameters
    ///
    /// * `a` - Source vector (does not need to be normalized)
    /// * `b` - Target vector (does not need to be normalized)
    ///
    /// # Returns
    ///
    /// * `Some(rotation)` - The rotation that aligns `a` with `b`
    /// * `None` - If vectors are anti-parallel (opposite directions), rotation is ambiguous
    ///
    /// # Examples
    ///
    /// ## Basic vector alignment
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{Vector3, Rotation3};
    /// let a = Vector3::new(1.0, 2.0, 3.0);
    /// let b = Vector3::new(3.0, 1.0, 2.0);
    ///
    /// let rot = Rotation3::rotation_between(&a, &b).unwrap();
    ///
    /// // Verify alignment
    /// assert_relative_eq!(rot * a, b, epsilon = 1.0e-6);
    /// assert_relative_eq!(rot.inverse() * b, a, epsilon = 1.0e-6);
    /// ```
    ///
    /// ## Game: Point object toward target
    /// ```
    /// # use nalgebra::{Vector3, Rotation3, Point3};
    /// // Object's current forward direction
    /// let current_forward = Vector3::z();
    ///
    /// // Target position relative to object
    /// let to_target = Vector3::new(5.0, 2.0, 3.0);
    ///
    /// // Rotate to face target
    /// if let Some(rotation) = Rotation3::rotation_between(&current_forward, &to_target) {
    ///     // Apply rotation to object
    /// }
    /// ```
    ///
    /// ## Handling None case (anti-parallel vectors)
    /// ```
    /// # use nalgebra::{Vector3, Rotation3};
    /// let forward = Vector3::z();
    /// let backward = -Vector3::z();
    ///
    /// // Anti-parallel vectors have ambiguous rotation (180° around any perpendicular axis)
    /// let result = Rotation3::rotation_between(&forward, &backward);
    /// assert!(result.is_none());
    ///
    /// // Handle this case explicitly
    /// let rotation = result.unwrap_or_else(|| {
    ///     // Choose an arbitrary perpendicular axis for 180° rotation
    ///     Rotation3::from_axis_angle(&Vector3::x_axis(), std::f32::consts::PI)
    /// });
    /// ```
    ///
    /// # See Also
    /// - [`scaled_rotation_between`](Self::scaled_rotation_between) - Partial rotation (animation)
    /// - [`face_towards`](Self::face_towards) - Orient object with up vector control
    #[inline]
    pub fn rotation_between<SB, SC>(a: &Vector<T, U3, SB>, b: &Vector<T, U3, SC>) -> Option<Self>
    where
        T: RealField,
        SB: Storage<T, U3>,
        SC: Storage<T, U3>,
    {
        Self::scaled_rotation_between(a, b, T::one())
    }

    /// Creates a partial rotation from `a` toward `b`, scaled by factor `n`.
    ///
    /// This is like [`rotation_between`](Self::rotation_between) but applies only a fraction of
    /// the full rotation. Perfect for smooth 3D animations, gradual reorientation, and interpolation.
    ///
    /// # Parameters
    ///
    /// * `a` - Source vector (does not need to be normalized)
    /// * `b` - Target vector (does not need to be normalized)
    /// * `n` - Scaling factor: 0.0 = no rotation, 0.5 = halfway, 1.0 = full rotation
    ///
    /// # Returns
    ///
    /// * `Some(rotation)` - The scaled rotation
    /// * `None` - If vectors are anti-parallel (ambiguous 180° rotation)
    ///
    /// # Examples
    ///
    /// ## Smooth 3D rotation animation
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{Vector3, Rotation3};
    /// let start = Vector3::x();
    /// let end = Vector3::z();
    ///
    /// // Animate over 5 frames
    /// for i in 0..=5 {
    ///     let t = i as f32 / 5.0;
    ///     if let Some(rot) = Rotation3::scaled_rotation_between(&start, &end, t) {
    ///         let current = rot * start;
    ///         // Smoothly rotates from X to Z
    ///     }
    /// }
    /// ```
    ///
    /// ## Verification example
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{Vector3, Rotation3};
    /// let a = Vector3::new(1.0, 2.0, 3.0);
    /// let b = Vector3::new(3.0, 1.0, 2.0);
    ///
    /// // 20% of rotation (1/5th)
    /// let rot2 = Rotation3::scaled_rotation_between(&a, &b, 0.2).unwrap();
    /// // Applying 5 times gives full rotation
    /// assert_relative_eq!(rot2 * rot2 * rot2 * rot2 * rot2 * a, b, epsilon = 1.0e-6);
    ///
    /// // 50% of rotation (half)
    /// let rot5 = Rotation3::scaled_rotation_between(&a, &b, 0.5).unwrap();
    /// // Applying twice gives full rotation
    /// assert_relative_eq!(rot5 * rot5 * a, b, epsilon = 1.0e-6);
    /// ```
    ///
    /// ## Game: Smooth turret tracking
    /// ```
    /// # use nalgebra::{Vector3, Rotation3};
    /// let turret_forward = Vector3::z();
    /// let target_direction = Vector3::new(1.0, 0.5, 1.0);
    ///
    /// // Rotate 15% toward target each frame
    /// let turn_rate = 0.15;
    /// if let Some(rotation) = Rotation3::scaled_rotation_between(
    ///     &turret_forward,
    ///     &target_direction,
    ///     turn_rate
    /// ) {
    ///     // Apply to turret orientation
    /// }
    /// ```
    ///
    /// # See Also
    /// - [`rotation_between`](Self::rotation_between) - Full rotation (n=1.0)
    /// - [`powf`](Self::powf) - Scale an existing rotation's angle
    #[inline]
    pub fn scaled_rotation_between<SB, SC>(
        a: &Vector<T, U3, SB>,
        b: &Vector<T, U3, SC>,
        n: T,
    ) -> Option<Self>
    where
        T: RealField,
        SB: Storage<T, U3>,
        SC: Storage<T, U3>,
    {
        // TODO: code duplication with Rotation.
        if let (Some(na), Some(nb)) = (a.try_normalize(T::zero()), b.try_normalize(T::zero())) {
            let c = na.cross(&nb);

            if let Some(axis) = Unit::try_new(c, T::default_epsilon()) {
                return Some(Self::from_axis_angle(&axis, na.dot(&nb).acos() * n));
            }

            // Zero or PI.
            if na.dot(&nb) < T::zero() {
                // PI
                //
                // The rotation axis is undefined but the angle not zero. This is not a
                // simple rotation.
                return None;
            }
        }

        Some(Self::identity())
    }

    /// Computes the rotation needed to transform this rotation into another.
    ///
    /// This calculates the **delta rotation** that, when applied to `self`, produces `other`.
    /// Mathematically: `self.rotation_to(other) * self == other`
    ///
    /// This is crucial for:
    /// - Computing relative orientations
    /// - Interpolating between rotations
    /// - Calculating angular changes over time
    ///
    /// # Parameters
    ///
    /// * `other` - The target rotation
    ///
    /// # Examples
    ///
    /// ## Basic delta rotation
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{Rotation3, Vector3};
    /// let rot1 = Rotation3::from_axis_angle(&Vector3::y_axis(), 1.0);
    /// let rot2 = Rotation3::from_axis_angle(&Vector3::x_axis(), 0.1);
    ///
    /// let rot_to = rot1.rotation_to(&rot2);
    ///
    /// // Verify the relationship
    /// assert_relative_eq!(rot_to * rot1, rot2, epsilon = 1.0e-6);
    /// ```
    ///
    /// ## Game: Calculate orientation change
    /// ```
    /// # use nalgebra::{Rotation3, Vector3};
    /// // Character orientation at two time points
    /// let orientation_t0 = Rotation3::from_axis_angle(&Vector3::y_axis(), 0.5);
    /// let orientation_t1 = Rotation3::from_axis_angle(&Vector3::y_axis(), 1.2);
    ///
    /// // How much did the character rotate?
    /// let delta = orientation_t0.rotation_to(&orientation_t1);
    ///
    /// // Could use delta to compute angular velocity, etc.
    /// ```
    ///
    /// # See Also
    /// - [`angle_to`](Self::angle_to) - Just the angle difference
    /// - [`powf`](Self::powf) - Scale a rotation's angle
    #[inline]
    #[must_use]
    pub fn rotation_to(&self, other: &Self) -> Self {
        other * self.inverse()
    }

    /// Raises the rotation to a power, scaling its angle by a factor.
    ///
    /// Returns a rotation with the **same axis** as `self` but with the angle multiplied by `n`.
    /// Perfect for animation, interpolation, and rotation scaling.
    ///
    /// # Parameters
    ///
    /// * `n` - The power/scaling factor
    ///   - `n = 2.0` → double the angle
    ///   - `n = 0.5` → half the angle
    ///   - `n = -1.0` → reverse (same as inverse)
    ///   - `n = 0.0` → identity
    ///
    /// # Examples
    ///
    /// ## Basic power operation
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{Rotation3, Vector3, Unit};
    /// let axis = Unit::new_normalize(Vector3::new(1.0, 2.0, 3.0));
    /// let angle = 1.2;
    /// let rot = Rotation3::from_axis_angle(&axis, angle);
    /// let pow = rot.powf(2.0);
    /// assert_relative_eq!(pow.axis().unwrap(), axis, epsilon = 1.0e-6);
    /// assert_eq!(pow.angle(), 2.4);
    /// ```
    ///
    /// ## Animation interpolation
    /// ```
    /// # use std::f32;
    /// # use nalgebra::{Rotation3, Vector3};
    /// let full_rotation = Rotation3::from_axis_angle(&Vector3::z_axis(), f32::consts::PI);
    ///
    /// // Animate from 0% to 100%
    /// let t = 0.3;
    /// let current = full_rotation.powf(t);
    /// // At 30%, angle is 30% of PI
    /// ```
    ///
    /// # See Also
    /// - [`scaled_rotation_between`](Self::scaled_rotation_between) - Scale rotation between vectors
    #[inline]
    #[must_use]
    pub fn powf(&self, n: T) -> Self
    where
        T: RealField,
    {
        match self.axis() {
            Some(axis) => Self::from_axis_angle(&axis, self.angle() * n),
            None => {
                if self.matrix()[(0, 0)] < T::zero() {
                    let minus_id = SMatrix::<T, 3, 3>::from_diagonal_element(-T::one());
                    Self::from_matrix_unchecked(minus_id)
                } else {
                    Self::identity()
                }
            }
        }
    }

    /// Builds a 3D rotation from three basis vectors without checking orthonormality.
    ///
    /// Creates a rotation from three vectors representing the rotated X, Y, and Z axes.
    /// **Warning:** Does not verify orthonormality. Use only with known-valid bases.
    ///
    /// # Parameters
    ///
    /// * `basis` - Array of three vectors: [X-axis, Y-axis, Z-axis]
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use std::f32;
    /// # use nalgebra::{Rotation3, Vector3};
    /// // Create 90-degree rotation around Z
    /// let x = Vector3::y();  // X becomes Y
    /// let y = -Vector3::x(); // Y becomes -X
    /// let z = Vector3::z();  // Z unchanged
    ///
    /// let rot = Rotation3::from_basis_unchecked(&[x, y, z]);
    /// assert_relative_eq!(rot * Vector3::x(), Vector3::y(), epsilon = 1.0e-6);
    /// ```
    ///
    /// # See Also
    /// - [`from_matrix`](Self::from_matrix) - Extracts rotation from any matrix (safe)
    pub fn from_basis_unchecked(basis: &[Vector3<T>; 3]) -> Self {
        let mat = Matrix3::from_columns(&basis[..]);
        Self::from_matrix_unchecked(mat)
    }

    /// Extracts the rotation component from any 3D transformation matrix.
    ///
    /// Takes any 3×3 matrix and finds the closest pure rotation. Useful for cleaning up
    /// matrices with scaling/shearing or accumulated numerical errors.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{Rotation3, Matrix3};
    /// // Matrix with rotation + scaling
    /// let m = Matrix3::new(
    ///     2.0, 0.0, 0.0,
    ///     0.0, 2.0, 0.0,
    ///     0.0, 0.0, 2.0
    /// );
    ///
    /// let rot = Rotation3::from_matrix(&m);
    /// // Extracted pure rotation (no scaling)
    /// assert_relative_eq!(rot.matrix().determinant(), 1.0, epsilon = 1.0e-6);
    /// ```
    ///
    /// # See Also
    /// - [`from_matrix_eps`](Self::from_matrix_eps) - Custom convergence parameters
    /// - [`renormalize`](Self::renormalize) - Clean up existing rotation
    pub fn from_matrix(m: &Matrix3<T>) -> Self
    where
        T: RealField,
    {
        Self::from_matrix_eps(m, T::default_epsilon(), 0, Self::identity())
    }

    /// Builds a rotation matrix by extracting the rotation part of the given transformation `m`.
    ///
    /// This implements "A Robust Method to Extract the Rotational Part of Deformations" by Müller et al.
    ///
    /// # Parameters
    ///
    /// * `m`: the matrix from which the rotational part is to be extracted.
    /// * `eps`: the angular errors tolerated between the current rotation and the optimal one.
    /// * `max_iter`: the maximum number of iterations. Loops indefinitely until convergence if set to `0`.
    /// * `guess`: a guess of the solution. Convergence will be significantly faster if an initial solution close
    ///   to the actual solution is provided. Can be set to `Rotation3::identity()` if no other
    ///   guesses come to mind.
    pub fn from_matrix_eps(m: &Matrix3<T>, eps: T, mut max_iter: usize, guess: Self) -> Self
    where
        T: RealField,
    {
        if max_iter == 0 {
            max_iter = usize::MAX;
        }

        // Using sqrt(eps) ensures we perturb with something larger than eps; clamp to eps to handle the case of eps > 1.0
        let eps_disturbance = eps.clone().sqrt().max(eps.clone() * eps.clone());
        let mut perturbation_axes = Vector3::x_axis();
        let mut rot = guess.into_inner();

        for _ in 0..max_iter {
            let axis = rot.column(0).cross(&m.column(0))
                + rot.column(1).cross(&m.column(1))
                + rot.column(2).cross(&m.column(2));
            let denom = rot.column(0).dot(&m.column(0))
                + rot.column(1).dot(&m.column(1))
                + rot.column(2).dot(&m.column(2));

            let axisangle = axis / (denom.abs() + T::default_epsilon());

            match Unit::try_new_and_get(axisangle, eps.clone()) {
                Some((axis, angle)) => {
                    rot = Rotation3::from_axis_angle(&axis, angle) * rot;
                }
                None => {
                    // Check if stuck in a maximum w.r.t. the norm (m - rot).norm()
                    let mut perturbed = rot.clone();
                    let norm_squared = (m - &rot).norm_squared();
                    let mut new_norm_squared: T;

                    // Perturb until the new norm is significantly different
                    loop {
                        perturbed *=
                            Rotation3::from_axis_angle(&perturbation_axes, eps_disturbance.clone());
                        new_norm_squared = (m - &perturbed).norm_squared();
                        if abs_diff_ne!(
                            norm_squared,
                            new_norm_squared,
                            epsilon = T::default_epsilon()
                        ) {
                            break;
                        }
                    }

                    // If new norm is larger, it's a minimum
                    if norm_squared < new_norm_squared {
                        break;
                    }

                    // If not, continue from perturbed rotation, but use a different axes for the next perturbation
                    perturbation_axes = UnitVector3::new_unchecked(perturbation_axes.yzx());
                    rot = perturbed;
                }
            }
        }

        Self::from_matrix_unchecked(rot)
    }

    /// Ensure this rotation is an orthonormal rotation matrix. This is useful when repeated
    /// computations might cause the matrix from progressively not being orthonormal anymore.
    #[inline]
    pub fn renormalize(&mut self)
    where
        T: RealField,
    {
        let mut c = UnitQuaternion::from(self.clone());
        let _ = c.renormalize();

        *self = Self::from_matrix_eps(self.matrix(), T::default_epsilon(), 0, c.into())
    }
}

/// # 3D axis and angle extraction
impl<T: SimdRealField> Rotation3<T> {
    /// Returns the rotation angle in radians, in the range [0, π].
    ///
    /// This extracts the magnitude of rotation from the rotation matrix, regardless of the
    /// rotation axis. The angle is always positive and represents the amount of rotation,
    /// not the direction (which is given by the axis).
    ///
    /// For a rotation of 0 radians, this returns 0. For a 180-degree rotation, this returns π.
    /// The angle is computed from the trace of the rotation matrix.
    ///
    /// # Returns
    ///
    /// The rotation angle in radians, always in the range [0, π]
    ///
    /// # Examples
    ///
    /// ## Basic angle extraction
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{Unit, Rotation3, Vector3};
    /// // Create a rotation with known angle
    /// let axis = Unit::new_normalize(Vector3::new(1.0, 2.0, 3.0));
    /// let rot = Rotation3::from_axis_angle(&axis, 1.78);
    ///
    /// // Extract the angle
    /// assert_relative_eq!(rot.angle(), 1.78, epsilon = 1.0e-6);
    /// ```
    ///
    /// ## Game object rotation speed
    /// ```
    /// # use std::f32;
    /// # use nalgebra::{Rotation3, Vector3};
    /// // Calculate how much an object has rotated
    /// let previous_rotation = Rotation3::identity();
    /// let current_rotation = Rotation3::from_axis_angle(
    ///     &Vector3::y_axis(),
    ///     f32::consts::FRAC_PI_4
    /// );
    ///
    /// // Get rotation difference
    /// let delta_rotation = current_rotation * previous_rotation.inverse();
    /// let rotation_amount = delta_rotation.angle();
    ///
    /// // Rotated by 45 degrees (π/4 radians)
    /// assert!((rotation_amount - f32::consts::FRAC_PI_4).abs() < 1e-6);
    /// ```
    ///
    /// ## Robotics: Joint angle monitoring
    /// ```
    /// # use std::f32;
    /// # use nalgebra::{Rotation3, Vector3};
    /// // Check if a joint has rotated beyond a safe limit
    /// let max_rotation = 90.0_f32.to_radians();
    ///
    /// let joint_rotation = Rotation3::from_axis_angle(
    ///     &Vector3::x_axis(),
    ///     75.0_f32.to_radians()
    /// );
    ///
    /// let current_angle = joint_rotation.angle();
    ///
    /// if current_angle > max_rotation {
    ///     // Exceeds safe rotation limit
    ///     println!("Warning: joint angle too large");
    /// }
    /// ```
    ///
    /// ## Animation: Rotation progress
    /// ```
    /// # use std::f32;
    /// # use nalgebra::{Rotation3, Vector3};
    /// // Measure animation progress by rotation angle
    /// let target_angle = f32::consts::PI; // 180 degrees
    /// let axis = Vector3::z_axis();
    ///
    /// // Current animation state
    /// let current_rotation = Rotation3::from_axis_angle(&axis, f32::consts::FRAC_PI_2);
    ///
    /// // Calculate progress (0.0 to 1.0)
    /// let progress = current_rotation.angle() / target_angle;
    ///
    /// assert!((progress - 0.5).abs() < 1e-6); // 50% complete
    /// ```
    ///
    /// # See Also
    /// - [`axis`](Self::axis) - Get the rotation axis
    /// - [`axis_angle`](Self::axis_angle) - Get both axis and angle together
    /// - [`scaled_axis`](Self::scaled_axis) - Get axis-angle representation as a vector
    #[inline]
    #[must_use]
    pub fn angle(&self) -> T {
        ((self.matrix()[(0, 0)].clone()
            + self.matrix()[(1, 1)].clone()
            + self.matrix()[(2, 2)].clone()
            - T::one())
            / crate::convert(2.0))
        .simd_acos()
    }

    /// Returns the unit vector representing the rotation axis.
    ///
    /// This extracts the axis around which the rotation occurs. The axis is a unit vector
    /// (length 1) pointing in the direction of the rotation axis according to the right-hand rule.
    ///
    /// # Returns
    ///
    /// - `Some(axis)` - A unit vector representing the rotation axis
    /// - `None` - If the rotation angle is zero (identity rotation) or π (180°)
    ///
    /// # Special Cases
    ///
    /// Returns `None` when:
    /// - The rotation is identity (no rotation, any axis would be valid)
    /// - The rotation is exactly 180° (the axis is ambiguous, multiple axes give same result)
    ///
    /// For these cases, use [`axis_angle`](Self::axis_angle) which handles them gracefully,
    /// or check the angle first with [`angle`](Self::angle).
    ///
    /// # Examples
    ///
    /// ## Basic axis extraction
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{Rotation3, Vector3, Unit};
    /// // Create rotation around a known axis
    /// let axis = Unit::new_normalize(Vector3::new(1.0, 2.0, 3.0));
    /// let angle = 1.2;
    /// let rot = Rotation3::from_axis_angle(&axis, angle);
    ///
    /// // Extract the axis back
    /// assert_relative_eq!(rot.axis().unwrap(), axis, epsilon = 1.0e-6);
    /// ```
    ///
    /// ## Handling identity rotation
    /// ```
    /// # use nalgebra::{Rotation3, Vector3};
    /// let axis = Vector3::z_axis();
    ///
    /// // Zero angle rotation (identity)
    /// let rot = Rotation3::from_axis_angle(&axis, 0.0);
    ///
    /// // Axis is undefined for identity rotation
    /// assert!(rot.axis().is_none());
    /// ```
    ///
    /// ## Game development: Projectile spin axis
    /// ```
    /// # use std::f32;
    /// # use nalgebra::{Rotation3, Vector3};
    /// // Bullet rotating as it flies
    /// let spin_axis = Vector3::new(0.0, 1.0, 0.0); // Vertical spin
    /// let spin_rate = 10.0; // radians per second
    /// let time = 0.5; // seconds
    ///
    /// let current_rotation = Rotation3::from_axis_angle(
    ///     &Vector3::y_axis(),
    ///     spin_rate * time
    /// );
    ///
    /// // Extract spin axis for physics calculations
    /// if let Some(axis) = current_rotation.axis() {
    ///     // Use axis for angular momentum calculations
    ///     assert!((axis.y - 1.0).abs() < 1e-6);
    /// }
    /// ```
    ///
    /// ## Robotics: Joint axis identification
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use std::f32;
    /// # use nalgebra::{Rotation3, Vector3, Unit};
    /// // Identify which joint caused a rotation
    /// let joint_rotation = Rotation3::from_axis_angle(
    ///     &Vector3::x_axis(),
    ///     f32::consts::FRAC_PI_4
    /// );
    ///
    /// if let Some(axis) = joint_rotation.axis() {
    ///     // Check if it's rotating around X axis (shoulder joint)
    ///     if (axis.x.abs() - 1.0).abs() < 1e-6 {
    ///         println!("Shoulder joint active");
    ///     }
    ///     assert_relative_eq!(axis, Vector3::x_axis(), epsilon = 1.0e-6);
    /// }
    /// ```
    ///
    /// ## Flight simulation: Control surface analysis
    /// ```
    /// # use std::f32;
    /// # use nalgebra::{Rotation3, Vector3};
    /// // Analyze aircraft rotation to determine which control surface was used
    /// let aircraft_rotation = Rotation3::from_axis_angle(
    ///     &Vector3::z_axis(),  // Roll axis
    ///     15.0_f32.to_radians()
    /// );
    ///
    /// if let Some(rotation_axis) = aircraft_rotation.axis() {
    ///     // Check primary rotation component
    ///     if rotation_axis.z.abs() > 0.9 {
    ///         println!("Primarily a roll maneuver (ailerons)");
    ///     }
    /// }
    /// ```
    ///
    /// # See Also
    /// - [`angle`](Self::angle) - Get just the rotation angle
    /// - [`axis_angle`](Self::axis_angle) - Get both axis and angle, handles edge cases better
    /// - [`scaled_axis`](Self::scaled_axis) - Get axis-angle as a single vector
    #[inline]
    #[must_use]
    pub fn axis(&self) -> Option<Unit<Vector3<T>>>
    where
        T: RealField,
    {
        let rotmat = self.matrix();
        let axis = SVector::<T, 3>::new(
            rotmat[(2, 1)].clone() - rotmat[(1, 2)].clone(),
            rotmat[(0, 2)].clone() - rotmat[(2, 0)].clone(),
            rotmat[(1, 0)].clone() - rotmat[(0, 1)].clone(),
        );

        Unit::try_new(axis, T::default_epsilon())
    }

    /// Returns the axis-angle representation as a single 3D vector.
    ///
    /// This combines the rotation axis and angle into a compact 3D vector representation where:
    /// - The **direction** of the vector is the rotation axis
    /// - The **magnitude** (length) of the vector is the rotation angle in radians
    ///
    /// This is the inverse of the [`new`](Self::new) constructor. It's also known as the
    /// "rotation vector" or "exponential coordinates" representation.
    ///
    /// # Returns
    ///
    /// A 3D vector whose direction is the axis and magnitude is the angle. Returns zero vector
    /// for identity rotation or 180° rotation.
    ///
    /// # Examples
    ///
    /// ## Basic round-trip conversion
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{Rotation3, Vector3};
    /// let axisangle = Vector3::new(0.1, 0.2, 0.3);
    /// let rot = Rotation3::new(axisangle);
    ///
    /// // Extract back the same representation
    /// assert_relative_eq!(rot.scaled_axis(), axisangle, epsilon = 1.0e-6);
    /// ```
    ///
    /// ## Interpolation between rotations
    /// ```
    /// # use std::f32;
    /// # use nalgebra::{Rotation3, Vector3};
    /// let start_rot = Rotation3::identity();
    /// let end_rot = Rotation3::new(Vector3::z() * f32::consts::PI);
    ///
    /// // Linear interpolation in axis-angle space
    /// let t = 0.5; // 50% of the way
    /// let start_aa = start_rot.scaled_axis();
    /// let end_aa = end_rot.scaled_axis();
    /// let interpolated_aa = start_aa + (end_aa - start_aa) * t;
    ///
    /// let interpolated_rot = Rotation3::new(interpolated_aa);
    /// ```
    ///
    /// ## Robotics: Velocity commands
    /// ```
    /// # use std::f32;
    /// # use nalgebra::{Rotation3, Vector3};
    /// // Angular velocity as axis-angle per time unit
    /// let angular_velocity = Vector3::new(0.0, 0.1, 0.0); // rad/sec around Y
    /// let time_step = 0.1; // seconds
    ///
    /// // Incremental rotation for this time step
    /// let delta_rotation = Rotation3::new(angular_velocity * time_step);
    ///
    /// // Can extract it back
    /// let delta_aa = delta_rotation.scaled_axis();
    /// assert!((delta_aa.magnitude() - 0.01).abs() < 1e-6);
    /// ```
    ///
    /// ## Serialization/storage
    /// ```
    /// # use nalgebra::{Rotation3, Vector3};
    /// let rotation = Rotation3::from_axis_angle(&Vector3::x_axis(), 1.5);
    ///
    /// // Compact 3-value representation for storage
    /// let compact_form = rotation.scaled_axis();
    /// let values = [compact_form.x, compact_form.y, compact_form.z];
    ///
    /// // Reconstruct later
    /// let reconstructed = Rotation3::new(Vector3::from_row_slice(&values));
    /// ```
    ///
    /// # See Also
    /// - [`new`](Self::new) - Construct a rotation from a scaled axis vector
    /// - [`axis`](Self::axis) - Get just the axis (unit vector)
    /// - [`angle`](Self::angle) - Get just the angle
    /// - [`axis_angle`](Self::axis_angle) - Get axis and angle as separate values
    #[inline]
    #[must_use]
    pub fn scaled_axis(&self) -> Vector3<T>
    where
        T: RealField,
    {
        match self.axis() {
            Some(axis) => axis.into_inner() * self.angle(),
            None => Vector::zero(),
        }
    }

    /// Returns both the rotation axis and angle as a tuple.
    ///
    /// This is a convenience method that extracts both the axis (as a unit vector) and
    /// the angle (in radians) in a single call. It's equivalent to calling
    /// [`axis()`](Self::axis) and [`angle()`](Self::angle) separately.
    ///
    /// # Returns
    ///
    /// - `Some((axis, angle))` - The rotation axis and angle in radians (angle in range [0, π])
    /// - `None` - If the rotation is identity (zero angle)
    ///
    /// # Examples
    ///
    /// ## Basic extraction
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{Rotation3, Vector3, Unit};
    /// let axis = Unit::new_normalize(Vector3::new(1.0, 2.0, 3.0));
    /// let angle = 1.2;
    /// let rot = Rotation3::from_axis_angle(&axis, angle);
    ///
    /// // Extract both at once
    /// let (extracted_axis, extracted_angle) = rot.axis_angle().unwrap();
    /// assert_relative_eq!(extracted_axis, axis, epsilon = 1.0e-6);
    /// assert_relative_eq!(extracted_angle, angle, epsilon = 1.0e-6);
    /// ```
    ///
    /// ## Handling identity rotation
    /// ```
    /// # use nalgebra::{Rotation3, Vector3};
    /// let axis = Vector3::z_axis();
    ///
    /// // Zero rotation
    /// let identity = Rotation3::from_axis_angle(&axis, 0.0);
    ///
    /// // No meaningful axis-angle representation
    /// assert!(identity.axis_angle().is_none());
    /// ```
    ///
    /// ## Game physics: Angular momentum
    /// ```
    /// # use std::f32;
    /// # use nalgebra::{Rotation3, Vector3};
    /// let object_rotation = Rotation3::from_axis_angle(
    ///     &Vector3::y_axis(),
    ///     f32::consts::FRAC_PI_4
    /// );
    ///
    /// if let Some((axis, angle)) = object_rotation.axis_angle() {
    ///     // Angular momentum = moment_of_inertia * axis * angular_velocity
    ///     let moment_of_inertia = 2.0;
    ///     let angular_velocity = angle / 0.1; // angle accumulated in 0.1 seconds
    ///
    ///     let angular_momentum = axis.into_inner() * moment_of_inertia * angular_velocity;
    /// }
    /// ```
    ///
    /// ## Robotics: Joint state reporting
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use std::f32;
    /// # use nalgebra::{Rotation3, Vector3, Unit};
    /// let joint_rotation = Rotation3::from_axis_angle(
    ///     &Vector3::x_axis(),
    ///     f32::consts::FRAC_PI_3
    /// );
    ///
    /// if let Some((joint_axis, joint_angle)) = joint_rotation.axis_angle() {
    ///     println!("Joint rotating around axis: {:?}", joint_axis);
    ///     println!("Joint angle: {} degrees", joint_angle.to_degrees());
    ///
    ///     assert_relative_eq!(joint_axis, Vector3::x_axis(), epsilon = 1.0e-6);
    ///     assert_relative_eq!(joint_angle, f32::consts::FRAC_PI_3, epsilon = 1.0e-6);
    /// }
    /// ```
    ///
    /// # See Also
    /// - [`axis`](Self::axis) - Get only the rotation axis
    /// - [`angle`](Self::angle) - Get only the rotation angle
    /// - [`scaled_axis`](Self::scaled_axis) - Get axis and angle as a single vector
    #[inline]
    #[must_use]
    pub fn axis_angle(&self) -> Option<(Unit<Vector3<T>>, T)>
    where
        T: RealField,
    {
        self.axis().map(|axis| (axis, self.angle()))
    }

    /// The rotation angle needed to make `self` and `other` coincide.
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{Rotation3, Vector3};
    /// let rot1 = Rotation3::from_axis_angle(&Vector3::y_axis(), 1.0);
    /// let rot2 = Rotation3::from_axis_angle(&Vector3::x_axis(), 0.1);
    /// assert_relative_eq!(rot1.angle_to(&rot2), 1.0045657, epsilon = 1.0e-6);
    /// ```
    #[inline]
    #[must_use]
    pub fn angle_to(&self, other: &Self) -> T
    where
        T::Element: SimdRealField,
    {
        self.rotation_to(other).angle()
    }

    /// Creates Euler angles from a rotation.
    ///
    /// The angles are produced in the form (roll, pitch, yaw).
    #[deprecated(note = "This is renamed to use `.euler_angles()`.")]
    pub fn to_euler_angles(self) -> (T, T, T)
    where
        T: RealField,
    {
        self.euler_angles()
    }

    /// Extracts Euler angles from this rotation.
    ///
    /// Returns the rotation decomposed into three sequential rotations around coordinate axes.
    /// This uses the **roll-pitch-yaw (XYZ)** convention, which is the inverse of
    /// [`from_euler_angles`](Self::from_euler_angles).
    ///
    /// The returned angles are in the form `(roll, pitch, yaw)`:
    /// - **Roll** - Rotation around X axis in radians (range: [-π, π])
    /// - **Pitch** - Rotation around Y axis in radians (range: [-π/2, π/2])
    /// - **Yaw** - Rotation around Z axis in radians (range: [-π, π])
    ///
    /// # Returns
    ///
    /// A tuple `(roll, pitch, yaw)` in radians
    ///
    /// # Gimbal Lock
    ///
    /// When pitch is exactly ±90° (±π/2), the rotation suffers from gimbal lock:
    /// - Roll and yaw become dependent (infinitely many combinations produce same rotation)
    /// - The extraction may return different values than used in construction
    /// - One of the angles will be set to zero by convention
    ///
    /// For orientations near gimbal lock, consider using quaternions instead.
    ///
    /// # Examples
    ///
    /// ## Basic extraction
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::Rotation3;
    /// // Create from Euler angles
    /// let rot = Rotation3::from_euler_angles(0.1, 0.2, 0.3);
    ///
    /// // Extract them back
    /// let (roll, pitch, yaw) = rot.euler_angles();
    /// assert_relative_eq!(roll, 0.1, epsilon = 1.0e-6);
    /// assert_relative_eq!(pitch, 0.2, epsilon = 1.0e-6);
    /// assert_relative_eq!(yaw, 0.3, epsilon = 1.0e-6);
    /// ```
    ///
    /// ## Aircraft orientation display
    /// ```
    /// # use std::f32;
    /// # use nalgebra::{Rotation3, Vector3};
    /// // Airplane's current orientation
    /// let aircraft_rotation = Rotation3::from_axis_angle(
    ///     &Vector3::y_axis(),
    ///     30.0_f32.to_radians()
    /// );
    ///
    /// // Extract for flight instruments
    /// let (roll, pitch, yaw) = aircraft_rotation.euler_angles();
    ///
    /// println!("Roll: {}°", roll.to_degrees());
    /// println!("Pitch: {}°", pitch.to_degrees());
    /// println!("Yaw: {}°", yaw.to_degrees());
    /// ```
    ///
    /// ## Game camera controls
    /// ```
    /// # use std::f32;
    /// # use nalgebra::Rotation3;
    /// // First-person camera orientation
    /// let camera_rotation = Rotation3::from_euler_angles(
    ///     0.0,                          // No roll
    ///     20.0_f32.to_radians(),        // Looking up 20°
    ///     45.0_f32.to_radians()         // Facing 45° right
    /// );
    ///
    /// // Extract for UI display or input handling
    /// let (roll, pitch, yaw) = camera_rotation.euler_angles();
    ///
    /// // Clamp pitch to prevent looking too far up/down
    /// let max_pitch = 89.0_f32.to_radians();
    /// let clamped_pitch = pitch.clamp(-max_pitch, max_pitch);
    /// ```
    ///
    /// ## Robotics: Joint angle readout
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use std::f32;
    /// # use nalgebra::Rotation3;
    /// // Robot end-effector orientation
    /// let effector_rot = Rotation3::from_euler_angles(
    ///     10.0_f32.to_radians(),   // Wrist roll
    ///     45.0_f32.to_radians(),   // Wrist pitch
    ///     0.0                      // No yaw
    /// );
    ///
    /// // Read individual joint angles for control system
    /// let (wrist_roll, wrist_pitch, wrist_yaw) = effector_rot.euler_angles();
    ///
    /// assert_relative_eq!(wrist_roll, 10.0_f32.to_radians(), epsilon = 1.0e-5);
    /// assert_relative_eq!(wrist_pitch, 45.0_f32.to_radians(), epsilon = 1.0e-5);
    /// ```
    ///
    /// ## Gimbal lock demonstration
    /// ```
    /// # use std::f32;
    /// # use nalgebra::Rotation3;
    /// // Create with pitch = 90° (gimbal lock)
    /// let gimbal_locked = Rotation3::from_euler_angles(
    ///     30.0_f32.to_radians(),
    ///     f32::consts::FRAC_PI_2,  // 90° pitch causes gimbal lock
    ///     45.0_f32.to_radians()
    /// );
    ///
    /// let (roll, pitch, yaw) = gimbal_locked.euler_angles();
    ///
    /// // Pitch is preserved
    /// assert!((pitch - f32::consts::FRAC_PI_2).abs() < 1e-6);
    ///
    /// // But roll and yaw may differ from input (they're coupled)
    /// // One will be set to zero by convention
    /// ```
    ///
    /// # See Also
    /// - [`from_euler_angles`](Self::from_euler_angles) - Construct from Euler angles
    /// - [`euler_angles_ordered`](Self::euler_angles_ordered) - Use custom Euler angle conventions
    /// - [`axis_angle`](Self::axis_angle) - Alternative representation without gimbal lock issues
    #[must_use]
    pub fn euler_angles(&self) -> (T, T, T)
    where
        T: RealField,
    {
        // Implementation informed by "Computing Euler angles from a rotation matrix", by Gregory G. Slabaugh
        //  https://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.371.6578
        //  where roll, pitch, yaw angles are referred to as ψ, θ, ϕ,
        if self[(2, 0)].clone().abs() < T::one() {
            let pitch = -self[(2, 0)].clone().asin();
            let theta_cos = pitch.clone().cos();
            let roll = (self[(2, 1)].clone() / theta_cos.clone())
                .atan2(self[(2, 2)].clone() / theta_cos.clone());
            let yaw =
                (self[(1, 0)].clone() / theta_cos.clone()).atan2(self[(0, 0)].clone() / theta_cos);
            (roll, pitch, yaw)
        } else if self[(2, 0)].clone() <= -T::one() {
            (
                self[(0, 1)].clone().atan2(self[(0, 2)].clone()),
                T::frac_pi_2(),
                T::zero(),
            )
        } else {
            (
                -self[(0, 1)].clone().atan2(-self[(0, 2)].clone()),
                -T::frac_pi_2(),
                T::zero(),
            )
        }
    }

    /// Represent this rotation as Euler angles.
    ///
    /// Returns the angles produced in the order provided by seq parameter, along with the
    /// observability flag. The Euler axes passed to seq must form an orthonormal basis. If the
    /// rotation is gimbal locked, then the observability flag is false.
    ///
    /// # Panics
    ///
    /// Panics if the Euler axes in `seq` are not orthonormal.
    ///
    /// # Example 1:
    /// ```
    /// use std::f64::consts::PI;
    /// use approx::assert_relative_eq;
    /// use nalgebra::{Matrix3, Rotation3, Unit, Vector3};
    ///
    /// // 3-1-2
    /// let n = [
    ///     Unit::new_unchecked(Vector3::new(0.0, 0.0, 1.0)),
    ///     Unit::new_unchecked(Vector3::new(1.0, 0.0, 0.0)),
    ///     Unit::new_unchecked(Vector3::new(0.0, 1.0, 0.0)),
    /// ];
    ///
    /// let r1 = Rotation3::from_axis_angle(&n[2], 20.0 * PI / 180.0);
    /// let r2 = Rotation3::from_axis_angle(&n[1], 30.0 * PI / 180.0);
    /// let r3 = Rotation3::from_axis_angle(&n[0], 45.0 * PI / 180.0);
    ///
    /// let d = r3 * r2 * r1;
    ///
    /// let (angles, observable) = d.euler_angles_ordered(n, false);
    /// assert!(observable);
    /// assert_relative_eq!(angles[0] * 180.0 / PI, 45.0, epsilon = 1e-12);
    /// assert_relative_eq!(angles[1] * 180.0 / PI, 30.0, epsilon = 1e-12);
    /// assert_relative_eq!(angles[2] * 180.0 / PI, 20.0, epsilon = 1e-12);
    /// ```
    ///
    /// # Example 2:
    /// ```
    /// use std::f64::consts::PI;
    /// use approx::assert_relative_eq;
    /// use nalgebra::{Matrix3, Rotation3, Unit, Vector3};
    ///
    /// let sqrt_2 = 2.0_f64.sqrt();
    /// let n = [
    ///     Unit::new_unchecked(Vector3::new(1.0 / sqrt_2, 1.0 / sqrt_2, 0.0)),
    ///     Unit::new_unchecked(Vector3::new(1.0 / sqrt_2, -1.0 / sqrt_2, 0.0)),
    ///     Unit::new_unchecked(Vector3::new(0.0, 0.0, 1.0)),
    /// ];
    ///
    /// let r1 = Rotation3::from_axis_angle(&n[2], 20.0 * PI / 180.0);
    /// let r2 = Rotation3::from_axis_angle(&n[1], 30.0 * PI / 180.0);
    /// let r3 = Rotation3::from_axis_angle(&n[0], 45.0 * PI / 180.0);
    ///
    /// let d = r3 * r2 * r1;
    ///
    /// let (angles, observable) = d.euler_angles_ordered(n, false);
    /// assert!(observable);
    /// assert_relative_eq!(angles[0] * 180.0 / PI, 45.0, epsilon = 1e-12);
    /// assert_relative_eq!(angles[1] * 180.0 / PI, 30.0, epsilon = 1e-12);
    /// assert_relative_eq!(angles[2] * 180.0 / PI, 20.0, epsilon = 1e-12);
    /// ```
    ///
    /// Algorithm based on:
    /// Malcolm D. Shuster, F. Landis Markley, “General formula for extraction the Euler
    /// angles”, Journal of guidance, control, and dynamics, vol. 29.1, pp. 215-221. 2006,
    /// and modified to be able to produce extrinsic rotations.
    #[must_use]
    pub fn euler_angles_ordered(
        &self,
        mut seq: [Unit<Vector3<T>>; 3],
        extrinsic: bool,
    ) -> ([T; 3], bool)
    where
        T: RealField + Copy,
    {
        let mut angles = [T::zero(); 3];
        let eps = T::from_subset(&1e-6);
        let two = T::from_subset(&2.0);

        if extrinsic {
            seq.reverse();
        }

        let [n1, n2, n3] = &seq;
        assert_relative_eq!(n1.dot(n2), T::zero(), epsilon = eps);
        assert_relative_eq!(n3.dot(n1), T::zero(), epsilon = eps);

        let n1_c_n2 = n1.cross(n2);
        let s1 = n1_c_n2.dot(n3);
        let c1 = n1.dot(n3);
        let lambda = s1.atan2(c1);

        let mut c = Matrix3::zeros();
        c.column_mut(0).copy_from(n2);
        c.column_mut(1).copy_from(&n1_c_n2);
        c.column_mut(2).copy_from(n1);
        c.transpose_mut();

        let r1l = Matrix3::new(
            T::one(),
            T::zero(),
            T::zero(),
            T::zero(),
            c1,
            s1,
            T::zero(),
            -s1,
            c1,
        );
        let o_t = c * self.matrix() * (c.transpose() * r1l);
        angles[1] = o_t.m33.acos();

        let safe1 = angles[1].abs() >= eps;
        let safe2 = (angles[1] - T::pi()).abs() >= eps;
        let observable = safe1 && safe2;
        angles[1] += lambda;

        if observable {
            angles[0] = o_t.m13.atan2(-o_t.m23);
            angles[2] = o_t.m31.atan2(o_t.m32);
        } else {
            // gimbal lock detected
            if extrinsic {
                // angle1 is initialized to zero
                if !safe1 {
                    angles[2] = (o_t.m12 - o_t.m21).atan2(o_t.m11 + o_t.m22);
                } else {
                    angles[2] = -(o_t.m12 + o_t.m21).atan2(o_t.m11 - o_t.m22);
                };
            } else {
                // angle3 is initialized to zero
                if !safe1 {
                    angles[0] = (o_t.m12 - o_t.m21).atan2(o_t.m11 + o_t.m22);
                } else {
                    angles[0] = (o_t.m12 + o_t.m21).atan2(o_t.m11 - o_t.m22);
                };
            };
        };

        let adjust = if seq[0] == seq[2] {
            // lambda = 0, so ensure angle2 -> [0, pi]
            angles[1] < T::zero() || angles[1] > T::pi()
        } else {
            // lambda = + or - pi/2, so ensure angle2 -> [-pi/2, pi/2]
            angles[1] < -T::frac_pi_2() || angles[1] > T::frac_pi_2()
        };

        // dont adjust gimbal locked rotation
        if adjust && observable {
            angles[0] += T::pi();
            angles[1] = two * lambda - angles[1];
            angles[2] -= T::pi();
        }

        // ensure all angles are within [-pi, pi]
        for angle in angles.as_mut_slice().iter_mut() {
            if *angle < -T::pi() {
                *angle += T::two_pi();
            } else if *angle > T::pi() {
                *angle -= T::two_pi();
            }
        }

        if extrinsic {
            angles.reverse();
        }

        (angles, observable)
    }
}

#[cfg(feature = "rand-no-std")]
impl<T: SimdRealField> Distribution<Rotation3<T>> for StandardUniform
where
    T::Element: SimdRealField,
    OpenClosed01: Distribution<T>,
    T: SampleUniform,
{
    /// Generate a uniformly distributed random rotation.
    #[inline]
    fn sample<'a, R: Rng + ?Sized>(&self, rng: &mut R) -> Rotation3<T> {
        // James Arvo.
        // Fast random rotation matrices.
        // In D. Kirk, editor, Graphics Gems III, pages 117-120. Academic, New York, 1992.

        // Compute a random rotation around Z
        let twopi = Uniform::new(T::zero(), T::simd_two_pi())
            .expect("Failed to construct `Uniform`, should be unreachable");
        let theta = rng.sample(&twopi);
        let (ts, tc) = theta.simd_sin_cos();
        let a = SMatrix::<T, 3, 3>::new(
            tc.clone(),
            ts.clone(),
            T::zero(),
            -ts,
            tc,
            T::zero(),
            T::zero(),
            T::zero(),
            T::one(),
        );

        // Compute a random rotation *of* Z
        let phi = rng.sample(&twopi);
        let z = rng.sample(OpenClosed01);
        let (ps, pc) = phi.simd_sin_cos();
        let sqrt_z = z.clone().simd_sqrt();
        let v = Vector3::new(pc * sqrt_z.clone(), ps * sqrt_z, (T::one() - z).simd_sqrt());
        let mut b = v.clone() * v.transpose();
        b += b.clone();
        b -= SMatrix::<T, 3, 3>::identity();

        Rotation3::from_matrix_unchecked(b * a)
    }
}

#[cfg(feature = "arbitrary")]
impl<T: SimdRealField + Arbitrary> Arbitrary for Rotation3<T>
where
    T::Element: SimdRealField,
    Owned<T, U3, U3>: Send,
    Owned<T, U3>: Send,
{
    #[inline]
    fn arbitrary(g: &mut Gen) -> Self {
        Self::new(SVector::arbitrary(g))
    }
}
