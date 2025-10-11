use num::{One, Zero};

use simba::scalar::{ClosedAddAssign, ClosedMulAssign, SupersetOf};

use crate::base::{SMatrix, Scalar};

use crate::geometry::Rotation;

impl<T, const D: usize> Default for Rotation<T, D>
where
    T: Scalar + Zero + One,
{
    fn default() -> Self {
        Self::identity()
    }
}

/// # Identity
impl<T, const D: usize> Rotation<T, D>
where
    T: Scalar + Zero + One,
{
    /// Creates a new identity rotation matrix.
    ///
    /// An identity rotation represents "no rotation" - it leaves all vectors unchanged when applied.
    /// This is analogous to multiplying by 1 in regular arithmetic. When you multiply any rotation
    /// by the identity rotation, you get the original rotation back unchanged.
    ///
    /// In mathematical terms, this creates a rotation matrix where all diagonal elements are 1
    /// and all off-diagonal elements are 0.
    ///
    /// # Use Cases
    ///
    /// Identity rotations are commonly used as:
    /// - Initial values when building up complex rotations incrementally
    /// - Default orientations for objects in games and simulations
    /// - Identity elements in rotation calculations and interpolations
    /// - Placeholder values when no rotation is needed
    ///
    /// # Examples
    ///
    /// ## Basic usage with 2D rotations
    /// ```
    /// # use nalgebra::{Rotation2, Point2};
    /// // Create an identity rotation (no rotation)
    /// let identity = Rotation2::identity();
    ///
    /// // Applying it to a point leaves the point unchanged
    /// let point = Point2::new(3.0, 4.0);
    /// let rotated = identity * point;
    /// assert_eq!(rotated, point);
    /// ```
    ///
    /// ## Identity as a multiplicative neutral element
    /// ```
    /// # use nalgebra::{Rotation2, Rotation3, Vector3};
    /// // For 2D rotations
    /// let rot2d = Rotation2::new(std::f32::consts::FRAC_PI_2); // 90 degree rotation
    /// let identity2d = Rotation2::identity();
    ///
    /// // Identity on either side leaves the rotation unchanged
    /// assert_eq!(identity2d * rot2d, rot2d);
    /// assert_eq!(rot2d * identity2d, rot2d);
    ///
    /// // For 3D rotations
    /// let rot3d = Rotation3::from_axis_angle(&Vector3::z_axis(), std::f32::consts::FRAC_PI_2);
    /// let identity3d = Rotation3::identity();
    ///
    /// assert_eq!(identity3d * rot3d, rot3d);
    /// assert_eq!(rot3d * identity3d, rot3d);
    /// ```
    ///
    /// ## Using identity as an initial state in game development
    /// ```
    /// # use nalgebra::{Rotation3, Vector3, Point3};
    /// // Initialize a game object with no rotation
    /// let mut player_rotation = Rotation3::identity();
    ///
    /// // Later, apply a rotation (e.g., player looks up 30 degrees)
    /// let look_up = Rotation3::from_axis_angle(&Vector3::x_axis(), 0.523); // ~30 degrees
    /// player_rotation = look_up * player_rotation;
    ///
    /// // Apply the rotation to the camera's forward direction
    /// let forward = Vector3::new(0.0, 0.0, -1.0);
    /// let new_forward = player_rotation * forward;
    /// ```
    ///
    /// ## Verifying a rotation is close to identity
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{Rotation3, Vector3};
    /// // Create a very small rotation
    /// let tiny_rotation = Rotation3::from_axis_angle(&Vector3::y_axis(), 0.0001);
    ///
    /// // Check if it's approximately equal to identity
    /// assert_relative_eq!(tiny_rotation, Rotation3::identity(), epsilon = 1.0e-3);
    /// ```
    ///
    /// # See Also
    ///
    /// - [`Rotation2::new`](crate::Rotation2::new) - Create a 2D rotation from an angle
    /// - [`Rotation3::from_axis_angle`](crate::Rotation3::from_axis_angle) - Create a 3D rotation from an axis and angle
    /// - [`Rotation::inverse`](crate::Rotation::inverse) - Get the inverse of a rotation
    #[inline]
    pub fn identity() -> Rotation<T, D> {
        Self::from_matrix_unchecked(SMatrix::<T, D, D>::identity())
    }
}

impl<T: Scalar, const D: usize> Rotation<T, D> {
    /// Converts this rotation to an equivalent rotation with a different scalar type.
    ///
    /// This function casts all the components of the rotation matrix from one numeric type
    /// to another (e.g., from `f64` to `f32`, or vice versa). This is useful when you need to
    /// interface with APIs that use different floating-point precision levels, or when you want
    /// to trade precision for performance or memory usage.
    ///
    /// The conversion uses the standard numeric casting mechanism, so the typical rules apply:
    /// - Converting from higher to lower precision (e.g., `f64` to `f32`) may lose precision
    /// - Converting from lower to higher precision (e.g., `f32` to `f64`) is lossless
    ///
    /// # Use Cases
    ///
    /// - **Interfacing with graphics APIs**: Many graphics libraries use `f32` for performance,
    ///   while your physics calculations might use `f64` for accuracy
    /// - **Serialization/deserialization**: Converting between different precision requirements
    /// - **Performance optimization**: Using `f32` can be faster on some hardware
    /// - **Memory optimization**: `f32` rotations use half the memory of `f64` rotations
    ///
    /// # Examples
    ///
    /// ## Converting from double to single precision
    /// ```
    /// # use nalgebra::{Rotation2, Rotation3, Vector3};
    /// // Create a rotation using double precision (f64)
    /// let rot_f64 = Rotation2::<f64>::new(1.5707963267948966); // π/2 in f64
    ///
    /// // Cast to single precision (f32) for use with a graphics API
    /// let rot_f32 = rot_f64.cast::<f32>();
    ///
    /// // Both rotations are approximately equal (within f32 precision)
    /// # #[macro_use] extern crate approx;
    /// assert_relative_eq!(rot_f32.angle(), std::f32::consts::FRAC_PI_2, epsilon = 1e-6);
    /// ```
    ///
    /// ## Converting from single to double precision
    /// ```
    /// # use nalgebra::{Rotation3, Vector3};
    /// // Create a rotation using single precision
    /// let rot_f32 = Rotation3::<f32>::from_axis_angle(
    ///     &Vector3::y_axis(),
    ///     std::f32::consts::FRAC_PI_4
    /// );
    ///
    /// // Cast to double precision for more accurate calculations
    /// let rot_f64 = rot_f32.cast::<f64>();
    ///
    /// // The conversion preserves the rotation accurately
    /// # #[macro_use] extern crate approx;
    /// assert_relative_eq!(rot_f64.angle(), std::f64::consts::FRAC_PI_4 as f64, epsilon = 1e-6);
    /// ```
    ///
    /// ## Round-trip conversion
    /// ```
    /// # use nalgebra::Rotation2;
    /// let original = Rotation2::<f64>::identity();
    /// let converted = original.cast::<f32>();
    /// let back = converted.cast::<f64>();
    ///
    /// // Identity rotation survives round-trip perfectly
    /// assert_eq!(back, original);
    /// ```
    ///
    /// ## Practical example: Physics to rendering pipeline
    /// ```
    /// # use nalgebra::{Rotation3, Vector3, Point3};
    /// // Physics simulation uses f64 for accuracy
    /// fn physics_update() -> Rotation3<f64> {
    ///     // Complex physics calculations with high precision
    ///     Rotation3::from_axis_angle(&Vector3::y_axis(), 0.1234567890123456)
    /// }
    ///
    /// // Rendering uses f32 for performance
    /// fn render(rotation: Rotation3<f32>) {
    ///     // Send to GPU, which typically uses f32
    ///     let point = Point3::new(1.0f32, 2.0, 3.0);
    ///     let transformed = rotation * point;
    ///     // ... render the transformed point
    /// }
    ///
    /// // Bridge between physics and rendering
    /// let physics_rotation = physics_update();
    /// let rendering_rotation = physics_rotation.cast::<f32>();
    /// render(rendering_rotation);
    /// ```
    ///
    /// ## Working with integer rotations (special cases)
    /// ```
    /// # use nalgebra::Rotation2;
    /// // While uncommon, you can use integer types in specific algorithms
    /// // (though rotations typically require floating-point for trigonometry)
    /// let identity_f32 = Rotation2::<f32>::identity();
    /// // Cast operations work with the type system
    /// let identity_f64 = identity_f32.cast::<f64>();
    /// ```
    ///
    /// # See Also
    ///
    /// - [`Rotation::identity`](crate::Rotation::identity) - Create an identity rotation
    /// - [`Rotation2::new`](crate::Rotation2::new) - Create a 2D rotation from an angle
    /// - [`Rotation3::from_axis_angle`](crate::Rotation3::from_axis_angle) - Create a 3D rotation from axis and angle
    pub fn cast<To: Scalar>(self) -> Rotation<To, D>
    where
        Rotation<To, D>: SupersetOf<Self>,
    {
        crate::convert(self)
    }
}

impl<T, const D: usize> One for Rotation<T, D>
where
    T: Scalar + Zero + One + ClosedAddAssign + ClosedMulAssign,
{
    #[inline]
    fn one() -> Self {
        Self::identity()
    }
}
