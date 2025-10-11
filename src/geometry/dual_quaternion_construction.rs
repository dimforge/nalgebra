use crate::{
    DualQuaternion, Isometry3, Quaternion, Scalar, SimdRealField, Translation3, UnitDualQuaternion,
    UnitQuaternion,
};
use num::{One, Zero};
#[cfg(feature = "arbitrary")]
use quickcheck::{Arbitrary, Gen};
use simba::scalar::SupersetOf;

impl<T: Scalar> DualQuaternion<T> {
    /// Creates a dual quaternion from its real and dual quaternion components.
    ///
    /// A dual quaternion consists of two parts:
    /// - **Real part**: Typically represents the rotation component
    /// - **Dual part**: Encodes the translation (for rigid body transformations)
    ///
    /// For most use cases, you should use higher-level constructors like
    /// [`from_parts`](UnitDualQuaternion::from_parts) instead, which directly
    /// take rotation and translation parameters.
    ///
    /// # Arguments
    ///
    /// - `real`: The real (primary) quaternion component
    /// - `dual`: The dual (secondary) quaternion component
    ///
    /// # Example
    ///
    /// ```
    /// # use nalgebra::{DualQuaternion, Quaternion};
    /// let real = Quaternion::new(1.0, 2.0, 3.0, 4.0);
    /// let dual = Quaternion::new(5.0, 6.0, 7.0, 8.0);
    ///
    /// let dq = DualQuaternion::from_real_and_dual(real, dual);
    /// assert_eq!(dq.real.w, 1.0);
    /// assert_eq!(dq.dual.w, 5.0);
    /// ```
    ///
    /// # See Also
    ///
    /// - [`from_real`](Self::from_real) - Creates from just the real part (no translation)
    /// - [`from_parts`](UnitDualQuaternion::from_parts) - More convenient constructor for transformations
    /// - [`identity`](Self::identity) - Creates the identity transformation
    #[inline]
    pub const fn from_real_and_dual(real: Quaternion<T>, dual: Quaternion<T>) -> Self {
        Self { real, dual }
    }

    /// The dual quaternion multiplicative identity.
    ///
    /// The identity dual quaternion represents "no transformation" - it leaves points
    /// and vectors unchanged when applied. It consists of:
    /// - Real part: (1, 0, 0, 0) - the identity rotation
    /// - Dual part: (0, 0, 0, 0) - zero translation
    ///
    /// When multiplied with any other dual quaternion, it returns that quaternion unchanged.
    ///
    /// # Returns
    ///
    /// The identity dual quaternion.
    ///
    /// # Example
    ///
    /// ```
    /// # use nalgebra::{DualQuaternion, Quaternion};
    /// let identity = DualQuaternion::identity();
    /// let dq = DualQuaternion::from_real_and_dual(
    ///     Quaternion::new(1.0, 2.0, 3.0, 4.0),
    ///     Quaternion::new(5.0, 6.0, 7.0, 8.0)
    /// );
    ///
    /// // Identity is the multiplicative neutral element
    /// assert_eq!(identity.clone() * dq.clone(), dq.clone());
    /// assert_eq!(dq.clone() * identity, dq);
    /// ```
    ///
    /// # Use Case: Initializing Transformations
    ///
    /// ```
    /// # use nalgebra::UnitDualQuaternion;
    /// // Start with identity and accumulate transformations
    /// let mut current_transform = UnitDualQuaternion::identity();
    ///
    /// // Apply transformations over time...
    /// // current_transform = current_transform * delta_transform;
    /// ```
    ///
    /// # See Also
    ///
    /// - [`UnitDualQuaternion::identity`] - Identity for unit dual quaternions (more commonly used)
    /// - [`zero`](Self::zero) - The additive identity (all zeros, rarely used)
    #[inline]
    pub fn identity() -> Self
    where
        T: SimdRealField,
    {
        Self::from_real_and_dual(
            Quaternion::from_real(T::one()),
            Quaternion::from_real(T::zero()),
        )
    }

    /// Converts the components of this dual quaternion to another scalar type.
    ///
    /// This performs a type conversion on all the scalar values in both the real and
    /// dual quaternion components. This is useful when you need to convert between
    /// different floating-point precisions (e.g., f32 to f64 or vice versa).
    ///
    /// # Type Parameters
    ///
    /// - `To`: The target scalar type (must be convertible from the current type)
    ///
    /// # Returns
    ///
    /// A new dual quaternion with components cast to the target type.
    ///
    /// # Example
    ///
    /// ```
    /// # use nalgebra::{Quaternion, DualQuaternion};
    /// // Create a dual quaternion with f64 precision
    /// let dq_f64 = DualQuaternion::from_real(Quaternion::new(1.0f64, 2.0, 3.0, 4.0));
    ///
    /// // Convert to f32 precision
    /// let dq_f32 = dq_f64.cast::<f32>();
    /// assert_eq!(dq_f32, DualQuaternion::from_real(Quaternion::new(1.0f32, 2.0, 3.0, 4.0)));
    /// ```
    ///
    /// # Use Case: GPU Interoperability
    ///
    /// ```
    /// # use nalgebra::{DualQuaternion, Quaternion};
    /// // CPU simulation uses f64 for precision
    /// let high_precision = DualQuaternion::from_real(Quaternion::new(1.0f64, 2.0, 3.0, 4.0));
    ///
    /// // GPU rendering uses f32 for performance
    /// let gpu_transform: DualQuaternion<f32> = high_precision.cast();
    /// // Send gpu_transform to GPU...
    /// ```
    ///
    /// # See Also
    ///
    /// - [`UnitDualQuaternion::cast`] - For unit dual quaternions
    pub fn cast<To: Scalar>(self) -> DualQuaternion<To>
    where
        DualQuaternion<To>: SupersetOf<Self>,
    {
        crate::convert(self)
    }
}

impl<T: SimdRealField> DualQuaternion<T>
where
    T::Element: SimdRealField,
{
    /// Creates a dual quaternion from only its real part, with zero dual part.
    ///
    /// This creates a dual quaternion that represents only a rotation (or orientation)
    /// with no translation component. The dual part is set to zero.
    ///
    /// This is useful when you want to represent pure rotation without any translation.
    ///
    /// # Arguments
    ///
    /// - `real`: The real quaternion component (typically represents rotation)
    ///
    /// # Returns
    ///
    /// A dual quaternion with the given real part and zero dual part.
    ///
    /// # Example
    ///
    /// ```
    /// # use nalgebra::{DualQuaternion, Quaternion};
    /// let rotation = Quaternion::new(1.0, 2.0, 3.0, 4.0);
    ///
    /// let dq = DualQuaternion::from_real(rotation);
    /// assert_eq!(dq.real.w, 1.0);
    /// assert_eq!(dq.dual.w, 0.0);
    /// assert_eq!(dq.dual.i, 0.0);
    /// ```
    ///
    /// # Use Case: Pure Rotation
    ///
    /// ```
    /// # use nalgebra::{DualQuaternion, UnitQuaternion, Vector3};
    /// // Create a rotation-only transformation
    /// let rotation = UnitQuaternion::from_euler_angles(0.0, 0.0, 1.57);
    /// let dq = DualQuaternion::from_real(rotation.into_inner());
    ///
    /// // This dual quaternion will rotate but not translate
    /// ```
    ///
    /// # See Also
    ///
    /// - [`from_real_and_dual`](Self::from_real_and_dual) - Creates from both real and dual parts
    /// - [`from_rotation`](UnitDualQuaternion::from_rotation) - For unit dual quaternions
    /// - [`identity`](Self::identity) - Creates the identity (no rotation, no translation)
    #[inline]
    pub fn from_real(real: Quaternion<T>) -> Self {
        Self {
            real,
            dual: Quaternion::zero(),
        }
    }
}

impl<T: SimdRealField> One for DualQuaternion<T>
where
    T::Element: SimdRealField,
{
    #[inline]
    fn one() -> Self {
        Self::identity()
    }
}

impl<T: SimdRealField> Zero for DualQuaternion<T>
where
    T::Element: SimdRealField,
{
    #[inline]
    fn zero() -> Self {
        DualQuaternion::from_real_and_dual(Quaternion::zero(), Quaternion::zero())
    }

    #[inline]
    fn is_zero(&self) -> bool {
        self.real.is_zero() && self.dual.is_zero()
    }
}

#[cfg(feature = "arbitrary")]
impl<T> Arbitrary for DualQuaternion<T>
where
    T: SimdRealField + Arbitrary + Send,
    T::Element: SimdRealField,
{
    #[inline]
    fn arbitrary(rng: &mut Gen) -> Self {
        Self::from_real_and_dual(Arbitrary::arbitrary(rng), Arbitrary::arbitrary(rng))
    }
}

impl<T: SimdRealField> UnitDualQuaternion<T> {
    /// The unit dual quaternion identity transformation.
    ///
    /// This represents the identity isometry - a transformation that does nothing.
    /// When applied to any point or vector, it returns the input unchanged.
    ///
    /// The identity consists of:
    /// - Rotation: Identity rotation (no rotation)
    /// - Translation: Zero translation (no movement)
    ///
    /// The identity is its own inverse: `identity * identity = identity` and
    /// `identity.inverse() = identity`.
    ///
    /// # Returns
    ///
    /// The identity transformation.
    ///
    /// # Example
    ///
    /// ```
    /// # use nalgebra::{UnitDualQuaternion, Point3};
    /// let identity = UnitDualQuaternion::identity();
    /// let point = Point3::new(1.0, -4.3, 3.33);
    ///
    /// // Identity transformation doesn't change the point
    /// assert_eq!(identity * point, point);
    ///
    /// // Identity is its own inverse
    /// assert_eq!(identity, identity.inverse());
    /// ```
    ///
    /// # Use Case: Default/Initial State
    ///
    /// ```
    /// # use nalgebra::{UnitDualQuaternion, UnitQuaternion, Vector3};
    /// // Common pattern: start with identity and build up transformation
    /// let mut object_transform = UnitDualQuaternion::identity();
    ///
    /// // Later, apply transformations
    /// let movement = UnitDualQuaternion::from_parts(
    ///     Vector3::new(1.0, 0.0, 0.0).into(),
    ///     UnitQuaternion::identity()
    /// );
    /// object_transform = object_transform * movement;
    /// ```
    ///
    /// # See Also
    ///
    /// - [`DualQuaternion::identity`] - Identity for non-unit dual quaternions
    /// - [`from_parts`](Self::from_parts) - Creates transformations from rotation and translation
    #[inline]
    pub fn identity() -> Self {
        Self::new_unchecked(DualQuaternion::identity())
    }

    /// Converts the components of this unit dual quaternion to another scalar type.
    ///
    /// This performs a type conversion on all the scalar values while maintaining
    /// the unit property. This is useful when you need to convert between different
    /// floating-point precisions (e.g., f32 to f64 or vice versa).
    ///
    /// # Type Parameters
    ///
    /// - `To`: The target scalar type (must be convertible from the current type)
    ///
    /// # Returns
    ///
    /// A new unit dual quaternion with components cast to the target type.
    ///
    /// # Example
    ///
    /// ```
    /// # use nalgebra::UnitDualQuaternion;
    /// // Create a unit dual quaternion with f64 precision
    /// let dq_f64 = UnitDualQuaternion::<f64>::identity();
    ///
    /// // Convert to f32 precision
    /// let dq_f32 = dq_f64.cast::<f32>();
    /// assert_eq!(dq_f32, UnitDualQuaternion::<f32>::identity());
    /// ```
    ///
    /// # Use Case: Mixed Precision Pipeline
    ///
    /// ```
    /// # use nalgebra::{UnitDualQuaternion, UnitQuaternion, Vector3};
    /// // Physics simulation computes in f64
    /// let physics_pose = UnitDualQuaternion::from_parts(
    ///     Vector3::new(1.0f64, 2.0, 3.0).into(),
    ///     UnitQuaternion::from_euler_angles(0.1, 0.2, 0.3)
    /// );
    ///
    /// // Rendering uses f32 for GPU efficiency
    /// let render_pose: UnitDualQuaternion<f32> = physics_pose.cast();
    /// ```
    ///
    /// # See Also
    ///
    /// - [`DualQuaternion::cast`] - For non-unit dual quaternions
    pub fn cast<To: Scalar>(self) -> UnitDualQuaternion<To>
    where
        UnitDualQuaternion<To>: SupersetOf<Self>,
    {
        crate::convert(self)
    }
}

impl<T: SimdRealField> UnitDualQuaternion<T>
where
    T::Element: SimdRealField,
{
    /// Creates a unit dual quaternion from separate translation and rotation components.
    ///
    /// This is the most common way to construct a rigid body transformation. It takes
    /// a translation vector and a rotation quaternion and combines them into a single
    /// unit dual quaternion that represents the complete transformation.
    ///
    /// The resulting transformation applies rotation first, then translation.
    ///
    /// # Arguments
    ///
    /// - `translation`: The translation to apply (as a `Translation3`)
    /// - `rotation`: The rotation to apply (as a `UnitQuaternion`)
    ///
    /// # Returns
    ///
    /// A unit dual quaternion representing the combined transformation.
    ///
    /// # Example
    ///
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{UnitDualQuaternion, UnitQuaternion, Vector3, Point3};
    /// // Create a transformation: rotate 90° around X-axis, then translate
    /// let dq = UnitDualQuaternion::from_parts(
    ///     Vector3::new(0.0, 3.0, 0.0).into(),
    ///     UnitQuaternion::from_euler_angles(std::f32::consts::FRAC_PI_2, 0.0, 0.0)
    /// );
    ///
    /// let point = Point3::new(1.0, 2.0, 3.0);
    /// let transformed = dq * point;
    ///
    /// assert_relative_eq!(transformed, Point3::new(1.0, 0.0, 2.0), epsilon = 1.0e-6);
    /// ```
    ///
    /// # Use Case: Building a Scene Graph
    ///
    /// ```
    /// # use nalgebra::{UnitDualQuaternion, UnitQuaternion, Vector3};
    /// // Robot arm with multiple joints
    /// let base_transform = UnitDualQuaternion::from_parts(
    ///     Vector3::new(0.0, 0.0, 1.0).into(),  // Base is 1m above ground
    ///     UnitQuaternion::identity()
    /// );
    ///
    /// let joint1_transform = UnitDualQuaternion::from_parts(
    ///     Vector3::new(0.0, 0.0, 0.5).into(),  // 0.5m up from base
    ///     UnitQuaternion::from_euler_angles(0.0, 0.0, 0.785)  // 45° rotation
    /// );
    ///
    /// // Compose transformations
    /// let world_to_joint1 = base_transform * joint1_transform;
    /// ```
    ///
    /// # See Also
    ///
    /// - [`from_isometry`](Self::from_isometry) - Creates from an `Isometry3`
    /// - [`from_rotation`](Self::from_rotation) - Creates from just a rotation
    /// - [`identity`](Self::identity) - Creates the identity transformation
    /// - [`translation`](crate::geometry::UnitDualQuaternion::translation) - Extracts the translation part
    /// - [`rotation`](crate::geometry::UnitDualQuaternion::rotation) - Extracts the rotation part
    #[inline]
    pub fn from_parts(translation: Translation3<T>, rotation: UnitQuaternion<T>) -> Self {
        let half: T = crate::convert(0.5f64);
        UnitDualQuaternion::new_unchecked(DualQuaternion {
            real: rotation.clone().into_inner(),
            dual: Quaternion::from_parts(T::zero(), translation.vector)
                * rotation.into_inner()
                * half,
        })
    }

    /// Creates a unit dual quaternion from an isometry.
    ///
    /// An `Isometry3` represents a 3D rigid body transformation (rotation + translation)
    /// using a separate rotation quaternion and translation vector. This method converts
    /// it into the equivalent unit dual quaternion representation.
    ///
    /// This is useful when interfacing with code that uses isometries and you want the
    /// benefits of dual quaternion representation (e.g., for smooth interpolation).
    ///
    /// # Arguments
    ///
    /// - `isometry`: The isometry to convert
    ///
    /// # Returns
    ///
    /// A unit dual quaternion representing the same transformation.
    ///
    /// # Example
    ///
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{Isometry3, UnitDualQuaternion, UnitQuaternion, Vector3, Point3};
    /// let iso = Isometry3::from_parts(
    ///     Vector3::new(0.0, 3.0, 0.0).into(),
    ///     UnitQuaternion::from_euler_angles(std::f32::consts::FRAC_PI_2, 0.0, 0.0)
    /// );
    /// let dq = UnitDualQuaternion::from_isometry(&iso);
    /// let point = Point3::new(1.0, 2.0, 3.0);
    ///
    /// // Both representations produce the same transformation
    /// assert_relative_eq!(dq * point, iso * point, epsilon = 1.0e-6);
    /// ```
    ///
    /// # Use Case: Converting Physics Engine Data
    ///
    /// ```
    /// # use nalgebra::{Isometry3, UnitDualQuaternion, UnitQuaternion, Vector3};
    /// // Many physics engines use Isometry3 for rigid body transforms
    /// let physics_transform = Isometry3::from_parts(
    ///     Vector3::new(1.0, 2.0, 3.0).into(),
    ///     UnitQuaternion::from_euler_angles(0.1, 0.2, 0.3)
    /// );
    ///
    /// // Convert to dual quaternion for smooth interpolation in rendering
    /// let render_transform = UnitDualQuaternion::from_isometry(&physics_transform);
    ///
    /// // Can now use sclerp for smooth animation between physics steps
    /// ```
    ///
    /// # See Also
    ///
    /// - [`to_isometry`](crate::geometry::UnitDualQuaternion::to_isometry) - Converts back to an isometry
    /// - [`from_parts`](Self::from_parts) - Creates directly from rotation and translation
    /// - `From<Isometry3<T>> for UnitDualQuaternion<T>` - Conversion trait implementation
    #[inline]
    pub fn from_isometry(isometry: &Isometry3<T>) -> Self {
        // TODO: take the isometry by-move instead of cloning it.
        let isometry = isometry.clone();
        UnitDualQuaternion::from_parts(isometry.translation, isometry.rotation)
    }

    /// Creates a unit dual quaternion from a pure rotation (no translation).
    ///
    /// This creates a transformation that only rotates, with zero translation.
    /// The resulting dual quaternion will have the given rotation as its real part
    /// and a zero dual part.
    ///
    /// This is useful when you want to represent orientation without position.
    ///
    /// # Arguments
    ///
    /// - `rotation`: The rotation to represent (as a `UnitQuaternion`)
    ///
    /// # Returns
    ///
    /// A unit dual quaternion representing the pure rotation.
    ///
    /// # Example
    ///
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{UnitQuaternion, UnitDualQuaternion, Quaternion};
    /// let rotation = UnitQuaternion::new_normalize(
    ///     Quaternion::new(1.0, 2.0, 3.0, 4.0)
    /// );
    ///
    /// let dq = UnitDualQuaternion::from_rotation(rotation);
    ///
    /// // The real part is the rotation (normalized)
    /// assert_relative_eq!(dq.as_ref().real.norm(), 1.0, epsilon = 1.0e-6);
    /// // The dual part is zero (no translation)
    /// assert_eq!(dq.as_ref().dual.norm(), 0.0);
    /// ```
    ///
    /// # Use Case: Representing Object Orientation
    ///
    /// ```
    /// # use nalgebra::{UnitDualQuaternion, UnitQuaternion, Vector3, Point3};
    /// // Represent just the orientation of a satellite (position tracked separately)
    /// let satellite_orientation = UnitQuaternion::from_euler_angles(0.1, 0.2, 0.3);
    /// let satellite_transform = UnitDualQuaternion::from_rotation(satellite_orientation);
    ///
    /// // Rotate a vector from satellite's local frame to world frame
    /// let local_antenna_direction = Vector3::new(1.0, 0.0, 0.0);
    /// let world_antenna_direction = satellite_transform * local_antenna_direction;
    /// ```
    ///
    /// # See Also
    ///
    /// - [`from_parts`](Self::from_parts) - Creates from both rotation and translation
    /// - [`rotation`](crate::geometry::UnitDualQuaternion::rotation) - Extracts the rotation component
    /// - [`identity`](Self::identity) - Creates identity transformation (no rotation, no translation)
    #[inline]
    pub fn from_rotation(rotation: UnitQuaternion<T>) -> Self {
        Self::new_unchecked(DualQuaternion::from_real(rotation.into_inner()))
    }
}

impl<T: SimdRealField> One for UnitDualQuaternion<T>
where
    T::Element: SimdRealField,
{
    #[inline]
    fn one() -> Self {
        Self::identity()
    }
}

#[cfg(feature = "arbitrary")]
impl<T> Arbitrary for UnitDualQuaternion<T>
where
    T: SimdRealField + Arbitrary + Send,
    T::Element: SimdRealField,
{
    #[inline]
    fn arbitrary(rng: &mut Gen) -> Self {
        Self::new_normalize(Arbitrary::arbitrary(rng))
    }
}
