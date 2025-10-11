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
    AbstractRotation, Isometry, Point, Point3, Rotation2, Rotation3, Scalar, Similarity,
    Translation, UnitComplex, UnitQuaternion,
};

impl<T: SimdRealField, R, const D: usize> Default for Similarity<T, R, D>
where
    T::Element: SimdRealField,
    R: AbstractRotation<T, D>,
{
    fn default() -> Self {
        Self::identity()
    }
}

impl<T: SimdRealField, R, const D: usize> Similarity<T, R, D>
where
    T::Element: SimdRealField,
    R: AbstractRotation<T, D>,
{
    /// Creates a new identity similarity transformation.
    ///
    /// An identity similarity does nothing - it leaves points and vectors unchanged.
    /// This is equivalent to a transformation with:
    /// - No translation (stays at origin)
    /// - No rotation (no orientation change)
    /// - Scale of 1.0 (no size change)
    ///
    /// This is useful as a starting point before applying transformations, or as a
    /// default/neutral transformation state.
    ///
    /// # Returns
    ///
    /// A new identity similarity transformation
    ///
    /// # Example: 2D Identity
    ///
    /// ```
    /// # use nalgebra::{Similarity2, Point2};
    /// let sim = Similarity2::identity();
    /// let pt = Point2::new(1.0, 2.0);
    ///
    /// // Point remains unchanged
    /// assert_eq!(sim * pt, pt);
    /// ```
    ///
    /// # Example: 3D Identity
    ///
    /// ```
    /// # use nalgebra::{Similarity3, Point3};
    /// let sim = Similarity3::identity();
    /// let pt = Point3::new(1.0, 2.0, 3.0);
    ///
    /// // Point remains unchanged
    /// assert_eq!(sim * pt, pt);
    /// ```
    ///
    /// # Example: Default Object Transform
    ///
    /// ```
    /// # use nalgebra::{Similarity3, Vector3};
    /// // Initialize objects with identity transform
    /// let mut object_transform = Similarity3::identity();
    ///
    /// // Later, modify as needed
    /// object_transform = Similarity3::new(
    ///     Vector3::new(10.0, 0.0, 0.0),
    ///     Vector3::zeros(),
    ///     2.0
    /// );
    /// ```
    ///
    /// # Example: Composing from Identity
    ///
    /// ```
    /// # use nalgebra::{Similarity2, Vector2};
    /// let mut transform = Similarity2::identity();
    ///
    /// // Build up transformation step by step
    /// transform = transform.append_scaling(2.0);
    /// // Now has 2x scale
    ///
    /// transform = Similarity2::new(Vector2::new(5.0, 0.0), 0.0, 1.0) * transform;
    /// // Now also translated
    /// ```
    ///
    /// # See Also
    ///
    /// * [`from_scaling`](Self::from_scaling) - Create a pure scaling transformation
    /// * [`new`](Self::new) - Create a complete similarity transformation
    #[inline]
    pub fn identity() -> Self {
        Self::from_isometry(Isometry::identity(), T::one())
    }
}

impl<T: SimdRealField, R, const D: usize> One for Similarity<T, R, D>
where
    T::Element: SimdRealField,
    R: AbstractRotation<T, D>,
{
    /// Creates a new identity similarity.
    #[inline]
    fn one() -> Self {
        Self::identity()
    }
}

#[cfg(feature = "rand-no-std")]
impl<T: crate::RealField, R, const D: usize> Distribution<Similarity<T, R, D>> for StandardUniform
where
    R: AbstractRotation<T, D>,
    StandardUniform: Distribution<T> + Distribution<R>,
{
    /// Generate an arbitrary random variate for testing purposes.
    #[inline]
    fn sample<'a, G: Rng + ?Sized>(&self, rng: &mut G) -> Similarity<T, R, D> {
        let mut s = rng.random();
        while relative_eq!(s, T::zero()) {
            s = rng.random()
        }

        Similarity::from_isometry(rng.random(), s)
    }
}

impl<T: SimdRealField, R, const D: usize> Similarity<T, R, D>
where
    T::Element: SimdRealField,
    R: AbstractRotation<T, D>,
{
    /// Creates a similarity that rotates around a specific point with scaling.
    ///
    /// This creates a similarity transformation that:
    /// 1. Scales objects uniformly by `scaling`
    /// 2. Rotates them by `r` around point `p` (not the origin!)
    ///
    /// The point `p` remains fixed - imagine pinning a piece of paper at point `p` and
    /// rotating it around that pin while also changing its size. This is useful for
    /// rotating and scaling objects around their centers or pivot points.
    ///
    /// # Parameters
    ///
    /// * `r` - The rotation to apply
    /// * `p` - The center point of rotation (remains invariant)
    /// * `scaling` - The uniform scaling factor
    ///
    /// # Returns
    ///
    /// A new similarity that scales and rotates around point `p`
    ///
    /// # Example: 2D Rotation Around Point
    ///
    /// ```
    /// # use nalgebra::{Similarity2, Point2, UnitComplex};
    /// # use approx::assert_relative_eq;
    /// # use std::f32;
    /// // Rotate 90° around point (3, 2), with 4x scaling
    /// let rot = UnitComplex::new(f32::consts::FRAC_PI_2);
    /// let pivot = Point2::new(3.0, 2.0);
    /// let sim = Similarity2::rotation_wrt_point(rot, pivot, 4.0);
    ///
    /// // The pivot point stays fixed
    /// assert_relative_eq!(sim * pivot, pivot, epsilon = 1.0e-6);
    ///
    /// // Other points rotate around the pivot
    /// let point = Point2::new(1.0, 2.0);
    /// let transformed = sim * point;
    /// assert_relative_eq!(transformed, Point2::new(-3.0, 3.0), epsilon = 1.0e-6);
    /// ```
    ///
    /// # Example: 3D Rotation Around Point
    ///
    /// ```
    /// # use nalgebra::{Similarity3, Point3, UnitQuaternion, Vector3};
    /// # use approx::assert_relative_eq;
    /// # use std::f32;
    /// // Rotate 180° around Y-axis, centered at (5, 0, 0), with 2x scaling
    /// let rotation = UnitQuaternion::from_axis_angle(
    ///     &Vector3::y_axis(),
    ///     f32::consts::PI
    /// );
    /// let center = Point3::new(5.0, 0.0, 0.0);
    /// let sim = Similarity3::rotation_wrt_point(rotation, center, 2.0);
    ///
    /// // Center point remains fixed
    /// assert_relative_eq!(sim * center, center, epsilon = 1.0e-6);
    /// ```
    ///
    /// # Example: Rotating UI Element Around Center
    ///
    /// ```
    /// # use nalgebra::{Similarity2, Point2, UnitComplex};
    /// # use std::f32;
    /// // UI element centered at (100, 50), rotating by angle
    /// let element_center = Point2::new(100.0, 50.0);
    /// let angle = f32::consts::FRAC_PI_4; // 45 degrees
    /// let rotation = UnitComplex::new(angle);
    ///
    /// // Rotate and scale around the center
    /// let transform = Similarity2::rotation_wrt_point(
    ///     rotation,
    ///     element_center,
    ///     1.2  // Slightly larger
    /// );
    ///
    /// // Element rotates in place and grows
    /// ```
    ///
    /// # See Also
    ///
    /// * [`new`](Self::new) - Create similarity with rotation around origin
    /// * [`from_parts`](Self::from_parts) - Create from individual components
    #[inline]
    pub fn rotation_wrt_point(r: R, p: Point<T, D>, scaling: T) -> Self {
        let shift = r.transform_vector(&-&p.coords);
        Self::from_parts(Translation::from(shift + p.coords), r, scaling)
    }
}

#[cfg(feature = "arbitrary")]
impl<T, R, const D: usize> Arbitrary for Similarity<T, R, D>
where
    T: crate::RealField + Arbitrary + Send,
    T::Element: crate::RealField,
    R: AbstractRotation<T, D> + Arbitrary + Send,
    Owned<T, crate::Const<D>>: Send,
{
    #[inline]
    fn arbitrary(rng: &mut Gen) -> Self {
        let mut s: T = Arbitrary::arbitrary(rng);
        while s.is_zero() {
            s = Arbitrary::arbitrary(rng)
        }

        Self::from_isometry(Arbitrary::arbitrary(rng), s)
    }
}

/*
 *
 * Constructors for various static dimensions.
 *
 */

// 2D similarity.
impl<T: SimdRealField> Similarity<T, Rotation2<T>, 2>
where
    T::Element: SimdRealField,
{
    /// Creates a new 2D similarity from translation, rotation angle, and scaling.
    ///
    /// This is the most convenient way to create a 2D similarity transformation. It combines
    /// all three components (translation, rotation, and uniform scaling) in one constructor.
    ///
    /// The transformation is applied in this order:
    /// 1. Scale the object by `scaling`
    /// 2. Rotate it by `angle` radians around the origin
    /// 3. Translate it by `translation`
    ///
    /// # Parameters
    ///
    /// * `translation` - The 2D translation vector (where to move)
    /// * `angle` - The rotation angle in radians (counterclockwise)
    /// * `scaling` - The uniform scaling factor (must be non-zero)
    ///
    /// # Returns
    ///
    /// A new 2D similarity transformation
    ///
    /// # Example: Basic 2D Similarity
    ///
    /// ```
    /// # use nalgebra::{SimilarityMatrix2, Similarity2, Vector2, Point2};
    /// # use approx::assert_relative_eq;
    /// # use std::f32;
    /// // Create a similarity: translate to (1, 2), rotate 90°, scale 3x
    /// let sim = SimilarityMatrix2::new(
    ///     Vector2::new(1.0, 2.0),
    ///     f32::consts::FRAC_PI_2,  // 90 degrees
    ///     3.0
    /// );
    ///
    /// let point = Point2::new(2.0, 4.0);
    /// let transformed = sim * point;
    ///
    /// assert_relative_eq!(transformed, Point2::new(-11.0, 8.0), epsilon = 1.0e-6);
    /// ```
    ///
    /// # Example: Game Sprite Transform
    ///
    /// ```
    /// # use nalgebra::{Similarity2, Vector2, Point2};
    /// # use approx::assert_relative_eq;
    /// # use std::f32;
    /// // Sprite at position (100, 200), facing up (no rotation), normal size
    /// let sprite_transform = Similarity2::new(
    ///     Vector2::new(100.0, 200.0),
    ///     0.0,    // No rotation
    ///     1.0     // Normal size
    /// );
    ///
    /// // Local sprite vertex at (1, 0)
    /// let local_vertex = Point2::new(1.0, 0.0);
    /// let world_vertex = sprite_transform * local_vertex;
    ///
    /// // Vertex is now at (101, 200) in world space
    /// assert_relative_eq!(world_vertex, Point2::new(101.0, 200.0), epsilon = 1.0e-6);
    /// ```
    ///
    /// # Example: Rotating and Scaling
    ///
    /// ```
    /// # use nalgebra::{Similarity2, Vector2, Point2};
    /// # use approx::assert_relative_eq;
    /// # use std::f32;
    /// // Object at origin, rotated 45°, doubled in size
    /// let sim = Similarity2::new(
    ///     Vector2::zeros(),
    ///     f32::consts::FRAC_PI_4,  // 45 degrees
    ///     2.0
    /// );
    ///
    /// let point = Point2::new(1.0, 0.0);
    /// let result = sim * point;
    ///
    /// // Point is scaled to (2, 0), then rotated 45° to (√2, √2)
    /// let sqrt2 = f32::sqrt(2.0);
    /// assert_relative_eq!(result, Point2::new(sqrt2, sqrt2), epsilon = 1.0e-6);
    /// ```
    ///
    /// # See Also
    ///
    /// * [`from_parts`](Self::from_parts) - Create from separate components
    /// * [`from_scaling`](Self::from_scaling) - Create pure scaling transformation
    /// * [`identity`](Self::identity) - Create identity transformation
    #[inline]
    pub fn new(translation: Vector2<T>, angle: T, scaling: T) -> Self {
        Self::from_parts(
            Translation::from(translation),
            Rotation2::new(angle),
            scaling,
        )
    }

    /// Converts the numeric type of this similarity to another type.
    ///
    /// This allows you to convert between different floating-point precisions (e.g., f32 to f64)
    /// or to custom scalar types, as long as the conversion is supported.
    ///
    /// # Type Parameters
    ///
    /// * `To` - The target scalar type
    ///
    /// # Returns
    ///
    /// A new similarity with components converted to type `To`
    ///
    /// # Example: f64 to f32
    ///
    /// ```
    /// # use nalgebra::SimilarityMatrix2;
    /// let sim = SimilarityMatrix2::<f64>::identity();
    /// let sim2 = sim.cast::<f32>();
    /// assert_eq!(sim2, SimilarityMatrix2::<f32>::identity());
    /// ```
    ///
    /// # Example: Converting Transformation Data
    ///
    /// ```
    /// # use nalgebra::{Similarity2, Vector2};
    /// // Compute in high precision
    /// let sim_f64 = Similarity2::<f64>::new(
    ///     Vector2::new(1.0, 2.0),
    ///     0.5,
    ///     2.0
    /// );
    ///
    /// // Convert to single precision for GPU
    /// let sim_f32: Similarity2<f32> = sim_f64.cast();
    /// ```
    ///
    /// # See Also
    ///
    /// * [`new`](Self::new) - Create a new similarity
    pub fn cast<To: Scalar>(self) -> Similarity<To, Rotation2<To>, 2>
    where
        Similarity<To, Rotation2<To>, 2>: SupersetOf<Self>,
    {
        crate::convert(self)
    }
}

impl<T: SimdRealField> Similarity<T, UnitComplex<T>, 2>
where
    T::Element: SimdRealField,
{
    /// Creates a new 2D similarity from translation, rotation angle, and scaling.
    ///
    /// This constructor is identical to `SimilarityMatrix2::new` but uses `UnitComplex`
    /// for internal rotation representation. Both produce the same transformations and can
    /// be used interchangeably based on your preference.
    ///
    /// # Parameters
    ///
    /// * `translation` - The 2D translation vector
    /// * `angle` - The rotation angle in radians (counterclockwise)
    /// * `scaling` - The uniform scaling factor (must be non-zero)
    ///
    /// # Returns
    ///
    /// A new 2D similarity transformation
    ///
    /// # Example
    ///
    /// ```
    /// # use nalgebra::{Similarity2, Vector2, Point2};
    /// # use approx::assert_relative_eq;
    /// # use std::f32;
    /// let sim = Similarity2::new(Vector2::new(1.0, 2.0), f32::consts::FRAC_PI_2, 3.0);
    ///
    /// assert_relative_eq!(sim * Point2::new(2.0, 4.0), Point2::new(-11.0, 8.0), epsilon = 1.0e-6);
    /// ```
    ///
    /// # See Also
    ///
    /// * [`identity`](Self::identity) - Create identity transformation
    /// * [`from_scaling`](Self::from_scaling) - Create pure scaling transformation
    #[inline]
    pub fn new(translation: Vector2<T>, angle: T, scaling: T) -> Self {
        Self::from_parts(
            Translation::from(translation),
            UnitComplex::new(angle),
            scaling,
        )
    }

    /// Converts the numeric type of this similarity to another type.
    ///
    /// See `SimilarityMatrix2::cast` for more details and examples.
    ///
    /// # Example
    ///
    /// ```
    /// # use nalgebra::Similarity2;
    /// let sim = Similarity2::<f64>::identity();
    /// let sim2 = sim.cast::<f32>();
    /// assert_eq!(sim2, Similarity2::<f32>::identity());
    /// ```
    pub fn cast<To: Scalar>(self) -> Similarity<To, UnitComplex<To>, 2>
    where
        Similarity<To, UnitComplex<To>, 2>: SupersetOf<Self>,
    {
        crate::convert(self)
    }
}

// 3D rotation.
macro_rules! similarity_construction_impl(
    ($Rot: ident) => {
        impl<T: SimdRealField> Similarity<T, $Rot<T>, 3>
        where T::Element: SimdRealField {
            /// Creates a new 3D similarity from translation, axis-angle rotation, and scaling.
            ///
            /// This is the main constructor for 3D similarity transformations. It combines
            /// translation, rotation (specified as axis-angle), and uniform scaling.
            ///
            /// The axis-angle representation is a 3D vector where:
            /// - The direction indicates the axis of rotation
            /// - The magnitude indicates the angle of rotation in radians
            ///
            /// The transformation is applied in this order:
            /// 1. Scale by `scaling`
            /// 2. Rotate by `axisangle`
            /// 3. Translate by `translation`
            ///
            /// # Parameters
            ///
            /// * `translation` - The 3D translation vector
            /// * `axisangle` - The rotation as an axis-angle vector (direction = axis, magnitude = angle in radians)
            /// * `scaling` - The uniform scaling factor (must be non-zero)
            ///
            /// # Returns
            ///
            /// A new 3D similarity transformation
            ///
            /// # Example: Basic 3D Similarity
            ///
            /// ```
            /// # use nalgebra::{Similarity3, SimilarityMatrix3, Point3, Vector3};
            /// # use approx::assert_relative_eq;
            /// # use std::f32;
            /// let axisangle = Vector3::y() * f32::consts::FRAC_PI_2; // 90° around Y-axis
            /// let translation = Vector3::new(1.0, 2.0, 3.0);
            ///
            /// // Similarity with rotation as UnitQuaternion (Similarity3)
            /// let sim = Similarity3::new(translation, axisangle, 3.0);
            ///
            /// let pt = Point3::new(4.0, 5.0, 6.0);
            /// assert_relative_eq!(sim * pt, Point3::new(19.0, 17.0, -9.0), epsilon = 1.0e-5);
            ///
            /// let vec = Vector3::new(4.0, 5.0, 6.0);
            /// assert_relative_eq!(sim * vec, Vector3::new(18.0, 15.0, -12.0), epsilon = 1.0e-5);
            /// ```
            ///
            /// # Example: Rotating Around Different Axes
            ///
            /// ```
            /// # use nalgebra::{Similarity3, Vector3};
            /// # use std::f32;
            /// // Rotate 180° around X-axis
            /// let sim_x = Similarity3::new(
            ///     Vector3::zeros(),
            ///     Vector3::x() * f32::consts::PI,
            ///     1.0
            /// );
            ///
            /// // Rotate 90° around a diagonal axis (normalized automatically)
            /// let axis = Vector3::new(1.0, 1.0, 0.0).normalize();
            /// let angle = f32::consts::FRAC_PI_2;
            /// let sim_diag = Similarity3::new(
            ///     Vector3::zeros(),
            ///     axis * angle,
            ///     1.0
            /// );
            /// ```
            ///
            /// # Example: 3D Game Object Transform
            ///
            /// ```
            /// # use nalgebra::{Similarity3, Vector3, Point3};
            /// # use std::f32;
            /// // Character at position (10, 0, 5), facing forward (no rotation), normal size
            /// let character = Similarity3::new(
            ///     Vector3::new(10.0, 0.0, 5.0),
            ///     Vector3::zeros(),  // No rotation
            ///     1.0                 // Normal size
            /// );
            ///
            /// // Boss enemy: larger and rotated
            /// let boss = Similarity3::new(
            ///     Vector3::new(20.0, 0.0, 10.0),
            ///     Vector3::y() * f32::consts::FRAC_PI_2,  // Facing different direction
            ///     2.5                                       // 2.5x larger
            /// );
            /// ```
            ///
            /// # See Also
            ///
            /// * [`from_parts`](Self::from_parts) - Create from separate components
            /// * [`identity`](Self::identity) - Create identity transformation
            /// * [`face_towards`](Self::face_towards) - Create from eye and target positions
            #[inline]
            pub fn new(translation: Vector3<T>, axisangle: Vector3<T>, scaling: T) -> Self
            {
                Self::from_isometry(Isometry::<_, $Rot<T>, 3>::new(translation, axisangle), scaling)
            }

            /// Converts the numeric type of this similarity to another type.
            ///
            /// See `SimilarityMatrix2::cast` for more details and examples.
            ///
            /// # Example
            ///
            /// ```
            /// # use nalgebra::Similarity3;
            /// let sim = Similarity3::<f64>::identity();
            /// let sim2 = sim.cast::<f32>();
            /// assert_eq!(sim2, Similarity3::<f32>::identity());
            /// ```
            pub fn cast<To: Scalar>(self) -> Similarity<To, $Rot<To>, 3>
            where
                Similarity<To, $Rot<To>, 3>: SupersetOf<Self>,
            {
                crate::convert(self)
            }

            /// Creates a similarity representing an observer's view, with scaling.
            ///
            /// This creates a coordinate frame for an observer at `eye` looking toward `target`,
            /// with an optional scaling factor. It's useful for placing and orienting cameras or
            /// objects that need to "look at" something while also being scaled.
            ///
            /// The resulting transformation:
            /// - Places the origin at `eye`
            /// - Orients so the positive Z-axis points from `eye` toward `target`
            /// - Scales by the given factor
            ///
            /// # Parameters
            ///
            /// * `eye` - The observer/camera position
            /// * `target` - The point to look at
            /// * `up` - The approximate "up" direction (must not be parallel to view direction)
            /// * `scaling` - The uniform scaling factor
            ///
            /// # Panics
            ///
            /// May produce unexpected results if `up` is parallel to `target - eye`.
            ///
            /// # Example: Camera Looking at Target
            ///
            /// ```
            /// # use nalgebra::{Similarity3, Point3, Vector3};
            /// # use approx::assert_relative_eq;
            /// let eye = Point3::new(1.0, 2.0, 3.0);
            /// let target = Point3::new(2.0, 2.0, 3.0);
            /// let up = Vector3::y();
            ///
            /// let sim = Similarity3::face_towards(&eye, &target, &up, 3.0);
            ///
            /// // Origin maps to eye position
            /// assert_eq!(sim * Point3::origin(), eye);
            ///
            /// // Z-axis points toward target (scaled)
            /// assert_relative_eq!(sim * Vector3::z(), Vector3::x() * 3.0, epsilon = 1.0e-6);
            /// ```
            ///
            /// # Example: Third-Person Camera
            ///
            /// ```
            /// # use nalgebra::{Similarity3, Point3, Vector3};
            /// // Camera behind and above the player
            /// let player_pos = Point3::new(0.0, 0.0, 0.0);
            /// let camera_pos = Point3::new(0.0, 5.0, -10.0);
            ///
            /// let camera_transform = Similarity3::face_towards(
            ///     &camera_pos,
            ///     &player_pos,
            ///     &Vector3::y(),
            ///     1.0
            /// );
            /// ```
            ///
            /// # See Also
            ///
            /// * [`look_at_rh`](Self::look_at_rh) - Right-handed look-at (for view matrices)
            /// * [`look_at_lh`](Self::look_at_lh) - Left-handed look-at (for view matrices)
            /// * [`new`](Self::new) - General constructor with explicit rotation
            #[inline]
            pub fn face_towards(eye:    &Point3<T>,
                                target: &Point3<T>,
                                up:     &Vector3<T>,
                                scaling: T)
                                -> Self {
                Self::from_isometry(Isometry::<_, $Rot<T>, 3>::face_towards(eye, target, up), scaling)
            }

            /// Deprecated: Use [`SimilarityMatrix3::face_towards`](Self::face_towards) instead.
            #[deprecated(note="renamed to `face_towards`")]
            pub fn new_observer_frames(eye:    &Point3<T>,
                                       target: &Point3<T>,
                                       up:     &Vector3<T>,
                                       scaling: T)
                                       -> Self {
                Self::face_towards(eye, target, up, scaling)
            }

            /// Builds a right-handed look-at view matrix including scaling factor.
            ///
            /// This conforms to the common notion of right handed look-at matrix from the computer
            /// graphics community.
            ///
            /// # Arguments
            ///   * eye - The eye position.
            ///   * target - The target position.
            ///   * up - A vector approximately aligned with required the vertical axis. The only
            ///     requirement of this parameter is to not be collinear to `target - eye`.
            ///
            /// # Example
            /// ```
            /// # #[macro_use] extern crate approx;
            /// # use std::f32;
            /// # use nalgebra::{Similarity3, SimilarityMatrix3, Point3, Vector3};
            /// let eye = Point3::new(1.0, 2.0, 3.0);
            /// let target = Point3::new(2.0, 2.0, 3.0);
            /// let up = Vector3::y();
            ///
            /// // Similarity with its rotation part represented as a UnitQuaternion
            /// let iso = Similarity3::look_at_rh(&eye, &target, &up, 3.0);
            /// assert_relative_eq!(iso * Vector3::x(), -Vector3::z() * 3.0, epsilon = 1.0e-6);
            ///
            /// // Similarity with its rotation part represented as Rotation3 (a 3x3 rotation matrix).
            /// let iso = SimilarityMatrix3::look_at_rh(&eye, &target, &up, 3.0);
            /// assert_relative_eq!(iso * Vector3::x(), -Vector3::z() * 3.0, epsilon = 1.0e-6);
            /// ```
            #[inline]
            pub fn look_at_rh(eye:     &Point3<T>,
                              target:  &Point3<T>,
                              up:      &Vector3<T>,
                              scaling: T)
                              -> Self {
                Self::from_isometry(Isometry::<_, $Rot<T>, 3>::look_at_rh(eye, target, up), scaling)
            }

            /// Builds a left-handed look-at view matrix including a scaling factor.
            ///
            /// This conforms to the common notion of left handed look-at matrix from the computer
            /// graphics community.
            ///
            /// # Arguments
            ///   * eye - The eye position.
            ///   * target - The target position.
            ///   * up - A vector approximately aligned with required the vertical axis. The only
            ///     requirement of this parameter is to not be collinear to `target - eye`.
            ///
            /// # Example
            /// ```
            /// # #[macro_use] extern crate approx;
            /// # use std::f32;
            /// # use nalgebra::{Similarity3, SimilarityMatrix3, Point3, Vector3};
            /// let eye = Point3::new(1.0, 2.0, 3.0);
            /// let target = Point3::new(2.0, 2.0, 3.0);
            /// let up = Vector3::y();
            ///
            /// // Similarity with its rotation part represented as a UnitQuaternion
            /// let sim = Similarity3::look_at_lh(&eye, &target, &up, 3.0);
            /// assert_relative_eq!(sim * Vector3::x(), Vector3::z() * 3.0, epsilon = 1.0e-6);
            ///
            /// // Similarity with its rotation part represented as Rotation3 (a 3x3 rotation matrix).
            /// let sim = SimilarityMatrix3::look_at_lh(&eye, &target, &up, 3.0);
            /// assert_relative_eq!(sim * Vector3::x(), Vector3::z() * 3.0, epsilon = 1.0e-6);
            /// ```
            #[inline]
            pub fn look_at_lh(eye:     &Point3<T>,
                              target:  &Point3<T>,
                              up:      &Vector3<T>,
                              scaling: T)
                              -> Self {
                Self::from_isometry(Isometry::<_, $Rot<T>, 3>::look_at_lh(eye, target, up), scaling)
            }
        }
    }
);

similarity_construction_impl!(Rotation3);
similarity_construction_impl!(UnitQuaternion);
