use approx::{AbsDiffEq, RelativeEq, UlpsEq};
use num_complex::Complex;
use std::fmt;

use crate::Scalar;
use crate::base::{Matrix2, Matrix3, Normed, Unit, Vector1, Vector2};
use crate::geometry::{Point2, Rotation2};
use simba::scalar::RealField;
use simba::simd::SimdRealField;
use std::cmp::{Eq, PartialEq};

/// A 2D rotation represented as a complex number with magnitude 1.
///
/// All the methods specific [`UnitComplex`] are listed here. You may also
/// read the documentation of the [`Complex`] type which
/// is used internally and accessible with `unit_complex.complex()`.
///
/// # Construction
/// * [Identity <span style="float:right;">`identity`</span>](#identity)
/// * [From a 2D rotation angle <span style="float:right;">`new`, `from_cos_sin_unchecked`…</span>](#construction-from-a-2d-rotation-angle)
/// * [From an existing 2D matrix or complex number <span style="float:right;">`from_matrix`, `rotation_to`, `powf`…</span>](#construction-from-an-existing-2d-matrix-or-complex-number)
/// * [From two vectors <span style="float:right;">`rotation_between`, `scaled_rotation_between_axis`…</span>](#construction-from-two-vectors)
///
/// # Transformation and composition
/// * [Angle extraction <span style="float:right;">`angle`, `angle_to`…</span>](#angle-extraction)
/// * [Transformation of a vector or a point <span style="float:right;">`transform_vector`, `inverse_transform_point`…</span>](#transformation-of-a-vector-or-a-point)
/// * [Conjugation and inversion <span style="float:right;">`conjugate`, `inverse_mut`…</span>](#conjugation-and-inversion)
/// * [Interpolation <span style="float:right;">`slerp`…</span>](#interpolation)
///
/// # Conversion
/// * [Conversion to a matrix <span style="float:right;">`to_rotation_matrix`, `to_homogeneous`…</span>](#conversion-to-a-matrix)
pub type UnitComplex<T> = Unit<Complex<T>>;

impl<T: Scalar + PartialEq> PartialEq for UnitComplex<T> {
    #[inline]
    fn eq(&self, rhs: &Self) -> bool {
        (**self).eq(&**rhs)
    }
}

impl<T: Scalar + Eq> Eq for UnitComplex<T> {}

impl<T: SimdRealField> Normed for Complex<T> {
    type Norm = T::SimdRealField;

    #[inline]
    fn norm(&self) -> T::SimdRealField {
        // We don't use `.norm_sqr()` because it requires
        // some very strong Num trait requirements.
        (self.re.clone() * self.re.clone() + self.im.clone() * self.im.clone()).simd_sqrt()
    }

    #[inline]
    fn norm_squared(&self) -> T::SimdRealField {
        // We don't use `.norm_sqr()` because it requires
        // some very strong Num trait requirements.
        self.re.clone() * self.re.clone() + self.im.clone() * self.im.clone()
    }

    #[inline]
    fn scale_mut(&mut self, n: Self::Norm) {
        self.re *= n.clone();
        self.im *= n;
    }

    #[inline]
    fn unscale_mut(&mut self, n: Self::Norm) {
        self.re /= n.clone();
        self.im /= n;
    }
}

/// # Angle extraction
impl<T: SimdRealField> UnitComplex<T>
where
    T::Element: SimdRealField,
{
    /// Returns the rotation angle in radians, in the range `]-pi; pi]`.
    ///
    /// A unit complex number represents a 2D rotation as a point on the unit circle.
    /// This method extracts the angle of rotation from the positive x-axis, measured
    /// counter-clockwise. The angle is always normalized to the range `]-pi; pi]`,
    /// which means angles wrap around: an angle of π + 0.1 becomes -π + 0.1.
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// # use nalgebra::UnitComplex;
    /// // Create a 45-degree rotation (π/4 radians)
    /// let rot = UnitComplex::new(std::f32::consts::FRAC_PI_4);
    /// assert_eq!(rot.angle(), std::f32::consts::FRAC_PI_4);
    /// ```
    ///
    /// The angle is always in the range `]-pi; pi]`:
    /// ```
    /// # use nalgebra::UnitComplex;
    /// # use std::f32;
    /// // Creating a rotation with angle > π wraps around
    /// let rot = UnitComplex::new(1.78);
    /// assert_eq!(rot.angle(), 1.78);
    ///
    /// // A full rotation (2π) is the same as no rotation (0)
    /// let full_rot = UnitComplex::new(2.0 * f32::consts::PI);
    /// assert!((full_rot.angle()).abs() < 1e-6);
    /// ```
    ///
    /// Practical use case - rotating a sprite in a 2D game:
    /// ```
    /// # use nalgebra::{UnitComplex, Vector2};
    /// // Get the current rotation angle of a game object
    /// let sprite_rotation = UnitComplex::new(0.5);
    /// let current_angle = sprite_rotation.angle();
    ///
    /// // Display the angle in degrees for debugging
    /// let degrees = current_angle.to_degrees();
    /// println!("Sprite is rotated {} degrees", degrees);
    /// ```
    ///
    /// # See Also
    /// * [`sin_angle`](Self::sin_angle) - Get only the sine component
    /// * [`cos_angle`](Self::cos_angle) - Get only the cosine component
    /// * [`angle_to`](Self::angle_to) - Calculate the angle between two rotations
    #[inline]
    #[must_use]
    pub fn angle(&self) -> T {
        self.im.clone().simd_atan2(self.re.clone())
    }

    /// Returns the sine of the rotation angle.
    ///
    /// Unit complex numbers store rotation using the cosine and sine of the angle
    /// (as the real and imaginary parts respectively). This method directly returns
    /// the sine component without computing the full angle, which is more efficient
    /// than calling `angle().sin()`.
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// # use nalgebra::UnitComplex;
    /// let angle = 1.78f32;
    /// let rot = UnitComplex::new(angle);
    /// assert_eq!(rot.sin_angle(), angle.sin());
    /// ```
    ///
    /// Using sine for vertical position calculations:
    /// ```
    /// # use nalgebra::UnitComplex;
    /// # use std::f32;
    /// // Calculate where a rotated point ends up on the y-axis
    /// let rotation = UnitComplex::new(f32::consts::FRAC_PI_6); // 30 degrees
    /// let radius = 10.0;
    /// let y_offset = radius * rotation.sin_angle();
    ///
    /// // At 30 degrees, sine is 0.5
    /// assert!((y_offset - 5.0).abs() < 1e-5);
    /// ```
    ///
    /// This is more efficient than computing the angle first:
    /// ```
    /// # use nalgebra::UnitComplex;
    /// let rot = UnitComplex::new(1.0);
    ///
    /// // Efficient: directly access the sine
    /// let sin1 = rot.sin_angle();
    ///
    /// // Less efficient: compute angle then sine
    /// let sin2 = rot.angle().sin();
    ///
    /// assert_eq!(sin1, sin2);
    /// ```
    ///
    /// # See Also
    /// * [`cos_angle`](Self::cos_angle) - Get the cosine of the rotation angle
    /// * [`angle`](Self::angle) - Get the full rotation angle
    #[inline]
    #[must_use]
    pub fn sin_angle(&self) -> T {
        self.im.clone()
    }

    /// Returns the cosine of the rotation angle.
    ///
    /// Unit complex numbers store rotation using the cosine and sine of the angle
    /// (as the real and imaginary parts respectively). This method directly returns
    /// the cosine component without computing the full angle, which is more efficient
    /// than calling `angle().cos()`.
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// # use nalgebra::UnitComplex;
    /// let angle = 1.78f32;
    /// let rot = UnitComplex::new(angle);
    /// assert_eq!(rot.cos_angle(), angle.cos());
    /// ```
    ///
    /// Using cosine for horizontal position calculations:
    /// ```
    /// # use nalgebra::UnitComplex;
    /// # use std::f32;
    /// // Calculate where a rotated point ends up on the x-axis
    /// let rotation = UnitComplex::new(f32::consts::FRAC_PI_3); // 60 degrees
    /// let radius = 10.0;
    /// let x_offset = radius * rotation.cos_angle();
    ///
    /// // At 60 degrees, cosine is 0.5
    /// assert!((x_offset - 5.0).abs() < 1e-5);
    /// ```
    ///
    /// Using both sine and cosine to transform a point:
    /// ```
    /// # use nalgebra::{UnitComplex, Point2};
    /// # use std::f32;
    /// let rotation = UnitComplex::new(f32::consts::FRAC_PI_4);
    /// let point = Point2::new(1.0, 0.0);
    ///
    /// // Manual rotation using cos and sin
    /// let cos = rotation.cos_angle();
    /// let sin = rotation.sin_angle();
    /// let rotated_x = cos * point.x - sin * point.y;
    /// let rotated_y = sin * point.x + cos * point.y;
    ///
    /// // Verify it matches the built-in transformation
    /// let rotated = rotation * point;
    /// assert!((rotated_x - rotated.x).abs() < 1e-6);
    /// assert!((rotated_y - rotated.y).abs() < 1e-6);
    /// ```
    ///
    /// # See Also
    /// * [`sin_angle`](Self::sin_angle) - Get the sine of the rotation angle
    /// * [`angle`](Self::angle) - Get the full rotation angle
    #[inline]
    #[must_use]
    pub fn cos_angle(&self) -> T {
        self.re.clone()
    }

    /// Returns the rotation angle as a 1-dimensional vector.
    ///
    /// This method wraps the rotation angle in a 1D vector for compatibility with
    /// generic programming interfaces that expect vector types. In most cases, you
    /// should use [`angle()`](Self::angle) instead, which returns the angle directly
    /// as a scalar value.
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// # use nalgebra::{UnitComplex, Vector1};
    /// let rotation = UnitComplex::new(1.5);
    /// let axis_angle = rotation.scaled_axis();
    ///
    /// assert_eq!(axis_angle, Vector1::new(1.5));
    /// ```
    ///
    /// Generic programming example:
    /// ```
    /// # use nalgebra::{UnitComplex, Vector1};
    /// fn get_rotation_vector<T>(rotation: &UnitComplex<T>) -> Vector1<T>
    /// where
    ///     T: nalgebra::RealField
    /// {
    ///     rotation.scaled_axis()
    /// }
    ///
    /// let rot = UnitComplex::new(0.7);
    /// let vec = get_rotation_vector(&rot);
    /// assert_eq!(vec[0], 0.7);
    /// ```
    ///
    /// # See Also
    /// * [`angle`](Self::angle) - Get the rotation angle as a scalar (recommended for most uses)
    /// * [`axis_angle`](Self::axis_angle) - Get the rotation as an axis-angle pair
    #[inline]
    #[must_use]
    pub fn scaled_axis(&self) -> Vector1<T> {
        Vector1::new(self.angle())
    }

    /// Returns the rotation axis and angle in the range (0, pi] of this unit complex number.
    ///
    /// For 2D rotations, the axis is always perpendicular to the plane (pointing either
    /// along +Z or -Z direction, represented as a 1D unit vector). This method is primarily
    /// used for generic programming to provide a consistent interface with 3D rotations.
    ///
    /// Returns `None` if the angle is zero (no rotation).
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// # use nalgebra::{UnitComplex, Vector1, Unit};
    /// // Positive rotation (counter-clockwise)
    /// let rotation = UnitComplex::new(1.0);
    /// if let Some((axis, angle)) = rotation.axis_angle() {
    ///     assert_eq!(axis, Unit::new_unchecked(Vector1::new(1.0)));
    ///     assert_eq!(angle, 1.0);
    /// }
    /// ```
    ///
    /// Negative rotation returns a flipped axis:
    /// ```
    /// # use nalgebra::{UnitComplex, Vector1, Unit};
    /// // Negative rotation (clockwise)
    /// let rotation = UnitComplex::new(-1.5);
    /// if let Some((axis, angle)) = rotation.axis_angle() {
    ///     assert_eq!(axis, Unit::new_unchecked(Vector1::new(-1.0)));
    ///     assert_eq!(angle, 1.5);
    /// }
    /// ```
    ///
    /// Identity rotation returns None:
    /// ```
    /// # use nalgebra::UnitComplex;
    /// let identity = UnitComplex::identity();
    /// assert!(identity.axis_angle().is_none());
    /// ```
    ///
    /// # See Also
    /// * [`angle`](Self::angle) - Get only the rotation angle (recommended for 2D)
    /// * [`scaled_axis`](Self::scaled_axis) - Get the angle as a 1D vector
    #[inline]
    #[must_use]
    pub fn axis_angle(&self) -> Option<(Unit<Vector1<T>>, T)>
    where
        T: RealField,
    {
        let ang = self.angle();

        if ang.is_zero() {
            None
        } else if ang.is_sign_positive() {
            Some((Unit::new_unchecked(Vector1::x()), ang))
        } else {
            Some((Unit::new_unchecked(-Vector1::<T>::x()), -ang))
        }
    }

    /// Returns the rotation angle needed to make `self` and `other` coincide.
    ///
    /// This computes the angular difference between two rotations. The result is the
    /// angle you need to rotate by to go from `self` to `other`. This is particularly
    /// useful for smooth rotation animations and calculating the shortest rotation path.
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::UnitComplex;
    /// let rot1 = UnitComplex::new(0.1);
    /// let rot2 = UnitComplex::new(1.7);
    /// let angle_diff = rot1.angle_to(&rot2);
    ///
    /// assert_relative_eq!(angle_diff, 1.6);
    /// ```
    ///
    /// The angle is always the shortest path:
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::UnitComplex;
    /// # use std::f32;
    /// // From 0 degrees to 350 degrees
    /// let rot1 = UnitComplex::new(0.0);
    /// let rot2 = UnitComplex::new(-10.0_f32.to_radians());
    /// let angle_diff = rot1.angle_to(&rot2);
    ///
    /// // Takes the short way: -10 degrees, not +350 degrees
    /// assert_relative_eq!(angle_diff, -10.0_f32.to_radians(), epsilon = 1e-6);
    /// ```
    ///
    /// Practical example - smooth rotation in a 2D game:
    /// ```
    /// # use nalgebra::UnitComplex;
    /// # use std::f32;
    /// // Current player rotation
    /// let current = UnitComplex::new(0.5);
    /// // Target rotation (where we want to face)
    /// let target = UnitComplex::new(2.0);
    ///
    /// // Calculate how much to rotate this frame
    /// let angle_to_target = current.angle_to(&target);
    /// let rotation_speed = 0.1; // radians per frame
    ///
    /// // Rotate a bit towards the target
    /// let rotation_this_frame = angle_to_target.signum() *
    ///     angle_to_target.abs().min(rotation_speed);
    /// let new_rotation = UnitComplex::new(current.angle() + rotation_this_frame);
    /// ```
    ///
    /// # See Also
    /// * [`angle`](Self::angle) - Get the rotation angle of a single rotation
    /// * [`rotation_to`](Self::rotation_to) - Get the rotation (not just angle) between two rotations
    #[inline]
    #[must_use]
    pub fn angle_to(&self, other: &Self) -> T {
        let delta = self.rotation_to(other);
        delta.angle()
    }
}

/// # Conjugation and inversion
impl<T: SimdRealField> UnitComplex<T>
where
    T::Element: SimdRealField,
{
    /// Computes the conjugate of this unit complex number.
    ///
    /// The conjugate of a unit complex number represents the inverse rotation.
    /// For a complex number `a + bi`, the conjugate is `a - bi`. Since unit complex
    /// numbers represent rotations, the conjugate represents rotating in the opposite
    /// direction by the same angle.
    ///
    /// This is equivalent to [`inverse()`](Self::inverse) for unit complex numbers.
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// # use nalgebra::UnitComplex;
    /// let rot = UnitComplex::new(1.78);
    /// let conj = rot.conjugate();
    ///
    /// // Imaginary part is negated
    /// assert_eq!(rot.complex().im, -conj.complex().im);
    /// // Real part stays the same
    /// assert_eq!(rot.complex().re, conj.complex().re);
    /// ```
    ///
    /// Conjugate reverses the rotation:
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{UnitComplex, Point2};
    /// # use std::f32;
    /// let rotation = UnitComplex::new(f32::consts::FRAC_PI_4);
    /// let point = Point2::new(1.0, 0.0);
    ///
    /// // Rotate then apply conjugate (reverse rotation)
    /// let rotated = rotation * point;
    /// let back = rotation.conjugate() * rotated;
    ///
    /// // We're back to the original point
    /// assert_relative_eq!(back, point, epsilon = 1e-6);
    /// ```
    ///
    /// Conjugate has the same angle magnitude but opposite sign:
    /// ```
    /// # use nalgebra::UnitComplex;
    /// let rotation = UnitComplex::new(1.5);
    /// let conjugate = rotation.conjugate();
    ///
    /// assert_eq!(rotation.angle(), -conjugate.angle());
    /// ```
    ///
    /// # See Also
    /// * [`inverse`](Self::inverse) - Same as conjugate for unit complex numbers
    /// * [`conjugate_mut`](Self::conjugate_mut) - In-place conjugation
    #[inline]
    #[must_use = "Did you mean to use conjugate_mut()?"]
    pub fn conjugate(&self) -> Self {
        Self::new_unchecked(self.conj())
    }

    /// Returns the inverse of this unit complex number.
    ///
    /// The inverse rotation undoes the original rotation. For unit complex numbers,
    /// the inverse is the same as the conjugate - it represents rotating in the opposite
    /// direction by the same angle.
    ///
    /// When you multiply a rotation by its inverse, you get the identity (no rotation).
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::UnitComplex;
    /// let rot = UnitComplex::new(1.2);
    /// let inv = rot.inverse();
    ///
    /// // Rotation times inverse gives identity
    /// assert_relative_eq!(rot * inv, UnitComplex::identity(), epsilon = 1.0e-6);
    /// assert_relative_eq!(inv * rot, UnitComplex::identity(), epsilon = 1.0e-6);
    /// ```
    ///
    /// Using inverse to undo a rotation:
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{UnitComplex, Point2};
    /// # use std::f32;
    /// let rotation = UnitComplex::new(f32::consts::FRAC_PI_3);
    /// let point = Point2::new(3.0, 4.0);
    ///
    /// // Apply rotation
    /// let rotated = rotation * point;
    /// // Undo with inverse
    /// let original = rotation.inverse() * rotated;
    ///
    /// assert_relative_eq!(original, point, epsilon = 1e-6);
    /// ```
    ///
    /// Practical example - reverting player rotation:
    /// ```
    /// # use nalgebra::{UnitComplex, Vector2};
    /// # use std::f32;
    /// // Player is rotated 45 degrees
    /// let player_rotation = UnitComplex::new(f32::consts::FRAC_PI_4);
    ///
    /// // We have a direction in world space
    /// let world_direction = Vector2::new(1.0, 1.0);
    ///
    /// // Convert to player's local space (undo player rotation)
    /// let local_direction = player_rotation.inverse() * world_direction;
    /// ```
    ///
    /// # See Also
    /// * [`conjugate`](Self::conjugate) - Same as inverse for unit complex numbers
    /// * [`inverse_mut`](Self::inverse_mut) - In-place inversion
    /// * [`inverse_transform_point`](Self::inverse_transform_point) - Transform a point by the inverse
    #[inline]
    #[must_use = "Did you mean to use inverse_mut()?"]
    pub fn inverse(&self) -> Self {
        self.conjugate()
    }

    /// Computes the conjugate of this unit complex number in-place.
    ///
    /// This modifies the unit complex number to its conjugate (inverse rotation) without
    /// allocating a new instance. The conjugate negates the imaginary part while keeping
    /// the real part the same, effectively reversing the rotation direction.
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// # use nalgebra::UnitComplex;
    /// let angle = 1.7;
    /// let original = UnitComplex::new(angle);
    /// let mut rotation = UnitComplex::new(angle);
    ///
    /// rotation.conjugate_mut();
    ///
    /// // Imaginary part is negated
    /// assert_eq!(original.complex().im, -rotation.complex().im);
    /// // Real part unchanged
    /// assert_eq!(original.complex().re, rotation.complex().re);
    /// ```
    ///
    /// In-place is more efficient when you don't need the original:
    /// ```
    /// # use nalgebra::UnitComplex;
    /// # use std::f32;
    /// let mut camera_rotation = UnitComplex::new(f32::consts::FRAC_PI_2);
    ///
    /// // Reverse the camera rotation in-place (no allocation)
    /// camera_rotation.conjugate_mut();
    ///
    /// assert_eq!(camera_rotation.angle(), -f32::consts::FRAC_PI_2);
    /// ```
    ///
    /// # See Also
    /// * [`conjugate`](Self::conjugate) - Returns a new conjugated rotation
    /// * [`inverse_mut`](Self::inverse_mut) - Same as conjugate_mut for unit complex numbers
    #[inline]
    pub fn conjugate_mut(&mut self) {
        let me = self.as_mut_unchecked();
        me.im = -me.im.clone();
    }

    /// Inverts this unit complex number in-place.
    ///
    /// This modifies the unit complex number to its inverse (opposite rotation) without
    /// allocating a new instance. For unit complex numbers, inversion is the same as
    /// conjugation - it negates the imaginary part to reverse the rotation direction.
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::UnitComplex;
    /// let angle = 1.7;
    /// let mut rot = UnitComplex::new(angle);
    /// rot.inverse_mut();
    ///
    /// // Now rot is the inverse of the original rotation
    /// assert_relative_eq!(rot * UnitComplex::new(angle), UnitComplex::identity());
    /// assert_relative_eq!(UnitComplex::new(angle) * rot, UnitComplex::identity());
    /// ```
    ///
    /// Practical example - toggling rotation direction:
    /// ```
    /// # use nalgebra::{UnitComplex, Vector2};
    /// # use std::f32;
    /// let mut rotation = UnitComplex::new(f32::consts::FRAC_PI_4);
    /// let vector = Vector2::new(1.0, 0.0);
    ///
    /// let rotated_cw = rotation * vector;
    ///
    /// // Reverse the rotation direction
    /// rotation.inverse_mut();
    /// let rotated_ccw = rotation * vector;
    ///
    /// // The two rotations are in opposite directions
    /// assert!((rotated_cw.y + rotated_ccw.y).abs() < 1e-6);
    /// ```
    ///
    /// # See Also
    /// * [`inverse`](Self::inverse) - Returns a new inverted rotation
    /// * [`conjugate_mut`](Self::conjugate_mut) - Same as inverse_mut for unit complex numbers
    #[inline]
    pub fn inverse_mut(&mut self) {
        self.conjugate_mut()
    }
}

/// # Conversion to a matrix
impl<T: SimdRealField> UnitComplex<T>
where
    T::Element: SimdRealField,
{
    /// Converts this unit complex number to its equivalent 2×2 rotation matrix.
    ///
    /// Unit complex numbers and 2D rotation matrices represent the same concept in
    /// different forms. This method converts from the complex number representation
    /// to the matrix representation. Both representations will produce identical
    /// rotations when applied to vectors or points.
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// # use nalgebra::{UnitComplex, Rotation2};
    /// # use std::f32;
    /// let rot = UnitComplex::new(f32::consts::FRAC_PI_6);
    /// let matrix = rot.to_rotation_matrix();
    ///
    /// let expected = Rotation2::new(f32::consts::FRAC_PI_6);
    /// assert_eq!(matrix, expected);
    /// ```
    ///
    /// Both representations produce the same rotation:
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{UnitComplex, Vector2};
    /// # use std::f32;
    /// let rotation = UnitComplex::new(f32::consts::FRAC_PI_4);
    /// let matrix = rotation.to_rotation_matrix();
    /// let vector = Vector2::new(1.0, 0.0);
    ///
    /// // Both produce the same result
    /// let result1 = rotation * vector;
    /// let result2 = matrix * vector;
    ///
    /// assert_relative_eq!(result1, result2, epsilon = 1e-6);
    /// ```
    ///
    /// When to use matrices vs unit complex numbers:
    /// ```
    /// # use nalgebra::{UnitComplex, Rotation2, Vector2};
    /// # use std::f32;
    /// let angle = f32::consts::FRAC_PI_3;
    ///
    /// // Unit complex: more compact, faster for rotations
    /// let uc = UnitComplex::new(angle);
    ///
    /// // Matrix: useful for interfacing with graphics APIs
    /// let mat = uc.to_rotation_matrix();
    ///
    /// // Both work the same way
    /// let v = Vector2::new(1.0, 1.0);
    /// let rotated1 = uc * v;
    /// let rotated2 = mat * v;
    /// ```
    ///
    /// # See Also
    /// * [`to_homogeneous`](Self::to_homogeneous) - Convert to a 3×3 homogeneous matrix
    /// * [`from_rotation_matrix`](Self::from_rotation_matrix) - Create from a rotation matrix
    #[inline]
    #[must_use]
    pub fn to_rotation_matrix(self) -> Rotation2<T> {
        let r = self.re.clone();
        let i = self.im.clone();

        Rotation2::from_matrix_unchecked(Matrix2::new(r.clone(), -i.clone(), i, r))
    }

    /// Converts this unit complex number into its equivalent 3×3 homogeneous transformation matrix.
    ///
    /// Homogeneous matrices are commonly used in computer graphics to represent transformations
    /// (rotation, translation, scaling) in a unified way. This method converts a 2D rotation
    /// into a 3×3 matrix where the top-left 2×2 block contains the rotation, and the rest
    /// represents no translation (identity). This format is compatible with most graphics APIs.
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{UnitComplex, Matrix3};
    /// # use std::f32;
    /// let rot = UnitComplex::new(f32::consts::FRAC_PI_6);
    /// let homogeneous = rot.to_homogeneous();
    ///
    /// let expected = Matrix3::new(
    ///     0.8660254, -0.5,      0.0,
    ///     0.5,       0.8660254, 0.0,
    ///     0.0,       0.0,       1.0
    /// );
    ///
    /// assert_relative_eq!(homogeneous, expected, epsilon = 1e-6);
    /// ```
    ///
    /// The homogeneous matrix has a specific structure:
    /// ```
    /// # use nalgebra::{UnitComplex, Matrix3};
    /// # use std::f32;
    /// let rotation = UnitComplex::new(1.0);
    /// let matrix = rotation.to_homogeneous();
    ///
    /// // Top-left 2×2 is the rotation
    /// let cos = rotation.cos_angle();
    /// let sin = rotation.sin_angle();
    /// assert_eq!(matrix[(0, 0)], cos);
    /// assert_eq!(matrix[(0, 1)], -sin);
    /// assert_eq!(matrix[(1, 0)], sin);
    /// assert_eq!(matrix[(1, 1)], cos);
    ///
    /// // Third row and column are identity (no translation)
    /// assert_eq!(matrix[(0, 2)], 0.0);
    /// assert_eq!(matrix[(1, 2)], 0.0);
    /// assert_eq!(matrix[(2, 2)], 1.0);
    /// ```
    ///
    /// Practical use - sending to a graphics shader:
    /// ```
    /// # use nalgebra::UnitComplex;
    /// # use std::f32;
    /// // Rotate a sprite 90 degrees
    /// let sprite_rotation = UnitComplex::new(f32::consts::FRAC_PI_2);
    ///
    /// // Convert to homogeneous matrix for shader uniform
    /// let transform_matrix = sprite_rotation.to_homogeneous();
    ///
    /// // In a real application, you'd send this to your GPU
    /// // gpu.set_uniform("u_transform", transform_matrix.as_slice());
    /// ```
    ///
    /// # See Also
    /// * [`to_rotation_matrix`](Self::to_rotation_matrix) - Convert to a 2×2 rotation matrix
    #[inline]
    #[must_use]
    pub fn to_homogeneous(self) -> Matrix3<T> {
        self.to_rotation_matrix().to_homogeneous()
    }
}

/// # Transformation of a vector or a point
impl<T: SimdRealField> UnitComplex<T>
where
    T::Element: SimdRealField,
{
    /// Rotates the given point by this unit complex number.
    ///
    /// This applies a 2D rotation to a point around the origin. The rotation is applied
    /// counter-clockwise for positive angles. This method is equivalent to using the
    /// multiplication operator `self * pt`, but may be more readable in some contexts.
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{UnitComplex, Point2};
    /// # use std::f32;
    /// // 90-degree rotation
    /// let rot = UnitComplex::new(f32::consts::FRAC_PI_2);
    /// let transformed_point = rot.transform_point(&Point2::new(1.0, 2.0));
    ///
    /// // Point (1, 2) rotates to approximately (-2, 1)
    /// assert_relative_eq!(transformed_point, Point2::new(-2.0, 1.0), epsilon = 1.0e-6);
    /// ```
    ///
    /// Same as using the multiplication operator:
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{UnitComplex, Point2};
    /// # use std::f32;
    /// let rotation = UnitComplex::new(f32::consts::FRAC_PI_4);
    /// let point = Point2::new(1.0, 0.0);
    ///
    /// let result1 = rotation.transform_point(&point);
    /// let result2 = rotation * point;
    ///
    /// assert_relative_eq!(result1, result2);
    /// ```
    ///
    /// Practical example - rotating a game object position:
    /// ```
    /// # use nalgebra::{UnitComplex, Point2};
    /// # use std::f32;
    /// // Enemy is at position (10, 5) relative to player
    /// let enemy_pos = Point2::new(10.0, 5.0);
    ///
    /// // Player rotates 45 degrees
    /// let player_rotation = UnitComplex::new(f32::consts::FRAC_PI_4);
    ///
    /// // Update enemy position relative to rotated player
    /// let new_enemy_pos = player_rotation.transform_point(&enemy_pos);
    /// ```
    ///
    /// Rotating multiple points:
    /// ```
    /// # use nalgebra::{UnitComplex, Point2};
    /// # use std::f32;
    /// let rotation = UnitComplex::new(f32::consts::FRAC_PI_6);
    /// let vertices = vec![
    ///     Point2::new(0.0, 0.0),
    ///     Point2::new(1.0, 0.0),
    ///     Point2::new(1.0, 1.0),
    /// ];
    ///
    /// let rotated: Vec<_> = vertices.iter()
    ///     .map(|p| rotation.transform_point(p))
    ///     .collect();
    /// ```
    ///
    /// # See Also
    /// * [`transform_vector`](Self::transform_vector) - Rotate a vector (same as point)
    /// * [`inverse_transform_point`](Self::inverse_transform_point) - Rotate by the inverse
    #[inline]
    #[must_use]
    pub fn transform_point(&self, pt: &Point2<T>) -> Point2<T> {
        self * pt
    }

    /// Rotates the given vector by this unit complex number.
    ///
    /// This applies a 2D rotation to a vector. For 2D rotations, rotating a vector is
    /// mathematically identical to rotating a point. This method is equivalent to using
    /// the multiplication operator `self * v`.
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{UnitComplex, Vector2};
    /// # use std::f32;
    /// // 90-degree rotation
    /// let rot = UnitComplex::new(f32::consts::FRAC_PI_2);
    /// let transformed_vector = rot.transform_vector(&Vector2::new(1.0, 2.0));
    ///
    /// // Vector (1, 2) rotates to approximately (-2, 1)
    /// assert_relative_eq!(transformed_vector, Vector2::new(-2.0, 1.0), epsilon = 1.0e-6);
    /// ```
    ///
    /// Practical example - rotating a velocity vector:
    /// ```
    /// # use nalgebra::{UnitComplex, Vector2};
    /// # use std::f32;
    /// // Projectile moving right at 10 units/sec
    /// let velocity = Vector2::new(10.0, 0.0);
    ///
    /// // Rotate 30 degrees to aim upward
    /// let aim_rotation = UnitComplex::new(30.0_f32.to_radians());
    /// let new_velocity = aim_rotation.transform_vector(&velocity);
    ///
    /// // Now projectile moves at an angle
    /// ```
    ///
    /// Rotating directional input:
    /// ```
    /// # use nalgebra::{UnitComplex, Vector2};
    /// # use std::f32;
    /// // Player input: moving up
    /// let input = Vector2::new(0.0, 1.0);
    ///
    /// // Camera is rotated 45 degrees
    /// let camera_rotation = UnitComplex::new(f32::consts::FRAC_PI_4);
    ///
    /// // Rotate input to match camera orientation
    /// let world_movement = camera_rotation.transform_vector(&input);
    /// ```
    ///
    /// Vectors and points rotate the same way in 2D:
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{UnitComplex, Vector2, Point2};
    /// # use std::f32;
    /// let rotation = UnitComplex::new(1.0);
    ///
    /// let vector = Vector2::new(3.0, 4.0);
    /// let point = Point2::new(3.0, 4.0);
    ///
    /// let rotated_vector = rotation.transform_vector(&vector);
    /// let rotated_point = rotation.transform_point(&point);
    ///
    /// assert_relative_eq!(rotated_vector.x, rotated_point.x);
    /// assert_relative_eq!(rotated_vector.y, rotated_point.y);
    /// ```
    ///
    /// # See Also
    /// * [`transform_point`](Self::transform_point) - Rotate a point (same for 2D)
    /// * [`inverse_transform_vector`](Self::inverse_transform_vector) - Rotate by the inverse
    #[inline]
    #[must_use]
    pub fn transform_vector(&self, v: &Vector2<T>) -> Vector2<T> {
        self * v
    }

    /// Rotates the given point by the inverse of this unit complex number.
    ///
    /// This applies the opposite rotation to a point. If the unit complex represents
    /// a rotation by angle θ, this method rotates by -θ. This is useful for undoing
    /// rotations or converting from world space to local space.
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{UnitComplex, Point2};
    /// # use std::f32;
    /// // 90-degree rotation
    /// let rot = UnitComplex::new(f32::consts::FRAC_PI_2);
    /// let transformed_point = rot.inverse_transform_point(&Point2::new(1.0, 2.0));
    ///
    /// // Point (1, 2) rotates by -90 degrees to (2, -1)
    /// assert_relative_eq!(transformed_point, Point2::new(2.0, -1.0), epsilon = 1.0e-6);
    /// ```
    ///
    /// Undoing a rotation:
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{UnitComplex, Point2};
    /// # use std::f32;
    /// let rotation = UnitComplex::new(f32::consts::FRAC_PI_3);
    /// let point = Point2::new(5.0, 3.0);
    ///
    /// // Rotate the point
    /// let rotated = rotation.transform_point(&point);
    ///
    /// // Undo the rotation
    /// let back = rotation.inverse_transform_point(&rotated);
    ///
    /// assert_relative_eq!(back, point, epsilon = 1e-6);
    /// ```
    ///
    /// Practical example - world to local space conversion:
    /// ```
    /// # use nalgebra::{UnitComplex, Point2};
    /// # use std::f32;
    /// // Player is facing 45 degrees from horizontal
    /// let player_rotation = UnitComplex::new(f32::consts::FRAC_PI_4);
    ///
    /// // Target is at this position in world space
    /// let target_world_pos = Point2::new(10.0, 5.0);
    ///
    /// // Convert target position to player's local coordinate system
    /// // (useful for determining if target is "in front" or "to the left")
    /// let target_local_pos = player_rotation.inverse_transform_point(&target_world_pos);
    ///
    /// // If target_local_pos.x > 0, target is to the player's right
    /// // If target_local_pos.y > 0, target is in front of the player
    /// ```
    ///
    /// # See Also
    /// * [`transform_point`](Self::transform_point) - Apply the forward rotation
    /// * [`inverse_transform_vector`](Self::inverse_transform_vector) - Same for vectors
    /// * [`inverse`](Self::inverse) - Get the inverse rotation
    #[inline]
    #[must_use]
    pub fn inverse_transform_point(&self, pt: &Point2<T>) -> Point2<T> {
        // TODO: would it be useful performance-wise not to call inverse explicitly (i-e. implement
        // the inverse transformation explicitly here) ?
        self.inverse() * pt
    }

    /// Rotates the given vector by the inverse of this unit complex number.
    ///
    /// This applies the opposite rotation to a vector. If the unit complex represents
    /// a rotation by angle θ, this method rotates by -θ. This is commonly used when
    /// converting directions from world space to local space.
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{UnitComplex, Vector2};
    /// # use std::f32;
    /// // 90-degree rotation
    /// let rot = UnitComplex::new(f32::consts::FRAC_PI_2);
    /// let transformed_vector = rot.inverse_transform_vector(&Vector2::new(1.0, 2.0));
    ///
    /// // Vector (1, 2) rotates by -90 degrees to (2, -1)
    /// assert_relative_eq!(transformed_vector, Vector2::new(2.0, -1.0), epsilon = 1.0e-6);
    /// ```
    ///
    /// Undoing a vector rotation:
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{UnitComplex, Vector2};
    /// let rotation = UnitComplex::new(1.5);
    /// let vector = Vector2::new(3.0, 4.0);
    ///
    /// // Rotate forward
    /// let rotated = rotation.transform_vector(&vector);
    ///
    /// // Rotate back
    /// let original = rotation.inverse_transform_vector(&rotated);
    ///
    /// assert_relative_eq!(original, vector, epsilon = 1e-6);
    /// ```
    ///
    /// Practical example - converting velocity to local space:
    /// ```
    /// # use nalgebra::{UnitComplex, Vector2};
    /// # use std::f32;
    /// // Vehicle is facing 30 degrees from horizontal
    /// let vehicle_rotation = UnitComplex::new(30.0_f32.to_radians());
    ///
    /// // Velocity in world coordinates
    /// let world_velocity = Vector2::new(5.0, 3.0);
    ///
    /// // Convert to vehicle's local coordinates to determine
    /// // forward/backward and left/right components
    /// let local_velocity = vehicle_rotation.inverse_transform_vector(&world_velocity);
    ///
    /// // local_velocity.x is sideways speed (positive = moving right)
    /// // local_velocity.y is forward speed (positive = moving forward)
    /// ```
    ///
    /// Camera-relative movement:
    /// ```
    /// # use nalgebra::{UnitComplex, Vector2};
    /// # use std::f32;
    /// // Camera is rotated
    /// let camera_rotation = UnitComplex::new(f32::consts::FRAC_PI_4);
    ///
    /// // Object is moving in world space
    /// let world_direction = Vector2::new(1.0, 0.0);
    ///
    /// // Convert to camera space to know if object moves left/right on screen
    /// let camera_relative = camera_rotation.inverse_transform_vector(&world_direction);
    /// ```
    ///
    /// # See Also
    /// * [`transform_vector`](Self::transform_vector) - Apply the forward rotation
    /// * [`inverse_transform_point`](Self::inverse_transform_point) - Same for points
    #[inline]
    #[must_use]
    pub fn inverse_transform_vector(&self, v: &Vector2<T>) -> Vector2<T> {
        self.inverse() * v
    }

    /// Rotates the given unit vector by the inverse of this unit complex number.
    ///
    /// This is a specialized version of [`inverse_transform_vector`](Self::inverse_transform_vector)
    /// for unit vectors (vectors with length 1). The result is guaranteed to remain a unit vector,
    /// which can avoid normalization in some algorithms.
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{UnitComplex, Vector2};
    /// # use std::f32;
    /// let rot = UnitComplex::new(f32::consts::FRAC_PI_2);
    /// let transformed = rot.inverse_transform_unit_vector(&Vector2::x_axis());
    ///
    /// // X-axis rotated by -90 degrees gives -Y axis
    /// assert_relative_eq!(transformed, -Vector2::y_axis(), epsilon = 1.0e-6);
    /// ```
    ///
    /// Unit vectors stay unit length after rotation:
    /// ```
    /// # use nalgebra::{UnitComplex, Vector2};
    /// # use std::f32;
    /// let rotation = UnitComplex::new(1.23);
    /// let unit_vec = Vector2::y_axis();
    ///
    /// let rotated = rotation.inverse_transform_unit_vector(&unit_vec);
    ///
    /// // Still has length 1
    /// assert!((rotated.norm() - 1.0).abs() < 1e-6);
    /// ```
    ///
    /// Practical example - converting facing direction:
    /// ```
    /// # use nalgebra::{UnitComplex, Vector2, Unit};
    /// # use std::f32;
    /// // Enemy is facing this direction in world space
    /// let enemy_facing = Unit::new_normalize(Vector2::new(1.0, 1.0));
    ///
    /// // Camera is rotated 45 degrees
    /// let camera_rotation = UnitComplex::new(f32::consts::FRAC_PI_4);
    ///
    /// // Convert enemy's facing to camera space
    /// let facing_in_camera_space = camera_rotation.inverse_transform_unit_vector(&enemy_facing);
    ///
    /// // Use this to draw an arrow showing which way the enemy faces on screen
    /// ```
    ///
    /// # See Also
    /// * [`inverse_transform_vector`](Self::inverse_transform_vector) - For non-unit vectors
    /// * [`transform_vector`](Self::transform_vector) - Apply the forward rotation
    #[inline]
    #[must_use]
    pub fn inverse_transform_unit_vector(&self, v: &Unit<Vector2<T>>) -> Unit<Vector2<T>> {
        self.inverse() * v
    }
}

/// # Interpolation
impl<T: SimdRealField> UnitComplex<T>
where
    T::Element: SimdRealField,
{
    /// Performs spherical linear interpolation (SLERP) between two rotations.
    ///
    /// SLERP produces a smooth rotation that interpolates between `self` and `other` along
    /// the shortest arc on the unit circle. The parameter `t` controls the interpolation:
    /// - `t = 0.0` returns `self`
    /// - `t = 1.0` returns `other`
    /// - `t = 0.5` returns the rotation halfway between
    ///
    /// This is essential for smooth rotation animations in games and graphics applications.
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::UnitComplex;
    /// # use std::f32;
    ///
    /// let rot1 = UnitComplex::new(f32::consts::FRAC_PI_4);  // 45 degrees
    /// let rot2 = UnitComplex::new(-f32::consts::PI);         // -180 degrees
    ///
    /// // One third of the way from rot1 to rot2
    /// let rot = rot1.slerp(&rot2, 1.0 / 3.0);
    ///
    /// assert_relative_eq!(rot.angle(), f32::consts::FRAC_PI_2); // 90 degrees
    /// ```
    ///
    /// Interpolation endpoints:
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::UnitComplex;
    /// let start = UnitComplex::new(0.5);
    /// let end = UnitComplex::new(2.0);
    ///
    /// // t = 0 gives the start rotation
    /// assert_relative_eq!(start.slerp(&end, 0.0), start);
    ///
    /// // t = 1 gives the end rotation
    /// assert_relative_eq!(start.slerp(&end, 1.0), end);
    ///
    /// // t = 0.5 gives the midpoint
    /// let mid = start.slerp(&end, 0.5);
    /// assert_relative_eq!(mid.angle(), 1.25);
    /// ```
    ///
    /// Smooth rotation animation:
    /// ```
    /// # use nalgebra::UnitComplex;
    /// # use std::f32;
    /// let current_rotation = UnitComplex::new(0.0);
    /// let target_rotation = UnitComplex::new(f32::consts::FRAC_PI_2);
    ///
    /// // Simulate animation over several frames
    /// let mut rotation = current_rotation;
    /// for frame in 0..10 {
    ///     let t = (frame as f32) / 10.0; // 0.0 to 0.9
    ///     rotation = current_rotation.slerp(&target_rotation, t);
    ///     // Draw sprite with 'rotation'
    /// }
    /// ```
    ///
    /// Practical example - smooth camera rotation:
    /// ```
    /// # use nalgebra::UnitComplex;
    /// # use std::f32;
    /// struct Camera {
    ///     rotation: UnitComplex<f32>,
    ///     target_rotation: UnitComplex<f32>,
    /// }
    ///
    /// impl Camera {
    ///     fn update(&mut self, delta_time: f32) {
    ///         // Smoothly rotate toward target (10% per second)
    ///         let t = (delta_time * 0.1).min(1.0);
    ///         self.rotation = self.rotation.slerp(&self.target_rotation, t);
    ///     }
    /// }
    /// ```
    ///
    /// # See Also
    /// * [`angle_to`](Self::angle_to) - Calculate the angular distance between rotations
    /// * [`rotation_to`](Self::rotation_to) - Get the rotation between two rotations
    #[inline]
    #[must_use]
    pub fn slerp(&self, other: &Self, t: T) -> Self {
        let delta = other / self;
        self * Self::new(delta.angle() * t)
    }
}

impl<T: RealField + fmt::Display> fmt::Display for UnitComplex<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "UnitComplex angle: {}", self.angle())
    }
}

impl<T: RealField> AbsDiffEq for UnitComplex<T> {
    type Epsilon = T;

    #[inline]
    fn default_epsilon() -> Self::Epsilon {
        T::default_epsilon()
    }

    #[inline]
    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        self.re.abs_diff_eq(&other.re, epsilon.clone()) && self.im.abs_diff_eq(&other.im, epsilon)
    }
}

impl<T: RealField> RelativeEq for UnitComplex<T> {
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
        self.re
            .relative_eq(&other.re, epsilon.clone(), max_relative.clone())
            && self.im.relative_eq(&other.im, epsilon, max_relative)
    }
}

impl<T: RealField> UlpsEq for UnitComplex<T> {
    #[inline]
    fn default_max_ulps() -> u32 {
        T::default_max_ulps()
    }

    #[inline]
    fn ulps_eq(&self, other: &Self, epsilon: Self::Epsilon, max_ulps: u32) -> bool {
        self.re.ulps_eq(&other.re, epsilon.clone(), max_ulps)
            && self.im.ulps_eq(&other.im, epsilon, max_ulps)
    }
}
