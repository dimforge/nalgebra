#[cfg(feature = "arbitrary")]
use quickcheck::{Arbitrary, Gen};

#[cfg(feature = "rand-no-std")]
use rand::{
    Rng,
    distr::{Distribution, StandardUniform},
};

use num::One;
use num_complex::Complex;

use crate::base::dimension::{U1, U2};
use crate::base::storage::Storage;
use crate::base::{Matrix2, Scalar, Unit, Vector, Vector2};
use crate::geometry::{Rotation2, UnitComplex};
use simba::scalar::{RealField, SupersetOf};
use simba::simd::SimdRealField;

impl<T: SimdRealField> Default for UnitComplex<T>
where
    T::Element: SimdRealField,
{
    fn default() -> Self {
        Self::identity()
    }
}

/// # Identity
impl<T: SimdRealField> UnitComplex<T>
where
    T::Element: SimdRealField,
{
    /// Returns the identity rotation (no rotation).
    ///
    /// The identity unit complex number represents zero rotation. It has an angle of 0
    /// and corresponds to the complex number `1 + 0i`. When you multiply any rotation
    /// by the identity, you get the original rotation back unchanged.
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// # use nalgebra::UnitComplex;
    /// let identity = UnitComplex::identity();
    /// let rotation = UnitComplex::new(1.7);
    ///
    /// // Identity times anything equals that thing
    /// assert_eq!(identity * rotation, rotation);
    /// assert_eq!(rotation * identity, rotation);
    /// ```
    ///
    /// Identity has zero angle:
    /// ```
    /// # use nalgebra::UnitComplex;
    /// let identity = UnitComplex::identity();
    ///
    /// assert_eq!(identity.angle(), 0.0);
    /// assert_eq!(identity.cos_angle(), 1.0);
    /// assert_eq!(identity.sin_angle(), 0.0);
    /// ```
    ///
    /// Identity rotation doesn't change points:
    /// ```
    /// # use nalgebra::{UnitComplex, Point2};
    /// let identity = UnitComplex::identity();
    /// let point = Point2::new(3.0, 4.0);
    ///
    /// let rotated = identity * point;
    /// assert_eq!(rotated, point);
    /// ```
    ///
    /// Practical example - initial rotation for game objects:
    /// ```
    /// # use nalgebra::UnitComplex;
    /// struct GameObject {
    ///     rotation: UnitComplex<f32>,
    /// }
    ///
    /// impl GameObject {
    ///     fn new() -> Self {
    ///         GameObject {
    ///             // Start with no rotation
    ///             rotation: UnitComplex::identity(),
    ///         }
    ///     }
    /// }
    /// ```
    ///
    /// # See Also
    /// * [`new`](Self::new) - Create a rotation with a specific angle
    #[inline]
    pub fn identity() -> Self {
        Self::new_unchecked(Complex::new(T::one(), T::zero()))
    }
}

/// # Construction from a 2D rotation angle
impl<T: SimdRealField> UnitComplex<T>
where
    T::Element: SimdRealField,
{
    /// Creates a unit complex number representing a rotation by the given angle in radians.
    ///
    /// This is the primary way to create a 2D rotation. The angle is measured in radians,
    /// with positive angles rotating counter-clockwise and negative angles rotating clockwise.
    ///
    /// Unit complex numbers are an efficient way to represent 2D rotations - they use only
    /// 2 floats (compared to 4 for a 2×2 matrix) and rotation operations are fast complex
    /// number multiplications.
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{UnitComplex, Point2};
    /// # use std::f32;
    ///
    /// // 90-degree rotation (π/2 radians)
    /// let rot = UnitComplex::new(f32::consts::FRAC_PI_2);
    ///
    /// // Rotating point (3, 4) by 90° gives (-4, 3)
    /// assert_relative_eq!(rot * Point2::new(3.0, 4.0), Point2::new(-4.0, 3.0));
    /// ```
    ///
    /// Positive angles rotate counter-clockwise:
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{UnitComplex, Vector2};
    /// # use std::f32;
    ///
    /// // 45-degree counter-clockwise rotation
    /// let rot = UnitComplex::new(f32::consts::FRAC_PI_4);
    /// let right = Vector2::new(1.0, 0.0);
    ///
    /// let rotated = rot * right;
    /// // Now pointing up and to the right
    /// assert_relative_eq!(rotated, Vector2::new(0.7071068, 0.7071068), epsilon = 1e-6);
    /// ```
    ///
    /// Negative angles rotate clockwise:
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{UnitComplex, Vector2};
    /// # use std::f32;
    ///
    /// // 90-degree clockwise rotation
    /// let rot = UnitComplex::new(-f32::consts::FRAC_PI_2);
    /// let up = Vector2::new(0.0, 1.0);
    ///
    /// let rotated = rot * up;
    /// // Now pointing right
    /// assert_relative_eq!(rotated, Vector2::new(1.0, 0.0), epsilon = 1e-6);
    /// ```
    ///
    /// Converting from degrees:
    /// ```
    /// # use nalgebra::UnitComplex;
    /// // Rotate by 30 degrees
    /// let angle_degrees = 30.0_f32;
    /// let angle_radians = angle_degrees.to_radians();
    /// let rot = UnitComplex::new(angle_radians);
    /// ```
    ///
    /// Practical example - rotating a 2D game sprite:
    /// ```
    /// # use nalgebra::{UnitComplex, Vector2};
    /// # use std::f32;
    /// struct Sprite {
    ///     position: Vector2<f32>,
    ///     rotation: UnitComplex<f32>,
    /// }
    ///
    /// impl Sprite {
    ///     fn new() -> Self {
    ///         Sprite {
    ///             position: Vector2::zeros(),
    ///             rotation: UnitComplex::identity(),
    ///         }
    ///     }
    ///
    ///     fn rotate(&mut self, angle_delta: f32) {
    ///         // Combine rotations by multiplying
    ///         let additional_rotation = UnitComplex::new(angle_delta);
    ///         self.rotation = additional_rotation * self.rotation;
    ///     }
    /// }
    /// ```
    ///
    /// # See Also
    /// * [`from_angle`](Self::from_angle) - Alias for `new`
    /// * [`from_cos_sin_unchecked`](Self::from_cos_sin_unchecked) - Create from pre-computed cos and sin
    /// * [`identity`](Self::identity) - Create a zero rotation
    #[inline]
    pub fn new(angle: T) -> Self {
        let (sin, cos) = angle.simd_sin_cos();
        Self::from_cos_sin_unchecked(cos, sin)
    }

    /// Creates a unit complex number representing a rotation by the given angle in radians.
    ///
    /// This is an alias for [`new`](Self::new) provided for API consistency. Both methods
    /// do exactly the same thing - create a 2D rotation from an angle.
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{UnitComplex, Point2};
    /// # use std::f32;
    /// let rot = UnitComplex::from_angle(f32::consts::FRAC_PI_2);
    ///
    /// assert_relative_eq!(rot * Point2::new(3.0, 4.0), Point2::new(-4.0, 3.0));
    /// ```
    ///
    /// Equivalent to `new`:
    /// ```
    /// # use nalgebra::UnitComplex;
    /// let rot1 = UnitComplex::new(1.5);
    /// let rot2 = UnitComplex::from_angle(1.5);
    ///
    /// assert_eq!(rot1, rot2);
    /// ```
    ///
    /// # See Also
    /// * [`new`](Self::new) - The preferred method (same functionality)
    // TODO: deprecate this.
    #[inline]
    pub fn from_angle(angle: T) -> Self {
        Self::new(angle)
    }

    /// Creates a unit complex number from pre-computed cosine and sine values.
    ///
    /// This is an advanced constructor that directly uses the cosine and sine of a rotation
    /// angle without computing them. This is useful when you already have these values
    /// from another source and want to avoid recomputing them.
    ///
    /// **Important**: The values are not validated! For a valid rotation, `cos² + sin² = 1`
    /// must hold. If you pass invalid values, the resulting "unit" complex number won't
    /// actually have unit magnitude, which will break rotations. Use [`new`](Self::new)
    /// instead unless you're certain the values are correct.
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{UnitComplex, Point2};
    /// # use std::f32;
    /// let angle = f32::consts::FRAC_PI_2;
    /// let cos = angle.cos();
    /// let sin = angle.sin();
    ///
    /// let rot = UnitComplex::from_cos_sin_unchecked(cos, sin);
    /// assert_relative_eq!(rot * Point2::new(3.0, 4.0), Point2::new(-4.0, 3.0));
    /// ```
    ///
    /// Avoiding recomputation when you already have cos/sin:
    /// ```
    /// # use nalgebra::UnitComplex;
    /// # use std::f32;
    /// // Imagine these come from a physics simulation
    /// let direction_cos = 0.8;
    /// let direction_sin = 0.6;
    ///
    /// // Create rotation without recomputing trig functions
    /// let rotation = UnitComplex::from_cos_sin_unchecked(direction_cos, direction_sin);
    /// ```
    ///
    /// **Warning**: Invalid input breaks rotations:
    /// ```
    /// # use nalgebra::{UnitComplex, Vector2};
    /// // These don't form a valid unit circle point!
    /// let bad_rotation = UnitComplex::from_cos_sin_unchecked(1.0, 1.0);
    ///
    /// let vec = Vector2::new(1.0, 0.0);
    /// let result = bad_rotation * vec;
    ///
    /// // Result will have wrong magnitude (not a pure rotation)
    /// // The vector length changed from 1.0 to ~1.41
    /// assert!((result.norm() - 1.0).abs() > 0.4);
    /// ```
    ///
    /// # See Also
    /// * [`new`](Self::new) - Safe constructor from angle (recommended)
    /// * [`from_complex`](Self::from_complex) - Create from a complex number (normalized)
    #[inline]
    pub const fn from_cos_sin_unchecked(cos: T, sin: T) -> Self {
        Self::new_unchecked(Complex::new(cos, sin))
    }

    /// Creates a unit complex number from an angle stored in a 1-dimensional vector.
    ///
    /// This method extracts the angle from a `Vector1` and creates the corresponding rotation.
    /// It's primarily used for generic programming where rotation parameters need to be
    /// represented as vectors for API consistency with higher-dimensional rotations.
    ///
    /// For normal use, [`new`](Self::new) is simpler and more direct.
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// # use nalgebra::{UnitComplex, Vector1};
    /// let angle_vec = Vector1::new(1.5);
    /// let rotation = UnitComplex::from_scaled_axis(angle_vec);
    ///
    /// assert_eq!(rotation.angle(), 1.5);
    /// ```
    ///
    /// Equivalent to using `new` directly:
    /// ```
    /// # use nalgebra::{UnitComplex, Vector1};
    /// let angle = 1.5;
    ///
    /// let rot1 = UnitComplex::new(angle);
    /// let rot2 = UnitComplex::from_scaled_axis(Vector1::new(angle));
    ///
    /// assert_eq!(rot1, rot2);
    /// ```
    ///
    /// # See Also
    /// * [`new`](Self::new) - Create from a scalar angle (recommended for most uses)
    /// * [`scaled_axis`](Self::scaled_axis) - Get the angle as a vector
    #[inline]
    pub fn from_scaled_axis<SB: Storage<T, U1>>(axisangle: Vector<T, U1, SB>) -> Self {
        Self::from_angle(axisangle[0].clone())
    }
}

/// # Construction from an existing 2D matrix or complex number
impl<T: SimdRealField> UnitComplex<T>
where
    T::Element: SimdRealField,
{
    /// Casts the components of this unit complex number to another type.
    ///
    /// This converts the internal floating-point representation from one type to another
    /// (e.g., from `f64` to `f32`). This is useful when interfacing between different
    /// precision requirements or different libraries.
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::UnitComplex;
    /// // Create with 64-bit precision
    /// let rotation_f64 = UnitComplex::new(1.0f64);
    ///
    /// // Cast to 32-bit precision
    /// let rotation_f32 = rotation_f64.cast::<f32>();
    ///
    /// assert_relative_eq!(rotation_f32, UnitComplex::new(1.0f32));
    /// ```
    ///
    /// Converting between precision levels:
    /// ```
    /// # use nalgebra::UnitComplex;
    /// # use std::f64;
    /// // High precision calculation
    /// let precise_angle = f64::consts::PI / 3.0;
    /// let precise_rotation = UnitComplex::new(precise_angle);
    ///
    /// // Cast to f32 for GPU upload
    /// let gpu_rotation = precise_rotation.cast::<f32>();
    /// ```
    ///
    /// # See Also
    /// * [`new`](Self::new) - Create a rotation at a specific type
    pub fn cast<To: Scalar>(self) -> UnitComplex<To>
    where
        UnitComplex<To>: SupersetOf<Self>,
    {
        crate::convert(self)
    }

    /// Returns a reference to the underlying complex number.
    ///
    /// Unit complex numbers are internally stored as `Complex<T>` (from the `num-complex` crate)
    /// where the real part is the cosine and the imaginary part is the sine of the rotation angle.
    /// This method gives you access to that internal representation.
    ///
    /// This is equivalent to calling `self.as_ref()`.
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// # extern crate num_complex;
    /// # use num_complex::Complex;
    /// # use nalgebra::UnitComplex;
    /// let angle = 1.78f32;
    /// let rot = UnitComplex::new(angle);
    ///
    /// let complex = rot.complex();
    /// assert_eq!(*complex, Complex::new(angle.cos(), angle.sin()));
    /// ```
    ///
    /// Accessing real and imaginary parts:
    /// ```
    /// # use nalgebra::UnitComplex;
    /// # use std::f32;
    /// let rotation = UnitComplex::new(f32::consts::FRAC_PI_4);
    /// let complex = rotation.complex();
    ///
    /// // Real part is cosine of angle
    /// assert!((complex.re - rotation.cos_angle()).abs() < 1e-6);
    /// // Imaginary part is sine of angle
    /// assert!((complex.im - rotation.sin_angle()).abs() < 1e-6);
    /// ```
    ///
    /// # See Also
    /// * [`from_complex`](Self::from_complex) - Create from a complex number
    /// * [`cos_angle`](Self::cos_angle) - Get the cosine (real part)
    /// * [`sin_angle`](Self::sin_angle) - Get the sine (imaginary part)
    #[inline]
    #[must_use]
    pub fn complex(&self) -> &Complex<T> {
        self.as_ref()
    }

    /// Creates a new unit complex number from a complex number by normalizing it.
    ///
    /// This takes any complex number and normalizes it to have magnitude 1, creating a valid
    /// unit complex number that represents a rotation. The angle of rotation is preserved
    /// from the input complex number.
    ///
    /// If you also need the original magnitude, use [`from_complex_and_get`](Self::from_complex_and_get).
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// # extern crate num_complex;
    /// # use num_complex::Complex;
    /// # use nalgebra::UnitComplex;
    /// // Create from a non-unit complex number
    /// let c = Complex::new(3.0, 4.0); // magnitude is 5.0
    /// let rotation = UnitComplex::from_complex(c);
    ///
    /// // Result is normalized to unit magnitude
    /// let result = rotation.complex();
    /// assert!((result.norm() - 1.0).abs() < 1e-6);
    /// ```
    ///
    /// The angle is preserved during normalization:
    /// ```
    /// # extern crate num_complex;
    /// # use num_complex::Complex;
    /// # use nalgebra::UnitComplex;
    /// let c = Complex::new(2.0, 2.0);
    /// let rotation = UnitComplex::from_complex(c);
    ///
    /// // Angle is atan2(2, 2) = π/4
    /// assert!((rotation.angle() - std::f64::consts::FRAC_PI_4).abs() < 1e-6);
    /// ```
    ///
    /// # See Also
    /// * [`from_complex_and_get`](Self::from_complex_and_get) - Also returns the original magnitude
    /// * [`complex`](Self::complex) - Get the internal complex number
    #[inline]
    pub fn from_complex(q: Complex<T>) -> Self {
        Self::from_complex_and_get(q).0
    }

    /// Creates a new unit complex number from a complex number and returns its original magnitude.
    ///
    /// This is like [`from_complex`](Self::from_complex), but also returns the magnitude of the
    /// input complex number before normalization. This is useful when you need both the rotation
    /// direction (as a unit complex) and the scale factor separately.
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// # extern crate num_complex;
    /// # use num_complex::Complex;
    /// # use nalgebra::UnitComplex;
    /// let c = Complex::new(3.0, 4.0);
    /// let (rotation, magnitude) = UnitComplex::from_complex_and_get(c);
    ///
    /// // The magnitude was 5.0
    /// assert!((magnitude - 5.0).abs() < 1e-6);
    ///
    /// // The rotation is normalized
    /// assert!((rotation.complex().norm() - 1.0).abs() < 1e-6);
    /// ```
    ///
    /// Practical example - decomposing a scaled rotation:
    /// ```
    /// # extern crate num_complex;
    /// # use num_complex::Complex;
    /// # use nalgebra::{UnitComplex, Vector2};
    /// // A vector that represents both direction and magnitude
    /// let combined = Complex::new(6.0, 8.0);
    ///
    /// // Separate into rotation and scale
    /// let (rotation, scale) = UnitComplex::from_complex_and_get(combined);
    ///
    /// // Use rotation for direction, scale for magnitude
    /// let direction = Vector2::new(1.0, 0.0);
    /// let result = (rotation * direction) * scale;
    ///
    /// assert!((result.x - 6.0).abs() < 1e-6);
    /// assert!((result.y - 8.0).abs() < 1e-6);
    /// ```
    ///
    /// # See Also
    /// * [`from_complex`](Self::from_complex) - Just get the rotation, discard magnitude
    #[inline]
    pub fn from_complex_and_get(q: Complex<T>) -> (Self, T) {
        let norm = (q.im.clone() * q.im.clone() + q.re.clone() * q.re.clone()).simd_sqrt();
        (Self::new_unchecked(q / norm.clone()), norm)
    }

    /// Creates a unit complex number from a 2D rotation matrix.
    ///
    /// This extracts the rotation represented by a `Rotation2` matrix and converts it into
    /// the equivalent unit complex number representation. Both represent the same rotation,
    /// just in different forms - the matrix form uses 4 numbers while unit complex uses only 2.
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// # use nalgebra::{Rotation2, UnitComplex};
    /// let matrix = Rotation2::new(1.7);
    /// let complex = UnitComplex::from_rotation_matrix(&matrix);
    ///
    /// assert_eq!(complex, UnitComplex::new(1.7));
    /// ```
    ///
    /// Converting between representations:
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{Rotation2, UnitComplex, Vector2};
    /// # use std::f32;
    /// let angle = f32::consts::FRAC_PI_3;
    /// let matrix = Rotation2::new(angle);
    /// let complex = UnitComplex::from_rotation_matrix(&matrix);
    ///
    /// // Both produce the same rotation
    /// let vec = Vector2::new(1.0, 2.0);
    /// assert_relative_eq!(matrix * vec, complex * vec, epsilon = 1e-6);
    /// ```
    ///
    /// Practical example - converting from a library that uses matrices:
    /// ```
    /// # use nalgebra::{Rotation2, UnitComplex};
    /// fn external_lib_rotation() -> Rotation2<f32> {
    ///     Rotation2::new(0.5)
    /// }
    ///
    /// // Convert to more compact unit complex representation
    /// let matrix_rot = external_lib_rotation();
    /// let compact_rot = UnitComplex::from_rotation_matrix(&matrix_rot);
    /// ```
    ///
    /// # See Also
    /// * [`to_rotation_matrix`](Self::to_rotation_matrix) - Convert back to matrix form
    /// * [`from_matrix`](Self::from_matrix) - Extract rotation from a general 2×2 matrix
    // TODO: add UnitComplex::from(...) instead?
    #[inline]
    pub fn from_rotation_matrix(rotmat: &Rotation2<T>) -> Self {
        Self::new_unchecked(Complex::new(rotmat[(0, 0)].clone(), rotmat[(1, 0)].clone()))
    }

    /// Creates a rotation from a basis assumed to be orthonormal.
    ///
    /// This constructs a unit complex number from two orthonormal basis vectors (the columns
    /// of a rotation matrix). For a valid rotation, the input must be an orthonormal basis:
    /// each vector must have length 1, and they must be perpendicular to each other.
    ///
    /// **Warning**: These constraints are NOT checked! Invalid input will produce incorrect rotations.
    /// Use this only when you're certain the input is valid, otherwise use [`from_matrix`](Self::from_matrix).
    ///
    /// # Examples
    ///
    /// Basic usage with a valid orthonormal basis:
    /// ```
    /// # use nalgebra::{UnitComplex, Vector2};
    /// # use std::f32;
    /// // 90-degree rotation basis vectors
    /// let basis = [
    ///     Vector2::new(0.0, 1.0),  // rotated x-axis
    ///     Vector2::new(-1.0, 0.0), // rotated y-axis
    /// ];
    ///
    /// let rotation = UnitComplex::from_basis_unchecked(&basis);
    /// assert!((rotation.angle() - f32::consts::FRAC_PI_2).abs() < 1e-6);
    /// ```
    ///
    /// # See Also
    /// * [`from_matrix`](Self::from_matrix) - Safe extraction from any 2×2 matrix
    /// * [`from_rotation_matrix`](Self::from_rotation_matrix) - From a validated rotation matrix
    pub fn from_basis_unchecked(basis: &[Vector2<T>; 2]) -> Self {
        let mat = Matrix2::from_columns(&basis[..]);
        let rot = Rotation2::from_matrix_unchecked(mat);
        Self::from_rotation_matrix(&rot)
    }

    /// Extracts the rotation part from a general 2×2 matrix.
    ///
    /// This method handles matrices that aren't pure rotations (e.g., matrices with scaling,
    /// shearing, or numerical errors). It iteratively extracts the closest pure rotation from
    /// the input matrix using the algorithm "A Robust Method to Extract the Rotational Part of
    /// Deformations" by Müller et al.
    ///
    /// For pure rotation matrices, use [`from_rotation_matrix`](Self::from_rotation_matrix) instead.
    /// For more control over convergence, use [`from_matrix_eps`](Self::from_matrix_eps).
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// # use nalgebra::{UnitComplex, Matrix2};
    /// let mat = Matrix2::new(0.0, -1.0,
    ///                        1.0,  0.0);
    /// let rotation = UnitComplex::from_matrix(&mat);
    ///
    /// // Extracted a 90-degree rotation
    /// assert!((rotation.angle() - std::f64::consts::FRAC_PI_2).abs() < 1e-6);
    /// ```
    ///
    /// Extracting rotation from a scaled matrix:
    /// ```
    /// # use nalgebra::{UnitComplex, Matrix2};
    /// # use std::f32;
    /// // A rotation scaled by 2
    /// let scale = 2.0;
    /// let angle = f32::consts::FRAC_PI_4;
    /// let cos = angle.cos() * scale;
    /// let sin = angle.sin() * scale;
    /// let mat = Matrix2::new(cos, -sin,
    ///                        sin,  cos);
    ///
    /// // Extract just the rotation part (ignoring scale)
    /// let rotation = UnitComplex::from_matrix(&mat);
    /// assert!((rotation.angle() - angle).abs() < 1e-6);
    /// ```
    ///
    /// # See Also
    /// * [`from_rotation_matrix`](Self::from_rotation_matrix) - For pure rotation matrices
    /// * [`from_matrix_eps`](Self::from_matrix_eps) - With custom convergence parameters
    pub fn from_matrix(m: &Matrix2<T>) -> Self
    where
        T: RealField,
    {
        Rotation2::from_matrix(m).into()
    }

    /// Extracts the rotation part from a general 2×2 matrix with custom convergence parameters.
    ///
    /// This is an advanced version of [`from_matrix`](Self::from_matrix) that gives you control
    /// over the iterative algorithm's convergence behavior and initial guess. It implements
    /// "A Robust Method to Extract the Rotational Part of Deformations" by Müller et al.
    ///
    /// # Parameters
    ///
    /// * `m`: The matrix from which to extract the rotation
    /// * `eps`: Angular tolerance for convergence (in radians). Smaller values = more precision
    /// * `max_iter`: Maximum iterations before giving up. Set to 0 to loop until convergence
    /// * `guess`: Initial guess for the rotation. Convergence is faster with a good guess.
    ///            Use `UnitComplex::identity()` if you have no better guess
    ///
    /// # Examples
    ///
    /// Basic usage with custom tolerance:
    /// ```
    /// # use nalgebra::{UnitComplex, Matrix2};
    /// let mat = Matrix2::new(2.0, -2.0,  // Scaled rotation
    ///                        2.0,  2.0);
    /// let guess = UnitComplex::identity();
    /// let rotation = UnitComplex::from_matrix_eps(&mat, 1e-6, 50, guess);
    /// ```
    ///
    /// Using a good initial guess for faster convergence:
    /// ```
    /// # use nalgebra::{UnitComplex, Matrix2};
    /// # use std::f64;
    /// let mat = Matrix2::new(0.0, -1.0,
    ///                        1.0,  0.0);
    ///
    /// // We know it's close to 90 degrees
    /// let good_guess = UnitComplex::new(f64::consts::FRAC_PI_2);
    /// let rotation = UnitComplex::from_matrix_eps(&mat, 1e-8, 10, good_guess);
    /// ```
    ///
    /// # See Also
    /// * [`from_matrix`](Self::from_matrix) - Simpler version with default parameters
    pub fn from_matrix_eps(m: &Matrix2<T>, eps: T, max_iter: usize, guess: Self) -> Self
    where
        T: RealField,
    {
        let guess = Rotation2::from(guess);
        Rotation2::from_matrix_eps(m, eps, max_iter, guess).into()
    }

    /// Returns the rotation needed to rotate from `self` to `other`.
    ///
    /// This computes the "delta rotation" that, when applied to `self`, produces `other`.
    /// In other words: `self.rotation_to(other) * self == other`.
    ///
    /// This is useful for finding how much more rotation is needed to reach a target orientation.
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::UnitComplex;
    /// let current = UnitComplex::new(0.1);
    /// let target = UnitComplex::new(1.7);
    /// let delta = current.rotation_to(&target);
    ///
    /// // Applying delta to current gives target
    /// assert_relative_eq!(delta * current, target);
    ///
    /// // Inverse delta goes from target back to current
    /// assert_relative_eq!(delta.inverse() * target, current);
    /// ```
    ///
    /// The delta rotation is just the difference in angles:
    /// ```
    /// # use nalgebra::UnitComplex;
    /// let rot1 = UnitComplex::new(0.5);
    /// let rot2 = UnitComplex::new(2.0);
    /// let delta = rot1.rotation_to(&rot2);
    ///
    /// // Delta angle is 1.5 radians
    /// assert!((delta.angle() - 1.5).abs() < 1e-6);
    /// ```
    ///
    /// Practical example - rotating toward a target:
    /// ```
    /// # use nalgebra::UnitComplex;
    /// # use std::f32;
    /// let player_rotation = UnitComplex::new(0.0);
    /// let target_rotation = UnitComplex::new(f32::consts::FRAC_PI_2);
    ///
    /// // How much rotation is needed?
    /// let needed_rotation = player_rotation.rotation_to(&target_rotation);
    ///
    /// // Rotate partway (20% of the needed rotation)
    /// let partial_rotation = UnitComplex::new(needed_rotation.angle() * 0.2);
    /// let new_player_rotation = partial_rotation * player_rotation;
    /// ```
    ///
    /// # See Also
    /// * [`angle_to`](Self::angle_to) - Just get the angle difference as a scalar
    #[inline]
    #[must_use]
    pub fn rotation_to(&self, other: &Self) -> Self {
        other / self
    }

    /// Raises this unit complex number to a floating-point power.
    ///
    /// This scales the rotation angle by the given power. For example, raising a rotation
    /// to the power of 2.0 doubles its angle, and raising to 0.5 gives the "square root"
    /// rotation (half the angle). The result is a rotation with angle = `self.angle() × n`.
    ///
    /// This is useful for scaling rotations and computing fractional rotations.
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::UnitComplex;
    /// let rot = UnitComplex::new(0.78);
    /// let doubled = rot.powf(2.0);
    ///
    /// // Angle is doubled
    /// assert_relative_eq!(doubled.angle(), 2.0 * 0.78);
    /// ```
    ///
    /// Computing a "half rotation":
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::UnitComplex;
    /// # use std::f32;
    /// let full_rotation = UnitComplex::new(f32::consts::PI);
    /// let half_rotation = full_rotation.powf(0.5);
    ///
    /// // Half of 180 degrees is 90 degrees
    /// assert_relative_eq!(half_rotation.angle(), f32::consts::FRAC_PI_2);
    /// ```
    ///
    /// Applying a rotation multiple times:
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::UnitComplex;
    /// let rotation = UnitComplex::new(0.1);
    ///
    /// // Apply rotation 3 times manually
    /// let manual = rotation * rotation * rotation;
    ///
    /// // Same as raising to power 3
    /// let powered = rotation.powf(3.0);
    ///
    /// assert_relative_eq!(manual, powered);
    /// ```
    ///
    /// Practical example - fractional rotation for animation:
    /// ```
    /// # use nalgebra::UnitComplex;
    /// # use std::f32;
    /// let target_rotation = UnitComplex::new(f32::consts::PI);
    ///
    /// // Create 10 animation steps
    /// for step in 0..=10 {
    ///     let t = (step as f32) / 10.0;
    ///     let current_rotation = target_rotation.powf(t);
    ///     // Draw object at current_rotation
    /// }
    /// ```
    ///
    /// # See Also
    /// * [`slerp`](Self::slerp) - Spherical linear interpolation between two rotations
    #[inline]
    #[must_use]
    pub fn powf(&self, n: T) -> Self {
        Self::from_angle(self.angle() * n)
    }
}

/// # Construction from two vectors
impl<T: SimdRealField> UnitComplex<T>
where
    T::Element: SimdRealField,
{
    /// Creates a rotation that aligns vector `a` with vector `b`.
    ///
    /// This computes the rotation needed to rotate vector `a` so that it points in the same
    /// direction as vector `b`. The vectors don't need to be normalized - their magnitudes
    /// are ignored, only their directions matter.
    ///
    /// This is extremely useful for "look at" functionality, aiming, and alignment tasks.
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{Vector2, UnitComplex};
    /// let a = Vector2::new(1.0, 2.0);
    /// let b = Vector2::new(2.0, 1.0);
    /// let rotation = UnitComplex::rotation_between(&a, &b);
    ///
    /// // Rotating a aligns it with b
    /// assert_relative_eq!(rotation * a, b);
    /// // Inverse rotation does the opposite
    /// assert_relative_eq!(rotation.inverse() * b, a);
    /// ```
    ///
    /// Vector magnitudes don't matter:
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{Vector2, UnitComplex};
    /// let a = Vector2::new(1.0, 0.0);
    /// let b = Vector2::new(0.0, 5.0);  // Different magnitude, same direction as (0, 1)
    ///
    /// let rot = UnitComplex::rotation_between(&a, &b);
    ///
    /// // This is a 90-degree rotation
    /// assert_relative_eq!(rot.angle(), std::f32::consts::FRAC_PI_2, epsilon = 1e-6);
    /// ```
    ///
    /// Practical example - point object toward target:
    /// ```
    /// # use nalgebra::{Vector2, UnitComplex};
    /// // Player is facing right
    /// let player_forward = Vector2::new(1.0, 0.0);
    ///
    /// // Enemy is in this direction
    /// let to_enemy = Vector2::new(3.0, 4.0);
    ///
    /// // Calculate rotation to face enemy
    /// let look_rotation = UnitComplex::rotation_between(&player_forward, &to_enemy);
    ///
    /// // Apply to player
    /// let new_forward = look_rotation * player_forward;
    /// ```
    ///
    /// Practical example - align projectile velocity:
    /// ```
    /// # use nalgebra::{Vector2, UnitComplex};
    /// // Projectile currently moving right
    /// let current_velocity = Vector2::new(10.0, 0.0);
    ///
    /// // We want it to move toward this target
    /// let target_direction = Vector2::new(1.0, 1.0);
    ///
    /// // Rotation needed to align the velocity
    /// let rotation = UnitComplex::rotation_between(&current_velocity, &target_direction);
    /// let new_velocity = rotation * current_velocity;
    /// ```
    ///
    /// # See Also
    /// * [`scaled_rotation_between`](Self::scaled_rotation_between) - With fractional rotation amount
    /// * [`rotation_between_axis`](Self::rotation_between_axis) - For unit vectors
    #[inline]
    pub fn rotation_between<SB, SC>(a: &Vector<T, U2, SB>, b: &Vector<T, U2, SC>) -> Self
    where
        T: RealField,
        SB: Storage<T, U2>,
        SC: Storage<T, U2>,
    {
        Self::scaled_rotation_between(a, b, T::one())
    }

    /// Creates a partial rotation that aligns vector `a` toward vector `b`.
    ///
    /// This is like [`rotation_between`](Self::rotation_between), but applies only a fraction
    /// of the full rotation. The parameter `s` controls how much rotation to apply:
    /// - `s = 0.0` gives no rotation (identity)
    /// - `s = 1.0` gives the full rotation from `a` to `b`
    /// - `s = 0.5` gives half the rotation (halfway between `a` and `b`)
    ///
    /// This is perfect for gradual rotation, smooth turning, and animation.
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{Vector2, UnitComplex};
    /// let a = Vector2::new(1.0, 2.0);
    /// let b = Vector2::new(2.0, 1.0);
    ///
    /// // Apply 1/5th of the rotation
    /// let partial = UnitComplex::scaled_rotation_between(&a, &b, 0.2);
    ///
    /// // Applying it 5 times fully aligns a with b
    /// let result = partial * partial * partial * partial * partial * a;
    /// assert_relative_eq!(result, b, epsilon = 1.0e-6);
    /// ```
    ///
    /// Half rotation example:
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{Vector2, UnitComplex};
    /// let a = Vector2::new(1.0, 2.0);
    /// let b = Vector2::new(2.0, 1.0);
    ///
    /// // Halfway rotation
    /// let half_rot = UnitComplex::scaled_rotation_between(&a, &b, 0.5);
    ///
    /// // Applying it twice gives full rotation
    /// assert_relative_eq!(half_rot * half_rot * a, b, epsilon = 1.0e-6);
    /// ```
    ///
    /// Practical example - gradually rotate toward target:
    /// ```
    /// # use nalgebra::{Vector2, UnitComplex};
    /// let current_direction = Vector2::new(1.0, 0.0);
    /// let target_direction = Vector2::new(0.0, 1.0);
    ///
    /// // Rotate 10% toward target each frame
    /// let rotation_speed = 0.1;
    /// let this_frame_rotation = UnitComplex::scaled_rotation_between(
    ///     &current_direction,
    ///     &target_direction,
    ///     rotation_speed
    /// );
    ///
    /// let new_direction = this_frame_rotation * current_direction;
    /// ```
    ///
    /// Practical example - smooth "look at" over time:
    /// ```
    /// # use nalgebra::{Vector2, UnitComplex};
    /// struct Player {
    ///     facing: Vector2<f32>,
    /// }
    ///
    /// impl Player {
    ///     fn update(&mut self, enemy_position: Vector2<f32>, delta_time: f32) {
    ///         // Turn toward enemy at 45 degrees per second
    ///         let turn_speed = std::f32::consts::FRAC_PI_4 * delta_time / self.facing.angle(&enemy_position);
    ///         let rotation = UnitComplex::scaled_rotation_between(
    ///             &self.facing,
    ///             &enemy_position,
    ///             turn_speed.min(1.0)
    ///         );
    ///         self.facing = rotation * self.facing;
    ///     }
    /// }
    /// ```
    ///
    /// # See Also
    /// * [`rotation_between`](Self::rotation_between) - Full rotation (equivalent to `s = 1.0`)
    /// * [`slerp`](Self::slerp) - Interpolate between two existing rotations
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
        // TODO: code duplication with Rotation.
        match (
            Unit::try_new(a.clone_owned(), T::zero()),
            Unit::try_new(b.clone_owned(), T::zero()),
        ) {
            (Some(na), Some(nb)) => Self::scaled_rotation_between_axis(&na, &nb, s),
            _ => Self::identity(),
        }
    }

    /// Creates a rotation that aligns unit vector `a` with unit vector `b`.
    ///
    /// This is a specialized version of [`rotation_between`](Self::rotation_between) for unit
    /// vectors (vectors with length 1). Since the inputs are already normalized, this can be
    /// slightly more efficient.
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{Unit, Vector2, UnitComplex};
    /// let a = Unit::new_normalize(Vector2::new(1.0, 2.0));
    /// let b = Unit::new_normalize(Vector2::new(2.0, 1.0));
    /// let rotation = UnitComplex::rotation_between_axis(&a, &b);
    ///
    /// assert_relative_eq!(rotation * a, b);
    /// assert_relative_eq!(rotation.inverse() * b, a);
    /// ```
    ///
    /// Using cardinal directions:
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{Vector2, UnitComplex};
    /// # use std::f32;
    /// let right = Vector2::x_axis();
    /// let up = Vector2::y_axis();
    ///
    /// let rotation = UnitComplex::rotation_between_axis(&right, &up);
    /// // 90-degree rotation
    /// assert_relative_eq!(rotation.angle(), f32::consts::FRAC_PI_2);
    /// ```
    ///
    /// # See Also
    /// * [`rotation_between`](Self::rotation_between) - For non-unit vectors
    /// * [`scaled_rotation_between_axis`](Self::scaled_rotation_between_axis) - Partial rotation
    #[inline]
    pub fn rotation_between_axis<SB, SC>(
        a: &Unit<Vector<T, U2, SB>>,
        b: &Unit<Vector<T, U2, SC>>,
    ) -> Self
    where
        SB: Storage<T, U2>,
        SC: Storage<T, U2>,
    {
        Self::scaled_rotation_between_axis(a, b, T::one())
    }

    /// Creates a partial rotation that aligns unit vector `a` toward unit vector `b`.
    ///
    /// This is a specialized version of [`scaled_rotation_between`](Self::scaled_rotation_between)
    /// for unit vectors. The parameter `s` controls how much of the rotation to apply (0.0 to 1.0).
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{Unit, Vector2, UnitComplex};
    /// let a = Unit::new_normalize(Vector2::new(1.0, 2.0));
    /// let b = Unit::new_normalize(Vector2::new(2.0, 1.0));
    ///
    /// // Apply 20% of the rotation
    /// let partial = UnitComplex::scaled_rotation_between_axis(&a, &b, 0.2);
    ///
    /// // Five times gives the full rotation
    /// let result = partial * partial * partial * partial * partial * a;
    /// assert_relative_eq!(result, b, epsilon = 1.0e-6);
    /// ```
    ///
    /// # See Also
    /// * [`scaled_rotation_between`](Self::scaled_rotation_between) - For non-unit vectors
    /// * [`rotation_between_axis`](Self::rotation_between_axis) - Full rotation (s = 1.0)
    #[inline]
    pub fn scaled_rotation_between_axis<SB, SC>(
        na: &Unit<Vector<T, U2, SB>>,
        nb: &Unit<Vector<T, U2, SC>>,
        s: T,
    ) -> Self
    where
        SB: Storage<T, U2>,
        SC: Storage<T, U2>,
    {
        let sang = na.perp(nb);
        let cang = na.dot(nb);

        Self::from_angle(sang.simd_atan2(cang) * s)
    }
}

impl<T: SimdRealField> One for UnitComplex<T>
where
    T::Element: SimdRealField,
{
    #[inline]
    fn one() -> Self {
        Self::identity()
    }
}

#[cfg(feature = "rand")]
impl<T: SimdRealField> Distribution<UnitComplex<T>> for StandardUniform
where
    T::Element: SimdRealField,
    rand_distr::UnitCircle: Distribution<[T; 2]>,
{
    /// Generate a uniformly distributed random `UnitComplex`.
    #[inline]
    fn sample<'a, R: Rng + ?Sized>(&self, rng: &mut R) -> UnitComplex<T> {
        let x = rng.sample(rand_distr::UnitCircle);
        UnitComplex::new_unchecked(Complex::new(x[0].clone(), x[1].clone()))
    }
}

#[cfg(feature = "arbitrary")]
impl<T: SimdRealField + Arbitrary> Arbitrary for UnitComplex<T>
where
    T::Element: SimdRealField,
{
    #[inline]
    fn arbitrary(g: &mut Gen) -> Self {
        UnitComplex::from_angle(T::arbitrary(g))
    }
}
