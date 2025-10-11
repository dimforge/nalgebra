//! Construction of Givens rotations.
//!
//! A Givens rotation is a 2×2 rotation matrix that can be used to zero out specific
//! elements in vectors and matrices. They are fundamental building blocks in numerical
//! linear algebra, particularly in QR decomposition, eigenvalue algorithms, and when
//! working with sparse matrices.
//!
//! The basic form of a Givens rotation is:
//! ```text
//! G = [ c  -s* ]
//!     [ s   c  ]
//! ```
//! where c (cosine) is real, s (sine) can be complex, and |c|² + |s|² = 1.

use num::{One, Zero};
use simba::scalar::ComplexField;

use crate::base::constraint::{DimEq, ShapeConstraint};
use crate::base::dimension::{Dim, U2};
use crate::base::storage::{Storage, StorageMut};
use crate::base::{Matrix, Vector};

/// A Givens rotation matrix that can zero out specific elements in vectors and matrices.
///
/// A Givens rotation is a 2×2 orthogonal (or unitary for complex numbers) rotation matrix
/// of the form:
/// ```text
/// G = [ c  -s* ]
///     [ s   c  ]
/// ```
/// where:
/// - `c` is the cosine component (always real, even for complex matrices)
/// - `s` is the sine component (can be complex)
/// - `s*` denotes the complex conjugate of `s`
/// - The normalization condition |c|² + |s|² = 1 holds
///
/// Givens rotations are used extensively in numerical linear algebra for:
/// - **QR decomposition**: Systematically zeroing out subdiagonal elements
/// - **Eigenvalue algorithms**: Used in the QR algorithm and Jacobi methods
/// - **Sparse matrices**: More efficient than Householder reflections when only specific elements need to be zeroed
/// - **Least squares problems**: Updating solutions incrementally
///
/// # Example
/// ```
/// use nalgebra::{Vector2, Matrix2};
/// use nalgebra::linalg::givens::GivensRotation;
///
/// // Create a Givens rotation that zeros out the second component of a vector
/// let v = Vector2::new(3.0, 4.0);
/// if let Some((rotation, norm)) = GivensRotation::cancel_y(&v) {
///     println!("Original vector: {}", v);
///     println!("Norm: {}", norm);
///
///     // The rotation transforms v into [norm, 0]
///     let mut result = v.clone();
///     rotation.rotate(&mut result.fixed_rows_mut::<2>(0));
///     println!("After rotation: {}", result);
///     // result is approximately [5.0, 0.0]
/// }
/// ```
#[derive(Debug, Clone, Copy)]
pub struct GivensRotation<T: ComplexField> {
    c: T::RealField,
    s: T,
}

// Matrix = UnitComplex * Matrix
impl<T: ComplexField> GivensRotation<T> {
    /// Creates the identity Givens rotation that leaves vectors unchanged.
    ///
    /// This rotation has `c = 1` and `s = 0`, corresponding to the 2×2 identity matrix:
    /// ```text
    /// I = [ 1  0 ]
    ///     [ 0  1 ]
    /// ```
    ///
    /// The identity rotation is useful as a default or initial value in algorithms,
    /// and represents "no rotation" in a sequence of transformations.
    ///
    /// # Example
    /// ```
    /// use nalgebra::linalg::givens::GivensRotation;
    /// use nalgebra::Vector2;
    ///
    /// let identity: GivensRotation<f64> = GivensRotation::identity();
    ///
    /// // The identity rotation doesn't change vectors
    /// let v = Vector2::new(3.0, 4.0);
    /// let mut result = v.clone();
    /// identity.rotate(&mut result.fixed_rows_mut::<2>(0));
    ///
    /// assert!((result - v).norm() < 1e-10);
    /// ```
    ///
    /// # See Also
    /// - [`new()`](Self::new): Create a Givens rotation from cosine and sine components
    /// - [`new_unchecked()`](Self::new_unchecked): Create without normalization
    pub fn identity() -> Self {
        Self {
            c: T::RealField::one(),
            s: T::zero(),
        }
    }

    /// Creates a Givens rotation from pre-computed cosine and sine components without validation.
    ///
    /// This constructor takes the cosine (`c`) and sine (`s`) components directly and does not
    /// verify that they satisfy the normalization condition |c|² + |s|² = 1. Use this only when
    /// you're certain the components are already correctly normalized.
    ///
    /// # Safety Considerations
    ///
    /// While this function is safe in Rust's memory safety sense, using it with invalid
    /// components (where |c|² + |s|² ≠ 1) will produce incorrect mathematical results.
    /// The rotation matrix will not be orthogonal/unitary, leading to numerical errors
    /// in subsequent operations.
    ///
    /// # When to Use
    ///
    /// Use this constructor when:
    /// - You've manually computed normalized components
    /// - You're deserializing from a trusted source
    /// - Performance is critical and you can guarantee correctness
    ///
    /// For most cases, prefer [`new()`](Self::new) which automatically normalizes the components.
    ///
    /// # Example
    /// ```
    /// use nalgebra::linalg::givens::GivensRotation;
    ///
    /// // Create a 45-degree rotation with pre-normalized components
    /// let cos_45 = std::f64::consts::FRAC_1_SQRT_2;
    /// let sin_45 = std::f64::consts::FRAC_1_SQRT_2;
    ///
    /// let rotation = GivensRotation::new_unchecked(cos_45, sin_45);
    ///
    /// // Verify normalization: c² + s² should equal 1
    /// let norm_squared = rotation.c() * rotation.c() + rotation.s() * rotation.s();
    /// assert!((norm_squared - 1.0).abs() < 1e-10);
    /// ```
    ///
    /// # See Also
    /// - [`new()`](Self::new): Safe constructor that normalizes components automatically
    /// - [`try_new()`](Self::try_new): Constructor with epsilon tolerance for normalization
    /// - [`identity()`](Self::identity): Create the identity rotation
    pub const fn new_unchecked(c: T::RealField, s: T) -> Self {
        Self { c, s }
    }

    /// Creates a Givens rotation from non-normalized cosine and sine components.
    ///
    /// This constructor takes arbitrary cosine (`c`) and sine (`s`) values and automatically
    /// normalizes them to create a valid Givens rotation. It computes the rotation and returns
    /// both the normalized rotation and the norm of the input vector (c, s).
    ///
    /// The normalization ensures that the resulting rotation matrix is orthogonal/unitary
    /// (i.e., |c'|² + |s'|² = 1, where c' and s' are the normalized components).
    ///
    /// # Returns
    ///
    /// A tuple `(rotation, norm)` where:
    /// - `rotation`: The normalized Givens rotation
    /// - `norm`: The magnitude of the original (c, s) vector, which equals √(|c|² + |s|²)
    ///
    /// If the norm is zero (both c and s are zero), returns the identity rotation with norm zero.
    ///
    /// # Example
    /// ```
    /// use nalgebra::linalg::givens::GivensRotation;
    /// use nalgebra::Vector2;
    ///
    /// // Create a rotation from non-normalized components
    /// let (rotation, norm): (GivensRotation<f64>, f64) = GivensRotation::new(3.0, 4.0);
    ///
    /// // The norm should be 5.0 (since sqrt(3² + 4²) = 5)
    /// assert!((norm - 5.0).abs() < 1e-10);
    ///
    /// // The rotation components are normalized: c = 3/5, s = 4/5
    /// assert!((rotation.c() - 0.6).abs() < 1e-10);
    /// assert!((rotation.s() - 0.8).abs() < 1e-10);
    ///
    /// // Verify the normalization condition
    /// let c = rotation.c();
    /// let s = rotation.s();
    /// assert!((c * c + s * s - 1.0).abs() < 1e-10);
    /// ```
    ///
    /// # Practical Use Case: Creating a Rotation Matrix
    /// ```
    /// use nalgebra::{Matrix2, linalg::givens::GivensRotation};
    ///
    /// // Create a rotation from components representing an angle
    /// // For example, c=0.6 and s=0.8 represents a specific rotation
    /// let (rotation, norm): (GivensRotation<f64>, f64) = GivensRotation::new(3.0, 4.0);
    ///
    /// // The norm is the magnitude of the input vector
    /// assert!((norm - 5.0).abs() < 1e-10);
    ///
    /// // Build the explicit rotation matrix
    /// let c = rotation.c();
    /// let s = rotation.s();
    /// let rot_matrix = Matrix2::new(
    ///     c, -s,
    ///     s,  c
    /// );
    ///
    /// // This matrix is orthogonal
    /// let product = rot_matrix.transpose() * rot_matrix;
    /// let identity = Matrix2::identity();
    /// assert!((product - identity).norm() < 1e-10);
    /// ```
    ///
    /// # See Also
    /// - [`try_new()`](Self::try_new): Version with configurable epsilon tolerance
    /// - [`new_unchecked()`](Self::new_unchecked): Skip normalization for pre-normalized values
    /// - [`cancel_y()`](Self::cancel_y): Create rotation to zero out the y-component of a vector
    /// - [`cancel_x()`](Self::cancel_x): Create rotation to zero out the x-component of a vector
    pub fn new(c: T, s: T) -> (Self, T) {
        Self::try_new(c, s, T::RealField::zero())
            .unwrap_or_else(|| (GivensRotation::identity(), T::zero()))
    }

    /// Attempts to create a Givens rotation from non-normalized components with epsilon tolerance.
    ///
    /// This is a more robust version of [`new()`](Self::new) that returns `None` if the
    /// magnitude of the input vector (c, s) is smaller than the specified epsilon tolerance.
    /// This is useful for avoiding numerical instability when working with very small values.
    ///
    /// # Parameters
    ///
    /// - `c`: The cosine component (can be complex)
    /// - `s`: The sine component (can be complex)
    /// - `eps`: Tolerance threshold for the norm. If √(|c|² + |s|²) ≤ eps, returns `None`
    ///
    /// # Returns
    ///
    /// - `Some((rotation, norm))`: If the norm exceeds epsilon, returns the normalized rotation
    ///   and the original norm
    /// - `None`: If the norm is less than or equal to epsilon, indicating the input is too small
    ///   to normalize reliably
    ///
    /// # When to Use
    ///
    /// Use this constructor when:
    /// - Working with potentially very small values that could cause numerical instability
    /// - You need to handle the degenerate case (near-zero inputs) explicitly
    /// - Implementing robust numerical algorithms that need to check for valid rotations
    ///
    /// # Example
    /// ```
    /// use nalgebra::linalg::givens::GivensRotation;
    ///
    /// // Normal case: components are large enough
    /// let result: Option<(GivensRotation<f64>, f64)> = GivensRotation::try_new(3.0, 4.0, 1e-10);
    /// assert!(result.is_some());
    /// let (rotation, norm) = result.unwrap();
    /// assert!((norm - 5.0).abs() < 1e-10);
    ///
    /// // Degenerate case: both components are too small
    /// let result: Option<(GivensRotation<f64>, f64)> = GivensRotation::try_new(1e-12, 1e-12, 1e-10);
    /// assert!(result.is_none());
    ///
    /// // One large component is enough
    /// let result: Option<(GivensRotation<f64>, f64)> = GivensRotation::try_new(1e-12, 3.0, 1e-10);
    /// assert!(result.is_some());
    /// ```
    ///
    /// # Practical Use Case: Robust Matrix Algorithms
    /// ```
    /// use nalgebra::{Matrix3, linalg::givens::GivensRotation};
    ///
    /// // In iterative algorithms, we might encounter very small values
    /// let matrix = Matrix3::new(
    ///     1.0,   0.0,     0.0,
    ///     0.0,   1e-15,   2.0,
    ///     0.0,   0.0,     3.0
    /// );
    ///
    /// // Try to create a rotation for elements (1,1) and (2,1)
    /// let eps = 1e-10;
    /// match GivensRotation::try_new(matrix[(1, 1)], matrix[(2, 1)], eps) {
    ///     Some((rotation, _norm)) => {
    ///         println!("Created valid rotation");
    ///         // Apply rotation...
    ///     }
    ///     None => {
    ///         println!("Value too small, skipping rotation");
    ///         // Skip this rotation or use identity
    ///     }
    /// }
    /// ```
    ///
    /// # See Also
    /// - [`new()`](Self::new): Simpler version that always returns a rotation (uses eps = 0)
    /// - [`new_unchecked()`](Self::new_unchecked): Create without normalization
    /// - [`identity()`](Self::identity): The rotation returned when inputs are too small
    pub fn try_new(c: T, s: T, eps: T::RealField) -> Option<(Self, T)> {
        let (mod0, sign0) = c.to_exp();
        let denom = (mod0.clone() * mod0.clone() + s.clone().modulus_squared()).sqrt();

        if denom > eps {
            let norm = sign0.scale(denom.clone());
            let c = mod0 / denom;
            let s = s / norm.clone();
            Some((Self { c, s }, norm))
        } else {
            None
        }
    }

    /// Creates a Givens rotation that zeros out the second component (y) of a 2D vector.
    ///
    /// Given a vector `v = [x, y]ᵀ`, this function computes a rotation `R` such that:
    /// ```text
    /// R * v = [‖v‖, 0]ᵀ
    /// ```
    /// where ‖v‖ is the Euclidean norm of the vector. This effectively rotates the vector
    /// to align it with the x-axis.
    ///
    /// This is one of the fundamental operations in numerical linear algebra, used extensively in:
    /// - **QR decomposition**: Systematically zeroing subdiagonal elements
    /// - **Least squares**: Transforming overdetermined systems to triangular form
    /// - **Eigenvalue algorithms**: Reducing matrices to tridiagonal or Hessenberg form
    ///
    /// # Parameters
    ///
    /// - `v`: A 2D vector whose second component should be zeroed
    ///
    /// # Returns
    ///
    /// - `Some((rotation, norm))`: The rotation and the norm of the original vector
    /// - `None`: If `v[1]` (the y-component) is already zero, no rotation is needed
    ///
    /// # Example
    /// ```
    /// use nalgebra::{Vector2, linalg::givens::GivensRotation};
    ///
    /// // Create a vector [3, 4]
    /// let v = Vector2::new(3.0_f64, 4.0);
    ///
    /// // Compute rotation to zero out the y-component
    /// if let Some((rotation, norm)) = GivensRotation::cancel_y(&v) {
    ///     // The norm should be 5 (sqrt(3² + 4²))
    ///     assert!((norm - 5.0).abs() < 1e-10);
    ///
    ///     // Apply the rotation
    ///     let mut result = v.clone();
    ///     rotation.rotate(&mut result.fixed_rows_mut::<2>(0));
    ///
    ///     // Result should be approximately [5, 0]
    ///     assert!((result[0] - 5.0).abs() < 1e-10);
    ///     assert!(result[1].abs() < 1e-10);
    /// }
    /// ```
    ///
    /// # Practical Use Case: QR Decomposition Step
    /// ```
    /// use nalgebra::{Matrix3x2, Vector2, linalg::givens::GivensRotation};
    ///
    /// // Consider the first column of a matrix in QR decomposition
    /// let mut matrix = Matrix3x2::new(
    ///     3.0_f64, 1.0,
    ///     4.0, 2.0,
    ///     0.0, 3.0
    /// );
    ///
    /// // Zero out element (1,0) using the first two rows of column 0
    /// let col = Vector2::new(matrix[(0, 0)], matrix[(1, 0)]);
    ///
    /// if let Some((rotation, _norm)) = GivensRotation::cancel_y(&col) {
    ///     // Apply to both columns of the first two rows
    ///     rotation.rotate(&mut matrix.fixed_rows_mut::<2>(0));
    ///
    ///     // Element (1,0) should now be zero
    ///     assert!(matrix[(1, 0)].abs() < 1e-10);
    ///     println!("Partially triangularized matrix:\n{}", matrix);
    /// }
    /// ```
    ///
    /// # See Also
    /// - [`cancel_x()`](Self::cancel_x): Zero out the first component (x) instead
    /// - [`rotate()`](Self::rotate): Apply the rotation to a matrix
    /// - [`new()`](Self::new): Create a rotation from cosine and sine components
    pub fn cancel_y<S: Storage<T, U2>>(v: &Vector<T, U2, S>) -> Option<(Self, T)> {
        if !v[1].is_zero() {
            let (mod0, sign0) = v[0].clone().to_exp();
            let denom = (mod0.clone() * mod0.clone() + v[1].clone().modulus_squared()).sqrt();
            let c = mod0 / denom.clone();
            let s = -v[1].clone() / sign0.clone().scale(denom.clone());
            let r = sign0.scale(denom);
            Some((Self { c, s }, r))
        } else {
            None
        }
    }

    /// Creates a Givens rotation that zeros out the first component (x) of a 2D vector.
    ///
    /// Given a vector `v = [x, y]ᵀ`, this function computes a rotation `R` such that:
    /// ```text
    /// R * v = [0, ‖v‖]ᵀ
    /// ```
    /// where ‖v‖ is the Euclidean norm of the vector. This effectively rotates the vector
    /// to align it with the y-axis.
    ///
    /// While [`cancel_y()`](Self::cancel_y) is more commonly used in standard algorithms,
    /// `cancel_x()` is useful in specialized scenarios where you need to preserve the second
    /// component or work with transposed matrices.
    ///
    /// # Parameters
    ///
    /// - `v`: A 2D vector whose first component should be zeroed
    ///
    /// # Returns
    ///
    /// - `Some((rotation, norm))`: The rotation and the norm of the original vector
    /// - `None`: If `v[0]` (the x-component) is already zero, no rotation is needed
    ///
    /// # Example
    /// ```
    /// use nalgebra::{Vector2, linalg::givens::GivensRotation};
    ///
    /// // Create a vector [3, 4]
    /// let v = Vector2::new(3.0_f64, 4.0);
    ///
    /// // Compute rotation to zero out the x-component
    /// if let Some((rotation, norm)) = GivensRotation::cancel_x(&v) {
    ///     // The norm should be 5 (sqrt(3² + 4²))
    ///     assert!((norm - 5.0).abs() < 1e-10);
    ///
    ///     // Apply the rotation
    ///     let mut result = v.clone();
    ///     rotation.rotate(&mut result.fixed_rows_mut::<2>(0));
    ///
    ///     // Result should be approximately [0, 5]
    ///     assert!(result[0].abs() < 1e-10);
    ///     assert!((result[1] - 5.0).abs() < 1e-10);
    /// }
    /// ```
    ///
    /// # Comparison with cancel_y
    /// ```
    /// use nalgebra::{Vector2, linalg::givens::GivensRotation};
    ///
    /// let v = Vector2::new(3.0, 4.0);
    ///
    /// // cancel_y zeros the second component: v → [5, 0]
    /// if let Some((rot_y, _)) = GivensRotation::cancel_y(&v) {
    ///     let mut result_y = v.clone();
    ///     rot_y.rotate(&mut result_y.fixed_rows_mut::<2>(0));
    ///     println!("cancel_y: [{:.1}, {:.1}]", result_y[0], result_y[1]);
    /// }
    ///
    /// // cancel_x zeros the first component: v → [0, 5]
    /// if let Some((rot_x, _)) = GivensRotation::cancel_x(&v) {
    ///     let mut result_x = v.clone();
    ///     rot_x.rotate(&mut result_x.fixed_rows_mut::<2>(0));
    ///     println!("cancel_x: [{:.1}, {:.1}]", result_x[0], result_x[1]);
    /// }
    /// ```
    ///
    /// # Practical Use Case: Column-wise Operations
    /// ```
    /// use nalgebra::{Matrix2, Vector2, linalg::givens::GivensRotation};
    ///
    /// // Working with a matrix where we want to zero the first row element
    /// let mut matrix = Matrix2::new(
    ///     3.0_f64, 1.0,
    ///     4.0, 2.0
    /// );
    ///
    /// // Get the first column
    /// let col = Vector2::new(matrix[(0, 0)], matrix[(1, 0)]);
    ///
    /// // Zero out the top element of the column
    /// if let Some((rotation, _norm)) = GivensRotation::cancel_x(&col) {
    ///     rotation.rotate(&mut matrix.fixed_rows_mut::<2>(0));
    ///
    ///     // Element (0,0) should now be zero
    ///     assert!(matrix[(0, 0)].abs() < 1e-10);
    ///     println!("Modified matrix:\n{}", matrix);
    /// }
    /// ```
    ///
    /// # See Also
    /// - [`cancel_y()`](Self::cancel_y): Zero out the second component (y) instead (more common)
    /// - [`rotate()`](Self::rotate): Apply the rotation to a matrix
    /// - [`new()`](Self::new): Create a rotation from cosine and sine components
    pub fn cancel_x<S: Storage<T, U2>>(v: &Vector<T, U2, S>) -> Option<(Self, T)> {
        if !v[0].is_zero() {
            let (mod1, sign1) = v[1].clone().to_exp();
            let denom = (mod1.clone() * mod1.clone() + v[0].clone().modulus_squared()).sqrt();
            let c = mod1 / denom.clone();
            let s = (v[0].clone().conjugate() * sign1.clone()).unscale(denom.clone());
            let r = sign1.scale(denom);
            Some((Self { c, s }, r))
        } else {
            None
        }
    }

    /// Returns the cosine component of this Givens rotation.
    ///
    /// The cosine component is always a real number (even for complex-valued rotations)
    /// and represents the diagonal elements of the 2×2 rotation matrix:
    /// ```text
    /// G = [ c  -s* ]
    ///     [ s   c  ]
    /// ```
    ///
    /// Together with the sine component, it satisfies the normalization condition:
    /// |c|² + |s|² = 1
    ///
    /// # Example
    /// ```
    /// use nalgebra::linalg::givens::GivensRotation;
    ///
    /// // Create a rotation from normalized components
    /// let (rotation, _norm): (GivensRotation<f64>, f64) = GivensRotation::new(3.0, 4.0);
    ///
    /// // The cosine should be 3/5 = 0.6
    /// let c = rotation.c();
    /// assert!((c - 0.6).abs() < 1e-10);
    ///
    /// // Verify normalization with the sine
    /// let s = rotation.s();
    /// assert!((c * c + s * s - 1.0).abs() < 1e-10);
    /// ```
    ///
    /// # Accessing Individual Components
    /// ```
    /// use nalgebra::{Vector2, linalg::givens::GivensRotation};
    ///
    /// let v = Vector2::new(1.0, 1.0);
    /// if let Some((rotation, _norm)) = GivensRotation::cancel_y(&v) {
    ///     let c = rotation.c();
    ///     let s = rotation.s();
    ///
    ///     println!("Rotation matrix:");
    ///     println!("[ {:.4}  {:.4} ]", c, -s);
    ///     println!("[ {:.4}  {:.4} ]", s, c);
    /// }
    /// ```
    ///
    /// # See Also
    /// - [`s()`](Self::s): Get the sine component
    /// - [`new_unchecked()`](Self::new_unchecked): Create from c and s directly
    #[must_use]
    pub fn c(&self) -> T::RealField {
        self.c.clone()
    }

    /// Returns the sine component of this Givens rotation.
    ///
    /// The sine component can be complex (for complex-valued matrices) and represents
    /// the off-diagonal elements of the 2×2 rotation matrix:
    /// ```text
    /// G = [ c  -s* ]
    ///     [ s   c  ]
    /// ```
    /// where s* denotes the complex conjugate of s.
    ///
    /// Together with the cosine component, it satisfies the normalization condition:
    /// |c|² + |s|² = 1
    ///
    /// # Example
    /// ```
    /// use nalgebra::linalg::givens::GivensRotation;
    ///
    /// // Create a rotation from normalized components
    /// let (rotation, _norm): (GivensRotation<f64>, f64) = GivensRotation::new(3.0, 4.0);
    ///
    /// // The sine should be 4/5 = 0.8
    /// let s = rotation.s();
    /// assert!((s - 0.8).abs() < 1e-10);
    ///
    /// // Verify normalization with the cosine
    /// let c = rotation.c();
    /// assert!((c * c + s * s - 1.0).abs() < 1e-10);
    /// ```
    ///
    /// # Building the Rotation Matrix
    /// ```
    /// use nalgebra::{Matrix2, linalg::givens::GivensRotation};
    ///
    /// let (rotation, _norm) = GivensRotation::new(3.0, 4.0);
    ///
    /// let c = rotation.c();
    /// let s = rotation.s();
    ///
    /// // Construct the explicit 2×2 rotation matrix
    /// let matrix = Matrix2::new(
    ///     c, -s,
    ///     s,  c
    /// );
    ///
    /// println!("Givens rotation matrix:\n{}", matrix);
    ///
    /// // Verify it's orthogonal: G^T * G = I
    /// let product = matrix.transpose() * matrix;
    /// let identity = Matrix2::identity();
    /// assert!((product - identity).norm() < 1e-10);
    /// ```
    ///
    /// # Complex Numbers
    /// ```
    /// use nalgebra::{linalg::givens::GivensRotation, Complex};
    ///
    /// // For complex matrices, the sine can be complex
    /// let c = Complex::new(0.6, 0.0);
    /// let s = Complex::new(0.48, 0.64);  // |s|² = 0.48² + 0.64² = 0.64
    ///
    /// let (rotation, _norm) = GivensRotation::new(c, s);
    ///
    /// let sine = rotation.s();
    /// println!("Complex sine: {}", sine);
    /// ```
    ///
    /// # See Also
    /// - [`c()`](Self::c): Get the cosine component
    /// - [`new_unchecked()`](Self::new_unchecked): Create from c and s directly
    #[must_use]
    pub fn s(&self) -> T {
        self.s.clone()
    }

    /// Returns the inverse (transpose) of this Givens rotation.
    ///
    /// For Givens rotations, the inverse is equal to the transpose because the rotation
    /// matrix is orthogonal (or unitary for complex numbers). The inverse simply negates
    /// the sine component while keeping the cosine unchanged.
    ///
    /// Given a rotation:
    /// ```text
    /// G = [ c  -s* ]
    ///     [ s   c  ]
    /// ```
    /// Its inverse is:
    /// ```text
    /// G^(-1) = G^T = [ c   s* ]
    ///                 [-s   c  ]
    /// ```
    ///
    /// This operation is useful for:
    /// - Undoing a rotation
    /// - Right-multiplying by a Givens rotation (equivalent to left-multiplying by the inverse)
    /// - Computing the transpose of a product of rotations
    ///
    /// # Example
    /// ```
    /// use nalgebra::{Vector2, linalg::givens::GivensRotation};
    ///
    /// let v = Vector2::new(3.0_f64, 4.0);
    ///
    /// // Create a rotation
    /// if let Some((rotation, _norm)) = GivensRotation::cancel_y(&v) {
    ///     // Apply the rotation
    ///     let mut rotated = v.clone();
    ///     rotation.rotate(&mut rotated.fixed_rows_mut::<2>(0));
    ///
    ///     // Apply the inverse rotation
    ///     let inverse = rotation.inverse();
    ///     let mut restored = rotated.clone();
    ///     inverse.rotate(&mut restored.fixed_rows_mut::<2>(0));
    ///
    ///     // Should get back the original vector
    ///     assert!((restored - v).norm() < 1e-10);
    /// }
    /// ```
    ///
    /// # Verifying the Inverse Property
    /// ```
    /// use nalgebra::{Matrix2, linalg::givens::GivensRotation};
    ///
    /// let (rotation, _norm) = GivensRotation::new(3.0, 4.0);
    /// let inverse = rotation.inverse();
    ///
    /// // Build explicit matrices
    /// let c = rotation.c();
    /// let s = rotation.s();
    ///
    /// let g = Matrix2::new(c, -s, s, c);
    ///
    /// let c_inv = inverse.c();
    /// let s_inv = inverse.s();
    /// let g_inv = Matrix2::new(c_inv, -s_inv, s_inv, c_inv);
    ///
    /// // G * G^(-1) should be identity
    /// let product = g * g_inv;
    /// let identity = Matrix2::identity();
    /// assert!((product - identity).norm() < 1e-10);
    /// ```
    ///
    /// # Computational Efficiency
    /// ```
    /// use nalgebra::linalg::givens::GivensRotation;
    ///
    /// let (rotation, _norm) = GivensRotation::new(3.0, 4.0);
    ///
    /// // Computing the inverse is very cheap - just negates the sine
    /// let inverse = rotation.inverse();
    ///
    /// // Components relationship
    /// assert_eq!(inverse.c(), rotation.c());
    /// assert_eq!(inverse.s(), -rotation.s());
    /// ```
    ///
    /// # See Also
    /// - [`rotate()`](Self::rotate): Apply the rotation (left-multiplication)
    /// - [`rotate_rows()`](Self::rotate_rows): Apply as right-multiplication
    #[must_use = "This function does not mutate self."]
    pub fn inverse(&self) -> Self {
        Self {
            c: self.c.clone(),
            s: -self.s.clone(),
        }
    }

    /// Applies this Givens rotation to a matrix by left-multiplication (in-place).
    ///
    /// Performs the operation `M = G * M` where `G` is this Givens rotation and `M` is
    /// the input matrix. This modifies the matrix in-place, transforming its first two rows
    /// according to the rotation.
    ///
    /// The matrix must have exactly 2 rows. The rotation is applied to all columns
    /// simultaneously, making this operation efficient for matrices with many columns.
    ///
    /// # Mathematical Operation
    ///
    /// For each column j, the rotation transforms the first two elements:
    /// ```text
    /// [ m₀ⱼ ]    [ c  -s* ] [ m₀ⱼ ]
    /// [ m₁ⱼ ] ← [ s   c  ] [ m₁ⱼ ]
    /// ```
    ///
    /// # Parameters
    ///
    /// - `rhs`: A mutable reference to a matrix with exactly 2 rows and any number of columns
    ///
    /// # Panics
    ///
    /// Panics if the matrix does not have exactly 2 rows.
    ///
    /// # Example
    /// ```
    /// use nalgebra::{Matrix2x3, Vector2, linalg::givens::GivensRotation};
    ///
    /// // Create a matrix to rotate
    /// let mut matrix = Matrix2x3::new(
    ///     3.0_f64, 1.0, 2.0,
    ///     4.0, 0.0, 1.0
    /// );
    ///
    /// // Create a rotation that zeros the (1,0) element
    /// let col0 = Vector2::new(matrix[(0, 0)], matrix[(1, 0)]);
    /// if let Some((rotation, _norm)) = GivensRotation::cancel_y(&col0) {
    ///     // Apply to the entire matrix
    ///     rotation.rotate(&mut matrix.fixed_rows_mut::<2>(0));
    ///
    ///     // The (1,0) element should now be zero
    ///     assert!(matrix[(1, 0)].abs() < 1e-10);
    ///     println!("Rotated matrix:\n{}", matrix);
    /// }
    /// ```
    ///
    /// # Practical Use Case: QR Decomposition
    /// ```
    /// use nalgebra::{Matrix3, Vector2, linalg::givens::GivensRotation};
    ///
    /// // Start with a 3×3 matrix
    /// let mut matrix = Matrix3::new(
    ///     4.0_f64, 1.0, 2.0,
    ///     3.0, 0.0, 1.0,
    ///     2.0, 1.0, 3.0
    /// );
    ///
    /// // Zero out element (1,0) - rotate rows 0 and 1
    /// let v = Vector2::new(matrix[(0, 0)], matrix[(1, 0)]);
    /// if let Some((g1, _)) = GivensRotation::cancel_y(&v) {
    ///     g1.rotate(&mut matrix.fixed_rows_mut::<2>(0));
    ///     assert!(matrix[(1, 0)].abs() < 1e-10);
    /// }
    ///
    /// // Zero out element (2,0) - rotate rows 0 and 2
    /// let v = Vector2::new(matrix[(0, 0)], matrix[(2, 0)]);
    /// if let Some((g2, _)) = GivensRotation::cancel_y(&v) {
    ///     // Need to work with rows 0 and 2
    ///     // For real numbers, conjugate = identity
    ///     for j in 0..3 {
    ///         let a = matrix[(0, j)];
    ///         let b = matrix[(2, j)];
    ///         matrix[(0, j)] = g2.c() * a - g2.s() * b;
    ///         matrix[(2, j)] = g2.s() * a + g2.c() * b;
    ///     }
    ///     assert!(matrix[(2, 0)].abs() < 1e-10);
    /// }
    ///
    /// println!("Upper triangular form:\n{}", matrix);
    /// ```
    ///
    /// # Performance Note
    ///
    /// This method processes all columns of the matrix in a single pass, making it efficient
    /// for wide matrices. The in-place operation avoids memory allocation.
    ///
    /// # See Also
    /// - [`rotate_rows()`](Self::rotate_rows): Right-multiplication (M = M * G)
    /// - [`cancel_y()`](Self::cancel_y): Create rotation to zero a specific element
    /// - [`inverse()`](Self::inverse): Get the inverse rotation
    pub fn rotate<R2: Dim, C2: Dim, S2: StorageMut<T, R2, C2>>(
        &self,
        rhs: &mut Matrix<T, R2, C2, S2>,
    ) where
        ShapeConstraint: DimEq<R2, U2>,
    {
        assert_eq!(
            rhs.nrows(),
            2,
            "Unit complex rotation: the input matrix must have exactly two rows."
        );
        let s = self.s.clone();
        let c = self.c.clone();

        for j in 0..rhs.ncols() {
            unsafe {
                let a = rhs.get_unchecked((0, j)).clone();
                let b = rhs.get_unchecked((1, j)).clone();

                *rhs.get_unchecked_mut((0, j)) =
                    a.clone().scale(c.clone()) - s.clone().conjugate() * b.clone();
                *rhs.get_unchecked_mut((1, j)) = s.clone() * a + b.scale(c.clone());
            }
        }
    }

    /// Applies this Givens rotation to a matrix by right-multiplication (in-place).
    ///
    /// Performs the operation `M = M * G` where `M` is the input matrix and `G` is this
    /// Givens rotation. This modifies the matrix in-place, transforming its first two columns
    /// according to the rotation.
    ///
    /// The matrix must have exactly 2 columns. The rotation is applied to all rows
    /// simultaneously, making this operation efficient for matrices with many rows.
    ///
    /// # Mathematical Operation
    ///
    /// For each row i, the rotation transforms the first two elements:
    /// ```text
    /// [mᵢ₀ mᵢ₁] ← [mᵢ₀ mᵢ₁] [ c   s* ]
    ///                        [-s   c  ]
    /// ```
    ///
    /// This is equivalent to left-multiplying by the transpose (inverse) of the rotation.
    ///
    /// # Parameters
    ///
    /// - `lhs`: A mutable reference to a matrix with any number of rows and exactly 2 columns
    ///
    /// # Panics
    ///
    /// Panics if the matrix does not have exactly 2 columns.
    ///
    /// # Example
    /// ```
    /// use nalgebra::{Matrix3x2, Vector2, linalg::givens::GivensRotation};
    ///
    /// // Create a matrix to rotate
    /// let mut matrix = Matrix3x2::new(
    ///     3.0, 4.0,
    ///     1.0, 0.0,
    ///     2.0, 1.0
    /// );
    ///
    /// // Create a rotation from the first row
    /// let row0 = Vector2::new(matrix[(0, 0)], matrix[(0, 1)]);
    /// if let Some((rotation, _norm)) = GivensRotation::cancel_y(&row0) {
    ///     // Apply to all rows
    ///     rotation.rotate_rows(&mut matrix.fixed_columns_mut::<2>(0));
    ///
    ///     println!("Rotated matrix:\n{}", matrix);
    /// }
    /// ```
    ///
    /// # Difference from rotate()
    /// ```
    /// use nalgebra::{Matrix2, linalg::givens::GivensRotation};
    ///
    /// let (rotation, _norm) = GivensRotation::new(3.0, 4.0);
    ///
    /// // Left multiplication: G * M (transforms rows)
    /// let mut m1 = Matrix2::new(
    ///     1.0, 2.0,
    ///     3.0, 4.0
    /// );
    /// rotation.rotate(&mut m1.fixed_rows_mut::<2>(0));
    /// println!("G * M =\n{}", m1);
    ///
    /// // Right multiplication: M * G (transforms columns)
    /// let mut m2 = Matrix2::new(
    ///     1.0, 2.0,
    ///     3.0, 4.0
    /// );
    /// rotation.rotate_rows(&mut m2.fixed_columns_mut::<2>(0));
    /// println!("M * G =\n{}", m2);
    /// ```
    ///
    /// # Practical Use Case: Eigenvector Computation
    /// ```
    /// use nalgebra::{Matrix2, Vector2, linalg::givens::GivensRotation};
    ///
    /// // In iterative eigenvalue algorithms, we accumulate rotations
    /// // to build the eigenvector matrix
    /// let mut eigenvectors = Matrix2::identity();
    ///
    /// // Apply a sequence of Givens rotations
    /// let (g1, _) = GivensRotation::new(0.8, 0.6);
    /// g1.rotate_rows(&mut eigenvectors.fixed_columns_mut::<2>(0));
    ///
    /// // The columns of eigenvectors now contain the rotated basis
    /// println!("Accumulated rotation:\n{}", eigenvectors);
    ///
    /// // Verify orthogonality
    /// let product = eigenvectors.transpose() * eigenvectors;
    /// let identity = Matrix2::identity();
    /// assert!((product - identity).norm() < 1e-10);
    /// ```
    ///
    /// # When to Use
    ///
    /// Use `rotate_rows()` when you need to:
    /// - Transform columns rather than rows
    /// - Accumulate rotations in eigenvalue algorithms
    /// - Build orthogonal transformation matrices
    /// - Apply the transpose of a rotation (equivalent to `inverse().rotate()`)
    ///
    /// # See Also
    /// - [`rotate()`](Self::rotate): Left-multiplication (G * M, transforms rows)
    /// - [`inverse()`](Self::inverse): Get the inverse rotation
    /// - [`cancel_y()`](Self::cancel_y): Create rotation to zero an element
    pub fn rotate_rows<R2: Dim, C2: Dim, S2: StorageMut<T, R2, C2>>(
        &self,
        lhs: &mut Matrix<T, R2, C2, S2>,
    ) where
        ShapeConstraint: DimEq<C2, U2>,
    {
        assert_eq!(
            lhs.ncols(),
            2,
            "Unit complex rotation: the input matrix must have exactly two columns."
        );
        let s = self.s.clone();
        let c = self.c.clone();

        // TODO: can we optimize that to iterate on one column at a time ?
        for j in 0..lhs.nrows() {
            unsafe {
                let a = lhs.get_unchecked((j, 0)).clone();
                let b = lhs.get_unchecked((j, 1)).clone();

                *lhs.get_unchecked_mut((j, 0)) = a.clone().scale(c.clone()) + s.clone() * b.clone();
                *lhs.get_unchecked_mut((j, 1)) = -s.clone().conjugate() * a + b.scale(c.clone());
            }
        }
    }
}
