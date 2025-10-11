#[cfg(feature = "serde-serialize-no-std")]
use serde::{Deserialize, Serialize};

use crate::allocator::Allocator;
use crate::base::{DefaultAllocator, OMatrix, OVector};
use crate::dimension::{Const, DimDiff, DimSub, U1};
use simba::scalar::ComplexField;

use crate::Matrix;
use crate::linalg::householder;
use std::mem::MaybeUninit;

/// Hessenberg decomposition of a general matrix.
///
/// The Hessenberg decomposition is a factorization of a square matrix **A** into the form:
/// **A = Q * H * Q<sup>T</sup>**
///
/// where:
/// - **Q** is an orthogonal (or unitary for complex matrices) matrix
/// - **H** is an upper Hessenberg matrix (almost triangular with zeros below the first subdiagonal)
/// - **Q<sup>T</sup>** is the transpose (or conjugate transpose) of Q
///
/// # What is Hessenberg Form?
///
/// An upper Hessenberg matrix has the property that all entries below the first subdiagonal
/// are zero. In other words, `H[i,j] = 0` for all `i > j+1`. For example, a 5×5 Hessenberg
/// matrix looks like:
///
/// ```text
/// ┌                     ┐
/// │ x  x  x  x  x │
/// │ x  x  x  x  x │
/// │ 0  x  x  x  x │
/// │ 0  0  x  x  x │
/// │ 0  0  0  x  x │
/// └                     ┘
/// ```
///
/// where 'x' represents potentially non-zero entries and '0' represents structural zeros.
///
/// # Why is it Useful?
///
/// The Hessenberg decomposition is a crucial intermediate step in many numerical algorithms:
///
/// - **Eigenvalue computation**: It's the first step in the QR algorithm for computing eigenvalues.
///   Converting to Hessenberg form reduces computational complexity from O(n⁴) to O(n³).
///
/// - **Control theory**: Used in solving matrix equations like the Lyapunov and Sylvester equations,
///   which appear frequently in stability analysis and controller design.
///
/// - **Solving linear systems**: For certain structured problems, Hessenberg form can be exploited
///   for more efficient solutions.
///
/// - **Matrix functions**: Computing matrix exponentials, logarithms, and other functions is more
///   efficient when starting from Hessenberg form.
///
/// # Performance
///
/// The reduction to Hessenberg form requires O(n³) operations, which is significantly less than
/// the O(n³) per iteration required by the QR algorithm on a general matrix. Once in Hessenberg
/// form, each QR iteration only requires O(n²) operations.
///
/// # Examples
///
/// Basic usage with a 3×3 matrix:
///
/// ```
/// use nalgebra::Matrix3;
///
/// let m = Matrix3::new(
///     1.0, 2.0, 3.0,
///     4.0, 5.0, 6.0,
///     7.0, 8.0, 9.0
/// );
///
/// let hess = m.hessenberg();
/// let (q, h) = hess.unpack();
///
/// // Verify the decomposition: A = Q * H * Q^T
/// assert!((m - q * h * q.transpose()).norm() < 1e-10);
///
/// // H should be in Hessenberg form (zeros below first subdiagonal)
/// assert!(h[(2, 0)].abs() < 1e-10);
/// ```
///
/// Using Hessenberg decomposition as a stepping stone to eigenvalues:
///
/// ```
/// use nalgebra::Matrix4;
///
/// let m = Matrix4::new(
///     4.0, 1.0, 0.0, 0.0,
///     1.0, 3.0, 1.0, 0.0,
///     0.0, 1.0, 2.0, 1.0,
///     0.0, 0.0, 1.0, 1.0,
/// );
///
/// // The Hessenberg decomposition is used internally by eigenvalue algorithms
/// let eigenvalues = m.eigenvalues();
/// println!("Eigenvalues: {:?}", eigenvalues);
/// ```
///
/// Working with dynamic matrices:
///
/// ```
/// use nalgebra::DMatrix;
///
/// let m = DMatrix::from_row_slice(4, 4, &[
///     1.0, 2.0, 3.0, 4.0,
///     5.0, 6.0, 7.0, 8.0,
///     9.0, 10.0, 11.0, 12.0,
///     13.0, 14.0, 15.0, 16.0,
/// ]);
///
/// let hess = m.hessenberg();
/// let h = hess.h();
///
/// // The Hessenberg matrix preserves eigenvalues
/// // All entries below the first subdiagonal should be zero
/// for i in 2..4 {
///     for j in 0..i-1 {
///         assert!(h[(i, j)].abs() < 1e-10);
///     }
/// }
/// ```
///
/// # See Also
///
/// - [`Hessenberg::new`] - Create a new Hessenberg decomposition
/// - [`Hessenberg::unpack`] - Extract Q and H matrices
/// - [`Hessenberg::h`] - Get just the H matrix
/// - [`Hessenberg::q`] - Get just the Q matrix
/// - [`Matrix::schur`](crate::linalg::Schur) - Uses Hessenberg form internally
/// - [`Matrix::eigenvalues`](crate::base::Matrix::eigenvalues) - Computes eigenvalues via Hessenberg and Schur
#[cfg_attr(feature = "serde-serialize-no-std", derive(Serialize, Deserialize))]
#[cfg_attr(
    feature = "serde-serialize-no-std",
    serde(bound(serialize = "DefaultAllocator: Allocator<D, D> +
                           Allocator<DimDiff<D, U1>>,
         OMatrix<T, D, D>: Serialize,
         OVector<T, DimDiff<D, U1>>: Serialize"))
)]
#[cfg_attr(
    feature = "serde-serialize-no-std",
    serde(bound(deserialize = "DefaultAllocator: Allocator<D, D> +
                           Allocator<DimDiff<D, U1>>,
         OMatrix<T, D, D>: Deserialize<'de>,
         OVector<T, DimDiff<D, U1>>: Deserialize<'de>"))
)]
#[cfg_attr(feature = "defmt", derive(defmt::Format))]
#[derive(Clone, Debug)]
pub struct Hessenberg<T: ComplexField, D: DimSub<U1>>
where
    DefaultAllocator: Allocator<D, D> + Allocator<DimDiff<D, U1>>,
{
    hess: OMatrix<T, D, D>,
    subdiag: OVector<T, DimDiff<D, U1>>,
}

impl<T: ComplexField, D: DimSub<U1>> Copy for Hessenberg<T, D>
where
    DefaultAllocator: Allocator<D, D> + Allocator<DimDiff<D, U1>>,
    OMatrix<T, D, D>: Copy,
    OVector<T, DimDiff<D, U1>>: Copy,
{
}

impl<T: ComplexField, D: DimSub<U1>> Hessenberg<T, D>
where
    DefaultAllocator: Allocator<D, D> + Allocator<D> + Allocator<DimDiff<D, U1>>,
{
    /// Computes the Hessenberg decomposition using Householder reflections.
    ///
    /// This method decomposes a square matrix **A** into **Q * H * Q<sup>T</sup>**, where
    /// **Q** is orthogonal/unitary and **H** is in upper Hessenberg form (zeros below the
    /// first subdiagonal).
    ///
    /// # What This Does
    ///
    /// This function transforms your square matrix into Hessenberg form, which is a
    /// "nearly triangular" form that makes subsequent computations much more efficient.
    /// The transformation preserves important properties like eigenvalues while making
    /// the matrix structure simpler to work with.
    ///
    /// # Algorithm
    ///
    /// The implementation uses Householder reflections, which are numerically stable
    /// orthogonal transformations. The algorithm systematically zeros out elements below
    /// the first subdiagonal, column by column, resulting in O(n³) complexity.
    ///
    /// # When to Use This
    ///
    /// - As a preprocessing step before eigenvalue computation
    /// - When solving matrix equations (Lyapunov, Sylvester)
    /// - For computing matrix functions efficiently
    /// - In control theory applications
    ///
    /// # Panics
    ///
    /// Panics if the input matrix is not square or is empty.
    ///
    /// # Example: Basic 3×3 Decomposition
    ///
    /// ```
    /// use nalgebra::Matrix3;
    ///
    /// let m = Matrix3::new(
    ///     1.0, 2.0, 3.0,
    ///     4.0, 5.0, 6.0,
    ///     7.0, 8.0, 9.0
    /// );
    ///
    /// let hess = m.clone().hessenberg();
    /// let (q, h) = hess.unpack();
    ///
    /// // Verify: A = Q * H * Q^T
    /// let reconstructed = &q * &h * q.transpose();
    /// assert!(reconstructed.relative_eq(&m, 1e-10, 1e-10));
    ///
    /// // Q is orthogonal: Q * Q^T = I
    /// let identity = &q * q.transpose();
    /// assert!(identity.is_identity(1e-10));
    ///
    /// // H is in Hessenberg form (zero below first subdiagonal)
    /// assert!(h[(2, 0)].abs() < 1e-10);
    /// ```
    ///
    /// # Example: Eigenvalue Computation Setup
    ///
    /// ```
    /// use nalgebra::Matrix4;
    ///
    /// // Create a matrix representing a dynamical system
    /// let system_matrix = Matrix4::new(
    ///     -0.5,  0.2,  0.0,  0.0,
    ///      0.1, -0.3,  0.1,  0.0,
    ///      0.0,  0.1, -0.4,  0.1,
    ///      0.0,  0.0,  0.2, -0.6
    /// );
    ///
    /// // Convert to Hessenberg form (first step for eigenvalue algorithms)
    /// let hess = system_matrix.hessenberg();
    ///
    /// // The Hessenberg form has the same eigenvalues
    /// // but is much faster to work with
    /// let h = hess.h();
    /// println!("Hessenberg matrix:\n{}", h);
    /// ```
    ///
    /// # Example: Working with Dynamic Matrices
    ///
    /// ```
    /// use nalgebra::DMatrix;
    ///
    /// // Create a 5×5 random-like matrix
    /// let data = vec![
    ///     1.0, 2.0, 3.0, 4.0, 5.0,
    ///     2.0, 3.0, 4.0, 5.0, 6.0,
    ///     3.0, 4.0, 5.0, 6.0, 7.0,
    ///     4.0, 5.0, 6.0, 7.0, 8.0,
    ///     5.0, 6.0, 7.0, 8.0, 9.0,
    /// ];
    /// let m = DMatrix::from_row_slice(5, 5, &data);
    ///
    /// let hess = m.clone().hessenberg();
    /// let h = hess.h();
    ///
    /// // Verify Hessenberg structure: all entries below
    /// // the first subdiagonal should be zero
    /// for i in 0..5 {
    ///     for j in 0..5 {
    ///         if i > j + 1 {
    ///             assert!(h[(i, j)].abs() < 1e-10,
    ///                 "Entry ({}, {}) should be zero, got {}", i, j, h[(i, j)]);
    ///         }
    ///     }
    /// }
    /// ```
    ///
    /// # Example: Control Theory - Lyapunov Equation
    ///
    /// ```
    /// use nalgebra::Matrix2;
    ///
    /// // System matrix for a stable linear system
    /// let a = Matrix2::new(
    ///     -1.0,  0.5,
    ///      0.0, -2.0
    /// );
    ///
    /// // Hessenberg form is often used as a preprocessing step
    /// // for solving Lyapunov equations: A*X + X*A^T = -Q
    /// let hess = a.hessenberg();
    /// let h = hess.h();
    /// let q = hess.q();
    ///
    /// // In a full implementation, you would solve in Hessenberg coordinates
    /// // for better numerical efficiency
    /// println!("Working in Hessenberg form is faster for large matrices");
    /// ```
    ///
    /// # Performance Notes
    ///
    /// - Time complexity: O(n³) for an n×n matrix
    /// - Space complexity: O(n²) for storing Q and H
    /// - For small matrices (n < 10), the overhead may not be worth it
    /// - For large matrices, this is essential for efficient eigenvalue computation
    ///
    /// # See Also
    ///
    /// - [`new_with_workspace`](#method.new_with_workspace): Version with pre-allocated workspace
    /// - [`unpack`](#method.unpack): Extract both Q and H matrices
    /// - [`h`](#method.h): Get just the Hessenberg matrix H
    /// - [`q`](#method.q): Get just the orthogonal matrix Q
    /// - [`Matrix::eigenvalues`](crate::base::Matrix::eigenvalues): Uses Hessenberg internally
    pub fn new(hess: OMatrix<T, D, D>) -> Self {
        let mut work = Matrix::zeros_generic(hess.shape_generic().0, Const::<1>);
        Self::new_with_workspace(hess, &mut work)
    }

    /// Computes the Hessenberg decomposition using Householder reflections with a pre-allocated workspace.
    ///
    /// This is a performance-optimized version of [`new`](#method.new) that allows you to
    /// provide your own workspace buffer, avoiding repeated allocations when computing multiple
    /// decompositions.
    ///
    /// # What This Does
    ///
    /// This function performs the same Hessenberg decomposition as [`new`](#method.new), but
    /// uses a caller-provided workspace vector instead of allocating its own. This is particularly
    /// useful when you need to compute many decompositions in a loop and want to avoid the overhead
    /// of repeated memory allocations.
    ///
    /// # Arguments
    ///
    /// * `hess` - The square matrix to decompose (will be consumed)
    /// * `work` - A workspace vector of length `D` (its initial contents don't matter; they will be overwritten)
    ///
    /// # When to Use This
    ///
    /// - When computing decompositions in a tight loop
    /// - When memory allocations are expensive in your context
    /// - When profiling shows that allocation is a bottleneck
    /// - In embedded systems or real-time applications with strict performance requirements
    ///
    /// For most use cases, the simpler [`new`](#method.new) method is more convenient.
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - The input matrix is not square
    /// - The input matrix is empty
    /// - The workspace vector length doesn't match the matrix dimension
    ///
    /// # Example: Basic Usage
    ///
    /// ```
    /// use nalgebra::{Matrix3, Vector3};
    ///
    /// let m = Matrix3::new(
    ///     1.0, 2.0, 3.0,
    ///     4.0, 5.0, 6.0,
    ///     7.0, 8.0, 9.0
    /// );
    ///
    /// // Pre-allocate workspace
    /// let mut workspace = Vector3::zeros();
    ///
    /// // Compute decomposition with our workspace
    /// let hess = m.clone().hessenberg_with_workspace(&mut workspace);
    /// let (q, h) = hess.unpack();
    ///
    /// // Verify the decomposition
    /// assert!((m - q * h * q.transpose()).norm() < 1e-10);
    /// ```
    ///
    /// # Example: Reusing Workspace in a Loop
    ///
    /// ```
    /// use nalgebra::{Matrix4, Vector4, DMatrix};
    ///
    /// // Simulate processing multiple matrices
    /// let matrices = vec![
    ///     Matrix4::new(
    ///         1.0, 2.0, 3.0, 4.0,
    ///         2.0, 3.0, 4.0, 5.0,
    ///         3.0, 4.0, 5.0, 6.0,
    ///         4.0, 5.0, 6.0, 7.0,
    ///     ),
    ///     Matrix4::new(
    ///         2.0, 1.0, 0.0, 0.0,
    ///         1.0, 2.0, 1.0, 0.0,
    ///         0.0, 1.0, 2.0, 1.0,
    ///         0.0, 0.0, 1.0, 2.0,
    ///     ),
    /// ];
    ///
    /// // Allocate workspace once
    /// let mut workspace = Vector4::zeros();
    ///
    /// // Reuse it for all decompositions
    /// for matrix in matrices {
    ///     let hess = matrix.hessenberg_with_workspace(&mut workspace);
    ///     let h = hess.h();
    ///     // Process the Hessenberg form...
    ///     println!("Computed Hessenberg form");
    /// }
    /// ```
    ///
    /// # Example: Performance Comparison
    ///
    /// ```
    /// use nalgebra::{Matrix3, Vector3};
    ///
    /// let m = Matrix3::new(
    ///     1.0, 2.0, 3.0,
    ///     4.0, 5.0, 6.0,
    ///     7.0, 8.0, 9.0
    /// );
    ///
    /// // Method 1: Standard (allocates workspace internally)
    /// let hess1 = m.clone().hessenberg();
    ///
    /// // Method 2: With workspace (no allocation if workspace is reused)
    /// let mut workspace = Vector3::zeros();
    /// let hess2 = m.clone().hessenberg_with_workspace(&mut workspace);
    ///
    /// // Both produce the same result
    /// let h1 = hess1.h();
    /// let h2 = hess2.h();
    /// assert!(h1.relative_eq(&h2, 1e-10, 1e-10));
    /// ```
    ///
    /// # Performance Notes
    ///
    /// - Same O(n³) time complexity as [`new`](#method.new)
    /// - Saves one O(n) allocation per call
    /// - The performance gain is most noticeable for:
    ///   - Small to medium matrices (where allocation overhead is significant)
    ///   - Tight loops with many decompositions
    ///   - Memory-constrained environments
    ///
    /// # See Also
    ///
    /// - [`new`](#method.new): Simpler version that handles workspace allocation automatically
    /// - [`unpack`](#method.unpack): Extract Q and H matrices
    /// - [`h`](#method.h): Get just the Hessenberg matrix
    /// - [`q`](#method.q): Get just the orthogonal matrix
    pub fn new_with_workspace(mut hess: OMatrix<T, D, D>, work: &mut OVector<T, D>) -> Self {
        assert!(
            hess.is_square(),
            "Cannot compute the hessenberg decomposition of a non-square matrix."
        );

        let dim = hess.shape_generic().0;

        assert!(
            dim.value() != 0,
            "Cannot compute the hessenberg decomposition of an empty matrix."
        );
        assert_eq!(
            dim.value(),
            work.len(),
            "Hessenberg: invalid workspace size."
        );

        if dim.value() == 0 {
            return Hessenberg {
                hess,
                subdiag: Matrix::zeros_generic(dim.sub(Const::<1>), Const::<1>),
            };
        }

        let mut subdiag = Matrix::uninit(dim.sub(Const::<1>), Const::<1>);

        for ite in 0..dim.value() - 1 {
            subdiag[ite] = MaybeUninit::new(householder::clear_column_unchecked(
                &mut hess,
                ite,
                1,
                Some(work),
            ));
        }

        // Safety: subdiag is now fully initialized.
        let subdiag = unsafe { subdiag.assume_init() };
        Hessenberg { hess, subdiag }
    }

    /// Retrieves the orthogonal matrix `Q` and the Hessenberg matrix `H` from this decomposition.
    ///
    /// This method consumes the Hessenberg decomposition and returns both matrices as a tuple `(Q, H)`.
    ///
    /// # What This Returns
    ///
    /// Returns a tuple `(Q, H)` where:
    /// - **Q**: An orthogonal (or unitary for complex matrices) matrix satisfying Q * Q<sup>T</sup> = I
    /// - **H**: An upper Hessenberg matrix (zeros below the first subdiagonal)
    ///
    /// The original matrix can be reconstructed as: **A = Q * H * Q<sup>T</sup>**
    ///
    /// # Understanding the Matrices
    ///
    /// **Q (Orthogonal Matrix)**:
    /// - Represents the transformation basis
    /// - Preserves lengths and angles (isometry)
    /// - Columns form an orthonormal basis
    /// - Numerically stable: condition number = 1
    ///
    /// **H (Hessenberg Matrix)**:
    /// - Almost upper triangular (one extra diagonal of non-zeros)
    /// - Has the same eigenvalues as the original matrix
    /// - Much faster for subsequent algorithms (QR iterations, etc.)
    /// - Preserves characteristic polynomial
    ///
    /// # When to Use This
    ///
    /// Use `unpack()` when you need both matrices, for example:
    /// - Verifying the decomposition accuracy
    /// - Implementing custom algorithms that need both Q and H
    /// - Transforming between original and Hessenberg coordinates
    ///
    /// If you only need one matrix, use [`h()`](#method.h) or [`q()`](#method.q) instead
    /// for better performance.
    ///
    /// # Example: Basic Unpacking and Verification
    ///
    /// ```
    /// use nalgebra::Matrix3;
    ///
    /// let m = Matrix3::new(
    ///     1.0, 2.0, 3.0,
    ///     4.0, 5.0, 6.0,
    ///     7.0, 8.0, 9.0
    /// );
    ///
    /// let hess = m.clone().hessenberg();
    /// let (q, h) = hess.unpack();
    ///
    /// // Verify: A = Q * H * Q^T
    /// let reconstructed = &q * &h * q.transpose();
    /// assert!(reconstructed.relative_eq(&m, 1e-10, 1e-10));
    ///
    /// // Verify Q is orthogonal: Q * Q^T = I
    /// let identity = &q * q.transpose();
    /// assert!(identity.is_identity(1e-10));
    ///
    /// // Verify H is in Hessenberg form
    /// assert!(h[(2, 0)].abs() < 1e-10); // Below first subdiagonal
    /// ```
    ///
    /// # Example: Analyzing Structure
    ///
    /// ```
    /// use nalgebra::Matrix4;
    ///
    /// let m = Matrix4::new(
    ///     1.0, 2.0, 3.0, 4.0,
    ///     5.0, 6.0, 7.0, 8.0,
    ///     9.0, 10.0, 11.0, 12.0,
    ///     13.0, 14.0, 15.0, 16.0,
    /// );
    ///
    /// let hess = m.hessenberg();
    /// let (q, h) = hess.unpack();
    ///
    /// println!("Original matrix:\n{}", m);
    /// println!("\nOrthogonal matrix Q:\n{}", q);
    /// println!("\nHessenberg matrix H:\n{}", h);
    ///
    /// // H should have zeros below the first subdiagonal
    /// for i in 0..4 {
    ///     for j in 0..4 {
    ///         if i > j + 1 {
    ///             assert!(h[(i, j)].abs() < 1e-10,
    ///                 "H[{}, {}] should be zero but is {}", i, j, h[(i, j)]);
    ///         }
    ///     }
    /// }
    /// ```
    ///
    /// # Example: Coordinate Transformation
    ///
    /// ```
    /// use nalgebra::{Matrix3, Vector3};
    ///
    /// let m = Matrix3::new(
    ///     2.0, 1.0, 0.5,
    ///     1.0, 3.0, 1.0,
    ///     0.5, 1.0, 4.0,
    /// );
    ///
    /// let hess = m.clone().hessenberg();
    /// let (q, h) = hess.unpack();
    ///
    /// // Transform a vector to Hessenberg coordinates
    /// let v = Vector3::new(1.0, 2.0, 3.0);
    /// let v_hess = q.transpose() * v;
    ///
    /// // Apply the matrix in Hessenberg form (potentially faster)
    /// let result_hess = h * v_hess;
    ///
    /// // Transform back to original coordinates
    /// let result = q * result_hess;
    ///
    /// // Should equal m * v
    /// let expected = m * v;
    /// assert!(result.relative_eq(&expected, 1e-10, 1e-10));
    /// ```
    ///
    /// # Example: Eigenvalue Preservation
    ///
    /// ```
    /// use nalgebra::Matrix2;
    ///
    /// let m = Matrix2::new(
    ///     3.0, 1.0,
    ///     1.0, 2.0
    /// );
    ///
    /// let hess = m.clone().hessenberg();
    /// let (q, h) = hess.unpack();
    ///
    /// // Both matrices have the same eigenvalues
    /// let eigs_original = m.eigenvalues().unwrap();
    /// let eigs_hessenberg = h.eigenvalues().unwrap();
    ///
    /// // They should match (possibly in different order)
    /// println!("Original eigenvalues: {}", eigs_original);
    /// println!("Hessenberg eigenvalues: {}", eigs_hessenberg);
    /// ```
    ///
    /// # Example: Control Theory Application
    ///
    /// ```
    /// use nalgebra::Matrix3;
    ///
    /// // State-space system matrix
    /// let a = Matrix3::new(
    ///     -1.0,  0.5,  0.0,
    ///      0.5, -2.0,  0.3,
    ///      0.0,  0.3, -3.0
    /// );
    ///
    /// let hess = a.clone().hessenberg();
    /// let (q, h) = hess.unpack();
    ///
    /// // H is better conditioned for computing matrix exponential
    /// // or solving Lyapunov equations
    /// println!("System in Hessenberg form is ready for stability analysis");
    ///
    /// // The transformation Q can be used to transform state variables
    /// // or observations between coordinate systems
    /// ```
    ///
    /// # Performance Notes
    ///
    /// - This method consumes `self`, moving ownership of the internal data
    /// - Computing Q requires O(n³) operations
    /// - If you only need H, use [`unpack_h()`](#method.unpack_h) which is faster
    /// - If you only need Q, use [`q()`](#method.q) (which clones H internally)
    ///
    /// # See Also
    ///
    /// - [`unpack_h`](#method.unpack_h): Get only H (consumes self, more efficient)
    /// - [`h`](#method.h): Get only H (clones, keeps self alive)
    /// - [`q`](#method.q): Get only Q
    /// - [`new`](#method.new): Create a Hessenberg decomposition
    #[inline]
    pub fn unpack(self) -> (OMatrix<T, D, D>, OMatrix<T, D, D>) {
        let q = self.q();

        (q, self.unpack_h())
    }

    /// Retrieves the upper Hessenberg matrix `H` of this decomposition.
    ///
    /// This method consumes the Hessenberg decomposition and returns only the H matrix,
    /// which is more efficient than calling [`unpack()`](#method.unpack) if you don't need Q.
    ///
    /// # What This Returns
    ///
    /// Returns the upper Hessenberg matrix **H**, which has the following properties:
    /// - All entries below the first subdiagonal are zero: `H[i,j] = 0` for `i > j+1`
    /// - Has the same eigenvalues as the original matrix
    /// - The characteristic polynomial is preserved
    /// - More efficient for subsequent numerical algorithms than the original matrix
    ///
    /// # Matrix Structure
    ///
    /// For a 5×5 Hessenberg matrix:
    /// ```text
    /// ┌                     ┐
    /// │ *  *  *  *  * │  ← Any values
    /// │ *  *  *  *  * │  ← Any values
    /// │ 0  *  *  *  * │  ← Zero, then any values
    /// │ 0  0  *  *  * │  ← Zeros, then any values
    /// │ 0  0  0  *  * │  ← Zeros, then any values
    /// └                     ┘
    /// ```
    ///
    /// # When to Use This
    ///
    /// Use `unpack_h()` when:
    /// - You only need the H matrix, not Q
    /// - You're implementing eigenvalue algorithms
    /// - You want to analyze the matrix structure
    /// - Memory efficiency is important (consumes self, no Q allocation)
    ///
    /// If you also need Q, use [`unpack()`](#method.unpack) instead. If you need to keep
    /// the decomposition alive, use [`h()`](#method.h).
    ///
    /// # Example: Basic Usage
    ///
    /// ```
    /// use nalgebra::Matrix3;
    ///
    /// let m = Matrix3::new(
    ///     1.0, 2.0, 3.0,
    ///     4.0, 5.0, 6.0,
    ///     7.0, 8.0, 9.0
    /// );
    ///
    /// let hess = m.clone().hessenberg();
    /// let h = hess.unpack_h();
    ///
    /// // H is in Hessenberg form
    /// assert!(h[(2, 0)].abs() < 1e-10);
    ///
    /// // Upper part and first subdiagonal may be non-zero
    /// println!("Hessenberg matrix:\n{}", h);
    /// ```
    ///
    /// # Example: Verifying Hessenberg Structure
    ///
    /// ```
    /// use nalgebra::Matrix4;
    ///
    /// let m = Matrix4::from_row_slice(&[
    ///     1.0, 2.0, 3.0, 4.0,
    ///     5.0, 6.0, 7.0, 8.0,
    ///     9.0, 10.0, 11.0, 12.0,
    ///     13.0, 14.0, 15.0, 16.0,
    /// ]);
    ///
    /// let hess = m.hessenberg();
    /// let h = hess.unpack_h();
    ///
    /// // Verify all elements below first subdiagonal are zero
    /// for i in 0..4 {
    ///     for j in 0..4 {
    ///         if i > j + 1 {
    ///             assert!(h[(i, j)].abs() < 1e-10,
    ///                 "H[{}, {}] = {} should be zero", i, j, h[(i, j)]);
    ///         }
    ///     }
    /// }
    ///
    /// println!("Verified: H is in proper Hessenberg form");
    /// ```
    ///
    /// # Example: Eigenvalue Preservation
    ///
    /// ```
    /// use nalgebra::Matrix3;
    ///
    /// let m = Matrix3::new(
    ///     3.0, 1.0, 0.5,
    ///     1.0, 4.0, 1.0,
    ///     0.5, 1.0, 5.0,
    /// );
    ///
    /// let hess = m.clone().hessenberg();
    /// let h = hess.unpack_h();
    ///
    /// // Both matrices should have the same eigenvalues
    /// let eigs_original = m.eigenvalues().unwrap();
    /// let eigs_hessenberg = h.eigenvalues().unwrap();
    ///
    /// // Eigenvalues match (may be in different order)
    /// println!("Original eigenvalues: {}", eigs_original);
    /// println!("Hessenberg eigenvalues: {}", eigs_hessenberg);
    /// ```
    ///
    /// # Example: Preparing for QR Algorithm
    ///
    /// ```
    /// use nalgebra::Matrix4;
    ///
    /// let m = Matrix4::new(
    ///     2.0, 1.0, 0.0, 0.0,
    ///     1.0, 3.0, 1.0, 0.0,
    ///     0.0, 1.0, 4.0, 1.0,
    ///     0.0, 0.0, 1.0, 5.0,
    /// );
    ///
    /// // First step: reduce to Hessenberg form
    /// let hess = m.hessenberg();
    /// let h = hess.unpack_h();
    ///
    /// // H is now ready for the QR algorithm (much faster than on original matrix)
    /// // The QR algorithm would iterate on H rather than m
    /// println!("Hessenberg form ready for QR iterations");
    /// ```
    ///
    /// # Example: Control Theory - System Analysis
    ///
    /// ```
    /// use nalgebra::Matrix3;
    ///
    /// // System dynamics matrix
    /// let a = Matrix3::new(
    ///     -1.0,  0.5,  0.0,
    ///      0.5, -2.0,  0.3,
    ///      0.0,  0.3, -3.0
    /// );
    ///
    /// let hess = a.hessenberg();
    /// let h = hess.unpack_h();
    ///
    /// // The Hessenberg form is often used in:
    /// // - Computing matrix exponentials (for system solution)
    /// // - Solving Lyapunov equations (for stability analysis)
    /// // - Computing controllability/observability gramians
    /// println!("System matrix in Hessenberg form:\n{}", h);
    /// ```
    ///
    /// # Performance Notes
    ///
    /// - This method consumes `self` (moves ownership)
    /// - O(n²) operation to finalize the H matrix
    /// - More efficient than `unpack()` since it doesn't compute Q
    /// - For repeated access, use [`h()`](#method.h) instead (clones but keeps decomposition)
    ///
    /// # See Also
    ///
    /// - [`h`](#method.h): Get H without consuming self (clones instead)
    /// - [`unpack`](#method.unpack): Get both Q and H
    /// - [`q`](#method.q): Get only the orthogonal matrix Q
    /// - [`new`](#method.new): Create the decomposition
    #[inline]
    pub fn unpack_h(mut self) -> OMatrix<T, D, D> {
        let dim = self.hess.nrows();
        self.hess.fill_lower_triangle(T::zero(), 2);
        self.hess
            .view_mut((1, 0), (dim - 1, dim - 1))
            .set_partial_diagonal(
                self.subdiag
                    .iter()
                    .map(|e| T::from_real(e.clone().modulus())),
            );
        self.hess
    }

    // TODO: add a h that moves out of self.
    /// Retrieves the upper Hessenberg matrix `H` of this decomposition.
    ///
    /// This method clones the internal data and returns the H matrix, keeping the
    /// decomposition alive for further use. If you don't need the decomposition afterward,
    /// use [`unpack_h()`](#method.unpack_h) instead for better performance.
    ///
    /// # What This Returns
    ///
    /// Returns a copy of the upper Hessenberg matrix **H**, which has:
    /// - Zeros below the first subdiagonal
    /// - The same eigenvalues as the original matrix
    /// - Preserved characteristic polynomial
    /// - Simpler structure than the original matrix
    ///
    /// # When to Use This
    ///
    /// Use `h()` when:
    /// - You need to access H multiple times
    /// - You want to keep the decomposition for later Q access
    /// - You're not concerned about the O(n²) cloning cost
    /// - You need both H and Q but want to access them separately
    ///
    /// For one-time use, [`unpack_h()`](#method.unpack_h) is more efficient.
    ///
    /// # Example: Multiple Accesses
    ///
    /// ```
    /// use nalgebra::Matrix3;
    ///
    /// let m = Matrix3::new(
    ///     1.0, 2.0, 3.0,
    ///     4.0, 5.0, 6.0,
    ///     7.0, 8.0, 9.0
    /// );
    ///
    /// let hess = m.hessenberg();
    ///
    /// // Get H without consuming hess
    /// let h1 = hess.h();
    /// println!("First access:\n{}", h1);
    ///
    /// // Can still access H again
    /// let h2 = hess.h();
    /// assert_eq!(h1, h2);
    ///
    /// // And we can still get Q
    /// let q = hess.q();
    /// ```
    ///
    /// # Example: Analyzing Properties
    ///
    /// ```
    /// use nalgebra::Matrix4;
    ///
    /// let m = Matrix4::new(
    ///     2.0, 1.0, 3.0, 1.0,
    ///     1.0, 3.0, 1.0, 2.0,
    ///     4.0, 2.0, 4.0, 1.0,
    ///     1.0, 3.0, 2.0, 3.0,
    /// );
    ///
    /// let hess = m.hessenberg();
    /// let h = hess.h();
    ///
    /// // Analyze the Hessenberg structure
    /// println!("Hessenberg matrix H:");
    /// for i in 0..4 {
    ///     for j in 0..4 {
    ///         if i > j + 1 {
    ///             // Below first subdiagonal should be zero
    ///             assert!(h[(i, j)].abs() < 1e-10);
    ///         }
    ///     }
    /// }
    ///
    /// // Check sparsity: count non-zero elements
    /// let mut nonzeros = 0;
    /// for i in 0..4 {
    ///     for j in 0..4 {
    ///         if h[(i, j)].abs() > 1e-10 {
    ///             nonzeros += 1;
    ///         }
    ///     }
    /// }
    /// println!("Non-zero elements: {} out of 16", nonzeros);
    /// ```
    ///
    /// # Example: Eigenvalue Computation
    ///
    /// ```
    /// use nalgebra::Matrix3;
    ///
    /// let m = Matrix3::new(
    ///     4.0, 2.0, 1.0,
    ///     2.0, 5.0, 3.0,
    ///     1.0, 3.0, 6.0,
    /// );
    ///
    /// let hess = m.clone().hessenberg();
    /// let h = hess.h();
    ///
    /// // H and m have the same eigenvalues
    /// let eigs_m = m.eigenvalues().unwrap();
    /// let eigs_h = h.eigenvalues().unwrap();
    ///
    /// println!("Original matrix eigenvalues: {}", eigs_m);
    /// println!("Hessenberg matrix eigenvalues: {}", eigs_h);
    ///
    /// // Trace is preserved (sum of eigenvalues)
    /// let trace_m = m[(0,0)] + m[(1,1)] + m[(2,2)];
    /// let trace_h = h[(0,0)] + h[(1,1)] + h[(2,2)];
    /// assert!((trace_m - trace_h).abs() < 1e-10);
    /// ```
    ///
    /// # Example: Using H for Iterative Methods
    ///
    /// ```
    /// use nalgebra::{Matrix3, Vector3};
    ///
    /// let m = Matrix3::new(
    ///     3.0, 1.0, 0.5,
    ///     1.0, 4.0, 1.0,
    ///     0.5, 1.0, 5.0,
    /// );
    ///
    /// let hess = m.hessenberg();
    /// let h = hess.h();
    ///
    /// // Power iteration on H (finds dominant eigenvector faster)
    /// let mut v = Vector3::new(1.0, 1.0, 1.0);
    /// v = v.normalize();
    ///
    /// for _ in 0..5 {
    ///     v = h * v;
    ///     v = v.normalize();
    /// }
    ///
    /// println!("Approximate dominant eigenvector: {}", v);
    /// ```
    ///
    /// # Example: Comparing with Original Matrix
    ///
    /// ```
    /// use nalgebra::Matrix2;
    ///
    /// let m = Matrix2::new(
    ///     5.0, 2.0,
    ///     2.0, 3.0
    /// );
    ///
    /// let hess = m.clone().hessenberg();
    /// let h = hess.h();
    /// let q = hess.q();
    ///
    /// // Reconstruct original: A = Q * H * Q^T
    /// let reconstructed = &q * &h * q.transpose();
    /// assert!(reconstructed.relative_eq(&m, 1e-10, 1e-10));
    ///
    /// // Determinant is preserved
    /// let det_m = m.determinant();
    /// let det_h = h.determinant();
    /// assert!((det_m - det_h).abs() < 1e-10);
    /// ```
    ///
    /// # Performance Notes
    ///
    /// - Clones the internal matrix: O(n²) memory and time
    /// - Less efficient than `unpack_h()` which moves the data
    /// - Use this when you need to keep the decomposition alive
    /// - For one-time access, prefer `unpack_h()`
    ///
    /// # See Also
    ///
    /// - [`unpack_h`](#method.unpack_h): Get H by consuming self (more efficient)
    /// - [`q`](#method.q): Get the orthogonal matrix Q
    /// - [`unpack`](#method.unpack): Get both Q and H, consuming self
    /// - [`new`](#method.new): Create the decomposition
    #[inline]
    #[must_use]
    pub fn h(&self) -> OMatrix<T, D, D> {
        let dim = self.hess.nrows();
        let mut res = self.hess.clone();
        res.fill_lower_triangle(T::zero(), 2);
        res.view_mut((1, 0), (dim - 1, dim - 1))
            .set_partial_diagonal(
                self.subdiag
                    .iter()
                    .map(|e| T::from_real(e.clone().modulus())),
            );
        res
    }

    /// Computes the orthogonal matrix `Q` of this decomposition.
    ///
    /// Returns the orthogonal (or unitary for complex matrices) transformation matrix that
    /// relates the original matrix to its Hessenberg form via **A = Q * H * Q<sup>T</sup>**.
    ///
    /// # What This Returns
    ///
    /// Returns the orthogonal matrix **Q** with the following properties:
    /// - **Orthogonality**: Q * Q<sup>T</sup> = Q<sup>T</sup> * Q = I
    /// - **Preserves norms**: ||Q*v|| = ||v|| for all vectors v
    /// - **Columns are orthonormal**: Each column has unit length and columns are perpendicular
    /// - **Determinant**: det(Q) = ±1 (or |det(Q)| = 1 for complex matrices)
    /// - **Numerically stable**: Condition number equals 1
    ///
    /// # Understanding Q
    ///
    /// The matrix Q represents a change of basis:
    /// - **Transforms** from the original coordinate system to the Hessenberg coordinate system
    /// - **Q<sup>T</sup>** transforms back from Hessenberg to original coordinates
    /// - **Preserves** lengths, angles, and all geometric properties
    /// - **Rotations and reflections** only (no scaling or shearing)
    ///
    /// # When to Use This
    ///
    /// Use `q()` when you need:
    /// - To transform vectors between coordinate systems
    /// - To verify the decomposition: A = Q * H * Q^T
    /// - To understand the geometric transformation
    /// - The transformation matrix for further computations
    ///
    /// # Example: Basic Usage
    ///
    /// ```
    /// use nalgebra::Matrix3;
    ///
    /// let m = Matrix3::new(
    ///     1.0, 2.0, 3.0,
    ///     4.0, 5.0, 6.0,
    ///     7.0, 8.0, 9.0
    /// );
    ///
    /// let hess = m.clone().hessenberg();
    /// let q = hess.q();
    ///
    /// // Q is orthogonal: Q * Q^T = I
    /// let identity = &q * q.transpose();
    /// assert!(identity.is_identity(1e-10));
    ///
    /// // Q preserves norms
    /// println!("Q is orthogonal");
    /// ```
    ///
    /// # Example: Verifying the Decomposition
    ///
    /// ```
    /// use nalgebra::Matrix3;
    ///
    /// let m = Matrix3::new(
    ///     2.0, 1.0, 0.5,
    ///     1.0, 3.0, 1.0,
    ///     0.5, 1.0, 4.0,
    /// );
    ///
    /// let hess = m.clone().hessenberg();
    /// let q = hess.q();
    /// let h = hess.h();
    ///
    /// // Verify: A = Q * H * Q^T
    /// let reconstructed = &q * &h * q.transpose();
    /// assert!(reconstructed.relative_eq(&m, 1e-10, 1e-10));
    ///
    /// // Verify Q is orthogonal
    /// let should_be_identity = &q * q.transpose();
    /// assert!(should_be_identity.is_identity(1e-10));
    ///
    /// println!("Decomposition verified!");
    /// ```
    ///
    /// # Example: Coordinate Transformation
    ///
    /// ```
    /// use nalgebra::{Matrix3, Vector3};
    ///
    /// let m = Matrix3::new(
    ///     3.0, 1.0, 0.0,
    ///     1.0, 4.0, 1.0,
    ///     0.0, 1.0, 5.0,
    /// );
    ///
    /// let hess = m.clone().hessenberg();
    /// let q = hess.q();
    /// let h = hess.h();
    ///
    /// // Original vector
    /// let v = Vector3::new(1.0, 2.0, 3.0);
    ///
    /// // Transform to Hessenberg coordinates
    /// let v_hess = q.transpose() * v;
    ///
    /// // Apply matrix in both coordinate systems
    /// let result_original = m * v;
    /// let result_hess = h * v_hess;
    ///
    /// // Transform back to original coordinates
    /// let result_transformed = q * result_hess;
    ///
    /// // Both approaches give the same result
    /// assert!(result_original.relative_eq(&result_transformed, 1e-10, 1e-10));
    /// ```
    ///
    /// # Example: Analyzing Orthogonality
    ///
    /// ```
    /// use nalgebra::Matrix4;
    ///
    /// let m = Matrix4::new(
    ///     1.0, 2.0, 3.0, 4.0,
    ///     2.0, 3.0, 4.0, 5.0,
    ///     3.0, 4.0, 5.0, 6.0,
    ///     4.0, 5.0, 6.0, 7.0,
    /// );
    ///
    /// let hess = m.hessenberg();
    /// let q = hess.q();
    ///
    /// // Check orthogonality of columns
    /// for i in 0..4 {
    ///     let col_i = q.column(i);
    ///
    ///     // Each column should have unit norm
    ///     assert!((col_i.norm() - 1.0).abs() < 1e-10);
    ///
    ///     // Columns should be orthogonal to each other
    ///     for j in 0..i {
    ///         let col_j = q.column(j);
    ///         let dot_product = col_i.dot(&col_j);
    ///         assert!(dot_product.abs() < 1e-10);
    ///     }
    /// }
    ///
    /// println!("All columns are orthonormal");
    /// ```
    ///
    /// # Example: Geometric Interpretation
    ///
    /// ```
    /// use nalgebra::{Matrix2, Vector2};
    ///
    /// let m = Matrix2::new(
    ///     3.0, 1.0,
    ///     1.0, 2.0
    /// );
    ///
    /// let hess = m.hessenberg();
    /// let q = hess.q();
    ///
    /// // Q represents a rotation (in 2D, possibly with reflection)
    /// let det = q.determinant();
    /// assert!((det.abs() - 1.0).abs() < 1e-10);
    ///
    /// if det > 0.0 {
    ///     println!("Q is a pure rotation");
    /// } else {
    ///     println!("Q is a rotation with reflection");
    /// }
    ///
    /// // Q preserves distances
    /// let v1 = Vector2::new(1.0, 0.0);
    /// let v2 = Vector2::new(0.0, 1.0);
    /// let distance_before = (v1 - v2).norm();
    /// let distance_after = (q * v1 - q * v2).norm();
    /// assert!((distance_before - distance_after).abs() < 1e-10);
    /// ```
    ///
    /// # Example: Multiple Transformations
    ///
    /// ```
    /// use nalgebra::{Matrix3, Vector3};
    ///
    /// let m = Matrix3::new(
    ///     4.0, 1.0, 0.0,
    ///     1.0, 3.0, 1.0,
    ///     0.0, 1.0, 2.0,
    /// );
    ///
    /// let hess = m.hessenberg();
    /// let q = hess.q();
    ///
    /// // Transform multiple vectors at once
    /// let v1 = Vector3::new(1.0, 0.0, 0.0);
    /// let v2 = Vector3::new(0.0, 1.0, 0.0);
    /// let v3 = Vector3::new(0.0, 0.0, 1.0);
    ///
    /// // Q transforms the standard basis
    /// let new_basis_1 = q * v1;
    /// let new_basis_2 = q * v2;
    /// let new_basis_3 = q * v3;
    ///
    /// // The new basis is still orthonormal
    /// assert!((new_basis_1.norm() - 1.0).abs() < 1e-10);
    /// assert!(new_basis_1.dot(&new_basis_2).abs() < 1e-10);
    /// println!("Transformed basis is orthonormal");
    /// ```
    ///
    /// # Performance Notes
    ///
    /// - Computing Q requires O(n³) operations
    /// - The result is a dense n×n matrix
    /// - This method assembles Q from the internal Householder reflections
    /// - If you need both Q and H, use [`unpack()`](#method.unpack) which is more efficient
    ///
    /// # See Also
    ///
    /// - [`h`](#method.h): Get the Hessenberg matrix H
    /// - [`unpack`](#method.unpack): Get both Q and H efficiently
    /// - [`unpack_h`](#method.unpack_h): Get only H (if you don't need Q)
    /// - [`new`](#method.new): Create the decomposition
    #[must_use]
    pub fn q(&self) -> OMatrix<T, D, D> {
        householder::assemble_q(&self.hess, self.subdiag.as_slice())
    }

    #[doc(hidden)]
    pub const fn hess_internal(&self) -> &OMatrix<T, D, D> {
        &self.hess
    }
}
