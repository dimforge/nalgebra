#[cfg(feature = "arbitrary")]
use crate::base::storage::Owned;
#[cfg(feature = "arbitrary")]
use quickcheck::{Arbitrary, Gen};

use crate::base::Scalar;
use crate::base::allocator::Allocator;
use crate::base::dimension::{Dim, Dyn};
use crate::base::{DefaultAllocator, OMatrix};
use simba::scalar::ComplexField;

use crate::debug::RandomOrthogonal;

/// A generator for random symmetric positive-definite (SPD) matrices.
///
/// A symmetric positive-definite matrix is a square matrix that satisfies:
/// - **Symmetric**: A = A^T (equals its own transpose)
/// - **Positive-definite**: x^T * A * x > 0 for all non-zero vectors x
///
/// # Mathematical Properties
///
/// For a symmetric positive-definite matrix A:
/// - All eigenvalues are real and positive
/// - All diagonal elements are positive
/// - The matrix is invertible (determinant > 0)
/// - The matrix has a unique Cholesky decomposition: A = L * L^T
/// - The matrix defines a valid inner product
///
/// # Why Use SPD Matrices?
///
/// Symmetric positive-definite matrices are crucial in many areas:
/// - **Optimization**: They appear in quadratic forms and Hessian matrices of convex functions
/// - **Statistics**: Covariance matrices are always SPD
/// - **Physics**: Mass and stiffness matrices in structural analysis
/// - **Machine Learning**: Kernel matrices in many algorithms
/// - **Numerical Methods**: Testing Cholesky decomposition, conjugate gradient, etc.
///
/// # Generation Method
///
/// This generator creates well-conditioned SPD matrices using the formula:
/// `A = Q * D * Q^T` where:
/// - Q is a random orthogonal matrix (from [`RandomOrthogonal`])
/// - D is a diagonal matrix with positive eigenvalues ≥ 1
///
/// The eigenvalues are computed as `1 + |random_value|`, ensuring they're all at least 1.
/// This makes the matrix "well-conditioned" (not too close to singular), which is important
/// for numerical stability in testing.
///
/// # Examples
///
/// ## Basic Usage
///
/// ```
/// use nalgebra::Matrix3;
/// use nalgebra::debug::RandomSDP;
///
/// // Simple random function
/// let mut counter = 0.0_f64;
/// let mut rand_fn = || { counter += 0.1; counter.abs() };
///
/// // Generate a random 3x3 SPD matrix
/// let spd_gen = RandomSDP::<f64, _>::new(
///     nalgebra::Const::<3>,
///     &mut rand_fn
/// );
/// let matrix = spd_gen.unwrap();
///
/// // Verify it's symmetric
/// assert!((matrix.transpose() - &matrix).norm() < 1e-10);
///
/// // Verify all eigenvalues are positive
/// let eigenvalues = matrix.symmetric_eigen().eigenvalues;
/// assert!(eigenvalues.iter().all(|&x| x > 0.0));
/// ```
///
/// ## Testing Cholesky Decomposition
///
/// ```
/// use nalgebra::Matrix4;
/// use nalgebra::debug::RandomSDP;
///
/// // Simple random function
/// let mut counter = 1.0;
/// let mut rand_fn = || { counter += 0.15; counter };
///
/// // Generate a random SPD matrix
/// let a = RandomSDP::<f64, _>::new(
///     nalgebra::Const::<4>,
///     &mut rand_fn
/// ).unwrap();
///
/// // Cholesky decomposition should succeed for SPD matrices
/// let cholesky = a.clone().cholesky().expect("Cholesky failed");
/// let l = cholesky.l();
///
/// // Verify: A = L * L^T
/// let reconstructed = l * l.transpose();
/// assert!((reconstructed - a).norm() < 1e-10);
/// ```
///
/// ## Testing with Positive-Definiteness Check
///
/// ```
/// use nalgebra::{Vector3, Matrix3};
/// use nalgebra::debug::RandomSDP;
///
/// // Simple random function
/// let mut counter = 2.0_f64;
/// let mut rand_fn = || { counter += 0.2; counter.abs() };
///
/// let spd = RandomSDP::<f64, _>::new(
///     nalgebra::Const::<3>,
///     &mut rand_fn
/// ).unwrap();
///
/// // For any non-zero vector x, x^T * A * x should be positive
/// let x = Vector3::new(1.0, 2.0, 3.0);
/// let result = x.transpose() * &spd * x;
///
/// assert!(result[(0, 0)] > 0.0);
/// ```
///
/// ## Dynamic-Size Matrix
///
/// ```
/// use nalgebra::DMatrix;
/// use nalgebra::debug::RandomSDP;
///
/// // Simple random function
/// let mut counter = 3.0;
/// let mut rand_fn = || { counter += 0.11; counter };
///
/// // Generate an SPD matrix of any size
/// let dim = 10;
/// let spd = RandomSDP::<f64, _>::new(
///     nalgebra::Dyn(dim),
///     &mut rand_fn
/// ).unwrap();
///
/// // Verify dimensions
/// assert_eq!(spd.nrows(), dim);
/// assert_eq!(spd.ncols(), dim);
///
/// // Verify symmetry
/// let diff = spd.transpose() - &spd;
/// assert!(diff.norm() < 1e-9);
/// ```
///
/// ## Testing Optimization Algorithms
///
/// ```
/// use nalgebra::{DVector, DMatrix};
/// use nalgebra::debug::RandomSDP;
///
/// // Simple random function
/// let mut counter = 5.0;
/// let mut rand_fn = || { counter += 0.25; counter };
///
/// // Create a quadratic form: f(x) = 0.5 * x^T * A * x + b^T * x
/// let n = 5;
/// let a = RandomSDP::<f64, _>::new(
///     nalgebra::Dyn(n),
///     &mut rand_fn
/// ).unwrap();
///
/// // SPD matrix ensures the quadratic form is convex
/// // (useful for testing optimization algorithms)
/// ```
///
/// # See Also
///
/// - [`RandomOrthogonal`](crate::debug::RandomOrthogonal): Generate random orthogonal matrices (used internally)
/// - [`Matrix::cholesky`](crate::linalg::Cholesky): Cholesky decomposition for SPD matrices
/// - [`Matrix::symmetric_eigen`](crate::linalg::SymmetricEigen): Eigendecomposition for symmetric matrices
#[derive(Clone, Debug)]
pub struct RandomSDP<T: Scalar, D: Dim = Dyn>
where
    DefaultAllocator: Allocator<D, D>,
{
    m: OMatrix<T, D, D>,
}

impl<T: ComplexField, D: Dim> RandomSDP<T, D>
where
    DefaultAllocator: Allocator<D, D>,
{
    /// Extracts the generated symmetric positive-definite matrix.
    ///
    /// This consumes the `RandomSDP` wrapper and returns the underlying matrix.
    /// After calling this method, you can use the matrix like any other `OMatrix`.
    ///
    /// # Examples
    ///
    /// ## Basic Usage
    ///
    /// ```
    /// use nalgebra::Matrix3;
    /// use nalgebra::debug::RandomSDP;
    ///
    /// // Simple random function
    /// let mut counter = 0.0;
    /// let mut rand_fn = || { counter += 0.1; counter };
    ///
    /// // Generate a random SPD matrix
    /// let generator = RandomSDP::<f64, _>::new(
    ///     nalgebra::Const::<3>,
    ///     &mut rand_fn
    /// );
    ///
    /// // Extract the matrix for use
    /// let matrix = generator.unwrap();
    ///
    /// // Now you can use it like any other matrix
    /// println!("Determinant: {}", matrix.determinant());
    /// ```
    ///
    /// ## Using for Cholesky Decomposition
    ///
    /// ```
    /// use nalgebra::Matrix4;
    /// use nalgebra::debug::RandomSDP;
    ///
    /// // Simple random function
    /// let mut counter = 1.0;
    /// let mut rand_fn = || { counter += 0.2; counter };
    ///
    /// let spd = RandomSDP::<f64, _>::new(
    ///     nalgebra::Const::<4>,
    ///     &mut rand_fn
    /// ).unwrap();
    ///
    /// // Since it's SPD, Cholesky decomposition will always succeed
    /// let cholesky = spd.cholesky().expect("Should succeed for SPD matrix");
    /// ```
    ///
    /// ## Using in Linear Systems
    ///
    /// ```
    /// use nalgebra::{Matrix3, Vector3};
    /// use nalgebra::debug::RandomSDP;
    ///
    /// // Simple random function
    /// let mut counter = 2.0;
    /// let mut rand_fn = || { counter += 0.15; counter };
    ///
    /// let a = RandomSDP::<f64, _>::new(
    ///     nalgebra::Const::<3>,
    ///     &mut rand_fn
    /// ).unwrap();
    ///
    /// // SPD matrices are always invertible
    /// let b = Vector3::new(1.0, 2.0, 3.0);
    /// let x = a.lu().solve(&b).expect("Should have unique solution");
    /// ```
    ///
    /// # See Also
    ///
    /// - [`new`](Self::new): Create a new random SPD matrix generator
    pub fn unwrap(self) -> OMatrix<T, D, D> {
        self.m
    }

    /// Creates a new random symmetric positive-definite matrix generator.
    ///
    /// This function generates a well-conditioned symmetric positive-definite (SPD) matrix
    /// using the eigendecomposition formula: A = Q * D * Q^T, where Q is a random orthogonal
    /// matrix and D is a diagonal matrix with positive eigenvalues.
    ///
    /// The eigenvalues are chosen as `1 + |random_value|`, ensuring they're all at least 1.
    /// This guarantees the matrix is "well-conditioned" (not too close to being singular),
    /// which is important for numerical stability in testing.
    ///
    /// # Parameters
    ///
    /// - `dim`: The dimension of the square matrix to generate. Can be a compile-time
    ///   constant like `Const::<3>` for a 3x3 matrix, or runtime size like `Dyn(n)`.
    /// - `rand`: A closure or function that generates random values of type `T`. This
    ///   function will be called multiple times. For real numbers, values in [0, 1]
    ///   work well (e.g., `rand::random::<f64>()`).
    ///
    /// # Returns
    ///
    /// A `RandomSDP` wrapper containing the generated matrix. Call [`unwrap`](Self::unwrap)
    /// to extract the actual matrix.
    ///
    /// # Examples
    ///
    /// ## Fixed-Size Matrix (Compile-Time Dimensions)
    ///
    /// ```
    /// use nalgebra::Matrix3;
    /// use nalgebra::debug::RandomSDP;
    ///
    /// // Simple random function
    /// let mut counter = 0.5;
    /// let mut rand_fn = || { counter += 0.1; counter };
    ///
    /// // Create a 3x3 SPD matrix
    /// let spd = RandomSDP::<f64, _>::new(
    ///     nalgebra::Const::<3>,
    ///     &mut rand_fn
    /// );
    /// let matrix = spd.unwrap();
    ///
    /// // Verify it's symmetric
    /// let diff = matrix.transpose() - &matrix;
    /// assert!(diff.norm() < 1e-10);
    ///
    /// // Verify it's positive-definite (all eigenvalues > 0)
    /// let eigenvalues = matrix.symmetric_eigen().eigenvalues;
    /// assert!(eigenvalues.iter().all(|&ev| ev > 0.0));
    /// ```
    ///
    /// ## Dynamic-Size Matrix (Runtime Dimensions)
    ///
    /// ```
    /// use nalgebra::DMatrix;
    /// use nalgebra::debug::RandomSDP;
    ///
    /// // Simple random function
    /// let mut counter = 1.0;
    /// let mut rand_fn = || { counter += 0.12; counter };
    ///
    /// // Create an SPD matrix with size determined at runtime
    /// let size = 8;
    /// let spd = RandomSDP::<f64, _>::new(
    ///     nalgebra::Dyn(size),
    ///     &mut rand_fn
    /// );
    /// let matrix = spd.unwrap();
    ///
    /// assert_eq!(matrix.nrows(), size);
    /// assert_eq!(matrix.ncols(), size);
    /// ```
    ///
    /// ## Testing Cholesky Decomposition
    ///
    /// ```
    /// use nalgebra::Matrix4;
    /// use nalgebra::debug::RandomSDP;
    ///
    /// // Simple random function
    /// let mut counter = 2.0;
    /// let mut rand_fn = || { counter += 0.18; counter };
    ///
    /// // Generate test matrix
    /// let original = RandomSDP::<f64, _>::new(
    ///     nalgebra::Const::<4>,
    ///     &mut rand_fn
    /// ).unwrap();
    ///
    /// // Compute Cholesky decomposition (will always succeed for SPD)
    /// let cholesky = original.clone().cholesky()
    ///     .expect("Cholesky should succeed for SPD matrix");
    /// let l = cholesky.l();
    ///
    /// // Verify: A = L * L^T
    /// let reconstructed = l * l.transpose();
    /// assert!((reconstructed - original).norm() < 1e-10);
    /// ```
    ///
    /// ## Using a Deterministic Function for Reproducible Tests
    ///
    /// ```
    /// use nalgebra::Matrix3;
    /// use nalgebra::debug::RandomSDP;
    ///
    /// // Use a deterministic function for reproducible tests
    /// let mut seed = 12345.0;
    /// let mut deterministic_rand = || {
    ///     seed = (seed * 9301.0 + 49297.0) % 233280.0;
    ///     seed / 233280.0
    /// };
    ///
    /// let spd = RandomSDP::<f64, _>::new(
    ///     nalgebra::Const::<3>,
    ///     &mut deterministic_rand
    /// );
    /// let matrix = spd.unwrap();
    ///
    /// // This will generate the same matrix every time with the same seed
    /// ```
    ///
    /// ## Testing Linear System Solvers
    ///
    /// ```
    /// use nalgebra::{Matrix3, Vector3};
    /// use nalgebra::debug::RandomSDP;
    ///
    /// // Simple random function
    /// let mut counter = 4.0;
    /// let mut rand_fn = || { counter += 0.2; counter };
    ///
    /// // Generate a test system
    /// let a = RandomSDP::<f64, _>::new(
    ///     nalgebra::Const::<3>,
    ///     &mut rand_fn
    /// ).unwrap();
    ///
    /// // Create a right-hand side
    /// let b = Vector3::new(1.0, 2.0, 3.0);
    ///
    /// // Solve Ax = b (SPD matrices are always invertible)
    /// let x = a.clone().lu().solve(&b).expect("Should have solution");
    ///
    /// // Verify the solution
    /// let residual = a * x - b;
    /// assert!(residual.norm() < 1e-10);
    /// ```
    ///
    /// ## Testing Eigenvalue Algorithms
    ///
    /// ```
    /// use nalgebra::Matrix3;
    /// use nalgebra::debug::RandomSDP;
    ///
    /// // Simple random function
    /// let mut counter = 5.0;
    /// let mut rand_fn = || { counter += 0.25; counter };
    ///
    /// let spd = RandomSDP::<f64, _>::new(
    ///     nalgebra::Const::<3>,
    ///     &mut rand_fn
    /// ).unwrap();
    ///
    /// // Compute eigendecomposition
    /// let eigen = spd.clone().symmetric_eigen();
    /// let eigenvalues = eigen.eigenvalues;
    /// let eigenvectors = eigen.eigenvectors;
    ///
    /// // For SPD matrices, all eigenvalues are positive
    /// assert!(eigenvalues.iter().all(|&ev| ev > 0.0));
    ///
    /// // Verify: A = V * D * V^T
    /// let d = nalgebra::Matrix3::from_diagonal(&eigenvalues);
    /// let reconstructed = &eigenvectors * d * eigenvectors.transpose();
    /// assert!((reconstructed - spd).norm() < 1e-10);
    /// ```
    ///
    /// # See Also
    ///
    /// - [`unwrap`](Self::unwrap): Extract the generated matrix
    /// - [`RandomOrthogonal::new`](crate::debug::RandomOrthogonal::new): Generate orthogonal matrices (used internally)
    /// - [`Matrix::cholesky`](crate::linalg::Cholesky): Cholesky decomposition for SPD matrices
    /// - [`Matrix::symmetric_eigen`](crate::linalg::SymmetricEigen): Eigendecomposition for symmetric matrices
    pub fn new<Rand: FnMut() -> T>(dim: D, mut rand: Rand) -> Self {
        let mut m = RandomOrthogonal::new(dim, &mut rand).unwrap();
        let mt = m.adjoint();

        for i in 0..dim.value() {
            let mut col = m.column_mut(i);
            let eigenval = T::one() + T::from_real(rand().modulus());
            col *= eigenval;
        }

        RandomSDP { m: m * mt }
    }
}

#[cfg(feature = "arbitrary")]
impl<T: ComplexField + Arbitrary + Send, D: Dim> Arbitrary for RandomSDP<T, D>
where
    DefaultAllocator: Allocator<D, D>,
    Owned<T, D, D>: Clone + Send,
{
    fn arbitrary(g: &mut Gen) -> Self {
        let dim = D::try_to_usize().unwrap_or(1 + usize::arbitrary(g) % 50);
        Self::new(D::from_usize(dim), || T::arbitrary(g))
    }
}
