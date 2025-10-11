#[cfg(feature = "arbitrary")]
use crate::base::storage::Owned;
#[cfg(feature = "arbitrary")]
use quickcheck::{Arbitrary, Gen};

use crate::base::Scalar;
use crate::base::allocator::Allocator;
use crate::base::dimension::{Dim, Dyn};
use crate::base::{DefaultAllocator, OMatrix};
use crate::linalg::givens::GivensRotation;
use simba::scalar::ComplexField;

/// A generator for random orthogonal matrices.
///
/// An orthogonal matrix is a square matrix whose columns (and rows) are orthonormal vectors.
/// This means:
/// - Each column has length 1 (unit vector)
/// - Every pair of columns is perpendicular (dot product = 0)
/// - The matrix satisfies: Q^T * Q = I (transpose times itself equals identity)
///
/// # Mathematical Properties
///
/// For an orthogonal matrix Q:
/// - `Q^T * Q = I` (transpose equals inverse)
/// - `det(Q) = ±1` (determinant is 1 or -1)
/// - Preserves lengths: `||Qv|| = ||v||` for any vector v
/// - Preserves angles between vectors
///
/// # Why Use Orthogonal Matrices?
///
/// Orthogonal matrices are fundamental in numerical linear algebra:
/// - They represent rotations and reflections without scaling
/// - They're numerically stable (don't amplify errors)
/// - They appear in QR decomposition, SVD, and eigenvalue algorithms
/// - Perfect for testing transformation and decomposition code
///
/// # Generation Method
///
/// This generator creates orthogonal matrices by composing random Givens rotations.
/// Each Givens rotation is a simple 2D rotation, and composing many of them creates
/// a uniformly distributed random orthogonal matrix.
///
/// # Examples
///
/// ## Basic Usage with Fixed-Size Matrix
///
/// ```
/// use nalgebra::Matrix3;
/// use nalgebra::debug::RandomOrthogonal;
///
/// // Simple counter-based random function for the example
/// let mut counter = 0.0_f64;
/// let mut rand_fn = || { counter += 0.1; counter.sin() };
///
/// // Generate a random 3x3 orthogonal matrix
/// let orthogonal = RandomOrthogonal::<f64, _>::new(
///     nalgebra::Const::<3>,
///     &mut rand_fn
/// );
/// let q = orthogonal.unwrap();
///
/// // Verify orthogonality: Q^T * Q should equal identity
/// let identity = q.transpose() * &q;
/// let identity_check = Matrix3::identity();
///
/// // Check that we're very close to identity (accounting for floating point errors)
/// assert!((identity - identity_check).norm() < 1e-10);
/// ```
///
/// ## Testing a Rotation Algorithm
///
/// ```
/// use nalgebra::{Vector3, Matrix3};
/// use nalgebra::debug::RandomOrthogonal;
///
/// // Simple random function
/// let mut counter = 1.0_f64;
/// let mut rand_fn = || { counter += 0.15; counter.cos() };
///
/// // Generate a random rotation matrix
/// let rotation = RandomOrthogonal::<f64, _>::new(
///     nalgebra::Const::<3>,
///     &mut rand_fn
/// ).unwrap();
///
/// // Create a test vector
/// let v = Vector3::new(1.0, 2.0, 3.0);
/// let original_length = v.norm();
///
/// // Apply rotation
/// let rotated = rotation * v;
///
/// // Verify that rotation preserves length
/// assert!((rotated.norm() - original_length).abs() < 1e-10);
/// ```
///
/// ## Dynamic-Size Matrix for Testing
///
/// ```
/// use nalgebra::DMatrix;
/// use nalgebra::debug::RandomOrthogonal;
///
/// // Simple random function
/// let mut counter = 2.0_f64;
/// let mut rand_fn = || { counter += 0.2; (counter * 0.7).sin() };
///
/// // Generate a random orthogonal matrix of any size
/// let dim = 5;
/// let orthogonal = RandomOrthogonal::<f64, _>::new(
///     nalgebra::Dyn(dim),
///     &mut rand_fn
/// );
/// let q = orthogonal.unwrap();
///
/// // Verify it's square
/// assert_eq!(q.nrows(), dim);
/// assert_eq!(q.ncols(), dim);
///
/// // Verify orthogonality
/// let product = q.transpose() * &q;
/// let identity = DMatrix::identity(dim, dim);
/// assert!((product - identity).norm() < 1e-9);
/// ```
///
/// # See Also
///
/// - [`RandomSDP`](crate::debug::RandomSDP): Generate random symmetric positive-definite matrices
/// - [`GivensRotation`](crate::linalg::givens::GivensRotation): The building block used internally
#[derive(Clone, Debug)]
pub struct RandomOrthogonal<T: Scalar, D: Dim = Dyn>
where
    DefaultAllocator: Allocator<D, D>,
{
    m: OMatrix<T, D, D>,
}

impl<T: ComplexField, D: Dim> RandomOrthogonal<T, D>
where
    DefaultAllocator: Allocator<D, D>,
{
    /// Extracts the generated orthogonal matrix.
    ///
    /// This consumes the `RandomOrthogonal` wrapper and returns the underlying matrix.
    /// After calling this method, you can use the matrix like any other `OMatrix`.
    ///
    /// # Examples
    ///
    /// ## Basic Usage
    ///
    /// ```
    /// use nalgebra::Matrix3;
    /// use nalgebra::debug::RandomOrthogonal;
    ///
    /// // Simple random function
    /// let mut counter = 0.0_f64;
    /// let mut rand_fn = || { counter += 0.1; counter.sin() };
    ///
    /// // Generate a random orthogonal matrix
    /// let generator = RandomOrthogonal::<f64, _>::new(
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
    /// ## Using in Computations
    ///
    /// ```
    /// use nalgebra::{Matrix3, Vector3};
    /// use nalgebra::debug::RandomOrthogonal;
    ///
    /// // Simple random function
    /// let mut counter = 1.0_f64;
    /// let mut rand_fn = || { counter += 0.1; counter.cos() };
    ///
    /// let q = RandomOrthogonal::<f64, _>::new(
    ///     nalgebra::Const::<3>,
    ///     &mut rand_fn
    /// ).unwrap();
    ///
    /// let v = Vector3::new(1.0, 0.0, 0.0);
    /// let rotated = q * v;
    ///
    /// // The length is preserved after rotation
    /// assert!((rotated.norm() - v.norm()).abs() < 1e-10);
    /// ```
    ///
    /// # See Also
    ///
    /// - [`new`](Self::new): Create a new random orthogonal matrix generator
    pub fn unwrap(self) -> OMatrix<T, D, D> {
        self.m
    }

    /// Creates a new random orthogonal matrix generator.
    ///
    /// This function generates a random orthogonal matrix by composing multiple random
    /// Givens rotations. A Givens rotation is a simple 2D rotation, and by applying many
    /// of them, we build up a full orthogonal matrix with random orientation.
    ///
    /// # Parameters
    ///
    /// - `dim`: The dimension of the square matrix to generate. Can be a compile-time
    ///   constant like `Const::<3>` for a 3x3 matrix, or runtime size like `Dyn(n)`.
    /// - `rand`: A closure or function that generates random values of type `T`. This
    ///   function will be called multiple times to generate the random Givens rotations.
    ///   For real numbers, values in the range [-1, 1] work well.
    ///
    /// # Returns
    ///
    /// A `RandomOrthogonal` wrapper containing the generated matrix. Call [`unwrap`](Self::unwrap)
    /// to extract the actual matrix.
    ///
    /// # Examples
    ///
    /// ## Fixed-Size Matrix (Compile-Time Dimensions)
    ///
    /// ```
    /// use nalgebra::Matrix4;
    /// use nalgebra::debug::RandomOrthogonal;
    ///
    /// // Simple random function
    /// let mut counter = 0.5_f64;
    /// let mut rand_fn = || { counter += 0.13; (counter * 1.3).sin() };
    ///
    /// // Create a 4x4 orthogonal matrix
    /// let orthogonal = RandomOrthogonal::<f64, _>::new(
    ///     nalgebra::Const::<4>,
    ///     &mut rand_fn
    /// );
    /// let matrix = orthogonal.unwrap();
    ///
    /// // Check the determinant (should be ±1)
    /// let det = matrix.determinant();
    /// assert!((det.abs() - 1.0).abs() < 1e-10);
    /// ```
    ///
    /// ## Dynamic-Size Matrix (Runtime Dimensions)
    ///
    /// ```
    /// use nalgebra::DMatrix;
    /// use nalgebra::debug::RandomOrthogonal;
    ///
    /// // Simple random function
    /// let mut counter = 3.0_f64;
    /// let mut rand_fn = || { counter += 0.17; counter.cos() };
    ///
    /// // Create an orthogonal matrix with size determined at runtime
    /// let size = 6;
    /// let orthogonal = RandomOrthogonal::<f64, _>::new(
    ///     nalgebra::Dyn(size),
    ///     &mut rand_fn
    /// );
    /// let q = orthogonal.unwrap();
    ///
    /// assert_eq!(q.nrows(), size);
    /// assert_eq!(q.ncols(), size);
    /// ```
    ///
    /// ## Using a Deterministic Random Function
    ///
    /// ```
    /// use nalgebra::Matrix3;
    /// use nalgebra::debug::RandomOrthogonal;
    ///
    /// // Use a deterministic function for reproducible tests
    /// let mut seed = 42.0;
    /// let mut deterministic_rand = || {
    ///     seed = (seed * 9301.0 + 49297.0) % 233280.0;
    ///     (seed / 233280.0) * 2.0 - 1.0
    /// };
    ///
    /// let orthogonal = RandomOrthogonal::<f64, _>::new(
    ///     nalgebra::Const::<3>,
    ///     &mut deterministic_rand
    /// );
    /// let matrix = orthogonal.unwrap();
    ///
    /// // This will generate the same matrix every time with the same seed
    /// ```
    ///
    /// ## Testing QR Decomposition
    ///
    /// ```
    /// use nalgebra::Matrix3;
    /// use nalgebra::debug::RandomOrthogonal;
    ///
    /// // Simple random function
    /// let mut counter = 7.0_f64;
    /// let mut rand_fn = || { counter += 0.21; (counter * 0.5).sin() };
    ///
    /// // Generate a test matrix
    /// let q_original = RandomOrthogonal::<f64, _>::new(
    ///     nalgebra::Const::<3>,
    ///     &mut rand_fn
    /// ).unwrap();
    ///
    /// // QR decompose it
    /// let qr = q_original.qr();
    /// let q = qr.q();
    ///
    /// // Q from QR decomposition should also be orthogonal
    /// let identity = q.transpose() * q;
    /// assert!((identity - Matrix3::identity()).norm() < 1e-10);
    /// ```
    ///
    /// # See Also
    ///
    /// - [`unwrap`](Self::unwrap): Extract the generated matrix
    /// - [`RandomSDP::new`](crate::debug::RandomSDP::new): Generate symmetric positive-definite matrices
    /// - [`GivensRotation`](crate::linalg::givens::GivensRotation): The rotation primitive used internally
    pub fn new<Rand: FnMut() -> T>(dim: D, mut rand: Rand) -> Self {
        let mut res = OMatrix::identity_generic(dim, dim);

        // Create an orthogonal matrix by composing random Givens rotations rotations.
        for i in 0..dim.value() - 1 {
            let rot = GivensRotation::new(rand(), rand()).0;
            rot.rotate(&mut res.fixed_rows_mut::<2>(i));
        }

        RandomOrthogonal { m: res }
    }
}

#[cfg(feature = "arbitrary")]
impl<T: ComplexField + Arbitrary + Send, D: Dim> Arbitrary for RandomOrthogonal<T, D>
where
    DefaultAllocator: Allocator<D, D>,
    Owned<T, D, D>: Clone + Send,
{
    fn arbitrary(g: &mut Gen) -> Self {
        let dim = D::try_to_usize().unwrap_or(1 + usize::arbitrary(g) % 50);
        Self::new(D::from_usize(dim), || T::arbitrary(g))
    }
}
