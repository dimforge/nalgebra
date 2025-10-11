#[cfg(feature = "serde-serialize-no-std")]
use serde::{Deserialize, Serialize};

use num::{One, Zero};
use simba::scalar::ComplexField;
use simba::simd::SimdComplexField;

use crate::allocator::Allocator;
use crate::base::{Const, DefaultAllocator, Matrix, OMatrix, Vector};
use crate::constraint::{SameNumberOfRows, ShapeConstraint};
use crate::dimension::{Dim, DimAdd, DimDiff, DimSub, DimSum, U1};
use crate::storage::{Storage, StorageMut};

/// The Cholesky decomposition of a symmetric positive-definite matrix.
///
/// # What is Cholesky Decomposition?
///
/// The Cholesky decomposition is a way to factorize a symmetric positive-definite matrix `M`
/// into the product of a lower triangular matrix `L` and its conjugate transpose:
///
/// ```text
/// M = L * L^H
/// ```
///
/// where `L^H` denotes the conjugate transpose (for real matrices, this is just the transpose).
///
/// # When to Use It?
///
/// This decomposition is useful for:
/// - Solving systems of linear equations efficiently
/// - Computing matrix inversions
/// - Numerical optimization (e.g., in machine learning, statistics)
/// - Generating correlated random variables
/// - Testing if a matrix is positive-definite
///
/// # Requirements
///
/// The input matrix must be:
/// - **Symmetric**: `M = M^T` (or `M = M^H` for complex matrices)
/// - **Positive-definite**: All eigenvalues are positive, or equivalently, `x^T M x > 0`
///   for all non-zero vectors `x`
///
/// Common positive-definite matrices include covariance matrices, Gram matrices, and
/// the Hessian matrices of strictly convex functions at their minima.
#[cfg_attr(feature = "serde-serialize-no-std", derive(Serialize, Deserialize))]
#[cfg_attr(
    feature = "serde-serialize-no-std",
    serde(bound(serialize = "DefaultAllocator: Allocator<D>,
         OMatrix<T, D, D>: Serialize"))
)]
#[cfg_attr(
    feature = "serde-serialize-no-std",
    serde(bound(deserialize = "DefaultAllocator: Allocator<D>,
         OMatrix<T, D, D>: Deserialize<'de>"))
)]
#[cfg_attr(feature = "defmt", derive(defmt::Format))]
#[derive(Clone, Debug)]
pub struct Cholesky<T: SimdComplexField, D: Dim>
where
    DefaultAllocator: Allocator<D, D>,
{
    chol: OMatrix<T, D, D>,
}

impl<T: SimdComplexField, D: Dim> Copy for Cholesky<T, D>
where
    DefaultAllocator: Allocator<D, D>,
    OMatrix<T, D, D>: Copy,
{
}

impl<T: SimdComplexField, D: Dim> Cholesky<T, D>
where
    DefaultAllocator: Allocator<D, D>,
{
    /// Computes the Cholesky decomposition of `matrix` without checking that the matrix is positive-definite.
    ///
    /// # Warning
    ///
    /// This is an **unsafe** operation in the logical sense: it assumes the input matrix is
    /// symmetric and positive-definite **without verification**. If the input matrix does not
    /// meet these requirements, the decomposition may contain invalid values (Inf, NaN, etc.),
    /// leading to incorrect results in subsequent computations.
    ///
    /// **Use this only when you are certain the matrix is positive-definite** and want to
    /// skip the validity checks for performance reasons. For general use, prefer [`new`](Self::new).
    ///
    /// # Arguments
    ///
    /// * `matrix` - A symmetric positive-definite matrix. Only the lower-triangular part is read.
    ///
    /// # Returns
    ///
    /// The Cholesky decomposition `L` such that `matrix = L * L^T`.
    ///
    /// # Example
    ///
    /// ```
    /// use nalgebra::{Matrix3, linalg::Cholesky};
    ///
    /// // A known positive-definite matrix (diagonal with positive entries)
    /// let m = Matrix3::new(
    ///     4.0, 0.0, 0.0,
    ///     0.0, 9.0, 0.0,
    ///     0.0, 0.0, 16.0,
    /// );
    ///
    /// // We know it's positive-definite, so we can use new_unchecked
    /// let cholesky = Cholesky::new_unchecked(m);
    /// let l = cholesky.l();
    ///
    /// // Verify: L * L^T should equal the original matrix
    /// let reconstructed = &l * l.transpose();
    /// assert!((reconstructed - m).norm() < 1e-10);
    /// ```
    ///
    /// # See Also
    ///
    /// * [`new`](Self::new) - Safe version that validates the matrix is positive-definite
    /// * [`new_with_substitute`](Self::new_with_substitute) - Handles numerical errors gracefully
    pub fn new_unchecked(mut matrix: OMatrix<T, D, D>) -> Self {
        assert!(matrix.is_square(), "The input matrix must be square.");

        let n = matrix.nrows();

        for j in 0..n {
            for k in 0..j {
                let factor = unsafe { -matrix.get_unchecked((j, k)).clone() };

                let (mut col_j, col_k) = matrix.columns_range_pair_mut(j, k);
                let mut col_j = col_j.rows_range_mut(j..);
                let col_k = col_k.rows_range(j..);
                col_j.axpy(factor.simd_conjugate(), &col_k, T::one());
            }

            let diag = unsafe { matrix.get_unchecked((j, j)).clone() };
            let denom = diag.simd_sqrt();

            unsafe {
                *matrix.get_unchecked_mut((j, j)) = denom.clone();
            }

            let mut col = matrix.view_range_mut(j + 1.., j);
            col /= denom;
        }

        Cholesky { chol: matrix }
    }

    /// Uses the given matrix as-is without any checks or modifications as the
    /// Cholesky decomposition.
    ///
    /// # Warning
    ///
    /// This function performs **no validation whatsoever**. It directly wraps the provided
    /// matrix as a Cholesky decomposition without checking if it's actually a valid lower
    /// triangular factor. Using this incorrectly will lead to completely invalid results.
    ///
    /// **It is entirely the user's responsibility to ensure all invariants hold:**
    /// - The matrix must be lower triangular (upper part is ignored)
    /// - It must be a valid Cholesky factor (i.e., `L * L^T` produces a positive-definite matrix)
    ///
    /// This function is primarily useful for:
    /// - Deserializing a pre-computed Cholesky decomposition
    /// - Advanced use cases where you've computed the decomposition through other means
    ///
    /// # Arguments
    ///
    /// * `matrix` - A matrix to treat as a Cholesky decomposition factor
    ///
    /// # Example
    ///
    /// ```
    /// use nalgebra::{Matrix3, linalg::Cholesky};
    ///
    /// // Manually create a lower triangular matrix that we know is a valid Cholesky factor
    /// let l = Matrix3::new(
    ///     2.0, 0.0, 0.0,
    ///     1.0, 3.0, 0.0,
    ///     2.0, 1.0, 4.0,
    /// );
    ///
    /// // Wrap it directly as a Cholesky decomposition
    /// let cholesky = Cholesky::pack_dirty(l.clone());
    ///
    /// // Verify it represents M = L * L^T
    /// let m = &l * l.transpose();
    /// let reconstructed = cholesky.l() * cholesky.l().transpose();
    /// assert!((reconstructed - m).norm() < 1e-10);
    /// ```
    ///
    /// # See Also
    ///
    /// * [`new`](Self::new) - Safe construction from a positive-definite matrix
    /// * [`new_unchecked`](Self::new_unchecked) - Fast construction, skips positive-definite checks
    /// * [`unpack_dirty`](Self::unpack_dirty) - Retrieves the matrix without zeroing upper triangle
    pub const fn pack_dirty(matrix: OMatrix<T, D, D>) -> Self {
        Cholesky { chol: matrix }
    }

    /// Retrieves the lower-triangular factor of the Cholesky decomposition with its strictly
    /// upper-triangular part filled with zeros.
    ///
    /// This method consumes the decomposition and returns the lower triangular matrix `L`
    /// such that `M = L * L^T` (or `M = L * L^H` for complex matrices), where `M` is the
    /// original matrix that was decomposed.
    ///
    /// The returned matrix has zeros in all positions above the main diagonal.
    ///
    /// # Returns
    ///
    /// The lower triangular factor `L` with the strictly upper triangular part zeroed out.
    ///
    /// # Example
    ///
    /// ```
    /// use nalgebra::{Matrix3, linalg::Cholesky};
    ///
    /// let m = Matrix3::new(
    ///     4.0, 2.0, 1.0,
    ///     2.0, 5.0, 3.0,
    ///     1.0, 3.0, 6.0,
    /// );
    ///
    /// let cholesky = Cholesky::new(m.clone()).unwrap();
    /// let l = cholesky.unpack();
    ///
    /// // L is lower triangular (upper part is zero)
    /// assert_eq!(l[(0, 1)], 0.0);
    /// assert_eq!(l[(0, 2)], 0.0);
    /// assert_eq!(l[(1, 2)], 0.0);
    ///
    /// // Verify: L * L^T reconstructs the original matrix
    /// let reconstructed = l.clone() * l.transpose();
    /// assert!((reconstructed - m).norm() < 1e-10);
    /// ```
    ///
    /// # See Also
    ///
    /// * [`l`](Self::l) - Non-consuming version that returns a copy
    /// * [`unpack_dirty`](Self::unpack_dirty) - Faster version that doesn't zero the upper triangle
    /// * [`l_dirty`](Self::l_dirty) - Returns a reference without zeroing the upper triangle
    pub fn unpack(mut self) -> OMatrix<T, D, D> {
        self.chol.fill_upper_triangle(T::zero(), 1);
        self.chol
    }

    /// Retrieves the lower-triangular factor of the Cholesky decomposition, without zeroing-out
    /// its strict upper-triangular part.
    ///
    /// This method is faster than [`unpack`](Self::unpack) because it avoids the overhead of
    /// filling the upper triangle with zeros. However, the values in the strictly upper-triangular
    /// part are **undefined garbage** and must be ignored in subsequent computations.
    ///
    /// **Use this only when:**
    /// - You need maximum performance
    /// - You will only access the lower triangular part
    /// - You're passing the result to functions that only read the lower triangle
    ///
    /// # Returns
    ///
    /// The lower triangular factor `L` where only the lower triangle and diagonal contain
    /// valid data. The upper triangle contains undefined values.
    ///
    /// # Example
    ///
    /// ```
    /// use nalgebra::{Matrix3, linalg::Cholesky};
    ///
    /// let m = Matrix3::new(
    ///     4.0, 2.0, 1.0,
    ///     2.0, 5.0, 3.0,
    ///     1.0, 3.0, 6.0,
    /// );
    ///
    /// let cholesky = Cholesky::new(m.clone()).unwrap();
    /// let l_dirty = cholesky.unpack_dirty();
    ///
    /// // Only access the lower triangle - upper triangle contains garbage!
    /// // Extract lower triangle manually
    /// let l = l_dirty.lower_triangle();
    ///
    /// // Now we can safely use L
    /// let reconstructed = &l * l.transpose();
    /// assert!((reconstructed - m).norm() < 1e-10);
    /// ```
    ///
    /// # See Also
    ///
    /// * [`unpack`](Self::unpack) - Safe version that zeros the upper triangle
    /// * [`l_dirty`](Self::l_dirty) - Non-consuming version returning a reference
    /// * [`l`](Self::l) - Non-consuming version with zeroed upper triangle
    pub fn unpack_dirty(self) -> OMatrix<T, D, D> {
        self.chol
    }

    /// Retrieves the lower-triangular factor of the Cholesky decomposition with its strictly
    /// upper-triangular part filled with zeros.
    ///
    /// This method returns a **copy** of the lower triangular matrix `L` such that
    /// `M = L * L^T` (or `M = L * L^H` for complex matrices), where `M` is the original
    /// matrix that was decomposed. All elements above the main diagonal are set to zero.
    ///
    /// Unlike [`unpack`](Self::unpack), this method does not consume the decomposition,
    /// allowing you to continue using it afterwards.
    ///
    /// # Returns
    ///
    /// A copy of the lower triangular factor with the upper triangle zeroed.
    ///
    /// # Example
    ///
    /// ```
    /// use nalgebra::{Matrix3, linalg::Cholesky};
    ///
    /// let m = Matrix3::new(
    ///     4.0, 2.0, 1.0,
    ///     2.0, 5.0, 3.0,
    ///     1.0, 3.0, 6.0,
    /// );
    ///
    /// let cholesky = Cholesky::new(m.clone()).unwrap();
    ///
    /// // Get the lower triangular factor (decomposition is not consumed)
    /// let l = cholesky.l();
    ///
    /// // We can still use the decomposition
    /// let det = cholesky.determinant();
    /// println!("Determinant: {}", det);
    ///
    /// // Verify: L * L^T = M
    /// let reconstructed = &l * l.transpose();
    /// assert!((reconstructed - m).norm() < 1e-10);
    /// ```
    ///
    /// # See Also
    ///
    /// * [`unpack`](Self::unpack) - Consuming version that returns the factor
    /// * [`l_dirty`](Self::l_dirty) - Faster version that returns a reference (upper triangle not zeroed)
    /// * [`unpack_dirty`](Self::unpack_dirty) - Consuming version without zeroing upper triangle
    #[must_use]
    pub fn l(&self) -> OMatrix<T, D, D> {
        self.chol.lower_triangle()
    }

    /// Retrieves the lower-triangular factor of the Cholesky decomposition, without zeroing-out
    /// its strict upper-triangular part.
    ///
    /// This is the most efficient way to access the Cholesky factor: it returns a **reference**
    /// to the internal matrix without any allocation or copying. However, the strictly
    /// upper-triangular part contains **undefined garbage values** that must be ignored.
    ///
    /// **Use this when:**
    /// - Performance is critical and you want to avoid allocations
    /// - You only need to read the lower triangular part
    /// - You're passing the matrix to methods that only access the lower triangle
    ///
    /// # Returns
    ///
    /// A reference to the internal matrix. Only the lower triangle and diagonal are valid;
    /// the strictly upper triangle contains undefined values.
    ///
    /// # Example
    ///
    /// ```
    /// use nalgebra::{Matrix3, linalg::Cholesky};
    ///
    /// let m = Matrix3::new(
    ///     4.0, 2.0, 1.0,
    ///     2.0, 5.0, 3.0,
    ///     1.0, 3.0, 6.0,
    /// );
    ///
    /// let cholesky = Cholesky::new(m).unwrap();
    ///
    /// // Get a reference to the factor (no allocation!)
    /// let l_ref = cholesky.l_dirty();
    ///
    /// // Access only the lower triangular elements
    /// println!("L[0,0] = {}", l_ref[(0, 0)]);
    /// println!("L[1,0] = {}", l_ref[(1, 0)]);
    /// println!("L[2,1] = {}", l_ref[(2, 1)]);
    ///
    /// // Don't access the upper triangle - it contains garbage!
    /// // BAD: let bad_value = l_ref[(0, 1)];
    ///
    /// // If you need the full lower triangular matrix, extract it:
    /// let l = l_ref.lower_triangle();
    /// ```
    ///
    /// # See Also
    ///
    /// * [`l`](Self::l) - Returns a copy with the upper triangle zeroed
    /// * [`unpack_dirty`](Self::unpack_dirty) - Consuming version
    /// * [`unpack`](Self::unpack) - Consuming version with zeroed upper triangle
    #[must_use]
    pub const fn l_dirty(&self) -> &OMatrix<T, D, D> {
        &self.chol
    }

    /// Solves the system `M * x = b` where `M` is the decomposed matrix and `x` is the unknown.
    ///
    /// This method solves the linear system **in-place**, storing the solution directly in `b`.
    /// It's more efficient than [`solve`](Self::solve) because it doesn't allocate a new matrix.
    ///
    /// # How It Works
    ///
    /// Given the Cholesky decomposition `M = L * L^T`, solving `M * x = b` is done in two steps:
    /// 1. Solve `L * y = b` for `y` (forward substitution)
    /// 2. Solve `L^T * x = y` for `x` (backward substitution)
    ///
    /// This is much faster than directly inverting `M`.
    ///
    /// # Arguments
    ///
    /// * `b` - On input, contains the right-hand side vector(s). On output, contains the
    ///         solution vector(s). Can be a vector or a matrix (to solve multiple systems at once).
    ///
    /// # Example: Solving a single system
    ///
    /// ```
    /// use nalgebra::{Matrix3, Vector3, linalg::Cholesky};
    ///
    /// // Positive-definite matrix
    /// let m = Matrix3::new(
    ///     4.0, 2.0, 1.0,
    ///     2.0, 5.0, 3.0,
    ///     1.0, 3.0, 6.0,
    /// );
    ///
    /// // Right-hand side
    /// let mut b = Vector3::new(1.0, 2.0, 3.0);
    ///
    /// let cholesky = Cholesky::new(m.clone()).unwrap();
    /// cholesky.solve_mut(&mut b);
    ///
    /// // Verify: M * x = original b
    /// let original_b = Vector3::new(1.0, 2.0, 3.0);
    /// let result = m * b;
    /// assert!((result - original_b).norm() < 1e-10);
    /// ```
    ///
    /// # Example: Solving multiple systems at once
    ///
    /// ```
    /// use nalgebra::{Matrix3, linalg::Cholesky};
    ///
    /// let m = Matrix3::new(
    ///     4.0, 2.0, 1.0,
    ///     2.0, 5.0, 3.0,
    ///     1.0, 3.0, 6.0,
    /// );
    ///
    /// // Multiple right-hand sides (as columns)
    /// let mut b = Matrix3::new(
    ///     1.0, 0.0, 0.0,
    ///     0.0, 1.0, 0.0,
    ///     0.0, 0.0, 1.0,
    /// );
    ///
    /// let cholesky = Cholesky::new(m.clone()).unwrap();
    /// cholesky.solve_mut(&mut b);
    ///
    /// // b now contains the inverse of m
    /// let identity = &m * &b;
    /// assert!((identity - Matrix3::identity()).norm() < 1e-10);
    /// ```
    ///
    /// # See Also
    ///
    /// * [`solve`](Self::solve) - Non-mutating version that returns a new matrix
    /// * [`inverse`](Self::inverse) - Computes the matrix inverse directly
    pub fn solve_mut<R2: Dim, C2: Dim, S2>(&self, b: &mut Matrix<T, R2, C2, S2>)
    where
        S2: StorageMut<T, R2, C2>,
        ShapeConstraint: SameNumberOfRows<R2, D>,
    {
        self.chol.solve_lower_triangular_unchecked_mut(b);
        self.chol.ad_solve_lower_triangular_unchecked_mut(b);
    }

    /// Returns the solution of the system `M * x = b` where `M` is the decomposed matrix and
    /// `x` is the unknown.
    ///
    /// This method creates a copy of `b` and solves the system, returning the solution as a
    /// new matrix. For better performance when you can modify `b` in-place, use
    /// [`solve_mut`](Self::solve_mut) instead.
    ///
    /// # How It Works
    ///
    /// Given the Cholesky decomposition `M = L * L^T`, solving `M * x = b` is accomplished by:
    /// 1. Solving `L * y = b` for `y` (forward substitution)
    /// 2. Solving `L^T * x = y` for `x` (backward substitution)
    ///
    /// This two-step approach is significantly more efficient than computing the inverse
    /// and multiplying: `O(n^2)` vs `O(n^3)` for an n×n system.
    ///
    /// # Arguments
    ///
    /// * `b` - The right-hand side vector(s). Can be a vector or a matrix of multiple
    ///         right-hand sides (each column is solved independently).
    ///
    /// # Returns
    ///
    /// The solution vector(s) `x` such that `M * x = b`.
    ///
    /// # Example: Basic linear system
    ///
    /// ```
    /// use nalgebra::{Matrix3, Vector3, linalg::Cholesky};
    ///
    /// let m = Matrix3::new(
    ///     4.0, 2.0, 1.0,
    ///     2.0, 5.0, 3.0,
    ///     1.0, 3.0, 6.0,
    /// );
    ///
    /// let b = Vector3::new(7.0, 14.0, 15.0);
    ///
    /// let cholesky = Cholesky::new(m.clone()).unwrap();
    /// let x = cholesky.solve(&b);
    ///
    /// // Verify the solution: M * x should equal b
    /// let result = m * x;
    /// assert!((result - b).norm() < 1e-10);
    /// ```
    ///
    /// # Example: Application in least squares
    ///
    /// ```
    /// use nalgebra::{Matrix3, Vector3, linalg::Cholesky};
    ///
    /// // Solving the normal equations: A^T * A * x = A^T * b
    /// // This is used in least squares fitting
    /// let a = Matrix3::new(
    ///     1.0, 1.0, 1.0,
    ///     1.0, 2.0, 4.0,
    ///     1.0, 3.0, 9.0,
    /// );
    /// let b = Vector3::new(2.1, 3.9, 7.8);
    ///
    /// // Form the normal equations matrix (always positive-definite for full-rank A)
    /// let ata = a.transpose() * &a;
    /// let atb = a.transpose() * b;
    ///
    /// // Solve using Cholesky
    /// let cholesky = Cholesky::new(ata).unwrap();
    /// let x = cholesky.solve(&atb);
    ///
    /// println!("Least squares solution: {:?}", x);
    /// ```
    ///
    /// # See Also
    ///
    /// * [`solve_mut`](Self::solve_mut) - In-place version for better performance
    /// * [`inverse`](Self::inverse) - Computes the full matrix inverse
    #[must_use = "Did you mean to use solve_mut()?"]
    pub fn solve<R2: Dim, C2: Dim, S2>(&self, b: &Matrix<T, R2, C2, S2>) -> OMatrix<T, R2, C2>
    where
        S2: Storage<T, R2, C2>,
        DefaultAllocator: Allocator<R2, C2>,
        ShapeConstraint: SameNumberOfRows<R2, D>,
    {
        let mut res = b.clone_owned();
        self.solve_mut(&mut res);
        res
    }

    /// Computes the inverse of the decomposed matrix.
    ///
    /// This method efficiently computes the inverse of the original positive-definite matrix `M`
    /// using its Cholesky decomposition. It's more numerically stable and efficient than
    /// general matrix inversion for positive-definite matrices.
    ///
    /// # How It Works
    ///
    /// Given `M = L * L^T`, the inverse is computed by solving `M * M^(-1) = I`, which is
    /// equivalent to solving `n` systems of equations (one for each column of the identity matrix).
    ///
    /// # Returns
    ///
    /// The inverse matrix `M^(-1)` of the original decomposed matrix.
    ///
    /// # Example: Basic matrix inversion
    ///
    /// ```
    /// use nalgebra::{Matrix3, linalg::Cholesky};
    ///
    /// let m = Matrix3::new(
    ///     4.0, 2.0, 1.0,
    ///     2.0, 5.0, 3.0,
    ///     1.0, 3.0, 6.0,
    /// );
    ///
    /// let cholesky = Cholesky::new(m.clone()).unwrap();
    /// let m_inv = cholesky.inverse();
    ///
    /// // Verify: M * M^(-1) = I
    /// let identity = m * m_inv;
    /// assert!((identity - Matrix3::identity()).norm() < 1e-10);
    /// ```
    ///
    /// # Example: Computing inverse of a covariance matrix
    ///
    /// ```
    /// use nalgebra::{Matrix2, linalg::Cholesky};
    ///
    /// // Covariance matrix (always positive-definite)
    /// let cov = Matrix2::new(
    ///     2.0, 0.8,
    ///     0.8, 1.5,
    /// );
    ///
    /// // Efficiently compute the precision matrix (inverse covariance)
    /// let cholesky = Cholesky::new(cov.clone()).unwrap();
    /// let precision = cholesky.inverse();
    ///
    /// // The precision matrix is useful in statistics and machine learning
    /// println!("Precision matrix:\n{}", precision);
    ///
    /// // Verify it's the inverse
    /// assert!((cov * precision - Matrix2::identity()).norm() < 1e-10);
    /// ```
    ///
    /// # Performance Note
    ///
    /// Computing the full inverse is `O(n^3)`. If you only need to solve `M * x = b`,
    /// use [`solve`](Self::solve) instead, which is `O(n^2)` and more numerically stable.
    ///
    /// # See Also
    ///
    /// * [`solve`](Self::solve) - Solve linear systems without computing the full inverse
    /// * [`solve_mut`](Self::solve_mut) - In-place version of solve
    /// * [`determinant`](Self::determinant) - Compute the determinant efficiently
    #[must_use]
    pub fn inverse(&self) -> OMatrix<T, D, D> {
        let shape = self.chol.shape_generic();
        let mut res = OMatrix::identity_generic(shape.0, shape.1);

        self.solve_mut(&mut res);
        res
    }

    /// Computes the determinant of the decomposed matrix.
    ///
    /// For a positive-definite matrix with Cholesky decomposition `M = L * L^T`, the
    /// determinant is computed very efficiently as the square of the product of the
    /// diagonal elements of `L`:
    ///
    /// ```text
    /// det(M) = det(L * L^T) = det(L)^2 = (∏ L[i,i])^2
    /// ```
    ///
    /// This is much more efficient and numerically stable than computing the determinant
    /// directly from the matrix.
    ///
    /// # Returns
    ///
    /// The determinant of the original matrix (always positive for positive-definite matrices).
    ///
    /// # Example: Basic determinant computation
    ///
    /// ```
    /// use nalgebra::{Matrix3, linalg::Cholesky};
    ///
    /// let m = Matrix3::new(
    ///     4.0, 2.0, 1.0,
    ///     2.0, 5.0, 3.0,
    ///     1.0, 3.0, 6.0,
    /// );
    ///
    /// let cholesky = Cholesky::new(m.clone()).unwrap();
    /// let det: f64 = cholesky.determinant();
    ///
    /// println!("Determinant: {}", det);
    ///
    /// // For a positive-definite matrix, the determinant is always positive
    /// assert!(det > 0.0);
    ///
    /// // Verify against the matrix's determinant method
    /// let m_det = m.determinant();
    /// assert!((det - m_det).abs() < 1e-10);
    /// ```
    ///
    /// # Example: Using determinant in statistics
    ///
    /// ```
    /// use nalgebra::{Matrix2, linalg::Cholesky};
    ///
    /// // Covariance matrix from a 2D Gaussian distribution
    /// let cov = Matrix2::new(
    ///     2.0, 0.5,
    ///     0.5, 1.0,
    /// );
    ///
    /// let cholesky = Cholesky::new(cov).unwrap();
    /// let det: f64 = cholesky.determinant();
    ///
    /// // The determinant appears in the normalization constant of the Gaussian PDF
    /// let norm_constant = 1.0 / ((2.0 * std::f64::consts::PI).sqrt() * det.sqrt());
    /// println!("Gaussian normalization: {}", norm_constant);
    /// ```
    ///
    /// # Warning
    ///
    /// For very small or very large determinants, this method may produce numerical
    /// overflow or underflow. In such cases, use [`ln_determinant`](Self::ln_determinant)
    /// which computes the natural logarithm of the determinant more stably.
    ///
    /// # See Also
    ///
    /// * [`ln_determinant`](Self::ln_determinant) - Computes log(det) for better numerical stability
    #[must_use]
    pub fn determinant(&self) -> T::SimdRealField {
        let dim = self.chol.nrows();
        let mut prod_diag = T::one();
        for i in 0..dim {
            prod_diag *= unsafe { self.chol.get_unchecked((i, i)).clone() };
        }
        prod_diag.simd_modulus_squared()
    }

    /// Computes the natural logarithm of the determinant of the decomposed matrix.
    ///
    /// This method is **more numerically stable** than [`determinant`](Self::determinant) for
    /// very small or very large determinants, as it avoids overflow and underflow by computing
    /// the logarithm directly without first computing the determinant itself.
    ///
    /// # How It Works
    ///
    /// For `M = L * L^T`, instead of computing `det(M) = (∏ L[i,i])^2`, this method computes:
    ///
    /// ```text
    /// ln(det(M)) = 2 * ln(∏ L[i,i]) = 2 * ∑ ln(L[i,i])
    /// ```
    ///
    /// This approach avoids numerical issues when the product of diagonal elements would
    /// overflow or underflow.
    ///
    /// # Returns
    ///
    /// The natural logarithm of the determinant: `ln(det(M))`.
    ///
    /// # Example: Basic usage
    ///
    /// ```
    /// use nalgebra::{Matrix3, linalg::Cholesky};
    ///
    /// let m = Matrix3::new(
    ///     4.0, 2.0, 1.0,
    ///     2.0, 5.0, 3.0,
    ///     1.0, 3.0, 6.0,
    /// );
    ///
    /// let cholesky = Cholesky::new(m.clone()).unwrap();
    /// let ln_det: f64 = cholesky.ln_determinant();
    /// let det: f64 = cholesky.determinant();
    ///
    /// // Verify: exp(ln_det) should equal det
    /// assert!((ln_det.exp() - det).abs() < 1e-10);
    /// ```
    ///
    /// # Example: Handling extreme values
    ///
    /// ```
    /// use nalgebra::{Matrix3, linalg::Cholesky};
    ///
    /// // Matrix with very small diagonal values (small determinant)
    /// let m = Matrix3::new(
    ///     1e-10, 0.0,   0.0,
    ///     0.0,   1e-10, 0.0,
    ///     0.0,   0.0,   1e-10,
    /// );
    ///
    /// let cholesky = Cholesky::new(m).unwrap();
    ///
    /// // determinant() might underflow to 0
    /// let det = cholesky.determinant();
    /// println!("Determinant: {} (may underflow)", det);
    ///
    /// // ln_determinant() handles this gracefully
    /// let ln_det = cholesky.ln_determinant();
    /// println!("ln(det): {} (stable)", ln_det);
    /// // Expected: ln(10^-30) = -30 * ln(10) ≈ -69.08
    /// ```
    ///
    /// # Example: Application in machine learning
    ///
    /// ```
    /// use nalgebra::{Matrix2, Vector2, linalg::Cholesky};
    ///
    /// // Computing the log-likelihood of a multivariate Gaussian
    /// let cov = Matrix2::new(
    ///     2.0, 0.5,
    ///     0.5, 1.0,
    /// );
    /// let mean = Vector2::new(0.0, 0.0);
    /// let x = Vector2::new(1.0, 1.0);
    ///
    /// let cholesky = Cholesky::new(cov.clone()).unwrap();
    /// let diff = x - mean;
    /// let inv_cov_diff = cholesky.solve(&diff);
    ///
    /// // Log-likelihood (avoiding overflow in the normalization term)
    /// let ln_det = cholesky.ln_determinant();
    /// let log_likelihood = -0.5 * (2.0 * std::f64::consts::PI.ln()
    ///                              + ln_det
    ///                              + diff.dot(&inv_cov_diff));
    /// println!("Log-likelihood: {}", log_likelihood);
    /// ```
    ///
    /// # See Also
    ///
    /// * [`determinant`](Self::determinant) - Direct determinant computation (may overflow/underflow)
    #[must_use]
    pub fn ln_determinant(&self) -> T::SimdRealField {
        let dim = self.chol.nrows();
        let mut sum_diag = T::SimdRealField::zero();
        for i in 0..dim {
            sum_diag += unsafe {
                self.chol
                    .get_unchecked((i, i))
                    .clone()
                    .simd_modulus_squared()
                    .simd_ln()
            };
        }
        sum_diag
    }
}

impl<T: ComplexField, D: Dim> Cholesky<T, D>
where
    DefaultAllocator: Allocator<D, D>,
{
    /// Attempts to compute the Cholesky decomposition of `matrix`.
    ///
    /// This is the **primary and safest way** to create a Cholesky decomposition. It validates
    /// that the matrix is positive-definite during decomposition and returns `None` if the
    /// matrix doesn't meet this requirement.
    ///
    /// # What It Does
    ///
    /// Decomposes a symmetric positive-definite matrix `M` into `L * L^T`, where `L` is a
    /// lower triangular matrix. This decomposition is unique and always exists for
    /// positive-definite matrices.
    ///
    /// # Arguments
    ///
    /// * `matrix` - A symmetric positive-definite matrix. Only the **lower-triangular part**
    ///              is read; the upper triangle is ignored.
    ///
    /// # Returns
    ///
    /// * `Some(Cholesky)` - If the matrix is positive-definite
    /// * `None` - If the matrix is not positive-definite (or has numerical issues)
    ///
    /// # What Makes a Matrix Positive-Definite?
    ///
    /// A symmetric matrix `M` is positive-definite if any of these equivalent conditions hold:
    /// - All eigenvalues are positive
    /// - For all non-zero vectors `x`: `x^T * M * x > 0`
    /// - All leading principal minors (top-left submatrices) have positive determinants
    ///
    /// Common positive-definite matrices include:
    /// - Covariance matrices (from data with independent columns)
    /// - Gram matrices `A^T * A` where `A` has full column rank
    /// - Diagonal matrices with all positive entries
    /// - The Hessian of a strictly convex function
    ///
    /// # Example: Basic decomposition
    ///
    /// ```
    /// use nalgebra::{Matrix3, linalg::Cholesky};
    ///
    /// // A positive-definite matrix
    /// let m = Matrix3::new(
    ///     4.0, 2.0, 1.0,
    ///     2.0, 5.0, 3.0,
    ///     1.0, 3.0, 6.0,
    /// );
    ///
    /// // Compute the decomposition
    /// let cholesky = Cholesky::new(m.clone()).unwrap();
    ///
    /// // Get the lower triangular factor
    /// let l = cholesky.l();
    ///
    /// // Verify: L * L^T = M
    /// let reconstructed = &l * l.transpose();
    /// assert!((reconstructed - m).norm() < 1e-10);
    /// ```
    ///
    /// # Example: Detecting non-positive-definite matrices
    ///
    /// ```
    /// use nalgebra::{Matrix3, linalg::Cholesky};
    ///
    /// // This matrix is NOT positive-definite (it's singular)
    /// let m = Matrix3::new(
    ///     1.0, 2.0, 3.0,
    ///     2.0, 4.0, 6.0,
    ///     3.0, 6.0, 9.0,
    /// );
    ///
    /// // new() returns None for non-positive-definite matrices
    /// assert!(Cholesky::new(m).is_none());
    /// ```
    ///
    /// # Example: Solving linear systems
    ///
    /// ```
    /// use nalgebra::{Matrix3, Vector3, linalg::Cholesky};
    ///
    /// let m = Matrix3::new(
    ///     4.0, 2.0, 1.0,
    ///     2.0, 5.0, 3.0,
    ///     1.0, 3.0, 6.0,
    /// );
    /// let b = Vector3::new(7.0, 14.0, 15.0);
    ///
    /// // Decompose once, solve efficiently
    /// let cholesky = Cholesky::new(m.clone()).unwrap();
    /// let x = cholesky.solve(&b);
    ///
    /// // Verify the solution
    /// assert!((m * x - b).norm() < 1e-10);
    /// ```
    ///
    /// # Example: Application in optimization
    ///
    /// ```
    /// use nalgebra::{Matrix3, Vector3, linalg::Cholesky};
    ///
    /// // Newton's method for optimization uses the Hessian (positive-definite at a minimum)
    /// let hessian = Matrix3::new(
    ///     2.0, 0.0, 0.0,
    ///     0.0, 4.0, 0.0,
    ///     0.0, 0.0, 6.0,
    /// );
    /// let gradient = Vector3::new(1.0, 2.0, 3.0);
    ///
    /// // Newton step: solve Hessian * step = -gradient
    /// let cholesky = Cholesky::new(hessian).unwrap();
    /// let step = cholesky.solve(&(-gradient));
    ///
    /// println!("Optimization step: {:?}", step);
    /// ```
    ///
    /// # Performance Note
    ///
    /// The decomposition is `O(n^3)` but subsequent operations (solving, determinant, etc.)
    /// are much faster. If you need to solve multiple systems with the same matrix, decompose
    /// once and reuse the result.
    ///
    /// # See Also
    ///
    /// * [`new_unchecked`](Self::new_unchecked) - Skips validation for performance (use carefully!)
    /// * [`new_with_substitute`](Self::new_with_substitute) - Handles near-positive-definite matrices
    /// * [`solve`](Self::solve) - Solve linear systems with the decomposition
    /// * [`determinant`](Self::determinant) - Compute determinant efficiently
    pub fn new(matrix: OMatrix<T, D, D>) -> Option<Self> {
        Self::new_internal(matrix, None)
    }

    /// Attempts to approximate the Cholesky decomposition of `matrix` by
    /// replacing non-positive values on the diagonals during the decomposition
    /// with the given `substitute`.
    ///
    /// This method is designed for matrices that are **theoretically** positive-definite but
    /// have small negative or zero diagonal values during decomposition due to numerical
    /// errors (e.g., from floating-point arithmetic or ill-conditioned matrices).
    ///
    /// # How It Works
    ///
    /// During the decomposition, whenever a diagonal element is encountered that is zero
    /// or has no valid square root (e.g., negative), the `substitute` value is used instead.
    /// [`try_sqrt`](ComplexField::try_sqrt) will be applied to the `substitute` when needed.
    ///
    /// If the input matrix naturally produces only positive diagonal values during decomposition,
    /// the `substitute` is never used and the result is identical to [`new`](Self::new).
    ///
    /// # Arguments
    ///
    /// * `matrix` - A matrix that should be positive-definite (only lower triangle is read)
    /// * `substitute` - Value to use when diagonal elements are non-positive (will be square-rooted)
    ///
    /// # Returns
    ///
    /// * `Some(Cholesky)` - An approximate decomposition
    /// * `None` - If even the substitute cannot produce a valid decomposition
    ///
    /// # Warning
    ///
    /// This is fundamentally an **approximation** and should be considered a numerical hack.
    /// The resulting decomposition may not accurately represent the original matrix.
    ///
    /// **Consider these alternatives instead:**
    /// - Improve numerical conditioning of your matrix
    /// - Use [`LU`](crate::linalg::LU) decomposition for general matrices
    /// - Use SVD for severely ill-conditioned problems
    /// - Regularize your matrix: `M + ε*I` for small `ε > 0`
    ///
    /// # Example: Handling numerical errors
    ///
    /// ```
    /// use nalgebra::{Matrix3, linalg::Cholesky};
    ///
    /// // A matrix that's nearly positive-definite but has numerical issues
    /// // (e.g., from accumulated floating-point errors)
    /// let m = Matrix3::new(
    ///     1.0,      0.9,      0.8,
    ///     0.9,      1.0,      0.9,
    ///     0.8,      0.9,      1.0,
    /// );
    ///
    /// // Regular decomposition might fail for ill-conditioned matrices
    /// if let None = Cholesky::new(m.clone()) {
    ///     // Try with a small substitute value
    ///     let cholesky = Cholesky::new_with_substitute(m, 1e-10);
    ///     assert!(cholesky.is_some());
    /// }
    /// ```
    ///
    /// # Example: Comparing with regularization
    ///
    /// ```
    /// use nalgebra::{Matrix3, linalg::Cholesky};
    ///
    /// // A matrix that's not quite positive-definite
    /// let m = Matrix3::new(
    ///     1.0,  0.99, 0.98,
    ///     0.99, 1.0,  0.99,
    ///     0.98, 0.99, 1.0,
    /// );
    ///
    /// // Method 1: Use substitute
    /// let chol1 = Cholesky::new_with_substitute(m.clone(), 1e-6);
    ///
    /// // Method 2: Regularization (often better!)
    /// let epsilon = 1e-6;
    /// let m_regularized = m + Matrix3::identity() * epsilon;
    /// let chol2 = Cholesky::new(m_regularized);
    ///
    /// // Both work, but regularization is more principled
    /// assert!(chol1.is_some());
    /// assert!(chol2.is_some());
    /// ```
    ///
    /// # Example: When it returns None
    ///
    /// ```
    /// use nalgebra::{Matrix2, linalg::Cholesky};
    ///
    /// // A matrix where the substitute also cannot help
    /// let m = Matrix2::new(
    ///     1.0,  2.0,
    ///     2.0,  1.0,  // This will cause issues during decomposition
    /// );
    ///
    /// // Even with a substitute, this returns None when decomposition fails
    /// let result = Cholesky::new_with_substitute(m, 1e-6);
    /// // Result depends on whether the substitute can fix the numerical issues
    /// ```
    ///
    /// # See Also
    ///
    /// * [`new`](Self::new) - Standard decomposition without substitution
    /// * [`new_unchecked`](Self::new_unchecked) - Fast decomposition without validation
    /// * [`LU`](crate::linalg::LU) - Alternative for non-positive-definite matrices
    pub fn new_with_substitute(matrix: OMatrix<T, D, D>, substitute: T) -> Option<Self> {
        Self::new_internal(matrix, Some(substitute))
    }

    /// Common implementation for `new` and `new_with_substitute`.
    fn new_internal(mut matrix: OMatrix<T, D, D>, substitute: Option<T>) -> Option<Self> {
        assert!(matrix.is_square(), "The input matrix must be square.");

        let n = matrix.nrows();

        for j in 0..n {
            for k in 0..j {
                let factor = unsafe { -matrix.get_unchecked((j, k)).clone() };

                let (mut col_j, col_k) = matrix.columns_range_pair_mut(j, k);
                let mut col_j = col_j.rows_range_mut(j..);
                let col_k = col_k.rows_range(j..);

                col_j.axpy(factor.conjugate(), &col_k, T::one());
            }

            let sqrt_denom = |v: T| {
                if v.is_zero() {
                    return None;
                }
                v.try_sqrt()
            };

            let diag = unsafe { matrix.get_unchecked((j, j)).clone() };

            if let Some(denom) =
                sqrt_denom(diag).or_else(|| substitute.clone().and_then(sqrt_denom))
            {
                unsafe {
                    *matrix.get_unchecked_mut((j, j)) = denom.clone();
                }

                let mut col = matrix.view_range_mut(j + 1.., j);
                col /= denom;
                continue;
            }

            // The diagonal element is either zero or its square root could not
            // be taken (e.g. for negative real numbers).
            return None;
        }

        Some(Cholesky { chol: matrix })
    }

    /// Updates the Cholesky decomposition after a rank-one modification to the original matrix.
    ///
    /// Given the Cholesky decomposition of a matrix `M`, a vector `x`, and a scalar `sigma`,
    /// this method efficiently updates the decomposition to represent `M + sigma * (x * x^H)`,
    /// where `x^H` denotes the conjugate transpose of `x` (for real matrices, this is just `x^T`).
    ///
    /// This is **much more efficient** than recomputing the entire decomposition from scratch,
    /// with complexity `O(n^2)` instead of `O(n^3)`.
    ///
    /// # How It Works
    ///
    /// A rank-one update modifies the matrix by adding (or subtracting) the outer product of a
    /// vector with itself:
    ///
    /// ```text
    /// M_new = M_old + σ * (x * x^T)
    /// ```
    ///
    /// For positive `sigma`, this adds a positive semi-definite matrix to `M`.
    /// For negative `sigma` (downdate), the result may not be positive-definite.
    ///
    /// # Arguments
    ///
    /// * `x` - The vector for the rank-one update
    /// * `sigma` - The scaling factor (positive for update, negative for downdate)
    ///
    /// # Panics
    ///
    /// Panics if `x` has a different size than the decomposed matrix.
    ///
    /// # Example: Basic rank-one update
    ///
    /// ```
    /// use nalgebra::{Matrix3, Vector3, linalg::Cholesky};
    ///
    /// // Start with a positive-definite matrix
    /// let m = Matrix3::new(
    ///     4.0, 2.0, 1.0,
    ///     2.0, 5.0, 3.0,
    ///     1.0, 3.0, 6.0,
    /// );
    ///
    /// let mut cholesky = Cholesky::new(m.clone()).unwrap();
    ///
    /// // Update vector
    /// let x = Vector3::new(1.0, 2.0, 3.0);
    /// let sigma = 0.5;
    ///
    /// // Perform the rank-one update
    /// cholesky.rank_one_update(&x, sigma);
    ///
    /// // Verify: the decomposition now represents M + sigma * (x * x^T)
    /// let x_outer = &x * x.transpose();
    /// let m_updated = m + x_outer * sigma;
    ///
    /// let l = cholesky.l();
    /// let reconstructed = &l * l.transpose();
    /// assert!((reconstructed - m_updated).norm() < 1e-10);
    /// ```
    ///
    /// # Example: Application in Kalman filtering
    ///
    /// ```
    /// use nalgebra::{Matrix2, Vector2, linalg::Cholesky};
    ///
    /// // Covariance matrix in a Kalman filter
    /// let mut cov = Matrix2::new(
    ///     1.0, 0.0,
    ///     0.0, 1.0,
    /// );
    ///
    /// let mut cholesky = Cholesky::new(cov.clone()).unwrap();
    ///
    /// // Process noise adds a rank-one component
    /// let noise_direction = Vector2::new(1.0, 0.5);
    /// let noise_variance = 0.1;
    ///
    /// // Update the covariance efficiently
    /// cholesky.rank_one_update(&noise_direction, noise_variance);
    ///
    /// // The decomposition now includes the process noise
    /// println!("Updated covariance:\n{}", cholesky.l() * cholesky.l().transpose());
    /// ```
    ///
    /// # Example: Sequential updates
    ///
    /// ```
    /// use nalgebra::{Matrix3, Vector3, linalg::Cholesky};
    ///
    /// let m = Matrix3::identity();
    /// let mut cholesky = Cholesky::new(m).unwrap();
    ///
    /// // Add multiple rank-one updates efficiently
    /// let updates = vec![
    ///     Vector3::new(1.0, 0.0, 0.0),
    ///     Vector3::new(0.0, 1.0, 0.0),
    ///     Vector3::new(0.0, 0.0, 1.0),
    /// ];
    ///
    /// for x in updates {
    ///     cholesky.rank_one_update(&x, 0.5);
    /// }
    ///
    /// // Result: I + 0.5 * sum(x_i * x_i^T)
    /// let result = cholesky.l() * cholesky.l().transpose();
    /// println!("After sequential updates:\n{}", result);
    /// ```
    ///
    /// # Performance Note
    ///
    /// This operation is `O(n^2)`, much faster than recomputing the decomposition from scratch
    /// which is `O(n^3)`. Use this when you need to incrementally update a decomposition.
    ///
    /// # See Also
    ///
    /// * [`insert_column`](Self::insert_column) - Add a row and column to the matrix
    /// * [`remove_column`](Self::remove_column) - Remove a row and column from the matrix
    #[inline]
    pub fn rank_one_update<R2: Dim, S2>(&mut self, x: &Vector<T, R2, S2>, sigma: T::RealField)
    where
        S2: Storage<T, R2, U1>,
        DefaultAllocator: Allocator<R2, U1>,
        ShapeConstraint: SameNumberOfRows<R2, D>,
    {
        Self::xx_rank_one_update(&mut self.chol, &mut x.clone_owned(), sigma)
    }

    /// Updates the decomposition such that we get the decomposition of a matrix with the given column `col` in the `j`th position.
    ///
    /// This method efficiently updates the Cholesky decomposition when inserting both a new
    /// row and column at position `j` in the original matrix. Since the matrix must remain
    /// symmetric, the inserted row equals the transpose of the inserted column.
    ///
    /// This is useful for dynamically growing the size of positive-definite matrices, such as
    /// incrementally building covariance matrices or kernel matrices in machine learning.
    ///
    /// # How It Works
    ///
    /// Given an `n×n` matrix `M` with decomposition `M = L * L^T`, this computes the
    /// decomposition of the `(n+1)×(n+1)` matrix formed by inserting a row and column at
    /// position `j`. The operation is more efficient than recomputing the entire decomposition.
    ///
    /// # Arguments
    ///
    /// * `j` - The position where the new row/column should be inserted (0-indexed, must be ≤ n)
    /// * `col` - The values for the new column (length must be `n+1`, including the new diagonal element)
    ///
    /// # Returns
    ///
    /// A new Cholesky decomposition of dimension `(n+1)×(n+1)`.
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - `col` doesn't have size `n+1`
    /// - `j` is out of bounds (must be ≤ n)
    ///
    /// # Example: Basic column insertion
    ///
    /// ```
    /// use nalgebra::{Matrix3, Matrix2, Vector3, linalg::Cholesky};
    ///
    /// // Start with a 2×2 positive-definite matrix
    /// let m2 = Matrix2::new(
    ///     4.0, 2.0,
    ///     2.0, 5.0,
    /// );
    ///
    /// let chol2 = Cholesky::new(m2).unwrap();
    ///
    /// // Insert a new row/column at position 1 (middle)
    /// // The new 3×3 matrix will have this structure:
    /// // [ 4  X  2 ]
    /// // [ X  3  X ]  where X values come from col
    /// // [ 2  X  5 ]
    /// let col = Vector3::new(1.0, 3.0, 1.5);  // values for position [0,1], [1,1], [2,1]
    /// let chol3 = chol2.insert_column(1, col);
    ///
    /// // Verify the size increased
    /// assert_eq!(chol3.l().nrows(), 3);
    /// ```
    ///
    /// # Example: Building a matrix incrementally
    ///
    /// ```
    /// use nalgebra::{Matrix1, Vector2, Vector3, linalg::Cholesky};
    ///
    /// // Start with a 1×1 matrix
    /// let m1 = Matrix1::new(4.0);
    /// let chol1 = Cholesky::new(m1).unwrap();
    ///
    /// // Grow to 2×2
    /// let col2 = Vector2::new(2.0, 5.0);
    /// let chol2 = chol1.insert_column(1, col2);
    ///
    /// // Grow to 3×3
    /// let col3 = Vector3::new(1.0, 3.0, 6.0);
    /// let chol3 = chol2.insert_column(2, col3);
    ///
    /// // We now have a 3×3 decomposition
    /// assert_eq!(chol3.l().nrows(), 3);
    /// println!("Final matrix:\n{}", chol3.l() * chol3.l().transpose());
    /// ```
    ///
    /// # Example: Inserting at different positions
    ///
    /// ```
    /// use nalgebra::{Matrix2, Vector3, linalg::Cholesky};
    ///
    /// let m = Matrix2::new(4.0, 1.0, 1.0, 3.0);
    /// let chol = Cholesky::new(m).unwrap();
    ///
    /// // Insert at the beginning (position 0)
    /// let col_start = Vector3::new(2.0, 0.5, 0.3);
    /// let chol_start = chol.clone().insert_column(0, col_start);
    /// assert_eq!(chol_start.l().nrows(), 3);
    ///
    /// // Insert at the end (position 2)
    /// let col_end = Vector3::new(0.5, 0.8, 2.5);
    /// let chol_end = chol.insert_column(2, col_end);
    /// assert_eq!(chol_end.l().nrows(), 3);
    /// ```
    ///
    /// # Performance Note
    ///
    /// This operation is more efficient than recomputing the entire decomposition, though
    /// it still requires `O(n^2)` work.
    ///
    /// # See Also
    ///
    /// * [`remove_column`](Self::remove_column) - Remove a row and column from the matrix
    /// * [`rank_one_update`](Self::rank_one_update) - Update with a rank-one modification
    pub fn insert_column<R2, S2>(
        &self,
        j: usize,
        col: Vector<T, R2, S2>,
    ) -> Cholesky<T, DimSum<D, U1>>
    where
        D: DimAdd<U1>,
        R2: Dim,
        S2: Storage<T, R2, U1>,
        DefaultAllocator: Allocator<DimSum<D, U1>, DimSum<D, U1>> + Allocator<R2>,
        ShapeConstraint: SameNumberOfRows<R2, DimSum<D, U1>>,
    {
        let mut col = col.into_owned();
        // for an explanation of the formulas, see https://en.wikipedia.org/wiki/Cholesky_decomposition#Updating_the_decomposition
        let n = col.nrows();
        assert_eq!(
            n,
            self.chol.nrows() + 1,
            "The new column must have the size of the factored matrix plus one."
        );
        assert!(j < n, "j needs to be within the bound of the new matrix.");

        // loads the data into a new matrix with an additional jth row/column
        // TODO: would it be worth it to avoid the zero-initialization?
        let mut chol = Matrix::zeros_generic(
            self.chol.shape_generic().0.add(Const::<1>),
            self.chol.shape_generic().1.add(Const::<1>),
        );
        chol.view_range_mut(..j, ..j)
            .copy_from(&self.chol.view_range(..j, ..j));
        chol.view_range_mut(..j, j + 1..)
            .copy_from(&self.chol.view_range(..j, j..));
        chol.view_range_mut(j + 1.., ..j)
            .copy_from(&self.chol.view_range(j.., ..j));
        chol.view_range_mut(j + 1.., j + 1..)
            .copy_from(&self.chol.view_range(j.., j..));

        // update the jth row
        let top_left_corner = self.chol.view_range(..j, ..j);

        let col_j = col[j].clone();
        let (mut new_rowj_adjoint, mut new_colj) = col.rows_range_pair_mut(..j, j + 1..);
        assert!(
            top_left_corner.solve_lower_triangular_mut(&mut new_rowj_adjoint),
            "Cholesky::insert_column : Unable to solve lower triangular system!"
        );

        new_rowj_adjoint.adjoint_to(&mut chol.view_range_mut(j, ..j));

        // update the center element
        let center_element = T::sqrt(col_j - T::from_real(new_rowj_adjoint.norm_squared()));
        chol[(j, j)] = center_element.clone();

        // update the jth column
        let bottom_left_corner = self.chol.view_range(j.., ..j);
        // new_colj = (col_jplus - bottom_left_corner * new_rowj.adjoint()) / center_element;
        new_colj.gemm(
            -T::one() / center_element.clone(),
            &bottom_left_corner,
            &new_rowj_adjoint,
            T::one() / center_element,
        );
        chol.view_range_mut(j + 1.., j).copy_from(&new_colj);

        // update the bottom right corner
        let mut bottom_right_corner = chol.view_range_mut(j + 1.., j + 1..);
        Self::xx_rank_one_update(
            &mut bottom_right_corner,
            &mut new_colj,
            -T::RealField::one(),
        );

        Cholesky { chol }
    }

    /// Updates the decomposition such that we get the decomposition of the factored matrix with its `j`th column removed.
    ///
    /// This method efficiently updates the Cholesky decomposition when removing both the `j`th
    /// row and column from the original matrix. Since the matrix must remain symmetric, both
    /// the row and column are removed together.
    ///
    /// This is useful for dynamically shrinking the size of positive-definite matrices, such as
    /// removing variables from covariance matrices or pruning kernel matrices in machine learning.
    ///
    /// # How It Works
    ///
    /// Given an `n×n` matrix `M` with decomposition `M = L * L^T`, this computes the
    /// decomposition of the `(n-1)×(n-1)` matrix formed by removing the `j`th row and column.
    /// The operation is more efficient than recomputing the entire decomposition.
    ///
    /// # Arguments
    ///
    /// * `j` - The index of the row/column to remove (0-indexed, must be < n)
    ///
    /// # Returns
    ///
    /// A new Cholesky decomposition of dimension `(n-1)×(n-1)`.
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - The matrix has size 0 (cannot remove from empty matrix)
    /// - `j` is out of bounds (must be < n)
    ///
    /// # Example: Basic column removal
    ///
    /// ```
    /// use nalgebra::{Matrix3, linalg::Cholesky};
    ///
    /// // Start with a 3×3 positive-definite matrix
    /// let m3 = Matrix3::new(
    ///     4.0, 2.0, 1.0,
    ///     2.0, 5.0, 3.0,
    ///     1.0, 3.0, 6.0,
    /// );
    ///
    /// let chol3 = Cholesky::new(m3).unwrap();
    ///
    /// // Remove the middle row/column (index 1)
    /// let chol2 = chol3.remove_column(1);
    ///
    /// // Verify the size decreased
    /// assert_eq!(chol2.l().nrows(), 2);
    ///
    /// // The resulting matrix should be:
    /// // [ 4  1 ]
    /// // [ 1  6 ]
    /// ```
    ///
    /// # Example: Shrinking a matrix incrementally
    ///
    /// ```
    /// use nalgebra::{Matrix3, linalg::Cholesky};
    ///
    /// let m3 = Matrix3::new(
    ///     4.0, 2.0, 1.0,
    ///     2.0, 5.0, 3.0,
    ///     1.0, 3.0, 6.0,
    /// );
    ///
    /// let chol3 = Cholesky::new(m3).unwrap();
    ///
    /// // Shrink to 2×2 by removing last column
    /// let chol2 = chol3.remove_column(2);
    /// assert_eq!(chol2.l().nrows(), 2);
    ///
    /// // Shrink to 1×1 by removing another column
    /// let chol1 = chol2.remove_column(1);
    /// assert_eq!(chol1.l().nrows(), 1);
    /// ```
    ///
    /// # Example: Removing from different positions
    ///
    /// ```
    /// use nalgebra::{Matrix3, linalg::Cholesky};
    ///
    /// let m = Matrix3::new(
    ///     4.0, 2.0, 1.0,
    ///     2.0, 5.0, 3.0,
    ///     1.0, 3.0, 6.0,
    /// );
    ///
    /// let chol = Cholesky::new(m).unwrap();
    ///
    /// // Remove first column (index 0)
    /// let chol_no_first = chol.clone().remove_column(0);
    /// assert_eq!(chol_no_first.l().nrows(), 2);
    ///
    /// // Remove last column (index 2)
    /// let chol_no_last = chol.clone().remove_column(2);
    /// assert_eq!(chol_no_last.l().nrows(), 2);
    ///
    /// // Remove middle column (index 1)
    /// let chol_no_middle = chol.remove_column(1);
    /// assert_eq!(chol_no_middle.l().nrows(), 2);
    /// ```
    ///
    /// # Example: Application in feature selection
    ///
    /// ```
    /// use nalgebra::{Matrix3, linalg::Cholesky};
    ///
    /// // Covariance matrix of three features
    /// let cov = Matrix3::new(
    ///     1.0, 0.8, 0.3,
    ///     0.8, 1.0, 0.2,
    ///     0.3, 0.2, 1.0,
    /// );
    ///
    /// let chol = Cholesky::new(cov).unwrap();
    ///
    /// // Remove feature 2 (index 1) - maybe it's redundant
    /// let chol_reduced = chol.remove_column(1);
    ///
    /// // Now we have a 2×2 covariance matrix with features 1 and 3
    /// println!("Reduced covariance:\n{}", chol_reduced.l() * chol_reduced.l().transpose());
    /// ```
    ///
    /// # Performance Note
    ///
    /// This operation is more efficient than recomputing the entire decomposition, though
    /// it still requires `O(n^2)` work due to the rank-one update needed to restore the
    /// decomposition after removal.
    ///
    /// # See Also
    ///
    /// * [`insert_column`](Self::insert_column) - Add a row and column to the matrix
    /// * [`rank_one_update`](Self::rank_one_update) - Update with a rank-one modification
    #[must_use]
    pub fn remove_column(&self, j: usize) -> Cholesky<T, DimDiff<D, U1>>
    where
        D: DimSub<U1>,
        DefaultAllocator: Allocator<DimDiff<D, U1>, DimDiff<D, U1>> + Allocator<D>,
    {
        let n = self.chol.nrows();
        assert!(n > 0, "The matrix needs at least one column.");
        assert!(j < n, "j needs to be within the bound of the matrix.");

        // loads the data into a new matrix except for the jth row/column
        // TODO: would it be worth it to avoid this zero initialization?
        let mut chol = Matrix::zeros_generic(
            self.chol.shape_generic().0.sub(Const::<1>),
            self.chol.shape_generic().1.sub(Const::<1>),
        );
        chol.view_range_mut(..j, ..j)
            .copy_from(&self.chol.view_range(..j, ..j));
        chol.view_range_mut(..j, j..)
            .copy_from(&self.chol.view_range(..j, j + 1..));
        chol.view_range_mut(j.., ..j)
            .copy_from(&self.chol.view_range(j + 1.., ..j));
        chol.view_range_mut(j.., j..)
            .copy_from(&self.chol.view_range(j + 1.., j + 1..));

        // updates the bottom right corner
        let mut bottom_right_corner = chol.view_range_mut(j.., j..);
        let mut workspace = self.chol.column(j).clone_owned();
        let mut old_colj = workspace.rows_range_mut(j + 1..);
        Self::xx_rank_one_update(&mut bottom_right_corner, &mut old_colj, T::RealField::one());

        Cholesky { chol }
    }

    /// Given the Cholesky decomposition of a matrix `M`, a scalar `sigma` and a vector `x`,
    /// performs a rank one update such that we end up with the decomposition of `M + sigma * (x * x.adjoint())`.
    ///
    /// This helper method is called by `rank_one_update` but also `insert_column` and `remove_column`
    /// where it is used on a square view of the decomposition
    fn xx_rank_one_update<Dm, Sm, Rx, Sx>(
        chol: &mut Matrix<T, Dm, Dm, Sm>,
        x: &mut Vector<T, Rx, Sx>,
        sigma: T::RealField,
    ) where
        //T: ComplexField,
        Dm: Dim,
        Rx: Dim,
        Sm: StorageMut<T, Dm, Dm>,
        Sx: StorageMut<T, Rx, U1>,
    {
        // heavily inspired by Eigen's `llt_rank_update_lower` implementation https://eigen.tuxfamily.org/dox/LLT_8h_source.html
        let n = x.nrows();
        assert_eq!(
            n,
            chol.nrows(),
            "The input vector must be of the same size as the factorized matrix."
        );

        let mut beta = crate::one::<T::RealField>();

        for j in 0..n {
            // updates the diagonal
            let diag = T::real(unsafe { chol.get_unchecked((j, j)).clone() });
            let diag2 = diag.clone() * diag.clone();
            let xj = unsafe { x.get_unchecked(j).clone() };
            let sigma_xj2 = sigma.clone() * T::modulus_squared(xj.clone());
            let gamma = diag2.clone() * beta.clone() + sigma_xj2.clone();
            let new_diag = (diag2.clone() + sigma_xj2.clone() / beta.clone()).sqrt();
            unsafe { *chol.get_unchecked_mut((j, j)) = T::from_real(new_diag.clone()) };
            beta += sigma_xj2 / diag2;
            // updates the terms of L
            let mut xjplus = x.rows_range_mut(j + 1..);
            let mut col_j = chol.view_range_mut(j + 1.., j);
            // temp_jplus -= (wj / T::from_real(diag)) * col_j;
            xjplus.axpy(-xj.clone() / T::from_real(diag.clone()), &col_j, T::one());
            if gamma != crate::zero::<T::RealField>() {
                // col_j = T::from_real(nljj / diag) * col_j  + (T::from_real(nljj * sigma / gamma) * T::conjugate(wj)) * temp_jplus;
                col_j.axpy(
                    T::from_real(new_diag.clone() * sigma.clone() / gamma) * T::conjugate(xj),
                    &xjplus,
                    T::from_real(new_diag / diag),
                );
            }
        }
    }
}
