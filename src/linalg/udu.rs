#[cfg(feature = "serde-serialize-no-std")]
use serde::{Deserialize, Serialize};

use crate::allocator::Allocator;
use crate::base::{Const, DefaultAllocator, OMatrix, OVector};
use crate::dimension::Dim;
use simba::scalar::RealField;

/// The UDU decomposition of a symmetric positive-definite matrix.
///
/// # What is UDU Decomposition?
///
/// The UDU decomposition is a way to factorize a symmetric positive-definite matrix `P`
/// into the product of an upper triangular matrix `U` with ones on the diagonal, a diagonal
/// matrix `D`, and the transpose of `U`:
///
/// ```text
/// P = U * D * U^T
/// ```
///
/// where:
/// - `U` is an upper triangular matrix with ones on the diagonal
/// - `D` is a diagonal matrix with positive elements
/// - `U^T` denotes the transpose of `U`
///
/// # Why Use UDU Instead of Cholesky?
///
/// The UDU decomposition is particularly useful when:
/// - **Numerical stability**: It avoids square roots, making it more numerically stable for
///   ill-conditioned matrices or when diagonal elements vary greatly in magnitude
/// - **Kalman filtering**: It's the preferred decomposition for covariance matrices in Kalman
///   filters, especially in embedded systems and control applications
/// - **Control theory**: Common in Riccati equation solvers and optimal control problems
/// - **Avoiding underflow/overflow**: Since it doesn't compute square roots, it's less prone
///   to numerical issues with very small or very large values
///
/// # When to Use It?
///
/// This decomposition is useful for:
/// - Implementing Kalman filters and state estimation algorithms
/// - Solving optimal control problems and Riccati equations
/// - Covariance matrix propagation in navigation systems
/// - Any application where you need Cholesky-like factorization but want to avoid square roots
///
/// # Requirements
///
/// The input matrix must be:
/// - **Symmetric**: `P = P^T`
/// - **Positive-definite**: All eigenvalues are positive, or equivalently, `x^T * P * x > 0`
///   for all non-zero vectors `x`
///
/// Only the **upper-triangular part** of the input matrix is read during decomposition.
///
/// # Comparison with Cholesky
///
/// | Feature | Cholesky (`L * L^T`) | UDU (`U * D * U^T`) |
/// |---------|---------------------|---------------------|
/// | Computational cost | Slightly faster | Slightly slower |
/// | Numerical stability | Good | Better (no square roots) |
/// | Memory usage | Same | Same |
/// | Common in | General linear algebra | Kalman filtering, control |
///
/// # Example
///
/// ```
/// use nalgebra::Matrix3;
///
/// // A symmetric positive-definite matrix
/// let p = Matrix3::new(
///     2.0, -1.0,  0.0,
///    -1.0,  2.0, -1.0,
///     0.0, -1.0,  2.0,
/// );
///
/// // Compute the UDU decomposition
/// let udu = p.udu().unwrap();
///
/// // Access the factors
/// let u = &udu.u;  // Upper triangular with ones on diagonal
/// let d = &udu.d;  // Diagonal elements as a vector
///
/// // Reconstruct the original matrix
/// let p_reconstructed = u * udu.d_matrix() * u.transpose();
/// assert!((p - p_reconstructed).norm() < 1e-10);
/// ```
#[cfg_attr(feature = "serde-serialize-no-std", derive(Serialize, Deserialize))]
#[cfg_attr(
    feature = "serde-serialize-no-std",
    serde(bound(serialize = "OVector<T, D>: Serialize, OMatrix<T, D, D>: Serialize"))
)]
#[cfg_attr(
    feature = "serde-serialize-no-std",
    serde(bound(
        deserialize = "OVector<T, D>: Deserialize<'de>, OMatrix<T, D, D>: Deserialize<'de>"
    ))
)]
#[cfg_attr(feature = "defmt", derive(defmt::Format))]
#[derive(Clone, Debug)]
pub struct UDU<T: RealField, D: Dim>
where
    DefaultAllocator: Allocator<D> + Allocator<D, D>,
{
    /// The upper triangular matrix `U` resulting from the factorization.
    ///
    /// This matrix has the following properties:
    /// - Upper triangular (all elements below the diagonal are zero)
    /// - Unit diagonal (all diagonal elements are exactly 1)
    /// - Together with `d`, satisfies: `P = U * D * U^T` where `P` is the original matrix
    pub u: OMatrix<T, D, D>,

    /// The diagonal elements `D` of the factorization, stored as a vector.
    ///
    /// These are the positive values that appear on the diagonal of the `D` matrix in
    /// the decomposition `P = U * D * U^T`. Use [`d_matrix()`](Self::d_matrix) to convert
    /// this vector into a full diagonal matrix.
    pub d: OVector<T, D>,
}

impl<T: RealField, D: Dim> Copy for UDU<T, D>
where
    DefaultAllocator: Allocator<D> + Allocator<D, D>,
    OVector<T, D>: Copy,
    OMatrix<T, D, D>: Copy,
{
}

impl<T: RealField, D: Dim> UDU<T, D>
where
    DefaultAllocator: Allocator<D> + Allocator<D, D>,
{
    /// Computes the UDU decomposition of a symmetric positive-definite matrix.
    ///
    /// This is the **primary way** to create a UDU decomposition. It validates that the matrix
    /// is positive-definite during decomposition and returns `None` if the matrix doesn't meet
    /// this requirement.
    ///
    /// # What It Does
    ///
    /// Decomposes a symmetric positive-definite matrix `P` into `U * D * U^T`, where:
    /// - `U` is an upper triangular matrix with ones on the diagonal
    /// - `D` is a diagonal matrix with positive elements
    ///
    /// This decomposition is unique for positive-definite matrices and avoids computing square
    /// roots, making it more numerically stable than Cholesky decomposition in many applications.
    ///
    /// # Arguments
    ///
    /// * `p` - A symmetric positive-definite matrix. Only the **upper-triangular part** is read;
    ///         the lower triangle is ignored. This matches the convention used in Kalman filtering
    ///         and control applications.
    ///
    /// # Returns
    ///
    /// * `Some(UDU)` - If the matrix is positive-definite
    /// * `None` - If the matrix is not positive-definite (has zero or negative eigenvalues, or
    ///            numerical issues occur during decomposition)
    ///
    /// # What Makes a Matrix Positive-Definite?
    ///
    /// A symmetric matrix `P` is positive-definite if:
    /// - All eigenvalues are positive
    /// - For all non-zero vectors `x`: `x^T * P * x > 0`
    /// - All leading principal minors have positive determinants
    ///
    /// Common positive-definite matrices include:
    /// - Covariance matrices (from data with independent columns)
    /// - Gram matrices `A^T * A` where `A` has full column rank
    /// - Solutions to Riccati equations in control theory
    /// - Innovation covariances in Kalman filters
    ///
    /// # Example: Basic decomposition
    ///
    /// ```
    /// use nalgebra::Matrix3;
    ///
    /// // A symmetric positive-definite matrix (tri-diagonal)
    /// let p = Matrix3::new(
    ///     2.0, -1.0,  0.0,
    ///    -1.0,  2.0, -1.0,
    ///     0.0, -1.0,  2.0,
    /// );
    ///
    /// // Compute the UDU decomposition
    /// let udu = p.udu().unwrap();
    ///
    /// // Access the factors
    /// println!("U matrix:\n{}", udu.u);
    /// println!("D vector: {}", udu.d);
    ///
    /// // Verify: U * D * U^T = P
    /// let reconstructed = &udu.u * udu.d_matrix() * udu.u.transpose();
    /// assert!((p - reconstructed).norm() < 1e-10);
    /// ```
    ///
    /// # Example: Kalman filter covariance matrix
    ///
    /// ```
    /// use nalgebra::Matrix2;
    ///
    /// // Covariance matrix from a 2D Kalman filter
    /// // Represents uncertainty in position (x, y) with correlation
    /// let covariance = Matrix2::new(
    ///     1.0, 0.3,  // Variance in x, covariance xy
    ///     0.3, 1.5,  // Covariance xy, variance in y
    /// );
    ///
    /// // UDU decomposition is preferred for covariance propagation
    /// let udu = covariance.udu().unwrap();
    ///
    /// // The diagonal elements show the "independent" variances
    /// println!("Independent variances: {}", udu.d);
    ///
    /// // U captures the correlations
    /// println!("Correlation structure:\n{}", udu.u);
    /// ```
    ///
    /// # Example: Detecting non-positive-definite matrices
    ///
    /// ```
    /// use nalgebra::Matrix2;
    ///
    /// // This matrix is NOT positive-definite (it's singular)
    /// let singular = Matrix2::new(
    ///     1.0, 1.0,
    ///     1.0, 1.0,
    /// );
    ///
    /// // new() returns None for non-positive-definite matrices
    /// assert!(singular.udu().is_none());
    ///
    /// // This matrix has a zero diagonal element (not positive-definite)
    /// let zero_diag = Matrix2::new(
    ///      0.0, 0.0,
    ///      0.0, 1.0,
    /// );
    ///
    /// assert!(zero_diag.udu().is_none());
    /// ```
    ///
    /// # Example: Control theory application
    ///
    /// ```
    /// use nalgebra::Matrix3;
    ///
    /// // Solution to a Riccati equation in optimal control
    /// // Represents the cost-to-go matrix
    /// let p = Matrix3::new(
    ///     4.0, 2.0, 1.0,
    ///     2.0, 5.0, 3.0,
    ///     1.0, 3.0, 6.0,
    /// );
    ///
    /// let udu = p.udu().unwrap();
    ///
    /// // In control theory, the UDU form is useful for:
    /// // - Numerically stable Riccati equation solving
    /// // - Efficient optimal control gain computation
    /// // - Avoiding numerical issues in embedded systems
    ///
    /// println!("Control cost structure (D): {}", udu.d);
    /// ```
    ///
    /// # Performance Note
    ///
    /// The decomposition is `O(n^3)` where `n` is the matrix dimension. Once computed,
    /// the decomposition can be reused for multiple operations, making it efficient for
    /// applications like Kalman filtering where the same covariance structure is used
    /// repeatedly.
    ///
    /// # Numerical Stability
    ///
    /// The UDU decomposition is generally more numerically stable than Cholesky decomposition
    /// because it avoids computing square roots. This makes it particularly suitable for:
    /// - Matrices with widely varying diagonal elements
    /// - Embedded systems with limited floating-point precision
    /// - Long-running filters where numerical errors accumulate
    ///
    /// # Reference
    ///
    /// The algorithm is based on: "Optimal control and estimation", Robert F. Stengel,
    /// Dover Publications (1994), page 360.
    ///
    /// # See Also
    ///
    /// * [`d_matrix`](Self::d_matrix) - Convert the diagonal vector to a full matrix
    /// * [`Cholesky`](crate::linalg::Cholesky) - Alternative decomposition using square roots
    pub fn new(p: OMatrix<T, D, D>) -> Option<Self> {
        let n = p.ncols();
        let n_dim = p.shape_generic().1;

        let mut d = OVector::zeros_generic(n_dim, Const::<1>);
        let mut u = OMatrix::zeros_generic(n_dim, n_dim);

        d[n - 1] = p[(n - 1, n - 1)].clone();

        if d[n - 1].is_zero() {
            return None;
        }

        u.column_mut(n - 1)
            .axpy(T::one() / d[n - 1].clone(), &p.column(n - 1), T::zero());

        for j in (0..n - 1).rev() {
            let mut d_j = d[j].clone();
            for k in j + 1..n {
                d_j += d[k].clone() * u[(j, k)].clone().powi(2);
            }

            d[j] = p[(j, j)].clone() - d_j;

            if d[j].is_zero() {
                return None;
            }

            for i in (0..=j).rev() {
                let mut u_ij = u[(i, j)].clone();
                for k in j + 1..n {
                    u_ij += d[k].clone() * u[(j, k)].clone() * u[(i, k)].clone();
                }

                u[(i, j)] = (p[(i, j)].clone() - u_ij) / d[j].clone();
            }

            u[(j, j)] = T::one();
        }

        Some(Self { u, d })
    }

    /// Converts the diagonal vector `d` into a full diagonal matrix `D`.
    ///
    /// This method constructs a square matrix with the diagonal elements from the decomposition
    /// on its main diagonal and zeros everywhere else. The resulting matrix `D` is used in the
    /// factorization `P = U * D * U^T`.
    ///
    /// # What It Returns
    ///
    /// A square diagonal matrix where:
    /// - Diagonal elements are the values from `self.d`
    /// - All off-diagonal elements are zero
    ///
    /// # When to Use This
    ///
    /// Use this method when you need:
    /// - To reconstruct the original matrix: `P = U * D * U^T`
    /// - To perform matrix operations that require `D` as a full matrix
    /// - To visualize or print the complete decomposition
    ///
    /// For most computational purposes, working directly with the diagonal vector `self.d`
    /// is more efficient than creating the full matrix.
    ///
    /// # Returns
    ///
    /// A diagonal matrix with `self.d` on the diagonal and zeros elsewhere.
    ///
    /// # Example: Basic usage
    ///
    /// ```
    /// use nalgebra::Matrix3;
    ///
    /// let p = Matrix3::new(
    ///     2.0, -1.0,  0.0,
    ///    -1.0,  2.0, -1.0,
    ///     0.0, -1.0,  2.0,
    /// );
    ///
    /// let udu = p.udu().unwrap();
    ///
    /// // Get D as a full diagonal matrix
    /// let d_matrix = udu.d_matrix();
    ///
    /// // D is diagonal (all off-diagonal elements are zero)
    /// assert_eq!(d_matrix[(0, 1)], 0.0);
    /// assert_eq!(d_matrix[(1, 0)], 0.0);
    ///
    /// // Diagonal elements come from udu.d
    /// assert_eq!(d_matrix[(0, 0)], udu.d[0]);
    /// assert_eq!(d_matrix[(1, 1)], udu.d[1]);
    /// assert_eq!(d_matrix[(2, 2)], udu.d[2]);
    /// ```
    ///
    /// # Example: Reconstructing the original matrix
    ///
    /// ```
    /// use nalgebra::Matrix2;
    ///
    /// let p = Matrix2::new(
    ///     4.0, 2.0,
    ///     2.0, 5.0,
    /// );
    ///
    /// let udu = p.udu().unwrap();
    ///
    /// // Reconstruct: P = U * D * U^T
    /// let p_reconstructed = &udu.u * udu.d_matrix() * udu.u.transpose();
    ///
    /// // Should match the original matrix
    /// assert!((p - p_reconstructed).norm() < 1e-10);
    /// ```
    ///
    /// # Example: Kalman filter covariance propagation
    ///
    /// ```
    /// use nalgebra::{Matrix2, Vector2};
    ///
    /// // Initial covariance in UDU form
    /// let p = Matrix2::new(
    ///     1.0, 0.2,
    ///     0.2, 1.5,
    /// );
    /// let udu = p.udu().unwrap();
    ///
    /// // State transition matrix
    /// let f = Matrix2::new(
    ///     1.0, 0.1,  // Simple position-velocity model
    ///     0.0, 1.0,
    /// );
    ///
    /// // Propagate covariance: P_new = F * P * F^T
    /// // Using UDU form for numerical stability
    /// let d_full = udu.d_matrix();
    /// let p_propagated = f * &udu.u * d_full * udu.u.transpose() * f.transpose();
    ///
    /// println!("Propagated covariance:\n{}", p_propagated);
    /// ```
    ///
    /// # Example: Comparing with vector form
    ///
    /// ```
    /// use nalgebra::Matrix3;
    ///
    /// let p = Matrix3::new(
    ///     4.0, 1.0, 0.5,
    ///     1.0, 3.0, 1.0,
    ///     0.5, 1.0, 2.0,
    /// );
    ///
    /// let udu = p.udu().unwrap();
    ///
    /// // Vector form (more efficient for storage and some computations)
    /// let d_vec = &udu.d;
    /// println!("D as vector: {}", d_vec);
    ///
    /// // Matrix form (needed for some matrix operations)
    /// let d_mat = udu.d_matrix();
    /// println!("D as matrix:\n{}", d_mat);
    ///
    /// // Both represent the same information
    /// for i in 0..3 {
    ///     assert_eq!(d_vec[i], d_mat[(i, i)]);
    /// }
    /// ```
    ///
    /// # Performance Note
    ///
    /// Creating the full diagonal matrix is `O(n^2)` in both time and space. If you only need
    /// the diagonal values for computations, consider using `self.d` directly for better performance.
    ///
    /// For example, instead of:
    /// ```ignore
    /// let d_full = udu.d_matrix();
    /// let result = some_matrix * d_full;
    /// ```
    ///
    /// You might be able to use:
    /// ```ignore
    /// let result = some_matrix.column_mut(i) *= udu.d[i];  // for each column
    /// ```
    ///
    /// # See Also
    ///
    /// * [`new`](Self::new) - Compute the UDU decomposition
    /// * [`d`](Self::d) - Access the diagonal elements as a vector (more efficient)
    #[must_use]
    pub fn d_matrix(&self) -> OMatrix<T, D, D> {
        OMatrix::from_diagonal(&self.d)
    }
}
