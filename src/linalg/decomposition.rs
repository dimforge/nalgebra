use crate::storage::Storage;
use crate::{
    Allocator, Bidiagonal, Cholesky, ColPivQR, ComplexField, DefaultAllocator, Dim, DimDiff,
    DimMin, DimMinimum, DimSub, FullPivLU, Hessenberg, LU, Matrix, OMatrix, QR, RealField, SVD,
    Schur, SymmetricEigen, SymmetricTridiagonal, U1, UDU,
};

/// # Rectangular matrix decomposition
///
/// This section contains the methods for computing some common decompositions of rectangular
/// matrices with real or complex components. The following are currently supported:
///
/// | Decomposition            | Factors             | Details |
/// | -------------------------|---------------------|--------------|
/// | QR                       | `Q * R`             | `Q` is an unitary matrix, and `R` is upper-triangular. |
/// | QR with column pivoting  | `Q * R * P⁻¹`       | `Q` is an unitary matrix, and `R` is upper-triangular. `P` is a permutation matrix. |
/// | LU with partial pivoting | `P⁻¹ * L * U`       | `L` is lower-triangular with a diagonal filled with `1` and `U` is upper-triangular. `P` is a permutation matrix. |
/// | LU with full pivoting    | `P⁻¹ * L * U * Q⁻¹` | `L` is lower-triangular with a diagonal filled with `1` and `U` is upper-triangular. `P` and `Q` are permutation matrices. |
/// | SVD                      | `U * Σ * Vᵀ`        | `U` and `V` are two orthogonal matrices and `Σ` is a diagonal matrix containing the singular values. |
/// | Polar (Left Polar)       | `P' * U`            | `U` is semi-unitary/unitary and `P'` is a positive semi-definite Hermitian Matrix
impl<T: ComplexField, R: Dim, C: Dim, S: Storage<T, R, C>> Matrix<T, R, C, S> {
    /// Computes the bidiagonalization of a matrix using Householder reflections.
    ///
    /// The bidiagonalization reduces a matrix to bidiagonal form, which means it has
    /// non-zero elements only on the main diagonal and either the first super-diagonal
    /// (if more rows than columns) or the first sub-diagonal (if more columns than rows).
    /// This decomposition is primarily used as an intermediate step in computing the
    /// Singular Value Decomposition (SVD).
    ///
    /// # When to use
    ///
    /// Use bidiagonalization when you need:
    /// - An intermediate step toward computing the SVD
    /// - A more efficient representation of a matrix for certain iterative algorithms
    /// - To analyze the structure of a matrix transformation
    ///
    /// For most applications, you'll want to use `svd()` directly rather than `bidiagonalize()`.
    ///
    /// # Example
    ///
    /// ```
    /// use nalgebra::Matrix3x2;
    ///
    /// let m = Matrix3x2::new(
    ///     1.0, 2.0,
    ///     3.0, 4.0,
    ///     5.0, 6.0,
    /// );
    ///
    /// let bidiag = m.bidiagonalize();
    ///
    /// // The bidiagonal matrix has non-zero elements only on the diagonal
    /// // and the first super-diagonal
    /// let bd = bidiag.diagonal();
    /// let off_diag = bidiag.off_diagonal();
    ///
    /// assert_eq!(bd.len(), 2);
    /// assert_eq!(off_diag.len(), 1);
    /// ```
    ///
    /// # See also
    ///
    /// - [`svd()`](Self::svd) - Computes the full Singular Value Decomposition
    /// - [`try_svd()`](Self::try_svd) - SVD with custom convergence parameters
    pub fn bidiagonalize(self) -> Bidiagonal<T, R, C>
    where
        R: DimMin<C>,
        DimMinimum<R, C>: DimSub<U1>,
        DefaultAllocator: Allocator<R, C>
            + Allocator<C>
            + Allocator<R>
            + Allocator<DimMinimum<R, C>>
            + Allocator<DimDiff<DimMinimum<R, C>, U1>>,
    {
        Bidiagonal::new(self.into_owned())
    }

    /// Computes the LU decomposition with full pivoting of a matrix.
    ///
    /// The LU decomposition with full pivoting factorizes a matrix `A` into the form
    /// `P * A * Q = L * U`, where:
    /// - `L` is a lower-triangular matrix with ones on the diagonal
    /// - `U` is an upper-triangular matrix
    /// - `P` and `Q` are permutation matrices (row and column permutations)
    ///
    /// Full pivoting performs both row and column exchanges to improve numerical stability,
    /// making this the most stable LU variant but also the most computationally expensive.
    ///
    /// # When to use
    ///
    /// Use full pivoting LU when you need:
    /// - Maximum numerical stability for ill-conditioned matrices
    /// - Accurate rank determination
    /// - To solve systems where partial pivoting might be insufficient
    ///
    /// For most applications, partial pivoting (`lu()`) offers a better balance of
    /// speed and stability.
    ///
    /// # Example
    ///
    /// ```
    /// use nalgebra::Matrix3;
    ///
    /// let m = Matrix3::new(
    ///     2.0, 1.0, 1.0,
    ///     4.0, 3.0, 3.0,
    ///     8.0, 7.0, 9.0,
    /// );
    ///
    /// let lu = m.full_piv_lu();
    ///
    /// // Extract the L and U factors
    /// let l = lu.l();
    /// let u = lu.u();
    ///
    /// // Verify the decomposition: P * A * Q = L * U
    /// let reconstructed = lu.p() * m * lu.q();
    /// assert!(reconstructed.relative_eq(&(l * u), 1e-10, 1e-10));
    /// ```
    ///
    /// # Practical application: solving a linear system
    ///
    /// ```
    /// use nalgebra::{Matrix3, Vector3};
    ///
    /// let a = Matrix3::new(
    ///     2.0, 1.0, 1.0,
    ///     4.0, 3.0, 3.0,
    ///     8.0, 7.0, 9.0,
    /// );
    /// let b = Vector3::new(2.0, 8.0, 22.0);
    ///
    /// let lu = a.full_piv_lu();
    /// let x = lu.solve(&b).expect("Linear resolution failed.");
    ///
    /// // Verify the solution
    /// assert!(a * x.relative_eq(&b, 1e-10, 1e-10));
    /// ```
    ///
    /// # See also
    ///
    /// - [`lu()`](Self::lu) - LU decomposition with partial (row) pivoting (faster)
    /// - [`qr()`](Self::qr) - QR decomposition, alternative for solving linear systems
    /// - [`svd()`](Self::svd) - SVD for solving rank-deficient systems
    pub fn full_piv_lu(self) -> FullPivLU<T, R, C>
    where
        R: DimMin<C>,
        DefaultAllocator: Allocator<R, C> + Allocator<DimMinimum<R, C>>,
    {
        FullPivLU::new(self.into_owned())
    }

    /// Computes the LU decomposition with partial (row) pivoting of a matrix.
    ///
    /// The LU decomposition with partial pivoting factorizes a matrix `A` into the form
    /// `P * A = L * U`, where:
    /// - `L` is a lower-triangular matrix with ones on the diagonal
    /// - `U` is an upper-triangular matrix
    /// - `P` is a permutation matrix representing row exchanges
    ///
    /// This is the standard LU decomposition used in most numerical libraries. It offers
    /// good numerical stability with reasonable computational cost, making it suitable for
    /// most applications.
    ///
    /// # When to use
    ///
    /// Use LU decomposition when you need to:
    /// - Solve linear systems of equations `Ax = b`
    /// - Compute the determinant of a matrix
    /// - Compute the inverse of a matrix
    /// - Solve multiple systems with the same matrix but different right-hand sides
    ///
    /// LU decomposition is generally faster than QR for square systems and provides
    /// good numerical stability for most well-conditioned matrices.
    ///
    /// # Example
    ///
    /// ```
    /// use nalgebra::Matrix3;
    ///
    /// let m = Matrix3::new(
    ///     1.0, 2.0, 3.0,
    ///     4.0, 5.0, 6.0,
    ///     7.0, 8.0, 10.0,
    /// );
    ///
    /// let lu = m.lu();
    ///
    /// // Check if the matrix is invertible
    /// assert!(lu.is_invertible());
    ///
    /// // Extract the L and U factors
    /// let l = lu.l();
    /// let u = lu.u();
    ///
    /// // Verify the decomposition: P * A = L * U
    /// let reconstructed = lu.p() * m;
    /// assert!(reconstructed.relative_eq(&(l * u), 1e-10, 1e-10));
    /// ```
    ///
    /// # Practical applications
    ///
    /// **Solving a linear system:**
    /// ```
    /// use nalgebra::{Matrix3, Vector3};
    ///
    /// let a = Matrix3::new(
    ///     2.0, 1.0, 1.0,
    ///     4.0, -6.0, 0.0,
    ///     -2.0, 7.0, 2.0,
    /// );
    /// let b = Vector3::new(5.0, -2.0, 9.0);
    ///
    /// let lu = a.lu();
    /// let x = lu.solve(&b).expect("Linear system has no solution");
    ///
    /// // Verify the solution
    /// assert!((a * x).relative_eq(&b, 1e-10, 1e-10));
    /// ```
    ///
    /// **Computing the determinant:**
    /// ```
    /// use nalgebra::Matrix3;
    ///
    /// let m = Matrix3::new(
    ///     1.0, 2.0, 3.0,
    ///     4.0, 5.0, 6.0,
    ///     7.0, 8.0, 9.0,
    /// );
    ///
    /// let lu = m.lu();
    /// let det = lu.determinant();
    ///
    /// assert!(det.abs() < 1e-10); // Matrix is singular
    /// ```
    ///
    /// # See also
    ///
    /// - [`full_piv_lu()`](Self::full_piv_lu) - LU with full pivoting for maximum stability
    /// - [`qr()`](Self::qr) - QR decomposition, better for overdetermined systems
    /// - [`cholesky()`](Self::cholesky) - More efficient for positive-definite matrices
    pub fn lu(self) -> LU<T, R, C>
    where
        R: DimMin<C>,
        DefaultAllocator: Allocator<R, C> + Allocator<DimMinimum<R, C>>,
    {
        LU::new(self.into_owned())
    }

    /// Computes the QR decomposition of a matrix.
    ///
    /// The QR decomposition factorizes a matrix `A` into the form `A = Q * R`, where:
    /// - `Q` is an orthogonal (or unitary for complex matrices) matrix
    /// - `R` is an upper-triangular matrix
    ///
    /// The QR decomposition is particularly useful for solving least-squares problems and
    /// for numerically stable solutions of linear systems, especially for tall matrices
    /// (more rows than columns).
    ///
    /// # When to use
    ///
    /// Use QR decomposition when you need to:
    /// - Solve overdetermined linear systems (least-squares problems)
    /// - Compute orthonormal bases
    /// - Solve linear systems where numerical stability is crucial
    /// - Compute eigenvalues (via the QR algorithm)
    ///
    /// QR is more numerically stable than LU decomposition but slightly more expensive
    /// computationally. It's the preferred method for rectangular matrices and
    /// ill-conditioned systems.
    ///
    /// # Example
    ///
    /// ```
    /// use nalgebra::Matrix3;
    ///
    /// let m = Matrix3::new(
    ///     1.0, 2.0, 3.0,
    ///     4.0, 5.0, 6.0,
    ///     7.0, 8.0, 9.0,
    /// );
    ///
    /// let qr = m.qr();
    ///
    /// // Extract the Q and R matrices
    /// let q = qr.q();
    /// let r = qr.r();
    ///
    /// // Q is orthogonal: Q * Q^T = Identity
    /// assert!(q.is_orthogonal(1e-10));
    ///
    /// // Verify the decomposition: A = Q * R
    /// assert!(m.relative_eq(&(q * r), 1e-10, 1e-10));
    /// ```
    ///
    /// # Practical applications
    ///
    /// **Solving a linear system:**
    /// ```
    /// use nalgebra::{Matrix3, Vector3};
    ///
    /// let a = Matrix3::new(
    ///     1.0, 2.0, 1.0,
    ///     2.0, 1.0, 3.0,
    ///     3.0, 1.0, 2.0,
    /// );
    /// let b = Vector3::new(1.0, 2.0, 3.0);
    ///
    /// let qr = a.qr();
    /// let x = qr.solve(&b).expect("QR solve failed");
    ///
    /// // Verify the solution
    /// assert!((a * x).relative_eq(&b, 1e-10, 1e-10));
    /// ```
    ///
    /// **Solving a least-squares problem (overdetermined system):**
    /// ```
    /// use nalgebra::{Matrix3x2, Vector3};
    ///
    /// // More equations than unknowns (3 equations, 2 unknowns)
    /// let a = Matrix3x2::new(
    ///     1.0, 0.0,
    ///     1.0, 1.0,
    ///     1.0, 2.0,
    /// );
    /// let b = Vector3::new(1.0, 2.0, 2.5);
    ///
    /// let qr = a.qr();
    /// // Finds the least-squares solution
    /// let x = qr.solve(&b).expect("QR solve failed");
    ///
    /// // x minimizes ||Ax - b||
    /// let residual = a * x - b;
    /// assert!(residual.norm() < 0.5);
    /// ```
    ///
    /// # See also
    ///
    /// - [`col_piv_qr()`](Self::col_piv_qr) - QR with column pivoting for rank-revealing
    /// - [`lu()`](Self::lu) - LU decomposition, faster for square well-conditioned systems
    /// - [`svd()`](Self::svd) - SVD for even more robust least-squares solutions
    pub fn qr(self) -> QR<T, R, C>
    where
        R: DimMin<C>,
        DefaultAllocator: Allocator<R, C> + Allocator<R> + Allocator<DimMinimum<R, C>>,
    {
        QR::new(self.into_owned())
    }

    /// Computes the QR decomposition with column pivoting of a matrix.
    ///
    /// The QR decomposition with column pivoting factorizes a matrix `A` into the form
    /// `A * P = Q * R`, where:
    /// - `Q` is an orthogonal (or unitary for complex matrices) matrix
    /// - `R` is an upper-triangular matrix
    /// - `P` is a permutation matrix representing column exchanges
    ///
    /// Column pivoting improves numerical stability and makes this decomposition
    /// "rank-revealing", meaning it can accurately determine the rank of a matrix
    /// even in the presence of numerical errors.
    ///
    /// # When to use
    ///
    /// Use QR with column pivoting when you need to:
    /// - Determine the rank of a matrix accurately
    /// - Solve rank-deficient least-squares problems
    /// - Find a numerically stable orthonormal basis
    /// - Handle matrices where column ordering affects numerical stability
    ///
    /// This is more expensive than regular QR but provides better rank detection
    /// and can handle rank-deficient matrices more gracefully.
    ///
    /// # Example
    ///
    /// ```
    /// use nalgebra::Matrix3;
    ///
    /// let m = Matrix3::new(
    ///     1.0, 2.0, 3.0,
    ///     4.0, 5.0, 6.0,
    ///     7.0, 8.0, 9.0,
    /// );
    ///
    /// let qr = m.col_piv_qr();
    ///
    /// // Check if the matrix is full rank
    /// assert!(!qr.is_invertible());
    ///
    /// // Get the rank of the matrix
    /// // (rank is determined by counting non-zero diagonal elements of R)
    /// let r = qr.r();
    /// let rank = r.diagonal().iter().filter(|&&x| x.abs() > 1e-10).count();
    /// assert_eq!(rank, 2); // Matrix is rank-deficient
    /// ```
    ///
    /// # Practical application: solving a rank-deficient system
    ///
    /// ```
    /// use nalgebra::{Matrix3, Vector3};
    ///
    /// // This matrix is rank-deficient (third row = 2*first row + second row)
    /// let a = Matrix3::new(
    ///     1.0, 0.0, 1.0,
    ///     0.0, 1.0, 1.0,
    ///     2.0, 1.0, 3.0,
    /// );
    /// let b = Vector3::new(1.0, 2.0, 4.0);
    ///
    /// let qr = a.col_piv_qr();
    ///
    /// // Can still solve if the system is consistent
    /// if let Some(x) = qr.solve(&b) {
    ///     // Verify the solution
    ///     assert!((a * x).relative_eq(&b, 1e-10, 1e-10));
    /// }
    /// ```
    ///
    /// # See also
    ///
    /// - [`qr()`](Self::qr) - Standard QR decomposition (faster, no pivoting)
    /// - [`full_piv_lu()`](Self::full_piv_lu) - Alternative rank-revealing decomposition
    /// - [`svd()`](Self::svd) - Most robust rank-revealing decomposition
    pub fn col_piv_qr(self) -> ColPivQR<T, R, C>
    where
        R: DimMin<C>,
        DefaultAllocator: Allocator<R, C>
            + Allocator<R>
            + Allocator<DimMinimum<R, C>>
            + Allocator<DimMinimum<R, C>>,
    {
        ColPivQR::new(self.into_owned())
    }

    /// Computes the Singular Value Decomposition (SVD) of a matrix.
    ///
    /// The SVD factorizes a matrix `A` into the form `A = U * Σ * Vᵀ`, where:
    /// - `U` is an orthogonal matrix containing the left singular vectors
    /// - `Σ` is a diagonal matrix containing the singular values (in descending order)
    /// - `V` is an orthogonal matrix containing the right singular vectors
    ///
    /// The SVD is the most robust matrix decomposition available and can handle any matrix,
    /// including rank-deficient and rectangular matrices. It's fundamental to many applications
    /// in numerical linear algebra, signal processing, and machine learning.
    ///
    /// # Parameters
    ///
    /// * `compute_u` - Whether to compute the left singular vectors (matrix `U`)
    /// * `compute_v` - Whether to compute the right singular vectors (matrix `V`)
    ///
    /// Set these to `false` if you only need the singular values, which saves computation time.
    ///
    /// # When to use
    ///
    /// Use SVD when you need to:
    /// - Solve least-squares problems robustly, even for rank-deficient matrices
    /// - Compute the pseudo-inverse of a matrix
    /// - Determine the rank of a matrix accurately
    /// - Find the best low-rank approximation of a matrix
    /// - Analyze the range and null space of a matrix
    /// - Perform Principal Component Analysis (PCA)
    /// - Compute matrix norms and condition numbers
    ///
    /// # Example
    ///
    /// ```
    /// use nalgebra::Matrix3x2;
    ///
    /// let m = Matrix3x2::new(
    ///     1.0, 2.0,
    ///     3.0, 4.0,
    ///     5.0, 6.0,
    /// );
    ///
    /// // Compute full SVD with both U and V
    /// let svd = m.svd(true, true);
    ///
    /// // Get singular values (in descending order)
    /// let singular_values = svd.singular_values;
    /// assert!(singular_values[0] >= singular_values[1]);
    ///
    /// // Verify the decomposition: A = U * Σ * Vᵀ
    /// let reconstructed = svd.recompose().unwrap();
    /// assert!(m.relative_eq(&reconstructed, 1e-10, 1e-10));
    /// ```
    ///
    /// # Practical applications
    ///
    /// **Computing the pseudo-inverse:**
    /// ```
    /// use nalgebra::Matrix2x3;
    ///
    /// let m = Matrix2x3::new(
    ///     1.0, 2.0, 3.0,
    ///     4.0, 5.0, 6.0,
    /// );
    ///
    /// let svd = m.svd(true, true);
    /// let pseudo_inv = svd.pseudo_inverse(1e-10).unwrap();
    ///
    /// // The pseudo-inverse satisfies: A * A⁺ * A = A
    /// let should_equal_m = &m * &pseudo_inv * &m;
    /// assert!(m.relative_eq(&should_equal_m, 1e-9, 1e-9));
    /// ```
    ///
    /// **Solving a least-squares problem:**
    /// ```
    /// use nalgebra::{Matrix3x2, Vector3};
    ///
    /// let a = Matrix3x2::new(
    ///     1.0, 2.0,
    ///     3.0, 4.0,
    ///     5.0, 6.0,
    /// );
    /// let b = Vector3::new(1.0, 2.0, 3.0);
    ///
    /// let svd = a.svd(true, true);
    /// let x = svd.solve(&b, 1e-10).unwrap();
    ///
    /// // x minimizes ||Ax - b||
    /// let residual = a * x - b;
    /// assert!(residual.norm() < 0.1);
    /// ```
    ///
    /// **Computing only singular values (faster):**
    /// ```
    /// use nalgebra::Matrix3;
    ///
    /// let m = Matrix3::identity();
    ///
    /// // Don't compute U and V, only singular values
    /// let svd = m.svd(false, false);
    /// let singular_values = svd.singular_values;
    ///
    /// // For an identity matrix, all singular values are 1
    /// assert!((singular_values[0] - 1.0).abs() < 1e-10);
    /// assert!((singular_values[1] - 1.0).abs() < 1e-10);
    /// assert!((singular_values[2] - 1.0).abs() < 1e-10);
    /// ```
    ///
    /// # See also
    ///
    /// - [`svd_unordered()`](Self::svd_unordered) - SVD without sorting singular values (slightly faster)
    /// - [`try_svd()`](Self::try_svd) - SVD with custom convergence parameters
    /// - [`qr()`](Self::qr) - Faster alternative for well-conditioned systems
    /// - [`polar()`](Self::polar) - Polar decomposition (uses SVD internally)
    pub fn svd(self, compute_u: bool, compute_v: bool) -> SVD<T, R, C>
    where
        R: DimMin<C>,
        DimMinimum<R, C>: DimSub<U1>, // for Bidiagonal.
        DefaultAllocator: Allocator<R, C>
            + Allocator<C>
            + Allocator<R>
            + Allocator<DimDiff<DimMinimum<R, C>, U1>>
            + Allocator<DimMinimum<R, C>, C>
            + Allocator<R, DimMinimum<R, C>>
            + Allocator<DimMinimum<R, C>>,
    {
        SVD::new(self.into_owned(), compute_u, compute_v)
    }

    /// Computes the Singular Value Decomposition (SVD) without sorting singular values.
    ///
    /// This function performs the same computation as [`svd()`](Self::svd), but the singular
    /// values are not guaranteed to be in any particular order. This saves a small amount
    /// of computation time if you don't need the values sorted.
    ///
    /// The SVD factorizes a matrix `A` into the form `A = U * Σ * Vᵀ`, where:
    /// - `U` is an orthogonal matrix containing the left singular vectors
    /// - `Σ` is a diagonal matrix containing the singular values (unsorted)
    /// - `V` is an orthogonal matrix containing the right singular vectors
    ///
    /// # Parameters
    ///
    /// * `compute_u` - Whether to compute the left singular vectors (matrix `U`)
    /// * `compute_v` - Whether to compute the right singular vectors (matrix `V`)
    ///
    /// # When to use
    ///
    /// Use `svd_unordered()` instead of `svd()` when:
    /// - You don't need the singular values in descending order
    /// - You want to save a small amount of computation time
    /// - You'll be processing all singular values anyway (e.g., computing matrix rank)
    ///
    /// For most applications, the difference in performance is negligible, so prefer
    /// [`svd()`](Self::svd) unless you have a specific reason to avoid sorting.
    ///
    /// # Example
    ///
    /// ```
    /// use nalgebra::Matrix3x2;
    ///
    /// let m = Matrix3x2::new(
    ///     1.0, 2.0,
    ///     3.0, 4.0,
    ///     5.0, 6.0,
    /// );
    ///
    /// let svd = m.svd_unordered(true, true);
    ///
    /// // Singular values are not in any particular order
    /// let singular_values = svd.singular_values;
    /// println!("Singular values (unsorted): {:?}", singular_values);
    ///
    /// // The decomposition still works correctly
    /// let reconstructed = svd.recompose().unwrap();
    /// assert!(m.relative_eq(&reconstructed, 1e-10, 1e-10));
    /// ```
    ///
    /// # See also
    ///
    /// - [`svd()`](Self::svd) - SVD with singular values sorted in descending order
    /// - [`try_svd_unordered()`](Self::try_svd_unordered) - SVD with custom convergence parameters
    pub fn svd_unordered(self, compute_u: bool, compute_v: bool) -> SVD<T, R, C>
    where
        R: DimMin<C>,
        DimMinimum<R, C>: DimSub<U1>, // for Bidiagonal.
        DefaultAllocator: Allocator<R, C>
            + Allocator<C>
            + Allocator<R>
            + Allocator<DimDiff<DimMinimum<R, C>, U1>>
            + Allocator<DimMinimum<R, C>, C>
            + Allocator<R, DimMinimum<R, C>>
            + Allocator<DimMinimum<R, C>>,
    {
        SVD::new_unordered(self.into_owned(), compute_u, compute_v)
    }

    /// Attempts to compute the Singular Value Decomposition (SVD) with custom convergence parameters.
    ///
    /// This function is similar to [`svd()`](Self::svd) but allows you to control the
    /// convergence criteria. It returns `None` if the algorithm fails to converge within
    /// the specified number of iterations. The singular values are sorted in descending order.
    ///
    /// # Arguments
    ///
    /// * `compute_u` - Set to `true` to compute the left singular vectors (matrix `U`)
    /// * `compute_v` - Set to `true` to compute the right singular vectors (matrix `V`)
    /// * `eps` - Tolerance used to determine when a value has converged to 0
    /// * `max_niter` - Maximum number of iterations. Returns `None` if exceeded.
    ///   Set to `0` for unlimited iterations (not recommended).
    ///
    /// # When to use
    ///
    /// Use `try_svd()` instead of `svd()` when:
    /// - You need finer control over convergence criteria
    /// - You want to handle convergence failures explicitly
    /// - You're working with particularly difficult matrices and need to adjust tolerances
    /// - You want to limit computation time by setting a maximum iteration count
    ///
    /// For most applications, the default `svd()` method is sufficient.
    ///
    /// # Example
    ///
    /// ```
    /// use nalgebra::Matrix3x2;
    ///
    /// let m = Matrix3x2::new(
    ///     1.0, 2.0,
    ///     3.0, 4.0,
    ///     5.0, 6.0,
    /// );
    ///
    /// // Try SVD with custom tolerance and iteration limit
    /// let eps = 1e-12;
    /// let max_iterations = 100;
    ///
    /// match m.try_svd(true, true, eps, max_iterations) {
    ///     Some(svd) => {
    ///         // Successfully computed SVD
    ///         let singular_values = svd.singular_values;
    ///         assert!(singular_values[0] >= singular_values[1]);
    ///     }
    ///     None => {
    ///         panic!("SVD failed to converge");
    ///     }
    /// }
    /// ```
    ///
    /// # Practical application: handling difficult matrices
    ///
    /// ```
    /// use nalgebra::Matrix2;
    ///
    /// let m = Matrix2::new(
    ///     1e-10, 2e-10,
    ///     3e-10, 4e-10,
    /// );
    ///
    /// // Use a looser tolerance for matrices with very small values
    /// let eps = 1e-15;
    /// let max_iterations = 200;
    ///
    /// if let Some(svd) = m.try_svd(true, true, eps, max_iterations) {
    ///     println!("Successfully computed SVD");
    ///     println!("Singular values: {:?}", svd.singular_values);
    /// }
    /// ```
    ///
    /// # See also
    ///
    /// - [`svd()`](Self::svd) - SVD with default convergence parameters
    /// - [`try_svd_unordered()`](Self::try_svd_unordered) - Without sorting singular values
    pub fn try_svd(
        self,
        compute_u: bool,
        compute_v: bool,
        eps: T::RealField,
        max_niter: usize,
    ) -> Option<SVD<T, R, C>>
    where
        R: DimMin<C>,
        DimMinimum<R, C>: DimSub<U1>, // for Bidiagonal.
        DefaultAllocator: Allocator<R, C>
            + Allocator<C>
            + Allocator<R>
            + Allocator<DimDiff<DimMinimum<R, C>, U1>>
            + Allocator<DimMinimum<R, C>, C>
            + Allocator<R, DimMinimum<R, C>>
            + Allocator<DimMinimum<R, C>>,
    {
        SVD::try_new(self.into_owned(), compute_u, compute_v, eps, max_niter)
    }

    /// Attempts to compute the SVD with custom convergence parameters, without sorting singular values.
    ///
    /// This function combines the features of [`try_svd()`](Self::try_svd) and
    /// [`svd_unordered()`](Self::svd_unordered): it allows custom convergence parameters
    /// and doesn't sort the resulting singular values, saving a small amount of computation time.
    ///
    /// # Arguments
    ///
    /// * `compute_u` - Set to `true` to compute the left singular vectors (matrix `U`)
    /// * `compute_v` - Set to `true` to compute the right singular vectors (matrix `V`)
    /// * `eps` - Tolerance used to determine when a value has converged to 0
    /// * `max_niter` - Maximum number of iterations. Returns `None` if exceeded.
    ///   Set to `0` for unlimited iterations (not recommended).
    ///
    /// # When to use
    ///
    /// Use this function when you need both:
    /// - Custom convergence control (like `try_svd`)
    /// - Unsorted singular values (like `svd_unordered`)
    ///
    /// # Example
    ///
    /// ```
    /// use nalgebra::Matrix3x2;
    ///
    /// let m = Matrix3x2::new(
    ///     1.0, 2.0,
    ///     3.0, 4.0,
    ///     5.0, 6.0,
    /// );
    ///
    /// let eps = 1e-12;
    /// let max_iterations = 100;
    ///
    /// if let Some(svd) = m.try_svd_unordered(true, true, eps, max_iterations) {
    ///     // Singular values are not sorted
    ///     println!("Singular values (unsorted): {:?}", svd.singular_values);
    /// } else {
    ///     println!("SVD failed to converge");
    /// }
    /// ```
    ///
    /// # See also
    ///
    /// - [`try_svd()`](Self::try_svd) - SVD with custom parameters and sorted singular values
    /// - [`svd_unordered()`](Self::svd_unordered) - SVD with default parameters, unsorted
    /// - [`svd()`](Self::svd) - Standard SVD with default parameters
    pub fn try_svd_unordered(
        self,
        compute_u: bool,
        compute_v: bool,
        eps: T::RealField,
        max_niter: usize,
    ) -> Option<SVD<T, R, C>>
    where
        R: DimMin<C>,
        DimMinimum<R, C>: DimSub<U1>, // for Bidiagonal.
        DefaultAllocator: Allocator<R, C>
            + Allocator<C>
            + Allocator<R>
            + Allocator<DimDiff<DimMinimum<R, C>, U1>>
            + Allocator<DimMinimum<R, C>, C>
            + Allocator<R, DimMinimum<R, C>>
            + Allocator<DimMinimum<R, C>>
            + Allocator<DimMinimum<R, C>>
            + Allocator<DimDiff<DimMinimum<R, C>, U1>>,
    {
        SVD::try_new_unordered(self.into_owned(), compute_u, compute_v, eps, max_niter)
    }

    /// Computes the Polar Decomposition of a matrix.
    ///
    /// The polar decomposition factorizes a matrix `A` into the form `A = U * P`, where:
    /// - `U` is a unitary/orthogonal matrix (or semi-unitary if `A` is not square)
    /// - `P` is a positive semi-definite Hermitian matrix
    ///
    /// This decomposition is analogous to writing a complex number in polar form (r * e^(iθ)),
    /// where `U` represents the "rotation" and `P` represents the "scaling".
    /// Internally, this uses the SVD: if `A = U * Σ * Vᵀ`, then the polar form is
    /// `A = (U * Vᵀ) * (V * Σ * Vᵀ)`.
    ///
    /// # When to use
    ///
    /// Use polar decomposition when you need to:
    /// - Separate rotation from scaling in a transformation
    /// - Find the nearest orthogonal matrix to a given matrix
    /// - Analyze deformations in mechanics or graphics
    /// - Compute matrix functions like the matrix square root
    ///
    /// # Example
    ///
    /// ```
    /// use nalgebra::Matrix2;
    ///
    /// let m = Matrix2::new(
    ///     3.0, 1.0,
    ///     1.0, 3.0,
    /// );
    ///
    /// let (u, p) = m.polar();
    ///
    /// // U is orthogonal
    /// assert!(u.is_orthogonal(1e-10));
    ///
    /// // P is positive semi-definite (all eigenvalues >= 0)
    /// let eigenvalues = p.symmetric_eigen().eigenvalues;
    /// assert!(eigenvalues.iter().all(|&x| x >= -1e-10));
    ///
    /// // Verify the decomposition: A = U * P
    /// assert!(m.relative_eq(&(u * p), 1e-10, 1e-10));
    /// ```
    ///
    /// # Practical application: finding nearest rotation matrix
    ///
    /// ```
    /// use nalgebra::Matrix3;
    ///
    /// // A matrix that's "almost" a rotation
    /// let almost_rotation = Matrix3::new(
    ///     0.9, -0.4, 0.1,
    ///     0.4,  0.9, 0.0,
    ///     0.0,  0.1, 1.0,
    /// );
    ///
    /// let (u, _p) = almost_rotation.polar();
    ///
    /// // U is the nearest orthogonal matrix to the input
    /// assert!(u.is_orthogonal(1e-10));
    /// assert!((u.determinant() - 1.0).abs() < 1e-10); // Proper rotation
    /// ```
    ///
    /// # See also
    ///
    /// - [`try_polar()`](Self::try_polar) - Polar decomposition with custom convergence parameters
    /// - [`svd()`](Self::svd) - Singular Value Decomposition (used internally)
    pub fn polar(self) -> (OMatrix<T, R, R>, OMatrix<T, R, C>)
    where
        R: DimMin<C>,
        DimMinimum<R, C>: DimSub<U1>, // for Bidiagonal.
        DefaultAllocator: Allocator<R, C>
            + Allocator<DimMinimum<R, C>, R>
            + Allocator<DimMinimum<R, C>>
            + Allocator<R, R>
            + Allocator<DimMinimum<R, C>, DimMinimum<R, C>>
            + Allocator<C>
            + Allocator<R>
            + Allocator<DimDiff<DimMinimum<R, C>, U1>>
            + Allocator<DimMinimum<R, C>, C>
            + Allocator<R, DimMinimum<R, C>>
            + Allocator<DimMinimum<R, C>>
            + Allocator<DimMinimum<R, C>>
            + Allocator<DimDiff<DimMinimum<R, C>, U1>>,
    {
        SVD::new_unordered(self.into_owned(), true, true)
            .to_polar()
            .unwrap()
    }

    /// Attempts to compute the Polar Decomposition with custom convergence parameters.
    ///
    /// This function is similar to [`polar()`](Self::polar) but allows you to control the
    /// convergence criteria for the underlying SVD computation. It returns `None` if the
    /// SVD algorithm fails to converge within the specified parameters.
    ///
    /// # Arguments
    ///
    /// * `eps` - Tolerance used to determine when a value has converged to 0 in the SVD
    /// * `max_niter` - Maximum number of iterations for the SVD algorithm
    ///
    /// # When to use
    ///
    /// Use `try_polar()` instead of `polar()` when:
    /// - You need to handle convergence failures explicitly
    /// - You're working with difficult matrices that might not converge
    /// - You want to limit computation time
    ///
    /// # Example
    ///
    /// ```
    /// use nalgebra::Matrix2;
    ///
    /// let m = Matrix2::new(
    ///     2.0, 1.0,
    ///     1.0, 2.0,
    /// );
    ///
    /// let eps = 1e-10;
    /// let max_iterations = 100;
    ///
    /// match m.try_polar(eps, max_iterations) {
    ///     Some((u, p)) => {
    ///         // Successfully computed polar decomposition
    ///         assert!(u.is_orthogonal(1e-10));
    ///         assert!(m.relative_eq(&(u * p), 1e-10, 1e-10));
    ///     }
    ///     None => {
    ///         println!("Polar decomposition failed to converge");
    ///     }
    /// }
    /// ```
    ///
    /// # See also
    ///
    /// - [`polar()`](Self::polar) - Polar decomposition with default convergence parameters
    /// - [`try_svd()`](Self::try_svd) - The underlying SVD computation
    pub fn try_polar(
        self,
        eps: T::RealField,
        max_niter: usize,
    ) -> Option<(OMatrix<T, R, R>, OMatrix<T, R, C>)>
    where
        R: DimMin<C>,
        DimMinimum<R, C>: DimSub<U1>, // for Bidiagonal.
        DefaultAllocator: Allocator<R, C>
            + Allocator<DimMinimum<R, C>, R>
            + Allocator<DimMinimum<R, C>>
            + Allocator<R, R>
            + Allocator<DimMinimum<R, C>, DimMinimum<R, C>>
            + Allocator<C>
            + Allocator<R>
            + Allocator<DimDiff<DimMinimum<R, C>, U1>>
            + Allocator<DimMinimum<R, C>, C>
            + Allocator<R, DimMinimum<R, C>>
            + Allocator<DimMinimum<R, C>>
            + Allocator<DimMinimum<R, C>>
            + Allocator<DimDiff<DimMinimum<R, C>, U1>>,
    {
        SVD::try_new_unordered(self.into_owned(), true, true, eps, max_niter)
            .and_then(|svd| svd.to_polar())
    }
}

/// # Square matrix decomposition
///
/// This section contains the methods for computing some common decompositions of square
/// matrices with real or complex components. The following are currently supported:
///
/// | Decomposition            | Factors                   | Details |
/// | -------------------------|---------------------------|--------------|
/// | Hessenberg               | `Q * H * Qᵀ`             | `Q` is a unitary matrix and `H` an upper-Hessenberg matrix. |
/// | Cholesky                 | `L * Lᵀ`                 | `L` is a lower-triangular matrix. |
/// | UDU                      | `U * D * Uᵀ`             | `U` is a upper-triangular matrix, and `D` a diagonal matrix. |
/// | Schur decomposition      | `Q * T * Qᵀ`             | `Q` is an unitary matrix and `T` a quasi-upper-triangular matrix. |
/// | Symmetric eigendecomposition | `Q ~ Λ ~ Qᵀ`   | `Q` is an unitary matrix, and `Λ` is a real diagonal matrix. |
/// | Symmetric tridiagonalization | `Q ~ T ~ Qᵀ`   | `Q` is an unitary matrix, and `T` is a tridiagonal matrix. |
impl<T: ComplexField, D: Dim, S: Storage<T, D, D>> Matrix<T, D, D, S> {
    /// Attempts to compute the Cholesky decomposition of a symmetric positive-definite matrix.
    ///
    /// The Cholesky decomposition factorizes a symmetric positive-definite matrix `A` into
    /// the form `A = L * Lᵀ`, where `L` is a lower-triangular matrix with positive diagonal
    /// elements. This is one of the most efficient matrix decompositions and is widely used
    /// in numerical computation.
    ///
    /// The input matrix is assumed to be symmetric, and only the lower-triangular part is read.
    /// Returns `None` if the matrix is not positive-definite.
    ///
    /// # When to use
    ///
    /// Use Cholesky decomposition when you have a symmetric positive-definite matrix and need to:
    /// - Solve linear systems `Ax = b` (2x faster than LU for SPD matrices)
    /// - Compute the determinant or inverse efficiently
    /// - Check if a matrix is positive-definite
    /// - Simulate random variables with a given covariance matrix
    /// - Solve least-squares problems (via normal equations)
    ///
    /// A matrix is positive-definite if all its eigenvalues are positive, which commonly
    /// occurs with covariance matrices, Gram matrices, and certain physics simulations.
    ///
    /// # Example
    ///
    /// ```
    /// use nalgebra::Matrix3;
    ///
    /// // A symmetric positive-definite matrix
    /// let m = Matrix3::new(
    ///     4.0, 2.0, 1.0,
    ///     2.0, 5.0, 3.0,
    ///     1.0, 3.0, 6.0,
    /// );
    ///
    /// match m.cholesky() {
    ///     Some(chol) => {
    ///         // Extract the L factor
    ///         let l = chol.l();
    ///
    ///         // Verify the decomposition: A = L * Lᵀ
    ///         assert!(m.relative_eq(&(l * l.transpose()), 1e-10, 1e-10));
    ///     }
    ///     None => {
    ///         println!("Matrix is not positive-definite");
    ///     }
    /// }
    /// ```
    ///
    /// # Practical applications
    ///
    /// **Solving a linear system:**
    /// ```
    /// use nalgebra::{Matrix3, Vector3};
    ///
    /// let a = Matrix3::new(
    ///     4.0, 2.0, 1.0,
    ///     2.0, 5.0, 3.0,
    ///     1.0, 3.0, 6.0,
    /// );
    /// let b = Vector3::new(1.0, 2.0, 3.0);
    ///
    /// if let Some(chol) = a.cholesky() {
    ///     let x = chol.solve(&b);
    ///     // Verify the solution
    ///     assert!((a * x).relative_eq(&b, 1e-10, 1e-10));
    /// }
    /// ```
    ///
    /// **Checking if a matrix is positive-definite:**
    /// ```
    /// use nalgebra::Matrix2;
    ///
    /// let positive_def = Matrix2::new(
    ///     2.0, 1.0,
    ///     1.0, 2.0,
    /// );
    /// assert!(positive_def.cholesky().is_some());
    ///
    /// let not_positive_def = Matrix2::new(
    ///     1.0, 2.0,
    ///     2.0, 1.0,
    /// );
    /// assert!(not_positive_def.cholesky().is_none());
    /// ```
    ///
    /// **Computing the determinant:**
    /// ```
    /// use nalgebra::Matrix3;
    ///
    /// let m = Matrix3::new(
    ///     4.0, 2.0, 1.0,
    ///     2.0, 5.0, 3.0,
    ///     1.0, 3.0, 6.0,
    /// );
    ///
    /// if let Some(chol) = m.cholesky() {
    ///     let det = chol.determinant();
    ///     println!("Determinant: {}", det);
    /// }
    /// ```
    ///
    /// # See also
    ///
    /// - [`lu()`](Self::lu) - General-purpose decomposition for non-symmetric matrices
    /// - [`udu()`](Self::udu) - Alternative for symmetric matrices, numerically robust
    /// - [`symmetric_eigen()`](Self::symmetric_eigen) - For computing eigenvalues/vectors
    pub fn cholesky(self) -> Option<Cholesky<T, D>>
    where
        DefaultAllocator: Allocator<D, D>,
    {
        Cholesky::new(self.into_owned())
    }

    /// Attempts to compute the UDU decomposition of a symmetric matrix.
    ///
    /// The UDU decomposition factorizes a symmetric matrix `A` into the form `A = U * D * Uᵀ`,
    /// where:
    /// - `U` is an upper-triangular matrix with ones on the diagonal
    /// - `D` is a diagonal matrix
    ///
    /// This decomposition is similar to Cholesky but works for indefinite matrices (matrices
    /// with both positive and negative eigenvalues). It's also more numerically stable than
    /// Cholesky when dealing with ill-conditioned matrices because it avoids computing
    /// square roots.
    ///
    /// The input matrix is assumed to be symmetric, and only the upper-triangular part is read.
    /// Returns `None` if the decomposition fails (very rare for symmetric matrices).
    ///
    /// # When to use
    ///
    /// Use UDU decomposition when you have a symmetric matrix that is:
    /// - Not positive-definite (Cholesky won't work)
    /// - Ill-conditioned (UDU is more stable than Cholesky)
    /// - Used in Kalman filtering or control theory applications
    ///
    /// UDU is particularly popular in numerical applications where numerical stability
    /// is critical, such as satellite navigation and aircraft control systems.
    ///
    /// # Example
    ///
    /// ```
    /// use nalgebra::Matrix3;
    ///
    /// // A symmetric but indefinite matrix
    /// let m = Matrix3::new(
    ///     2.0,  1.0, 0.0,
    ///     1.0,  2.0, 1.0,
    ///     0.0,  1.0, 2.0,
    /// );
    ///
    /// match m.udu() {
    ///     Some(udu) => {
    ///         // Extract U and D factors
    ///         let u = udu.u();
    ///         let d = udu.d();
    ///
    ///         // Verify the decomposition: A = U * D * Uᵀ
    ///         let reconstructed = &u * d.map(|e| e) * u.transpose();
    ///         assert!(m.relative_eq(&reconstructed, 1e-10, 1e-10));
    ///     }
    ///     None => {
    ///         println!("UDU decomposition failed");
    ///     }
    /// }
    /// ```
    ///
    /// # Practical application: solving with an indefinite matrix
    ///
    /// ```
    /// use nalgebra::{Matrix3, Vector3};
    ///
    /// // Symmetric indefinite matrix
    /// let a = Matrix3::new(
    ///     2.0,  1.0,  0.0,
    ///     1.0,  2.0,  1.0,
    ///     0.0,  1.0, -1.0,
    /// );
    /// let b = Vector3::new(1.0, 2.0, 3.0);
    ///
    /// // UDU works even though Cholesky would fail
    /// assert!(a.cholesky().is_none());
    ///
    /// if let Some(udu) = a.udu() {
    ///     let x = udu.solve(&b);
    ///     // Verify the solution
    ///     assert!((a * x).relative_eq(&b, 1e-9, 1e-9));
    /// }
    /// ```
    ///
    /// # See also
    ///
    /// - [`cholesky()`](Self::cholesky) - More efficient for positive-definite matrices
    /// - [`lu()`](Self::lu) - General-purpose for non-symmetric matrices
    /// - [`symmetric_eigen()`](Self::symmetric_eigen) - For eigenvalue decomposition
    pub fn udu(self) -> Option<UDU<T, D>>
    where
        T: RealField,
        DefaultAllocator: Allocator<D> + Allocator<D, D>,
    {
        UDU::new(self.into_owned())
    }

    /// Computes the Hessenberg decomposition of a square matrix.
    ///
    /// The Hessenberg decomposition transforms a matrix `A` into the form `A = Q * H * Qᵀ`,
    /// where:
    /// - `Q` is an orthogonal (or unitary for complex matrices) matrix
    /// - `H` is an upper-Hessenberg matrix (zero below the first subdiagonal)
    ///
    /// An upper-Hessenberg matrix has zeros below the first subdiagonal, making it
    /// "almost" upper-triangular. This form is particularly useful as an intermediate
    /// step in eigenvalue computations.
    ///
    /// # When to use
    ///
    /// Use Hessenberg decomposition when you need:
    /// - An intermediate step for computing eigenvalues (used by the QR algorithm)
    /// - A more compact form for iterative algorithms
    /// - To reduce computational cost in repeated matrix operations
    ///
    /// Most users won't need this directly; it's primarily used internally by
    /// eigenvalue algorithms. If you need eigenvalues, use [`symmetric_eigen()`](Self::symmetric_eigen)
    /// for symmetric matrices or the Schur decomposition for general matrices.
    ///
    /// # Example
    ///
    /// ```
    /// use nalgebra::Matrix3;
    ///
    /// let m = Matrix3::new(
    ///     1.0, 2.0, 3.0,
    ///     4.0, 5.0, 6.0,
    ///     7.0, 8.0, 9.0,
    /// );
    ///
    /// let hess = m.hessenberg();
    ///
    /// // Get the Hessenberg matrix H
    /// let h = hess.h();
    ///
    /// // H is upper-Hessenberg: zeros below first subdiagonal
    /// assert!(h[(2, 0)].abs() < 1e-10);
    ///
    /// // Get the Q matrix
    /// let q = hess.q();
    ///
    /// // Verify the decomposition: A = Q * H * Qᵀ
    /// let reconstructed = &q * &h * q.transpose();
    /// assert!(m.relative_eq(&reconstructed, 1e-10, 1e-10));
    /// ```
    ///
    /// # Practical application
    ///
    /// ```
    /// use nalgebra::Matrix4;
    ///
    /// let m = Matrix4::new(
    ///     4.0, 1.0, 2.0, 3.0,
    ///     1.0, 4.0, 1.0, 2.0,
    ///     2.0, 1.0, 4.0, 1.0,
    ///     3.0, 2.0, 1.0, 4.0,
    /// );
    ///
    /// let hess = m.hessenberg();
    /// let h = hess.h();
    ///
    /// // The Hessenberg form has a simpler structure
    /// println!("Hessenberg matrix:\n{}", h);
    ///
    /// // Verify it's upper-Hessenberg (all elements below
    /// // the first subdiagonal are zero)
    /// for i in 2..4 {
    ///     for j in 0..i-1 {
    ///         assert!(h[(i, j)].abs() < 1e-10);
    ///     }
    /// }
    /// ```
    ///
    /// # See also
    ///
    /// - [`schur()`](Self::schur) - Schur decomposition for computing eigenvalues
    /// - [`symmetric_eigen()`](Self::symmetric_eigen) - Eigendecomposition for symmetric matrices
    /// - [`qr()`](Self::qr) - QR decomposition
    pub fn hessenberg(self) -> Hessenberg<T, D>
    where
        D: DimSub<U1>,
        DefaultAllocator: Allocator<D, D> + Allocator<D> + Allocator<DimDiff<D, U1>>,
    {
        Hessenberg::new(self.into_owned())
    }

    /// Computes the Schur decomposition of a square matrix.
    ///
    /// The Schur decomposition transforms a matrix `A` into the form `A = Q * T * Qᵀ`, where:
    /// - `Q` is an orthogonal (or unitary for complex matrices) matrix
    /// - `T` is a quasi-upper-triangular matrix (upper-triangular for complex matrices,
    ///   block upper-triangular for real matrices with complex eigenvalues)
    ///
    /// The Schur form reveals the eigenvalues of the matrix: they appear on the diagonal
    /// (or as eigenvalues of 2×2 blocks) of the `T` matrix. This decomposition is fundamental
    /// for computing eigenvalues and is used in many numerical algorithms.
    ///
    /// # When to use
    ///
    /// Use Schur decomposition when you need to:
    /// - Compute eigenvalues of non-symmetric matrices
    /// - Compute matrix functions (exponential, logarithm, etc.)
    /// - Solve matrix equations like the Sylvester equation
    /// - Analyze the stability of dynamical systems
    ///
    /// For symmetric matrices, [`symmetric_eigen()`](Self::symmetric_eigen) is more
    /// efficient and provides eigenvectors directly.
    ///
    /// # Example
    ///
    /// ```
    /// use nalgebra::Matrix3;
    ///
    /// let m = Matrix3::new(
    ///     1.0, 2.0, 3.0,
    ///     4.0, 5.0, 6.0,
    ///     7.0, 8.0, 9.0,
    /// );
    ///
    /// let schur = m.schur();
    ///
    /// // Get the Q and T matrices
    /// let q = schur.unpack().0;
    /// let t = schur.unpack().1;
    ///
    /// // Q is orthogonal
    /// assert!(q.is_orthogonal(1e-10));
    ///
    /// // Verify the decomposition: A = Q * T * Qᵀ
    /// let reconstructed = &q * &t * q.transpose();
    /// assert!(m.relative_eq(&reconstructed, 1e-10, 1e-10));
    /// ```
    ///
    /// # Practical applications
    ///
    /// **Computing eigenvalues:**
    /// ```
    /// use nalgebra::Matrix2;
    ///
    /// let m = Matrix2::new(
    ///     4.0, -2.0,
    ///     1.0,  1.0,
    /// );
    ///
    /// let schur = m.schur();
    /// let eigenvalues = schur.complex_eigenvalues();
    ///
    /// println!("Eigenvalues: {:?}", eigenvalues);
    /// ```
    ///
    /// **Analyzing stability:**
    /// ```
    /// use nalgebra::Matrix2;
    ///
    /// // A dynamical system matrix
    /// let a = Matrix2::new(
    ///     -1.0,  2.0,
    ///     -2.0, -1.0,
    /// );
    ///
    /// let schur = a.schur();
    /// let eigenvalues = schur.complex_eigenvalues();
    ///
    /// // System is stable if all eigenvalues have negative real parts
    /// let is_stable = eigenvalues.iter().all(|e| e.re < 0.0);
    /// assert!(is_stable);
    /// ```
    ///
    /// # See also
    ///
    /// - [`try_schur()`](Self::try_schur) - Schur decomposition with custom convergence parameters
    /// - [`symmetric_eigen()`](Self::symmetric_eigen) - More efficient for symmetric matrices
    /// - [`hessenberg()`](Self::hessenberg) - Intermediate step used by Schur decomposition
    pub fn schur(self) -> Schur<T, D>
    where
        D: DimSub<U1>, // For Hessenberg.
        DefaultAllocator: Allocator<D, DimDiff<D, U1>>
            + Allocator<DimDiff<D, U1>>
            + Allocator<D, D>
            + Allocator<D>,
    {
        Schur::new(self.into_owned())
    }

    /// Attempts to compute the Schur decomposition with custom convergence parameters.
    ///
    /// This function is similar to [`schur()`](Self::schur) but allows you to control the
    /// convergence criteria. It returns `None` if the algorithm fails to converge within
    /// the specified number of iterations.
    ///
    /// If you only need eigenvalues, it's more efficient to call the matrix method
    /// `.eigenvalues()` instead.
    ///
    /// # Arguments
    ///
    /// * `eps` - Tolerance used to determine when a value has converged to 0
    /// * `max_niter` - Maximum number of iterations. Returns `None` if exceeded.
    ///   Set to `0` for unlimited iterations (not recommended).
    ///
    /// # When to use
    ///
    /// Use `try_schur()` instead of `schur()` when:
    /// - You need to handle convergence failures explicitly
    /// - You're working with difficult matrices (e.g., nearly defective)
    /// - You want to limit computation time
    /// - You need finer control over numerical precision
    ///
    /// # Example
    ///
    /// ```
    /// use nalgebra::Matrix3;
    ///
    /// let m = Matrix3::new(
    ///     1.0, 2.0, 3.0,
    ///     4.0, 5.0, 6.0,
    ///     7.0, 8.0, 9.0,
    /// );
    ///
    /// let eps = 1e-10;
    /// let max_iterations = 100;
    ///
    /// match m.try_schur(eps, max_iterations) {
    ///     Some(schur) => {
    ///         let eigenvalues = schur.complex_eigenvalues();
    ///         println!("Eigenvalues: {:?}", eigenvalues);
    ///     }
    ///     None => {
    ///         println!("Schur decomposition failed to converge");
    ///     }
    /// }
    /// ```
    ///
    /// # Practical application: handling difficult matrices
    ///
    /// ```
    /// use nalgebra::Matrix2;
    ///
    /// let m = Matrix2::new(
    ///     1.0, 1e10,
    ///     1e-10, 1.0,
    /// );
    ///
    /// // Use looser tolerance for ill-conditioned matrices
    /// let eps = 1e-8;
    /// let max_iterations = 500;
    ///
    /// if let Some(schur) = m.try_schur(eps, max_iterations) {
    ///     println!("Successfully computed Schur decomposition");
    /// }
    /// ```
    ///
    /// # See also
    ///
    /// - [`schur()`](Self::schur) - Schur decomposition with default convergence parameters
    /// - [`symmetric_eigen()`](Self::symmetric_eigen) - For symmetric matrices
    pub fn try_schur(self, eps: T::RealField, max_niter: usize) -> Option<Schur<T, D>>
    where
        D: DimSub<U1>, // For Hessenberg.
        DefaultAllocator: Allocator<D, DimDiff<D, U1>>
            + Allocator<DimDiff<D, U1>>
            + Allocator<D, D>
            + Allocator<D>,
    {
        Schur::try_new(self.into_owned(), eps, max_niter)
    }

    /// Computes the eigendecomposition of a symmetric matrix.
    ///
    /// The symmetric eigendecomposition factorizes a symmetric matrix `A` into the form
    /// `A = Q * Λ * Qᵀ`, where:
    /// - `Q` is an orthogonal matrix whose columns are the eigenvectors
    /// - `Λ` is a diagonal matrix containing the real eigenvalues
    ///
    /// For symmetric matrices, all eigenvalues are real and eigenvectors are orthogonal,
    /// making this decomposition particularly clean and numerically stable. This is one of
    /// the most important decompositions in numerical linear algebra.
    ///
    /// Only the lower-triangular part (including the diagonal) of the matrix is read.
    ///
    /// # When to use
    ///
    /// Use symmetric eigendecomposition when you have a symmetric matrix and need to:
    /// - Find eigenvalues and eigenvectors
    /// - Perform Principal Component Analysis (PCA)
    /// - Diagonalize a symmetric matrix
    /// - Analyze quadratic forms
    /// - Compute matrix powers or functions
    /// - Determine definiteness (positive/negative definite, etc.)
    ///
    /// This is faster and more accurate than general eigendecomposition methods
    /// for symmetric matrices.
    ///
    /// # Example
    ///
    /// ```
    /// use nalgebra::Matrix3;
    ///
    /// // A symmetric matrix
    /// let m = Matrix3::new(
    ///     4.0, 2.0, 1.0,
    ///     2.0, 5.0, 3.0,
    ///     1.0, 3.0, 6.0,
    /// );
    ///
    /// let eigen = m.symmetric_eigen();
    ///
    /// // Get eigenvalues (real for symmetric matrices)
    /// let eigenvalues = eigen.eigenvalues;
    /// println!("Eigenvalues: {}", eigenvalues);
    ///
    /// // Get eigenvectors as columns of Q
    /// let eigenvectors = eigen.eigenvectors;
    ///
    /// // Eigenvectors are orthonormal
    /// assert!(eigenvectors.is_orthogonal(1e-10));
    ///
    /// // Verify the decomposition: A = Q * Λ * Qᵀ
    /// let reconstructed = &eigenvectors
    ///     * nalgebra::Matrix3::from_diagonal(&eigenvalues)
    ///     * eigenvectors.transpose();
    /// assert!(m.relative_eq(&reconstructed, 1e-10, 1e-10));
    /// ```
    ///
    /// # Practical applications
    ///
    /// **Principal Component Analysis (PCA):**
    /// ```
    /// use nalgebra::Matrix3;
    ///
    /// // Covariance matrix of some data
    /// let cov = Matrix3::new(
    ///     2.0, 1.0, 0.5,
    ///     1.0, 3.0, 1.0,
    ///     0.5, 1.0, 2.5,
    /// );
    ///
    /// let eigen = cov.symmetric_eigen();
    ///
    /// // Principal components are the eigenvectors
    /// // sorted by eigenvalue magnitude (largest first)
    /// let principal_components = eigen.eigenvectors;
    /// let explained_variance = eigen.eigenvalues;
    ///
    /// println!("Explained variance: {}", explained_variance);
    /// ```
    ///
    /// **Checking matrix definiteness:**
    /// ```
    /// use nalgebra::Matrix2;
    ///
    /// let m = Matrix2::new(
    ///     2.0, 1.0,
    ///     1.0, 2.0,
    /// );
    ///
    /// let eigen = m.symmetric_eigen();
    ///
    /// // Positive definite if all eigenvalues > 0
    /// let is_positive_definite = eigen.eigenvalues.iter().all(|&x| x > 0.0);
    /// assert!(is_positive_definite);
    /// ```
    ///
    /// **Computing matrix power:**
    /// ```
    /// use nalgebra::Matrix2;
    ///
    /// let m = Matrix2::new(
    ///     2.0, 1.0,
    ///     1.0, 2.0,
    /// );
    ///
    /// let eigen = m.symmetric_eigen();
    ///
    /// // Compute m^(1/2) (matrix square root)
    /// let sqrt_eigenvalues = eigen.eigenvalues.map(|x| x.sqrt());
    /// let m_sqrt = &eigen.eigenvectors
    ///     * Matrix2::from_diagonal(&sqrt_eigenvalues)
    ///     * eigen.eigenvectors.transpose();
    ///
    /// // Verify: m_sqrt * m_sqrt ≈ m
    /// assert!((&m_sqrt * &m_sqrt).relative_eq(&m, 1e-10, 1e-10));
    /// ```
    ///
    /// # See also
    ///
    /// - [`try_symmetric_eigen()`](Self::try_symmetric_eigen) - With custom convergence parameters
    /// - [`symmetric_tridiagonalize()`](Self::symmetric_tridiagonalize) - Intermediate form
    /// - [`schur()`](Self::schur) - For non-symmetric matrices
    pub fn symmetric_eigen(self) -> SymmetricEigen<T, D>
    where
        D: DimSub<U1>,
        DefaultAllocator:
            Allocator<D, D> + Allocator<DimDiff<D, U1>> + Allocator<D> + Allocator<DimDiff<D, U1>>,
    {
        SymmetricEigen::new(self.into_owned())
    }

    /// Computes the eigendecomposition of a symmetric matrix with custom convergence parameters.
    ///
    /// This function is similar to [`symmetric_eigen()`](Self::symmetric_eigen) but allows
    /// you to control the convergence criteria. It returns `None` if the algorithm fails to
    /// converge within the specified number of iterations.
    ///
    /// Only the lower-triangular part (including the diagonal) of the matrix is read.
    ///
    /// # Arguments
    ///
    /// * `eps` - Tolerance used to determine when a value has converged to 0
    /// * `max_niter` - Maximum number of iterations. Returns `None` if exceeded.
    ///   Set to `0` for unlimited iterations (not recommended).
    ///
    /// # When to use
    ///
    /// Use `try_symmetric_eigen()` instead of `symmetric_eigen()` when:
    /// - You need to handle convergence failures explicitly
    /// - You're working with difficult matrices
    /// - You want to limit computation time
    /// - You need custom tolerance levels
    ///
    /// # Example
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
    /// let eps = 1e-12;
    /// let max_iterations = 100;
    ///
    /// match m.try_symmetric_eigen(eps, max_iterations) {
    ///     Some(eigen) => {
    ///         println!("Eigenvalues: {}", eigen.eigenvalues);
    ///         println!("Converged successfully");
    ///     }
    ///     None => {
    ///         println!("Eigendecomposition failed to converge");
    ///     }
    /// }
    /// ```
    ///
    /// # Practical application
    ///
    /// ```
    /// use nalgebra::Matrix2;
    ///
    /// // Matrix with very small off-diagonal elements
    /// let m = Matrix2::new(
    ///     1.0, 1e-12,
    ///     1e-12, 2.0,
    /// );
    ///
    /// // Use appropriate tolerance
    /// let eps = 1e-15;
    /// let max_iterations = 50;
    ///
    /// if let Some(eigen) = m.try_symmetric_eigen(eps, max_iterations) {
    ///     // Successfully computed eigendecomposition
    ///     assert!(eigen.eigenvalues[0] > 0.0);
    ///     assert!(eigen.eigenvalues[1] > 0.0);
    /// }
    /// ```
    ///
    /// # See also
    ///
    /// - [`symmetric_eigen()`](Self::symmetric_eigen) - With default convergence parameters
    /// - [`try_schur()`](Self::try_schur) - For non-symmetric matrices
    pub fn try_symmetric_eigen(
        self,
        eps: T::RealField,
        max_niter: usize,
    ) -> Option<SymmetricEigen<T, D>>
    where
        D: DimSub<U1>,
        DefaultAllocator:
            Allocator<D, D> + Allocator<DimDiff<D, U1>> + Allocator<D> + Allocator<DimDiff<D, U1>>,
    {
        SymmetricEigen::try_new(self.into_owned(), eps, max_niter)
    }

    /// Computes the tridiagonalization of a symmetric matrix.
    ///
    /// The symmetric tridiagonalization transforms a symmetric matrix `A` into the form
    /// `A = Q * T * Qᵀ`, where:
    /// - `Q` is an orthogonal matrix
    /// - `T` is a tridiagonal matrix (non-zero only on main diagonal and adjacent diagonals)
    ///
    /// A tridiagonal matrix has non-zero elements only on the main diagonal and the two
    /// adjacent diagonals (one above and one below). This form is particularly useful as
    /// an intermediate step in computing eigenvalues of symmetric matrices.
    ///
    /// Only the lower-triangular part (including the diagonal) of the matrix is read.
    ///
    /// # When to use
    ///
    /// Use symmetric tridiagonalization when you need:
    /// - An intermediate step for computing eigenvalues (used by symmetric eigendecomposition)
    /// - A more compact representation for iterative algorithms
    /// - To analyze the structure of symmetric matrices
    ///
    /// Most users won't need this directly; it's primarily used internally by the
    /// [`symmetric_eigen()`](Self::symmetric_eigen) method. If you need eigenvalues,
    /// use that method instead.
    ///
    /// # Example
    ///
    /// ```
    /// use nalgebra::Matrix4;
    ///
    /// // A symmetric matrix
    /// let m = Matrix4::new(
    ///     4.0, 2.0, 1.0, 0.5,
    ///     2.0, 5.0, 3.0, 1.0,
    ///     1.0, 3.0, 6.0, 2.0,
    ///     0.5, 1.0, 2.0, 3.0,
    /// );
    ///
    /// let tridiag = m.symmetric_tridiagonalize();
    ///
    /// // Get the tridiagonal matrix T
    /// let t = tridiag.recompose();
    ///
    /// // T is tridiagonal: all elements more than one position
    /// // away from the diagonal are zero
    /// for i in 0..4 {
    ///     for j in 0..4 {
    ///         if i.abs_diff(j) > 1 {
    ///             assert!(t[(i, j)].abs() < 1e-10);
    ///         }
    ///     }
    /// }
    /// ```
    ///
    /// # Practical application: viewing the structure
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
    /// let tridiag = m.symmetric_tridiagonalize();
    ///
    /// // Get the diagonal and off-diagonal elements
    /// let diagonal = tridiag.diagonal();
    /// let off_diagonal = tridiag.off_diagonal();
    ///
    /// println!("Diagonal: {:?}", diagonal);
    /// println!("Off-diagonal: {:?}", off_diagonal);
    ///
    /// // The tridiagonal form is much simpler
    /// let t = tridiag.recompose();
    /// println!("Tridiagonal matrix:\n{}", t);
    /// ```
    ///
    /// # See also
    ///
    /// - [`symmetric_eigen()`](Self::symmetric_eigen) - Eigendecomposition (uses this internally)
    /// - [`hessenberg()`](Self::hessenberg) - Similar reduction for general matrices
    /// - [`bidiagonalize()`](Self::bidiagonalize) - For rectangular matrices
    pub fn symmetric_tridiagonalize(self) -> SymmetricTridiagonal<T, D>
    where
        D: DimSub<U1>,
        DefaultAllocator: Allocator<D, D> + Allocator<DimDiff<D, U1>>,
    {
        SymmetricTridiagonal::new(self.into_owned())
    }
}
