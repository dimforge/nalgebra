use num::Zero;
#[cfg(feature = "serde-serialize-no-std")]
use serde::{Deserialize, Serialize};

use crate::allocator::{Allocator, Reallocator};
use crate::base::{DefaultAllocator, Matrix, OMatrix, OVector, Unit};
use crate::constraint::{SameNumberOfRows, ShapeConstraint};
use crate::dimension::{Const, Dim, DimMin, DimMinimum};
use crate::storage::{Storage, StorageMut};
use simba::scalar::ComplexField;

use crate::geometry::Reflection;
use crate::linalg::householder;
use std::mem::MaybeUninit;

/// The QR decomposition of a general matrix.
///
/// # What is QR Decomposition?
///
/// The QR decomposition is a way to factorize any matrix `A` (not necessarily square) into
/// the product of two matrices:
///
/// ```text
/// A = Q * R
/// ```
///
/// where:
/// - `Q` is an **orthogonal matrix** (its columns are orthonormal vectors: `Q^T * Q = I`)
/// - `R` is an **upper triangular matrix** (all entries below the diagonal are zero)
///
/// For a matrix with more rows than columns (m > n), `Q` is m×n and `R` is n×n.
///
/// # When to Use It?
///
/// QR decomposition is particularly useful for:
/// - **Solving linear systems**: Especially overdetermined systems (more equations than unknowns)
/// - **Least squares problems**: Finding the best-fit solution when exact solutions don't exist
/// - **Orthogonalization**: Converting a set of vectors into orthonormal vectors (Gram-Schmidt)
/// - **Eigenvalue computation**: QR algorithm for finding eigenvalues
/// - **Numerical stability**: More stable than normal equations for least squares
///
/// # How It Works
///
/// This implementation uses **Householder reflections**, which is numerically stable and
/// efficient. Each reflection zeros out elements below the diagonal one column at a time.
///
/// # Example: Basic decomposition
///
/// ```
/// use nalgebra::{Matrix3x2, linalg::QR};
///
/// let m = Matrix3x2::new(
///     1.0, 2.0,
///     3.0, 4.0,
///     5.0, 6.0,
/// );
///
/// // Compute the QR decomposition
/// let qr = QR::new(m.clone());
///
/// // Extract Q and R
/// let q = qr.q();
/// let r = qr.r();
///
/// // Verify: Q * R should equal the original matrix
/// let reconstructed = q * r;
/// assert!((reconstructed - m).norm() < 1e-10);
///
/// // Verify Q is orthogonal: Q^T * Q = I (size matches Q's column dimension)
/// let qtq = q.transpose() * q;
/// let identity2x2 = Matrix2::identity();
/// assert!((qtq - identity2x2).norm() < 1e-10);
/// ```
#[cfg_attr(feature = "serde-serialize-no-std", derive(Serialize, Deserialize))]
#[cfg_attr(
    feature = "serde-serialize-no-std",
    serde(bound(serialize = "DefaultAllocator: Allocator<R, C> +
                           Allocator<DimMinimum<R, C>>,
         OMatrix<T, R, C>: Serialize,
         OVector<T, DimMinimum<R, C>>: Serialize"))
)]
#[cfg_attr(
    feature = "serde-serialize-no-std",
    serde(bound(deserialize = "DefaultAllocator: Allocator<R, C> +
                           Allocator<DimMinimum<R, C>>,
         OMatrix<T, R, C>: Deserialize<'de>,
         OVector<T, DimMinimum<R, C>>: Deserialize<'de>"))
)]
#[cfg_attr(feature = "defmt", derive(defmt::Format))]
#[derive(Clone, Debug)]
pub struct QR<T: ComplexField, R: DimMin<C>, C: Dim>
where
    DefaultAllocator: Allocator<R, C> + Allocator<DimMinimum<R, C>>,
{
    qr: OMatrix<T, R, C>,
    diag: OVector<T, DimMinimum<R, C>>,
}

impl<T: ComplexField, R: DimMin<C>, C: Dim> Copy for QR<T, R, C>
where
    DefaultAllocator: Allocator<R, C> + Allocator<DimMinimum<R, C>>,
    OMatrix<T, R, C>: Copy,
    OVector<T, DimMinimum<R, C>>: Copy,
{
}

impl<T: ComplexField, R: DimMin<C>, C: Dim> QR<T, R, C>
where
    DefaultAllocator: Allocator<R, C> + Allocator<R> + Allocator<DimMinimum<R, C>>,
{
    /// Computes the QR decomposition using Householder reflections.
    ///
    /// This is the primary way to create a QR decomposition. It works for any matrix
    /// (square or rectangular) and uses the numerically stable Householder reflection method.
    ///
    /// # How It Works
    ///
    /// The decomposition factors the input matrix `A` into `A = Q * R` where:
    /// - `Q` is an orthogonal matrix (columns are orthonormal)
    /// - `R` is upper triangular
    ///
    /// The algorithm uses Householder reflections to progressively eliminate entries below
    /// the diagonal, column by column. This is more stable than Gram-Schmidt orthogonalization.
    ///
    /// # Arguments
    ///
    /// * `matrix` - The matrix to decompose (can be any size, not just square)
    ///
    /// # Returns
    ///
    /// The QR decomposition stored in a compact form. Use [`q()`](Self::q) and [`r()`](Self::r)
    /// to extract the individual factors.
    ///
    /// # Example: Square matrix
    ///
    /// ```
    /// use nalgebra::{Matrix3, linalg::QR};
    ///
    /// let m = Matrix3::new(
    ///     12.0, -51.0,   4.0,
    ///      6.0, 167.0, -68.0,
    ///     -4.0,  24.0, -41.0,
    /// );
    ///
    /// let qr = QR::new(m.clone());
    /// let q = qr.q();
    /// let r = qr.r();
    ///
    /// // Verify: Q * R = A
    /// let reconstructed = q * r;
    /// assert!((reconstructed - m).norm() < 1e-7);
    ///
    /// // Q is orthogonal: Q^T * Q = I
    /// let identity = q.transpose() * &q;
    /// assert!((identity - Matrix3::identity()).norm() < 1e-10);
    /// ```
    ///
    /// # Example: Rectangular matrix (more rows than columns)
    ///
    /// ```
    /// use nalgebra::{Matrix3x2, Matrix2, linalg::QR};
    ///
    /// // A tall matrix (3 rows, 2 columns)
    /// let m = Matrix3x2::new(
    ///     1.0, 2.0,
    ///     3.0, 4.0,
    ///     5.0, 6.0,
    /// );
    ///
    /// let qr = QR::new(m.clone());
    /// let q = qr.q();  // 3x2 orthogonal matrix
    /// let r = qr.r();  // 2x2 upper triangular
    ///
    /// // Q has orthonormal columns
    /// let qtq = q.transpose() * &q;
    /// assert!((qtq - Matrix2::identity()).norm() < 1e-10);
    ///
    /// // Q * R reconstructs the original matrix
    /// assert!((q * r - m).norm() < 1e-10);
    /// ```
    ///
    /// # Example: Application in least squares
    ///
    /// ```
    /// use nalgebra::{Matrix3x2, Vector3, linalg::QR};
    ///
    /// // Overdetermined system: 3 equations, 2 unknowns
    /// // We want to fit a line y = a*x + b to points
    /// let a = Matrix3x2::new(
    ///     1.0, 1.0,  // point at x=1
    ///     2.0, 1.0,  // point at x=2
    ///     3.0, 1.0,  // point at x=3
    /// );
    ///
    /// let b = Vector3::new(1.9, 4.1, 5.8);  // observed y values (with noise)
    ///
    /// // QR decomposition for least squares
    /// let qr = QR::new(a.clone());
    ///
    /// // The least squares solution minimizes ||A*x - b||
    /// // This gives us the best-fit line parameters [a, b]
    /// let q = qr.q();
    /// let r = qr.r();
    ///
    /// // Compute x = R^(-1) * Q^T * b
    /// let qt_b = q.transpose() * b;
    /// // For a full solution, we'd solve R * x = qt_b
    /// // (shown in the solve() method documentation)
    /// ```
    ///
    /// # Example: Orthogonalizing vectors
    ///
    /// ```
    /// use nalgebra::{Matrix3, linalg::QR};
    ///
    /// // Three vectors (as columns) that we want to orthogonalize
    /// let m = Matrix3::new(
    ///     1.0, 1.0, 1.0,
    ///     0.0, 1.0, 1.0,
    ///     0.0, 0.0, 1.0,
    /// );
    ///
    /// let qr = QR::new(m);
    /// let q = qr.q();
    ///
    /// // The columns of Q are orthonormal versions of the original columns
    /// // Verify orthonormality: column_i · column_j = 0 for i ≠ j
    /// let col0 = q.column(0);
    /// let col1 = q.column(1);
    /// let col2 = q.column(2);
    ///
    /// assert!(col0.dot(&col1).abs() < 1e-10);  // perpendicular
    /// assert!(col0.dot(&col2).abs() < 1e-10);  // perpendicular
    /// assert!(col1.dot(&col2).abs() < 1e-10);  // perpendicular
    /// assert!((col0.norm() - 1.0).abs() < 1e-10);  // unit length
    /// assert!((col1.norm() - 1.0).abs() < 1e-10);  // unit length
    /// assert!((col2.norm() - 1.0).abs() < 1e-10);  // unit length
    /// ```
    ///
    /// # Performance Note
    ///
    /// The decomposition is `O(n^2 * m)` for an m×n matrix. For square n×n matrices,
    /// this is `O(n^3)`. The decomposition is done once; subsequent operations like
    /// solving linear systems are much faster.
    ///
    /// # See Also
    ///
    /// * [`q()`](Self::q) - Extract the orthogonal matrix Q
    /// * [`r()`](Self::r) - Extract the upper triangular matrix R
    /// * [`unpack()`](Self::unpack) - Get both Q and R at once
    /// * [`solve()`](Self::solve) - Solve linear systems using the decomposition (for square matrices)
    pub fn new(mut matrix: OMatrix<T, R, C>) -> Self {
        let (nrows, ncols) = matrix.shape_generic();
        let min_nrows_ncols = nrows.min(ncols);

        if min_nrows_ncols.value() == 0 {
            return QR {
                qr: matrix,
                diag: Matrix::zeros_generic(min_nrows_ncols, Const::<1>),
            };
        }

        let mut diag = Matrix::uninit(min_nrows_ncols, Const::<1>);

        for i in 0..min_nrows_ncols.value() {
            diag[i] =
                MaybeUninit::new(householder::clear_column_unchecked(&mut matrix, i, 0, None));
        }

        // Safety: diag is now fully initialized.
        let diag = unsafe { diag.assume_init() };
        QR { qr: matrix, diag }
    }

    /// Retrieves the upper trapezoidal submatrix `R` of this decomposition.
    ///
    /// This method returns a **copy** of the upper triangular (or trapezoidal) matrix `R`
    /// from the QR decomposition `A = Q * R`. All entries below the main diagonal are
    /// guaranteed to be zero.
    ///
    /// For a matrix with more rows than columns (m > n), `R` will be n×n.
    /// For a matrix with more columns than rows (m < n), `R` will be m×n and trapezoidal.
    ///
    /// # Returns
    ///
    /// The upper triangular matrix `R` such that `A = Q * R`.
    ///
    /// # Example: Square matrix
    ///
    /// ```
    /// use nalgebra::{Matrix3, linalg::QR};
    ///
    /// let m = Matrix3::new(
    ///     12.0, -51.0,   4.0,
    ///      6.0, 167.0, -68.0,
    ///     -4.0,  24.0, -41.0,
    /// );
    ///
    /// let qr = QR::new(m.clone());
    /// let r = qr.r();
    ///
    /// // R is upper triangular (all zeros below diagonal)
    /// assert!(r[(1, 0)].abs() < 1e-10);
    /// assert!(r[(2, 0)].abs() < 1e-10);
    /// assert!(r[(2, 1)].abs() < 1e-10);
    ///
    /// // Verify Q * R = A
    /// let q = qr.q();
    /// assert!((q * r - m).norm() < 1e-7);
    /// ```
    ///
    /// # Example: Rectangular matrix (tall)
    ///
    /// ```
    /// use nalgebra::{Matrix3x2, linalg::QR};
    ///
    /// let m = Matrix3x2::new(
    ///     1.0, 2.0,
    ///     3.0, 4.0,
    ///     5.0, 6.0,
    /// );
    ///
    /// let qr = QR::new(m.clone());
    /// let r = qr.r();  // 2x2 upper triangular
    ///
    /// // Check dimensions
    /// assert_eq!(r.nrows(), 2);
    /// assert_eq!(r.ncols(), 2);
    ///
    /// // R is upper triangular
    /// assert!(r[(1, 0)].abs() < 1e-10);
    ///
    /// // Reconstruct original matrix
    /// let q = qr.q();
    /// assert!((q * r - m).norm() < 1e-10);
    /// ```
    ///
    /// # Example: Using R to solve triangular systems
    ///
    /// ```
    /// use nalgebra::{Matrix3, Vector3, linalg::QR};
    ///
    /// let m = Matrix3::new(
    ///     2.0, 1.0, 1.0,
    ///     4.0, 1.0, 0.0,
    ///     2.0, 2.0, 1.0,
    /// );
    ///
    /// let qr = QR::new(m);
    /// let r = qr.r();
    ///
    /// // Upper triangular matrices are easy to solve
    /// // For R * x = b, we can use back-substitution
    /// // (This is done automatically by the solve() method)
    ///
    /// // Display the upper triangular structure
    /// println!("R matrix:\n{:.2}", r);
    /// ```
    ///
    /// # Performance Note
    ///
    /// This method allocates and returns a new matrix. If you're done with the QR
    /// decomposition and want to avoid the copy, use [`unpack_r()`](Self::unpack_r) instead,
    /// which consumes the decomposition.
    ///
    /// # See Also
    ///
    /// * [`q()`](Self::q) - Get the orthogonal matrix Q
    /// * [`unpack_r()`](Self::unpack_r) - Get R by consuming the decomposition (more efficient)
    /// * [`unpack()`](Self::unpack) - Get both Q and R at once
    #[inline]
    #[must_use]
    pub fn r(&self) -> OMatrix<T, DimMinimum<R, C>, C>
    where
        DefaultAllocator: Allocator<DimMinimum<R, C>, C>,
    {
        let (nrows, ncols) = self.qr.shape_generic();
        let mut res = self.qr.rows_generic(0, nrows.min(ncols)).upper_triangle();
        res.set_partial_diagonal(self.diag.iter().map(|e| T::from_real(e.clone().modulus())));
        res
    }

    /// Retrieves the upper trapezoidal submatrix `R` of this decomposition.
    ///
    /// This method consumes the decomposition and returns the upper triangular matrix `R`.
    /// It's **more efficient** than [`r()`](Self::r) because it reuses the internal storage
    /// instead of allocating a new matrix, but you can no longer use the decomposition afterward.
    ///
    /// Use this when you only need `R` and don't need to keep the QR decomposition around.
    ///
    /// # Returns
    ///
    /// The upper triangular matrix `R` such that `A = Q * R`. All entries below the
    /// diagonal are zero.
    ///
    /// # Example: Basic usage
    ///
    /// ```
    /// use nalgebra::{Matrix3, linalg::QR};
    ///
    /// let m = Matrix3::new(
    ///     12.0, -51.0,   4.0,
    ///      6.0, 167.0, -68.0,
    ///     -4.0,  24.0, -41.0,
    /// );
    ///
    /// let qr = QR::new(m);
    ///
    /// // Get R efficiently (decomposition is consumed)
    /// let r = qr.unpack_r();
    ///
    /// // R is upper triangular
    /// assert!(r[(1, 0)].abs() < 1e-10);
    /// assert!(r[(2, 0)].abs() < 1e-10);
    /// assert!(r[(2, 1)].abs() < 1e-10);
    ///
    /// // qr is no longer available here
    /// ```
    ///
    /// # Example: When you only need R
    ///
    /// ```
    /// use nalgebra::{Matrix3x2, linalg::QR};
    ///
    /// let m = Matrix3x2::new(
    ///     1.0, 2.0,
    ///     3.0, 4.0,
    ///     5.0, 6.0,
    /// );
    ///
    /// let qr = QR::new(m);
    ///
    /// // If we only need R (e.g., for analyzing matrix rank or condition)
    /// let r = qr.unpack_r();
    ///
    /// // Check if matrix has full rank by examining diagonal of R
    /// let is_full_rank = r[(0, 0)].abs() > 1e-10 && r[(1, 1)].abs() > 1e-10;
    /// assert!(is_full_rank);
    /// ```
    ///
    /// # Example: Comparing with r()
    ///
    /// ```
    /// use nalgebra::{Matrix3, linalg::QR};
    ///
    /// let m = Matrix3::new(
    ///     1.0, 2.0, 3.0,
    ///     0.0, 4.0, 5.0,
    ///     0.0, 0.0, 6.0,
    /// );
    ///
    /// // Using r() - keeps the decomposition
    /// let qr1 = QR::new(m.clone());
    /// let r1 = qr1.r();
    /// // Can still use qr1 here
    /// let q1 = qr1.q();
    ///
    /// // Using unpack_r() - consumes the decomposition (more efficient)
    /// let qr2 = QR::new(m);
    /// let r2 = qr2.unpack_r();
    /// // Cannot use qr2 here anymore
    ///
    /// // Both produce the same result
    /// assert!((r1 - r2).norm() < 1e-10);
    /// ```
    ///
    /// # Performance Note
    ///
    /// This method is faster than [`r()`](Self::r) because it reuses the internal matrix
    /// storage. Use this when you're done with the QR decomposition and only need `R`.
    ///
    /// # See Also
    ///
    /// * [`r()`](Self::r) - Non-consuming version that keeps the decomposition
    /// * [`unpack()`](Self::unpack) - Get both Q and R at once
    /// * [`q()`](Self::q) - Get the orthogonal matrix Q
    #[inline]
    pub fn unpack_r(self) -> OMatrix<T, DimMinimum<R, C>, C>
    where
        DefaultAllocator: Reallocator<T, R, C, DimMinimum<R, C>, C>,
    {
        let (nrows, ncols) = self.qr.shape_generic();
        let mut res = self.qr.resize_generic(nrows.min(ncols), ncols, T::zero());
        res.fill_lower_triangle(T::zero(), 1);
        res.set_partial_diagonal(self.diag.iter().map(|e| T::from_real(e.clone().modulus())));
        res
    }

    /// Computes the orthogonal matrix `Q` of this decomposition.
    ///
    /// This method returns the orthogonal (or orthonormal) matrix `Q` from the QR decomposition
    /// `A = Q * R`. The columns of `Q` form an orthonormal basis - they are unit vectors that
    /// are mutually perpendicular.
    ///
    /// For a matrix with more rows than columns (m > n), `Q` will be m×n with n orthonormal columns.
    ///
    /// # What Makes Q Special?
    ///
    /// The matrix `Q` has these important properties:
    /// - **Orthonormal columns**: Each column has length 1 and is perpendicular to all others
    /// - **Preserves lengths**: `||Q*x|| = ||x||` for any vector `x`
    /// - **Preserves angles**: `Q` represents a rotation and/or reflection
    /// - **Easy to invert**: `Q^T * Q = I` (for full square Q) or `Q^T` is a left inverse
    ///
    /// # Returns
    ///
    /// The orthogonal matrix `Q` such that `A = Q * R`.
    ///
    /// # Example: Verify orthogonality
    ///
    /// ```
    /// use nalgebra::{Matrix3, linalg::QR};
    ///
    /// let m = Matrix3::new(
    ///     12.0, -51.0,   4.0,
    ///      6.0, 167.0, -68.0,
    ///     -4.0,  24.0, -41.0,
    /// );
    ///
    /// let qr = QR::new(m);
    /// let q = qr.q();
    ///
    /// // Q is orthogonal: Q^T * Q = I
    /// let identity = q.transpose() * &q;
    /// assert!((identity - Matrix3::identity()).norm() < 1e-10);
    ///
    /// // Each column has unit length
    /// assert!((q.column(0).norm() - 1.0).abs() < 1e-10);
    /// assert!((q.column(1).norm() - 1.0).abs() < 1e-10);
    /// assert!((q.column(2).norm() - 1.0).abs() < 1e-10);
    ///
    /// // Columns are perpendicular to each other
    /// assert!(q.column(0).dot(&q.column(1)).abs() < 1e-10);
    /// assert!(q.column(0).dot(&q.column(2)).abs() < 1e-10);
    /// assert!(q.column(1).dot(&q.column(2)).abs() < 1e-10);
    /// ```
    ///
    /// # Example: Using Q for orthogonalization
    ///
    /// ```
    /// use nalgebra::{Matrix3, linalg::QR};
    ///
    /// // Start with arbitrary vectors (as columns)
    /// let m = Matrix3::new(
    ///     1.0, 1.0, 1.0,
    ///     2.0, 1.0, 0.0,
    ///     3.0, 1.0, 1.0,
    /// );
    ///
    /// let qr = QR::new(m);
    /// let q = qr.q();
    ///
    /// // Q contains orthonormalized versions of the original columns
    /// // The columns of Q span the same space as the columns of m
    /// // but are now orthonormal
    ///
    /// println!("Orthonormal basis:\n{:.4}", q);
    /// ```
    ///
    /// # Example: Rectangular matrix
    ///
    /// ```
    /// use nalgebra::{Matrix3x2, Matrix2, linalg::QR};
    ///
    /// let m = Matrix3x2::new(
    ///     1.0, 0.0,
    ///     0.0, 1.0,
    ///     1.0, 1.0,
    /// );
    ///
    /// let qr = QR::new(m);
    /// let q = qr.q();  // 3x2 matrix
    ///
    /// // Q has orthonormal columns (but Q is not square)
    /// let qtq = q.transpose() * &q;
    /// assert!((qtq - Matrix2::identity()).norm() < 1e-10);
    ///
    /// // Q * Q^T is NOT identity (Q is not square)
    /// let qqt = &q * q.transpose();
    /// assert!((qqt - Matrix3::identity()).norm() > 0.1);
    /// ```
    ///
    /// # Example: Application in projection
    ///
    /// ```
    /// use nalgebra::{Matrix3, Vector3, linalg::QR};
    ///
    /// let m = Matrix3::new(
    ///     1.0, 0.0, 0.0,
    ///     0.0, 1.0, 0.0,
    ///     1.0, 1.0, 0.0,
    /// );
    ///
    /// let qr = QR::new(m);
    /// let q = qr.q();
    ///
    /// // Project a vector onto the column space of the original matrix
    /// let v = Vector3::new(1.0, 2.0, 3.0);
    /// let projection = &q * (q.transpose() * v);
    ///
    /// println!("Original vector: {}", v);
    /// println!("Projected vector: {}", projection);
    /// ```
    ///
    /// # Performance Note
    ///
    /// Computing `Q` explicitly requires `O(n^2 * m)` operations for an m×n matrix.
    /// If you only need to multiply by `Q` or `Q^T`, consider using [`q_tr_mul()`](Self::q_tr_mul)
    /// which is more efficient.
    ///
    /// # See Also
    ///
    /// * [`r()`](Self::r) - Get the upper triangular matrix R
    /// * [`unpack()`](Self::unpack) - Get both Q and R at once
    /// * [`q_tr_mul()`](Self::q_tr_mul) - Multiply by Q^T without forming Q explicitly
    #[must_use]
    pub fn q(&self) -> OMatrix<T, R, DimMinimum<R, C>>
    where
        DefaultAllocator: Allocator<R, DimMinimum<R, C>>,
    {
        let (nrows, ncols) = self.qr.shape_generic();

        // NOTE: we could build the identity matrix and call q_mul on it.
        // Instead we don't so that we take in account the matrix sparseness.
        let mut res = Matrix::identity_generic(nrows, nrows.min(ncols));
        let dim = self.diag.len();

        for i in (0..dim).rev() {
            let axis = self.qr.view_range(i.., i);
            // TODO: sometimes, the axis might have a zero magnitude.
            let refl = Reflection::new(Unit::new_unchecked(axis), T::zero());

            let mut res_rows = res.view_range_mut(i.., i..);
            refl.reflect_with_sign(&mut res_rows, self.diag[i].clone().signum());
        }

        res
    }

    /// Unpacks this decomposition into its two matrix factors.
    ///
    /// This method consumes the QR decomposition and returns both the orthogonal matrix `Q`
    /// and the upper triangular matrix `R` as a tuple `(Q, R)`.
    ///
    /// This is a convenience method that's equivalent to calling `q()` and `unpack_r()`,
    /// but it's slightly more efficient when you need both matrices.
    ///
    /// # Returns
    ///
    /// A tuple `(Q, R)` where:
    /// - `Q` is the orthogonal matrix (with orthonormal columns)
    /// - `R` is the upper triangular matrix
    ///
    /// Together, they satisfy `A = Q * R` where `A` is the original matrix.
    ///
    /// # Example: Basic unpacking
    ///
    /// ```
    /// use nalgebra::{Matrix3, linalg::QR};
    ///
    /// let m = Matrix3::new(
    ///     12.0, -51.0,   4.0,
    ///      6.0, 167.0, -68.0,
    ///     -4.0,  24.0, -41.0,
    /// );
    ///
    /// let qr = QR::new(m.clone());
    ///
    /// // Get both Q and R at once
    /// let (q, r) = qr.unpack();
    ///
    /// // Verify: Q * R = A
    /// let reconstructed = q * r;
    /// assert!((reconstructed - m).norm() < 1e-7);
    /// ```
    ///
    /// # Example: Verifying QR properties
    ///
    /// ```
    /// use nalgebra::{Matrix3, linalg::QR};
    ///
    /// let m = Matrix3::new(
    ///     1.0, 2.0, 3.0,
    ///     4.0, 5.0, 6.0,
    ///     7.0, 8.0, 10.0,
    /// );
    ///
    /// let qr = QR::new(m.clone());
    /// let (q, r) = qr.unpack();
    ///
    /// // Property 1: Q is orthogonal (Q^T * Q = I)
    /// let identity = q.transpose() * &q;
    /// assert!((identity - Matrix3::identity()).norm() < 1e-10);
    ///
    /// // Property 2: R is upper triangular
    /// assert!(r[(1, 0)].abs() < 1e-10);
    /// assert!(r[(2, 0)].abs() < 1e-10);
    /// assert!(r[(2, 1)].abs() < 1e-10);
    ///
    /// // Property 3: Q * R reconstructs the original
    /// assert!((q * r - m).norm() < 1e-10);
    /// ```
    ///
    /// # Example: Using unpacked factors
    ///
    /// ```
    /// use nalgebra::{Matrix3, Vector3, linalg::QR};
    ///
    /// let a = Matrix3::new(
    ///     2.0, 1.0, 1.0,
    ///     4.0, 1.0, 0.0,
    ///     2.0, 2.0, 1.0,
    /// );
    ///
    /// let b = Vector3::new(5.0, 5.0, 6.0);
    ///
    /// // Solve A*x = b using QR decomposition
    /// let qr = QR::new(a);
    /// let (q, r) = qr.unpack();
    ///
    /// // Step 1: Compute Q^T * b
    /// let qt_b = q.transpose() * b;
    ///
    /// // Step 2: Solve R * x = Q^T * b (R is upper triangular)
    /// // (In practice, use the solve() method which does this for you)
    /// ```
    ///
    /// # Example: Rectangular matrix
    ///
    /// ```
    /// use nalgebra::{Matrix3x2, Matrix2, linalg::QR};
    ///
    /// let m = Matrix3x2::new(
    ///     1.0, 2.0,
    ///     3.0, 4.0,
    ///     5.0, 6.0,
    /// );
    ///
    /// let qr = QR::new(m.clone());
    /// let (q, r) = qr.unpack();
    ///
    /// // Q is 3x2 (tall), R is 2x2 (square)
    /// assert_eq!(q.nrows(), 3);
    /// assert_eq!(q.ncols(), 2);
    /// assert_eq!(r.nrows(), 2);
    /// assert_eq!(r.ncols(), 2);
    ///
    /// // Q has orthonormal columns
    /// let qtq = q.transpose() * &q;
    /// assert!((qtq - Matrix2::identity()).norm() < 1e-10);
    ///
    /// // Reconstruction works
    /// assert!((q * r - m).norm() < 1e-10);
    /// ```
    ///
    /// # Performance Note
    ///
    /// This method is more efficient than calling `q()` and `r()` separately because
    /// it reuses the internal storage for `R`.
    ///
    /// # See Also
    ///
    /// * [`q()`](Self::q) - Get only Q (keeps the decomposition)
    /// * [`r()`](Self::r) - Get only R (keeps the decomposition)
    /// * [`unpack_r()`](Self::unpack_r) - Get only R (consumes the decomposition)
    pub fn unpack(
        self,
    ) -> (
        OMatrix<T, R, DimMinimum<R, C>>,
        OMatrix<T, DimMinimum<R, C>, C>,
    )
    where
        DimMinimum<R, C>: DimMin<C, Output = DimMinimum<R, C>>,
        DefaultAllocator:
            Allocator<R, DimMinimum<R, C>> + Reallocator<T, R, C, DimMinimum<R, C>, C>,
    {
        (self.q(), self.unpack_r())
    }

    #[doc(hidden)]
    pub const fn qr_internal(&self) -> &OMatrix<T, R, C> {
        &self.qr
    }

    #[must_use]
    pub(crate) const fn diag_internal(&self) -> &OVector<T, DimMinimum<R, C>> {
        &self.diag
    }

    /// Multiplies the provided matrix by the transpose of the `Q` matrix of this decomposition.
    ///
    /// This method efficiently computes `Q^T * rhs` **in-place**, overwriting the input
    /// matrix `rhs` with the result. It's significantly more efficient than computing `Q`
    /// explicitly and then multiplying, especially for large matrices.
    ///
    /// # How It Works
    ///
    /// Instead of forming the full `Q` matrix (which requires `O(n^2 * m)` operations and
    /// storage), this method applies the Householder reflections directly to `rhs`.
    /// This is both faster and uses less memory.
    ///
    /// # Arguments
    ///
    /// * `rhs` - The matrix to multiply by `Q^T`. It will be modified in-place to contain
    ///           the result `Q^T * rhs`.
    ///
    /// # Example: Basic usage
    ///
    /// ```
    /// use nalgebra::{Matrix3, Vector3, linalg::QR};
    ///
    /// let m = Matrix3::new(
    ///     12.0, -51.0,   4.0,
    ///      6.0, 167.0, -68.0,
    ///     -4.0,  24.0, -41.0,
    /// );
    ///
    /// let qr = QR::new(m);
    ///
    /// let mut v = Vector3::new(1.0, 2.0, 3.0);
    ///
    /// // Compute Q^T * v in-place
    /// qr.q_tr_mul(&mut v);
    ///
    /// // Verify by comparing with explicit Q
    /// let m2 = Matrix3::new(
    ///     12.0, -51.0,   4.0,
    ///      6.0, 167.0, -68.0,
    ///     -4.0,  24.0, -41.0,
    /// );
    /// let qr2 = QR::new(m2);
    /// let q = qr2.q();
    /// let v2 = Vector3::new(1.0, 2.0, 3.0);
    /// let expected = q.transpose() * v2;
    ///
    /// assert!((v - expected).norm() < 1e-10);
    /// ```
    ///
    /// # Example: Application in solving linear systems
    ///
    /// ```
    /// use nalgebra::{Matrix3, Vector3, linalg::QR};
    ///
    /// let a = Matrix3::new(
    ///     2.0, 1.0, 1.0,
    ///     4.0, 1.0, 0.0,
    ///     2.0, 2.0, 1.0,
    /// );
    ///
    /// let mut b = Vector3::new(5.0, 5.0, 6.0);
    ///
    /// let qr = QR::new(a);
    ///
    /// // First step in solving A*x = b is computing Q^T * b
    /// qr.q_tr_mul(&mut b);
    ///
    /// // Now b contains Q^T * original_b
    /// // Next step would be solving R * x = b (upper triangular system)
    /// // (This is what the solve() method does for you)
    /// ```
    ///
    /// # Example: Multiple columns at once
    ///
    /// ```
    /// use nalgebra::{Matrix3, linalg::QR};
    ///
    /// let m = Matrix3::new(
    ///     1.0, 2.0, 3.0,
    ///     4.0, 5.0, 6.0,
    ///     7.0, 8.0, 10.0,
    /// );
    ///
    /// let qr = QR::new(m);
    ///
    /// // Multiply Q^T with multiple vectors at once (as columns)
    /// let mut matrix = Matrix3::new(
    ///     1.0, 0.0, 0.0,
    ///     0.0, 1.0, 0.0,
    ///     0.0, 0.0, 1.0,
    /// );
    ///
    /// qr.q_tr_mul(&mut matrix);
    ///
    /// // matrix now contains Q^T (the columns of Q^T)
    /// // Verify this matches Q.transpose()
    /// let m2 = Matrix3::new(
    ///     1.0, 2.0, 3.0,
    ///     4.0, 5.0, 6.0,
    ///     7.0, 8.0, 10.0,
    /// );
    /// let qr2 = QR::new(m2);
    /// let qt = qr2.q().transpose();
    ///
    /// assert!((matrix - qt).norm() < 1e-10);
    /// ```
    ///
    /// # Example: Computing projections efficiently
    ///
    /// ```
    /// use nalgebra::{Matrix3, Vector3, linalg::QR};
    ///
    /// let m = Matrix3::new(
    ///     1.0, 0.0, 0.0,
    ///     0.0, 1.0, 0.0,
    ///     1.0, 1.0, 0.0,
    /// );
    ///
    /// let qr = QR::new(m);
    ///
    /// let v = Vector3::new(1.0, 2.0, 3.0);
    ///
    /// // Project v onto the column space: projection = Q * (Q^T * v)
    /// let mut qt_v = v.clone();
    /// qr.q_tr_mul(&mut qt_v);  // Efficient: Q^T * v
    ///
    /// let projection = qr.q() * qt_v;  // Then Q * result
    ///
    /// println!("Projection: {}", projection);
    /// ```
    ///
    /// # Performance Note
    ///
    /// This method is **much more efficient** than computing `Q` explicitly:
    /// - **Memory**: No need to allocate the full `Q` matrix
    /// - **Speed**: Applies reflections directly, avoiding matrix construction
    /// - **Complexity**: `O(n*m*k)` vs `O(n^2*m + n*m*k)` for explicit Q
    ///
    /// where the input matrix is n×m and rhs is n×k.
    ///
    /// # See Also
    ///
    /// * [`q()`](Self::q) - Get the explicit Q matrix (if you really need it)
    /// * [`solve_mut()`](Self::solve_mut) - Solve linear systems (uses this internally)
    pub fn q_tr_mul<R2: Dim, C2: Dim, S2>(&self, rhs: &mut Matrix<T, R2, C2, S2>)
    // TODO: do we need a static constraint on the number of rows of rhs?
    where
        S2: StorageMut<T, R2, C2>,
    {
        let dim = self.diag.len();

        for i in 0..dim {
            let axis = self.qr.view_range(i.., i);
            let refl = Reflection::new(Unit::new_unchecked(axis), T::zero());

            let mut rhs_rows = rhs.rows_range_mut(i..);
            refl.reflect_with_sign(&mut rhs_rows, self.diag[i].clone().signum().conjugate());
        }
    }
}

impl<T: ComplexField, D: DimMin<D, Output = D>> QR<T, D, D>
where
    DefaultAllocator: Allocator<D, D> + Allocator<D>,
{
    /// Solves the linear system `self * x = b`, where `x` is the unknown to be determined.
    ///
    /// This method solves the system `A * x = b` for the unknown vector(s) `x`, where `A`
    /// is the matrix that was decomposed. It returns a new matrix/vector containing the solution.
    ///
    /// **Important**: This method only works for **square matrices** (same number of rows and columns).
    /// For overdetermined systems (more rows than columns), use least squares methods instead.
    ///
    /// # How It Works
    ///
    /// Given the QR decomposition `A = Q * R`, solving `A * x = b` becomes:
    ///
    /// 1. Multiply both sides by `Q^T`: `R * x = Q^T * b`
    /// 2. Solve the upper triangular system `R * x = Q^T * b` by back-substitution
    ///
    /// This is efficient because:
    /// - Multiplying by `Q^T` is fast (orthogonal matrix)
    /// - Solving triangular systems is `O(n^2)`, much faster than general solving
    ///
    /// # Arguments
    ///
    /// * `b` - The right-hand side of the equation. Can be a vector or matrix of multiple
    ///         right-hand sides (one per column).
    ///
    /// # Returns
    ///
    /// * `Some(x)` - The solution if the matrix is invertible
    /// * `None` - If the matrix is singular (not invertible)
    ///
    /// # Example: Solving a single linear system
    ///
    /// ```
    /// use nalgebra::{Matrix3, Vector3, linalg::QR};
    ///
    /// let a = Matrix3::new(
    ///     2.0, 1.0, 1.0,
    ///     4.0, 1.0, 0.0,
    ///     2.0, 2.0, 1.0,
    /// );
    ///
    /// let b = Vector3::new(5.0, 5.0, 6.0);
    ///
    /// let qr = QR::new(a.clone());
    /// let x = qr.solve(&b).expect("Matrix is invertible");
    ///
    /// // Verify: A * x = b
    /// let result = a * x;
    /// assert!((result - b).norm() < 1e-10);
    /// ```
    ///
    /// # Example: Solving multiple systems at once
    ///
    /// ```
    /// use nalgebra::{Matrix3, linalg::QR};
    ///
    /// let a = Matrix3::new(
    ///     2.0, 1.0, 0.0,
    ///     1.0, 3.0, 1.0,
    ///     0.0, 1.0, 2.0,
    /// );
    ///
    /// // Solve A * X = B where B has multiple columns
    /// let b = Matrix3::new(
    ///     1.0, 0.0, 0.0,
    ///     0.0, 1.0, 0.0,
    ///     0.0, 0.0, 1.0,
    /// );
    ///
    /// let qr = QR::new(a.clone());
    /// let x = qr.solve(&b).expect("Matrix is invertible");
    ///
    /// // x is actually the inverse of a (since we solved A * X = I)
    /// let identity = a * x;
    /// assert!((identity - Matrix3::identity()).norm() < 1e-10);
    /// ```
    ///
    /// # Example: Detecting singular matrices
    ///
    /// ```
    /// use nalgebra::{Matrix3, Vector3, linalg::QR};
    ///
    /// // A singular matrix (zero diagonal element)
    /// let a = Matrix3::new(
    ///     1.0, 2.0, 3.0,
    ///     0.0, 0.0, 0.0,  // Zero row makes it singular
    ///     3.0, 2.0, 1.0,
    /// );
    ///
    /// let b = Vector3::new(1.0, 2.0, 3.0);
    ///
    /// let qr = QR::new(a);
    /// let result = qr.solve(&b);
    ///
    /// // May return None for clearly singular matrices
    /// // For subtly singular matrices, may still return a solution
    /// println!("Solution found: {}", result.is_some());
    /// ```
    ///
    /// # Example: Application in engineering - circuit analysis
    ///
    /// ```
    /// use nalgebra::{Matrix3, Vector3, linalg::QR};
    ///
    /// // Kirchhoff's laws for a circuit with 3 unknown currents
    /// // Each row represents a loop or node equation
    /// let circuit_matrix = Matrix3::new(
    ///     10.0,  5.0,  0.0,  // Loop 1
    ///      5.0, 15.0, 10.0,  // Loop 2
    ///      0.0, 10.0, 20.0,  // Loop 3
    /// );
    ///
    /// let voltages = Vector3::new(12.0, 15.0, 18.0);  // Applied voltages
    ///
    /// let qr = QR::new(circuit_matrix);
    /// let currents = qr.solve(&voltages).expect("System is solvable");
    ///
    /// println!("Currents in the circuit: {:.3}", currents);
    /// ```
    ///
    /// # Performance Note
    ///
    /// This method is `O(n^2)` after the decomposition is computed. If you need to solve
    /// multiple systems with the same matrix, compute the QR decomposition once and reuse it.
    ///
    /// For in-place solving (which avoids allocating a new matrix), use [`solve_mut()`](Self::solve_mut).
    ///
    /// # See Also
    ///
    /// * [`solve_mut()`](Self::solve_mut) - In-place version (more efficient, modifies input)
    /// * [`try_inverse()`](Self::try_inverse) - Compute the matrix inverse
    /// * [`is_invertible()`](Self::is_invertible) - Check if the matrix is invertible
    #[must_use = "Did you mean to use solve_mut()?"]
    pub fn solve<R2: Dim, C2: Dim, S2>(
        &self,
        b: &Matrix<T, R2, C2, S2>,
    ) -> Option<OMatrix<T, R2, C2>>
    where
        S2: Storage<T, R2, C2>,
        ShapeConstraint: SameNumberOfRows<R2, D>,
        DefaultAllocator: Allocator<R2, C2>,
    {
        let mut res = b.clone_owned();

        if self.solve_mut(&mut res) {
            Some(res)
        } else {
            None
        }
    }

    /// Solves the linear system `self * x = b`, where `x` is the unknown to be determined.
    ///
    /// This method solves the system `A * x = b` **in-place**, storing the solution directly
    /// in `b`. It's more efficient than [`solve()`](Self::solve) because it doesn't allocate
    /// a new matrix, but it overwrites the input.
    ///
    /// **Important**: This method only works for **square matrices** (same number of rows and columns).
    ///
    /// # How It Works
    ///
    /// Given the QR decomposition `A = Q * R`, solving `A * x = b` is done in two steps:
    ///
    /// 1. Compute `Q^T * b` (stored back in `b`)
    /// 2. Solve the upper triangular system `R * x = b` by back-substitution
    ///
    /// # Arguments
    ///
    /// * `b` - On input: the right-hand side of the equation. On output: the solution `x`
    ///         (if successful) or garbage (if the matrix is singular).
    ///
    /// # Returns
    ///
    /// * `true` - If the system was solved successfully, `b` now contains the solution
    /// * `false` - If the matrix is singular (not invertible), `b` contains garbage
    ///
    /// # Warning
    ///
    /// If the decomposed matrix is not invertible, this returns `false` and its input `b` is
    /// overwritten with garbage. Always check the return value!
    ///
    /// # Example: Basic in-place solving
    ///
    /// ```
    /// use nalgebra::{Matrix3, Vector3, linalg::QR};
    ///
    /// let a = Matrix3::new(
    ///     2.0, 1.0, 1.0,
    ///     4.0, 1.0, 0.0,
    ///     2.0, 2.0, 1.0,
    /// );
    ///
    /// let mut b = Vector3::new(5.0, 5.0, 6.0);
    /// let b_original = b.clone();
    ///
    /// let qr = QR::new(a.clone());
    /// let success = qr.solve_mut(&mut b);
    ///
    /// assert!(success, "Matrix should be invertible");
    ///
    /// // b now contains the solution
    /// let result = a * b;
    /// assert!((result - b_original).norm() < 1e-10);
    /// ```
    ///
    /// # Example: Efficient solving of multiple systems
    ///
    /// ```
    /// use nalgebra::{Matrix3, Vector3, linalg::QR};
    ///
    /// let a = Matrix3::new(
    ///     2.0, 1.0, 0.0,
    ///     1.0, 3.0, 1.0,
    ///     0.0, 1.0, 2.0,
    /// );
    ///
    /// let qr = QR::new(a.clone());
    ///
    /// // Solve multiple systems efficiently
    /// let systems = vec![
    ///     Vector3::new(1.0, 0.0, 0.0),
    ///     Vector3::new(0.0, 1.0, 0.0),
    ///     Vector3::new(0.0, 0.0, 1.0),
    /// ];
    ///
    /// for mut b in systems {
    ///     let original_b = b.clone();
    ///     if qr.solve_mut(&mut b) {
    ///         // Verify solution
    ///         let result = &a * &b;
    ///         assert!((result - original_b).norm() < 1e-10);
    ///     }
    /// }
    /// ```
    ///
    /// # Example: Handling singular matrices
    ///
    /// ```
    /// use nalgebra::{Matrix3, Vector3, linalg::QR};
    ///
    /// // A singular matrix (zero row)
    /// let a = Matrix3::new(
    ///     1.0, 2.0, 3.0,
    ///     0.0, 0.0, 0.0,  // Zero row makes it singular
    ///     4.0, 5.0, 6.0,
    /// );
    ///
    /// let mut b = Vector3::new(1.0, 2.0, 3.0);
    ///
    /// let qr = QR::new(a);
    /// let success = qr.solve_mut(&mut b);
    ///
    /// // Check for failure
    /// if !success {
    ///     println!("Matrix is singular, cannot solve!");
    ///     // b now contains garbage, don't use it
    /// } else {
    ///     println!("Solution found (matrix may be numerically singular)");
    /// }
    /// ```
    ///
    /// # Example: Computing columns of an inverse
    ///
    /// ```
    /// use nalgebra::{Matrix3, Vector3, linalg::QR};
    ///
    /// let a = Matrix3::new(
    ///     2.0, 1.0, 0.0,
    ///     1.0, 3.0, 1.0,
    ///     0.0, 1.0, 2.0,
    /// );
    ///
    /// let qr = QR::new(a.clone());
    ///
    /// // Compute inverse column by column
    /// let mut col1 = Vector3::new(1.0, 0.0, 0.0);
    /// let mut col2 = Vector3::new(0.0, 1.0, 0.0);
    /// let mut col3 = Vector3::new(0.0, 0.0, 1.0);
    ///
    /// qr.solve_mut(&mut col1);
    /// qr.solve_mut(&mut col2);
    /// qr.solve_mut(&mut col3);
    ///
    /// // Build the inverse from the columns
    /// let inverse = Matrix3::from_columns(&[col1, col2, col3]);
    ///
    /// // Verify: A * A^(-1) = I
    /// let identity = a * inverse;
    /// assert!((identity - Matrix3::identity()).norm() < 1e-10);
    /// ```
    ///
    /// # Performance Note
    ///
    /// This method is more efficient than [`solve()`](Self::solve) because:
    /// - No allocation of a new result matrix
    /// - Solution is computed directly in the provided storage
    /// - Ideal for performance-critical code
    ///
    /// Use this when you don't need to keep the original right-hand side `b`.
    ///
    /// # See Also
    ///
    /// * [`solve()`](Self::solve) - Allocating version that doesn't modify input
    /// * [`try_inverse()`](Self::try_inverse) - Compute the full matrix inverse
    /// * [`is_invertible()`](Self::is_invertible) - Check invertibility before solving
    pub fn solve_mut<R2: Dim, C2: Dim, S2>(&self, b: &mut Matrix<T, R2, C2, S2>) -> bool
    where
        S2: StorageMut<T, R2, C2>,
        ShapeConstraint: SameNumberOfRows<R2, D>,
    {
        assert_eq!(
            self.qr.nrows(),
            b.nrows(),
            "QR solve matrix dimension mismatch."
        );
        assert!(
            self.qr.is_square(),
            "QR solve: unable to solve a non-square system."
        );

        self.q_tr_mul(b);
        self.solve_upper_triangular_mut(b)
    }

    // TODO: duplicate code from the `solve` module.
    fn solve_upper_triangular_mut<R2: Dim, C2: Dim, S2>(
        &self,
        b: &mut Matrix<T, R2, C2, S2>,
    ) -> bool
    where
        S2: StorageMut<T, R2, C2>,
        ShapeConstraint: SameNumberOfRows<R2, D>,
    {
        let dim = self.qr.nrows();

        for k in 0..b.ncols() {
            let mut b = b.column_mut(k);
            for i in (0..dim).rev() {
                let coeff;

                unsafe {
                    let diag = self.diag.vget_unchecked(i).clone().modulus();

                    if diag.is_zero() {
                        return false;
                    }

                    coeff = b.vget_unchecked(i).clone().unscale(diag);
                    *b.vget_unchecked_mut(i) = coeff.clone();
                }

                b.rows_range_mut(..i)
                    .axpy(-coeff, &self.qr.view_range(..i, i), T::one());
            }
        }

        true
    }

    /// Computes the inverse of the decomposed matrix.
    ///
    /// This method computes the inverse `A^(-1)` of the matrix `A` that was decomposed.
    /// For an invertible matrix, `A * A^(-1) = I` where `I` is the identity matrix.
    ///
    /// **Note**: Computing the inverse is `O(n^3)`. If you only need to solve `A*x = b`,
    /// use [`solve()`](Self::solve) instead, which is faster and more numerically stable.
    ///
    /// # How It Works
    ///
    /// The inverse is computed by solving the system `A * X = I` where `I` is the
    /// identity matrix. This is equivalent to solving `n` separate systems (one for
    /// each column of the identity matrix).
    ///
    /// # Returns
    ///
    /// * `Some(A^(-1))` - The inverse matrix if `A` is invertible
    /// * `None` - If the matrix is singular (not invertible)
    ///
    /// # Example: Basic matrix inversion
    ///
    /// ```
    /// use nalgebra::{Matrix3, linalg::QR};
    ///
    /// let a = Matrix3::new(
    ///     2.0, 1.0, 0.0,
    ///     1.0, 3.0, 1.0,
    ///     0.0, 1.0, 2.0,
    /// );
    ///
    /// let qr = QR::new(a.clone());
    /// let a_inv = qr.try_inverse().expect("Matrix is invertible");
    ///
    /// // Verify: A * A^(-1) = I
    /// let identity = a * a_inv;
    /// assert!((identity - Matrix3::identity()).norm() < 1e-10);
    ///
    /// // Also: A^(-1) * A = I
    /// let identity2 = a_inv * a;
    /// assert!((identity2 - Matrix3::identity()).norm() < 1e-10);
    /// ```
    ///
    /// # Example: Detecting non-invertible matrices
    ///
    /// ```
    /// use nalgebra::{Matrix3, linalg::QR};
    ///
    /// // A singular matrix (zero row)
    /// let a = Matrix3::new(
    ///     1.0, 2.0, 3.0,
    ///     0.0, 0.0, 0.0,  // Zero row makes it singular
    ///     3.0, 5.0, 7.0,
    /// );
    ///
    /// let qr = QR::new(a);
    /// let inverse = qr.try_inverse();
    ///
    /// // Singular matrices have no inverse (or return None)
    /// println!("Has inverse: {}", inverse.is_some());
    /// ```
    ///
    /// # Example: Using inverse vs solve
    ///
    /// ```
    /// use nalgebra::{Matrix3, Vector3, linalg::QR};
    ///
    /// let a = Matrix3::new(
    ///     2.0, 1.0, 1.0,
    ///     4.0, 1.0, 0.0,
    ///     2.0, 2.0, 1.0,
    /// );
    ///
    /// let b = Vector3::new(5.0, 5.0, 6.0);
    ///
    /// let qr = QR::new(a.clone());
    ///
    /// // Method 1: Using inverse (slower, less stable)
    /// let a_inv = qr.try_inverse().unwrap();
    /// let x1 = a_inv * b;
    ///
    /// // Method 2: Using solve (faster, more stable)
    /// let qr2 = QR::new(a.clone());
    /// let x2 = qr2.solve(&b).unwrap();
    ///
    /// // Both give the same answer
    /// assert!((x1 - x2).norm() < 1e-10);
    ///
    /// // But solve() is preferred for solving linear systems
    /// ```
    ///
    /// # Example: Application - transformation inversion
    ///
    /// ```
    /// use nalgebra::{Matrix3, Vector3, linalg::QR};
    ///
    /// // A transformation matrix (rotation + scaling)
    /// let transform = Matrix3::new(
    ///     2.0,  0.0, 1.0,
    ///     0.0,  2.0, 1.0,
    ///     0.0,  0.0, 1.0,
    /// );
    ///
    /// let qr = QR::new(transform.clone());
    /// let inverse_transform = qr.try_inverse().unwrap();
    ///
    /// // Apply transformation and its inverse
    /// let point = Vector3::new(1.0, 2.0, 3.0);
    /// let transformed = transform * point;
    /// let back = inverse_transform * transformed;
    ///
    /// // Should get back the original point
    /// assert!((back - point).norm() < 1e-10);
    /// ```
    ///
    /// # Example: Checking invertibility first
    ///
    /// ```
    /// use nalgebra::{Matrix3, linalg::QR};
    ///
    /// let a = Matrix3::new(
    ///     1.0, 2.0, 3.0,
    ///     0.0, 1.0, 4.0,
    ///     5.0, 6.0, 0.0,
    /// );
    ///
    /// let qr = QR::new(a);
    ///
    /// // Check before computing
    /// if qr.is_invertible() {
    ///     let inverse = qr.try_inverse().unwrap();
    ///     println!("Inverse computed: {:.3}", inverse);
    /// } else {
    ///     println!("Matrix is not invertible!");
    /// }
    /// ```
    ///
    /// # Performance Note
    ///
    /// Computing the inverse is `O(n^3)` and requires solving `n` linear systems.
    /// **Avoid computing the inverse when possible:**
    ///
    /// - To solve `A*x = b`, use [`solve()`](Self::solve) instead (faster and more accurate)
    /// - To compute `A^(-1)*b`, also use `solve()` - mathematically equivalent but better
    /// - Only compute the inverse when you truly need the full inverse matrix
    ///
    /// # See Also
    ///
    /// * [`is_invertible()`](Self::is_invertible) - Check if the matrix is invertible
    /// * [`solve()`](Self::solve) - Solve `A*x = b` without computing the inverse
    /// * [`solve_mut()`](Self::solve_mut) - In-place version of solve
    #[must_use]
    pub fn try_inverse(&self) -> Option<OMatrix<T, D, D>> {
        assert!(
            self.qr.is_square(),
            "QR inverse: unable to compute the inverse of a non-square matrix."
        );

        // TODO: is there a less naive method ?
        let (nrows, ncols) = self.qr.shape_generic();
        let mut res = OMatrix::identity_generic(nrows, ncols);

        if self.solve_mut(&mut res) {
            Some(res)
        } else {
            None
        }
    }

    /// Indicates if the decomposed matrix is invertible.
    ///
    /// A matrix is invertible (also called non-singular or non-degenerate) if it has an inverse.
    /// This is equivalent to:
    /// - The matrix has full rank
    /// - The determinant is non-zero
    /// - All rows (and columns) are linearly independent
    /// - The matrix can be used to solve systems of equations uniquely
    ///
    /// This method checks invertibility by examining the diagonal of the `R` matrix in the
    /// QR decomposition. Since `R` is upper triangular, the matrix is invertible if and only
    /// if all diagonal elements of `R` are non-zero.
    ///
    /// # Returns
    ///
    /// * `true` - The matrix is invertible (has an inverse)
    /// * `false` - The matrix is singular (no inverse exists)
    ///
    /// # Example: Checking before inverting
    ///
    /// ```
    /// use nalgebra::{Matrix3, linalg::QR};
    ///
    /// let a = Matrix3::new(
    ///     2.0, 1.0, 0.0,
    ///     1.0, 3.0, 1.0,
    ///     0.0, 1.0, 2.0,
    /// );
    ///
    /// let qr = QR::new(a);
    ///
    /// if qr.is_invertible() {
    ///     let inverse = qr.try_inverse().unwrap();
    ///     println!("Matrix has an inverse: {:.3}", inverse);
    /// } else {
    ///     println!("Matrix is singular, no inverse exists");
    /// }
    ///
    /// assert!(qr.is_invertible());
    /// ```
    ///
    /// # Example: Detecting singular matrices
    ///
    /// ```
    /// use nalgebra::{Matrix3, linalg::QR};
    ///
    /// // A clearly singular matrix (zero row)
    /// let singular = Matrix3::new(
    ///     1.0, 0.0, 0.0,
    ///     0.0, 0.0, 0.0,  // This row is all zeros
    ///     0.0, 0.0, 1.0,
    /// );
    ///
    /// let qr = QR::new(singular);
    /// // Note: QR decomposition may not always detect all singular matrices
    /// // For guaranteed detection, check the condition number or use SVD
    /// println!("Is invertible: {}", qr.is_invertible());
    ///
    /// // An invertible matrix
    /// let invertible = Matrix3::new(
    ///     1.0, 2.0, 3.0,
    ///     0.0, 1.0, 4.0,
    ///     5.0, 6.0, 0.0,
    /// );
    ///
    /// let qr2 = QR::new(invertible);
    /// assert!(qr2.is_invertible());
    /// ```
    ///
    /// # Example: Avoiding unnecessary computations
    ///
    /// ```
    /// use nalgebra::{Matrix3, Vector3, linalg::QR};
    ///
    /// let a = Matrix3::new(
    ///     1.0, 2.0, 3.0,
    ///     4.0, 5.0, 6.0,
    ///     7.0, 8.0, 9.0,  // Linearly dependent rows
    /// );
    ///
    /// let b = Vector3::new(1.0, 2.0, 3.0);
    ///
    /// let qr = QR::new(a);
    ///
    /// // Check before trying to solve
    /// if qr.is_invertible() {
    ///     let x = qr.solve(&b).unwrap();
    ///     println!("Solution: {}", x);
    /// } else {
    ///     println!("Cannot solve: matrix is singular");
    ///     // Handle the error appropriately
    /// }
    /// ```
    ///
    /// # Example: Numerical tolerance edge case
    ///
    /// ```
    /// use nalgebra::{Matrix3, linalg::QR};
    ///
    /// // A matrix that's numerically close to singular
    /// let nearly_singular = Matrix3::new(
    ///     1.0,   0.0,      0.0,
    ///     0.0,   1.0,      0.0,
    ///     0.0,   0.0,   1e-100,  // Very small but non-zero
    /// );
    ///
    /// let qr = QR::new(nearly_singular);
    ///
    /// // Technically invertible (diagonal element is non-zero)
    /// // But numerical issues may arise in practice
    /// println!("Is invertible: {}", qr.is_invertible());
    /// ```
    ///
    /// # Example: Relationship to matrix rank
    ///
    /// ```
    /// use nalgebra::{Matrix3, linalg::QR};
    ///
    /// let full_rank = Matrix3::new(
    ///     1.0, 0.0, 0.0,
    ///     0.0, 1.0, 0.0,
    ///     0.0, 0.0, 1.0,
    /// );
    ///
    /// let rank_deficient = Matrix3::new(
    ///     1.0, 2.0, 3.0,
    ///     0.0, 0.0, 0.0,  // Zero row means rank < 3
    ///     4.0, 5.0, 6.0,
    /// );
    ///
    /// let qr1 = QR::new(full_rank);
    /// let qr2 = QR::new(rank_deficient);
    ///
    /// // Full rank square matrices are invertible
    /// assert!(qr1.is_invertible());
    ///
    /// // Rank deficient matrices are not invertible (but QR may not always detect this)
    /// println!("Rank deficient is invertible: {}", qr2.is_invertible());
    /// ```
    ///
    /// # Performance Note
    ///
    /// This method is very fast - it only needs to check `n` diagonal elements of `R`.
    /// It's `O(n)` time complexity.
    ///
    /// Use this to avoid expensive operations (like computing the inverse or solving)
    /// when you're not sure if the matrix is invertible.
    ///
    /// # See Also
    ///
    /// * [`try_inverse()`](Self::try_inverse) - Compute the inverse (returns None if not invertible)
    /// * [`solve()`](Self::solve) - Solve linear systems (returns None if not invertible)
    /// * [`solve_mut()`](Self::solve_mut) - In-place solve (returns false if not invertible)
    #[must_use]
    pub fn is_invertible(&self) -> bool {
        assert!(
            self.qr.is_square(),
            "QR: unable to test the invertibility of a non-square matrix."
        );

        for i in 0..self.diag.len() {
            if self.diag[i].is_zero() {
                return false;
            }
        }

        true
    }

    // /// Computes the determinant of the decomposed matrix.
    // pub fn determinant(&self) -> T {
    //     let dim = self.qr.nrows();
    //     assert!(self.qr.is_square(), "QR determinant: unable to compute the determinant of a non-square matrix.");

    //     let mut res = T::one();
    //     for i in 0 .. dim {
    //         res *= unsafe { *self.diag.vget_unchecked(i) };
    //     }

    //     res self.q_determinant()
    // }
}
