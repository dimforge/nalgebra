#[cfg(feature = "serde-serialize-no-std")]
use serde::{Deserialize, Serialize};

use crate::allocator::Allocator;
use crate::base::{DefaultAllocator, OMatrix, OVector};
use crate::dimension::{Const, DimDiff, DimSub, U1};
use simba::scalar::ComplexField;

use crate::Matrix;
use crate::linalg::householder;
use std::mem::MaybeUninit;

/// Tridiagonalization of a symmetric matrix.
///
/// # What is Symmetric Tridiagonal Decomposition?
///
/// The symmetric tridiagonal decomposition transforms a symmetric matrix `M` into the form:
///
/// ```text
/// M = Q * T * Q^T
/// ```
///
/// where:
/// - `Q` is an orthogonal matrix (i.e., `Q^T * Q = I`)
/// - `T` is a tridiagonal matrix (non-zero elements only on the main diagonal and the
///   diagonals immediately above and below it)
///
/// A tridiagonal matrix has the form:
///
/// ```text
/// [ d₀  e₀  0   0   ... ]
/// [ e₀  d₁  e₁  0   ... ]
/// [ 0   e₁  d₂  e₂  ... ]
/// [ 0   0   e₂  d₃  ... ]
/// [ ...                 ]
/// ```
///
/// where `dᵢ` are the diagonal elements and `eᵢ` are the off-diagonal elements.
///
/// # Why is This Useful?
///
/// Tridiagonal decomposition is an essential preprocessing step for:
/// - **Computing eigenvalues and eigenvectors** of symmetric matrices efficiently
/// - **Solving symmetric linear systems** in specialized algorithms
/// - **Matrix computations** where the tridiagonal form simplifies operations
///
/// Many eigenvalue algorithms (like the QR algorithm) work much more efficiently on
/// tridiagonal matrices, reducing complexity from O(n³) per iteration to O(n) per iteration.
///
/// # Requirements
///
/// The input matrix must be **symmetric** (or Hermitian for complex matrices):
/// - For real matrices: `M = M^T`
/// - For complex matrices: `M = M^H` (where `^H` denotes conjugate transpose)
///
/// Only the **lower-triangular part** (including the diagonal) of the input matrix is read.
///
/// # Example: Basic decomposition
///
/// ```
/// use nalgebra::{Matrix3, linalg::SymmetricTridiagonal};
///
/// // A symmetric matrix
/// let m = Matrix3::new(
///     2.0, 1.0, 0.0,
///     1.0, 3.0, 1.0,
///     0.0, 1.0, 2.0,
/// );
///
/// // Compute the tridiagonal decomposition
/// let tri = SymmetricTridiagonal::new(m.clone());
///
/// // Get the orthogonal transformation matrix Q
/// let q = tri.q();
///
/// // Get the diagonal and off-diagonal elements
/// let diag = tri.diagonal();
/// let off_diag = tri.off_diagonal();
///
/// println!("Diagonal: {:?}", diag);
/// println!("Off-diagonal: {:?}", off_diag);
/// ```
///
/// # Example: Eigenvalue computation preparation
///
/// ```
/// use nalgebra::{Matrix4, linalg::SymmetricTridiagonal};
///
/// // A symmetric matrix whose eigenvalues we want to compute
/// let m = Matrix4::new(
///     4.0, 1.0, 0.5, 0.2,
///     1.0, 3.0, 0.3, 0.1,
///     0.5, 0.3, 2.0, 0.4,
///     0.2, 0.1, 0.4, 1.0,
/// );
///
/// // First step: reduce to tridiagonal form
/// let tri = SymmetricTridiagonal::new(m);
///
/// // The tridiagonal form is much more efficient for eigenvalue algorithms
/// let (q, diag, off_diag) = tri.unpack();
///
/// println!("Tridiagonal diagonal: {:?}", diag);
/// println!("Tridiagonal off-diagonal: {:?}", off_diag);
/// // Further eigenvalue computation would work on (diag, off_diag)
/// ```
///
/// # See Also
///
/// * [`SymmetricEigen`](crate::linalg::SymmetricEigen) - Computes eigenvalues/eigenvectors of symmetric matrices (uses this decomposition internally)
/// * [`Hessenberg`](crate::linalg::Hessenberg) - Similar decomposition for general (non-symmetric) matrices
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
pub struct SymmetricTridiagonal<T: ComplexField, D: DimSub<U1>>
where
    DefaultAllocator: Allocator<D, D> + Allocator<DimDiff<D, U1>>,
{
    tri: OMatrix<T, D, D>,
    off_diagonal: OVector<T, DimDiff<D, U1>>,
}

impl<T: ComplexField, D: DimSub<U1>> Copy for SymmetricTridiagonal<T, D>
where
    DefaultAllocator: Allocator<D, D> + Allocator<DimDiff<D, U1>>,
    OMatrix<T, D, D>: Copy,
    OVector<T, DimDiff<D, U1>>: Copy,
{
}

impl<T: ComplexField, D: DimSub<U1>> SymmetricTridiagonal<T, D>
where
    DefaultAllocator: Allocator<D, D> + Allocator<DimDiff<D, U1>>,
{
    /// Computes the tridiagonal decomposition of a symmetric matrix.
    ///
    /// This is the **primary way** to create a symmetric tridiagonal decomposition. It transforms
    /// a symmetric matrix into tridiagonal form using Householder reflections, which is an
    /// essential first step for many eigenvalue algorithms.
    ///
    /// # What It Does
    ///
    /// Given a symmetric matrix `M`, this computes matrices `Q` (orthogonal) and `T` (tridiagonal)
    /// such that:
    ///
    /// ```text
    /// M = Q * T * Q^T
    /// ```
    ///
    /// The transformation preserves:
    /// - The eigenvalues of the matrix (they're the same for `M` and `T`)
    /// - Symmetry properties
    /// - Numerical stability through orthogonal transformations
    ///
    /// # Arguments
    ///
    /// * `m` - A symmetric matrix. Only the **lower-triangular part** (including diagonal)
    ///         is read; the upper triangle is ignored.
    ///
    /// # Returns
    ///
    /// A `SymmetricTridiagonal` decomposition that can be queried for:
    /// - The orthogonal matrix `Q` via [`q()`](Self::q)
    /// - The diagonal elements via [`diagonal()`](Self::diagonal)
    /// - The off-diagonal elements via [`off_diagonal()`](Self::off_diagonal)
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - The matrix is not square
    /// - The matrix is empty (0×0)
    ///
    /// # Example: Basic usage
    ///
    /// ```
    /// use nalgebra::{Matrix3, linalg::SymmetricTridiagonal};
    ///
    /// // A symmetric matrix
    /// let m = Matrix3::new(
    ///     4.0, 2.0, 1.0,
    ///     2.0, 5.0, 3.0,
    ///     1.0, 3.0, 6.0,
    /// );
    ///
    /// // Compute the tridiagonal decomposition
    /// let tri = SymmetricTridiagonal::new(m.clone());
    ///
    /// // Get the components
    /// let q = tri.q();
    /// let diag = tri.diagonal();
    /// let off_diag = tri.off_diagonal();
    ///
    /// println!("Q (orthogonal matrix):\n{}", q);
    /// println!("Diagonal: {:?}", diag);
    /// println!("Off-diagonal: {:?}", off_diag);
    /// ```
    ///
    /// # Example: Verifying the decomposition
    ///
    /// ```
    /// use nalgebra::{Matrix3, linalg::SymmetricTridiagonal};
    ///
    /// let m = Matrix3::new(
    ///     4.0, 2.0, 1.0,
    ///     2.0, 5.0, 3.0,
    ///     1.0, 3.0, 6.0,
    /// );
    ///
    /// let tri = SymmetricTridiagonal::new(m.clone());
    ///
    /// // Reconstruct the original matrix
    /// let m_reconstructed = tri.recompose();
    ///
    /// // Verify: should be very close to the original
    /// assert!((m_reconstructed - m).norm() < 1e-10);
    /// ```
    ///
    /// # Example: Preparing for eigenvalue computation
    ///
    /// ```
    /// use nalgebra::{Matrix4, linalg::SymmetricTridiagonal};
    ///
    /// // A symmetric matrix
    /// let m = Matrix4::new(
    ///     3.0, 1.0, 0.5, 0.2,
    ///     1.0, 4.0, 0.3, 0.1,
    ///     0.5, 0.3, 2.0, 0.4,
    ///     0.2, 0.1, 0.4, 1.0,
    /// );
    ///
    /// // Reduce to tridiagonal form (O(n³) operation)
    /// let tri = SymmetricTridiagonal::new(m);
    ///
    /// // Extract diagonal and off-diagonal for eigenvalue algorithms
    /// // Working with tridiagonal form makes eigenvalue computation much faster
    /// let diag = tri.diagonal();
    /// let off_diag = tri.off_diagonal();
    ///
    /// // These compact vectors can be used with specialized eigenvalue algorithms
    /// println!("Tridiagonal form - Diagonal: {:?}", diag);
    /// println!("Tridiagonal form - Off-diagonal: {:?}", off_diag);
    /// ```
    ///
    /// # Example: Working with covariance matrices
    ///
    /// ```
    /// use nalgebra::Matrix3;
    /// use nalgebra::linalg::SymmetricTridiagonal;
    ///
    /// // A covariance matrix (always symmetric)
    /// let cov = Matrix3::new(
    ///     2.0, 0.8, 0.3,
    ///     0.8, 1.5, 0.4,
    ///     0.3, 0.4, 1.0,
    /// );
    ///
    /// // Tridiagonalize as a first step toward computing principal components
    /// let tri = SymmetricTridiagonal::new(cov);
    ///
    /// // The eigenvalues of the tridiagonal form are the same as the original
    /// // (they represent the variance along principal components)
    /// println!("Diagonal elements: {:?}", tri.diagonal());
    /// ```
    ///
    /// # Performance Note
    ///
    /// This operation has O(n³) complexity. The resulting tridiagonal form enables
    /// subsequent eigenvalue computations to run in O(n²) time instead of O(n³).
    ///
    /// # See Also
    ///
    /// * [`unpack`](Self::unpack) - Extract Q, diagonal, and off-diagonal elements
    /// * [`unpack_tridiagonal`](Self::unpack_tridiagonal) - Extract only diagonal and off-diagonal
    /// * [`recompose`](Self::recompose) - Reconstruct the original matrix
    /// * [`SymmetricEigen::new`](crate::linalg::SymmetricEigen::new) - Compute eigenvalues/eigenvectors directly
    pub fn new(mut m: OMatrix<T, D, D>) -> Self {
        let dim = m.shape_generic().0;

        assert!(
            m.is_square(),
            "Unable to compute the symmetric tridiagonal decomposition of a non-square matrix."
        );
        assert!(
            dim.value() != 0,
            "Unable to compute the symmetric tridiagonal decomposition of an empty matrix."
        );

        let mut off_diagonal = Matrix::uninit(dim.sub(Const::<1>), Const::<1>);
        let mut p = Matrix::zeros_generic(dim.sub(Const::<1>), Const::<1>);

        for i in 0..dim.value() - 1 {
            let mut m = m.rows_range_mut(i + 1..);
            let (mut axis, mut m) = m.columns_range_pair_mut(i, i + 1..);

            let (norm, not_zero) = householder::reflection_axis_mut(&mut axis);
            off_diagonal[i] = MaybeUninit::new(norm);

            if not_zero {
                let mut p = p.rows_range_mut(i..);

                p.hegemv(crate::convert(2.0), &m, &axis, T::zero());

                let dot = axis.dotc(&p);
                m.hegerc(-T::one(), &p, &axis, T::one());
                m.hegerc(-T::one(), &axis, &p, T::one());
                m.hegerc(dot * crate::convert(2.0), &axis, &axis, T::one());
            }
        }

        // Safety: off_diagonal has been fully initialized.
        let off_diagonal = unsafe { off_diagonal.assume_init() };
        Self {
            tri: m,
            off_diagonal,
        }
    }

    #[doc(hidden)]
    // For debugging.
    pub const fn internal_tri(&self) -> &OMatrix<T, D, D> {
        &self.tri
    }

    /// Extracts the orthogonal matrix Q and the tridiagonal components.
    ///
    /// This method consumes the decomposition and returns all three components:
    /// the orthogonal transformation matrix `Q`, the diagonal elements, and the
    /// off-diagonal elements of the tridiagonal matrix `T`.
    ///
    /// # What You Get
    ///
    /// Returns a tuple `(Q, diagonal, off_diagonal)` where:
    /// - `Q`: An orthogonal matrix such that `Q^T * Q = I`
    /// - `diagonal`: A vector containing the main diagonal of the tridiagonal matrix `T`
    /// - `off_diagonal`: A vector containing the elements immediately below/above the main diagonal
    ///
    /// Together, these satisfy: `M = Q * T * Q^T`, where `M` is the original matrix.
    ///
    /// # Returns
    ///
    /// A tuple of three elements:
    /// 1. The orthogonal matrix `Q` (size n×n)
    /// 2. The diagonal vector (size n)
    /// 3. The off-diagonal vector (size n-1)
    ///
    /// # Example: Basic unpacking
    ///
    /// ```
    /// use nalgebra::{Matrix3, linalg::SymmetricTridiagonal};
    ///
    /// let m = Matrix3::new(
    ///     4.0, 2.0, 1.0,
    ///     2.0, 5.0, 3.0,
    ///     1.0, 3.0, 6.0,
    /// );
    ///
    /// let tri = SymmetricTridiagonal::new(m.clone());
    ///
    /// // Extract all components
    /// let (q, diag, off_diag) = tri.unpack();
    ///
    /// println!("Orthogonal matrix Q:\n{}", q);
    /// println!("Diagonal: {:?}", diag);
    /// println!("Off-diagonal: {:?}", off_diag);
    ///
    /// // Verify Q is orthogonal: Q^T * Q = I
    /// let identity = q.transpose() * &q;
    /// assert!((identity - Matrix3::identity()).norm() < 1e-10);
    /// ```
    ///
    /// # Example: Reconstructing the tridiagonal matrix
    ///
    /// ```
    /// use nalgebra::{Matrix3, DMatrix, linalg::SymmetricTridiagonal};
    ///
    /// let m = Matrix3::new(
    ///     4.0, 2.0, 1.0,
    ///     2.0, 5.0, 3.0,
    ///     1.0, 3.0, 6.0,
    /// );
    ///
    /// let tri = SymmetricTridiagonal::new(m.clone());
    /// let (q, diag, off_diag) = tri.unpack();
    ///
    /// // Manually build the tridiagonal matrix T
    /// let mut t = DMatrix::zeros(3, 3);
    /// for i in 0..3 {
    ///     t[(i, i)] = diag[i];
    /// }
    /// for i in 0..2 {
    ///     t[(i, i + 1)] = off_diag[i];
    ///     t[(i + 1, i)] = off_diag[i];
    /// }
    ///
    /// // Verify: M = Q * T * Q^T
    /// let m_reconstructed = &q * &t * q.transpose();
    /// assert!((m_reconstructed - m).norm() < 1e-10);
    /// ```
    ///
    /// # Example: Using components for eigenvalue computation
    ///
    /// ```
    /// use nalgebra::{Matrix4, linalg::SymmetricTridiagonal};
    ///
    /// let m = Matrix4::new(
    ///     5.0, 1.0, 0.5, 0.2,
    ///     1.0, 4.0, 0.3, 0.1,
    ///     0.5, 0.3, 3.0, 0.4,
    ///     0.2, 0.1, 0.4, 2.0,
    /// );
    ///
    /// let tri = SymmetricTridiagonal::new(m);
    /// let (q, diag, off_diag) = tri.unpack();
    ///
    /// // The diagonal and off-diagonal vectors define a tridiagonal matrix
    /// // that has the same eigenvalues as the original matrix
    /// // These compact vectors are much more efficient for eigenvalue algorithms
    ///
    /// println!("Tridiagonal diagonal: {:?}", diag);
    /// println!("Tridiagonal off-diagonal: {:?}", off_diag);
    /// println!("Transformation matrix Q has {} columns", q.ncols());
    /// ```
    ///
    /// # Performance Note
    ///
    /// This method consumes the decomposition. If you need to keep the decomposition
    /// for later use, use the non-consuming methods [`q()`](Self::q),
    /// [`diagonal()`](Self::diagonal), and [`off_diagonal()`](Self::off_diagonal) instead
    /// (though they will allocate and copy the data).
    ///
    /// # See Also
    ///
    /// * [`unpack_tridiagonal`](Self::unpack_tridiagonal) - Extract only diagonal and off-diagonal (no Q)
    /// * [`q`](Self::q) - Get Q without consuming the decomposition
    /// * [`diagonal`](Self::diagonal) - Get diagonal without consuming the decomposition
    /// * [`off_diagonal`](Self::off_diagonal) - Get off-diagonal without consuming the decomposition
    /// * [`recompose`](Self::recompose) - Reconstruct the original matrix
    pub fn unpack(
        self,
    ) -> (
        OMatrix<T, D, D>,
        OVector<T::RealField, D>,
        OVector<T::RealField, DimDiff<D, U1>>,
    )
    where
        DefaultAllocator: Allocator<D> + Allocator<DimDiff<D, U1>>,
    {
        let diag = self.diagonal();
        let q = self.q();

        (q, diag, self.off_diagonal.map(T::modulus))
    }

    /// Extracts only the tridiagonal components without computing Q.
    ///
    /// This method is more efficient than [`unpack`](Self::unpack) when you only need the
    /// tridiagonal form and don't need the orthogonal transformation matrix `Q`. This is
    /// particularly useful when preparing data for eigenvalue algorithms that only work
    /// with the tridiagonal representation.
    ///
    /// # What You Get
    ///
    /// Returns a tuple `(diagonal, off_diagonal)` where:
    /// - `diagonal`: A vector containing the main diagonal of the tridiagonal matrix `T`
    /// - `off_diagonal`: A vector containing the elements immediately below/above the main diagonal
    ///
    /// These two vectors completely define the tridiagonal matrix `T` such that `M = Q * T * Q^T`.
    ///
    /// # Returns
    ///
    /// A tuple of two elements:
    /// 1. The diagonal vector (size n)
    /// 2. The off-diagonal vector (size n-1)
    ///
    /// # Example: Basic usage
    ///
    /// ```
    /// use nalgebra::{Matrix3, linalg::SymmetricTridiagonal};
    ///
    /// let m = Matrix3::new(
    ///     4.0, 2.0, 1.0,
    ///     2.0, 5.0, 3.0,
    ///     1.0, 3.0, 6.0,
    /// );
    ///
    /// let tri = SymmetricTridiagonal::new(m);
    ///
    /// // Extract only the tridiagonal components (faster than unpack)
    /// let (diag, off_diag) = tri.unpack_tridiagonal();
    ///
    /// println!("Diagonal: {:?}", diag);
    /// println!("Off-diagonal: {:?}", off_diag);
    /// ```
    ///
    /// # Example: Preparing for specialized eigenvalue algorithms
    ///
    /// ```
    /// use nalgebra::{Matrix4, linalg::SymmetricTridiagonal};
    ///
    /// let m = Matrix4::new(
    ///     5.0, 1.0, 0.5, 0.2,
    ///     1.0, 4.0, 0.3, 0.1,
    ///     0.5, 0.3, 3.0, 0.4,
    ///     0.2, 0.1, 0.4, 2.0,
    /// );
    ///
    /// let tri = SymmetricTridiagonal::new(m);
    ///
    /// // Get just the tridiagonal form
    /// let (diag, off_diag) = tri.unpack_tridiagonal();
    ///
    /// // These compact vectors can be fed to specialized tridiagonal eigenvalue solvers
    /// // which are much faster than general eigenvalue algorithms
    /// println!("Ready for eigenvalue computation:");
    /// println!("  Diagonal ({} elements): {:?}", diag.len(), diag);
    /// println!("  Off-diagonal ({} elements): {:?}", off_diag.len(), off_diag);
    /// ```
    ///
    /// # Example: Understanding the tridiagonal structure
    ///
    /// ```
    /// use nalgebra::{Matrix3, DMatrix, linalg::SymmetricTridiagonal};
    ///
    /// let m = Matrix3::new(
    ///     3.0, 1.0, 0.5,
    ///     1.0, 2.0, 0.8,
    ///     0.5, 0.8, 4.0,
    /// );
    ///
    /// let tri = SymmetricTridiagonal::new(m);
    /// let (diag, off_diag) = tri.unpack_tridiagonal();
    ///
    /// // Build the tridiagonal matrix T explicitly to see its structure
    /// let mut t = DMatrix::zeros(3, 3);
    ///
    /// // Fill diagonal
    /// for i in 0..3 {
    ///     t[(i, i)] = diag[i];
    /// }
    ///
    /// // Fill off-diagonal (both above and below main diagonal)
    /// for i in 0..2 {
    ///     t[(i, i + 1)] = off_diag[i];
    ///     t[(i + 1, i)] = off_diag[i];
    /// }
    ///
    /// println!("Tridiagonal matrix T:\n{}", t);
    /// // T has non-zero elements only on three diagonals:
    /// // [ d₀  e₀  0  ]
    /// // [ e₀  d₁  e₁ ]
    /// // [ 0   e₁  d₂ ]
    /// ```
    ///
    /// # Example: Comparing storage efficiency
    ///
    /// ```
    /// use nalgebra::{Matrix5, linalg::SymmetricTridiagonal};
    ///
    /// let m = Matrix5::from_diagonal_element(2.0) + Matrix5::from_element(0.1);
    ///
    /// let tri = SymmetricTridiagonal::new(m);
    /// let (diag, off_diag) = tri.unpack_tridiagonal();
    ///
    /// // Original matrix: 5×5 = 25 elements
    /// // Tridiagonal form: 5 + 4 = 9 elements
    /// println!("Original matrix storage: 25 elements");
    /// println!("Tridiagonal form storage: {} + {} = {} elements",
    ///          diag.len(), off_diag.len(), diag.len() + off_diag.len());
    /// println!("Storage reduction: {}%",
    ///          100.0 * (1.0 - (diag.len() + off_diag.len()) as f64 / 25.0));
    /// ```
    ///
    /// # Performance Note
    ///
    /// This method is more efficient than [`unpack`](Self::unpack) because it doesn't
    /// compute the orthogonal matrix `Q`, which requires additional allocations and
    /// computation. Use this when you only need the tridiagonal form itself.
    ///
    /// # See Also
    ///
    /// * [`unpack`](Self::unpack) - Extract Q, diagonal, and off-diagonal
    /// * [`diagonal`](Self::diagonal) - Get only the diagonal without consuming
    /// * [`off_diagonal`](Self::off_diagonal) - Get only the off-diagonal without consuming
    /// * [`SymmetricEigen`](crate::linalg::SymmetricEigen) - Compute eigenvalues from tridiagonal form
    pub fn unpack_tridiagonal(
        self,
    ) -> (
        OVector<T::RealField, D>,
        OVector<T::RealField, DimDiff<D, U1>>,
    )
    where
        DefaultAllocator: Allocator<D> + Allocator<DimDiff<D, U1>>,
    {
        (self.diagonal(), self.off_diagonal.map(T::modulus))
    }

    /// Extracts the main diagonal of the tridiagonal matrix.
    ///
    /// This method returns a **copy** of the diagonal elements of the tridiagonal matrix `T`
    /// without consuming the decomposition. The diagonal contains the elements that would
    /// appear on the main diagonal of `T` in the decomposition `M = Q * T * Q^T`.
    ///
    /// # What You Get
    ///
    /// A vector containing the diagonal elements `[d₀, d₁, d₂, ..., dₙ₋₁]` of the
    /// tridiagonal matrix. For a symmetric matrix reduced to tridiagonal form, these
    /// diagonal elements are always real, even for complex input matrices.
    ///
    /// # Returns
    ///
    /// A vector of length n containing the diagonal elements.
    ///
    /// # Example: Basic usage
    ///
    /// ```
    /// use nalgebra::{Matrix3, linalg::SymmetricTridiagonal};
    ///
    /// let m = Matrix3::new(
    ///     4.0, 2.0, 1.0,
    ///     2.0, 5.0, 3.0,
    ///     1.0, 3.0, 6.0,
    /// );
    ///
    /// let tri = SymmetricTridiagonal::new(m);
    /// let diag = tri.diagonal();
    ///
    /// println!("Diagonal elements: {:?}", diag);
    /// println!("Number of diagonal elements: {}", diag.len());
    /// ```
    ///
    /// # Example: Using with off-diagonal for eigenvalue computation
    ///
    /// ```
    /// use nalgebra::{Matrix4, linalg::SymmetricTridiagonal};
    ///
    /// let m = Matrix4::new(
    ///     5.0, 1.0, 0.5, 0.2,
    ///     1.0, 4.0, 0.3, 0.1,
    ///     0.5, 0.3, 3.0, 0.4,
    ///     0.2, 0.1, 0.4, 2.0,
    /// );
    ///
    /// let tri = SymmetricTridiagonal::new(m);
    ///
    /// // Get diagonal and off-diagonal separately
    /// let diag = tri.diagonal();
    /// let off_diag = tri.off_diagonal();
    ///
    /// // These define a tridiagonal matrix with the same eigenvalues as the original
    /// println!("Diagonal: {:?}", diag);
    /// println!("Off-diagonal: {:?}", off_diag);
    ///
    /// // Can still use tri for other operations
    /// let q = tri.q();
    /// println!("Q matrix size: {}×{}", q.nrows(), q.ncols());
    /// ```
    ///
    /// # Example: Examining the tridiagonal structure
    ///
    /// ```
    /// use nalgebra::{Matrix3, DMatrix, linalg::SymmetricTridiagonal};
    ///
    /// let m = Matrix3::new(
    ///     2.0, 1.0, 0.5,
    ///     1.0, 3.0, 0.8,
    ///     0.5, 0.8, 4.0,
    /// );
    ///
    /// let tri = SymmetricTridiagonal::new(m);
    /// let diag = tri.diagonal();
    /// let off_diag = tri.off_diagonal();
    ///
    /// // Construct the tridiagonal matrix explicitly
    /// let mut t = DMatrix::zeros(3, 3);
    /// for i in 0..diag.len() {
    ///     t[(i, i)] = diag[i];
    /// }
    /// for i in 0..off_diag.len() {
    ///     t[(i, i + 1)] = off_diag[i];
    ///     t[(i + 1, i)] = off_diag[i];
    /// }
    ///
    /// println!("Tridiagonal matrix T:\n{}", t);
    /// ```
    ///
    /// # Example: Comparing with the original matrix
    ///
    /// ```
    /// use nalgebra::{Matrix3, linalg::SymmetricTridiagonal};
    ///
    /// let m = Matrix3::new(
    ///     5.0, 1.0, 0.0,
    ///     1.0, 4.0, 1.0,
    ///     0.0, 1.0, 3.0,
    /// );
    ///
    /// let tri = SymmetricTridiagonal::new(m.clone());
    /// let diag = tri.diagonal();
    ///
    /// // The trace (sum of diagonal elements) is preserved
    /// let original_trace: f64 = (0..3).map(|i| m[(i, i)]).sum();
    /// let tridiag_trace: f64 = diag.iter().sum();
    ///
    /// println!("Original trace: {}", original_trace);
    /// println!("Tridiagonal trace: {}", tridiag_trace);
    /// assert!((original_trace - tridiag_trace).abs() < 1e-10);
    /// ```
    ///
    /// # Performance Note
    ///
    /// This method creates a copy of the diagonal elements. It does not consume the
    /// decomposition, allowing you to continue using it for other operations.
    ///
    /// # See Also
    ///
    /// * [`off_diagonal`](Self::off_diagonal) - Get the off-diagonal elements
    /// * [`unpack_tridiagonal`](Self::unpack_tridiagonal) - Get both diagonal and off-diagonal at once
    /// * [`unpack`](Self::unpack) - Get Q, diagonal, and off-diagonal together
    #[must_use]
    pub fn diagonal(&self) -> OVector<T::RealField, D>
    where
        DefaultAllocator: Allocator<D>,
    {
        self.tri.map_diagonal(|e| e.real())
    }

    /// Extracts the off-diagonal elements of the tridiagonal matrix.
    ///
    /// This method returns a **copy** of the off-diagonal elements (also called the
    /// super-diagonal or sub-diagonal) of the tridiagonal matrix `T` without consuming
    /// the decomposition. These elements appear immediately above and below the main
    /// diagonal in the tridiagonal form.
    ///
    /// # What You Get
    ///
    /// A vector containing the off-diagonal elements `[e₀, e₁, e₂, ..., eₙ₋₂]` of the
    /// tridiagonal matrix. For a symmetric tridiagonal matrix, the same values appear
    /// both above and below the diagonal (i.e., `T[i, i+1] = T[i+1, i] = eᵢ`).
    ///
    /// For an n×n matrix, there are n-1 off-diagonal elements.
    ///
    /// # Returns
    ///
    /// A vector of length n-1 containing the off-diagonal elements.
    ///
    /// # Example: Basic usage
    ///
    /// ```
    /// use nalgebra::{Matrix3, linalg::SymmetricTridiagonal};
    ///
    /// let m = Matrix3::new(
    ///     4.0, 2.0, 1.0,
    ///     2.0, 5.0, 3.0,
    ///     1.0, 3.0, 6.0,
    /// );
    ///
    /// let tri = SymmetricTridiagonal::new(m);
    /// let off_diag = tri.off_diagonal();
    ///
    /// println!("Off-diagonal elements: {:?}", off_diag);
    /// println!("Number of off-diagonal elements: {}", off_diag.len());
    /// // For a 3×3 matrix, there are 2 off-diagonal elements
    /// assert_eq!(off_diag.len(), 2);
    /// ```
    ///
    /// # Example: Understanding the tridiagonal structure
    ///
    /// ```
    /// use nalgebra::{Matrix4, DMatrix, linalg::SymmetricTridiagonal};
    ///
    /// let m = Matrix4::new(
    ///     5.0, 1.0, 0.5, 0.2,
    ///     1.0, 4.0, 0.3, 0.1,
    ///     0.5, 0.3, 3.0, 0.4,
    ///     0.2, 0.1, 0.4, 2.0,
    /// );
    ///
    /// let tri = SymmetricTridiagonal::new(m);
    /// let diag = tri.diagonal();
    /// let off_diag = tri.off_diagonal();
    ///
    /// // Build the full tridiagonal matrix to visualize
    /// let mut t = DMatrix::zeros(4, 4);
    ///
    /// // Diagonal: d₀, d₁, d₂, d₃
    /// for i in 0..diag.len() {
    ///     t[(i, i)] = diag[i];
    /// }
    ///
    /// // Off-diagonal: e₀, e₁, e₂ appear both above and below diagonal
    /// for i in 0..off_diag.len() {
    ///     t[(i, i + 1)] = off_diag[i];  // Above diagonal
    ///     t[(i + 1, i)] = off_diag[i];  // Below diagonal (symmetric)
    /// }
    ///
    /// println!("Tridiagonal matrix T:\n{}", t);
    /// // Structure:
    /// // [ d₀  e₀  0   0  ]
    /// // [ e₀  d₁  e₁  0  ]
    /// // [ 0   e₁  d₂  e₂ ]
    /// // [ 0   0   e₂  d₃ ]
    /// ```
    ///
    /// # Example: Combining with diagonal for eigenvalue computation
    ///
    /// ```
    /// use nalgebra::{Matrix3, linalg::SymmetricTridiagonal};
    ///
    /// let m = Matrix3::new(
    ///     3.0, 1.0, 0.5,
    ///     1.0, 2.0, 0.8,
    ///     0.5, 0.8, 4.0,
    /// );
    ///
    /// let tri = SymmetricTridiagonal::new(m);
    ///
    /// // Get both components
    /// let diag = tri.diagonal();
    /// let off_diag = tri.off_diagonal();
    ///
    /// println!("Tridiagonal form:");
    /// println!("  Diagonal ({} elements): {:?}", diag.len(), diag);
    /// println!("  Off-diagonal ({} elements): {:?}", off_diag.len(), off_diag);
    ///
    /// // These compact vectors contain all information needed for eigenvalue algorithms
    /// // Much more efficient than working with the full n×n matrix
    /// ```
    ///
    /// # Example: Storage efficiency
    ///
    /// ```
    /// use nalgebra::{Matrix6, linalg::SymmetricTridiagonal};
    ///
    /// // Create a 6×6 symmetric matrix
    /// let m = Matrix6::from_diagonal_element(3.0) + Matrix6::from_element(0.2);
    ///
    /// let tri = SymmetricTridiagonal::new(m);
    /// let diag = tri.diagonal();
    /// let off_diag = tri.off_diagonal();
    ///
    /// println!("Original matrix: 6×6 = 36 elements");
    /// println!("Tridiagonal form: {} diagonal + {} off-diagonal = {} total",
    ///          diag.len(), off_diag.len(), diag.len() + off_diag.len());
    /// println!("Storage saved: {} elements", 36 - (diag.len() + off_diag.len()));
    /// ```
    ///
    /// # Example: Non-consuming access
    ///
    /// ```
    /// use nalgebra::{Matrix3, linalg::SymmetricTridiagonal};
    ///
    /// let m = Matrix3::new(
    ///     4.0, 2.0, 1.0,
    ///     2.0, 5.0, 3.0,
    ///     1.0, 3.0, 6.0,
    /// );
    ///
    /// let tri = SymmetricTridiagonal::new(m);
    ///
    /// // Get off-diagonal without consuming the decomposition
    /// let off_diag = tri.off_diagonal();
    /// println!("Off-diagonal: {:?}", off_diag);
    ///
    /// // Can still use tri for other operations
    /// let diag = tri.diagonal();
    /// println!("Diagonal: {:?}", diag);
    ///
    /// let q = tri.q();
    /// println!("Q matrix:\n{}", q);
    /// ```
    ///
    /// # Performance Note
    ///
    /// This method creates a copy of the off-diagonal elements. It does not consume the
    /// decomposition, allowing you to continue using it for other operations.
    ///
    /// # See Also
    ///
    /// * [`diagonal`](Self::diagonal) - Get the diagonal elements
    /// * [`unpack_tridiagonal`](Self::unpack_tridiagonal) - Get both diagonal and off-diagonal at once
    /// * [`unpack`](Self::unpack) - Get Q, diagonal, and off-diagonal together
    #[must_use]
    pub fn off_diagonal(&self) -> OVector<T::RealField, DimDiff<D, U1>>
    where
        DefaultAllocator: Allocator<DimDiff<D, U1>>,
    {
        self.off_diagonal.map(T::modulus)
    }

    /// Computes the orthogonal transformation matrix Q.
    ///
    /// This method constructs and returns the orthogonal matrix `Q` that transforms the
    /// original symmetric matrix into tridiagonal form. In the decomposition `M = Q * T * Q^T`,
    /// this is the `Q` matrix.
    ///
    /// # What is Q?
    ///
    /// The matrix `Q` is an **orthogonal matrix**, meaning:
    /// - `Q^T * Q = I` (the identity matrix)
    /// - `Q * Q^T = I`
    /// - All columns of `Q` are orthonormal (mutually perpendicular with unit length)
    /// - `Q` preserves lengths and angles (it's an isometry)
    ///
    /// For complex matrices, `Q` is unitary: `Q^H * Q = I` (where `^H` is conjugate transpose).
    ///
    /// # Returns
    ///
    /// An n×n orthogonal matrix `Q`.
    ///
    /// # Example: Basic usage
    ///
    /// ```
    /// use nalgebra::{Matrix3, linalg::SymmetricTridiagonal};
    ///
    /// let m = Matrix3::new(
    ///     4.0, 2.0, 1.0,
    ///     2.0, 5.0, 3.0,
    ///     1.0, 3.0, 6.0,
    /// );
    ///
    /// let tri = SymmetricTridiagonal::new(m);
    /// let q = tri.q();
    ///
    /// println!("Orthogonal matrix Q:\n{}", q);
    ///
    /// // Verify Q is orthogonal: Q^T * Q should be the identity
    /// let identity = q.transpose() * &q;
    /// println!("Q^T * Q:\n{}", identity);
    /// assert!((identity - Matrix3::identity()).norm() < 1e-10);
    /// ```
    ///
    /// # Example: Verifying the decomposition
    ///
    /// ```
    /// use nalgebra::{Matrix3, DMatrix, linalg::SymmetricTridiagonal};
    ///
    /// let m = Matrix3::new(
    ///     4.0, 2.0, 1.0,
    ///     2.0, 5.0, 3.0,
    ///     1.0, 3.0, 6.0,
    /// );
    ///
    /// let tri = SymmetricTridiagonal::new(m.clone());
    ///
    /// // Get all components
    /// let q = tri.q();
    /// let diag = tri.diagonal();
    /// let off_diag = tri.off_diagonal();
    ///
    /// // Build the tridiagonal matrix T
    /// let mut t = DMatrix::zeros(3, 3);
    /// for i in 0..3 {
    ///     t[(i, i)] = diag[i];
    /// }
    /// for i in 0..2 {
    ///     t[(i, i + 1)] = off_diag[i];
    ///     t[(i + 1, i)] = off_diag[i];
    /// }
    ///
    /// // Verify: M = Q * T * Q^T
    /// let m_reconstructed = &q * &t * q.transpose();
    /// assert!((m_reconstructed - m).norm() < 1e-10);
    /// ```
    ///
    /// # Example: Understanding orthogonality
    ///
    /// ```
    /// use nalgebra::{Matrix3, linalg::SymmetricTridiagonal};
    ///
    /// let m = Matrix3::new(
    ///     3.0, 1.0, 0.5,
    ///     1.0, 2.0, 0.8,
    ///     0.5, 0.8, 4.0,
    /// );
    ///
    /// let tri = SymmetricTridiagonal::new(m);
    /// let q = tri.q();
    ///
    /// // Check orthogonality: columns should be orthonormal
    /// for i in 0..3 {
    ///     let col_i = q.column(i);
    ///
    ///     // Each column should have unit length
    ///     let norm: f64 = col_i.norm();
    ///     assert!((norm - 1.0).abs() < 1e-10);
    ///
    ///     // Different columns should be perpendicular
    ///     for j in (i + 1)..3 {
    ///         let col_j = q.column(j);
    ///         let dot_product: f64 = col_i.dot(&col_j);
    ///         assert!(dot_product.abs() < 1e-10);
    ///     }
    /// }
    /// println!("Q has orthonormal columns!");
    /// ```
    ///
    /// # Example: Using Q to transform vectors
    ///
    /// ```
    /// use nalgebra::{Matrix3, Vector3, linalg::SymmetricTridiagonal};
    ///
    /// let m = Matrix3::new(
    ///     4.0, 2.0, 1.0,
    ///     2.0, 5.0, 3.0,
    ///     1.0, 3.0, 6.0,
    /// );
    ///
    /// let tri = SymmetricTridiagonal::new(m);
    /// let q = tri.q();
    ///
    /// // Q transforms vectors between the original and tridiagonal coordinate systems
    /// let v = Vector3::new(1.0, 2.0, 3.0);
    ///
    /// // Transform to tridiagonal coordinates
    /// let v_tri = q.transpose() * &v;
    ///
    /// // Transform back to original coordinates
    /// let v_back = &q * v_tri;
    ///
    /// // Should get the original vector back (Q is orthogonal)
    /// assert!((v_back - v).norm() < 1e-10);
    /// println!("Orthogonal transformation preserves vectors!");
    /// ```
    ///
    /// # Example: Q preserves the determinant sign
    ///
    /// ```
    /// use nalgebra::{Matrix3, linalg::SymmetricTridiagonal};
    ///
    /// let m = Matrix3::new(
    ///     4.0, 2.0, 1.0,
    ///     2.0, 5.0, 3.0,
    ///     1.0, 3.0, 6.0,
    /// );
    ///
    /// let tri = SymmetricTridiagonal::new(m.clone());
    /// let q = tri.q();
    ///
    /// // Q is orthogonal, so det(Q) = ±1
    /// let det_q: f64 = q.determinant();
    /// println!("det(Q) = {}", det_q);
    /// assert!((det_q.abs() - 1.0).abs() < 1e-10);
    ///
    /// // The determinant is preserved through the transformation
    /// let det_m: f64 = m.determinant();
    /// let det_t: f64 = tri.recompose().determinant();
    /// assert!((det_m - det_t).abs() < 1e-10);
    /// ```
    ///
    /// # Performance Note
    ///
    /// Computing `Q` requires assembling the Householder reflections used during
    /// the decomposition, which is an O(n³) operation. If you only need the tridiagonal
    /// components, use [`unpack_tridiagonal`](Self::unpack_tridiagonal) instead.
    ///
    /// # See Also
    ///
    /// * [`unpack`](Self::unpack) - Get Q along with diagonal and off-diagonal
    /// * [`diagonal`](Self::diagonal) - Get the diagonal elements
    /// * [`off_diagonal`](Self::off_diagonal) - Get the off-diagonal elements
    /// * [`recompose`](Self::recompose) - Reconstruct the original matrix using Q
    #[must_use]
    pub fn q(&self) -> OMatrix<T, D, D> {
        householder::assemble_q(&self.tri, self.off_diagonal.as_slice())
    }

    /// Reconstructs the original symmetric matrix from the decomposition.
    ///
    /// This method consumes the decomposition and reconstructs the original matrix `M`
    /// using the formula `M = Q * T * Q^T`, where `Q` is the orthogonal matrix and `T`
    /// is the tridiagonal matrix.
    ///
    /// This is primarily useful for:
    /// - **Verifying** that the decomposition is correct
    /// - **Testing** and validation
    /// - Understanding how the decomposition works
    ///
    /// In practice, you rarely need to reconstruct the original matrix since the
    /// decomposition itself is more useful for computations.
    ///
    /// # Returns
    ///
    /// The original symmetric matrix that was decomposed.
    ///
    /// # Example: Basic reconstruction
    ///
    /// ```
    /// use nalgebra::{Matrix3, linalg::SymmetricTridiagonal};
    ///
    /// let m = Matrix3::new(
    ///     4.0, 2.0, 1.0,
    ///     2.0, 5.0, 3.0,
    ///     1.0, 3.0, 6.0,
    /// );
    ///
    /// let tri = SymmetricTridiagonal::new(m.clone());
    ///
    /// // Reconstruct the original matrix
    /// let m_reconstructed = tri.recompose();
    ///
    /// // Should be very close to the original
    /// assert!((m_reconstructed - m).norm() < 1e-10);
    /// println!("Successfully reconstructed the matrix!");
    /// ```
    ///
    /// # Example: Verifying the decomposition preserves properties
    ///
    /// ```
    /// use nalgebra::{Matrix3, linalg::SymmetricTridiagonal};
    ///
    /// let m = Matrix3::new(
    ///     5.0, 1.0, 0.5,
    ///     1.0, 4.0, 0.8,
    ///     0.5, 0.8, 3.0,
    /// );
    ///
    /// // Compute various properties of the original matrix
    /// let trace_orig: f64 = m.trace();
    /// let det_orig: f64 = m.determinant();
    ///
    /// // Decompose and recompose
    /// let tri = SymmetricTridiagonal::new(m.clone());
    /// let m_recomposed = tri.recompose();
    ///
    /// // Properties should be preserved
    /// let trace_recomp: f64 = m_recomposed.trace();
    /// let det_recomp: f64 = m_recomposed.determinant();
    ///
    /// assert!((trace_orig - trace_recomp).abs() < 1e-10);
    /// assert!((det_orig - det_recomp).abs() < 1e-10);
    ///
    /// println!("Trace preserved: {} ≈ {}", trace_orig, trace_recomp);
    /// println!("Determinant preserved: {} ≈ {}", det_orig, det_recomp);
    /// ```
    ///
    /// # Example: Understanding the decomposition formula
    ///
    /// ```
    /// use nalgebra::{Matrix3, DMatrix, linalg::SymmetricTridiagonal};
    ///
    /// let m = Matrix3::new(
    ///     4.0, 2.0, 1.0,
    ///     2.0, 5.0, 3.0,
    ///     1.0, 3.0, 6.0,
    /// );
    ///
    /// let tri = SymmetricTridiagonal::new(m.clone());
    ///
    /// // Manually compute M = Q * T * Q^T to understand recompose()
    /// let q = tri.q();
    /// let diag = tri.diagonal();
    /// let off_diag = tri.off_diagonal();
    ///
    /// // Build T explicitly
    /// let mut t = DMatrix::zeros(3, 3);
    /// for i in 0..3 {
    ///     t[(i, i)] = diag[i];
    /// }
    /// for i in 0..2 {
    ///     t[(i, i + 1)] = off_diag[i];
    ///     t[(i + 1, i)] = off_diag[i];
    /// }
    ///
    /// // M = Q * T * Q^T
    /// let m_manual = &q * &t * q.transpose();
    ///
    /// // This should equal recompose()
    /// let m_recomposed = SymmetricTridiagonal::new(m.clone()).recompose();
    /// assert!((m_manual - m_recomposed).norm() < 1e-10);
    /// ```
    ///
    /// # Example: Testing numerical stability
    ///
    /// ```
    /// use nalgebra::{Matrix4, linalg::SymmetricTridiagonal};
    ///
    /// // Create a matrix with a wide range of values
    /// let m = Matrix4::new(
    ///     100.0,   1.0,   0.1,  0.01,
    ///       1.0,  50.0,   2.0,   0.5,
    ///       0.1,   2.0,  25.0,   3.0,
    ///      0.01,   0.5,   3.0,  10.0,
    /// );
    ///
    /// let tri = SymmetricTridiagonal::new(m.clone());
    /// let m_reconstructed = tri.recompose();
    ///
    /// // Check relative error
    /// let error = (m_reconstructed - m.clone()).norm() / m.norm();
    /// println!("Relative reconstruction error: {}", error);
    /// assert!(error < 1e-10);
    /// ```
    ///
    /// # Performance Note
    ///
    /// This method is O(n³) because it:
    /// 1. Computes the orthogonal matrix Q (O(n³))
    /// 2. Performs matrix multiplications Q * T and (Q*T) * Q^T (O(n³))
    ///
    /// Use this primarily for verification and testing, not in performance-critical code.
    ///
    /// # See Also
    ///
    /// * [`new`](Self::new) - Create the decomposition
    /// * [`q`](Self::q) - Get the orthogonal matrix Q
    /// * [`unpack`](Self::unpack) - Extract all components without reconstructing
    /// * [`unpack_tridiagonal`](Self::unpack_tridiagonal) - Get just the tridiagonal form
    pub fn recompose(mut self) -> OMatrix<T, D, D> {
        let q = self.q();
        self.tri.fill_lower_triangle(T::zero(), 2);
        self.tri.fill_upper_triangle(T::zero(), 2);

        for i in 0..self.off_diagonal.len() {
            let val = T::from_real(self.off_diagonal[i].clone().modulus());
            self.tri[(i + 1, i)] = val.clone();
            self.tri[(i, i + 1)] = val;
        }

        &q * self.tri * q.adjoint()
    }
}
