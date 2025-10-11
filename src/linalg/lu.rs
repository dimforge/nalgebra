#[cfg(feature = "serde-serialize-no-std")]
use serde::{Deserialize, Serialize};

use crate::allocator::{Allocator, Reallocator};
use crate::base::{DefaultAllocator, Matrix, OMatrix, Scalar};
use crate::constraint::{SameNumberOfRows, ShapeConstraint};
use crate::dimension::{Dim, DimMin, DimMinimum};
use crate::storage::{Storage, StorageMut};
use simba::scalar::{ComplexField, Field};
use std::mem;

use crate::linalg::PermutationSequence;

/// LU decomposition with partial (row) pivoting.
///
/// # What is LU Decomposition?
///
/// LU decomposition is a fundamental matrix factorization technique that breaks down a square
/// matrix `A` into the product of two triangular matrices:
/// - **L**: A lower triangular matrix (all entries above the diagonal are zero)
/// - **U**: An upper triangular matrix (all entries below the diagonal are zero)
/// - **P**: A permutation matrix (for numerical stability)
///
/// The decomposition satisfies: `P * A = L * U`
///
/// LU decomposition is particularly useful for:
/// - Solving systems of linear equations efficiently
/// - Computing matrix determinants
/// - Finding matrix inverses
/// - Solving multiple systems with the same coefficient matrix
///
/// # Why Pivoting?
///
/// Partial (row) pivoting improves numerical stability by reordering rows during decomposition
/// to avoid division by very small numbers, which can lead to numerical errors.
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
/// let lu = m.lu();
///
/// // Extract the L, U, and P components
/// let (p, l, u) = lu.unpack();
///
/// // Verify that P * A = L * U
/// let mut m_copy = m.clone();
/// p.permute_rows(&mut m_copy);
/// assert!((m_copy - l * u).norm() < 1e-10);
/// ```
///
/// # See Also
///
/// - [`LU::new`]: Constructs a new LU decomposition
/// - [`LU::solve`]: Solves linear systems using this decomposition
/// - [`LU::determinant`]: Computes the determinant using this decomposition
/// - [`LU::try_inverse`]: Computes the matrix inverse using this decomposition
#[cfg_attr(feature = "serde-serialize-no-std", derive(Serialize, Deserialize))]
#[cfg_attr(
    feature = "serde-serialize-no-std",
    serde(bound(serialize = "DefaultAllocator: Allocator<R, C> +
                           Allocator<DimMinimum<R, C>>,
         OMatrix<T, R, C>: Serialize,
         PermutationSequence<DimMinimum<R, C>>: Serialize"))
)]
#[cfg_attr(
    feature = "serde-serialize-no-std",
    serde(bound(deserialize = "DefaultAllocator: Allocator<R, C> +
                           Allocator<DimMinimum<R, C>>,
         OMatrix<T, R, C>: Deserialize<'de>,
         PermutationSequence<DimMinimum<R, C>>: Deserialize<'de>"))
)]
#[cfg_attr(feature = "defmt", derive(defmt::Format))]
#[derive(Clone, Debug)]
pub struct LU<T: ComplexField, R: DimMin<C>, C: Dim>
where
    DefaultAllocator: Allocator<R, C> + Allocator<DimMinimum<R, C>>,
{
    lu: OMatrix<T, R, C>,
    p: PermutationSequence<DimMinimum<R, C>>,
}

impl<T: ComplexField, R: DimMin<C>, C: Dim> Copy for LU<T, R, C>
where
    DefaultAllocator: Allocator<R, C> + Allocator<DimMinimum<R, C>>,
    OMatrix<T, R, C>: Copy,
    PermutationSequence<DimMinimum<R, C>>: Copy,
{
}

/// Performs an LU decomposition to compute the inverse of a matrix, writing the result to `out`.
///
/// This function computes the inverse of the given `matrix` using LU decomposition and stores
/// the result in the provided `out` matrix. This is useful when you want to reuse an existing
/// matrix buffer to avoid allocations.
///
/// # Arguments
///
/// * `matrix` - The square matrix to invert (will be consumed by the operation)
/// * `out` - A mutable reference to a matrix where the inverse will be stored
///
/// # Returns
///
/// Returns `true` if the matrix is invertible and the inverse was successfully computed.
/// Returns `false` if the matrix is singular (not invertible), in which case `out` may
/// contain invalid data and should not be used.
///
/// # Panics
///
/// Panics if the input matrix is not square.
///
/// # Example
///
/// ```
/// use nalgebra::Matrix3;
///
/// let m = Matrix3::new(
///     2.0, 1.0, 0.0,
///     1.0, 2.0, 1.0,
///     0.0, 1.0, 2.0,
/// );
///
/// let mut out = Matrix3::zeros();
/// let success = nalgebra::linalg::try_invert_to(m, &mut out);
///
/// assert!(success);
///
/// // Verify that m * out is approximately the identity matrix
/// let m_original = Matrix3::new(
///     2.0, 1.0, 0.0,
///     1.0, 2.0, 1.0,
///     0.0, 1.0, 2.0,
/// );
/// let identity = Matrix3::identity();
/// assert!((m_original * out - identity).norm() < 1e-10);
/// ```
///
/// # Example: Non-invertible Matrix
///
/// ```
/// use nalgebra::Matrix2;
///
/// // Create a singular (non-invertible) matrix
/// let singular = Matrix2::new(
///     1.0, 2.0,
///     2.0, 4.0,  // Second row is a multiple of the first
/// );
///
/// let mut out = Matrix2::zeros();
/// let success = nalgebra::linalg::try_invert_to(singular, &mut out);
///
/// assert!(!success);  // Matrix is not invertible
/// ```
///
/// # See Also
///
/// - [`LU::try_inverse`]: Computes the inverse using an existing LU decomposition
/// - [`LU::try_inverse_to`]: Similar method on the LU struct
/// - [`Matrix::try_inverse`]: Direct matrix inversion method
pub fn try_invert_to<T: ComplexField, D: Dim, S>(
    mut matrix: OMatrix<T, D, D>,
    out: &mut Matrix<T, D, D, S>,
) -> bool
where
    S: StorageMut<T, D, D>,
    DefaultAllocator: Allocator<D, D>,
{
    assert!(
        matrix.is_square(),
        "LU inversion: unable to invert a rectangular matrix."
    );
    let dim = matrix.nrows();

    out.fill_with_identity();

    for i in 0..dim {
        let piv = matrix.view_range(i.., i).icamax() + i;
        let diag = matrix[(piv, i)].clone();

        if diag.is_zero() {
            return false;
        }

        if piv != i {
            out.swap_rows(i, piv);
            matrix.columns_range_mut(..i).swap_rows(i, piv);
            gauss_step_swap(&mut matrix, diag, i, piv);
        } else {
            gauss_step(&mut matrix, diag, i);
        }
    }

    let _ = matrix.solve_lower_triangular_with_diag_mut(out, T::one());
    matrix.solve_upper_triangular_mut(out)
}

impl<T: ComplexField, R: DimMin<C>, C: Dim> LU<T, R, C>
where
    DefaultAllocator: Allocator<R, C> + Allocator<DimMinimum<R, C>>,
{
    /// Computes the LU decomposition with partial (row) pivoting of `matrix`.
    ///
    /// This is the primary constructor for creating an LU decomposition. It performs the
    /// factorization `P * A = L * U`, where:
    /// - `P` is a permutation matrix (represented as a permutation sequence)
    /// - `L` is a lower triangular matrix with ones on the diagonal
    /// - `U` is an upper triangular matrix
    /// - `A` is the original matrix
    ///
    /// # How It Works
    ///
    /// The algorithm uses Gaussian elimination with partial pivoting:
    /// 1. For each column, find the row with the largest absolute value (pivoting)
    /// 2. Swap rows if necessary to bring this element to the diagonal
    /// 3. Eliminate entries below the diagonal using row operations
    /// 4. Store the elimination factors in the lower triangle
    ///
    /// Partial pivoting improves numerical stability by avoiding division by small numbers.
    ///
    /// # Arguments
    ///
    /// * `matrix` - The matrix to decompose (can be rectangular, though square is most common)
    ///
    /// # Example: Basic Usage
    ///
    /// ```
    /// use nalgebra::{Matrix3, LU};
    ///
    /// let m = Matrix3::new(
    ///     1.0, 2.0, 3.0,
    ///     4.0, 5.0, 6.0,
    ///     7.0, 8.0, 9.0,
    /// );
    ///
    /// let lu = LU::new(m);
    ///
    /// // The decomposition stores both L and U in a compact form
    /// let l = lu.l();
    /// let u = lu.u();
    /// let p = lu.p();
    ///
    /// // Verify: P * A = L * U
    /// let mut m_copy = m.clone();
    /// p.permute_rows(&mut m_copy);
    /// assert!((m_copy - l * u).norm() < 1e-10);
    /// ```
    ///
    /// # Example: Solving a Linear System
    ///
    /// ```
    /// use nalgebra::{Matrix3, Vector3};
    ///
    /// // Solve the system: A * x = b
    /// let a = Matrix3::new(
    ///     2.0, 1.0, 1.0,
    ///     4.0, 3.0, 3.0,
    ///     8.0, 7.0, 9.0,
    /// );
    ///
    /// let b = Vector3::new(4.0, 10.0, 24.0);
    ///
    /// let lu = a.lu();
    /// let x = lu.solve(&b).expect("Linear system has no solution");
    ///
    /// // Verify the solution
    /// assert!((a * x - b).norm() < 1e-10);
    /// ```
    ///
    /// # Example: Rectangular Matrix
    ///
    /// ```
    /// use nalgebra::{Matrix3x2, LU};
    ///
    /// // LU decomposition works on rectangular matrices too
    /// let m = Matrix3x2::new(
    ///     1.0, 2.0,
    ///     3.0, 4.0,
    ///     5.0, 6.0,
    /// );
    ///
    /// let lu = LU::new(m);
    /// let l = lu.l();
    /// let u = lu.u();
    ///
    /// // L is 3x2 and U is 2x2
    /// assert_eq!(l.shape(), (3, 2));
    /// assert_eq!(u.shape(), (2, 2));
    /// ```
    ///
    /// # See Also
    ///
    /// - [`LU::solve`]: Solve linear systems using this decomposition
    /// - [`LU::determinant`]: Compute determinants efficiently
    /// - [`LU::try_inverse`]: Compute matrix inverses
    /// - [`LU::unpack`]: Extract P, L, and U components
    pub fn new(mut matrix: OMatrix<T, R, C>) -> Self {
        let (nrows, ncols) = matrix.shape_generic();
        let min_nrows_ncols = nrows.min(ncols);

        let mut p = PermutationSequence::identity_generic(min_nrows_ncols);

        if min_nrows_ncols.value() == 0 {
            return LU { lu: matrix, p };
        }

        for i in 0..min_nrows_ncols.value() {
            let piv = matrix.view_range(i.., i).icamax() + i;
            let diag = matrix[(piv, i)].clone();

            if diag.is_zero() {
                // No non-zero entries on this column.
                continue;
            }

            if piv != i {
                p.append_permutation(i, piv);
                matrix.columns_range_mut(..i).swap_rows(i, piv);
                gauss_step_swap(&mut matrix, diag, i, piv);
            } else {
                gauss_step(&mut matrix, diag, i);
            }
        }

        LU { lu: matrix, p }
    }

    #[doc(hidden)]
    pub const fn lu_internal(&self) -> &OMatrix<T, R, C> {
        &self.lu
    }

    /// Extracts the lower triangular matrix `L` from this LU decomposition.
    ///
    /// The lower triangular matrix `L` has the following properties:
    /// - All entries above the main diagonal are zero
    /// - All diagonal entries are one
    /// - The entries below the diagonal contain the elimination multipliers from Gaussian elimination
    ///
    /// This matrix, combined with `U`, satisfies: `P * A = L * U`
    ///
    /// # Returns
    ///
    /// A lower triangular matrix with ones on the diagonal.
    ///
    /// # Example
    ///
    /// ```
    /// use nalgebra::Matrix3;
    ///
    /// let m = Matrix3::new(
    ///     1.0, 2.0, 3.0,
    ///     4.0, 5.0, 6.0,
    ///     7.0, 8.0, 10.0,  // Changed last element to make it non-singular
    /// );
    ///
    /// let lu = m.lu();
    /// let l = lu.l();
    ///
    /// // L is lower triangular with ones on the diagonal
    /// assert_eq!(l[(0, 0)], 1.0);
    /// assert_eq!(l[(1, 1)], 1.0);
    /// assert_eq!(l[(2, 2)], 1.0);
    ///
    /// // Upper triangle is zero
    /// assert_eq!(l[(0, 1)], 0.0);
    /// assert_eq!(l[(0, 2)], 0.0);
    /// assert_eq!(l[(1, 2)], 0.0);
    /// ```
    ///
    /// # Performance Note
    ///
    /// This method allocates a new matrix and copies data from the internal storage.
    /// If you need both `L` and `U`, consider using [`LU::unpack`] instead, which can be
    /// more efficient.
    ///
    /// # See Also
    ///
    /// - [`LU::u`]: Extracts the upper triangular matrix
    /// - [`LU::p`]: Gets the permutation sequence
    /// - [`LU::unpack`]: Extracts all components (P, L, U) at once
    /// - [`LU::l_unpack`]: More efficient version that consumes self
    #[inline]
    #[must_use]
    pub fn l(&self) -> OMatrix<T, R, DimMinimum<R, C>>
    where
        DefaultAllocator: Allocator<R, DimMinimum<R, C>>,
    {
        let (nrows, ncols) = self.lu.shape_generic();
        let mut m = self.lu.columns_generic(0, nrows.min(ncols)).into_owned();
        m.fill_upper_triangle(T::zero(), 1);
        m.fill_diagonal(T::one());
        m
    }

    /// The lower triangular matrix of this decomposition.
    fn l_unpack_with_p(
        self,
    ) -> (
        OMatrix<T, R, DimMinimum<R, C>>,
        PermutationSequence<DimMinimum<R, C>>,
    )
    where
        DefaultAllocator: Reallocator<T, R, C, R, DimMinimum<R, C>>,
    {
        let (nrows, ncols) = self.lu.shape_generic();
        let mut m = self.lu.resize_generic(nrows, nrows.min(ncols), T::zero());
        m.fill_upper_triangle(T::zero(), 1);
        m.fill_diagonal(T::one());
        (m, self.p)
    }

    /// Extracts the lower triangular matrix `L`, consuming the LU decomposition.
    ///
    /// This is a more efficient version of [`LU::l`] that consumes the LU decomposition
    /// and reuses its internal storage when possible, avoiding unnecessary allocations.
    ///
    /// Use this method when you no longer need the LU decomposition after extracting L.
    ///
    /// # Returns
    ///
    /// A lower triangular matrix with ones on the diagonal.
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
    /// let lu = m.lu();
    ///
    /// // Consume the LU decomposition to get L more efficiently
    /// let l = lu.l_unpack();
    ///
    /// // Verify L is lower triangular with ones on diagonal
    /// assert_eq!(l[(0, 0)], 1.0);
    /// assert_eq!(l[(1, 1)], 1.0);
    /// assert_eq!(l[(2, 2)], 1.0);
    /// ```
    ///
    /// # See Also
    ///
    /// - [`LU::l`]: Non-consuming version that returns a copy
    /// - [`LU::unpack`]: Extracts P, L, and U all at once
    #[inline]
    pub fn l_unpack(self) -> OMatrix<T, R, DimMinimum<R, C>>
    where
        DefaultAllocator: Reallocator<T, R, C, R, DimMinimum<R, C>>,
    {
        let (nrows, ncols) = self.lu.shape_generic();
        let mut m = self.lu.resize_generic(nrows, nrows.min(ncols), T::zero());
        m.fill_upper_triangle(T::zero(), 1);
        m.fill_diagonal(T::one());
        m
    }

    /// Extracts the upper triangular matrix `U` from this LU decomposition.
    ///
    /// The upper triangular matrix `U` has the following properties:
    /// - All entries below the main diagonal are zero
    /// - The diagonal and upper entries contain the result of Gaussian elimination
    ///
    /// This matrix, combined with `L`, satisfies: `P * A = L * U`
    ///
    /// # Returns
    ///
    /// An upper triangular matrix.
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
    /// let u = lu.u();
    ///
    /// // U is upper triangular - all entries below diagonal are zero
    /// assert_eq!(u[(1, 0)], 0.0);
    /// assert_eq!(u[(2, 0)], 0.0);
    /// assert_eq!(u[(2, 1)], 0.0);
    ///
    /// // Upper triangle and diagonal have non-zero values
    /// let diag: f64 = u[(0, 0)];
    /// assert!(diag.abs() > 1e-10);
    /// ```
    ///
    /// # Example: Verifying the Decomposition
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
    /// let lu = m.lu();
    /// let l = lu.l();
    /// let u = lu.u();
    /// let p = lu.p();
    ///
    /// // Verify that P * A = L * U
    /// let mut m_copy = m.clone();
    /// p.permute_rows(&mut m_copy);
    /// let l_times_u = l * u;
    /// assert!((m_copy - l_times_u).norm() < 1e-10);
    /// ```
    ///
    /// # Performance Note
    ///
    /// This method allocates a new matrix and copies data from the internal storage.
    /// If you need both `L` and `U`, consider using [`LU::unpack`] instead.
    ///
    /// # See Also
    ///
    /// - [`LU::l`]: Extracts the lower triangular matrix
    /// - [`LU::p`]: Gets the permutation sequence
    /// - [`LU::unpack`]: Extracts all components (P, L, U) at once
    #[inline]
    #[must_use]
    pub fn u(&self) -> OMatrix<T, DimMinimum<R, C>, C>
    where
        DefaultAllocator: Allocator<DimMinimum<R, C>, C>,
    {
        let (nrows, ncols) = self.lu.shape_generic();
        self.lu.rows_generic(0, nrows.min(ncols)).upper_triangle()
    }

    /// Returns a reference to the row permutation sequence `P` of this decomposition.
    ///
    /// The permutation sequence represents the row swaps performed during LU decomposition
    /// with partial pivoting. It can be used to reconstruct the permutation matrix P that
    /// satisfies: `P * A = L * U`
    ///
    /// # What is a Permutation Sequence?
    ///
    /// A permutation sequence is a compact representation of a permutation matrix. Instead of
    /// storing a full matrix (which would be mostly zeros), it stores only the sequence of
    /// row swaps needed to apply the permutation.
    ///
    /// # Returns
    ///
    /// A reference to the permutation sequence.
    ///
    /// # Example
    ///
    /// ```
    /// use nalgebra::Matrix3;
    ///
    /// let m = Matrix3::new(
    ///     0.0, 1.0, 2.0,
    ///     3.0, 4.0, 5.0,
    ///     6.0, 7.0, 8.0,
    /// );
    ///
    /// let lu = m.lu();
    /// let p = lu.p();
    ///
    /// // Apply the permutation to the original matrix
    /// let mut permuted = m.clone();
    /// p.permute_rows(&mut permuted);
    ///
    /// // The permuted matrix can be reconstructed as L * U
    /// let l = lu.l();
    /// let u = lu.u();
    /// assert!((permuted - l * u).norm() < 1e-10);
    /// ```
    ///
    /// # Example: Determinant of Permutation
    ///
    /// ```
    /// use nalgebra::Matrix3;
    ///
    /// let m = Matrix3::new(
    ///     1.0, 0.0, 0.0,
    ///     0.0, 1.0, 0.0,
    ///     0.0, 0.0, 1.0,
    /// );
    ///
    /// let lu = m.lu();
    ///
    /// // For the identity matrix, no pivoting is needed
    /// // The permutation determinant should be 1
    /// assert_eq!(lu.p().determinant::<i32>(), 1);
    /// ```
    ///
    /// # See Also
    ///
    /// - [`LU::l`]: Extracts the lower triangular matrix
    /// - [`LU::u`]: Extracts the upper triangular matrix
    /// - [`LU::unpack`]: Extracts all components (P, L, U) at once
    /// - [`PermutationSequence::permute`]: Apply the permutation to a matrix
    #[inline]
    #[must_use]
    pub const fn p(&self) -> &PermutationSequence<DimMinimum<R, C>> {
        &self.p
    }

    /// Extracts all components of the LU decomposition: `(P, L, U)`.
    ///
    /// This method consumes the LU decomposition and returns all three components:
    /// - `P`: The permutation sequence representing row swaps
    /// - `L`: The lower triangular matrix with ones on the diagonal
    /// - `U`: The upper triangular matrix
    ///
    /// These components satisfy the equation: `P * A = L * U`
    ///
    /// # Returns
    ///
    /// A tuple `(P, L, U)` containing the permutation sequence and both triangular matrices.
    ///
    /// # Performance Note
    ///
    /// This method is more efficient than calling `p()`, `l()`, and `u()` separately,
    /// as it can reuse internal storage and avoid redundant allocations.
    ///
    /// # Example: Basic Usage
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
    /// let lu = m.lu();
    /// let (p, l, u) = lu.unpack();
    ///
    /// // Verify the decomposition: P * A = L * U
    /// let mut m_copy = m.clone();
    /// p.permute_rows(&mut m_copy);
    /// let reconstructed = l * u;
    /// assert!((m_copy - reconstructed).norm() < 1e-10);
    /// ```
    ///
    /// # Example: Understanding the Components
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
    /// let (p, l, u) = lu.unpack();
    ///
    /// // L is lower triangular with ones on diagonal
    /// assert_eq!(l[(0, 0)], 1.0);
    /// assert_eq!(l[(1, 1)], 1.0);
    /// assert_eq!(l[(2, 2)], 1.0);
    /// assert_eq!(l[(0, 1)], 0.0);
    /// assert_eq!(l[(0, 2)], 0.0);
    ///
    /// // U is upper triangular
    /// assert_eq!(u[(1, 0)], 0.0);
    /// assert_eq!(u[(2, 0)], 0.0);
    /// assert_eq!(u[(2, 1)], 0.0);
    /// ```
    ///
    /// # Example: Solving a System Manually
    ///
    /// ```
    /// use nalgebra::{Matrix3, Vector3};
    ///
    /// let a = Matrix3::new(
    ///     2.0, 1.0, 1.0,
    ///     4.0, 3.0, 3.0,
    ///     8.0, 7.0, 9.0,
    /// );
    /// let b = Vector3::new(4.0, 10.0, 24.0);
    ///
    /// let lu = a.lu();
    /// let (p, l, u) = lu.unpack();
    ///
    /// // To solve A*x = b, we solve:
    /// // 1. P*A*x = P*b  =>  L*U*x = P*b
    /// // 2. Let y = U*x, solve L*y = P*b for y
    /// // 3. Then solve U*x = y for x
    ///
    /// let mut pb = b.clone();
    /// p.permute_rows(&mut pb);
    /// let y = l.solve_lower_triangular(&pb).unwrap();
    /// let x = u.solve_upper_triangular(&y).unwrap();
    ///
    /// // Verify the solution
    /// assert!((a * x - b).norm() < 1e-10);
    /// ```
    ///
    /// # See Also
    ///
    /// - [`LU::l`]: Extract only the lower triangular matrix
    /// - [`LU::u`]: Extract only the upper triangular matrix
    /// - [`LU::p`]: Get only the permutation sequence
    /// - [`LU::l_unpack`]: Extract only L, consuming self
    #[inline]
    pub fn unpack(
        self,
    ) -> (
        PermutationSequence<DimMinimum<R, C>>,
        OMatrix<T, R, DimMinimum<R, C>>,
        OMatrix<T, DimMinimum<R, C>, C>,
    )
    where
        DefaultAllocator: Allocator<R, DimMinimum<R, C>>
            + Allocator<DimMinimum<R, C>, C>
            + Reallocator<T, R, C, R, DimMinimum<R, C>>,
    {
        // Use reallocation for either l or u.
        let u = self.u();
        let (l, p) = self.l_unpack_with_p();

        (p, l, u)
    }
}

impl<T: ComplexField, D: DimMin<D, Output = D>> LU<T, D, D>
where
    DefaultAllocator: Allocator<D, D> + Allocator<D>,
{
    /// Solves the linear system `A * x = b` using this LU decomposition, where `x` is the unknown.
    ///
    /// This method efficiently solves systems of linear equations by leveraging the precomputed
    /// LU decomposition. Given a matrix `A` (which was decomposed to create this `LU` object)
    /// and a right-hand side `b`, it finds the solution vector `x` such that `A * x = b`.
    ///
    /// # How It Works
    ///
    /// The solution process involves two triangular system solves:
    /// 1. Forward substitution: Solve `L * y = P * b` for `y`
    /// 2. Back substitution: Solve `U * x = y` for `x`
    ///
    /// This is much faster than computing `A^(-1) * b`, especially for large systems.
    ///
    /// # Arguments
    ///
    /// * `b` - The right-hand side of the equation `A * x = b`. Can be a vector or matrix.
    ///
    /// # Returns
    ///
    /// * `Some(x)` - The solution to the system if `A` is invertible
    /// * `None` - If the matrix is singular (not invertible) and no unique solution exists
    ///
    /// # Example: Solving a Linear System
    ///
    /// ```
    /// use nalgebra::{Matrix3, Vector3};
    ///
    /// // Define the system: A * x = b
    /// let a = Matrix3::new(
    ///     2.0, 1.0, 1.0,
    ///     4.0, 3.0, 3.0,
    ///     8.0, 7.0, 9.0,
    /// );
    /// let b = Vector3::new(4.0, 10.0, 24.0);
    ///
    /// // Decompose and solve
    /// let lu = a.lu();
    /// let x = lu.solve(&b).expect("Linear system has no solution");
    ///
    /// // Verify: A * x = b
    /// assert!((a * x - b).norm() < 1e-10);
    /// ```
    ///
    /// # Example: Multiple Right-Hand Sides
    ///
    /// ```
    /// use nalgebra::{Matrix3, Matrix3x2};
    ///
    /// let a = Matrix3::new(
    ///     3.0, 1.0, 0.0,
    ///     1.0, 3.0, 1.0,
    ///     0.0, 1.0, 3.0,
    /// );
    ///
    /// // Solve for two right-hand sides at once
    /// let b = Matrix3x2::new(
    ///     1.0, 0.0,
    ///     2.0, 1.0,
    ///     3.0, 2.0,
    /// );
    ///
    /// let lu = a.lu();
    /// let x = lu.solve(&b).expect("Linear system has no solution");
    ///
    /// // Verify both solutions
    /// assert!((a * x - b).norm() < 1e-10);
    /// ```
    ///
    /// # Example: Singular Matrix
    ///
    /// ```
    /// use nalgebra::{Matrix2, Vector2};
    ///
    /// // Create a singular matrix (second row is twice the first)
    /// let a = Matrix2::new(
    ///     1.0, 2.0,
    ///     2.0, 4.0,
    /// );
    /// let b = Vector2::new(1.0, 2.0);
    ///
    /// let lu = a.lu();
    /// let result = lu.solve(&b);
    ///
    /// // System has no unique solution
    /// assert!(result.is_none());
    /// ```
    ///
    /// # Performance
    ///
    /// This method allocates a new matrix/vector for the result. If you want to reuse
    /// an existing buffer, use [`LU::solve_mut`] instead.
    ///
    /// # See Also
    ///
    /// - [`LU::solve_mut`]: In-place version that modifies `b` directly
    /// - [`LU::try_inverse`]: Computes the matrix inverse
    /// - [`Matrix::lu`]: Creates an LU decomposition
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

    /// Solves the linear system `A * x = b` in-place, overwriting `b` with the solution `x`.
    ///
    /// This is the in-place version of [`LU::solve`]. It modifies the input `b` directly,
    /// replacing it with the solution `x`. This is more memory-efficient than `solve()` when
    /// you don't need to preserve the original `b` vector.
    ///
    /// # Arguments
    ///
    /// * `b` - The right-hand side vector/matrix, which will be overwritten with the solution
    ///
    /// # Returns
    ///
    /// * `true` - If the system was solved successfully (matrix is invertible)
    /// * `false` - If the matrix is singular (not invertible). In this case, `b` contains
    ///   invalid data and should not be used.
    ///
    /// # Panics
    ///
    /// Panics if the matrix is not square or if dimensions don't match.
    ///
    /// # Example: Basic Usage
    ///
    /// ```
    /// use nalgebra::{Matrix3, Vector3};
    ///
    /// let a = Matrix3::new(
    ///     2.0, 1.0, 1.0,
    ///     4.0, 3.0, 3.0,
    ///     8.0, 7.0, 9.0,
    /// );
    ///
    /// let mut b = Vector3::new(4.0, 10.0, 24.0);
    /// let original_b = b.clone();
    ///
    /// let lu = a.lu();
    /// let success = lu.solve_mut(&mut b);
    ///
    /// assert!(success);
    /// // b now contains the solution x
    /// assert!((a * b - original_b).norm() < 1e-10);
    /// ```
    ///
    /// # Example: Multiple Systems with Same Coefficient Matrix
    ///
    /// ```
    /// use nalgebra::{Matrix3, Vector3};
    ///
    /// let a = Matrix3::new(
    ///     3.0, 1.0, 0.0,
    ///     1.0, 3.0, 1.0,
    ///     0.0, 1.0, 3.0,
    /// );
    ///
    /// // Decompose once
    /// let lu = a.lu();
    ///
    /// // Solve multiple systems efficiently by reusing the decomposition
    /// let mut b1 = Vector3::new(1.0, 2.0, 3.0);
    /// let mut b2 = Vector3::new(0.0, 1.0, 2.0);
    /// let mut b3 = Vector3::new(1.0, 1.0, 1.0);
    ///
    /// lu.solve_mut(&mut b1);
    /// lu.solve_mut(&mut b2);
    /// lu.solve_mut(&mut b3);
    ///
    /// // All three systems are now solved
    /// ```
    ///
    /// # Example: Checking for Singular Matrix
    ///
    /// ```
    /// use nalgebra::{Matrix2, Vector2};
    ///
    /// let singular = Matrix2::new(
    ///     1.0, 2.0,
    ///     2.0, 4.0,  // Linearly dependent rows
    /// );
    ///
    /// let mut b = Vector2::new(1.0, 2.0);
    /// let lu = singular.lu();
    /// let success = lu.solve_mut(&mut b);
    ///
    /// assert!(!success);  // Cannot solve with singular matrix
    /// // Don't use b here - it contains invalid data
    /// ```
    ///
    /// # Performance
    ///
    /// This method is faster than [`LU::solve`] because it doesn't allocate new memory.
    /// When solving multiple systems with the same coefficient matrix, decompose once
    /// and call `solve_mut` multiple times for maximum efficiency.
    ///
    /// # See Also
    ///
    /// - [`LU::solve`]: Allocating version that returns a new vector
    /// - [`LU::is_invertible`]: Check if the matrix is invertible before solving
    pub fn solve_mut<R2: Dim, C2: Dim, S2>(&self, b: &mut Matrix<T, R2, C2, S2>) -> bool
    where
        S2: StorageMut<T, R2, C2>,
        ShapeConstraint: SameNumberOfRows<R2, D>,
    {
        assert_eq!(
            self.lu.nrows(),
            b.nrows(),
            "LU solve matrix dimension mismatch."
        );
        assert!(
            self.lu.is_square(),
            "LU solve: unable to solve a non-square system."
        );

        self.p.permute_rows(b);
        let _ = self.lu.solve_lower_triangular_with_diag_mut(b, T::one());
        self.lu.solve_upper_triangular_mut(b)
    }

    /// Computes the inverse of the decomposed matrix using the LU decomposition.
    ///
    /// Matrix inversion finds a matrix `A^(-1)` such that `A * A^(-1) = I`, where `I` is
    /// the identity matrix. The LU decomposition provides an efficient way to compute this.
    ///
    /// # Returns
    ///
    /// * `Some(A_inv)` - The inverse matrix if it exists
    /// * `None` - If the matrix is singular (determinant is zero) and has no inverse
    ///
    /// # When to Use Matrix Inversion
    ///
    /// **Important:** For solving linear systems `A * x = b`, use [`LU::solve`] instead of
    /// computing `A^(-1) * b`. It's faster and more numerically stable.
    ///
    /// Matrix inversion is useful when you need to:
    /// - Solve many systems with different right-hand sides stored as columns
    /// - Apply the same transformation multiple times
    /// - Compute expressions involving the inverse matrix
    ///
    /// # Panics
    ///
    /// Panics if the decomposed matrix is not square.
    ///
    /// # Example: Basic Usage
    ///
    /// ```
    /// use nalgebra::Matrix3;
    ///
    /// let m = Matrix3::new(
    ///     2.0, 1.0, 0.0,
    ///     1.0, 2.0, 1.0,
    ///     0.0, 1.0, 2.0,
    /// );
    ///
    /// let lu = m.lu();
    /// let m_inv = lu.try_inverse().expect("Matrix is not invertible");
    ///
    /// // Verify that m * m_inv = I
    /// let identity = Matrix3::identity();
    /// assert!((m * m_inv - identity).norm() < 1e-10);
    /// ```
    ///
    /// # Example: Singular Matrix
    ///
    /// ```
    /// use nalgebra::Matrix2;
    ///
    /// // Create a singular matrix (rows are linearly dependent)
    /// let singular = Matrix2::new(
    ///     1.0, 2.0,
    ///     2.0, 4.0,
    /// );
    ///
    /// let lu = singular.lu();
    /// let result = lu.try_inverse();
    ///
    /// assert!(result.is_none());  // No inverse exists
    /// ```
    ///
    /// # Example: Prefer solve() for Linear Systems
    ///
    /// ```
    /// use nalgebra::{Matrix3, Vector3};
    ///
    /// let a = Matrix3::new(
    ///     2.0, 1.0, 1.0,
    ///     4.0, 3.0, 3.0,
    ///     8.0, 7.0, 9.0,
    /// );
    /// let b = Vector3::new(4.0, 10.0, 24.0);
    ///
    /// let lu = a.lu();
    ///
    /// // GOOD: Use solve() - faster and more accurate
    /// let x_good = lu.solve(&b).unwrap();
    ///
    /// // AVOID: Computing inverse then multiplying - slower and less accurate
    /// let x_bad = lu.try_inverse().unwrap() * b;
    ///
    /// // Both give the same result, but solve() is better
    /// assert!((x_good - x_bad).norm() < 1e-10);
    /// ```
    ///
    /// # Performance
    ///
    /// This method allocates a new matrix. If you want to reuse an existing buffer,
    /// use [`LU::try_inverse_to`] instead.
    ///
    /// # See Also
    ///
    /// - [`LU::try_inverse_to`]: In-place version that writes to an existing matrix
    /// - [`LU::solve`]: Better choice for solving linear systems
    /// - [`LU::is_invertible`]: Check if a matrix is invertible without computing the inverse
    /// - [`Matrix::try_inverse`]: Direct matrix inversion without explicit LU decomposition
    #[must_use]
    pub fn try_inverse(&self) -> Option<OMatrix<T, D, D>> {
        assert!(
            self.lu.is_square(),
            "LU inverse: unable to compute the inverse of a non-square matrix."
        );

        let (nrows, ncols) = self.lu.shape_generic();
        let mut res = OMatrix::identity_generic(nrows, ncols);
        if self.try_inverse_to(&mut res) {
            Some(res)
        } else {
            None
        }
    }

    /// Computes the inverse of the decomposed matrix, writing the result to `out`.
    ///
    /// This is the in-place version of [`LU::try_inverse`]. It writes the computed inverse
    /// directly into the provided matrix `out`, which is useful for reusing existing buffers
    /// and avoiding allocations.
    ///
    /// # Arguments
    ///
    /// * `out` - A mutable reference to a matrix where the inverse will be written
    ///
    /// # Returns
    ///
    /// * `true` - If the inverse was successfully computed and written to `out`
    /// * `false` - If the matrix is singular (not invertible). In this case, `out` may contain
    ///   invalid data and should not be used.
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - The decomposed matrix is not square
    /// - The output matrix `out` has different dimensions than the decomposed matrix
    ///
    /// # Example: Basic Usage
    ///
    /// ```
    /// use nalgebra::Matrix3;
    ///
    /// let m = Matrix3::new(
    ///     2.0, 1.0, 0.0,
    ///     1.0, 2.0, 1.0,
    ///     0.0, 1.0, 2.0,
    /// );
    ///
    /// let lu = m.lu();
    /// let mut out = Matrix3::zeros();
    /// let success = lu.try_inverse_to(&mut out);
    ///
    /// assert!(success);
    ///
    /// // Verify the inverse
    /// let identity = Matrix3::identity();
    /// assert!((m * out - identity).norm() < 1e-10);
    /// ```
    ///
    /// # Example: Reusing Buffer
    ///
    /// ```
    /// use nalgebra::Matrix3;
    ///
    /// let m1 = Matrix3::new(
    ///     2.0, 1.0, 0.0,
    ///     1.0, 2.0, 1.0,
    ///     0.0, 1.0, 2.0,
    /// );
    ///
    /// let m2 = Matrix3::new(
    ///     3.0, 0.0, 0.0,
    ///     0.0, 3.0, 0.0,
    ///     0.0, 0.0, 3.0,
    /// );
    ///
    /// // Reuse the same output buffer for multiple inversions
    /// let mut out = Matrix3::zeros();
    ///
    /// m1.lu().try_inverse_to(&mut out);
    /// // Use out...
    ///
    /// m2.lu().try_inverse_to(&mut out);
    /// // Use out again...
    /// ```
    ///
    /// # Example: Handling Singular Matrices
    ///
    /// ```
    /// use nalgebra::Matrix2;
    ///
    /// let singular = Matrix2::new(
    ///     1.0, 2.0,
    ///     2.0, 4.0,
    /// );
    ///
    /// let lu = singular.lu();
    /// let mut out = Matrix2::zeros();
    /// let success = lu.try_inverse_to(&mut out);
    ///
    /// assert!(!success);
    /// // Don't use out - it contains invalid data
    /// ```
    ///
    /// # Performance
    ///
    /// This method avoids allocating a new matrix, making it more efficient than
    /// [`LU::try_inverse`] when you already have a buffer available or need to compute
    /// many inverses.
    ///
    /// # See Also
    ///
    /// - [`LU::try_inverse`]: Allocating version that returns a new matrix
    /// - [`LU::solve_mut`]: In-place linear system solver
    /// - [`try_invert_to`]: Free function that combines decomposition and inversion
    pub fn try_inverse_to<S2: StorageMut<T, D, D>>(&self, out: &mut Matrix<T, D, D, S2>) -> bool {
        assert!(
            self.lu.is_square(),
            "LU inverse: unable to compute the inverse of a non-square matrix."
        );
        assert!(
            self.lu.shape() == out.shape(),
            "LU inverse: mismatched output shape."
        );

        out.fill_with_identity();
        self.solve_mut(out)
    }

    /// Computes the determinant of the decomposed matrix using the LU decomposition.
    ///
    /// The determinant is a scalar value that provides important information about a matrix:
    /// - If `det(A) = 0`, the matrix is singular (not invertible)
    /// - If `det(A) ≠ 0`, the matrix is invertible
    /// - The absolute value indicates how much the matrix scales volumes
    ///
    /// # How It Works
    ///
    /// For an LU decomposition where `P * A = L * U`:
    /// - `det(A) = det(L) * det(U) * det(P)`
    /// - `det(L) = 1` (unit diagonal)
    /// - `det(U)` = product of diagonal elements
    /// - `det(P)` = ±1 (sign depends on number of row swaps)
    ///
    /// This makes computing the determinant very efficient using LU decomposition.
    ///
    /// # Panics
    ///
    /// Panics if the decomposed matrix is not square.
    ///
    /// # Example: Basic Usage
    ///
    /// ```
    /// use nalgebra::Matrix3;
    ///
    /// let m = Matrix3::new(
    ///     1.0, 2.0, 3.0,
    ///     0.0, 1.0, 4.0,
    ///     5.0, 6.0, 0.0,
    /// );
    ///
    /// let lu = m.lu();
    /// let det = lu.determinant();
    ///
    /// println!("Determinant: {}", det);
    /// ```
    ///
    /// # Example: Checking Invertibility
    ///
    /// ```
    /// use nalgebra::Matrix3;
    ///
    /// let invertible = Matrix3::new(
    ///     2.0, 1.0, 1.0,
    ///     4.0, 3.0, 3.0,
    ///     8.0, 7.0, 9.0,
    /// );
    ///
    /// let singular = Matrix3::new(
    ///     1.0, 2.0, 3.0,
    ///     2.0, 4.0, 6.0,  // Second row is twice the first
    ///     4.0, 5.0, 6.0,
    /// );
    ///
    /// let det1: f64 = invertible.lu().determinant();
    /// let det2: f64 = singular.lu().determinant();
    ///
    /// assert!(det1.abs() > 1e-10);  // Non-zero determinant
    /// assert!(det2.abs() < 1e-10);  // Zero determinant (within numerical tolerance)
    /// ```
    ///
    /// # Example: Computing Volume Scale Factor
    ///
    /// ```
    /// use nalgebra::Matrix2;
    ///
    /// // A transformation matrix that scales by 2 in x and 3 in y
    /// let scale = Matrix2::new(
    ///     2.0, 0.0,
    ///     0.0, 3.0,
    /// );
    ///
    /// let det: f64 = scale.lu().determinant();
    ///
    /// // The determinant tells us the area scaling factor
    /// assert!((det - 6.0).abs() < 1e-10);  // Areas are scaled by 6
    /// ```
    ///
    /// # Example: Sign of Determinant
    ///
    /// ```
    /// use nalgebra::Matrix2;
    ///
    /// // A matrix that represents a reflection (flips orientation)
    /// let reflection = Matrix2::new(
    ///     1.0,  0.0,
    ///     0.0, -1.0,
    /// );
    ///
    /// let det: f64 = reflection.lu().determinant();
    ///
    /// // Negative determinant indicates orientation reversal
    /// assert!((det + 1.0).abs() < 1e-10);  // det = -1
    /// ```
    ///
    /// # Performance
    ///
    /// Computing the determinant using LU decomposition is O(n³) if you need to decompose
    /// the matrix first, but only O(n) if you already have the LU decomposition. If you
    /// only need the determinant and don't need to solve systems or invert the matrix,
    /// you can use [`Matrix::determinant`] directly.
    ///
    /// # See Also
    ///
    /// - [`LU::is_invertible`]: Check invertibility without computing the actual determinant
    /// - [`Matrix::determinant`]: Direct determinant computation
    #[must_use]
    pub fn determinant(&self) -> T {
        let dim = self.lu.nrows();
        assert!(
            self.lu.is_square(),
            "LU determinant: unable to compute the determinant of a non-square matrix."
        );

        let mut res = T::one();
        for i in 0..dim {
            res *= unsafe { self.lu.get_unchecked((i, i)).clone() };
        }

        res * self.p.determinant()
    }

    /// Checks whether the decomposed matrix is invertible (non-singular).
    ///
    /// A matrix is invertible if and only if:
    /// - It is square
    /// - Its determinant is non-zero
    /// - All diagonal elements of `U` in the LU decomposition are non-zero
    ///
    /// This method efficiently checks invertibility by examining the diagonal elements
    /// of the upper triangular matrix `U`, which is much faster than computing the
    /// full determinant.
    ///
    /// # Returns
    ///
    /// * `true` - If the matrix is invertible (has an inverse)
    /// * `false` - If the matrix is singular (has no inverse)
    ///
    /// # Panics
    ///
    /// Panics if the decomposed matrix is not square.
    ///
    /// # Example: Basic Usage
    ///
    /// ```
    /// use nalgebra::Matrix3;
    ///
    /// let invertible = Matrix3::new(
    ///     2.0, 1.0, 1.0,
    ///     4.0, 3.0, 3.0,
    ///     8.0, 7.0, 9.0,
    /// );
    ///
    /// let lu = invertible.lu();
    /// assert!(lu.is_invertible());
    ///
    /// // Safe to call try_inverse() now
    /// let inv = lu.try_inverse().unwrap();
    /// ```
    ///
    /// # Example: Singular Matrix
    ///
    /// ```
    /// use nalgebra::Matrix2;
    ///
    /// // Create a singular matrix (linearly dependent rows)
    /// let singular = Matrix2::new(
    ///     1.0, 2.0,
    ///     2.0, 4.0,
    /// );
    ///
    /// let lu = singular.lu();
    /// assert!(!lu.is_invertible());
    ///
    /// // try_inverse() would return None
    /// assert!(lu.try_inverse().is_none());
    /// ```
    ///
    /// # Example: Checking Before Solving
    ///
    /// ```
    /// use nalgebra::{Matrix3, Vector3};
    ///
    /// let a = Matrix3::new(
    ///     1.0, 0.0, 0.0,
    ///     0.0, 1.0, 0.0,
    ///     0.0, 0.0, 1.0,
    /// );
    /// let b = Vector3::new(1.0, 2.0, 3.0);
    ///
    /// let lu = a.lu();
    ///
    /// if lu.is_invertible() {
    ///     let x = lu.solve(&b).unwrap();
    ///     println!("Solution: {:?}", x);
    /// } else {
    ///     println!("System has no unique solution");
    /// }
    /// ```
    ///
    /// # Example: Nearly Singular Matrix
    ///
    /// ```
    /// use nalgebra::Matrix2;
    ///
    /// // A matrix that's nearly singular (very small determinant)
    /// let nearly_singular = Matrix2::new(
    ///     1.0,       1.0,
    ///     1.0,       1.0 + 1e-15,
    /// );
    ///
    /// let lu = nearly_singular.lu();
    ///
    /// // Technically invertible, but numerically unstable
    /// if lu.is_invertible() {
    ///     println!("Matrix is invertible (but may be ill-conditioned)");
    /// }
    /// ```
    ///
    /// # Performance
    ///
    /// This is a very fast O(n) operation that only checks the diagonal elements of `U`.
    /// It's much faster than computing the determinant if you only need to know whether
    /// the matrix is invertible.
    ///
    /// # Numerical Considerations
    ///
    /// This method checks if diagonal elements are exactly zero using the `is_zero()`
    /// method. In practice, very small (but non-zero) diagonal elements may indicate
    /// numerical instability, even though the method returns `true`.
    ///
    /// # See Also
    ///
    /// - [`LU::determinant`]: Compute the actual determinant value
    /// - [`LU::try_inverse`]: Attempt to compute the inverse
    /// - [`LU::solve`]: Solve a linear system (also requires invertibility)
    #[must_use]
    pub fn is_invertible(&self) -> bool {
        assert!(
            self.lu.is_square(),
            "LU: unable to test the invertibility of a non-square matrix."
        );

        for i in 0..self.lu.nrows() {
            if self.lu[(i, i)].is_zero() {
                return false;
            }
        }

        true
    }
}

#[doc(hidden)]
/// Executes one step of gaussian elimination on the i-th row and column of `matrix`. The diagonal
/// element `matrix[(i, i)]` is provided as argument.
pub fn gauss_step<T, R: Dim, C: Dim, S>(matrix: &mut Matrix<T, R, C, S>, diag: T, i: usize)
where
    T: Scalar + Field,
    S: StorageMut<T, R, C>,
{
    let mut submat = matrix.view_range_mut(i.., i..);

    let inv_diag = T::one() / diag;

    let (mut coeffs, mut submat) = submat.columns_range_pair_mut(0, 1..);

    let mut coeffs = coeffs.rows_range_mut(1..);
    coeffs *= inv_diag;

    let (pivot_row, mut down) = submat.rows_range_pair_mut(0, 1..);

    for k in 0..pivot_row.ncols() {
        down.column_mut(k)
            .axpy(-pivot_row[k].clone(), &coeffs, T::one());
    }
}

#[doc(hidden)]
/// Swaps the rows `i` with the row `piv` and executes one step of gaussian elimination on the i-th
/// row and column of `matrix`. The diagonal element `matrix[(i, i)]` is provided as argument.
pub fn gauss_step_swap<T, R: Dim, C: Dim, S>(
    matrix: &mut Matrix<T, R, C, S>,
    diag: T,
    i: usize,
    piv: usize,
) where
    T: Scalar + Field,
    S: StorageMut<T, R, C>,
{
    let piv = piv - i;
    let mut submat = matrix.view_range_mut(i.., i..);

    let inv_diag = T::one() / diag;

    let (mut coeffs, mut submat) = submat.columns_range_pair_mut(0, 1..);

    coeffs.swap((0, 0), (piv, 0));
    let mut coeffs = coeffs.rows_range_mut(1..);
    coeffs *= inv_diag;

    let (mut pivot_row, mut down) = submat.rows_range_pair_mut(0, 1..);

    for k in 0..pivot_row.ncols() {
        mem::swap(&mut pivot_row[k], &mut down[(piv - 1, k)]);
        down.column_mut(k)
            .axpy(-pivot_row[k].clone(), &coeffs, T::one());
    }
}
