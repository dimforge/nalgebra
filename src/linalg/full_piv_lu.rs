#[cfg(feature = "serde-serialize-no-std")]
use serde::{Deserialize, Serialize};

use crate::allocator::Allocator;
use crate::base::{DefaultAllocator, Matrix, OMatrix};
use crate::constraint::{SameNumberOfRows, ShapeConstraint};
use crate::dimension::{Dim, DimMin, DimMinimum};
use crate::storage::{Storage, StorageMut};
use simba::scalar::ComplexField;

use crate::linalg::PermutationSequence;
use crate::linalg::lu;

/// LU decomposition with full row and column pivoting.
///
/// # What is Full Pivoting LU Decomposition?
///
/// Full pivoting LU decomposition is an advanced matrix factorization technique that breaks down
/// a matrix `A` into the product of:
/// - **L**: A lower triangular matrix (all entries above the diagonal are zero)
/// - **U**: An upper triangular matrix (all entries below the diagonal are zero)
/// - **P**: A row permutation matrix
/// - **Q**: A column permutation matrix
///
/// The decomposition satisfies: `P * A * Q = L * U`
///
/// # Full Pivoting vs. Partial Pivoting
///
/// Full pivoting differs from partial pivoting (used in [`LU`](crate::linalg::LU)) in an important way:
/// - **Partial pivoting** (standard LU): Only searches for the largest element in the current column
///   and swaps rows. This gives the relationship `P * A = L * U`.
/// - **Full pivoting** (this type): Searches for the largest element in the entire remaining
///   submatrix and swaps both rows and columns. This gives `P * A * Q = L * U`.
///
/// # Why Use Full Pivoting?
///
/// Full pivoting provides:
/// - **Maximum numerical stability**: By choosing the largest possible pivot element, it minimizes
///   rounding errors in floating-point arithmetic
/// - **Better handling of ill-conditioned matrices**: More robust for matrices that are nearly singular
/// - **More accurate results**: When numerical precision is critical
///
/// The trade-off is performance: full pivoting is slower than partial pivoting because it searches
/// a larger area for pivot elements. For most applications, partial pivoting ([`LU`](crate::linalg::LU))
/// is sufficient.
///
/// # When to Use Full Pivoting
///
/// Choose full pivoting when:
/// - Working with ill-conditioned matrices (small singular values)
/// - Maximum numerical stability is required
/// - The matrix might have unusual pivot patterns
/// - Accuracy is more important than speed
///
/// Choose partial pivoting ([`LU`](crate::linalg::LU)) when:
/// - Speed is important
/// - The matrix is well-conditioned
/// - Standard numerical stability is sufficient
///
/// # Common Applications
///
/// Full pivoting LU decomposition is particularly useful for:
/// - Solving systems of linear equations with maximum accuracy
/// - Computing accurate matrix determinants
/// - Finding matrix inverses when stability is crucial
/// - Rank determination
/// - Numerical analysis requiring high precision
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
/// // Extract the L, U, P, and Q components
/// let (p, l, u, q) = lu.unpack();
///
/// // Verify that P * A * Q = L * U
/// let mut m_copy = m.clone();
/// p.permute_rows(&mut m_copy);
/// q.permute_columns(&mut m_copy);
/// assert!((m_copy - l * u).norm() < 1e-10);
/// ```
///
/// # See Also
///
/// - [`FullPivLU::new`]: Constructs a new full pivoting LU decomposition
/// - [`FullPivLU::solve`]: Solves linear systems using this decomposition
/// - [`FullPivLU::determinant`]: Computes the determinant
/// - [`FullPivLU::try_inverse`]: Computes the matrix inverse
/// - [`LU`](crate::linalg::LU): Standard LU decomposition with partial (row) pivoting
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
pub struct FullPivLU<T: ComplexField, R: DimMin<C>, C: Dim>
where
    DefaultAllocator: Allocator<R, C> + Allocator<DimMinimum<R, C>>,
{
    lu: OMatrix<T, R, C>,
    p: PermutationSequence<DimMinimum<R, C>>,
    q: PermutationSequence<DimMinimum<R, C>>,
}

impl<T: ComplexField, R: DimMin<C>, C: Dim> Copy for FullPivLU<T, R, C>
where
    DefaultAllocator: Allocator<R, C> + Allocator<DimMinimum<R, C>>,
    OMatrix<T, R, C>: Copy,
    PermutationSequence<DimMinimum<R, C>>: Copy,
{
}

impl<T: ComplexField, R: DimMin<C>, C: Dim> FullPivLU<T, R, C>
where
    DefaultAllocator: Allocator<R, C> + Allocator<DimMinimum<R, C>>,
{
    /// Computes the LU decomposition with full pivoting of `matrix`.
    ///
    /// This is the primary constructor for creating a full pivoting LU decomposition. It performs
    /// the factorization `P * A * Q = L * U`, where:
    /// - `P` is a row permutation matrix (represented as a permutation sequence)
    /// - `A` is the original matrix
    /// - `Q` is a column permutation matrix (represented as a permutation sequence)
    /// - `L` is a lower triangular matrix with ones on the diagonal
    /// - `U` is an upper triangular matrix
    ///
    /// # How It Works
    ///
    /// The algorithm uses Gaussian elimination with full pivoting:
    /// 1. For each diagonal position, search the entire remaining submatrix for the element
    ///    with the largest absolute value
    /// 2. Swap both rows and columns to bring this element to the diagonal position
    /// 3. Eliminate entries below the diagonal using row operations
    /// 4. Store the elimination factors in the lower triangle
    ///
    /// Full pivoting provides maximum numerical stability by always choosing the largest
    /// available pivot element, minimizing the impact of rounding errors.
    ///
    /// # Arguments
    ///
    /// * `matrix` - The matrix to decompose (can be rectangular, though square is most common)
    ///
    /// # Example: Basic Usage
    ///
    /// ```
    /// use nalgebra::{Matrix3, FullPivLU};
    ///
    /// let m = Matrix3::new(
    ///     1.0, 2.0, 3.0,
    ///     4.0, 5.0, 6.0,
    ///     7.0, 8.0, 9.0,
    /// );
    ///
    /// let lu = FullPivLU::new(m);
    ///
    /// // The decomposition stores both L and U in a compact form
    /// // along with both row and column permutations
    /// let l = lu.l();
    /// let u = lu.u();
    /// let p = lu.p();
    /// let q = lu.q();
    ///
    /// // Verify: P * A * Q = L * U
    /// let mut m_copy = m.clone();
    /// p.permute_rows(&mut m_copy);
    /// q.permute_columns(&mut m_copy);
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
    /// let lu = a.full_piv_lu();
    /// let x = lu.solve(&b).expect("Linear system has no solution");
    ///
    /// // Verify the solution
    /// assert!((a * x - b).norm() < 1e-10);
    /// ```
    ///
    /// # Example: Comparing with Partial Pivoting
    ///
    /// ```
    /// use nalgebra::Matrix3;
    ///
    /// // For a well-conditioned matrix, both methods give similar results
    /// let m = Matrix3::new(
    ///     2.0, 1.0, 1.0,
    ///     4.0, 3.0, 3.0,
    ///     8.0, 7.0, 9.0,
    /// );
    ///
    /// // Full pivoting (more stable, slower)
    /// let full_piv_lu = m.full_piv_lu();
    /// let det_full = full_piv_lu.determinant();
    ///
    /// // Partial pivoting (faster, usually sufficient)
    /// let partial_lu = m.lu();
    /// let det_partial: f64 = partial_lu.determinant();
    ///
    /// // Results should be very close
    /// assert!((det_full - det_partial).abs() < 1e-10);
    /// ```
    ///
    /// # Example: Ill-Conditioned Matrix
    ///
    /// ```
    /// use nalgebra::Matrix3;
    ///
    /// // An ill-conditioned matrix where full pivoting helps
    /// let ill_conditioned = Matrix3::new(
    ///     1e-10, 1.0,   2.0,
    ///     2.0,   3.0,   4.0,
    ///     5.0,   6.0,   7.0,
    /// );
    ///
    /// // Full pivoting will handle this more accurately by
    /// // avoiding the tiny 1e-10 element as a pivot
    /// let lu = ill_conditioned.full_piv_lu();
    ///
    /// // The decomposition succeeds with better numerical properties
    /// let det: f64 = lu.determinant();
    /// ```
    ///
    /// # Example: Rectangular Matrix
    ///
    /// ```
    /// use nalgebra::{Matrix3x2, FullPivLU};
    ///
    /// // Full pivoting LU works on rectangular matrices too
    /// let m = Matrix3x2::new(
    ///     1.0, 2.0,
    ///     3.0, 4.0,
    ///     5.0, 6.0,
    /// );
    ///
    /// let lu = FullPivLU::new(m);
    /// let l = lu.l();
    /// let u = lu.u();
    ///
    /// // L is 3x2 and U is 2x2
    /// assert_eq!(l.shape(), (3, 2));
    /// assert_eq!(u.shape(), (2, 2));
    /// ```
    ///
    /// # Performance Considerations
    ///
    /// Full pivoting requires O(n³) time, similar to partial pivoting, but with a larger
    /// constant factor due to the search over the entire submatrix at each step. For an
    /// n×n matrix, full pivoting typically takes 2-3 times longer than partial pivoting.
    ///
    /// If speed is critical and the matrix is well-conditioned, consider using
    /// [`LU::new`](crate::linalg::LU::new) (partial pivoting) instead.
    ///
    /// # See Also
    ///
    /// - [`FullPivLU::solve`]: Solve linear systems using this decomposition
    /// - [`FullPivLU::determinant`]: Compute determinants efficiently
    /// - [`FullPivLU::try_inverse`]: Compute matrix inverses
    /// - [`FullPivLU::unpack`]: Extract P, L, U, and Q components
    /// - [`LU::new`](crate::linalg::LU::new): Faster alternative with partial pivoting
    pub fn new(mut matrix: OMatrix<T, R, C>) -> Self {
        let (nrows, ncols) = matrix.shape_generic();
        let min_nrows_ncols = nrows.min(ncols);

        let mut p = PermutationSequence::identity_generic(min_nrows_ncols);
        let mut q = PermutationSequence::identity_generic(min_nrows_ncols);

        if min_nrows_ncols.value() == 0 {
            return Self { lu: matrix, p, q };
        }

        for i in 0..min_nrows_ncols.value() {
            let piv = matrix.view_range(i.., i..).icamax_full();
            let row_piv = piv.0 + i;
            let col_piv = piv.1 + i;
            let diag = matrix[(row_piv, col_piv)].clone();

            if diag.is_zero() {
                // The remaining of the matrix is zero.
                break;
            }

            matrix.swap_columns(i, col_piv);
            q.append_permutation(i, col_piv);

            if row_piv != i {
                p.append_permutation(i, row_piv);
                matrix.columns_range_mut(..i).swap_rows(i, row_piv);
                lu::gauss_step_swap(&mut matrix, diag, i, row_piv);
            } else {
                lu::gauss_step(&mut matrix, diag, i);
            }
        }

        Self { lu: matrix, p, q }
    }

    #[doc(hidden)]
    pub const fn lu_internal(&self) -> &OMatrix<T, R, C> {
        &self.lu
    }

    /// Extracts the lower triangular matrix `L` from this full pivoting LU decomposition.
    ///
    /// The lower triangular matrix `L` has the following properties:
    /// - All entries above the main diagonal are zero
    /// - All diagonal entries are one
    /// - The entries below the diagonal contain the elimination multipliers from Gaussian elimination
    ///
    /// This matrix, combined with `U` and the permutations `P` and `Q`, satisfies: `P * A * Q = L * U`
    ///
    /// # Returns
    ///
    /// A lower triangular matrix with ones on the diagonal.
    ///
    /// # Example: Basic Usage
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
    /// let lu = m.full_piv_lu();
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
    /// let lu = m.full_piv_lu();
    /// let l = lu.l();
    /// let u = lu.u();
    /// let p = lu.p();
    /// let q = lu.q();
    ///
    /// // Verify that P * A * Q = L * U
    /// let mut m_permuted = m.clone();
    /// p.permute_rows(&mut m_permuted);
    /// q.permute_columns(&mut m_permuted);
    /// let reconstructed = l * u;
    /// assert!((m_permuted - reconstructed).norm() < 1e-10);
    /// ```
    ///
    /// # Example: Understanding the Lower Triangular Structure
    ///
    /// ```
    /// use nalgebra::Matrix4;
    ///
    /// let m = Matrix4::new(
    ///     1.0,  2.0,  3.0,  4.0,
    ///     5.0,  6.0,  7.0,  8.0,
    ///     9.0, 10.0, 11.0, 12.0,
    ///     13.0, 14.0, 15.0, 17.0,
    /// );
    ///
    /// let lu = m.full_piv_lu();
    /// let l = lu.l();
    ///
    /// // The lower triangular matrix has a specific structure:
    /// // [ 1   0   0   0 ]
    /// // [ *   1   0   0 ]
    /// // [ *   *   1   0 ]
    /// // [ *   *   *   1 ]
    /// // where * represents the elimination multipliers
    ///
    /// // Verify the structure
    /// for i in 0..4 {
    ///     // Diagonal elements are all 1
    ///     assert_eq!(l[(i, i)], 1.0);
    ///
    ///     // Upper triangle is all zeros
    ///     for j in (i + 1)..4 {
    ///         assert_eq!(l[(i, j)], 0.0);
    ///     }
    /// }
    /// ```
    ///
    /// # Performance Note
    ///
    /// This method allocates a new matrix and copies data from the internal storage.
    /// If you need both `L` and `U`, consider using [`FullPivLU::unpack`] instead, which
    /// can be more efficient as it extracts all components at once.
    ///
    /// # See Also
    ///
    /// - [`FullPivLU::u`]: Extracts the upper triangular matrix
    /// - [`FullPivLU::p`]: Gets the row permutation sequence
    /// - [`FullPivLU::q`]: Gets the column permutation sequence
    /// - [`FullPivLU::unpack`]: Extracts all components (P, L, U, Q) at once
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

    /// Extracts the upper triangular matrix `U` from this full pivoting LU decomposition.
    ///
    /// The upper triangular matrix `U` has the following properties:
    /// - All entries below the main diagonal are zero
    /// - The diagonal and upper entries contain the result of Gaussian elimination
    ///
    /// This matrix, combined with `L` and the permutations `P` and `Q`, satisfies: `P * A * Q = L * U`
    ///
    /// # Returns
    ///
    /// An upper triangular matrix.
    ///
    /// # Example: Basic Usage
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
    /// let lu = m.full_piv_lu();
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
    /// let lu = m.full_piv_lu();
    /// let l = lu.l();
    /// let u = lu.u();
    /// let p = lu.p();
    /// let q = lu.q();
    ///
    /// // Verify that P * A * Q = L * U
    /// let mut m_permuted = m.clone();
    /// p.permute_rows(&mut m_permuted);
    /// q.permute_columns(&mut m_permuted);
    /// let l_times_u = l * u;
    /// assert!((m_permuted - l_times_u).norm() < 1e-10);
    /// ```
    ///
    /// # Example: Analyzing Numerical Stability
    ///
    /// ```
    /// use nalgebra::Matrix3;
    ///
    /// // Full pivoting ensures the largest element is always chosen as pivot
    /// let m = Matrix3::new(
    ///     1e-10, 1.0,   2.0,
    ///     2.0,   3.0,   4.0,
    ///     5.0,   6.0,   7.0,
    /// );
    ///
    /// let lu = m.full_piv_lu();
    /// let u = lu.u();
    ///
    /// // The diagonal elements of U should be relatively large
    /// // because full pivoting avoids tiny pivot elements
    /// for i in 0..3 {
    ///     let diag: f64 = u[(i, i)];
    ///     // Diagonal elements should not be extremely small
    ///     assert!(diag.abs() > 1e-12);
    /// }
    /// ```
    ///
    /// # Example: Understanding the Upper Triangular Structure
    ///
    /// ```
    /// use nalgebra::Matrix4;
    ///
    /// let m = Matrix4::new(
    ///     1.0,  2.0,  3.0,  4.0,
    ///     5.0,  6.0,  7.0,  8.0,
    ///     9.0, 10.0, 11.0, 12.0,
    ///     13.0, 14.0, 15.0, 17.0,
    /// );
    ///
    /// let lu = m.full_piv_lu();
    /// let u = lu.u();
    ///
    /// // The upper triangular matrix has a specific structure:
    /// // [ *   *   *   * ]
    /// // [ 0   *   *   * ]
    /// // [ 0   0   *   * ]
    /// // [ 0   0   0   * ]
    /// // where * represents non-zero values
    ///
    /// // Verify the structure
    /// for i in 0..4 {
    ///     // Lower triangle is all zeros
    ///     for j in 0..i {
    ///         assert_eq!(u[(i, j)], 0.0);
    ///     }
    /// }
    /// ```
    ///
    /// # Performance Note
    ///
    /// This method allocates a new matrix and copies data from the internal storage.
    /// If you need both `L` and `U`, consider using [`FullPivLU::unpack`] instead.
    ///
    /// # See Also
    ///
    /// - [`FullPivLU::l`]: Extracts the lower triangular matrix
    /// - [`FullPivLU::p`]: Gets the row permutation sequence
    /// - [`FullPivLU::q`]: Gets the column permutation sequence
    /// - [`FullPivLU::unpack`]: Extracts all components (P, L, U, Q) at once
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
    /// The row permutation sequence represents the row swaps performed during full pivoting
    /// LU decomposition. It can be used to reconstruct the permutation matrix P that, together
    /// with the column permutation Q, satisfies: `P * A * Q = L * U`
    ///
    /// # What is a Permutation Sequence?
    ///
    /// A permutation sequence is a compact representation of a permutation matrix. Instead of
    /// storing a full matrix (which would be mostly zeros), it stores only the sequence of
    /// row swaps needed to apply the permutation.
    ///
    /// # Returns
    ///
    /// A reference to the row permutation sequence.
    ///
    /// # Example: Basic Usage
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
    /// let lu = m.full_piv_lu();
    /// let p = lu.p();
    /// let q = lu.q();
    ///
    /// // Apply the permutations to the original matrix
    /// let mut permuted = m.clone();
    /// p.permute_rows(&mut permuted);
    /// q.permute_columns(&mut permuted);
    ///
    /// // The permuted matrix can be reconstructed as L * U
    /// let l = lu.l();
    /// let u = lu.u();
    /// assert!((permuted - l * u).norm() < 1e-10);
    /// ```
    ///
    /// # Example: Understanding Row Permutations
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
    /// let lu = m.full_piv_lu();
    /// let p = lu.p();
    ///
    /// // The permutation affects which rows are in which positions
    /// let mut m_copy = m.clone();
    /// p.permute_rows(&mut m_copy);
    ///
    /// // m_copy now has rows reordered according to the pivoting strategy
    /// ```
    ///
    /// # Example: Determinant of Permutation
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
    /// let p = lu.p();
    ///
    /// // The determinant of a permutation matrix is ±1
    /// let det_p = p.determinant::<i32>();
    /// assert!(det_p == 1 || det_p == -1);
    /// ```
    ///
    /// # See Also
    ///
    /// - [`FullPivLU::q`]: Gets the column permutation sequence
    /// - [`FullPivLU::l`]: Extracts the lower triangular matrix
    /// - [`FullPivLU::u`]: Extracts the upper triangular matrix
    /// - [`FullPivLU::unpack`]: Extracts all components (P, L, U, Q) at once
    /// - [`PermutationSequence::permute`]: Apply the permutation to a matrix
    #[inline]
    #[must_use]
    pub const fn p(&self) -> &PermutationSequence<DimMinimum<R, C>> {
        &self.p
    }

    /// Returns a reference to the column permutation sequence `Q` of this decomposition.
    ///
    /// The column permutation sequence represents the column swaps performed during full pivoting
    /// LU decomposition. This is a key difference from partial pivoting, which only swaps rows.
    /// Together with the row permutation P, it satisfies: `P * A * Q = L * U`
    ///
    /// # What Makes Full Pivoting Different?
    ///
    /// In full pivoting, we search for the largest element in the entire remaining submatrix
    /// and swap both rows AND columns to bring it to the pivot position. The column permutation
    /// Q records these column swaps, providing maximum numerical stability.
    ///
    /// # What is a Permutation Sequence?
    ///
    /// A permutation sequence is a compact representation of a permutation matrix. Instead of
    /// storing a full matrix (which would be mostly zeros), it stores only the sequence of
    /// column swaps needed to apply the permutation.
    ///
    /// # Returns
    ///
    /// A reference to the column permutation sequence.
    ///
    /// # Example: Basic Usage
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
    /// let lu = m.full_piv_lu();
    /// let p = lu.p();
    /// let q = lu.q();
    ///
    /// // Apply both permutations to the original matrix
    /// let mut permuted = m.clone();
    /// p.permute_rows(&mut permuted);
    /// q.permute_columns(&mut permuted);
    ///
    /// // The permuted matrix equals L * U
    /// let l = lu.l();
    /// let u = lu.u();
    /// assert!((permuted - l * u).norm() < 1e-10);
    /// ```
    ///
    /// # Example: Understanding Column Permutations
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
    /// let lu = m.full_piv_lu();
    /// let q = lu.q();
    ///
    /// // The permutation affects which columns are in which positions
    /// let mut m_copy = m.clone();
    /// q.permute_columns(&mut m_copy);
    ///
    /// // m_copy now has columns reordered according to the pivoting strategy
    /// ```
    ///
    /// # Example: Comparing with Partial Pivoting
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
    /// // Full pivoting has BOTH row and column permutations
    /// let full_lu = m.full_piv_lu();
    /// let p = full_lu.p();  // Row permutation
    /// let q = full_lu.q();  // Column permutation (unique to full pivoting)
    ///
    /// // Partial pivoting only has row permutations
    /// let partial_lu = m.lu();
    /// let p_partial = partial_lu.p();  // Only row permutation
    /// // No column permutation in partial pivoting!
    /// ```
    ///
    /// # Example: Determinant of Permutation
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
    /// let q = lu.q();
    ///
    /// // The determinant of a permutation matrix is ±1
    /// let det_q = q.determinant::<i32>();
    /// assert!(det_q == 1 || det_q == -1);
    /// ```
    ///
    /// # See Also
    ///
    /// - [`FullPivLU::p`]: Gets the row permutation sequence
    /// - [`FullPivLU::l`]: Extracts the lower triangular matrix
    /// - [`FullPivLU::u`]: Extracts the upper triangular matrix
    /// - [`FullPivLU::unpack`]: Extracts all components (P, L, U, Q) at once
    /// - [`PermutationSequence::permute`]: Apply the permutation to a matrix
    #[inline]
    #[must_use]
    pub const fn q(&self) -> &PermutationSequence<DimMinimum<R, C>> {
        &self.q
    }

    /// Extracts all components of the full pivoting LU decomposition: `(P, L, U, Q)`.
    ///
    /// This method consumes the full pivoting LU decomposition and returns all four components:
    /// - `P`: The row permutation sequence representing row swaps
    /// - `L`: The lower triangular matrix with ones on the diagonal
    /// - `U`: The upper triangular matrix
    /// - `Q`: The column permutation sequence representing column swaps
    ///
    /// These components satisfy the equation: `P * A * Q = L * U`
    ///
    /// # Returns
    ///
    /// A tuple `(P, L, U, Q)` containing both permutation sequences and both triangular matrices.
    ///
    /// # Performance Note
    ///
    /// This method is more efficient than calling `p()`, `l()`, `u()`, and `q()` separately,
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
    /// let lu = m.full_piv_lu();
    /// let (p, l, u, q) = lu.unpack();
    ///
    /// // Verify the decomposition: P * A * Q = L * U
    /// let mut m_copy = m.clone();
    /// p.permute_rows(&mut m_copy);
    /// q.permute_columns(&mut m_copy);
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
    /// let lu = m.full_piv_lu();
    /// let (p, l, u, q) = lu.unpack();
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
    ///
    /// // P and Q are permutation sequences
    /// // Their determinants are ±1
    /// let det_p = p.determinant::<i32>();
    /// let det_q = q.determinant::<i32>();
    /// assert!(det_p == 1 || det_p == -1);
    /// assert!(det_q == 1 || det_q == -1);
    /// ```
    ///
    /// # Example: Manually Solving a Linear System
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
    /// let lu = a.full_piv_lu();
    /// let (p, l, u, q) = lu.unpack();
    ///
    /// // To solve A*x = b with full pivoting:
    /// // 1. P*A*Q*y = P*b  =>  L*U*y = P*b  (where y = Q^(-1)*x)
    /// // 2. Let z = U*y, solve L*z = P*b for z
    /// // 3. Solve U*y = z for y
    /// // 4. Solve Q*y = x for x (or x = Q^(-1)*y)
    ///
    /// let mut pb = b.clone();
    /// p.permute_rows(&mut pb);
    /// let z = l.solve_lower_triangular(&pb).unwrap();
    /// let mut y = u.solve_upper_triangular(&z).unwrap();
    /// q.inv_permute_rows(&mut y);
    /// let x = y;
    ///
    /// // Verify the solution
    /// assert!((a * x - b).norm() < 1e-10);
    /// ```
    ///
    /// # Example: Comparing Full vs Partial Pivoting
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
    /// // Full pivoting gives P, L, U, Q
    /// let (p_full, l_full, u_full, q_full) = m.full_piv_lu().unpack();
    ///
    /// // Partial pivoting gives only P, L, U (no Q)
    /// let (p_partial, l_partial, u_partial) = m.lu().unpack();
    ///
    /// // Full pivoting has the additional column permutation Q
    /// // Both methods reconstruct the original matrix (with permutations)
    /// let mut m1 = m.clone();
    /// p_full.permute_rows(&mut m1);
    /// q_full.permute_columns(&mut m1);
    /// assert!((m1 - l_full * u_full).norm() < 1e-10);
    ///
    /// let mut m2 = m.clone();
    /// p_partial.permute_rows(&mut m2);
    /// assert!((m2 - l_partial * u_partial).norm() < 1e-10);
    /// ```
    ///
    /// # Example: Analyzing Numerical Stability
    ///
    /// ```
    /// use nalgebra::Matrix3;
    ///
    /// // An ill-conditioned matrix
    /// let ill_cond = Matrix3::new(
    ///     1e-10, 1.0,   2.0,
    ///     2.0,   3.0,   4.0,
    ///     5.0,   6.0,   7.0,
    /// );
    ///
    /// let (p, l, u, q) = ill_cond.full_piv_lu().unpack();
    ///
    /// // Full pivoting should have avoided the tiny 1e-10 as a pivot
    /// // Check that diagonal elements of U are not extremely small
    /// for i in 0..3 {
    ///     let diag: f64 = u[(i, i)];
    ///     // Diagonal should be reasonably large due to full pivoting
    ///     assert!(diag.abs() > 1e-12);
    /// }
    /// ```
    ///
    /// # See Also
    ///
    /// - [`FullPivLU::l`]: Extract only the lower triangular matrix
    /// - [`FullPivLU::u`]: Extract only the upper triangular matrix
    /// - [`FullPivLU::p`]: Get only the row permutation sequence
    /// - [`FullPivLU::q`]: Get only the column permutation sequence
    /// - [`LU::unpack`](crate::linalg::LU::unpack): Similar for partial pivoting (returns P, L, U)
    #[inline]
    pub fn unpack(
        self,
    ) -> (
        PermutationSequence<DimMinimum<R, C>>,
        OMatrix<T, R, DimMinimum<R, C>>,
        OMatrix<T, DimMinimum<R, C>, C>,
        PermutationSequence<DimMinimum<R, C>>,
    )
    where
        DefaultAllocator: Allocator<R, DimMinimum<R, C>> + Allocator<DimMinimum<R, C>, C>,
    {
        // Use reallocation for either l or u.
        let l = self.l();
        let u = self.u();
        let p = self.p;
        let q = self.q;

        (p, l, u, q)
    }
}

impl<T: ComplexField, D: DimMin<D, Output = D>> FullPivLU<T, D, D>
where
    DefaultAllocator: Allocator<D, D> + Allocator<D>,
{
    /// Solves the linear system `A * x = b` using this full pivoting LU decomposition, where `x` is the unknown.
    ///
    /// This method efficiently solves systems of linear equations by leveraging the precomputed
    /// full pivoting LU decomposition. Given a matrix `A` (which was decomposed to create this
    /// `FullPivLU` object) and a right-hand side `b`, it finds the solution vector `x` such that
    /// `A * x = b`.
    ///
    /// Full pivoting provides maximum numerical stability when solving linear systems, making this
    /// method particularly suitable for ill-conditioned matrices where standard partial pivoting
    /// might produce inaccurate results.
    ///
    /// # How It Works
    ///
    /// The solution process with full pivoting involves:
    /// 1. Apply row permutation: Compute `P * b`
    /// 2. Forward substitution: Solve `L * y = P * b` for `y`
    /// 3. Back substitution: Solve `U * z = y` for `z`
    /// 4. Apply inverse column permutation: Compute `x = Q^(-1) * z`
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
    /// let lu = a.full_piv_lu();
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
    /// let lu = a.full_piv_lu();
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
    /// let lu = a.full_piv_lu();
    /// let result = lu.solve(&b);
    ///
    /// // System has no unique solution
    /// assert!(result.is_none());
    /// ```
    ///
    /// # Example: Ill-Conditioned System
    ///
    /// ```
    /// use nalgebra::{Matrix3, Vector3};
    ///
    /// // An ill-conditioned matrix where full pivoting helps
    /// let a = Matrix3::new(
    ///     1e-10, 1.0,   2.0,
    ///     2.0,   3.0,   4.0,
    ///     5.0,   6.0,   7.0,
    /// );
    /// let b = Vector3::new(3.0, 9.0, 18.0);
    ///
    /// // Full pivoting provides better numerical stability
    /// let lu = a.full_piv_lu();
    /// let x = lu.solve(&b).expect("Linear system has no solution");
    ///
    /// // Verify the solution
    /// assert!((a * x - b).norm() < 1e-8);
    /// ```
    ///
    /// # Example: Reusing Decomposition for Multiple Solves
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
    /// // Decompose once
    /// let lu = a.full_piv_lu();
    ///
    /// // Solve multiple systems efficiently
    /// let b1 = Vector3::new(4.0, 10.0, 24.0);
    /// let b2 = Vector3::new(1.0, 2.0, 3.0);
    /// let b3 = Vector3::new(0.0, 1.0, 2.0);
    ///
    /// let x1 = lu.solve(&b1).unwrap();
    /// let x2 = lu.solve(&b2).unwrap();
    /// let x3 = lu.solve(&b3).unwrap();
    ///
    /// // All solutions are correct
    /// assert!((a * x1 - b1).norm() < 1e-10);
    /// assert!((a * x2 - b2).norm() < 1e-10);
    /// assert!((a * x3 - b3).norm() < 1e-10);
    /// ```
    ///
    /// # Performance
    ///
    /// This method allocates a new matrix/vector for the result. If you want to reuse
    /// an existing buffer, use [`FullPivLU::solve_mut`] instead.
    ///
    /// # See Also
    ///
    /// - [`FullPivLU::solve_mut`]: In-place version that modifies `b` directly
    /// - [`FullPivLU::try_inverse`]: Computes the matrix inverse
    /// - [`FullPivLU::is_invertible`]: Check if the matrix is invertible before solving
    /// - [`LU::solve`](crate::linalg::LU::solve): Faster alternative with partial pivoting
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
    /// This is the in-place version of [`FullPivLU::solve`]. It modifies the input `b` directly,
    /// replacing it with the solution `x`. This is more memory-efficient than `solve()` when
    /// you don't need to preserve the original `b` vector.
    ///
    /// Full pivoting provides maximum numerical stability when solving linear systems, making this
    /// method particularly suitable for ill-conditioned matrices.
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
    /// let lu = a.full_piv_lu();
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
    /// let lu = a.full_piv_lu();
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
    /// let lu = singular.full_piv_lu();
    /// let success = lu.solve_mut(&mut b);
    ///
    /// assert!(!success);  // Cannot solve with singular matrix
    /// // Don't use b here - it contains invalid data
    /// ```
    ///
    /// # Example: Ill-Conditioned System with Full Pivoting
    ///
    /// ```
    /// use nalgebra::{Matrix3, Vector3};
    ///
    /// // An ill-conditioned matrix
    /// let a = Matrix3::new(
    ///     1e-10, 1.0,   2.0,
    ///     2.0,   3.0,   4.0,
    ///     5.0,   6.0,   7.0,
    /// );
    ///
    /// let mut b = Vector3::new(3.0, 9.0, 18.0);
    /// let original_b = b.clone();
    ///
    /// // Full pivoting provides better numerical stability
    /// let lu = a.full_piv_lu();
    /// let success = lu.solve_mut(&mut b);
    ///
    /// assert!(success);
    /// // Verify the solution
    /// assert!((a * b - original_b).norm() < 1e-8);
    /// ```
    ///
    /// # Example: Comparing with Partial Pivoting
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
    /// let mut b_full = Vector3::new(4.0, 10.0, 24.0);
    /// let mut b_partial = b_full.clone();
    /// let original_b = b_full.clone();
    ///
    /// // Full pivoting (more stable)
    /// let full_lu = a.full_piv_lu();
    /// full_lu.solve_mut(&mut b_full);
    ///
    /// // Partial pivoting (faster)
    /// let partial_lu = a.lu();
    /// partial_lu.solve_mut(&mut b_partial);
    ///
    /// // Both should give similar results for well-conditioned matrices
    /// assert!((a * b_full - original_b).norm() < 1e-10);
    /// assert!((a * b_partial - original_b).norm() < 1e-10);
    /// assert!((b_full - b_partial).norm() < 1e-10);
    /// ```
    ///
    /// # Performance
    ///
    /// This method is faster than [`FullPivLU::solve`] because it doesn't allocate new memory.
    /// When solving multiple systems with the same coefficient matrix, decompose once
    /// and call `solve_mut` multiple times for maximum efficiency.
    ///
    /// # See Also
    ///
    /// - [`FullPivLU::solve`]: Allocating version that returns a new vector
    /// - [`FullPivLU::is_invertible`]: Check if the matrix is invertible before solving
    /// - [`LU::solve_mut`](crate::linalg::LU::solve_mut): Faster alternative with partial pivoting
    pub fn solve_mut<R2: Dim, C2: Dim, S2>(&self, b: &mut Matrix<T, R2, C2, S2>) -> bool
    where
        S2: StorageMut<T, R2, C2>,
        ShapeConstraint: SameNumberOfRows<R2, D>,
    {
        assert_eq!(
            self.lu.nrows(),
            b.nrows(),
            "FullPivLU solve matrix dimension mismatch."
        );
        assert!(
            self.lu.is_square(),
            "FullPivLU solve: unable to solve a non-square system."
        );

        if self.is_invertible() {
            self.p.permute_rows(b);
            let _ = self.lu.solve_lower_triangular_with_diag_mut(b, T::one());
            let _ = self.lu.solve_upper_triangular_mut(b);
            self.q.inv_permute_rows(b);

            true
        } else {
            false
        }
    }

    /// Computes the inverse of the decomposed matrix using the full pivoting LU decomposition.
    ///
    /// Matrix inversion finds a matrix `A^(-1)` such that `A * A^(-1) = I`, where `I` is
    /// the identity matrix. The full pivoting LU decomposition provides maximum numerical
    /// stability for computing matrix inverses, especially for ill-conditioned matrices.
    ///
    /// # Returns
    ///
    /// * `Some(A_inv)` - The inverse matrix if it exists
    /// * `None` - If the matrix is singular (determinant is zero) and has no inverse
    ///
    /// # When to Use Matrix Inversion
    ///
    /// **Important:** For solving linear systems `A * x = b`, use [`FullPivLU::solve`] instead of
    /// computing `A^(-1) * b`. It's faster and more numerically stable.
    ///
    /// Matrix inversion is useful when you need to:
    /// - Solve many systems with different right-hand sides stored as columns
    /// - Apply the same transformation multiple times
    /// - Compute expressions involving the inverse matrix
    ///
    /// # When to Use Full Pivoting for Inversion
    ///
    /// Use full pivoting for matrix inversion when:
    /// - The matrix is ill-conditioned (has small singular values)
    /// - Maximum numerical accuracy is required
    /// - The matrix might have unusual pivot patterns
    ///
    /// For well-conditioned matrices, partial pivoting ([`LU::try_inverse`](crate::linalg::LU::try_inverse))
    /// is usually sufficient and faster.
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
    /// let lu = m.full_piv_lu();
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
    /// let lu = singular.full_piv_lu();
    /// let result = lu.try_inverse();
    ///
    /// assert!(result.is_none());  // No inverse exists
    /// ```
    ///
    /// # Example: Ill-Conditioned Matrix
    ///
    /// ```
    /// use nalgebra::Matrix3;
    ///
    /// // An ill-conditioned matrix where full pivoting helps
    /// let ill_cond = Matrix3::new(
    ///     1e-10, 1.0,   2.0,
    ///     2.0,   3.0,   4.0,
    ///     5.0,   6.0,   8.0,
    /// );
    ///
    /// // Full pivoting provides better numerical stability
    /// let lu = ill_cond.full_piv_lu();
    /// let inv = lu.try_inverse().expect("Matrix is not invertible");
    ///
    /// // Verify the inverse (with relaxed tolerance due to conditioning)
    /// let identity = Matrix3::identity();
    /// assert!((ill_cond * inv - identity).norm() < 1e-5);
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
    /// let lu = a.full_piv_lu();
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
    /// # Example: Comparing Full vs Partial Pivoting
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
    /// // Full pivoting (more stable, slower)
    /// let inv_full = m.full_piv_lu().try_inverse().unwrap();
    ///
    /// // Partial pivoting (faster, usually sufficient)
    /// let inv_partial = m.lu().try_inverse().unwrap();
    ///
    /// // For well-conditioned matrices, both give similar results
    /// assert!((inv_full - inv_partial).norm() < 1e-10);
    /// ```
    ///
    /// # Example: Checking Invertibility First
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
    /// // Check before attempting inversion
    /// if lu.is_invertible() {
    ///     let inv = lu.try_inverse().unwrap();
    ///     println!("Matrix is invertible");
    /// } else {
    ///     println!("Matrix is singular");
    /// }
    /// ```
    ///
    /// # See Also
    ///
    /// - [`FullPivLU::solve`]: Better choice for solving linear systems
    /// - [`FullPivLU::is_invertible`]: Check if a matrix is invertible without computing the inverse
    /// - [`FullPivLU::determinant`]: Compute the determinant
    /// - [`LU::try_inverse`](crate::linalg::LU::try_inverse): Faster alternative with partial pivoting
    /// - [`Matrix::try_inverse`]: Direct matrix inversion without explicit decomposition
    #[must_use]
    pub fn try_inverse(&self) -> Option<OMatrix<T, D, D>> {
        assert!(
            self.lu.is_square(),
            "FullPivLU inverse: unable to compute the inverse of a non-square matrix."
        );

        let (nrows, ncols) = self.lu.shape_generic();

        let mut res = OMatrix::identity_generic(nrows, ncols);
        if self.solve_mut(&mut res) {
            Some(res)
        } else {
            None
        }
    }

    /// Checks whether the decomposed matrix is invertible (non-singular).
    ///
    /// A matrix is invertible if and only if:
    /// - It is square
    /// - Its determinant is non-zero
    /// - All diagonal elements of `U` in the LU decomposition are non-zero
    ///
    /// This method efficiently checks invertibility by examining only the last diagonal element
    /// of the upper triangular matrix `U`. Since full pivoting ensures the largest available
    /// element is always chosen as pivot, if the last diagonal element is non-zero, all
    /// previous diagonal elements must also be non-zero.
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
    /// let lu = invertible.full_piv_lu();
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
    /// let lu = singular.full_piv_lu();
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
    /// let lu = a.full_piv_lu();
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
    /// let lu = nearly_singular.full_piv_lu();
    ///
    /// // Technically invertible, but numerically unstable
    /// if lu.is_invertible() {
    ///     println!("Matrix is invertible (but may be ill-conditioned)");
    /// }
    /// ```
    ///
    /// # Example: Comparing with Determinant Check
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
    /// // is_invertible() is faster than computing the determinant
    /// let invertible = lu.is_invertible();
    /// let det: f64 = lu.determinant();
    ///
    /// // Both should agree (determinant is non-zero iff matrix is invertible)
    /// assert_eq!(invertible, det.abs() > 1e-10);
    /// ```
    ///
    /// # Example: Full Pivoting Stability
    ///
    /// ```
    /// use nalgebra::Matrix3;
    ///
    /// // An ill-conditioned matrix
    /// let ill_cond = Matrix3::new(
    ///     1e-10, 1.0,   2.0,
    ///     2.0,   3.0,   4.0,
    ///     5.0,   6.0,   7.0,
    /// );
    ///
    /// let lu = ill_cond.full_piv_lu();
    ///
    /// // Full pivoting correctly identifies this as invertible
    /// // (even though it has a very small element)
    /// assert!(lu.is_invertible());
    /// ```
    ///
    /// # Performance
    ///
    /// This is a very fast O(1) operation for full pivoting LU (checks only one element).
    /// It's much faster than computing the determinant if you only need to know whether
    /// the matrix is invertible.
    ///
    /// # Numerical Considerations
    ///
    /// This method checks if the last diagonal element is exactly zero using the `is_zero()`
    /// method. In practice, very small (but non-zero) diagonal elements may indicate
    /// numerical instability, even though the method returns `true`.
    ///
    /// Full pivoting provides better numerical stability than partial pivoting because it
    /// ensures the largest available element is always used as pivot, making the test
    /// more reliable.
    ///
    /// # See Also
    ///
    /// - [`FullPivLU::determinant`]: Compute the actual determinant value
    /// - [`FullPivLU::try_inverse`]: Attempt to compute the inverse
    /// - [`FullPivLU::solve`]: Solve a linear system (also requires invertibility)
    /// - [`LU::is_invertible`](crate::linalg::LU::is_invertible): Similar for partial pivoting
    #[must_use]
    pub fn is_invertible(&self) -> bool {
        assert!(
            self.lu.is_square(),
            "FullPivLU: unable to test the invertibility of a non-square matrix."
        );

        let dim = self.lu.nrows();
        !self.lu[(dim - 1, dim - 1)].is_zero()
    }

    /// Computes the determinant of the decomposed matrix using the full pivoting LU decomposition.
    ///
    /// The determinant is a scalar value that provides important information about a matrix:
    /// - If `det(A) = 0`, the matrix is singular (not invertible)
    /// - If `det(A) ≠ 0`, the matrix is invertible
    /// - The absolute value indicates how much the matrix scales volumes
    /// - The sign indicates whether the transformation preserves or reverses orientation
    ///
    /// Full pivoting provides maximum numerical accuracy when computing determinants,
    /// especially for ill-conditioned matrices.
    ///
    /// # How It Works
    ///
    /// For a full pivoting LU decomposition where `P * A * Q = L * U`:
    /// - `det(A) = det(L) * det(U) / (det(P) * det(Q))`
    /// - `det(L) = 1` (unit diagonal)
    /// - `det(U)` = product of diagonal elements
    /// - `det(P)` = ±1 (sign depends on number of row swaps)
    /// - `det(Q)` = ±1 (sign depends on number of column swaps)
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
    /// let lu = m.full_piv_lu();
    /// let det: f64 = lu.determinant();
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
    /// let det1: f64 = invertible.full_piv_lu().determinant();
    /// let det2: f64 = singular.full_piv_lu().determinant();
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
    /// let det: f64 = scale.full_piv_lu().determinant();
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
    /// let det: f64 = reflection.full_piv_lu().determinant();
    ///
    /// // Negative determinant indicates orientation reversal
    /// assert!((det + 1.0).abs() < 1e-10);  // det = -1
    /// ```
    ///
    /// # Example: Comparing Full vs Partial Pivoting
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
    /// // Full pivoting (more accurate for ill-conditioned matrices)
    /// let det_full: f64 = m.full_piv_lu().determinant();
    ///
    /// // Partial pivoting (faster, usually sufficient)
    /// let det_partial: f64 = m.lu().determinant();
    ///
    /// // For well-conditioned matrices, both give the same result
    /// assert!((det_full - det_partial).abs() < 1e-10);
    /// ```
    ///
    /// # Example: Ill-Conditioned Matrix
    ///
    /// ```
    /// use nalgebra::Matrix3;
    ///
    /// // An ill-conditioned matrix where full pivoting provides better accuracy
    /// let ill_cond = Matrix3::new(
    ///     1e-10, 1.0,   2.0,
    ///     2.0,   3.0,   4.0,
    ///     5.0,   6.0,   7.0,
    /// );
    ///
    /// // Full pivoting provides better numerical stability
    /// let det: f64 = ill_cond.full_piv_lu().determinant();
    ///
    /// // The determinant should be computed accurately despite the tiny element
    /// ```
    ///
    /// # Example: Determinant Properties
    ///
    /// ```
    /// use nalgebra::Matrix3;
    ///
    /// let a = Matrix3::new(
    ///     1.0, 2.0, 3.0,
    ///     4.0, 5.0, 6.0,
    ///     7.0, 8.0, 10.0,
    /// );
    ///
    /// let b = Matrix3::new(
    ///     2.0, 0.0, 0.0,
    ///     0.0, 3.0, 0.0,
    ///     0.0, 0.0, 4.0,
    /// );
    ///
    /// let det_a: f64 = a.full_piv_lu().determinant();
    /// let det_b: f64 = b.full_piv_lu().determinant();
    /// let det_ab: f64 = (a * b).full_piv_lu().determinant();
    ///
    /// // Property: det(AB) = det(A) * det(B)
    /// assert!((det_ab - det_a * det_b).abs() < 1e-9);
    /// ```
    ///
    /// # Example: Using Determinant for Matrix Properties
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
    /// let det: f64 = lu.determinant();
    ///
    /// // Use determinant to check various properties
    /// if det.abs() < 1e-10 {
    ///     println!("Matrix is singular");
    /// } else if det > 0.0 {
    ///     println!("Matrix preserves orientation (det = {})", det);
    /// } else {
    ///     println!("Matrix reverses orientation (det = {})", det);
    /// }
    ///
    /// // Check if matrix is invertible
    /// assert_eq!(lu.is_invertible(), det.abs() > 1e-10);
    /// ```
    ///
    /// # Performance
    ///
    /// Computing the determinant using full pivoting LU decomposition is O(n³) if you need to
    /// decompose the matrix first, but only O(n) if you already have the LU decomposition.
    /// If you only need the determinant and don't need to solve systems or invert the matrix,
    /// you can use [`Matrix::determinant`] directly.
    ///
    /// # Numerical Accuracy
    ///
    /// Full pivoting provides better numerical accuracy than partial pivoting when computing
    /// determinants of ill-conditioned matrices. This is because full pivoting always chooses
    /// the largest available element as pivot, minimizing rounding errors.
    ///
    /// # See Also
    ///
    /// - [`FullPivLU::is_invertible`]: Check invertibility without computing the actual determinant
    /// - [`FullPivLU::try_inverse`]: Compute the matrix inverse
    /// - [`LU::determinant`](crate::linalg::LU::determinant): Faster alternative with partial pivoting
    /// - [`Matrix::determinant`]: Direct determinant computation
    #[must_use]
    pub fn determinant(&self) -> T {
        assert!(
            self.lu.is_square(),
            "FullPivLU determinant: unable to compute the determinant of a non-square matrix."
        );

        let dim = self.lu.nrows();
        let mut res = self.lu[(dim - 1, dim - 1)].clone();
        if !res.is_zero() {
            for i in 0..dim - 1 {
                res *= unsafe { self.lu.get_unchecked((i, i)).clone() };
            }

            res * self.p.determinant() * self.q.determinant()
        } else {
            T::zero()
        }
    }
}
