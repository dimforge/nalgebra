use num::Zero;
#[cfg(feature = "serde-serialize-no-std")]
use serde::{Deserialize, Serialize};

use crate::ComplexField;
use crate::allocator::{Allocator, Reallocator};
use crate::base::{Const, DefaultAllocator, Matrix, OMatrix, OVector, Unit};
use crate::constraint::{SameNumberOfRows, ShapeConstraint};
use crate::dimension::{Dim, DimMin, DimMinimum};
use crate::storage::StorageMut;

use crate::geometry::Reflection;
use crate::linalg::{PermutationSequence, householder};
use std::mem::MaybeUninit;

/// The QR decomposition with column pivoting of a general matrix.
///
/// # What is Column Pivoting QR Decomposition?
///
/// Column Pivoting QR (ColPivQR) is an enhanced version of the standard QR decomposition that
/// provides better numerical stability and handles rank-deficient matrices more reliably.
///
/// Given a matrix **A** of size **m×n**, the decomposition computes:
///
/// **AP = QR**
///
/// where:
/// - **Q** is an **m×min(m,n)** orthogonal matrix (its columns are orthonormal vectors)
/// - **R** is an **min(m,n)×n** upper trapezoidal matrix
/// - **P** is an **n×n** permutation matrix that reorders columns
///
/// # Why Use Column Pivoting?
///
/// Column pivoting improves numerical stability by:
/// 1. **Selecting the largest column norm** at each step of the decomposition
/// 2. **Better handling of rank-deficient matrices** by moving near-zero columns to the right
/// 3. **Providing reliable rank estimation** through the diagonal elements of R
/// 4. **Reducing rounding errors** in computations
///
/// # Common Use Cases
///
/// - **Solving linear systems** with potentially rank-deficient matrices
/// - **Least squares problems** where the matrix might not have full rank
/// - **Rank determination** of a matrix
/// - **Computing matrix inverses** with better numerical stability
///
/// # Example
///
/// ```
/// use nalgebra::Matrix3x2;
///
/// let m = Matrix3x2::new(
///     1.0, 2.0,
///     3.0, 4.0,
///     5.0, 6.0
/// );
///
/// // Compute the ColPivQR decomposition
/// let qr = m.col_piv_qr();
///
/// // Extract the Q, R matrices and permutation
/// let (q, r, p) = qr.unpack();
///
/// // Verify: A * P = Q * R (within numerical precision)
/// let a_p = m * p.inverse();
/// let q_r = q * r;
/// assert!((a_p - q_r).norm() < 1e-10);
/// ```
#[cfg_attr(feature = "serde-serialize-no-std", derive(Serialize, Deserialize))]
#[cfg_attr(
    feature = "serde-serialize-no-std",
    serde(bound(serialize = "DefaultAllocator: Allocator<R, C> +
                           Allocator<DimMinimum<R, C>>,
         OMatrix<T, R, C>: Serialize,
         PermutationSequence<DimMinimum<R, C>>: Serialize,
         OVector<T, DimMinimum<R, C>>: Serialize"))
)]
#[cfg_attr(
    feature = "serde-serialize-no-std",
    serde(bound(deserialize = "DefaultAllocator: Allocator<R, C> +
                           Allocator<DimMinimum<R, C>>,
         OMatrix<T, R, C>: Deserialize<'de>,
         PermutationSequence<DimMinimum<R, C>>: Deserialize<'de>,
         OVector<T, DimMinimum<R, C>>: Deserialize<'de>"))
)]
#[cfg_attr(feature = "defmt", derive(defmt::Format))]
#[derive(Clone, Debug)]
pub struct ColPivQR<T: ComplexField, R: DimMin<C>, C: Dim>
where
    DefaultAllocator: Allocator<R, C> + Allocator<DimMinimum<R, C>>,
{
    col_piv_qr: OMatrix<T, R, C>,
    p: PermutationSequence<DimMinimum<R, C>>,
    diag: OVector<T, DimMinimum<R, C>>,
}

impl<T: ComplexField, R: DimMin<C>, C: Dim> Copy for ColPivQR<T, R, C>
where
    DefaultAllocator: Allocator<R, C> + Allocator<DimMinimum<R, C>>,
    OMatrix<T, R, C>: Copy,
    PermutationSequence<DimMinimum<R, C>>: Copy,
    OVector<T, DimMinimum<R, C>>: Copy,
{
}

impl<T: ComplexField, R: DimMin<C>, C: Dim> ColPivQR<T, R, C>
where
    DefaultAllocator: Allocator<R, C> + Allocator<R> + Allocator<DimMinimum<R, C>>,
{
    /// Computes the ColPivQR decomposition of a matrix using Householder reflections.
    ///
    /// This method performs QR decomposition with column pivoting, which selects columns
    /// in order of decreasing norm at each step. This pivoting strategy improves numerical
    /// stability and handles rank-deficient matrices better than standard QR decomposition.
    ///
    /// # Algorithm
    ///
    /// The algorithm works by:
    /// 1. At each step, finding the column with the largest norm
    /// 2. Swapping that column to the current position
    /// 3. Recording the permutation
    /// 4. Applying a Householder reflection to zero out elements below the diagonal
    ///
    /// # Arguments
    ///
    /// * `matrix` - The matrix to decompose (consumed by this method)
    ///
    /// # Returns
    ///
    /// A `ColPivQR` structure containing the decomposition factors.
    ///
    /// # Examples
    ///
    /// ## Basic usage with a simple matrix
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
    /// let qr = m.col_piv_qr();
    /// println!("Decomposition computed successfully");
    /// ```
    ///
    /// ## Handling a rank-deficient matrix
    ///
    /// ```
    /// use nalgebra::Matrix3;
    ///
    /// // Create a rank-2 matrix (third column = first + second)
    /// let m = Matrix3::new(
    ///     1.0, 0.0, 1.0,
    ///     0.0, 1.0, 1.0,
    ///     1.0, 1.0, 2.0
    /// );
    ///
    /// let qr = m.col_piv_qr();
    ///
    /// // Column pivoting will move the dependent column to the right
    /// // The diagonal of R will reveal the rank
    /// let r = qr.r();
    /// println!("R matrix:\n{}", r);
    /// ```
    ///
    /// ## Rectangular matrix (overdetermined system)
    ///
    /// ```
    /// use nalgebra::Matrix3x2;
    ///
    /// // More rows than columns
    /// let m = Matrix3x2::new(
    ///     1.0, 2.0,
    ///     3.0, 4.0,
    ///     5.0, 6.0
    /// );
    ///
    /// let qr = m.col_piv_qr();
    /// let q = qr.q();
    /// let r = qr.r();
    ///
    /// // Q is 3x2, R is 2x2
    /// assert_eq!(q.nrows(), 3);
    /// assert_eq!(q.ncols(), 2);
    /// assert_eq!(r.nrows(), 2);
    /// assert_eq!(r.ncols(), 2);
    /// ```
    ///
    /// # See Also
    ///
    /// - [`q()`](Self::q) - Extract the orthogonal matrix Q
    /// - [`r()`](Self::r) - Extract the upper triangular matrix R
    /// - [`p()`](Self::p) - Get the column permutation
    /// - [`unpack()`](Self::unpack) - Extract all components at once
    pub fn new(mut matrix: OMatrix<T, R, C>) -> Self {
        let (nrows, ncols) = matrix.shape_generic();
        let min_nrows_ncols = nrows.min(ncols);
        let mut p = PermutationSequence::identity_generic(min_nrows_ncols);

        if min_nrows_ncols.value() == 0 {
            return ColPivQR {
                col_piv_qr: matrix,
                p,
                diag: Matrix::zeros_generic(min_nrows_ncols, Const::<1>),
            };
        }

        let mut diag = Matrix::uninit(min_nrows_ncols, Const::<1>);

        for i in 0..min_nrows_ncols.value() {
            let piv = matrix.view_range(i.., i..).icamax_full();
            let col_piv = piv.1 + i;
            matrix.swap_columns(i, col_piv);
            p.append_permutation(i, col_piv);

            diag[i] =
                MaybeUninit::new(householder::clear_column_unchecked(&mut matrix, i, 0, None));
        }

        // Safety: diag is now fully initialized.
        let diag = unsafe { diag.assume_init() };

        ColPivQR {
            col_piv_qr: matrix,
            p,
            diag,
        }
    }

    /// Retrieves the upper trapezoidal submatrix R of this decomposition.
    ///
    /// The R matrix is an upper triangular (or upper trapezoidal for non-square matrices)
    /// matrix that contains the transformed data after applying the orthogonal Q matrix
    /// and column permutations. The diagonal elements of R provide information about
    /// the rank of the original matrix - small or zero diagonal elements indicate
    /// linear dependence.
    ///
    /// # Returns
    ///
    /// An upper trapezoidal matrix of size **min(m,n)×n** where the original matrix
    /// was **m×n**. This method creates a copy of the R matrix.
    ///
    /// # Examples
    ///
    /// ## Basic usage
    ///
    /// ```
    /// use nalgebra::Matrix3;
    ///
    /// let m = Matrix3::new(
    ///     2.0, 1.0, 0.0,
    ///     1.0, 2.0, 1.0,
    ///     0.0, 1.0, 2.0
    /// );
    ///
    /// let qr = m.col_piv_qr();
    /// let r = qr.r();
    ///
    /// // R is upper triangular
    /// println!("R matrix:\n{}", r);
    /// assert!(r[(1, 0)].abs() < 1e-10); // Below diagonal should be zero
    /// assert!(r[(2, 0)].abs() < 1e-10);
    /// assert!(r[(2, 1)].abs() < 1e-10);
    /// ```
    ///
    /// ## Checking matrix rank
    ///
    /// ```
    /// use nalgebra::Matrix3;
    ///
    /// // Rank-deficient matrix (third row = first row + second row)
    /// let m = Matrix3::new(
    ///     1.0, 0.0, 0.0,
    ///     0.0, 1.0, 0.0,
    ///     1.0, 1.0, 0.0
    /// );
    ///
    /// let qr = m.col_piv_qr();
    /// let r = qr.r();
    ///
    /// // Check diagonal elements to estimate rank
    /// let tolerance = 1e-10;
    /// let mut rank = 0;
    /// for i in 0..3 {
    ///     if r[(i, i)].abs() > tolerance {
    ///         rank += 1;
    ///     }
    /// }
    /// println!("Estimated rank: {}", rank); // Should be 2
    /// ```
    ///
    /// ## Rectangular matrix
    ///
    /// ```
    /// use nalgebra::Matrix3x2;
    ///
    /// let m = Matrix3x2::new(
    ///     1.0, 0.0,
    ///     2.0, 1.0,
    ///     3.0, 2.0
    /// );
    ///
    /// let qr = m.col_piv_qr();
    /// let r = qr.r();
    ///
    /// // For a 3x2 matrix, R is 2x2 (min(3,2) x 2)
    /// assert_eq!(r.nrows(), 2);
    /// assert_eq!(r.ncols(), 2);
    /// ```
    ///
    /// # See Also
    ///
    /// - [`unpack_r()`](Self::unpack_r) - A faster version that consumes self
    /// - [`q()`](Self::q) - Get the orthogonal matrix Q
    /// - [`p()`](Self::p) - Get the column permutation
    /// - [`unpack()`](Self::unpack) - Get Q, R, and P all at once
    #[inline]
    #[must_use]
    pub fn r(&self) -> OMatrix<T, DimMinimum<R, C>, C>
    where
        DefaultAllocator: Allocator<DimMinimum<R, C>, C>,
    {
        let (nrows, ncols) = self.col_piv_qr.shape_generic();
        let mut res = self
            .col_piv_qr
            .rows_generic(0, nrows.min(ncols))
            .upper_triangle();
        res.set_partial_diagonal(self.diag.iter().map(|e| T::from_real(e.clone().modulus())));
        res
    }

    /// Retrieves the upper trapezoidal submatrix R of this decomposition by consuming self.
    ///
    /// This method is more efficient than [`r()`](Self::r) because it reuses the internal
    /// storage instead of creating a copy. Use this when you no longer need the ColPivQR
    /// structure and only want the R matrix.
    ///
    /// # Returns
    ///
    /// An upper trapezoidal matrix of size **min(m,n)×n** where the original matrix
    /// was **m×n**.
    ///
    /// # Examples
    ///
    /// ## Basic usage
    ///
    /// ```
    /// use nalgebra::Matrix3;
    ///
    /// let m = Matrix3::new(
    ///     3.0, 1.0, 4.0,
    ///     1.0, 5.0, 9.0,
    ///     2.0, 6.0, 5.0
    /// );
    ///
    /// let qr = m.col_piv_qr();
    ///
    /// // If we only need R and don't need Q or P
    /// let r = qr.unpack_r();
    /// println!("R matrix:\n{}", r);
    /// ```
    ///
    /// ## Comparing with r() method
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
    /// let qr1 = m.col_piv_qr();
    /// let qr2 = m.col_piv_qr();
    ///
    /// // r() borrows self - we can still use qr1 afterwards
    /// let r1 = qr1.r();
    /// let q1 = qr1.q(); // Still possible
    ///
    /// // unpack_r() consumes self - more efficient but qr2 is moved
    /// let r2 = qr2.unpack_r();
    /// // let q2 = qr2.q(); // ERROR: qr2 was consumed
    ///
    /// // Both produce the same result
    /// assert!((r1 - r2).norm() < 1e-10);
    /// ```
    ///
    /// ## When to use unpack_r()
    ///
    /// ```
    /// use nalgebra::Matrix4x3;
    ///
    /// let data = Matrix4x3::new(
    ///     1.0, 2.0, 3.0,
    ///     4.0, 5.0, 6.0,
    ///     7.0, 8.0, 9.0,
    ///     10.0, 11.0, 12.0
    /// );
    ///
    /// // Use unpack_r() when:
    /// // 1. You only need R (not Q or P)
    /// // 2. Performance matters
    /// // 3. You're done with the decomposition
    /// let r = data.col_piv_qr().unpack_r();
    ///
    /// assert_eq!(r.nrows(), 3); // min(4, 3)
    /// assert_eq!(r.ncols(), 3);
    /// ```
    ///
    /// # See Also
    ///
    /// - [`r()`](Self::r) - Non-consuming version that borrows self
    /// - [`unpack()`](Self::unpack) - Get Q, R, and P all at once
    /// - [`q()`](Self::q) - Get the orthogonal matrix Q
    #[inline]
    pub fn unpack_r(self) -> OMatrix<T, DimMinimum<R, C>, C>
    where
        DefaultAllocator: Reallocator<T, R, C, DimMinimum<R, C>, C>,
    {
        let (nrows, ncols) = self.col_piv_qr.shape_generic();
        let mut res = self
            .col_piv_qr
            .resize_generic(nrows.min(ncols), ncols, T::zero());
        res.fill_lower_triangle(T::zero(), 1);
        res.set_partial_diagonal(self.diag.iter().map(|e| T::from_real(e.clone().modulus())));
        res
    }

    /// Computes the orthogonal matrix Q of this decomposition.
    ///
    /// The Q matrix is an orthogonal (or unitary for complex matrices) matrix whose columns
    /// form an orthonormal basis. In the decomposition **AP = QR**, Q represents the
    /// orthogonal transformation applied to the original matrix.
    ///
    /// # Properties of Q
    ///
    /// - **Orthonormal columns**: Each column has unit length and columns are perpendicular
    /// - **Preserves lengths**: Multiplying by Q doesn't change vector lengths
    /// - **Q^T * Q = I** (or Q^H * Q = I for complex matrices)
    ///
    /// # Returns
    ///
    /// An orthogonal matrix of size **m×min(m,n)** where the original matrix was **m×n**.
    ///
    /// # Examples
    ///
    /// ## Basic usage
    ///
    /// ```
    /// use nalgebra::Matrix3;
    ///
    /// let m = Matrix3::new(
    ///     1.0, 2.0, 3.0,
    ///     0.0, 4.0, 5.0,
    ///     0.0, 0.0, 6.0
    /// );
    ///
    /// let qr = m.col_piv_qr();
    /// let q = qr.q();
    ///
    /// println!("Q matrix:\n{}", q);
    ///
    /// // Q should have orthonormal columns
    /// let q_t_q = q.transpose() * &q;
    /// let identity = Matrix3::identity();
    /// assert!((q_t_q - identity).norm() < 1e-10);
    /// ```
    ///
    /// ## Verifying the decomposition
    ///
    /// ```
    /// use nalgebra::Matrix3;
    ///
    /// let m = Matrix3::new(
    ///     12.0, -51.0, 4.0,
    ///     6.0, 167.0, -68.0,
    ///     -4.0, 24.0, -41.0
    /// );
    ///
    /// let qr = m.col_piv_qr();
    /// let q = qr.q();
    /// let r = qr.r();
    /// let p = qr.p();
    ///
    /// // Verify: A * P = Q * R
    /// let ap = &m * p.inverse();
    /// let qr_product = q * r;
    ///
    /// assert!((ap - qr_product).norm() < 1e-10);
    /// ```
    ///
    /// ## Using Q for orthogonal transformations
    ///
    /// ```
    /// use nalgebra::{Matrix3, Vector3};
    ///
    /// let m = Matrix3::new(
    ///     1.0, 0.0, 0.0,
    ///     0.0, 1.0, 0.0,
    ///     0.0, 0.0, 1.0
    /// );
    ///
    /// let qr = m.col_piv_qr();
    /// let q = qr.q();
    ///
    /// let v = Vector3::new(1.0, 2.0, 3.0);
    /// let transformed = q * v;
    ///
    /// // Q preserves vector length (isometry)
    /// assert!((transformed.norm() - v.norm()).abs() < 1e-10);
    /// ```
    ///
    /// ## Rectangular matrix (overdetermined system)
    ///
    /// ```
    /// use nalgebra::Matrix3x2;
    ///
    /// // 3 rows, 2 columns
    /// let m = Matrix3x2::new(
    ///     1.0, 2.0,
    ///     3.0, 4.0,
    ///     5.0, 6.0
    /// );
    ///
    /// let qr = m.col_piv_qr();
    /// let q = qr.q();
    ///
    /// // Q is 3x2 for a 3x2 input matrix
    /// assert_eq!(q.nrows(), 3);
    /// assert_eq!(q.ncols(), 2);
    ///
    /// // Columns of Q are orthonormal
    /// assert!((q.column(0).norm() - 1.0).abs() < 1e-10);
    /// assert!((q.column(1).norm() - 1.0).abs() < 1e-10);
    /// assert!(q.column(0).dot(&q.column(1)).abs() < 1e-10);
    /// ```
    ///
    /// # See Also
    ///
    /// - [`r()`](Self::r) - Get the upper triangular matrix R
    /// - [`p()`](Self::p) - Get the column permutation
    /// - [`unpack()`](Self::unpack) - Get Q, R, and P all at once
    /// - [`q_tr_mul()`](Self::q_tr_mul) - Multiply by Q^T without forming Q explicitly
    #[must_use]
    pub fn q(&self) -> OMatrix<T, R, DimMinimum<R, C>>
    where
        DefaultAllocator: Allocator<R, DimMinimum<R, C>>,
    {
        let (nrows, ncols) = self.col_piv_qr.shape_generic();

        // NOTE: we could build the identity matrix and call q_mul on it.
        // Instead we don't so that we take in account the matrix sparseness.
        let mut res = Matrix::identity_generic(nrows, nrows.min(ncols));
        let dim = self.diag.len();

        for i in (0..dim).rev() {
            let axis = self.col_piv_qr.view_range(i.., i);
            // TODO: sometimes, the axis might have a zero magnitude.
            let refl = Reflection::new(Unit::new_unchecked(axis), T::zero());

            let mut res_rows = res.view_range_mut(i.., i..);
            refl.reflect_with_sign(&mut res_rows, self.diag[i].clone().signum());
        }

        res
    }
    /// Retrieves the column permutation of this decomposition.
    ///
    /// The permutation matrix P represents how the columns were reordered during the
    /// decomposition process. Column pivoting selects columns in order of decreasing norm,
    /// which improves numerical stability and helps identify rank deficiency.
    ///
    /// In the decomposition **AP = QR**, P records which column of A ended up in each
    /// position after pivoting.
    ///
    /// # Returns
    ///
    /// A reference to the permutation sequence that can be used to permute other matrices
    /// or to recover the original column ordering.
    ///
    /// # Examples
    ///
    /// ## Basic usage
    ///
    /// ```
    /// use nalgebra::Matrix3;
    ///
    /// let m = Matrix3::new(
    ///     1.0, 100.0, 2.0,
    ///     3.0, 200.0, 4.0,
    ///     5.0, 300.0, 6.0
    /// );
    ///
    /// let qr = m.col_piv_qr();
    /// let p = qr.p();
    ///
    /// // The permutation will likely put the largest column (second one) first
    /// println!("Permutation: {:?}", p);
    /// ```
    ///
    /// ## Understanding the permutation
    ///
    /// ```
    /// use nalgebra::Matrix3;
    ///
    /// let m = Matrix3::new(
    ///     0.001, 1.0, 0.002,
    ///     0.003, 2.0, 0.004,
    ///     0.005, 3.0, 0.006
    /// );
    ///
    /// let qr = m.col_piv_qr();
    /// let q = qr.q();
    /// let r = qr.r();
    /// let p = qr.p();
    ///
    /// // Verify the decomposition: A * P = Q * R
    /// let a_permuted = &m * p.inverse();
    /// let q_times_r = q * r;
    ///
    /// assert!((a_permuted - q_times_r).norm() < 1e-10);
    /// ```
    ///
    /// ## Using permutation to reorder results
    ///
    /// ```
    /// use nalgebra::{Matrix3, Vector3};
    ///
    /// let m = Matrix3::new(
    ///     2.0, 1.0, 3.0,
    ///     4.0, 5.0, 6.0,
    ///     7.0, 8.0, 9.0
    /// );
    ///
    /// let qr = m.col_piv_qr();
    /// let p = qr.p();
    ///
    /// // If we solve A*x = b, we get x in permuted order
    /// // We need P to restore the original order
    /// let b = Vector3::new(1.0, 2.0, 3.0);
    ///
    /// if let Some(x_permuted) = qr.solve(&b) {
    ///     // x_permuted has components in permuted order
    ///     // To get the original order: x = P * x_permuted
    ///     let x = p.inverse() * x_permuted;
    ///
    ///     // Verify: A * x = b
    ///     assert!((&m * x - b).norm() < 1e-10);
    /// }
    /// ```
    ///
    /// ## Examining which columns were selected
    ///
    /// ```
    /// use nalgebra::Matrix3;
    ///
    /// let m = Matrix3::new(
    ///     1.0, 0.0, 1.0,
    ///     0.0, 1.0, 1.0,
    ///     0.0, 0.0, 0.0
    /// );
    ///
    /// let qr = m.col_piv_qr();
    /// let p = qr.p();
    /// let r = qr.r();
    ///
    /// // The permutation moved columns to highlight linear dependencies
    /// // Check the diagonal of R to see the effect
    /// println!("Permutation moved columns to: {:?}", p);
    /// println!("R diagonal: [{}, {}, {}]", r[(0,0)], r[(1,1)], r[(2,2)]);
    /// ```
    ///
    /// # See Also
    ///
    /// - [`q()`](Self::q) - Get the orthogonal matrix Q
    /// - [`r()`](Self::r) - Get the upper triangular matrix R
    /// - [`unpack()`](Self::unpack) - Get Q, R, and P all at once
    /// - [`solve()`](Self::solve) - Solve linear systems (handles permutation automatically)
    #[inline]
    #[must_use]
    pub const fn p(&self) -> &PermutationSequence<DimMinimum<R, C>> {
        &self.p
    }

    /// Unpacks this decomposition into its three components: Q, R, and P.
    ///
    /// This method extracts all three components of the ColPivQR decomposition in one call,
    /// consuming the decomposition structure. It returns the orthogonal matrix Q, the upper
    /// triangular matrix R, and the permutation P.
    ///
    /// The decomposition satisfies: **AP = QR**
    ///
    /// # Returns
    ///
    /// A tuple `(Q, R, P)` where:
    /// - **Q**: Orthogonal matrix of size **m×min(m,n)**
    /// - **R**: Upper trapezoidal matrix of size **min(m,n)×n**
    /// - **P**: Column permutation
    ///
    /// # Examples
    ///
    /// ## Basic usage
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
    /// let qr = m.col_piv_qr();
    ///
    /// // Extract all three components at once
    /// let (q, r, p) = qr.unpack();
    ///
    /// println!("Q:\n{}", q);
    /// println!("R:\n{}", r);
    /// println!("P:\n{:?}", p);
    ///
    /// // Verify the decomposition
    /// let reconstructed = q * r * p.clone().inverse();
    /// assert!((m - reconstructed).norm() < 1e-10);
    /// ```
    ///
    /// ## Verifying orthogonality and decomposition
    ///
    /// ```
    /// use nalgebra::Matrix3;
    ///
    /// let m = Matrix3::new(
    ///     12.0, -51.0, 4.0,
    ///     6.0, 167.0, -68.0,
    ///     -4.0, 24.0, -41.0
    /// );
    ///
    /// let (q, r, p) = m.col_piv_qr().unpack();
    ///
    /// // Check that Q is orthogonal: Q^T * Q = I
    /// let q_t_q = q.transpose() * &q;
    /// let identity = Matrix3::identity();
    /// assert!((q_t_q - identity).norm() < 1e-10);
    ///
    /// // Check the decomposition: A * P = Q * R
    /// let a_p = &m * p.inverse();
    /// let q_r = q * r;
    /// assert!((a_p - q_r).norm() < 1e-10);
    /// ```
    ///
    /// ## Working with rectangular matrices
    ///
    /// ```
    /// use nalgebra::Matrix3x2;
    ///
    /// let m = Matrix3x2::new(
    ///     1.0, 2.0,
    ///     3.0, 4.0,
    ///     5.0, 6.0
    /// );
    ///
    /// let (q, r, p) = m.col_piv_qr().unpack();
    ///
    /// // For a 3x2 matrix:
    /// // Q is 3x2
    /// assert_eq!(q.nrows(), 3);
    /// assert_eq!(q.ncols(), 2);
    ///
    /// // R is 2x2
    /// assert_eq!(r.nrows(), 2);
    /// assert_eq!(r.ncols(), 2);
    ///
    /// // Verify decomposition
    /// let ap = m * p.inverse();
    /// let qr = q * r;
    /// assert!((ap - qr).norm() < 1e-10);
    /// ```
    ///
    /// ## Analyzing a rank-deficient matrix
    ///
    /// ```
    /// use nalgebra::Matrix3;
    ///
    /// // Create a rank-2 matrix (third column = first + second)
    /// let m = Matrix3::new(
    ///     1.0, 0.0, 1.0,
    ///     0.0, 1.0, 1.0,
    ///     1.0, 1.0, 2.0
    /// );
    ///
    /// let (q, r, p) = m.col_piv_qr().unpack();
    ///
    /// // The diagonal of R reveals the rank
    /// let tolerance = 1e-10;
    /// let mut rank = 0;
    /// for i in 0..3 {
    ///     if r[(i, i)].abs() > tolerance {
    ///         rank += 1;
    ///     }
    /// }
    ///
    /// println!("Matrix rank: {}", rank); // Should be 2
    /// assert_eq!(rank, 2);
    /// ```
    ///
    /// # See Also
    ///
    /// - [`q()`](Self::q) - Get only Q (borrows self)
    /// - [`r()`](Self::r) - Get only R (borrows self)
    /// - [`p()`](Self::p) - Get only P (borrows self)
    /// - [`unpack_r()`](Self::unpack_r) - Get only R (consumes self)
    pub fn unpack(
        self,
    ) -> (
        OMatrix<T, R, DimMinimum<R, C>>,
        OMatrix<T, DimMinimum<R, C>, C>,
        PermutationSequence<DimMinimum<R, C>>,
    )
    where
        DimMinimum<R, C>: DimMin<C, Output = DimMinimum<R, C>>,
        DefaultAllocator: Allocator<R, DimMinimum<R, C>>
            + Reallocator<T, R, C, DimMinimum<R, C>, C>
            + Allocator<DimMinimum<R, C>>,
    {
        (self.q(), self.r(), self.p)
    }

    #[doc(hidden)]
    pub const fn col_piv_qr_internal(&self) -> &OMatrix<T, R, C> {
        &self.col_piv_qr
    }

    /// Multiplies the provided matrix by the transpose of the Q matrix of this decomposition.
    ///
    /// This method computes **Q^T * rhs** (or **Q^H * rhs** for complex matrices) without
    /// explicitly forming the Q matrix. This is more efficient than first computing Q
    /// with [`q()`](Self::q) and then multiplying, especially for large matrices.
    ///
    /// The operation modifies `rhs` in place, replacing it with **Q^T * rhs**.
    ///
    /// # Why Use This?
    ///
    /// - **Memory efficient**: Doesn't allocate the full Q matrix
    /// - **Faster**: Uses the implicit Householder representation
    /// - **Common operation**: Q^T multiplication appears in many algorithms (solving systems, least squares)
    ///
    /// # Arguments
    ///
    /// * `rhs` - The matrix to multiply (modified in place)
    ///
    /// # Examples
    ///
    /// ## Basic usage
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
    /// let qr = m.col_piv_qr();
    /// let mut v = Vector3::new(1.0, 2.0, 3.0);
    ///
    /// // Compute Q^T * v efficiently
    /// qr.q_tr_mul(&mut v);
    ///
    /// println!("Q^T * v = {}", v);
    /// ```
    ///
    /// ## Comparing with explicit Q multiplication
    ///
    /// ```
    /// use nalgebra::{Matrix3, Vector3};
    ///
    /// let m = Matrix3::new(
    ///     2.0, 1.0, 0.0,
    ///     1.0, 2.0, 1.0,
    ///     0.0, 1.0, 2.0
    /// );
    ///
    /// let qr = m.col_piv_qr();
    /// let v = Vector3::new(1.0, 2.0, 3.0);
    ///
    /// // Method 1: Using q_tr_mul (efficient)
    /// let mut result1 = v.clone();
    /// qr.q_tr_mul(&mut result1);
    ///
    /// // Method 2: Explicit Q matrix (less efficient)
    /// let q = qr.q();
    /// let result2 = q.transpose() * v;
    ///
    /// // Both methods produce the same result
    /// assert!((result1 - result2).norm() < 1e-10);
    /// ```
    ///
    /// ## Using in solving systems (internal step)
    ///
    /// ```
    /// use nalgebra::{Matrix3, Vector3};
    ///
    /// let m = Matrix3::new(
    ///     4.0, 1.0, 2.0,
    ///     1.0, 5.0, 3.0,
    ///     2.0, 3.0, 6.0
    /// );
    ///
    /// let qr = m.col_piv_qr();
    /// let mut b = Vector3::new(7.0, 9.0, 11.0);
    ///
    /// // First step of solving A*x = b is computing Q^T * b
    /// qr.q_tr_mul(&mut b);
    ///
    /// // Now b = Q^T * original_b
    /// // Next we would solve R*x = b (upper triangular system)
    /// println!("Q^T * b = {}", b);
    /// ```
    ///
    /// ## Working with matrices (not just vectors)
    ///
    /// ```
    /// use nalgebra::Matrix3;
    ///
    /// let m = Matrix3::new(
    ///     1.0, 0.0, 0.0,
    ///     0.0, 1.0, 0.0,
    ///     0.0, 0.0, 1.0
    /// );
    ///
    /// let qr = m.col_piv_qr();
    ///
    /// // Can multiply a matrix, not just a vector
    /// let mut rhs = Matrix3::new(
    ///     1.0, 2.0, 3.0,
    ///     4.0, 5.0, 6.0,
    ///     7.0, 8.0, 9.0
    /// );
    ///
    /// qr.q_tr_mul(&mut rhs);
    /// println!("Q^T * rhs =\n{}", rhs);
    /// ```
    ///
    /// ## Least squares application
    ///
    /// ```
    /// use nalgebra::{Matrix3x2, Vector3};
    ///
    /// // Overdetermined system: 3 equations, 2 unknowns
    /// let a = Matrix3x2::new(
    ///     1.0, 1.0,
    ///     2.0, 1.0,
    ///     3.0, 1.0
    /// );
    ///
    /// let b = Vector3::new(2.0, 3.0, 4.0);
    ///
    /// let qr = a.col_piv_qr();
    /// let mut b_transformed = b.clone();
    ///
    /// // Transform b: Q^T * b
    /// // This is a key step in least squares solution
    /// qr.q_tr_mul(&mut b_transformed);
    ///
    /// println!("Transformed b for least squares: {}", b_transformed);
    /// ```
    ///
    /// # See Also
    ///
    /// - [`q()`](Self::q) - Get the explicit Q matrix
    /// - [`solve()`](Self::solve) - Solve linear systems (uses q_tr_mul internally)
    /// - [`solve_mut()`](Self::solve_mut) - In-place version of solve
    pub fn q_tr_mul<R2: Dim, C2: Dim, S2>(&self, rhs: &mut Matrix<T, R2, C2, S2>)
    where
        S2: StorageMut<T, R2, C2>,
    {
        let dim = self.diag.len();

        for i in 0..dim {
            let axis = self.col_piv_qr.view_range(i.., i);
            let refl = Reflection::new(Unit::new_unchecked(axis), T::zero());

            let mut rhs_rows = rhs.rows_range_mut(i..);
            refl.reflect_with_sign(&mut rhs_rows, self.diag[i].clone().signum().conjugate());
        }
    }
}

impl<T: ComplexField, D: DimMin<D, Output = D>> ColPivQR<T, D, D>
where
    DefaultAllocator: Allocator<D, D> + Allocator<D> + Allocator<DimMinimum<D, D>>,
{
    /// Solves the linear system **Ax = b** using this ColPivQR decomposition.
    ///
    /// This method finds the solution vector **x** to the equation **Ax = b**, where **A**
    /// is the matrix that was decomposed. The column pivoting improves numerical stability
    /// and allows the method to detect singular (non-invertible) matrices reliably.
    ///
    /// # How It Works
    ///
    /// Given **AP = QR**, the system **Ax = b** is solved by:
    /// 1. Computing **Q^T * b**
    /// 2. Solving the upper triangular system **Ry = Q^T * b**
    /// 3. Applying the inverse permutation: **x = P^(-1) * y**
    ///
    /// # Arguments
    ///
    /// * `b` - The right-hand side vector or matrix (must have same number of rows as A)
    ///
    /// # Returns
    ///
    /// - `Some(x)` if the system has a unique solution
    /// - `None` if the matrix is singular (not invertible)
    ///
    /// # Examples
    ///
    /// ## Solving a simple linear system
    ///
    /// ```
    /// use nalgebra::{Matrix3, Vector3};
    ///
    /// let a = Matrix3::new(
    ///     2.0, 1.0, 0.0,
    ///     1.0, 2.0, 1.0,
    ///     0.0, 1.0, 2.0
    /// );
    ///
    /// let b = Vector3::new(1.0, 2.0, 3.0);
    ///
    /// let qr = a.col_piv_qr();
    ///
    /// if let Some(x) = qr.solve(&b) {
    ///     println!("Solution: {}", x);
    ///
    ///     // Verify: A * x = b
    ///     let result = a * x;
    ///     assert!((result - b).norm() < 1e-10);
    /// }
    /// ```
    ///
    /// ## Detecting singular matrices
    ///
    /// ```
    /// use nalgebra::{Matrix3, Vector3};
    ///
    /// // Singular matrix (third row = first row + second row)
    /// let a = Matrix3::new(
    ///     1.0, 0.0, 0.0,
    ///     0.0, 1.0, 0.0,
    ///     1.0, 1.0, 0.0
    /// );
    ///
    /// let b = Vector3::new(1.0, 2.0, 3.0);
    ///
    /// let qr = a.col_piv_qr();
    ///
    /// // This will return None because the matrix is singular
    /// match qr.solve(&b) {
    ///     Some(x) => println!("Solution found: {}", x),
    ///     None => println!("Matrix is singular - no unique solution exists")
    /// }
    /// ```
    ///
    /// ## Solving multiple right-hand sides
    ///
    /// ```
    /// use nalgebra::{Matrix3, Matrix3x2};
    ///
    /// let a = Matrix3::new(
    ///     4.0, 1.0, 2.0,
    ///     1.0, 5.0, 3.0,
    ///     2.0, 3.0, 6.0
    /// );
    ///
    /// // Multiple right-hand sides (each column is a separate system)
    /// let b = Matrix3x2::new(
    ///     1.0, 4.0,
    ///     2.0, 5.0,
    ///     3.0, 6.0
    /// );
    ///
    /// let qr = a.col_piv_qr();
    ///
    /// if let Some(x) = qr.solve(&b) {
    ///     // x has 2 columns, one solution for each RHS
    ///     println!("Solutions:\n{}", x);
    ///
    ///     // Verify each solution
    ///     assert!((a * x - b).norm() < 1e-10);
    /// }
    /// ```
    ///
    /// ## Well-conditioned system
    ///
    /// ```
    /// use nalgebra::{Matrix3, Vector3};
    ///
    /// // Well-conditioned matrix
    /// let a = Matrix3::new(
    ///     10.0, 1.0, 1.0,
    ///     1.0, 10.0, 1.0,
    ///     1.0, 1.0, 10.0
    /// );
    ///
    /// let b = Vector3::new(12.0, 12.0, 12.0);
    ///
    /// let qr = a.col_piv_qr();
    /// let x = qr.solve(&b).expect("Should have a solution");
    ///
    /// println!("Solution: {}", x);
    /// assert!((a * x - b).norm() < 1e-10);
    /// ```
    ///
    /// ## Comparing with direct solve
    ///
    /// ```
    /// use nalgebra::{Matrix3, Vector3};
    ///
    /// let a = Matrix3::new(
    ///     3.0, 1.0, 4.0,
    ///     1.0, 5.0, 9.0,
    ///     2.0, 6.0, 5.0
    /// );
    ///
    /// let b = Vector3::new(1.0, 2.0, 3.0);
    ///
    /// // Using ColPivQR (more stable for ill-conditioned matrices)
    /// let qr = a.col_piv_qr();
    /// let x1 = qr.solve(&b).unwrap();
    ///
    /// // Both should give the same answer for well-conditioned matrices
    /// println!("ColPivQR solution: {}", x1);
    /// ```
    ///
    /// # See Also
    ///
    /// - [`solve_mut()`](Self::solve_mut) - In-place version (more efficient)
    /// - [`try_inverse()`](Self::try_inverse) - Compute matrix inverse
    /// - [`is_invertible()`](Self::is_invertible) - Check if matrix is invertible
    /// - [`q_tr_mul()`](Self::q_tr_mul) - Part of the solving process
    #[must_use = "Did you mean to use solve_mut()?"]
    pub fn solve<R2: Dim, C2: Dim, S2>(
        &self,
        b: &Matrix<T, R2, C2, S2>,
    ) -> Option<OMatrix<T, R2, C2>>
    where
        S2: StorageMut<T, R2, C2>,
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

    /// Solves the linear system **Ax = b** in-place, overwriting **b** with the solution.
    ///
    /// This is a more memory-efficient version of [`solve()`](Self::solve) that modifies
    /// the right-hand side vector/matrix in place instead of allocating new storage.
    ///
    /// # Arguments
    ///
    /// * `b` - On input: the right-hand side. On output: the solution **x** (if successful)
    ///
    /// # Returns
    ///
    /// - `true` if the system was solved successfully
    /// - `false` if the matrix is singular (not invertible)
    ///
    /// # Warning
    ///
    /// If this method returns `false`, the contents of `b` will be corrupted and should
    /// not be used. Always check the return value before using the result.
    ///
    /// # Examples
    ///
    /// ## Basic usage
    ///
    /// ```
    /// use nalgebra::{Matrix3, Vector3};
    ///
    /// let a = Matrix3::new(
    ///     2.0, 1.0, 0.0,
    ///     1.0, 2.0, 1.0,
    ///     0.0, 1.0, 2.0
    /// );
    ///
    /// let mut b = Vector3::new(1.0, 2.0, 3.0);
    /// let b_original = b.clone();
    ///
    /// let qr = a.col_piv_qr();
    ///
    /// if qr.solve_mut(&mut b) {
    ///     // b now contains the solution x
    ///     println!("Solution: {}", b);
    ///
    ///     // Verify: A * x = b_original
    ///     assert!((a * b - b_original).norm() < 1e-10);
    /// }
    /// ```
    ///
    /// ## Handling singular matrices
    ///
    /// ```
    /// use nalgebra::{Matrix3, Vector3};
    ///
    /// // Singular matrix
    /// let a = Matrix3::new(
    ///     1.0, 2.0, 3.0,
    ///     2.0, 4.0, 6.0,  // Second row = 2 * first row
    ///     4.0, 5.0, 6.0
    /// );
    ///
    /// let mut b = Vector3::new(1.0, 2.0, 3.0);
    ///
    /// let qr = a.col_piv_qr();
    ///
    /// if qr.solve_mut(&mut b) {
    ///     println!("Solution: {}", b);
    /// } else {
    ///     println!("Matrix is singular - cannot solve");
    ///     // DO NOT use b here - it contains garbage!
    /// }
    /// ```
    ///
    /// ## Multiple right-hand sides
    ///
    /// ```
    /// use nalgebra::{Matrix3, Matrix3x2};
    ///
    /// let a = Matrix3::new(
    ///     3.0, 1.0, 2.0,
    ///     1.0, 4.0, 1.0,
    ///     2.0, 1.0, 3.0
    /// );
    ///
    /// let mut b = Matrix3x2::new(
    ///     1.0, 4.0,
    ///     2.0, 5.0,
    ///     3.0, 6.0
    /// );
    ///
    /// let b_original = b.clone();
    ///
    /// let qr = a.col_piv_qr();
    ///
    /// if qr.solve_mut(&mut b) {
    ///     // Each column of b now contains a solution
    ///     println!("Solutions:\n{}", b);
    ///     assert!((a * b - b_original).norm() < 1e-10);
    /// }
    /// ```
    ///
    /// ## Memory-efficient solving
    ///
    /// ```
    /// use nalgebra::{Matrix3, Vector3};
    ///
    /// let a = Matrix3::new(
    ///     5.0, 2.0, 1.0,
    ///     2.0, 6.0, 2.0,
    ///     1.0, 2.0, 7.0
    /// );
    ///
    /// let qr = a.col_piv_qr();
    ///
    /// // Solve multiple systems efficiently by reusing the decomposition
    /// for i in 1..=3 {
    ///     let mut b = Vector3::new(i as f64, i as f64 * 2.0, i as f64 * 3.0);
    ///     let b_original = b.clone();
    ///
    ///     if qr.solve_mut(&mut b) {
    ///         println!("Solution {}: {}", i, b);
    ///         assert!((a * b - b_original).norm() < 1e-10);
    ///     }
    /// }
    /// ```
    ///
    /// ## Comparing with solve()
    ///
    /// ```
    /// use nalgebra::{Matrix3, Vector3};
    ///
    /// let a = Matrix3::new(
    ///     4.0, 1.0, 2.0,
    ///     1.0, 5.0, 3.0,
    ///     2.0, 3.0, 6.0
    /// );
    ///
    /// let b = Vector3::new(1.0, 2.0, 3.0);
    ///
    /// let qr = a.col_piv_qr();
    ///
    /// // Method 1: solve() - allocates new vector
    /// let x1 = qr.solve(&b).unwrap();
    ///
    /// // Method 2: solve_mut() - more efficient, no allocation
    /// let mut b_mut = b.clone();
    /// assert!(qr.solve_mut(&mut b_mut));
    /// let x2 = b_mut;
    ///
    /// // Both produce the same result
    /// assert!((x1 - x2).norm() < 1e-10);
    /// ```
    ///
    /// # See Also
    ///
    /// - [`solve()`](Self::solve) - Non-mutating version that returns Option
    /// - [`try_inverse()`](Self::try_inverse) - Compute matrix inverse
    /// - [`is_invertible()`](Self::is_invertible) - Check if matrix is invertible first
    pub fn solve_mut<R2: Dim, C2: Dim, S2>(&self, b: &mut Matrix<T, R2, C2, S2>) -> bool
    where
        S2: StorageMut<T, R2, C2>,
        ShapeConstraint: SameNumberOfRows<R2, D>,
    {
        assert_eq!(
            self.col_piv_qr.nrows(),
            b.nrows(),
            "ColPivQR solve matrix dimension mismatch."
        );
        assert!(
            self.col_piv_qr.is_square(),
            "ColPivQR solve: unable to solve a non-square system."
        );

        self.q_tr_mul(b);
        let solved = self.solve_upper_triangular_mut(b);
        self.p.inv_permute_rows(b);

        solved
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
        let dim = self.col_piv_qr.nrows();

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
                    .axpy(-coeff, &self.col_piv_qr.view_range(..i, i), T::one());
            }
        }

        true
    }

    /// Computes the inverse of the decomposed matrix, if it exists.
    ///
    /// This method computes **A^(-1)** where **A** is the original matrix that was
    /// decomposed. The column pivoting improves numerical stability compared to
    /// standard matrix inversion methods, making it more reliable for matrices
    /// that are close to singular.
    ///
    /// # Returns
    ///
    /// - `Some(A_inv)` if the matrix is invertible
    /// - `None` if the matrix is singular (determinant is zero)
    ///
    /// # Examples
    ///
    /// ## Basic usage
    ///
    /// ```
    /// use nalgebra::Matrix3;
    ///
    /// let a = Matrix3::new(
    ///     2.0, 1.0, 0.0,
    ///     1.0, 2.0, 1.0,
    ///     0.0, 1.0, 2.0
    /// );
    ///
    /// let qr = a.col_piv_qr();
    ///
    /// if let Some(a_inv) = qr.try_inverse() {
    ///     println!("Inverse:\n{}", a_inv);
    ///
    ///     // Verify: A * A^(-1) = I
    ///     let identity = &a * &a_inv;
    ///     let expected = Matrix3::identity();
    ///     assert!((identity - expected).norm() < 1e-10);
    ///
    ///     // Verify: A^(-1) * A = I
    ///     let identity2 = a_inv * a;
    ///     assert!((identity2 - expected).norm() < 1e-10);
    /// }
    /// ```
    ///
    /// ## Detecting singular matrices
    ///
    /// ```
    /// use nalgebra::Matrix3;
    ///
    /// // Singular matrix (third row = first + second row)
    /// let a = Matrix3::new(
    ///     1.0, 0.0, 0.0,
    ///     0.0, 1.0, 0.0,
    ///     1.0, 1.0, 0.0
    /// );
    ///
    /// let qr = a.col_piv_qr();
    ///
    /// match qr.try_inverse() {
    ///     Some(inv) => println!("Inverse found:\n{}", inv),
    ///     None => println!("Matrix is singular - no inverse exists")
    /// }
    /// ```
    ///
    /// ## Well-conditioned matrix
    ///
    /// ```
    /// use nalgebra::Matrix3;
    ///
    /// // Diagonal-dominant matrix (well-conditioned)
    /// let a = Matrix3::new(
    ///     10.0, 1.0, 1.0,
    ///     1.0, 10.0, 1.0,
    ///     1.0, 1.0, 10.0
    /// );
    ///
    /// let qr = a.col_piv_qr();
    /// let a_inv = qr.try_inverse().expect("Should be invertible");
    ///
    /// // Check the result
    /// let identity = a * a_inv;
    /// let expected = Matrix3::identity();
    /// assert!((identity - expected).norm() < 1e-10);
    /// ```
    ///
    /// ## Using inverse to solve systems
    ///
    /// ```
    /// use nalgebra::{Matrix3, Vector3};
    ///
    /// let a = Matrix3::new(
    ///     3.0, 1.0, 2.0,
    ///     1.0, 4.0, 1.0,
    ///     2.0, 1.0, 3.0
    /// );
    ///
    /// let qr = a.col_piv_qr();
    ///
    /// if let Some(a_inv) = qr.try_inverse() {
    ///     // Solve A*x = b using the inverse
    ///     let b = Vector3::new(1.0, 2.0, 3.0);
    ///     let x = a_inv * b;
    ///
    ///     // Verify the solution
    ///     assert!((a * x - b).norm() < 1e-10);
    ///
    ///     // Note: For solving systems, solve() is usually more efficient
    ///     // than computing the full inverse
    /// }
    /// ```
    ///
    /// ## Checking before inverting
    ///
    /// ```
    /// use nalgebra::Matrix3;
    ///
    /// let a = Matrix3::new(
    ///     1.0, 2.0, 3.0,
    ///     4.0, 5.0, 6.0,
    ///     7.0, 8.0, 9.0
    /// );
    ///
    /// let qr = a.col_piv_qr();
    ///
    /// // Check invertibility first
    /// if qr.is_invertible() {
    ///     let a_inv = qr.try_inverse().unwrap();
    ///     println!("Inverse:\n{}", a_inv);
    /// } else {
    ///     println!("Matrix is not invertible");
    /// }
    /// ```
    ///
    /// ## Comparing with direct inversion
    ///
    /// ```
    /// use nalgebra::Matrix3;
    ///
    /// let a = Matrix3::new(
    ///     4.0, 7.0, 2.0,
    ///     3.0, 6.0, 1.0,
    ///     2.0, 5.0, 3.0
    /// );
    ///
    /// // Using ColPivQR (more numerically stable)
    /// let qr = a.col_piv_qr();
    /// let inv1 = qr.try_inverse().unwrap();
    ///
    /// // Using direct method
    /// let inv2 = a.try_inverse().unwrap();
    ///
    /// // Both should give similar results for well-conditioned matrices
    /// assert!((inv1 - inv2).norm() < 1e-10);
    /// ```
    ///
    /// # Performance Note
    ///
    /// Computing the inverse is generally expensive (O(n³)). If you only need to
    /// solve **Ax = b** for one or a few right-hand sides, use [`solve()`](Self::solve)
    /// or [`solve_mut()`](Self::solve_mut) instead, which are more efficient.
    ///
    /// # See Also
    ///
    /// - [`is_invertible()`](Self::is_invertible) - Check if matrix is invertible without computing inverse
    /// - [`solve()`](Self::solve) - Solve linear systems (more efficient than using inverse)
    /// - [`determinant()`](Self::determinant) - Compute the determinant
    #[must_use]
    pub fn try_inverse(&self) -> Option<OMatrix<T, D, D>> {
        assert!(
            self.col_piv_qr.is_square(),
            "ColPivQR inverse: unable to compute the inverse of a non-square matrix."
        );

        // TODO: is there a less naive method ?
        let (nrows, ncols) = self.col_piv_qr.shape_generic();
        let mut res = OMatrix::identity_generic(nrows, ncols);

        if self.solve_mut(&mut res) {
            Some(res)
        } else {
            None
        }
    }

    /// Checks whether the decomposed matrix is invertible.
    ///
    /// This method determines if the original matrix **A** has an inverse by checking
    /// if all diagonal elements of the **R** matrix in the QR decomposition are non-zero.
    /// Thanks to column pivoting, this test is numerically reliable.
    ///
    /// A matrix is invertible if and only if:
    /// - It is square
    /// - It has full rank (no zero diagonal elements in R)
    /// - Its determinant is non-zero
    ///
    /// # Returns
    ///
    /// - `true` if the matrix is invertible
    /// - `false` if the matrix is singular (not invertible)
    ///
    /// # Examples
    ///
    /// ## Invertible matrix
    ///
    /// ```
    /// use nalgebra::Matrix3;
    ///
    /// let a = Matrix3::new(
    ///     2.0, 1.0, 0.0,
    ///     1.0, 2.0, 1.0,
    ///     0.0, 1.0, 2.0
    /// );
    ///
    /// let qr = a.col_piv_qr();
    ///
    /// if qr.is_invertible() {
    ///     println!("Matrix is invertible");
    ///     let inv = qr.try_inverse().unwrap();
    ///     println!("Inverse:\n{}", inv);
    /// }
    /// ```
    ///
    /// ## Singular matrix
    ///
    /// ```
    /// use nalgebra::Matrix3;
    ///
    /// // Singular matrix (rank-deficient)
    /// let a = Matrix3::new(
    ///     1.0, 2.0, 3.0,
    ///     2.0, 4.0, 6.0,  // Second row = 2 * first row
    ///     3.0, 6.0, 9.0   // Third row = 3 * first row
    /// );
    ///
    /// let qr = a.col_piv_qr();
    ///
    /// assert!(!qr.is_invertible());
    /// println!("Matrix is singular - cannot be inverted");
    /// ```
    ///
    /// ## Checking before operations
    ///
    /// ```
    /// use nalgebra::{Matrix3, Vector3};
    ///
    /// let a = Matrix3::new(
    ///     3.0, 1.0, 2.0,
    ///     1.0, 4.0, 1.0,
    ///     2.0, 1.0, 3.0
    /// );
    ///
    /// let b = Vector3::new(1.0, 2.0, 3.0);
    ///
    /// let qr = a.col_piv_qr();
    ///
    /// // Check invertibility before attempting to solve
    /// if qr.is_invertible() {
    ///     let x = qr.solve(&b).unwrap();
    ///     println!("Solution: {}", x);
    /// } else {
    ///     println!("Cannot solve - matrix is singular");
    /// }
    /// ```
    ///
    /// ## Comparing different matrices
    ///
    /// ```
    /// use nalgebra::Matrix3;
    ///
    /// let matrices = vec![
    ///     // Invertible: identity matrix
    ///     Matrix3::identity(),
    ///
    ///     // Invertible: diagonal matrix
    ///     Matrix3::new(
    ///         2.0, 0.0, 0.0,
    ///         0.0, 3.0, 0.0,
    ///         0.0, 0.0, 4.0
    ///     ),
    ///
    ///     // Singular: zero row
    ///     Matrix3::new(
    ///         1.0, 2.0, 3.0,
    ///         0.0, 0.0, 0.0,
    ///         4.0, 5.0, 6.0
    ///     ),
    /// ];
    ///
    /// for (i, m) in matrices.iter().enumerate() {
    ///     let qr = m.col_piv_qr();
    ///     println!("Matrix {}: {}", i, if qr.is_invertible() {
    ///         "invertible"
    ///     } else {
    ///         "singular"
    ///     });
    /// }
    /// ```
    ///
    /// ## Near-singular matrix
    ///
    /// ```
    /// use nalgebra::Matrix3;
    ///
    /// // Nearly singular matrix (small but non-zero determinant)
    /// let epsilon = 1e-15;
    /// let a = Matrix3::new(
    ///     1.0, 0.0, 0.0,
    ///     0.0, 1.0, 0.0,
    ///     0.0, 0.0, epsilon
    /// );
    ///
    /// let qr = a.col_piv_qr();
    ///
    /// // Technically invertible, but may be numerically problematic
    /// if qr.is_invertible() {
    ///     println!("Matrix is invertible (but nearly singular)");
    ///     println!("Determinant: {}", qr.determinant());
    /// }
    /// ```
    ///
    /// ## Using with determinant
    ///
    /// ```
    /// use nalgebra::Matrix3;
    ///
    /// let a = Matrix3::new(
    ///     1.0, 2.0, 3.0,
    ///     0.0, 1.0, 4.0,
    ///     5.0, 6.0, 0.0
    /// );
    ///
    /// let qr = a.col_piv_qr();
    ///
    /// // These should be consistent
    /// let is_inv = qr.is_invertible();
    /// let det = qr.determinant();
    ///
    /// assert_eq!(is_inv, !det.is_zero());
    /// println!("Invertible: {}, Determinant: {}", is_inv, det);
    /// ```
    ///
    /// # See Also
    ///
    /// - [`try_inverse()`](Self::try_inverse) - Compute the inverse if it exists
    /// - [`determinant()`](Self::determinant) - Compute the determinant
    /// - [`solve()`](Self::solve) - Solve linear systems
    #[must_use]
    pub fn is_invertible(&self) -> bool {
        assert!(
            self.col_piv_qr.is_square(),
            "ColPivQR: unable to test the invertibility of a non-square matrix."
        );

        for i in 0..self.diag.len() {
            if self.diag[i].is_zero() {
                return false;
            }
        }

        true
    }

    /// Computes the determinant of the decomposed matrix.
    ///
    /// The determinant is a scalar value that provides important information about the matrix:
    /// - **det(A) = 0**: Matrix is singular (not invertible)
    /// - **det(A) ≠ 0**: Matrix is invertible
    /// - **|det(A)| < 1**: Matrix "shrinks" volumes
    /// - **|det(A)| > 1**: Matrix "expands" volumes
    ///
    /// For the ColPivQR decomposition **AP = QR**, the determinant is computed as:
    /// **det(A) = det(Q) × det(R) × det(P^(-1))** = **(product of R's diagonal) × det(P)**
    ///
    /// # Returns
    ///
    /// The determinant of the original matrix.
    ///
    /// # Examples
    ///
    /// ## Basic usage
    ///
    /// ```
    /// use nalgebra::Matrix3;
    ///
    /// let a = Matrix3::new(
    ///     1.0, 2.0, 3.0,
    ///     0.0, 1.0, 4.0,
    ///     5.0, 6.0, 0.0
    /// );
    ///
    /// let qr = a.col_piv_qr();
    /// let det = qr.determinant();
    ///
    /// println!("Determinant: {}", det);
    /// assert!((det - 1.0).abs() < 1e-10);
    /// ```
    ///
    /// ## Detecting singular matrices
    ///
    /// ```
    /// use nalgebra::Matrix3;
    ///
    /// // Singular matrix (rank-deficient)
    /// let a = Matrix3::new(
    ///     1.0, 2.0, 3.0,
    ///     2.0, 4.0, 6.0,  // Second row = 2 * first row
    ///     4.0, 5.0, 6.0
    /// );
    ///
    /// let qr = a.col_piv_qr();
    /// let det = qr.determinant();
    ///
    /// println!("Determinant: {}", det);
    /// assert!(det.abs() < 1e-10); // Should be approximately zero
    /// ```
    ///
    /// ## Identity matrix
    ///
    /// ```
    /// use nalgebra::Matrix3;
    ///
    /// let identity = Matrix3::identity();
    /// let qr = identity.col_piv_qr();
    /// let det = qr.determinant();
    ///
    /// // Identity matrix has determinant 1
    /// assert!((det - 1.0).abs() < 1e-10);
    /// ```
    ///
    /// ## Diagonal matrix
    ///
    /// ```
    /// use nalgebra::Matrix3;
    ///
    /// let a = Matrix3::new(
    ///     2.0, 0.0, 0.0,
    ///     0.0, 3.0, 0.0,
    ///     0.0, 0.0, 4.0
    /// );
    ///
    /// let qr = a.col_piv_qr();
    /// let det = qr.determinant();
    ///
    /// // For diagonal matrix, det = product of diagonal elements
    /// assert!((det - 24.0).abs() < 1e-10); // 2 * 3 * 4 = 24
    /// ```
    ///
    /// ## Relationship with invertibility
    ///
    /// ```
    /// use nalgebra::Matrix3;
    ///
    /// let matrices = vec![
    ///     // Invertible
    ///     Matrix3::new(
    ///         2.0, 1.0, 0.0,
    ///         1.0, 2.0, 1.0,
    ///         0.0, 1.0, 2.0
    ///     ),
    ///     // Singular
    ///     Matrix3::new(
    ///         1.0, 2.0, 3.0,
    ///         2.0, 4.0, 6.0,
    ///         3.0, 6.0, 9.0
    ///     ),
    /// ];
    ///
    /// for m in matrices.iter() {
    ///     let qr = m.col_piv_qr();
    ///     let det = qr.determinant();
    ///     let is_invertible = qr.is_invertible();
    ///
    ///     println!("Determinant: {}", det);
    ///     println!("Invertible: {}", is_invertible);
    ///     assert_eq!(is_invertible, !det.is_zero());
    /// }
    /// ```
    ///
    /// ## Volume scaling interpretation
    ///
    /// ```
    /// use nalgebra::Matrix3;
    ///
    /// // Scaling matrix: doubles all dimensions
    /// let scale = Matrix3::new(
    ///     2.0, 0.0, 0.0,
    ///     0.0, 2.0, 0.0,
    ///     0.0, 0.0, 2.0
    /// );
    ///
    /// let qr = scale.col_piv_qr();
    /// let det = qr.determinant();
    ///
    /// // Determinant = 8 means volumes are scaled by factor of 8
    /// // (2x in each dimension = 2³ = 8x volume)
    /// assert!((det - 8.0).abs() < 1e-10);
    /// ```
    ///
    /// ## Comparing with direct computation
    ///
    /// ```
    /// use nalgebra::Matrix3;
    ///
    /// let a = Matrix3::new(
    ///     3.0, 1.0, 4.0,
    ///     1.0, 5.0, 9.0,
    ///     2.0, 6.0, 5.0
    /// );
    ///
    /// // Using ColPivQR
    /// let qr = a.col_piv_qr();
    /// let det1 = qr.determinant();
    ///
    /// // Using direct method
    /// let det2 = a.determinant();
    ///
    /// // Both should give the same result
    /// assert!((det1 - det2).abs() < 1e-10);
    /// println!("Determinant: {}", det1);
    /// ```
    ///
    /// ## Sign of determinant
    ///
    /// ```
    /// use nalgebra::Matrix3;
    ///
    /// let a = Matrix3::new(
    ///     1.0, 0.0, 0.0,
    ///     0.0, 1.0, 0.0,
    ///     0.0, 0.0, 1.0
    /// );
    ///
    /// // Swap two rows (changes sign of determinant)
    /// let b = Matrix3::new(
    ///     0.0, 1.0, 0.0,
    ///     1.0, 0.0, 0.0,
    ///     0.0, 0.0, 1.0
    /// );
    ///
    /// let det_a = a.col_piv_qr().determinant();
    /// let det_b = b.col_piv_qr().determinant();
    ///
    /// println!("det(A) = {}, det(B) = {}", det_a, det_b);
    /// assert!((det_a - 1.0).abs() < 1e-10);
    /// assert!((det_b + 1.0).abs() < 1e-10); // Negative due to swap
    /// ```
    ///
    /// # See Also
    ///
    /// - [`is_invertible()`](Self::is_invertible) - Check if determinant is non-zero
    /// - [`try_inverse()`](Self::try_inverse) - Compute matrix inverse
    #[must_use]
    pub fn determinant(&self) -> T {
        let dim = self.col_piv_qr.nrows();
        assert!(
            self.col_piv_qr.is_square(),
            "ColPivQR determinant: unable to compute the determinant of a non-square matrix."
        );

        let mut res = T::one();
        for i in 0..dim {
            res *= unsafe { self.diag.vget_unchecked(i).clone() };
        }

        res * self.p.determinant()
    }
}
