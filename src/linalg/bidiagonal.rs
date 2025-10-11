#[cfg(feature = "serde-serialize-no-std")]
use serde::{Deserialize, Serialize};

use crate::allocator::Allocator;
use crate::base::{DefaultAllocator, Matrix, OMatrix, OVector, Unit};
use crate::dimension::{Const, Dim, DimDiff, DimMin, DimMinimum, DimSub, U1};
use simba::scalar::ComplexField;

use crate::geometry::Reflection;
use crate::linalg::householder;
use crate::num::Zero;
use std::mem::MaybeUninit;

/// The bidiagonal decomposition of a general matrix.
///
/// # What is Bidiagonal Decomposition?
///
/// Bidiagonal decomposition (also called bidiagonalization) is a matrix factorization that
/// decomposes any matrix M into the product of three matrices: `M = U * D * V^T`, where:
/// - `U` is an orthogonal matrix (satisfies U^T * U = I)
/// - `D` is a bidiagonal matrix (has non-zero values only on the main diagonal and one adjacent diagonal)
/// - `V^T` is the transpose of an orthogonal matrix V
///
/// A bidiagonal matrix looks like this (upper bidiagonal example for a 4x4 matrix):
/// ```text
/// ┌           ┐
/// │ d₁ e₁ 0  0│
/// │ 0  d₂ e₂ 0│
/// │ 0  0  d₃ e₃│
/// │ 0  0  0  d₄│
/// └           ┘
/// ```
///
/// # Why is it useful?
///
/// Bidiagonal decomposition is important because:
/// 1. It's an intermediate step in computing the Singular Value Decomposition (SVD)
/// 2. It's numerically stable and efficient to compute
/// 3. It reduces a general matrix to a simpler form while preserving important properties
/// 4. It's useful for solving least-squares problems and computing matrix ranks
///
/// # Example
///
/// ```
/// use nalgebra::{Matrix3x2, Matrix2x3};
///
/// // Decompose a 3x2 matrix (produces upper bidiagonal form)
/// let m = Matrix3x2::new(
///     1.0, 2.0,
///     3.0, 4.0,
///     5.0, 6.0,
/// );
///
/// let bidiag = m.bidiagonalize();
///
/// // Check that the decomposition works
/// let (u, d, v_t) = bidiag.unpack();
/// let reconstructed = u * d * v_t;
/// assert!((m - reconstructed).norm() < 1e-10);
///
/// // Decompose a 2x3 matrix (produces lower bidiagonal form)
/// let m2 = Matrix2x3::new(
///     1.0, 2.0, 3.0,
///     4.0, 5.0, 6.0,
/// );
///
/// let bidiag2 = m2.bidiagonalize();
/// let (u2, d2, v_t2) = bidiag2.unpack();
/// let reconstructed2 = u2 * d2 * v_t2;
/// assert!((m2 - reconstructed2).norm() < 1e-10);
/// ```
///
/// # See Also
///
/// - [`SVD`](crate::linalg::SVD) - Singular Value Decomposition, which builds on bidiagonalization
/// - [`QR`](crate::linalg::QR) - QR decomposition, another important matrix factorization
#[cfg_attr(feature = "serde-serialize-no-std", derive(Serialize, Deserialize))]
#[cfg_attr(
    feature = "serde-serialize-no-std",
    serde(bound(serialize = "DimMinimum<R, C>: DimSub<U1>,
         DefaultAllocator: Allocator<R, C>             +
                           Allocator<DimMinimum<R, C>> +
                           Allocator<DimDiff<DimMinimum<R, C>, U1>>,
         OMatrix<T, R, C>: Serialize,
         OVector<T, DimMinimum<R, C>>: Serialize,
         OVector<T, DimDiff<DimMinimum<R, C>, U1>>: Serialize"))
)]
#[cfg_attr(
    feature = "serde-serialize-no-std",
    serde(bound(deserialize = "DimMinimum<R, C>: DimSub<U1>,
         DefaultAllocator: Allocator<R, C>             +
                           Allocator<DimMinimum<R, C>> +
                           Allocator<DimDiff<DimMinimum<R, C>, U1>>,
         OMatrix<T, R, C>: Deserialize<'de>,
         OVector<T, DimMinimum<R, C>>: Deserialize<'de>,
         OVector<T, DimDiff<DimMinimum<R, C>, U1>>: Deserialize<'de>"))
)]
#[cfg_attr(feature = "defmt", derive(defmt::Format))]
#[derive(Clone, Debug)]
pub struct Bidiagonal<T: ComplexField, R: DimMin<C>, C: Dim>
where
    DimMinimum<R, C>: DimSub<U1>,
    DefaultAllocator:
        Allocator<R, C> + Allocator<DimMinimum<R, C>> + Allocator<DimDiff<DimMinimum<R, C>, U1>>,
{
    // TODO: perhaps we should pack the axes into different vectors so that axes for `v_t` are
    // contiguous. This prevents some useless copies.
    uv: OMatrix<T, R, C>,
    /// The diagonal elements of the decomposed matrix.
    diagonal: OVector<T, DimMinimum<R, C>>,
    /// The off-diagonal elements of the decomposed matrix.
    off_diagonal: OVector<T, DimDiff<DimMinimum<R, C>, U1>>,
    upper_diagonal: bool,
}

impl<T: ComplexField, R: DimMin<C>, C: Dim> Copy for Bidiagonal<T, R, C>
where
    DimMinimum<R, C>: DimSub<U1>,
    DefaultAllocator:
        Allocator<R, C> + Allocator<DimMinimum<R, C>> + Allocator<DimDiff<DimMinimum<R, C>, U1>>,
    OMatrix<T, R, C>: Copy,
    OVector<T, DimMinimum<R, C>>: Copy,
    OVector<T, DimDiff<DimMinimum<R, C>, U1>>: Copy,
{
}

impl<T: ComplexField, R: DimMin<C>, C: Dim> Bidiagonal<T, R, C>
where
    DimMinimum<R, C>: DimSub<U1>,
    DefaultAllocator: Allocator<R, C>
        + Allocator<C>
        + Allocator<R>
        + Allocator<DimMinimum<R, C>>
        + Allocator<DimDiff<DimMinimum<R, C>, U1>>,
{
    /// Computes the bidiagonal decomposition using Householder reflections.
    ///
    /// This function decomposes the input matrix `M` into the product `M = U * D * V^T`, where
    /// `U` and `V` are orthogonal matrices and `D` is a bidiagonal matrix.
    ///
    /// # What are Householder Reflections?
    ///
    /// Householder reflections are a numerically stable way to zero out elements in a matrix.
    /// This method uses a series of Householder reflections to progressively reduce the matrix
    /// to bidiagonal form, alternating between columns and rows.
    ///
    /// # Matrix Shape
    ///
    /// - If the matrix has more rows than columns (tall matrix), the result is an **upper bidiagonal** matrix
    /// - If the matrix has more columns than rows (wide matrix), the result is a **lower bidiagonal** matrix
    ///
    /// # Panics
    ///
    /// Panics if the matrix is empty (has zero rows or columns).
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
    /// // Compute the bidiagonal decomposition
    /// let bidiag = nalgebra::linalg::Bidiagonal::new(m);
    ///
    /// // This is a tall matrix, so we get an upper bidiagonal form
    /// assert!(bidiag.is_upper_diagonal());
    ///
    /// // Verify the decomposition
    /// let (u, d, v_t) = bidiag.unpack();
    /// let reconstructed = u * d * v_t;
    /// assert!((m - reconstructed).norm() < 1e-10);
    /// ```
    ///
    /// # Practical Use Case: Preparing for SVD
    ///
    /// ```
    /// use nalgebra::Matrix4x3;
    ///
    /// // Large matrix that we want to analyze
    /// let m = Matrix4x3::new(
    ///     1.0, 2.0, 3.0,
    ///     4.0, 5.0, 6.0,
    ///     7.0, 8.0, 9.0,
    ///     10.0, 11.0, 12.0,
    /// );
    ///
    /// // Bidiagonalization is the first step toward computing SVD
    /// let bidiag = nalgebra::linalg::Bidiagonal::new(m);
    ///
    /// // The bidiagonal form is much simpler to work with
    /// let d = bidiag.d();
    /// println!("Bidiagonal matrix has {} non-zero diagonals",
    ///          if bidiag.is_upper_diagonal() { 2 } else { 2 });
    /// ```
    ///
    /// # See Also
    ///
    /// - [`unpack`](Self::unpack) - Extract the U, D, and V^T matrices
    /// - [`is_upper_diagonal`](Self::is_upper_diagonal) - Check the bidiagonal form type
    /// - [`SVD::new`](crate::linalg::SVD::new) - Full singular value decomposition
    pub fn new(mut matrix: OMatrix<T, R, C>) -> Self {
        let (nrows, ncols) = matrix.shape_generic();
        let min_nrows_ncols = nrows.min(ncols);
        let dim = min_nrows_ncols.value();
        assert!(
            dim != 0,
            "Cannot compute the bidiagonalization of an empty matrix."
        );

        let mut diagonal = Matrix::uninit(min_nrows_ncols, Const::<1>);
        let mut off_diagonal = Matrix::uninit(min_nrows_ncols.sub(Const::<1>), Const::<1>);
        let mut axis_packed = Matrix::zeros_generic(ncols, Const::<1>);
        let mut work = Matrix::zeros_generic(nrows, Const::<1>);

        let upper_diagonal = nrows.value() >= ncols.value();
        if upper_diagonal {
            for ite in 0..dim - 1 {
                diagonal[ite] = MaybeUninit::new(householder::clear_column_unchecked(
                    &mut matrix,
                    ite,
                    0,
                    None,
                ));
                off_diagonal[ite] = MaybeUninit::new(householder::clear_row_unchecked(
                    &mut matrix,
                    &mut axis_packed,
                    &mut work,
                    ite,
                    1,
                ));
            }

            diagonal[dim - 1] = MaybeUninit::new(householder::clear_column_unchecked(
                &mut matrix,
                dim - 1,
                0,
                None,
            ));
        } else {
            for ite in 0..dim - 1 {
                diagonal[ite] = MaybeUninit::new(householder::clear_row_unchecked(
                    &mut matrix,
                    &mut axis_packed,
                    &mut work,
                    ite,
                    0,
                ));
                off_diagonal[ite] = MaybeUninit::new(householder::clear_column_unchecked(
                    &mut matrix,
                    ite,
                    1,
                    None,
                ));
            }

            diagonal[dim - 1] = MaybeUninit::new(householder::clear_row_unchecked(
                &mut matrix,
                &mut axis_packed,
                &mut work,
                dim - 1,
                0,
            ));
        }

        // Safety: diagonal and off_diagonal have been fully initialized.
        let (diagonal, off_diagonal) =
            unsafe { (diagonal.assume_init(), off_diagonal.assume_init()) };

        Bidiagonal {
            uv: matrix,
            diagonal,
            off_diagonal,
            upper_diagonal,
        }
    }

    /// Indicates whether this decomposition contains an upper-diagonal matrix.
    ///
    /// Returns `true` if the bidiagonal matrix `D` is upper bidiagonal (non-zero elements on the
    /// main diagonal and the superdiagonal above it). Returns `false` if it's lower bidiagonal
    /// (non-zero elements on the main diagonal and the subdiagonal below it).
    ///
    /// # How to Determine the Form
    ///
    /// The form of the bidiagonal matrix depends on the shape of the original matrix:
    /// - **Tall matrices** (rows >= cols) produce **upper bidiagonal** form
    /// - **Wide matrices** (rows < cols) produce **lower bidiagonal** form
    ///
    /// # Example
    ///
    /// ```
    /// use nalgebra::{Matrix3x2, Matrix2x3};
    ///
    /// // Tall matrix (3 rows, 2 cols) produces upper bidiagonal
    /// let tall = Matrix3x2::new(
    ///     1.0, 2.0,
    ///     3.0, 4.0,
    ///     5.0, 6.0,
    /// );
    /// let bidiag_tall = nalgebra::linalg::Bidiagonal::new(tall);
    /// assert!(bidiag_tall.is_upper_diagonal());
    ///
    /// // Wide matrix (2 rows, 3 cols) produces lower bidiagonal
    /// let wide = Matrix2x3::new(
    ///     1.0, 2.0, 3.0,
    ///     4.0, 5.0, 6.0,
    /// );
    /// let bidiag_wide = nalgebra::linalg::Bidiagonal::new(wide);
    /// assert!(!bidiag_wide.is_upper_diagonal());
    /// ```
    ///
    /// # Visual Representation
    ///
    /// Upper bidiagonal (3x3):
    /// ```text
    /// ┌        ┐
    /// │ x x 0 │
    /// │ 0 x x │
    /// │ 0 0 x │
    /// └        ┘
    /// ```
    ///
    /// Lower bidiagonal (3x3):
    /// ```text
    /// ┌        ┐
    /// │ x 0 0 │
    /// │ x x 0 │
    /// │ 0 x x │
    /// └        ┘
    /// ```
    ///
    /// # See Also
    ///
    /// - [`diagonal`](Self::diagonal) - Get the main diagonal values
    /// - [`off_diagonal`](Self::off_diagonal) - Get the off-diagonal values
    /// - [`d`](Self::d) - Get the full bidiagonal matrix
    #[inline]
    #[must_use]
    pub const fn is_upper_diagonal(&self) -> bool {
        self.upper_diagonal
    }

    #[inline]
    const fn axis_shift(&self) -> (usize, usize) {
        if self.upper_diagonal { (0, 1) } else { (1, 0) }
    }

    /// Unpacks this decomposition into its three matrix factors `(U, D, V^T)`.
    ///
    /// This method extracts the three matrices from the bidiagonal decomposition such that
    /// the original matrix `M` can be reconstructed as `M = U * D * V^T`.
    ///
    /// # Return Values
    ///
    /// Returns a tuple `(U, D, V^T)` where:
    /// - `U` is an orthogonal matrix with orthonormal columns
    /// - `D` is the bidiagonal matrix (only diagonal and one off-diagonal are non-zero)
    /// - `V^T` is the transpose of an orthogonal matrix with orthonormal rows
    ///
    /// # When to Use This
    ///
    /// Use `unpack()` when you need to:
    /// - Verify the decomposition by reconstructing the original matrix
    /// - Analyze the individual components of the decomposition
    /// - Use the orthogonal matrices U and V for further computations
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
    /// let bidiag = nalgebra::linalg::Bidiagonal::new(m);
    ///
    /// // Unpack into three matrices
    /// let (u, d, v_t) = bidiag.unpack();
    ///
    /// // Verify that U is orthogonal (U^T * U = I)
    /// let u_t_u = u.transpose() * &u;
    /// let identity = nalgebra::Matrix2::identity();
    /// assert!((u_t_u - identity).norm() < 1e-10);
    ///
    /// // Verify that V^T * V = I (V^T has orthonormal rows)
    /// let v = v_t.transpose();
    /// let v_t_v = v_t * v;
    /// let identity2 = nalgebra::Matrix2::identity();
    /// assert!((v_t_v - identity2).norm() < 1e-10);
    ///
    /// // Verify the decomposition: M = U * D * V^T
    /// let reconstructed = u * d * v_t;
    /// assert!((m - reconstructed).norm() < 1e-10);
    /// ```
    ///
    /// # Practical Use Case: Matrix Analysis
    ///
    /// ```
    /// use nalgebra::Matrix4x3;
    ///
    /// // Analyze a data matrix
    /// let data = Matrix4x3::new(
    ///     2.0, 1.0, 0.5,
    ///     1.0, 2.0, 1.5,
    ///     0.5, 1.5, 2.0,
    ///     0.2, 0.8, 1.0,
    /// );
    ///
    /// let bidiag = nalgebra::linalg::Bidiagonal::new(data);
    /// let (u, d, v_t) = bidiag.unpack();
    ///
    /// // The bidiagonal matrix D reveals the structure of the data
    /// println!("Bidiagonal form simplifies the matrix structure:");
    /// println!("D = {}", d);
    ///
    /// // U and V^T can be used for basis transformations
    /// println!("U provides an orthonormal basis for the column space");
    /// println!("V^T provides an orthonormal basis for the row space");
    /// ```
    ///
    /// # See Also
    ///
    /// - [`u`](Self::u) - Get only the U matrix
    /// - [`d`](Self::d) - Get only the bidiagonal matrix D
    /// - [`v_t`](Self::v_t) - Get only the V^T matrix
    /// - [`diagonal`](Self::diagonal) - Get just the diagonal values
    #[inline]
    pub fn unpack(
        self,
    ) -> (
        OMatrix<T, R, DimMinimum<R, C>>,
        OMatrix<T, DimMinimum<R, C>, DimMinimum<R, C>>,
        OMatrix<T, DimMinimum<R, C>, C>,
    )
    where
        DefaultAllocator: Allocator<DimMinimum<R, C>, DimMinimum<R, C>>
            + Allocator<R, DimMinimum<R, C>>
            + Allocator<DimMinimum<R, C>, C>,
    {
        // TODO: optimize by calling a reallocator.
        (self.u(), self.d(), self.v_t())
    }

    /// Retrieves the bidiagonal matrix `D` of this decomposition.
    ///
    /// The bidiagonal matrix contains non-zero values only on the main diagonal and one
    /// adjacent diagonal (either above or below the main diagonal, depending on whether
    /// this is an upper or lower bidiagonal form).
    ///
    /// # Return Value
    ///
    /// Returns a square matrix where:
    /// - The main diagonal contains the diagonal values from the decomposition
    /// - One off-diagonal (super or sub) contains the off-diagonal values
    /// - All other elements are zero
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
    /// let bidiag = nalgebra::linalg::Bidiagonal::new(m);
    /// let d = bidiag.d();
    ///
    /// // D is a 2x2 bidiagonal matrix
    /// println!("Bidiagonal matrix D:");
    /// println!("{}", d);
    ///
    /// // For an upper bidiagonal matrix, check the structure
    /// if bidiag.is_upper_diagonal() {
    ///     // d[(0,0)] and d[(1,1)] are on the main diagonal
    ///     // d[(0,1)] is on the superdiagonal
    ///     // d[(1,0)] should be zero
    ///     assert!(d[(1, 0)] < 1e-10);
    /// }
    /// ```
    ///
    /// # Inspecting the Bidiagonal Structure
    ///
    /// ```
    /// use nalgebra::Matrix4x3;
    ///
    /// let m = Matrix4x3::new(
    ///     1.0, 0.0, 0.0,
    ///     0.0, 2.0, 0.0,
    ///     0.0, 0.0, 3.0,
    ///     0.0, 0.0, 0.0,
    /// );
    ///
    /// let bidiag = nalgebra::linalg::Bidiagonal::new(m);
    /// let d = bidiag.d();
    ///
    /// // Count non-zero elements - should be at most 2n-1 for an nxn matrix
    /// let mut non_zero_count = 0;
    /// for i in 0..d.nrows() {
    ///     for j in 0..d.ncols() {
    ///         if d[(i, j)] > 1e-10 {
    ///             non_zero_count += 1;
    ///         }
    ///     }
    /// }
    /// println!("Non-zero elements in D: {}", non_zero_count);
    /// ```
    ///
    /// # See Also
    ///
    /// - [`diagonal`](Self::diagonal) - Get only the main diagonal values as a vector
    /// - [`off_diagonal`](Self::off_diagonal) - Get only the off-diagonal values as a vector
    /// - [`is_upper_diagonal`](Self::is_upper_diagonal) - Check if this is upper or lower bidiagonal
    /// - [`unpack`](Self::unpack) - Get U, D, and V^T together
    #[inline]
    #[must_use]
    pub fn d(&self) -> OMatrix<T, DimMinimum<R, C>, DimMinimum<R, C>>
    where
        DefaultAllocator: Allocator<DimMinimum<R, C>, DimMinimum<R, C>>,
    {
        let (nrows, ncols) = self.uv.shape_generic();

        let d = nrows.min(ncols);
        let mut res = OMatrix::identity_generic(d, d);
        res.set_partial_diagonal(
            self.diagonal
                .iter()
                .map(|e| T::from_real(e.clone().modulus())),
        );

        let start = self.axis_shift();
        res.view_mut(start, (d.value() - 1, d.value() - 1))
            .set_partial_diagonal(
                self.off_diagonal
                    .iter()
                    .map(|e| T::from_real(e.clone().modulus())),
            );
        res
    }

    /// Computes the orthogonal matrix `U` of this `U * D * V^T` decomposition.
    ///
    /// The matrix `U` is an orthogonal matrix, meaning its columns are orthonormal vectors.
    /// In other words, `U^T * U = I` (the identity matrix).
    ///
    /// # Properties of U
    ///
    /// - **Orthogonality**: The columns of U are mutually perpendicular unit vectors
    /// - **Preservation of length**: Multiplying by U doesn't change vector lengths
    /// - **Shape**: U has the same number of rows as the original matrix, with columns equal to min(rows, cols)
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
    /// let bidiag = nalgebra::linalg::Bidiagonal::new(m);
    /// let u = bidiag.u();
    ///
    /// // U is a 3x2 matrix with orthonormal columns
    /// println!("U matrix: {}", u);
    ///
    /// // Verify orthogonality: U^T * U = I
    /// let u_t_u = u.transpose() * &u;
    /// let identity = nalgebra::Matrix2::identity();
    /// assert!((u_t_u - identity).norm() < 1e-10);
    /// ```
    ///
    /// # Verifying the Decomposition
    ///
    /// ```
    /// use nalgebra::Matrix4x3;
    ///
    /// let m = Matrix4x3::new(
    ///     2.0, 1.0, 0.0,
    ///     1.0, 2.0, 1.0,
    ///     0.0, 1.0, 2.0,
    ///     0.0, 0.0, 1.0,
    /// );
    ///
    /// let bidiag = nalgebra::linalg::Bidiagonal::new(m);
    /// let u = bidiag.u();
    /// let d = bidiag.d();
    /// let v_t = bidiag.v_t();
    ///
    /// // Reconstruct the original matrix
    /// let reconstructed = u * d * v_t;
    /// assert!((m - reconstructed).norm() < 1e-10);
    /// ```
    ///
    /// # Practical Use: Basis Transformation
    ///
    /// ```
    /// use nalgebra::{Matrix3, Vector3};
    ///
    /// let m = Matrix3::new(
    ///     1.0, 2.0, 3.0,
    ///     2.0, 4.0, 5.0,
    ///     3.0, 5.0, 6.0,
    /// );
    ///
    /// let bidiag = nalgebra::linalg::Bidiagonal::new(m);
    /// let u = bidiag.u();
    ///
    /// // U provides an orthonormal basis for the column space
    /// // We can project any vector onto this basis
    /// let v = Vector3::new(1.0, 2.0, 3.0);
    /// let projected = u.transpose() * v;
    /// println!("Vector in the new basis: {}", projected);
    /// ```
    ///
    /// # See Also
    ///
    /// - [`v_t`](Self::v_t) - Get the V^T matrix
    /// - [`d`](Self::d) - Get the bidiagonal matrix D
    /// - [`unpack`](Self::unpack) - Get U, D, and V^T together
    // TODO: code duplication with householder::assemble_q.
    // Except that we are returning a rectangular matrix here.
    #[must_use]
    pub fn u(&self) -> OMatrix<T, R, DimMinimum<R, C>>
    where
        DefaultAllocator: Allocator<R, DimMinimum<R, C>>,
    {
        let (nrows, ncols) = self.uv.shape_generic();

        let mut res = Matrix::identity_generic(nrows, nrows.min(ncols));
        let dim = self.diagonal.len();
        let shift = self.axis_shift().0;

        for i in (0..dim - shift).rev() {
            let axis = self.uv.view_range(i + shift.., i);

            // Sometimes, the axis might have a zero magnitude.
            if axis.norm_squared().is_zero() {
                continue;
            }
            let refl = Reflection::new(Unit::new_unchecked(axis), T::zero());

            let mut res_rows = res.view_range_mut(i + shift.., i..);

            let sign = if self.upper_diagonal {
                self.diagonal[i].clone().signum()
            } else {
                self.off_diagonal[i].clone().signum()
            };

            refl.reflect_with_sign(&mut res_rows, sign);
        }

        res
    }

    /// Computes the orthogonal matrix `V^T` (V-transpose) of this `U * D * V^T` decomposition.
    ///
    /// The matrix `V^T` is the transpose of an orthogonal matrix `V`, meaning its rows are
    /// orthonormal vectors. This is equivalent to saying that `V^T * V = I` and `V * V^T = I`.
    ///
    /// # Properties of V^T
    ///
    /// - **Orthogonality**: The rows of V^T are mutually perpendicular unit vectors
    /// - **Inverse relationship**: V^T is both the transpose and inverse of V
    /// - **Shape**: V^T has rows equal to min(rows, cols) and columns equal to the original matrix's columns
    ///
    /// # Why V^T instead of V?
    ///
    /// We return V^T (instead of V) because:
    /// 1. It's more efficient to compute
    /// 2. It directly participates in the decomposition: M = U * D * V^T
    /// 3. It's the standard convention in numerical linear algebra
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
    /// let bidiag = nalgebra::linalg::Bidiagonal::new(m);
    /// let v_t = bidiag.v_t();
    ///
    /// // V^T is a 2x2 matrix with orthonormal rows
    /// println!("V^T matrix: {}", v_t);
    ///
    /// // Verify orthogonality: V^T * V = I
    /// let v = v_t.transpose();
    /// let v_t_v = &v_t * v;
    /// let identity = nalgebra::Matrix2::identity();
    /// assert!((v_t_v - identity).norm() < 1e-10);
    /// ```
    ///
    /// # Reconstructing the Original Matrix
    ///
    /// ```
    /// use nalgebra::Matrix2x3;
    ///
    /// let m = Matrix2x3::new(
    ///     1.0, 2.0, 3.0,
    ///     4.0, 5.0, 6.0,
    /// );
    ///
    /// let bidiag = nalgebra::linalg::Bidiagonal::new(m);
    /// let u = bidiag.u();
    /// let d = bidiag.d();
    /// let v_t = bidiag.v_t();
    ///
    /// // The original matrix equals U * D * V^T
    /// let reconstructed = u * d * v_t;
    /// assert!((m - reconstructed).norm() < 1e-10);
    /// ```
    ///
    /// # Practical Use: Row Space Basis
    ///
    /// ```
    /// use nalgebra::{Matrix3, Vector3};
    ///
    /// let m = Matrix3::new(
    ///     1.0, 2.0, 3.0,
    ///     2.0, 4.0, 5.0,
    ///     3.0, 5.0, 6.0,
    /// );
    ///
    /// let bidiag = nalgebra::linalg::Bidiagonal::new(m);
    /// let v_t = bidiag.v_t();
    ///
    /// // The rows of V^T form an orthonormal basis for the row space
    /// // Get the transpose to work with V's columns more easily
    /// let v = v_t.transpose();
    ///
    /// // Each column of V is an orthonormal basis vector
    /// for i in 0..v.ncols() {
    ///     let col = v.column(i);
    ///     println!("Basis vector {}: {}", i, col);
    ///     // Each column has unit length
    ///     let norm: f64 = col.norm();
    ///     assert!((norm - 1.0).abs() < 1e-10);
    /// }
    /// ```
    ///
    /// # See Also
    ///
    /// - [`u`](Self::u) - Get the U matrix
    /// - [`d`](Self::d) - Get the bidiagonal matrix D
    /// - [`unpack`](Self::unpack) - Get U, D, and V^T together
    #[must_use]
    pub fn v_t(&self) -> OMatrix<T, DimMinimum<R, C>, C>
    where
        DefaultAllocator: Allocator<DimMinimum<R, C>, C>,
    {
        let (nrows, ncols) = self.uv.shape_generic();
        let min_nrows_ncols = nrows.min(ncols);

        let mut res = Matrix::identity_generic(min_nrows_ncols, ncols);
        let mut work = Matrix::zeros_generic(min_nrows_ncols, Const::<1>);
        let mut axis_packed = Matrix::zeros_generic(ncols, Const::<1>);

        let shift = self.axis_shift().1;

        for i in (0..min_nrows_ncols.value() - shift).rev() {
            let axis = self.uv.view_range(i, i + shift..);
            let mut axis_packed = axis_packed.rows_range_mut(i + shift..);
            axis_packed.tr_copy_from(&axis);

            // Sometimes, the axis might have a zero magnitude.
            if axis_packed.norm_squared().is_zero() {
                continue;
            }
            let refl = Reflection::new(Unit::new_unchecked(axis_packed), T::zero());

            let mut res_rows = res.view_range_mut(i.., i + shift..);

            let sign = if self.upper_diagonal {
                self.off_diagonal[i].clone().signum()
            } else {
                self.diagonal[i].clone().signum()
            };

            refl.reflect_rows_with_sign(&mut res_rows, &mut work.rows_range_mut(i..), sign);
        }

        res
    }

    /// The main diagonal elements of the bidiagonal matrix `D`.
    ///
    /// Returns a vector containing the values on the main diagonal of the bidiagonal matrix.
    /// These are the primary elements that characterize the matrix's structure.
    ///
    /// # Return Value
    ///
    /// A vector of length min(rows, cols) containing the main diagonal values.
    /// All values are real non-negative numbers (magnitudes).
    ///
    /// # Difference from `d()`
    ///
    /// - `diagonal()` returns just the main diagonal values as a **vector**
    /// - `d()` returns the full bidiagonal **matrix** including both diagonals
    ///
    /// Use `diagonal()` when you only need the main diagonal values for analysis.
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
    /// let bidiag = nalgebra::linalg::Bidiagonal::new(m);
    /// let diag = bidiag.diagonal();
    ///
    /// // diag is a 2-element vector
    /// println!("Main diagonal values: {}", diag);
    /// assert_eq!(diag.len(), 2);
    ///
    /// // All values are non-negative (magnitudes)
    /// for i in 0..diag.len() {
    ///     assert!(diag[i] >= 0.0);
    /// }
    /// ```
    ///
    /// # Analyzing Matrix Properties
    ///
    /// ```
    /// use nalgebra::Matrix4x3;
    ///
    /// let m = Matrix4x3::new(
    ///     2.0, 0.0, 0.0,
    ///     0.0, 3.0, 0.0,
    ///     0.0, 0.0, 4.0,
    ///     0.0, 0.0, 0.0,
    /// );
    ///
    /// let bidiag = nalgebra::linalg::Bidiagonal::new(m);
    /// let diag = bidiag.diagonal();
    ///
    /// // The diagonal values reveal the matrix's scale in each dimension
    /// let max_value = diag.max();
    /// let min_value = diag.min();
    ///
    /// // Condition number estimate (ratio of largest to smallest)
    /// if min_value > 1e-10 {
    ///     let condition_estimate = max_value / min_value;
    ///     println!("Approximate condition number: {}", condition_estimate);
    /// }
    /// ```
    ///
    /// # Practical Use: Checking for Rank Deficiency
    ///
    /// ```
    /// use nalgebra::Matrix3;
    ///
    /// let m = Matrix3::new(
    ///     1.0, 2.0, 3.0,
    ///     2.0, 4.0, 6.0,  // Second row is 2x first row
    ///     3.0, 6.0, 9.0,  // Third row is 3x first row
    /// );
    ///
    /// let bidiag = nalgebra::linalg::Bidiagonal::new(m);
    /// let diag = bidiag.diagonal();
    ///
    /// // Count "significant" diagonal values to estimate rank
    /// let tolerance = 1e-10;
    /// let estimated_rank = diag.iter().filter(|&&x| x > tolerance).count();
    /// println!("Estimated matrix rank: {}", estimated_rank);
    /// // For this rank-1 matrix, we expect only 1 significant value
    /// ```
    ///
    /// # See Also
    ///
    /// - [`off_diagonal`](Self::off_diagonal) - Get the off-diagonal values
    /// - [`d`](Self::d) - Get the full bidiagonal matrix (includes both diagonals)
    /// - [`is_upper_diagonal`](Self::is_upper_diagonal) - Check the bidiagonal form type
    #[must_use]
    pub fn diagonal(&self) -> OVector<T::RealField, DimMinimum<R, C>>
    where
        DefaultAllocator: Allocator<DimMinimum<R, C>>,
    {
        self.diagonal.map(|e| e.modulus())
    }

    /// The off-diagonal elements of the bidiagonal matrix `D`.
    ///
    /// Returns a vector containing the values on the off-diagonal (either superdiagonal or
    /// subdiagonal) of the bidiagonal matrix. These elements, together with the main diagonal,
    /// completely define the bidiagonal matrix.
    ///
    /// # Return Value
    ///
    /// A vector of length min(rows, cols) - 1 containing the off-diagonal values.
    /// All values are real non-negative numbers (magnitudes).
    ///
    /// # Which Off-Diagonal?
    ///
    /// - For **upper bidiagonal** matrices: these are the **superdiagonal** elements (above main diagonal)
    /// - For **lower bidiagonal** matrices: these are the **subdiagonal** elements (below main diagonal)
    ///
    /// Use [`is_upper_diagonal()`](Self::is_upper_diagonal) to determine which case applies.
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
    /// let bidiag = nalgebra::linalg::Bidiagonal::new(m);
    /// let off_diag = bidiag.off_diagonal();
    ///
    /// // off_diag has length min(3,2) - 1 = 1
    /// println!("Off-diagonal values: {}", off_diag);
    /// assert_eq!(off_diag.len(), 1);
    ///
    /// // All values are non-negative
    /// for i in 0..off_diag.len() {
    ///     assert!(off_diag[i] >= 0.0);
    /// }
    ///
    /// // Check if it's super or sub diagonal
    /// if bidiag.is_upper_diagonal() {
    ///     println!("These are superdiagonal elements (above main diagonal)");
    /// } else {
    ///     println!("These are subdiagonal elements (below main diagonal)");
    /// }
    /// ```
    ///
    /// # Visualizing the Structure
    ///
    /// ```
    /// use nalgebra::Matrix4x3;
    ///
    /// let m = Matrix4x3::new(
    ///     1.0, 2.0, 3.0,
    ///     4.0, 5.0, 6.0,
    ///     7.0, 8.0, 9.0,
    ///     10.0, 11.0, 12.0,
    /// );
    ///
    /// let bidiag = nalgebra::linalg::Bidiagonal::new(m);
    /// let diag = bidiag.diagonal();
    /// let off_diag = bidiag.off_diagonal();
    ///
    /// println!("Bidiagonal structure:");
    /// println!("Main diagonal has {} elements", diag.len());
    /// println!("Off-diagonal has {} elements", off_diag.len());
    ///
    /// // For upper bidiagonal (4x3 is tall, so upper):
    /// // D looks like:
    /// // [ d0  e0   0 ]
    /// // [  0  d1  e1 ]
    /// // [  0   0  d2 ]
    /// // where d_i are diagonal elements and e_i are off-diagonal
    /// ```
    ///
    /// # Practical Use: Iterative Algorithms
    ///
    /// ```
    /// use nalgebra::Matrix3;
    ///
    /// let m = Matrix3::new(
    ///     1.0, 2.0, 3.0,
    ///     2.0, 4.0, 5.0,
    ///     3.0, 5.0, 6.0,
    /// );
    ///
    /// let bidiag = nalgebra::linalg::Bidiagonal::new(m);
    /// let diag = bidiag.diagonal();
    /// let off_diag = bidiag.off_diagonal();
    ///
    /// // Many iterative algorithms for SVD work with just these two vectors
    /// // rather than the full matrix, making them very efficient
    /// println!("Compact representation:");
    /// println!("  Diagonal:     {:?}", diag.as_slice());
    /// println!("  Off-diagonal: {:?}", off_diag.as_slice());
    ///
    /// // This compact form uses O(n) storage instead of O(n²)
    /// ```
    ///
    /// # See Also
    ///
    /// - [`diagonal`](Self::diagonal) - Get the main diagonal values
    /// - [`d`](Self::d) - Get the full bidiagonal matrix
    /// - [`is_upper_diagonal`](Self::is_upper_diagonal) - Check if upper or lower bidiagonal
    #[must_use]
    pub fn off_diagonal(&self) -> OVector<T::RealField, DimDiff<DimMinimum<R, C>, U1>>
    where
        DefaultAllocator: Allocator<DimDiff<DimMinimum<R, C>, U1>>,
    {
        self.off_diagonal.map(|e| e.modulus())
    }

    #[doc(hidden)]
    pub const fn uv_internal(&self) -> &OMatrix<T, R, C> {
        &self.uv
    }
}

// impl<T: ComplexField, D: DimMin<D, Output = D> + DimSub<Dyn>> Bidiagonal<T, D, D>
//     where DefaultAllocator: Allocator<D, D> +
//                             Allocator<D> {
//     /// Solves the linear system `self * x = b`, where `x` is the unknown to be determined.
//     pub fn solve<R2: Dim, C2: Dim, S2>(&self, b: &Matrix<T, R2, C2, S2>) -> OMatrix<T, R2, C2>
//         where S2: StorageMut<T, R2, C2>,
//               ShapeConstraint: SameNumberOfRows<R2, D> {
//         let mut res = b.clone_owned();
//         self.solve_mut(&mut res);
//         res
//     }
//
//     /// Solves the linear system `self * x = b`, where `x` is the unknown to be determined.
//     pub fn solve_mut<R2: Dim, C2: Dim, S2>(&self, b: &mut Matrix<T, R2, C2, S2>)
//         where S2: StorageMut<T, R2, C2>,
//               ShapeConstraint: SameNumberOfRows<R2, D> {
//
//         assert_eq!(self.uv.nrows(), b.nrows(), "Bidiagonal solve matrix dimension mismatch.");
//         assert!(self.uv.is_square(), "Bidiagonal solve: unable to solve a non-square system.");
//
//         self.q_tr_mul(b);
//         self.solve_upper_triangular_mut(b);
//     }
//
//     // TODO: duplicate code from the `solve` module.
//     fn solve_upper_triangular_mut<R2: Dim, C2: Dim, S2>(&self, b: &mut Matrix<T, R2, C2, S2>)
//         where S2: StorageMut<T, R2, C2>,
//               ShapeConstraint: SameNumberOfRows<R2, D> {
//
//         let dim  = self.uv.nrows();
//
//         for k in 0 .. b.ncols() {
//             let mut b = b.column_mut(k);
//             for i in (0 .. dim).rev() {
//                 let coeff;
//
//                 unsafe {
//                     let diag = *self.diag.vget_unchecked(i);
//                     coeff = *b.vget_unchecked(i) / diag;
//                     *b.vget_unchecked_mut(i) = coeff;
//                 }
//
//                 b.rows_range_mut(.. i).axpy(-coeff, &self.uv.view_range(.. i, i), T::one());
//             }
//         }
//     }
//
//     /// Computes the inverse of the decomposed matrix.
//     pub fn inverse(&self) -> OMatrix<T, D, D> {
//         assert!(self.uv.is_square(), "Bidiagonal inverse: unable to compute the inverse of a non-square matrix.");
//
//         // TODO: is there a less naive method ?
//         let (nrows, ncols) = self.uv.shape_generic();
//         let mut res = OMatrix::identity_generic(nrows, ncols);
//         self.solve_mut(&mut res);
//         res
//     }
//
//     // /// Computes the determinant of the decomposed matrix.
//     // pub fn determinant(&self) -> T {
//     //     let dim = self.uv.nrows();
//     //     assert!(self.uv.is_square(), "Bidiagonal determinant: unable to compute the determinant of a non-square matrix.");
//
//     //     let mut res = T::one();
//     //     for i in 0 .. dim {
//     //         res *= unsafe { *self.diag.vget_unchecked(i) };
//     //     }
//
//     //     res self.q_determinant()
//     // }
// }
