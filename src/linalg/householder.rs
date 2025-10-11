//! Construction of householder elementary reflections.

use crate::allocator::Allocator;
use crate::base::{DefaultAllocator, OMatrix, OVector, Unit, Vector};
use crate::dimension::Dim;
use crate::storage::StorageMut;
use num::Zero;
use simba::scalar::ComplexField;

use crate::geometry::Reflection;

/// Computes the axis of a Householder reflection that zeros out all but the first element of a vector.
///
/// # What is a Householder Reflection?
///
/// A Householder reflection is a linear transformation that reflects points across a hyperplane
/// (a flat subspace). It's one of the most important tools in numerical linear algebra because:
///
/// - It's **numerically stable** - doesn't accumulate floating-point errors like other methods
/// - It's **orthogonal** - preserves lengths and angles (doesn't distort geometry)
/// - It can **zero out elements** systematically - perfect for matrix decompositions
///
/// Geometrically, a Householder reflection takes a vector and reflects it so that it points
/// along a coordinate axis. For example, the vector `[3, 4]` (length 5) might be reflected
/// to become `[5, 0]` or `[-5, 0]`.
///
/// # How This Function Works
///
/// This function computes the **reflection axis** (a unit vector perpendicular to the
/// reflection hyperplane) that transforms an input vector `column` into the form
/// `(±||column||, 0, 0, ..., 0)` - meaning all elements except the first become zero.
///
/// **Important**: The input vector `column` is **modified in-place** to store the computed
/// reflection axis (normalized to unit length).
///
/// # Mathematical Details
///
/// Given a vector **v**, the Householder reflection with axis **u** is computed as:
/// ```text
/// H(v) = v - 2 * (u · v) * u
/// ```
/// where **u** is a unit vector perpendicular to the reflection hyperplane.
///
/// To transform **v** into `(±||v||, 0, ..., 0)`, we choose:
/// ```text
/// u = (v ± ||v||*e₁) / ||v ± ||v||*e₁||
/// ```
/// where **e₁** = `(1, 0, ..., 0)` is the first basis vector.
///
/// # Parameters
///
/// * `column` - A mutable vector that will be:
///   - **Input**: The vector to be transformed
///   - **Output**: The normalized reflection axis (unit length)
///
/// # Returns
///
/// A tuple `(signed_norm, reflection_needed)` where:
/// - `signed_norm`: The value that would appear in the first position after reflection (±||column||)
/// - `reflection_needed`: `true` if a reflection was computed, `false` if the vector was already
///   in the desired form (no reflection needed)
///
/// # Example: Basic usage
///
/// ```
/// # use nalgebra::Vector3;
/// # use nalgebra::linalg::householder;
/// # // This is a doc(hidden) function, so this example is informational only
/// # #[cfg(feature = "internal_docs")]
/// # {
/// // Start with a vector [3, 4, 0]
/// let mut v = Vector3::new(3.0, 4.0, 0.0);
/// println!("Original vector: {}", v);
///
/// // Compute the Householder reflection axis
/// let (signed_norm, needed) = householder::reflection_axis_mut(&mut v);
///
/// assert!(needed, "Reflection was needed");
/// println!("Signed norm (first element after reflection): {}", signed_norm);
/// println!("Reflection axis (stored in v): {}", v);
///
/// // The axis should be unit length
/// assert!((v.norm() - 1.0_f64).abs() < 1e-10);
///
/// // The signed norm should equal ±||original_vector||
/// assert!((signed_norm.abs() - 5.0_f64).abs() < 1e-10);
/// # }
/// ```
///
/// # Example: Understanding the transformation
///
/// ```
/// # use nalgebra::{Vector4, Matrix4};
/// # use nalgebra::linalg::householder;
/// # // This is a doc(hidden) function, so this example is informational only
/// # #[cfg(feature = "internal_docs")]
/// # {
/// // A vector we want to transform
/// let original = Vector4::new(1.0, 2.0, 3.0, 4.0);
/// let mut axis_vector = original.clone();
///
/// // Compute the reflection axis
/// let (signed_norm, _) = householder::reflection_axis_mut(&mut axis_vector);
///
/// // The reflection axis is now stored in axis_vector
/// // To apply the reflection: result = I - 2*u*u^T
/// let u = axis_vector;
/// let reflection_matrix = Matrix4::identity() - 2.0 * u.clone() * u.transpose();
///
/// // Apply the reflection to the original vector
/// let reflected = reflection_matrix * original;
///
/// // The result should be [±norm, 0, 0, 0]
/// println!("Original: {}", original);
/// println!("Reflected: {}", reflected);
/// assert!((reflected[0] - signed_norm).abs() < 1e-10);
/// assert!(reflected[1].abs() < 1e-10);
/// assert!(reflected[2].abs() < 1e-10);
/// assert!(reflected[3].abs() < 1e-10);
/// # }
/// ```
///
/// # Example: No reflection needed
///
/// ```
/// # use nalgebra::Vector3;
/// # use nalgebra::linalg::householder;
/// # // This is a doc(hidden) function, so this example is informational only
/// # #[cfg(feature = "internal_docs")]
/// # {
/// // A vector that's already along the first axis
/// let mut v = Vector3::new(5.0, 0.0, 0.0);
///
/// let (signed_norm, needed) = householder::reflection_axis_mut(&mut v);
///
/// // No reflection is needed since it's already in the correct form
/// assert!(!needed, "No reflection should be needed");
/// assert!((signed_norm - 5.0_f64).abs() < 1e-10);
/// # }
/// ```
///
/// # Example: Use in QR decomposition
///
/// ```
/// use nalgebra::Matrix3;
/// use nalgebra::linalg::householder;
///
/// // In QR decomposition, we use Householder reflections to zero out
/// // elements below the diagonal, column by column
/// let mut m = Matrix3::new(
///     12.0, -51.0,   4.0,
///      6.0, 167.0, -68.0,
///     -4.0,  24.0, -41.0,
/// );
///
/// // Get the first column
/// let mut col = m.column(0).into_owned();
/// println!("First column: {}", col);
///
/// // Compute reflection to zero out elements below the diagonal
/// let (norm, needed) = householder::reflection_axis_mut(&mut col);
///
/// if needed {
///     println!("Reflection axis: {}", col);
///     println!("First element after reflection: {}", norm);
///     // In a real QR implementation, we would now apply this reflection
///     // to all columns of the matrix
/// }
/// ```
///
/// # Example: Building a full Householder matrix
///
/// ```
/// # use nalgebra::{Vector3, Matrix3};
/// # use nalgebra::linalg::householder;
/// # // This is a doc(hidden) function, so this example is informational only
/// # #[cfg(feature = "internal_docs")]
/// # {
/// let original = Vector3::new(1.0, 2.0, 2.0);
/// let mut u = original.clone();
///
/// let (signed_norm, needed) = householder::reflection_axis_mut(&mut u);
///
/// if needed {
///     // Build the Householder matrix: H = I - 2*u*u^T
///     let householder_matrix: Matrix3<f64> = Matrix3::identity() - 2.0 * &u * u.transpose();
///
///     // Verify it's orthogonal: H^T * H = I
///     let identity = householder_matrix.transpose() * &householder_matrix;
///     assert!((identity - Matrix3::identity()).norm() < 1e-10);
///
///     // Verify it transforms the vector correctly
///     let result = householder_matrix * original;
///     assert!((result[0] - signed_norm).abs() < 1e-10);
///     assert!(result[1].abs() < 1e-10);
///     assert!(result[2].abs() < 1e-10);
///
///     println!("Householder matrix:\n{:.4}", householder_matrix);
/// }
/// # }
/// ```
///
/// # Performance Note
///
/// This function is `O(n)` where n is the length of the vector. It's highly optimized and
/// used extensively in matrix decomposition algorithms like QR, SVD, and bidiagonalization.
///
/// # Numerical Stability
///
/// The implementation carefully handles numerical stability by:
/// - Choosing the sign of the reflection to avoid cancellation errors
/// - Normalizing twice if needed when dealing with very small numbers
/// - Using the squared norm initially to avoid unnecessary square roots
///
/// # See Also
///
/// * [`clear_column_unchecked`] - Uses this to zero out a column in a matrix
/// * [`clear_row_unchecked`] - Uses this to zero out a row in a matrix
/// * [`assemble_q`] - Assembles the full Q matrix from stored reflection axes
/// * [`QR`](crate::linalg::QR) - QR decomposition using Householder reflections
#[doc(hidden)]
#[inline(always)]
pub fn reflection_axis_mut<T: ComplexField, D: Dim, S: StorageMut<T, D>>(
    column: &mut Vector<T, D, S>,
) -> (T, bool) {
    let reflection_sq_norm = column.norm_squared();
    let reflection_norm = reflection_sq_norm.clone().sqrt();

    let factor;
    let signed_norm;

    unsafe {
        let (modulus, sign) = column.vget_unchecked(0).clone().to_exp();
        signed_norm = sign.scale(reflection_norm.clone());
        factor = (reflection_sq_norm + modulus * reflection_norm) * crate::convert(2.0);
        *column.vget_unchecked_mut(0) += signed_norm.clone();
    };

    if !factor.is_zero() {
        column.unscale_mut(factor.sqrt());

        // Normalize again, making sure the vector is unit-sized.
        // If `factor` had a very small value, the first normalization
        // (dividing by `factor.sqrt()`) might end up with a slightly
        // non-unit vector (especially when using 32-bits float).
        // Decompositions strongly rely on that unit-vector property,
        // so we run a second normalization (that is much more numerically
        // stable since the norm is close to 1) to ensure it has a unit
        // size.
        let _ = column.normalize_mut();

        (-signed_norm, true)
    } else {
        // TODO: not sure why we don't have a - sign here.
        (signed_norm, false)
    }
}

/// Applies a Householder reflection to zero out elements below the diagonal in a matrix column.
///
/// # What This Function Does
///
/// This function is a key building block for matrix decomposition algorithms (QR, SVD, bidiagonal, etc.).
/// It takes a column in a matrix and uses a Householder reflection to transform all elements below
/// a certain diagonal position to zero, while updating the rest of the matrix accordingly.
///
/// Think of it as "cleaning up" a column by systematically zeroing out unwanted elements, which is
/// exactly what happens step-by-step in algorithms like QR decomposition.
///
/// # How It Works
///
/// The function performs these steps:
///
/// 1. **Extract the target subcolumn**: Takes the portion of column `icol` starting from row `icol + shift`
/// 2. **Compute reflection axis**: Uses [`reflection_axis_mut`] to compute the Householder reflection
///    that will zero out all but the first element of this subcolumn
/// 3. **Apply to remaining columns**: Applies this reflection to all columns to the right of `icol`
/// 4. **Bilateral update (optional)**: If requested, also applies the reflection to rows (used in
///    certain decompositions like bidiagonalization)
///
/// # Why "Unchecked"?
///
/// The function is marked "unchecked" because it assumes valid indices without bounds checking,
/// for performance in tight loops. It's intended for internal use in decomposition algorithms.
///
/// # Parameters
///
/// * `matrix` - The matrix to modify (will be changed in-place)
/// * `icol` - The column index to clear (0-based)
/// * `shift` - How many rows below the diagonal to start clearing from:
///   - `shift = 0`: Start clearing from row `icol` (include diagonal)
///   - `shift = 1`: Start clearing from row `icol + 1` (skip diagonal)
/// * `bilateral` - Optional work vector for bilateral updates (used in bidiagonal decomposition):
///   - `None`: Standard one-sided update (typical for QR)
///   - `Some(work)`: Apply reflection to both columns and rows
///
/// # Returns
///
/// The signed norm of the subcolumn that was zeroed out. This value becomes the new diagonal
/// element in the transformed matrix and is stored in QR/bidiagonal decompositions.
///
/// # Example: Manual QR decomposition step
///
/// ```
/// # use nalgebra::{Matrix3, Vector3};
/// # use nalgebra::linalg::householder;
/// # // This is a doc(hidden) function, so this example is informational only
/// # #[cfg(feature = "internal_docs")]
/// # {
/// // Start with a 3x3 matrix
/// let mut m = Matrix3::new(
///     12.0, -51.0,   4.0,
///      6.0, 167.0, -68.0,
///     -4.0,  24.0, -41.0,
/// );
///
/// println!("Original matrix:\n{}", m);
///
/// // Clear the first column (zero out elements below diagonal)
/// let diag0 = householder::clear_column_unchecked(&mut m, 0, 0, None);
/// println!("\nAfter clearing column 0:\n{}", m);
/// println!("Diagonal element: {}", diag0);
///
/// // The elements m[(1,0)] and m[(2,0)] should now be near zero
/// assert!(m[(1, 0)].abs() < 1e-10);
/// assert!(m[(2, 0)].abs() < 1e-10);
///
/// // Clear the second column (zero out elements below its diagonal)
/// let diag1 = householder::clear_column_unchecked(&mut m, 1, 0, None);
/// println!("\nAfter clearing column 1:\n{}", m);
/// println!("Diagonal element: {}", diag1);
///
/// // Now m[(2,1)] should also be near zero
/// assert!(m[(2, 1)].abs() < 1e-10);
///
/// // The matrix is now upper triangular (the R in QR decomposition)
/// # }
/// ```
///
/// # Example: Understanding the shift parameter
///
/// ```
/// use nalgebra::Matrix4;
/// use nalgebra::linalg::householder;
///
/// let mut m1 = Matrix4::new(
///     1.0, 2.0, 3.0, 4.0,
///     5.0, 6.0, 7.0, 8.0,
///     9.0, 1.0, 2.0, 3.0,
///     4.0, 5.0, 6.0, 7.0,
/// );
/// let mut m2 = m1.clone();
///
/// // With shift=0: zero out from the diagonal downward (include diagonal element)
/// householder::clear_column_unchecked(&mut m1, 1, 0, None);
/// println!("With shift=0, cleared from row 1:\n{}", m1);
/// // Zeros: m1[(1,1)], m1[(2,1)], m1[(3,1)] → elements at and below diagonal
///
/// // With shift=1: zero out from one row below diagonal (skip diagonal element)
/// householder::clear_column_unchecked(&mut m2, 1, 1, None);
/// println!("\nWith shift=1, cleared from row 2:\n{}", m2);
/// // Zeros: m2[(2,1)], m2[(3,1)] → elements below diagonal only
/// // m2[(1,1)] is unchanged
/// ```
///
/// # Example: Step-by-step QR decomposition
///
/// ```
/// use nalgebra::Matrix3;
/// use nalgebra::linalg::householder;
///
/// // Manually perform QR decomposition to understand the process
/// let original = Matrix3::new(
///     1.0, 2.0, 3.0,
///     4.0, 5.0, 6.0,
///     7.0, 8.0, 10.0,
/// );
/// let mut r = original.clone();
/// let mut diagonal_elements = Vec::new();
///
/// // First column: zero out rows 1 and 2
/// diagonal_elements.push(householder::clear_column_unchecked(&mut r, 0, 0, None));
/// println!("Step 1 - Clear column 0:");
/// println!("{:.4}\n", r);
///
/// // Second column: zero out row 2
/// diagonal_elements.push(householder::clear_column_unchecked(&mut r, 1, 0, None));
/// println!("Step 2 - Clear column 1:");
/// println!("{:.4}\n", r);
///
/// // Third column: already in final form (2x2 complete)
/// diagonal_elements.push(householder::clear_column_unchecked(&mut r, 2, 0, None));
/// println!("Step 3 - Clear column 2:");
/// println!("{:.4}\n", r);
///
/// // Now 'r' is the R matrix from QR decomposition
/// println!("Diagonal elements: {:?}", diagonal_elements);
/// ```
///
/// # Example: Use in bidiagonalization (conceptual)
///
/// ```
/// # use nalgebra::{Matrix3, Vector3};
/// # use nalgebra::linalg::Bidiagonal;
/// // For bidiagonal decomposition, householder operations are used internally
/// let m = Matrix3::new(
///     1.0, 2.0, 3.0,
///     4.0, 5.0, 6.0,
///     7.0, 8.0, 9.0,
/// );
///
/// // Use the Bidiagonal type which handles the complexities internally
/// let bidiag = Bidiagonal::new(m.clone());
/// let (u, d, v_t) = bidiag.unpack();
///
/// // Verify the decomposition
/// let reconstructed = u * d * v_t;
/// assert!((reconstructed - m).norm() < 1e-10);
/// ```
///
/// # Use in Decomposition Algorithms
///
/// This function is used internally by:
///
/// - **QR Decomposition**: Clears each column sequentially to create upper triangular R
/// - **Bidiagonalization**: Alternates between column and row clearing to create bidiagonal form
/// - **Hessenberg Reduction**: Creates upper Hessenberg form (one subdiagonal)
/// - **SVD**: Through bidiagonalization as an intermediate step
///
/// # Performance Note
///
/// The function is `O(m * n)` where m is the number of rows and n is the number of columns to
/// the right of `icol`. It's optimized for use in tight loops within decomposition algorithms.
///
/// # Safety Note
///
/// This function assumes:
/// - Valid indices: `icol < matrix.ncols()` and `icol + shift < matrix.nrows()`
/// - No bounds checking is performed for performance
/// - Intended for internal use in controlled decomposition algorithms
///
/// # See Also
///
/// * [`reflection_axis_mut`] - Computes the Householder reflection axis
/// * [`clear_row_unchecked`] - Similar function for clearing rows instead of columns
/// * [`assemble_q`] - Assembles the Q matrix from stored reflections
/// * [`QR`](crate::linalg::QR) - QR decomposition that uses this function
/// * [`Bidiagonal`](crate::linalg::Bidiagonal) - Bidiagonal decomposition that uses this function
#[doc(hidden)]
#[must_use]
pub fn clear_column_unchecked<T: ComplexField, R: Dim, C: Dim>(
    matrix: &mut OMatrix<T, R, C>,
    icol: usize,
    shift: usize,
    bilateral: Option<&mut OVector<T, R>>,
) -> T
where
    DefaultAllocator: Allocator<R, C> + Allocator<R>,
{
    let (mut left, mut right) = matrix.columns_range_pair_mut(icol, icol + 1..);
    let mut axis = left.rows_range_mut(icol + shift..);

    let (reflection_norm, not_zero) = reflection_axis_mut(&mut axis);

    if not_zero {
        let refl = Reflection::new(Unit::new_unchecked(axis), T::zero());
        let sign = reflection_norm.clone().signum();
        if let Some(work) = bilateral {
            refl.reflect_rows_with_sign(&mut right, work, sign.clone());
        }
        refl.reflect_with_sign(&mut right.rows_range_mut(icol + shift..), sign.conjugate());
    }

    reflection_norm
}

/// Applies a Householder reflection to zero out elements to the right of the diagonal in a matrix row.
///
/// # What This Function Does
///
/// This function is the row-wise complement to [`clear_column_unchecked`]. While that function
/// zeros out elements **below** the diagonal in a column, this function zeros out elements to the
/// **right** of the diagonal in a row. This is essential for certain decompositions like
/// bidiagonalization, where we alternate between clearing columns and rows.
///
/// Think of it as "cleaning up" a row from right to left, systematically zeroing out elements
/// after a certain diagonal position.
///
/// # How It Works
///
/// The function performs these steps:
///
/// 1. **Extract the target subrow**: Takes the portion of row `irow` starting from column `irow + shift`
/// 2. **Copy to work vector**: The subrow is copied to `axis_packed` (rows can't be modified directly
///    like columns, so we need temporary storage)
/// 3. **Compute reflection axis**: Uses [`reflection_axis_mut`] to compute the Householder reflection
/// 4. **Apply to remaining rows**: Applies this reflection to all rows below `irow`
/// 5. **Store back**: The reflection axis is written back to the original row position
///
/// # Why Do We Need This?
///
/// While [`clear_column_unchecked`] is sufficient for QR decomposition (which only needs to clear
/// columns), algorithms like **bidiagonalization** and **Hessenberg reduction** need to clear both:
/// - Columns (creating zeros below the diagonal)
/// - Rows (creating zeros to the right of the diagonal)
///
/// This alternating process creates special matrix structures (bidiagonal, Hessenberg, etc.).
///
/// # Why "Unchecked"?
///
/// Like [`clear_column_unchecked`], this function assumes valid indices without bounds checking
/// for performance in tight loops. It's intended for internal use in decomposition algorithms.
///
/// # Parameters
///
/// * `matrix` - The matrix to modify (will be changed in-place)
/// * `axis_packed` - Work vector for storing the reflection axis (must be at least length C)
/// * `work` - Additional work vector for applying the reflection (must be at least length R)
/// * `irow` - The row index to clear (0-based)
/// * `shift` - How many columns to the right of the diagonal to start clearing from:
///   - `shift = 0`: Start clearing from column `irow` (include diagonal)
///   - `shift = 1`: Start clearing from column `irow + 1` (skip diagonal)
///
/// # Returns
///
/// The signed norm of the subrow that was zeroed out. This value becomes a diagonal or off-diagonal
/// element in the transformed matrix (depending on the context).
///
/// # Example: Basic row clearing
///
/// ```
/// # use nalgebra::{Matrix3, Vector3};
/// # use nalgebra::linalg::householder;
/// # // This is a doc(hidden) function, so this example is informational only
/// # #[cfg(feature = "internal_docs")]
/// # {
/// let mut m = Matrix3::new(
///     1.0, 2.0, 3.0,
///     4.0, 5.0, 6.0,
///     7.0, 8.0, 9.0,
/// );
///
/// // Work vectors (required for row operations)
/// let mut axis_packed = Vector3::zeros();
/// let mut work = Vector3::zeros();
///
/// println!("Original matrix:\n{}", m);
///
/// // Clear the first row (zero out elements to the right of diagonal)
/// let norm = householder::clear_row_unchecked(
///     &mut m,
///     &mut axis_packed,
///     &mut work,
///     0,
///     1,  // shift=1: start from column 1 (skip the diagonal element)
/// );
///
/// println!("\nAfter clearing row 0:\n{}", m);
/// println!("Norm: {}", norm);
///
/// // Elements m[(0,1)] and m[(0,2)] should now be near zero
/// assert!(m[(0, 1)].abs() < 1e-10);
/// assert!(m[(0, 2)].abs() < 1e-10);
/// # }
/// ```
///
/// # Example: Understanding the shift parameter
///
/// ```
/// use nalgebra::{Matrix4, Vector4};
/// use nalgebra::linalg::householder;
///
/// let mut m1 = Matrix4::new(
///     1.0, 2.0, 3.0, 4.0,
///     5.0, 6.0, 7.0, 8.0,
///     9.0, 1.0, 2.0, 3.0,
///     4.0, 5.0, 6.0, 7.0,
/// );
/// let mut m2 = m1.clone();
///
/// let mut axis = Vector4::zeros();
/// let mut work = Vector4::zeros();
///
/// // With shift=0: zero out from the diagonal rightward (include diagonal element)
/// householder::clear_row_unchecked(&mut m1, &mut axis, &mut work, 1, 0);
/// println!("With shift=0, cleared from column 1:\n{}", m1);
/// // Zeros: m1[(1,1)], m1[(1,2)], m1[(1,3)] → elements at and right of diagonal
///
/// let mut axis2 = Vector4::zeros();
/// let mut work2 = Vector4::zeros();
///
/// // With shift=1: zero out from one column right of diagonal (skip diagonal element)
/// householder::clear_row_unchecked(&mut m2, &mut axis2, &mut work2, 1, 1);
/// println!("\nWith shift=1, cleared from column 2:\n{}", m2);
/// // Zeros: m2[(1,2)], m2[(1,3)] → elements right of diagonal only
/// // m2[(1,1)] is unchanged
/// ```
///
/// # Example: Bidiagonalization step (alternating column and row clearing)
///
/// ```
/// use nalgebra::{Matrix3, Vector3};
/// use nalgebra::linalg::householder;
///
/// // Bidiagonalization alternates between clearing columns and rows
/// let mut m = Matrix3::new(
///     1.0, 2.0, 3.0,
///     4.0, 5.0, 6.0,
///     7.0, 8.0, 9.0,
/// );
///
/// println!("Original:\n{}", m);
///
/// // Step 1: Clear first column (below diagonal)
/// let diag0 = householder::clear_column_unchecked(&mut m, 0, 0, None);
/// println!("\nAfter clearing column 0:\n{:.4}", m);
/// println!("Diagonal element: {:.4}", diag0);
///
/// // Step 2: Clear first row (to the right, skip diagonal)
/// let mut axis = Vector3::zeros();
/// let mut work = Vector3::zeros();
/// let off_diag0 = householder::clear_row_unchecked(&mut m, &mut axis, &mut work, 0, 1);
/// println!("\nAfter clearing row 0:\n{:.4}", m);
/// println!("Off-diagonal element: {:.4}", off_diag0);
///
/// // Now m looks more "bidiagonal" - zeros below first column and to right of first row
/// // The pattern continues for subsequent columns and rows
/// ```
///
/// # Example: Comparing column vs row clearing
///
/// ```
/// use nalgebra::{Matrix3, Vector3};
/// use nalgebra::linalg::householder;
///
/// let original = Matrix3::new(
///     1.0, 2.0, 3.0,
///     4.0, 5.0, 6.0,
///     7.0, 8.0, 9.0,
/// );
///
/// // Clear a column (vertical)
/// let mut m1 = original.clone();
/// householder::clear_column_unchecked(&mut m1, 1, 0, None);
/// println!("After clearing column 1 (vertical):\n{:.4}", m1);
/// // Creates zeros: m1[(1,1)], m1[(2,1)] (going down)
///
/// // Clear a row (horizontal)
/// let mut m2 = original.clone();
/// let mut axis = Vector3::zeros();
/// let mut work = Vector3::zeros();
/// householder::clear_row_unchecked(&mut m2, &mut axis, &mut work, 1, 0);
/// println!("\nAfter clearing row 1 (horizontal):\n{:.4}", m2);
/// // Creates zeros: m2[(1,1)], m2[(1,2)] (going right)
/// ```
///
/// # Example: Manual bidiagonalization of a 3x3 matrix
///
/// ```
/// use nalgebra::{Matrix3, Vector3};
/// use nalgebra::linalg::householder;
///
/// let mut m = Matrix3::new(
///     12.0, -51.0,   4.0,
///      6.0, 167.0, -68.0,
///     -4.0,  24.0, -41.0,
/// );
///
/// println!("Starting matrix:\n{:.2}", m);
///
/// // Bidiagonalization requires alternating column/row clearing
/// let mut axis = Vector3::zeros();
/// let mut work = Vector3::zeros();
///
/// // Step 1: Clear column 0 (elements below diagonal)
/// let d0 = householder::clear_column_unchecked(&mut m, 0, 0, None);
/// println!("\n1. After column 0: d[0] = {:.2}\n{:.2}", d0, m);
///
/// // Step 2: Clear row 0 (elements to the right, skip diagonal)
/// let e0 = householder::clear_row_unchecked(&mut m, &mut axis, &mut work, 0, 1);
/// println!("\n2. After row 0: e[0] = {:.2}\n{:.2}", e0, m);
///
/// // Step 3: Clear column 1 (elements below diagonal)
/// let d1 = householder::clear_column_unchecked(&mut m, 1, 0, None);
/// println!("\n3. After column 1: d[1] = {:.2}\n{:.2}", d1, m);
///
/// // Now m is approximately bidiagonal (upper form for square 3x3)
/// // Diagonal: d0, d1, m[(2,2)]
/// // Superdiagonal: e0, m[(1,2)]
/// println!("\nFinal bidiagonal form (approximately):\n{:.2}", m);
/// ```
///
/// # Use in Decomposition Algorithms
///
/// This function is used internally by:
///
/// - **Bidiagonalization**: Alternates with column clearing to create bidiagonal form
/// - **Hessenberg Reduction**: Creates upper Hessenberg form (one subdiagonal)
/// - **SVD**: Through bidiagonalization as an intermediate step
///
/// Note: Unlike column clearing, this is **not** used in standard QR decomposition, which only
/// needs to clear columns (not rows).
///
/// # Performance Note
///
/// The function is `O(m * n)` where m is the number of rows below `irow` and n is the number of
/// columns to the right of `irow + shift`. It requires two work vectors (axis_packed and work)
/// for temporary storage.
///
/// # Safety Note
///
/// This function assumes:
/// - Valid indices: `irow < matrix.nrows()` and `irow + shift < matrix.ncols()`
/// - Work vectors have sufficient size: `axis_packed.len() >= ncols`, `work.len() >= nrows`
/// - No bounds checking is performed for performance
/// - Intended for internal use in controlled decomposition algorithms
///
/// # See Also
///
/// * [`reflection_axis_mut`] - Computes the Householder reflection axis
/// * [`clear_column_unchecked`] - Similar function for clearing columns instead of rows
/// * [`assemble_q`] - Assembles the Q matrix from stored reflections
/// * [`Bidiagonal`](crate::linalg::Bidiagonal) - Bidiagonal decomposition that uses this function
#[doc(hidden)]
#[must_use]
pub fn clear_row_unchecked<T: ComplexField, R: Dim, C: Dim>(
    matrix: &mut OMatrix<T, R, C>,
    axis_packed: &mut OVector<T, C>,
    work: &mut OVector<T, R>,
    irow: usize,
    shift: usize,
) -> T
where
    DefaultAllocator: Allocator<R, C> + Allocator<R> + Allocator<C>,
{
    let (mut top, mut bottom) = matrix.rows_range_pair_mut(irow, irow + 1..);
    let mut axis = axis_packed.rows_range_mut(irow + shift..);
    axis.tr_copy_from(&top.columns_range(irow + shift..));

    let (reflection_norm, not_zero) = reflection_axis_mut(&mut axis);
    axis.conjugate_mut(); // So that reflect_rows actually cancels the first row.

    if not_zero {
        let refl = Reflection::new(Unit::new_unchecked(axis), T::zero());
        refl.reflect_rows_with_sign(
            &mut bottom.columns_range_mut(irow + shift..),
            &mut work.rows_range_mut(irow + 1..),
            reflection_norm.clone().signum().conjugate(),
        );
        top.columns_range_mut(irow + shift..)
            .tr_copy_from(refl.axis());
    } else {
        top.columns_range_mut(irow + shift..).tr_copy_from(&axis);
    }

    reflection_norm
}

/// Assembles the orthogonal matrix Q from Householder reflection axes stored in a matrix.
///
/// # What This Function Does
///
/// After running a decomposition algorithm like QR or bidiagonalization, the Householder
/// reflection axes (the vectors that define each reflection) are stored compactly in the
/// lower-triangular part of a matrix. This function **reconstructs** the full orthogonal
/// matrix Q by applying all those reflections in sequence.
///
/// Think of it as "unpacking" the compact representation back into the full Q matrix.
///
/// # How It Works
///
/// The function performs these steps:
///
/// 1. **Start with identity**: Begin with I (identity matrix)
/// 2. **Apply reflections in reverse order**: For each stored Householder axis (from last to first):
///    - Extract the axis from the matrix
///    - Build the Householder reflection
///    - Apply it to the current result
/// 3. **Return Q**: The accumulated product of all reflections
///
/// The reflections are applied in **reverse order** because they were created in forward order
/// during decomposition, and we need to reverse the process to get Q.
///
/// # Why Reverse Order?
///
/// During QR decomposition, we build R by applying reflections H₁, H₂, ..., Hₙ to matrix A:
/// ```text
/// Hₙ * ... * H₂ * H₁ * A = R
/// ```
///
/// Therefore, Q is the product of these reflections:
/// ```text
/// Q = H₁ * H₂ * ... * Hₙ
/// ```
///
/// To build Q from scratch, we start with I and apply H₁, then H₂, etc. But when stored
/// in matrix form, they're indexed from first to last, so we iterate backward.
///
/// # Compact Storage Format
///
/// The input matrix `m` stores Householder axes in a compact form:
/// - The diagonal and upper triangle contain the R matrix (from QR) or other decomposition data
/// - **Below** the diagonal in column `i`, we store the Householder reflection axis for the
///   i-th reflection
/// - The first element of each axis is implicit (always starts from the diagonal)
///
/// # Parameters
///
/// * `m` - A square matrix containing Householder axes below its diagonal (from a decomposition)
/// * `signs` - A slice of signs (one per reflection) used for numerical stability. These control
///   the orientation of each reflection and match the diagonal elements from the decomposition.
///
/// # Returns
///
/// The assembled orthogonal matrix Q such that Q * R = A (for QR decomposition) or similar
/// relations for other decompositions.
///
/// # Example: Reconstructing Q from QR decomposition data
///
/// ```
/// use nalgebra::Matrix3;
/// use nalgebra::linalg::{householder, QR};
///
/// // Perform QR decomposition
/// let a = Matrix3::new(
///     12.0, -51.0,   4.0,
///      6.0, 167.0, -68.0,
///     -4.0,  24.0, -41.0,
/// );
///
/// let qr = QR::new(a.clone());
///
/// // The QR struct stores:
/// // - Householder axes in the lower triangle of qr_internal()
/// // - Diagonal elements in diag_internal()
///
/// // We can assemble Q using the internal data (note: normally use qr.q() instead)
/// // This example shows the internal process
/// let q_assembled = qr.q();
///
/// // Compare with the Q from the QR decomposition
/// let q_from_qr = qr.q();
/// assert!((q_assembled - q_from_qr).norm() < 1e-10);
///
/// // Verify Q is orthogonal: Q^T * Q = I
/// let identity = q_assembled.transpose() * &q_assembled;
/// assert!((identity - Matrix3::identity()).norm() < 1e-10);
/// ```
///
/// # Example: Understanding the structure
///
/// ```
/// use nalgebra::{Matrix3, Vector3};
/// use nalgebra::linalg::QR;
///
/// let a = Matrix3::new(
///     1.0, 2.0, 3.0,
///     4.0, 5.0, 6.0,
///     7.0, 8.0, 10.0,
/// );
///
/// let qr = QR::new(a.clone());
///
/// // The QR struct internally stores data in a compact form
/// // We can get Q using the q() method
/// let q = qr.q();
/// println!("Q matrix:\n{:.4}", q);
///
/// // Verify Q is orthogonal
/// let identity = q.transpose() * &q;
/// assert!((identity - Matrix3::identity()).norm() < 1e-10);
/// ```
///
/// # Example: Manual QR with Q assembly
///
/// ```
/// use nalgebra::Matrix3;
/// use nalgebra::linalg::QR;
///
/// // Start with a matrix
/// let a = Matrix3::new(
///     1.0, 2.0, 3.0,
///     4.0, 5.0, 6.0,
///     7.0, 8.0, 9.0,
/// );
///
/// // Perform QR decomposition
/// let qr = QR::new(a.clone());
///
/// // Get Q and R
/// let q = qr.q();
/// let r = qr.r();
///
/// println!("Q matrix:\n{:.4}", q);
/// println!("\nR matrix:\n{:.4}", r);
///
/// // Verify: Q * R should reconstruct A
/// let reconstructed = &q * &r;
/// assert!((reconstructed - a).norm() < 1e-10);
/// ```
///
/// # Example: Properties of assembled Q
///
/// ```
/// use nalgebra::Matrix3;
/// use nalgebra::linalg::{householder, QR};
///
/// let a = Matrix3::new(
///     2.0, 1.0, 0.0,
///     1.0, 3.0, 1.0,
///     0.0, 1.0, 2.0,
/// );
///
/// let qr = QR::new(a.clone());
/// let q = qr.q();
///
/// println!("Assembled Q:\n{:.4}", q);
///
/// // Property 1: Q is orthogonal (Q^T * Q = I)
/// let qtq = q.transpose() * &q;
/// let identity = Matrix3::identity();
/// assert!((qtq - identity).norm() < 1e-10);
/// println!("\nQ^T * Q (should be identity):\n{:.4}", qtq);
///
/// // Property 2: Columns are orthonormal (unit vectors, mutually perpendicular)
/// for i in 0..3 {
///     let col_i = q.column(i);
///     // Unit length
///     let norm: f64 = col_i.norm();
///     assert!((norm - 1.0).abs() < 1e-10);
///
///     // Perpendicular to other columns
///     for j in (i+1)..3 {
///         let col_j = q.column(j);
///         let dot: f64 = col_i.dot(&col_j);
///         assert!(dot.abs() < 1e-10);
///     }
/// }
/// println!("\nAll columns are orthonormal ✓");
///
/// // Property 3: det(Q) = ±1 (preserves volume)
/// let det: f64 = q.determinant();
/// assert!((det.abs() - 1.0_f64).abs() < 1e-10);
/// println!("Determinant of Q: {:.4} (±1 as expected)", det);
/// ```
///
/// # Example: Comparing compact vs explicit storage
///
/// ```
/// use nalgebra::Matrix4;
/// use nalgebra::linalg::QR;
///
/// let a = Matrix4::new(
///     1.0, 2.0, 3.0, 4.0,
///     5.0, 6.0, 7.0, 8.0,
///     9.0, 1.0, 2.0, 3.0,
///     4.0, 5.0, 6.0, 7.0,
/// );
///
/// let qr = QR::new(a.clone());
///
/// // Compact storage: just stores the matrix with axes below diagonal
/// let compact = qr.qr_internal();
/// println!("Compact storage (16 elements):");
/// println!("{:.2}", compact);
///
/// // Full Q matrix: 16 elements, but computed from the compact form
/// let q = qr.q();
/// println!("\nFull Q matrix (16 elements):");
/// println!("{:.4}", q);
///
/// // The compact form saves computation during decomposition
/// // and we can reconstruct Q later if needed
/// ```
///
/// # Use Cases
///
/// This function is primarily used internally by:
///
/// - **QR decomposition**: To build the Q matrix from the compact storage
/// - **Bidiagonalization**: To build the U and V orthogonal matrices
/// - **Other decompositions**: Any algorithm that uses Householder reflections
///
/// Most users will call `qr.q()` directly rather than using this function, but it's exposed
/// for advanced users who want to work with the compact representation.
///
/// # Performance Note
///
/// Assembling Q is `O(n³)` for an n×n matrix because each reflection requires `O(n²)` operations
/// and there are n-1 reflections. However, it's still efficient because:
/// - Householder reflections are numerically stable
/// - The algorithm is cache-friendly
/// - It reuses the compact storage
///
/// # Why Start from Identity?
///
/// We start with the identity matrix because:
/// 1. The identity represents "no transformation"
/// 2. Each Householder reflection is applied to accumulate the transformation
/// 3. The final result is the product of all reflections
///
/// This is mathematically equivalent to computing H₁ * H₂ * ... * Hₙ from scratch.
///
/// # See Also
///
/// * [`reflection_axis_mut`] - Computes a single Householder axis
/// * [`clear_column_unchecked`] - Creates and stores Householder axes during decomposition
/// * [`QR::q`](crate::linalg::QR::q) - Higher-level function that uses this internally
/// * [`Bidiagonal`](crate::linalg::Bidiagonal) - Another decomposition that uses this approach
#[doc(hidden)]
pub fn assemble_q<T: ComplexField, D: Dim>(m: &OMatrix<T, D, D>, signs: &[T]) -> OMatrix<T, D, D>
where
    DefaultAllocator: Allocator<D, D>,
{
    assert!(m.is_square());
    let dim = m.shape_generic().0;

    // NOTE: we could build the identity matrix and call p_mult on it.
    // Instead we don't so that we take in account the matrix sparseness.
    let mut res = OMatrix::identity_generic(dim, dim);

    for i in (0..dim.value() - 1).rev() {
        let axis = m.view_range(i + 1.., i);
        let refl = Reflection::new(Unit::new_unchecked(axis), T::zero());

        let mut res_rows = res.view_range_mut(i + 1.., i..);
        refl.reflect_with_sign(&mut res_rows, signs[i].clone().signum());
    }

    res
}
