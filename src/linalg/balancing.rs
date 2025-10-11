//! Functions for balancing a matrix.

use simba::scalar::RealField;
use std::ops::{DivAssign, MulAssign};

use crate::allocator::Allocator;
use crate::base::dimension::Dim;
use crate::base::{Const, DefaultAllocator, OMatrix, OVector};

/// Applies in-place a modified Parlett and Reinsch matrix balancing with 2-norm to the matrix and returns
/// the corresponding diagonal transformation.
///
/// # What is Matrix Balancing?
///
/// Matrix balancing is a preprocessing technique that improves the numerical stability of
/// eigenvalue computations and other matrix operations. It works by finding a diagonal matrix `D`
/// such that the balanced matrix `D⁻¹ * A * D` has rows and columns with similar norms.
///
/// This is particularly useful when:
/// - Matrix entries vary widely in magnitude (poorly scaled matrices)
/// - Computing eigenvalues or eigenvectors
/// - Solving linear systems where numerical precision is critical
///
/// The balancing transformation preserves eigenvalues but can significantly improve the
/// accuracy of their computation by reducing the condition number of the matrix.
///
/// # Algorithm
///
/// This function implements the Parlett-Reinsch balancing algorithm with 2-norm, which
/// iteratively scales rows and columns to balance their norms. The algorithm is based on
/// the paper: <https://arxiv.org/pdf/1401.5766.pdf>
///
/// # Arguments
///
/// * `matrix` - A mutable reference to a square matrix that will be balanced in-place.
///              The matrix is modified to contain the balanced version `D⁻¹ * A * D`.
///
/// # Returns
///
/// A vector `d` containing the diagonal entries of the balancing matrix `D`.
/// To reverse the balancing transformation, use [`unbalance`] with this vector.
///
/// # Panics
///
/// Panics if the input matrix is not square.
///
/// # Examples
///
/// ## Basic Usage
///
/// ```
/// use nalgebra::{Matrix3, balancing};
///
/// // Create a matrix with entries of varying magnitudes
/// let mut m = Matrix3::new(
///     1.0,    1000.0, 0.001,
///     0.001,  1.0,    1000.0,
///     1000.0, 0.001,  1.0
/// );
///
/// // Balance the matrix
/// let d = balancing::balance_parlett_reinsch(&mut m);
///
/// // The balanced matrix m now has better numerical properties
/// // d contains the scaling factors used
/// ```
///
/// ## Improving Eigenvalue Computation
///
/// ```
/// use nalgebra::{Matrix3, balancing};
///
/// // A poorly scaled matrix
/// let original = Matrix3::new(
///     1e-8, 1.0,  0.0,
///     0.0,  1e8,  1.0,
///     1.0,  0.0,  1.0
/// );
///
/// // Clone for balancing
/// let mut balanced = original.clone();
/// let d = balancing::balance_parlett_reinsch(&mut balanced);
///
/// // Now compute eigenvalues on the balanced matrix for better accuracy
/// // The eigenvalues of `balanced` are the same as `original`,
/// // but they can be computed more accurately due to better scaling
/// ```
///
/// ## Roundtrip: Balance and Unbalance
///
/// ```
/// use nalgebra::{Matrix3, balancing};
///
/// let original = Matrix3::new(
///     1.0, 2.0, 3.0,
///     4.0, 5.0, 6.0,
///     7.0, 8.0, 9.0
/// );
///
/// let mut m = original.clone();
/// let d = balancing::balance_parlett_reinsch(&mut m);
///
/// // Reverse the balancing
/// balancing::unbalance(&mut m, &d);
///
/// // m is now back to the original matrix (within floating point precision)
/// assert!((m - original).norm() < 1e-10);
/// ```
///
/// ## Working with Dynamic Matrices
///
/// ```
/// use nalgebra::{DMatrix, balancing};
///
/// // Create a 4x4 dynamic matrix
/// let mut m = DMatrix::from_row_slice(4, 4, &[
///     1.0,    100.0,  0.01,   1.0,
///     0.01,   1.0,    100.0,  1.0,
///     100.0,  0.01,   1.0,    1.0,
///     1.0,    1.0,    1.0,    1.0,
/// ]);
///
/// let d = balancing::balance_parlett_reinsch(&mut m);
/// // m is now balanced with improved numerical properties
/// ```
///
/// # See Also
///
/// * [`unbalance`] - Reverses the balancing transformation
/// * [`Schur`](crate::linalg::Schur) - Schur decomposition, often used with balancing for eigenvalue computation
/// * [`Eigen`](crate::linalg::Eigen) - Eigendecomposition that benefits from balanced matrices
pub fn balance_parlett_reinsch<T: RealField, D: Dim>(matrix: &mut OMatrix<T, D, D>) -> OVector<T, D>
where
    DefaultAllocator: Allocator<D, D> + Allocator<D>,
{
    assert!(matrix.is_square(), "Unable to balance a non-square matrix.");

    let dim = matrix.shape_generic().0;
    let radix: T = crate::convert(2.0f64);
    let mut d = OVector::from_element_generic(dim, Const::<1>, T::one());

    let mut converged = false;

    while !converged {
        converged = true;

        for i in 0..dim.value() {
            let mut n_col = matrix.column(i).norm_squared();
            let mut n_row = matrix.row(i).norm_squared();
            let mut f = T::one();

            let s = n_col.clone() + n_row.clone();
            n_col = n_col.sqrt();
            n_row = n_row.sqrt();

            if n_col.clone().is_zero() || n_row.clone().is_zero() {
                continue;
            }

            while n_col.clone() < n_row.clone() / radix.clone() {
                n_col *= radix.clone();
                n_row /= radix.clone();
                f *= radix.clone();
            }

            while n_col.clone() >= n_row.clone() * radix.clone() {
                n_col /= radix.clone();
                n_row *= radix.clone();
                f /= radix.clone();
            }

            let eps: T = crate::convert(0.95);
            #[allow(clippy::suspicious_operation_groupings)]
            if n_col.clone() * n_col + n_row.clone() * n_row < eps * s {
                converged = false;
                d[i] *= f.clone();
                matrix.column_mut(i).mul_assign(f.clone());
                matrix.row_mut(i).div_assign(f.clone());
            }
        }
    }

    d
}

/// Computes in-place `D * m * D.inverse()`, where `D` is the matrix with diagonal `d`.
///
/// # What is Unbalancing?
///
/// Unbalancing is the reverse operation of matrix balancing. After balancing a matrix `A` to get
/// `D⁻¹ * A * D` and performing computations on it (such as eigenvalue decomposition), you often
/// need to transform the results back to the original space. This function performs that
/// transformation: given a matrix `m` and the balancing vector `d`, it computes `D * m * D⁻¹`.
///
/// # Common Use Cases
///
/// - **Recovering eigenvectors**: After computing eigenvectors of a balanced matrix, unbalance
///   them to get eigenvectors of the original matrix.
/// - **Verification**: Unbalance a balanced matrix to recover the original (useful for testing).
/// - **Transforming results**: Any matrix that was computed from a balanced matrix may need to
///   be unbalanced to relate back to the original problem.
///
/// # Mathematical Background
///
/// If you balanced matrix `A` to get `B = D⁻¹ * A * D`, then:
/// - The eigenvalues of `A` and `B` are identical
/// - If `v` is an eigenvector of `B`, then `D * v` is an eigenvector of `A`
/// - To transform any matrix `M` computed from `B` back to the space of `A`, compute `D * M * D⁻¹`
///
/// # Arguments
///
/// * `m` - A mutable reference to a square matrix that will be unbalanced in-place.
///         The matrix is modified to contain `D * m * D⁻¹`.
/// * `d` - A vector containing the diagonal entries of the balancing matrix `D`.
///         This should be the vector returned by [`balance_parlett_reinsch`].
///
/// # Panics
///
/// - Panics if `m` is not square.
/// - Panics if the dimensions of `m` and `d` don't match.
///
/// # Examples
///
/// ## Basic Roundtrip
///
/// ```
/// use nalgebra::{Matrix3, balancing};
///
/// let original = Matrix3::new(
///     1.0, 2.0, 3.0,
///     4.0, 5.0, 6.0,
///     7.0, 8.0, 9.0
/// );
///
/// // Balance the matrix
/// let mut m = original.clone();
/// let d = balancing::balance_parlett_reinsch(&mut m);
///
/// // Now unbalance to recover the original
/// balancing::unbalance(&mut m, &d);
///
/// // m is back to the original matrix (within floating point precision)
/// assert!((m - original).norm() < 1e-10);
/// ```
///
/// ## Recovering Eigenvectors
///
/// ```
/// use nalgebra::{Matrix3, balancing};
///
/// // Original matrix
/// let original = Matrix3::new(
///     1000.0, 0.001,  0.0,
///     0.001,  1.0,    1000.0,
///     0.0,    1000.0, 0.001
/// );
///
/// // Balance it
/// let mut balanced = original.clone();
/// let d = balancing::balance_parlett_reinsch(&mut balanced);
///
/// // After computing eigenvectors of `balanced` (represented here as identity for demo)
/// // you would unbalance them to get eigenvectors of `original`
/// let mut eigenvectors = Matrix3::identity();
/// balancing::unbalance(&mut eigenvectors, &d);
///
/// // `eigenvectors` now relates to the original unbalanced matrix
/// ```
///
/// ## Verifying Balance Properties
///
/// ```
/// use nalgebra::{Matrix3, balancing};
///
/// let mut m = Matrix3::new(
///     1.0,   100.0,  0.01,
///     0.01,  1.0,    100.0,
///     100.0, 0.01,   1.0
/// );
///
/// let original = m.clone();
/// let d = balancing::balance_parlett_reinsch(&mut m);
///
/// // Store the balanced matrix
/// let balanced = m.clone();
///
/// // Unbalance it
/// balancing::unbalance(&mut m, &d);
///
/// // Verify we get back the original (within floating point precision)
/// assert!((m - original).norm() < 1e-10);
/// ```
///
/// ## Working with Dynamic Matrices
///
/// ```
/// use nalgebra::{DMatrix, balancing};
///
/// let original = DMatrix::from_row_slice(4, 4, &[
///     1.0,    100.0,  0.01,   1.0,
///     0.01,   1.0,    100.0,  1.0,
///     100.0,  0.01,   1.0,    1.0,
///     1.0,    1.0,    1.0,    1.0,
/// ]);
///
/// let mut m = original.clone();
/// let d = balancing::balance_parlett_reinsch(&mut m);
///
/// // Perform some computation on balanced matrix...
///
/// // Unbalance to get back to original space
/// balancing::unbalance(&mut m, &d);
/// assert!((m - original).norm() < 1e-10);
/// ```
///
/// ## Understanding the Transformation
///
/// ```
/// use nalgebra::{Matrix2, Vector2, balancing};
///
/// let mut m = Matrix2::new(
///     1.0,    1000.0,
///     0.001,  1.0
/// );
///
/// let d = balancing::balance_parlett_reinsch(&mut m);
///
/// // The balancing vector contains the scaling factors
/// // Each entry d[i] scales row i and column i
/// println!("Balancing factors: {:?}", d);
///
/// // Unbalancing reverses this transformation
/// balancing::unbalance(&mut m, &d);
/// ```
///
/// # See Also
///
/// * [`balance_parlett_reinsch`] - Performs the balancing transformation that this function reverses
/// * [`Schur`](crate::linalg::Schur) - Schur decomposition, where balancing/unbalancing is useful
/// * [`Eigen`](crate::linalg::Eigen) - Eigendecomposition that benefits from balanced matrices
pub fn unbalance<T: RealField, D: Dim>(m: &mut OMatrix<T, D, D>, d: &OVector<T, D>)
where
    DefaultAllocator: Allocator<D, D> + Allocator<D>,
{
    assert!(m.is_square(), "Unable to unbalance a non-square matrix.");
    assert_eq!(m.nrows(), d.len(), "Unbalancing: mismatched dimensions.");

    for j in 0..d.len() {
        let mut col = m.column_mut(j);
        let denom = T::one() / d[j].clone();

        for i in 0..d.len() {
            col[i] *= d[i].clone() * denom.clone();
        }
    }
}
