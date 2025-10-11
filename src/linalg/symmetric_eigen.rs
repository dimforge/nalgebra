#[cfg(feature = "serde-serialize-no-std")]
use serde::{Deserialize, Serialize};

use approx::AbsDiffEq;
use num::Zero;

use crate::allocator::Allocator;
use crate::base::{DefaultAllocator, Matrix2, OMatrix, OVector, SquareMatrix, Vector2};
use crate::dimension::{Dim, DimDiff, DimSub, U1};
use crate::storage::Storage;
use simba::scalar::ComplexField;

use crate::linalg::SymmetricTridiagonal;
use crate::linalg::givens::GivensRotation;

/// Eigendecomposition of a symmetric matrix.
///
/// # What is Symmetric Eigenvalue Decomposition?
///
/// Symmetric eigenvalue decomposition breaks down a symmetric matrix into its fundamental
/// components: eigenvalues and eigenvectors. For a symmetric matrix **A**, we find:
///
/// **A = Q Λ Q<sup>T</sup>**
///
/// Where:
/// - **Q** is an orthogonal matrix whose columns are the eigenvectors
/// - **Λ** (Lambda) is a diagonal matrix containing the eigenvalues
/// - **Q<sup>T</sup>** is the transpose of **Q**
///
/// # Understanding Eigenvalues and Eigenvectors
///
/// Think of eigenvalues and eigenvectors as describing the "natural directions" and "strengths"
/// of a matrix transformation:
///
/// - **Eigenvectors**: Special directions that remain unchanged (only scaled) when the matrix
///   is applied. For symmetric matrices, these directions are always perpendicular to each other.
/// - **Eigenvalues**: The scaling factors along each eigenvector direction. They tell you how
///   much the matrix stretches or compresses in each direction.
///
/// # Common Applications
///
/// 1. **Principal Component Analysis (PCA)**: Finding the most important patterns in data by
///    computing eigenvalues/eigenvectors of the covariance matrix.
///
/// 2. **Physics Simulations**: Analyzing vibration modes in mechanical systems, where eigenvalues
///    represent frequencies and eigenvectors represent vibration patterns.
///
/// 3. **Stress Analysis**: In structural engineering, eigenvalues identify principal stresses
///    and eigenvectors show their directions.
///
/// 4. **Stability Analysis**: Determining if a system is stable by examining the signs of eigenvalues.
///
/// # Example: Basic Usage
///
/// ```
/// use nalgebra::{Matrix3, Vector3};
///
/// // Create a symmetric 3x3 matrix
/// let m = Matrix3::new(
///     2.0, 1.0, 0.0,
///     1.0, 2.0, 1.0,
///     0.0, 1.0, 2.0,
/// );
///
/// // Compute the eigendecomposition
/// let eigen = m.symmetric_eigen();
///
/// // Access eigenvalues (as a vector)
/// let eigenvalues = &eigen.eigenvalues;
/// println!("Eigenvalues: {}", eigenvalues);
///
/// // Access eigenvectors (as columns of a matrix)
/// let eigenvectors = &eigen.eigenvectors;
/// println!("Eigenvectors:\n{}", eigenvectors);
///
/// // Verify the decomposition: A = Q Λ Q^T
/// let reconstructed = eigen.recompose();
/// assert!((m - reconstructed).norm() < 1e-10);
/// ```
///
/// # Example: Principal Component Analysis (PCA)
///
/// ```
/// use nalgebra::Matrix2;
///
/// // Covariance matrix from data (2 features)
/// let cov = Matrix2::new(
///     4.0, 2.0,
///     2.0, 3.0,
/// );
///
/// // Eigendecomposition gives us principal components
/// let eigen = cov.symmetric_eigen();
///
/// // Eigenvalues tell us the variance explained by each component
/// println!("Variance explained: {}", eigen.eigenvalues);
///
/// // Eigenvectors are the principal component directions
/// println!("Principal components:\n{}", eigen.eigenvectors);
/// ```
///
/// # Example: Vibration Analysis
///
/// ```
/// use nalgebra::Matrix2;
///
/// // Mass-normalized stiffness matrix for a 2-DOF system
/// // (e.g., two masses connected by springs)
/// let stiffness = Matrix2::new(
///     2.0, -1.0,
///    -1.0,  2.0,
/// );
///
/// // Eigenvalues are the squared natural frequencies (ω²)
/// let eigen = stiffness.symmetric_eigen();
///
/// // Natural frequencies of vibration
/// println!("Eigenvalues (ω²): {}", eigen.eigenvalues);
///
/// // Eigenvectors show the vibration mode shapes
/// println!("Mode shapes:\n{}", eigen.eigenvectors);
/// ```
///
/// # Fields
///
/// The decomposition exposes two public fields:
///
/// - `eigenvalues`: A vector containing the eigenvalues (may be unsorted)
/// - `eigenvectors`: A matrix where each column is an eigenvector corresponding to
///   the eigenvalue at the same index
///
/// # See Also
///
/// - [`SymmetricEigen::new`]: Compute the eigendecomposition with default parameters
/// - [`SymmetricEigen::try_new`]: Compute with custom convergence parameters
/// - [`SymmetricEigen::recompose`]: Reconstruct the original matrix
/// - [`SquareMatrix::symmetric_eigenvalues`]: Compute only eigenvalues (faster)
#[cfg_attr(feature = "serde-serialize-no-std", derive(Serialize, Deserialize))]
#[cfg_attr(
    feature = "serde-serialize-no-std",
    serde(bound(serialize = "DefaultAllocator: Allocator<D, D> +
                           Allocator<D>,
         OVector<T::RealField, D>: Serialize,
         OMatrix<T, D, D>: Serialize"))
)]
#[cfg_attr(
    feature = "serde-serialize-no-std",
    serde(bound(deserialize = "DefaultAllocator: Allocator<D, D> +
                           Allocator<D>,
         OVector<T::RealField, D>: Deserialize<'de>,
         OMatrix<T, D, D>: Deserialize<'de>"))
)]
#[cfg_attr(feature = "defmt", derive(defmt::Format))]
#[derive(Clone, Debug)]
pub struct SymmetricEigen<T: ComplexField, D: Dim>
where
    DefaultAllocator: Allocator<D, D> + Allocator<D>,
{
    /// The eigenvectors of the decomposed matrix.
    pub eigenvectors: OMatrix<T, D, D>,

    /// The unsorted eigenvalues of the decomposed matrix.
    pub eigenvalues: OVector<T::RealField, D>,
}

impl<T: ComplexField, D: Dim> Copy for SymmetricEigen<T, D>
where
    DefaultAllocator: Allocator<D, D> + Allocator<D>,
    OMatrix<T, D, D>: Copy,
    OVector<T::RealField, D>: Copy,
{
}

impl<T: ComplexField, D: Dim> SymmetricEigen<T, D>
where
    DefaultAllocator: Allocator<D, D> + Allocator<D>,
{
    /// Computes the eigendecomposition of the given symmetric matrix.
    ///
    /// This is the primary method for performing symmetric eigenvalue decomposition. It uses
    /// default convergence parameters that work well for most cases. The algorithm decomposes
    /// the symmetric matrix **A** into **A = Q Λ Q<sup>T</sup>**, where **Q** contains the
    /// eigenvectors and **Λ** contains the eigenvalues.
    ///
    /// # Important Notes
    ///
    /// - Only the **lower-triangular part** (including the diagonal) of the matrix is read.
    ///   The upper-triangular part is ignored, so the matrix doesn't need to be perfectly
    ///   symmetric in memory - only the lower half matters.
    ///
    /// - The returned eigenvalues are **not sorted**. If you need them in a specific order
    ///   (e.g., largest to smallest), you'll need to sort them yourself.
    ///
    /// - This method panics if convergence fails. For more control, use [`try_new`](Self::try_new).
    ///
    /// # Arguments
    ///
    /// * `m` - A square symmetric matrix to decompose
    ///
    /// # Returns
    ///
    /// A `SymmetricEigen` struct containing:
    /// - `eigenvalues`: Vector of eigenvalues (unsorted)
    /// - `eigenvectors`: Matrix where each column is an eigenvector
    ///
    /// # Panics
    ///
    /// Panics if the iterative algorithm fails to converge (very rare with default parameters).
    ///
    /// # Example: Basic 2×2 Matrix
    ///
    /// ```
    /// use nalgebra::Matrix2;
    ///
    /// // Create a symmetric matrix
    /// let m = Matrix2::new(
    ///     3.0, 1.0,
    ///     1.0, 3.0,
    /// );
    ///
    /// // Compute eigendecomposition
    /// let eigen = m.symmetric_eigen();
    ///
    /// // The eigenvalues for this matrix are 2.0 and 4.0
    /// println!("Eigenvalues: {}", eigen.eigenvalues);
    ///
    /// // Verify: A * v = λ * v for each eigenvector
    /// for i in 0..2 {
    ///     let v = eigen.eigenvectors.column(i);
    ///     let lambda = eigen.eigenvalues[i];
    ///     let lhs = m * v;
    ///     let rhs = v * lambda;
    ///     assert!((lhs - rhs).norm() < 1e-10);
    /// }
    /// ```
    ///
    /// # Example: Identifying Principal Stresses
    ///
    /// ```
    /// use nalgebra::Matrix3;
    ///
    /// // Stress tensor in material (symmetric)
    /// let stress = Matrix3::new(
    ///     10.0,  2.0,  1.0,
    ///      2.0,  8.0,  0.5,
    ///      1.0,  0.5,  6.0,
    /// );
    ///
    /// let eigen = stress.symmetric_eigen();
    ///
    /// // Principal stresses are the eigenvalues
    /// let mut principal_stresses: Vec<f64> = eigen.eigenvalues.iter().copied().collect();
    /// principal_stresses.sort_by(|a, b| b.partial_cmp(a).unwrap());
    ///
    /// println!("Maximum principal stress: {:.2}", principal_stresses[0]);
    /// println!("Minimum principal stress: {:.2}", principal_stresses[2]);
    ///
    /// // Principal directions are the eigenvectors
    /// println!("Principal directions:\n{}", eigen.eigenvectors);
    /// ```
    ///
    /// # Example: Only Lower Triangle Matters
    ///
    /// ```
    /// use nalgebra::Matrix2;
    ///
    /// // These two matrices give the same result because
    /// // only the lower triangle (including diagonal) is used
    /// let m1 = Matrix2::new(
    ///     5.0, 99.0,  // Upper triangle is ignored
    ///     2.0,  3.0,
    /// );
    ///
    /// let m2 = Matrix2::new(
    ///     5.0, -42.0,  // Different upper triangle, same result
    ///     2.0,   3.0,
    /// );
    ///
    /// let eigen1 = m1.symmetric_eigen();
    /// let eigen2 = m2.symmetric_eigen();
    ///
    /// // Same eigenvalues (the lower triangle defines the matrix)
    /// assert!((eigen1.eigenvalues - eigen2.eigenvalues).norm() < 1e-10);
    /// ```
    ///
    /// # See Also
    ///
    /// - [`SymmetricEigen::try_new`]: Version with custom convergence parameters
    /// - [`SymmetricEigen::recompose`]: Reconstruct the original matrix from the decomposition
    /// - [`SquareMatrix::symmetric_eigenvalues`]: Compute only eigenvalues (faster, no eigenvectors)
    pub fn new(m: OMatrix<T, D, D>) -> Self
    where
        D: DimSub<U1>,
        DefaultAllocator: Allocator<DimDiff<D, U1>> + Allocator<DimDiff<D, U1>>,
    {
        Self::try_new(m, T::RealField::default_epsilon(), 0).unwrap()
    }

    /// Computes the eigendecomposition of the given symmetric matrix with user-specified
    /// convergence parameters.
    ///
    /// This method gives you fine-grained control over the iterative eigenvalue algorithm.
    /// Use this when you need to:
    /// - Handle potential convergence failures gracefully (returns `None` instead of panicking)
    /// - Adjust convergence tolerance for your specific accuracy requirements
    /// - Limit computation time by setting a maximum iteration count
    ///
    /// # Important Notes
    ///
    /// - Only the **lower-triangular part** (including the diagonal) of `m` is read
    /// - Returns `None` if the algorithm doesn't converge within `max_niter` iterations
    /// - Setting `max_niter = 0` means no limit - the algorithm runs until convergence
    ///
    /// # Arguments
    ///
    /// * `m` - A square symmetric matrix to decompose
    /// * `eps` - Tolerance used to determine when a value has converged to 0.
    ///   Smaller values give higher accuracy but may require more iterations.
    ///   Typical value: `1e-10` for `f64`, `1e-5` for `f32`
    /// * `max_niter` - Maximum number of iterations. If exceeded, returns `None`.
    ///   If `max_niter == 0`, the algorithm continues indefinitely until convergence.
    ///
    /// # Returns
    ///
    /// - `Some(SymmetricEigen)` - Successfully computed eigendecomposition
    /// - `None` - Algorithm failed to converge within the iteration limit
    ///
    /// # Example: Basic Usage with Error Handling
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
    /// // Use custom tolerance and iteration limit
    /// let eps = 1e-12;
    /// let max_iter = 100;
    ///
    /// match SymmetricEigen::try_new(m, eps, max_iter) {
    ///     Some(eigen) => {
    ///         println!("Success! Eigenvalues: {}", eigen.eigenvalues);
    ///     },
    ///     None => {
    ///         println!("Failed to converge within {} iterations", max_iter);
    ///     }
    /// }
    /// ```
    ///
    /// # Example: High-Precision Computation
    ///
    /// ```
    /// use nalgebra::Matrix2;
    ///
    /// let m = Matrix2::new(
    ///     1.0, 0.5,
    ///     0.5, 1.0,
    /// );
    ///
    /// // Require very high precision
    /// let high_precision = 1e-15;
    /// let no_iter_limit = 0;  // Run until convergence
    ///
    /// let eigen = SymmetricEigen::try_new(m, high_precision, no_iter_limit)
    ///     .expect("Should converge for this simple matrix");
    ///
    /// // Verify accuracy
    /// let reconstructed = eigen.recompose();
    /// assert!((m - reconstructed).norm() < 1e-14);
    /// ```
    ///
    /// # Example: Fast Approximation with Limited Iterations
    ///
    /// ```
    /// use nalgebra::Matrix4;
    ///
    /// // Large matrix where we want quick results
    /// let m = Matrix4::new(
    ///     10.0, 1.0, 0.5, 0.2,
    ///      1.0, 8.0, 0.3, 0.1,
    ///      0.5, 0.3, 6.0, 0.4,
    ///      0.2, 0.1, 0.4, 4.0,
    /// );
    ///
    /// // Use relaxed tolerance and limit iterations for speed
    /// let relaxed_eps = 1e-6;
    /// let quick_iters = 20;
    ///
    /// if let Some(eigen) = SymmetricEigen::try_new(m, relaxed_eps, quick_iters) {
    ///     // Good enough approximation for this use case
    ///     println!("Approximate eigenvalues: {}", eigen.eigenvalues);
    /// } else {
    ///     // Fall back to default settings if quick method fails
    ///     let eigen = m.symmetric_eigen();
    ///     println!("Using default settings: {}", eigen.eigenvalues);
    /// }
    /// ```
    ///
    /// # Example: Comparing Different Tolerance Settings
    ///
    /// ```
    /// use nalgebra::Matrix2;
    ///
    /// let m = Matrix2::new(
    ///     2.0, 1.0,
    ///     1.0, 2.0,
    /// );
    ///
    /// // Low precision (faster)
    /// let eigen_fast = SymmetricEigen::try_new(m, 1e-6, 0).unwrap();
    ///
    /// // High precision (slower but more accurate)
    /// let eigen_precise = SymmetricEigen::try_new(m, 1e-14, 0).unwrap();
    ///
    /// // Both should give similar results for this simple matrix
    /// let diff = (eigen_fast.eigenvalues - eigen_precise.eigenvalues).norm();
    /// println!("Difference between tolerances: {}", diff);
    /// assert!(diff < 1e-6);
    /// ```
    ///
    /// # See Also
    ///
    /// - [`SymmetricEigen::new`]: Simpler version with default parameters
    /// - [`SymmetricEigen::recompose`]: Reconstruct the original matrix
    /// - [`SquareMatrix::symmetric_eigenvalues`]: Compute only eigenvalues
    pub fn try_new(m: OMatrix<T, D, D>, eps: T::RealField, max_niter: usize) -> Option<Self>
    where
        D: DimSub<U1>,
        DefaultAllocator: Allocator<DimDiff<D, U1>> + Allocator<DimDiff<D, U1>>,
    {
        Self::do_decompose(m, true, eps, max_niter).map(|(vals, vecs)| SymmetricEigen {
            eigenvectors: vecs.unwrap(),
            eigenvalues: vals,
        })
    }

    fn do_decompose(
        mut matrix: OMatrix<T, D, D>,
        eigenvectors: bool,
        eps: T::RealField,
        max_niter: usize,
    ) -> Option<(OVector<T::RealField, D>, Option<OMatrix<T, D, D>>)>
    where
        D: DimSub<U1>,
        DefaultAllocator: Allocator<DimDiff<D, U1>> + Allocator<DimDiff<D, U1>>,
    {
        assert!(
            matrix.is_square(),
            "Unable to compute the eigendecomposition of a non-square matrix."
        );
        let dim = matrix.nrows();
        let m_amax = matrix.camax();

        if !m_amax.is_zero() {
            matrix.unscale_mut(m_amax.clone());
        }

        let (mut q_mat, mut diag, mut off_diag);

        if eigenvectors {
            let res = SymmetricTridiagonal::new(matrix).unpack();
            q_mat = Some(res.0);
            diag = res.1;
            off_diag = res.2;
        } else {
            let res = SymmetricTridiagonal::new(matrix).unpack_tridiagonal();
            q_mat = None;
            diag = res.0;
            off_diag = res.1;
        }

        if dim == 1 {
            diag.scale_mut(m_amax);
            return Some((diag, q_mat));
        }

        let mut niter = 0;
        let (mut start, mut end) =
            Self::delimit_subproblem(&diag, &mut off_diag, dim - 1, eps.clone());

        while end != start {
            let subdim = end - start + 1;

            #[allow(clippy::comparison_chain)]
            if subdim > 2 {
                let m = end - 1;
                let n = end;

                let mut vec = Vector2::new(
                    diag[start].clone()
                        - wilkinson_shift(
                            diag[m].clone().clone(),
                            diag[n].clone(),
                            off_diag[m].clone().clone(),
                        ),
                    off_diag[start].clone(),
                );

                for i in start..n {
                    let j = i + 1;

                    match GivensRotation::cancel_y(&vec) {
                        Some((rot, norm)) => {
                            if i > start {
                                // Not the first iteration.
                                off_diag[i - 1] = norm;
                            }

                            let mii = diag[i].clone();
                            let mjj = diag[j].clone();
                            let mij = off_diag[i].clone();

                            let cc = rot.c() * rot.c();
                            let ss = rot.s() * rot.s();
                            let cs = rot.c() * rot.s();

                            let b = cs.clone() * crate::convert(2.0) * mij.clone();

                            diag[i] =
                                (cc.clone() * mii.clone() + ss.clone() * mjj.clone()) - b.clone();
                            diag[j] = (ss.clone() * mii.clone() + cc.clone() * mjj.clone()) + b;
                            off_diag[i] = cs * (mii - mjj) + mij * (cc - ss);

                            if i != n - 1 {
                                vec.x = off_diag[i].clone();
                                vec.y = -rot.s() * off_diag[i + 1].clone();
                                off_diag[i + 1] *= rot.c();
                            }

                            if let Some(ref mut q) = q_mat {
                                let rot =
                                    GivensRotation::new_unchecked(rot.c(), T::from_real(rot.s()));
                                rot.inverse().rotate_rows(&mut q.fixed_columns_mut::<2>(i));
                            }
                        }
                        None => {
                            break;
                        }
                    }
                }

                if off_diag[m].clone().norm1()
                    <= eps.clone() * (diag[m].clone().norm1() + diag[n].clone().norm1())
                {
                    end -= 1;
                }
            } else if subdim == 2 {
                let m = Matrix2::new(
                    diag[start].clone(),
                    off_diag[start].clone().conjugate(),
                    off_diag[start].clone(),
                    diag[start + 1].clone(),
                );
                let eigvals = m.eigenvalues().unwrap();
                let basis = Vector2::new(
                    eigvals.x.clone() - diag[start + 1].clone(),
                    off_diag[start].clone(),
                );

                diag[start] = eigvals[0].clone();
                diag[start + 1] = eigvals[1].clone();

                if let Some(ref mut q) = q_mat {
                    if let Some((rot, _)) =
                        GivensRotation::try_new(basis.x.clone(), basis.y.clone(), eps.clone())
                    {
                        let rot = GivensRotation::new_unchecked(rot.c(), T::from_real(rot.s()));
                        rot.rotate_rows(&mut q.fixed_columns_mut::<2>(start));
                    }
                }

                end -= 1;
            }

            // Re-delimit the subproblem in case some decoupling occurred.
            let sub = Self::delimit_subproblem(&diag, &mut off_diag, end, eps.clone());

            start = sub.0;
            end = sub.1;

            niter += 1;
            if niter == max_niter {
                return None;
            }
        }

        diag.scale_mut(m_amax);

        Some((diag, q_mat))
    }

    fn delimit_subproblem(
        diag: &OVector<T::RealField, D>,
        off_diag: &mut OVector<T::RealField, DimDiff<D, U1>>,
        end: usize,
        eps: T::RealField,
    ) -> (usize, usize)
    where
        D: DimSub<U1>,
        DefaultAllocator: Allocator<DimDiff<D, U1>>,
    {
        let mut n = end;

        while n > 0 {
            let m = n - 1;

            if off_diag[m].clone().norm1()
                > eps.clone() * (diag[n].clone().norm1() + diag[m].clone().norm1())
            {
                break;
            }

            n -= 1;
        }

        if n == 0 {
            return (0, 0);
        }

        let mut new_start = n - 1;
        while new_start > 0 {
            let m = new_start - 1;

            if off_diag[m].clone().is_zero()
                || off_diag[m].clone().norm1()
                    <= eps.clone() * (diag[new_start].clone().norm1() + diag[m].clone().norm1())
            {
                off_diag[m] = T::RealField::zero();
                break;
            }

            new_start -= 1;
        }

        (new_start, n)
    }

    /// Rebuild the original matrix from the eigendecomposition.
    ///
    /// Reconstructs the symmetric matrix using the formula **A = Q Λ Q<sup>T</sup>**, where:
    /// - **Q** is the matrix of eigenvectors (`self.eigenvectors`)
    /// - **Λ** is a diagonal matrix of eigenvalues (`self.eigenvalues`)
    /// - **Q<sup>T</sup>** is the transpose (or adjoint) of **Q**
    ///
    /// This is particularly useful for:
    /// - Verifying the correctness of the decomposition
    /// - Creating modified matrices by adjusting eigenvalues (e.g., filtering, damping)
    /// - Understanding what the eigendecomposition represents geometrically
    ///
    /// # Returns
    ///
    /// The reconstructed symmetric matrix. If the eigenvalues haven't been modified,
    /// this should match the original input matrix (within numerical precision).
    ///
    /// # Example: Verifying the Decomposition
    ///
    /// ```
    /// use nalgebra::Matrix3;
    ///
    /// let original = Matrix3::new(
    ///     5.0, 2.0, 1.0,
    ///     2.0, 6.0, 3.0,
    ///     1.0, 3.0, 7.0,
    /// );
    ///
    /// let eigen = original.symmetric_eigen();
    /// let reconstructed = eigen.recompose();
    ///
    /// // The reconstructed matrix should match the original
    /// assert!((original - reconstructed).norm() < 1e-10);
    /// ```
    ///
    /// # Example: Filtering Small Eigenvalues (Denoising)
    ///
    /// ```
    /// use nalgebra::Matrix3;
    ///
    /// // Covariance matrix with some numerical noise
    /// let covariance = Matrix3::new(
    ///     4.0, 1.0, 0.01,
    ///     1.0, 3.0, 0.02,
    ///     0.01, 0.02, 0.1,
    /// );
    ///
    /// let mut eigen = covariance.symmetric_eigen();
    ///
    /// // Filter out small eigenvalues (noise) by setting them to zero
    /// let threshold: f64 = 0.5;
    /// for i in 0..eigen.eigenvalues.len() {
    ///     if eigen.eigenvalues[i].abs() < threshold {
    ///         eigen.eigenvalues[i] = 0.0;
    ///     }
    /// }
    ///
    /// // Reconstruct the denoised matrix
    /// let denoised = eigen.recompose();
    ///
    /// println!("Original:\n{}", covariance);
    /// println!("Denoised:\n{}", denoised);
    /// ```
    ///
    /// # Example: Dimensionality Reduction (PCA)
    ///
    /// ```
    /// use nalgebra::Matrix3;
    ///
    /// let covariance = Matrix3::new(
    ///     10.0, 2.0, 1.0,
    ///      2.0, 5.0, 0.5,
    ///      1.0, 0.5, 1.0,
    /// );
    ///
    /// let mut eigen = covariance.symmetric_eigen();
    ///
    /// // Keep only the top 2 principal components by zeroing the smallest eigenvalue
    /// // (assumes eigenvalues are already sorted or we find the minimum)
    /// let min_idx = eigen.eigenvalues.imin();
    /// eigen.eigenvalues[min_idx] = 0.0;
    ///
    /// // Reconstruct with reduced dimensionality
    /// let reduced = eigen.recompose();
    ///
    /// // The reduced covariance explains most of the variance
    /// // but with simplified structure
    /// println!("Reduced covariance:\n{}", reduced);
    /// ```
    ///
    /// # Example: Ensuring Positive Definiteness
    ///
    /// ```
    /// use nalgebra::Matrix2;
    ///
    /// // Nearly positive definite matrix (might have small negative eigenvalues
    /// // due to numerical errors)
    /// let m = Matrix2::new(
    ///     2.0, 1.9,
    ///     1.9, 2.0,
    /// );
    ///
    /// let mut eigen = m.symmetric_eigen();
    ///
    /// // Ensure all eigenvalues are positive (required for some algorithms)
    /// for i in 0..eigen.eigenvalues.len() {
    ///     if eigen.eigenvalues[i] < 0.0 {
    ///         eigen.eigenvalues[i] = 1e-10;  // Small positive value
    ///     }
    /// }
    ///
    /// // Reconstruct the positive definite matrix
    /// let positive_definite = eigen.recompose();
    ///
    /// // Verify all eigenvalues are now positive
    /// let new_eigen = positive_definite.symmetric_eigen();
    /// for eigenvalue in new_eigen.eigenvalues.iter() {
    ///     assert!(*eigenvalue > 0.0);
    /// }
    /// ```
    ///
    /// # Example: Matrix Square Root
    ///
    /// ```
    /// use nalgebra::Matrix2;
    ///
    /// // Positive definite matrix
    /// let m = Matrix2::new(
    ///     5.0, 2.0,
    ///     2.0, 2.0,
    /// );
    ///
    /// let mut eigen = m.symmetric_eigen();
    ///
    /// // Take square root of eigenvalues to get matrix square root
    /// for i in 0..eigen.eigenvalues.len() {
    ///     let val: f64 = eigen.eigenvalues[i];
    ///     eigen.eigenvalues[i] = val.sqrt();
    /// }
    ///
    /// let sqrt_m = eigen.recompose();
    ///
    /// // Verify: sqrt(A) * sqrt(A) = A
    /// let reconstructed = &sqrt_m * &sqrt_m;
    /// assert!((m - reconstructed).norm() < 1e-10);
    /// ```
    ///
    /// # See Also
    ///
    /// - [`SymmetricEigen::new`]: Create the eigendecomposition
    /// - [`SymmetricEigen::try_new`]: Create with custom parameters
    /// - [`SquareMatrix::symmetric_eigenvalues`]: Compute only eigenvalues
    #[must_use]
    pub fn recompose(&self) -> OMatrix<T, D, D> {
        let mut u_t = self.eigenvectors.clone();
        for i in 0..self.eigenvalues.len() {
            let val = self.eigenvalues[i].clone();
            u_t.column_mut(i).scale_mut(val);
        }
        u_t.adjoint_mut();
        &self.eigenvectors * u_t
    }
}

/// Computes the Wilkinson shift for accelerating QR iteration convergence.
///
/// The Wilkinson shift is a sophisticated technique used in eigenvalue algorithms to
/// dramatically speed up convergence. It selects a strategic "shift" value that makes
/// the QR iteration algorithm converge much faster, especially for the smallest or
/// largest eigenvalue.
///
/// # What is a Shift?
///
/// In iterative eigenvalue algorithms, a "shift" is a value subtracted from the diagonal
/// before each iteration. The Wilkinson shift chooses this value intelligently by looking
/// at a 2×2 submatrix and picking the eigenvalue closest to the bottom-right corner.
/// This choice ensures rapid convergence.
///
/// # Technical Details
///
/// Given a 2×2 symmetric matrix:
/// ```text
/// ┌           ┐
/// │ tmm  tmn  │
/// │ tmn  tnn  │
/// └           ┘
/// ```
///
/// The function returns the eigenvalue of this matrix that is closer to `tnn`
/// (the bottom-right element). This is the optimal shift for QR iteration targeting
/// the bottom corner of a tridiagonal matrix.
///
/// # Arguments
///
/// * `tmm` - The top-left element (position [0,0])
/// * `tnn` - The bottom-right element (position [1,1])
/// * `tmn` - The off-diagonal element (position [0,1] or [1,0])
///
/// # Returns
///
/// The eigenvalue of the 2×2 matrix that is closest to `tnn`. If `tmn == 0`
/// (diagonal matrix), returns `tnn` directly.
///
/// # Example: Basic Usage
///
/// ```
/// use nalgebra::linalg::wilkinson_shift;
///
/// // 2x2 symmetric matrix:
/// // [ 3.0  1.0 ]
/// // [ 1.0  2.0 ]
/// let shift = wilkinson_shift(3.0, 2.0, 1.0);
///
/// // The eigenvalues of this matrix are approximately 1.382 and 3.618
/// // The shift will be ~1.382 (closer to tnn=2.0)
/// println!("Wilkinson shift: {}", shift);
/// let expected = 1.382_f64;
/// assert!((shift - expected).abs() < 0.01);
/// ```
///
/// # Example: Diagonal Matrix (Trivial Case)
///
/// ```
/// use nalgebra::linalg::wilkinson_shift;
///
/// // Diagonal matrix (off-diagonal is zero):
/// // [ 5.0  0.0 ]
/// // [ 0.0  3.0 ]
/// let shift = wilkinson_shift(5.0, 3.0, 0.0);
///
/// // For diagonal matrices, the shift is simply tnn
/// assert_eq!(shift, 3.0);
/// ```
///
/// # Example: Symmetric Matrix with Coupling
///
/// ```
/// use nalgebra::linalg::wilkinson_shift;
///
/// // Matrix with strong off-diagonal coupling:
/// // [ 10.0  8.0 ]
/// // [  8.0  6.0 ]
/// let shift = wilkinson_shift(10.0, 6.0, 8.0);
///
/// // The shift helps the algorithm converge quickly to the eigenvalues
/// println!("Shift for strongly coupled system: {}", shift);
///
/// // Verify it's one of the actual eigenvalues
/// // Eigenvalues are: 8 ± sqrt(68) ≈ 16.246 and -0.246
/// let expected_eigenvalue = -0.246_f64;
/// assert!((shift - expected_eigenvalue).abs() < 0.01);
/// ```
///
/// # Example: Understanding Convergence Acceleration
///
/// ```
/// use nalgebra::linalg::wilkinson_shift;
///
/// // Matrix from a tridiagonal eigenvalue problem:
/// // [ 4.0  2.0 ]
/// // [ 2.0  5.0 ]
/// let shift = wilkinson_shift(4.0, 5.0, 2.0);
///
/// // The Wilkinson shift is designed to make the off-diagonal
/// // element converge to zero quickly in QR iteration
/// println!("Optimal shift: {}", shift);
///
/// // This shift will be close to one eigenvalue, making
/// // that eigenvalue converge very quickly
/// ```
///
/// # Why This Matters
///
/// The Wilkinson shift is crucial for the efficiency of symmetric eigenvalue
/// decomposition algorithms. Without it, QR iteration can converge very slowly.
/// With it, convergence is typically quadratic - each iteration roughly doubles
/// the number of correct digits.
///
/// # Algorithm Background
///
/// The Wilkinson shift is computed using the formula:
/// ```text
/// δ = (tmm - tnn) / 2
/// shift = tnn - (tmn² / (δ + sign(δ) * sqrt(δ² + tmn²)))
/// ```
///
/// This formula is numerically stable and avoids catastrophic cancellation.
///
/// # See Also
///
/// - [`SymmetricEigen::new`]: Uses Wilkinson shifts internally
/// - [`SymmetricEigen::try_new`]: Custom convergence parameters
pub fn wilkinson_shift<T: ComplexField>(tmm: T, tnn: T, tmn: T) -> T {
    let sq_tmn = tmn.clone() * tmn;
    if !sq_tmn.is_zero() {
        // We have the guarantee that the denominator won't be zero.
        let d = (tmm - tnn.clone()) * crate::convert(0.5);
        tnn - sq_tmn.clone() / (d.clone() + d.clone().signum() * (d.clone() * d + sq_tmn).sqrt())
    } else {
        tnn
    }
}

/*
 *
 * Computations of eigenvalues for symmetric matrices.
 *
 */
impl<T: ComplexField, D: DimSub<U1>, S: Storage<T, D, D>> SquareMatrix<T, D, S>
where
    DefaultAllocator:
        Allocator<D, D> + Allocator<DimDiff<D, U1>> + Allocator<D> + Allocator<DimDiff<D, U1>>,
{
    /// Computes only the eigenvalues of this symmetric matrix (without eigenvectors).
    ///
    /// This is a faster alternative to [`SymmetricEigen::new`] when you only need the
    /// eigenvalues and don't need the eigenvectors. It uses the same robust algorithm but
    /// skips the eigenvector computation, saving both time and memory.
    ///
    /// # When to Use This
    ///
    /// Use this method when you need to:
    /// - Check matrix properties (positive definiteness, stability, condition number)
    /// - Compute matrix invariants (trace, determinant via eigenvalues)
    /// - Analyze spectral properties without needing the actual eigenvector directions
    /// - Process many matrices quickly where eigenvectors aren't needed
    ///
    /// # Performance
    ///
    /// Computing only eigenvalues is significantly faster than computing both eigenvalues
    /// and eigenvectors, especially for larger matrices. The time savings grow with matrix size.
    ///
    /// # Important Notes
    ///
    /// - Only the **lower-triangular part** (including diagonal) of the matrix is read
    /// - The eigenvalues are returned in **no particular order** (unsorted)
    /// - The matrix must be symmetric (or only the lower triangle needs to represent a symmetric matrix)
    ///
    /// # Returns
    ///
    /// A vector containing all eigenvalues of the matrix (unsorted).
    ///
    /// # Example: Basic Usage
    ///
    /// ```
    /// use nalgebra::Matrix3;
    ///
    /// let m = Matrix3::new(
    ///     4.0, 1.0, 2.0,
    ///     1.0, 5.0, 3.0,
    ///     2.0, 3.0, 6.0,
    /// );
    ///
    /// let eigenvalues = m.symmetric_eigenvalues();
    /// println!("Eigenvalues: {}", eigenvalues);
    ///
    /// // The sum of eigenvalues equals the trace
    /// let trace = m.trace();
    /// let sum_eigenvalues: f64 = eigenvalues.iter().sum();
    /// assert!((trace - sum_eigenvalues).abs() < 1e-10);
    /// ```
    ///
    /// # Example: Checking Positive Definiteness
    ///
    /// ```
    /// use nalgebra::Matrix3;
    ///
    /// let m = Matrix3::new(
    ///     5.0, 1.0, 0.5,
    ///     1.0, 4.0, 0.2,
    ///     0.5, 0.2, 3.0,
    /// );
    ///
    /// let eigenvalues = m.symmetric_eigenvalues();
    ///
    /// // A matrix is positive definite if all eigenvalues are positive
    /// let is_positive_definite = eigenvalues.iter().all(|&lambda| lambda > 0.0);
    /// println!("Is positive definite: {}", is_positive_definite);
    ///
    /// if is_positive_definite {
    ///     println!("Safe to use Cholesky decomposition!");
    /// }
    /// ```
    ///
    /// # Example: Computing Matrix Determinant
    ///
    /// ```
    /// use nalgebra::Matrix2;
    ///
    /// let m = Matrix2::new(
    ///     3.0, 1.0,
    ///     1.0, 2.0,
    /// );
    ///
    /// let eigenvalues = m.symmetric_eigenvalues();
    ///
    /// // The determinant is the product of all eigenvalues
    /// let det_via_eigenvalues: f64 = eigenvalues.iter().product();
    /// let det_direct = m.determinant();
    ///
    /// assert!((det_via_eigenvalues - det_direct).abs() < 1e-10);
    /// println!("Determinant: {}", det_via_eigenvalues);
    /// ```
    ///
    /// # Example: Condition Number Estimation
    ///
    /// ```
    /// use nalgebra::Matrix3;
    ///
    /// let m = Matrix3::new(
    ///     10.0, 1.0, 0.1,
    ///      1.0, 5.0, 0.2,
    ///      0.1, 0.2, 1.0,
    /// );
    ///
    /// let eigenvalues = m.symmetric_eigenvalues();
    ///
    /// // Find the largest and smallest eigenvalue magnitudes
    /// let max_eigenvalue = eigenvalues.iter()
    ///     .map(|&x| x.abs())
    ///     .fold(0.0, f64::max);
    /// let min_eigenvalue = eigenvalues.iter()
    ///     .map(|&x| x.abs())
    ///     .fold(f64::INFINITY, f64::min);
    ///
    /// // Condition number (for symmetric matrices)
    /// let condition_number = max_eigenvalue / min_eigenvalue;
    /// println!("Condition number: {:.2}", condition_number);
    ///
    /// if condition_number > 1000.0 {
    ///     println!("Warning: Matrix is ill-conditioned!");
    /// }
    /// ```
    ///
    /// # Example: Stability Analysis
    ///
    /// ```
    /// use nalgebra::Matrix2;
    ///
    /// // Jacobian matrix from a dynamical system
    /// let jacobian = Matrix2::new(
    ///     -2.0,  1.0,
    ///      1.0, -3.0,
    /// );
    ///
    /// let eigenvalues = jacobian.symmetric_eigenvalues();
    ///
    /// // System is stable if all eigenvalues are negative (for continuous systems)
    /// let is_stable = eigenvalues.iter().all(|&lambda| lambda < 0.0);
    /// println!("System is stable: {}", is_stable);
    ///
    /// if is_stable {
    ///     println!("The equilibrium point is stable.");
    /// }
    /// ```
    ///
    /// # Example: Spectral Radius
    ///
    /// ```
    /// use nalgebra::Matrix3;
    ///
    /// let m = Matrix3::new(
    ///     2.0, 1.0, 0.5,
    ///     1.0, 3.0, 0.2,
    ///     0.5, 0.2, 1.5,
    /// );
    ///
    /// let eigenvalues = m.symmetric_eigenvalues();
    ///
    /// // Spectral radius: maximum absolute value of eigenvalues
    /// let spectral_radius = eigenvalues.iter()
    ///     .map(|&x| x.abs())
    ///     .fold(0.0, f64::max);
    ///
    /// println!("Spectral radius: {:.4}", spectral_radius);
    ///
    /// // For iterative methods, convergence requires spectral radius < 1
    /// if spectral_radius < 1.0 {
    ///     println!("Iterative method will converge!");
    /// }
    /// ```
    ///
    /// # Example: Energy Levels in Quantum Systems
    ///
    /// ```
    /// use nalgebra::Matrix3;
    ///
    /// // Hamiltonian matrix for a 3-state quantum system
    /// let hamiltonian = Matrix3::new(
    ///     2.0, 0.5, 0.0,
    ///     0.5, 3.0, 0.5,
    ///     0.0, 0.5, 4.0,
    /// );
    ///
    /// // Eigenvalues represent energy levels
    /// let mut energy_levels = hamiltonian.symmetric_eigenvalues();
    ///
    /// // Sort to get ground state and excited states
    /// let mut levels_vec: Vec<f64> = energy_levels.iter().copied().collect();
    /// levels_vec.sort_by(|a, b| a.partial_cmp(b).unwrap());
    ///
    /// println!("Ground state energy: {:.4}", levels_vec[0]);
    /// println!("First excited state: {:.4}", levels_vec[1]);
    /// println!("Second excited state: {:.4}", levels_vec[2]);
    /// ```
    ///
    /// # Performance Comparison
    ///
    /// ```
    /// use nalgebra::Matrix3;
    ///
    /// let m = Matrix3::new(
    ///     4.0, 1.0, 0.5,
    ///     1.0, 3.0, 0.2,
    ///     0.5, 0.2, 2.0,
    /// );
    ///
    /// // Fast: only eigenvalues
    /// let eigenvalues = m.symmetric_eigenvalues();
    /// println!("Got {} eigenvalues", eigenvalues.len());
    ///
    /// // Slower: eigenvalues + eigenvectors
    /// let full_decomp = m.symmetric_eigen();
    /// println!("Got {} eigenvalues and eigenvectors", full_decomp.eigenvalues.len());
    ///
    /// // Both give the same eigenvalues
    /// ```
    ///
    /// # See Also
    ///
    /// - [`SymmetricEigen::new`]: Compute both eigenvalues and eigenvectors
    /// - [`SymmetricEigen::try_new`]: With custom convergence parameters
    /// - [`SymmetricEigen::recompose`]: Reconstruct matrix from decomposition
    #[must_use]
    pub fn symmetric_eigenvalues(&self) -> OVector<T::RealField, D> {
        SymmetricEigen::do_decompose(
            self.clone_owned(),
            false,
            T::RealField::default_epsilon(),
            0,
        )
        .unwrap()
        .0
    }
}

#[cfg(test)]
mod test {
    use crate::base::Matrix2;

    fn expected_shift(m: Matrix2<f64>) -> f64 {
        let vals = m.eigenvalues().unwrap();

        if (vals.x - m.m22).abs() < (vals.y - m.m22).abs() {
            vals.x
        } else {
            vals.y
        }
    }

    #[cfg(feature = "rand")]
    #[test]
    fn wilkinson_shift_random() {
        for _ in 0..1000 {
            let m = Matrix2::<f64>::new_random();
            let m = m * m.transpose();

            let expected = expected_shift(m);
            let computed = super::wilkinson_shift(m.m11, m.m22, m.m12);
            assert!(relative_eq!(expected, computed, epsilon = 1.0e-7));
        }
    }

    #[test]
    fn wilkinson_shift_zero() {
        let m = Matrix2::new(0.0, 0.0, 0.0, 0.0);
        assert!(relative_eq!(
            expected_shift(m),
            super::wilkinson_shift(m.m11, m.m22, m.m12)
        ));
    }

    #[test]
    fn wilkinson_shift_zero_diagonal() {
        let m = Matrix2::new(0.0, 42.0, 42.0, 0.0);
        assert!(relative_eq!(
            expected_shift(m),
            super::wilkinson_shift(m.m11, m.m22, m.m12)
        ));
    }

    #[test]
    fn wilkinson_shift_zero_off_diagonal() {
        let m = Matrix2::new(42.0, 0.0, 0.0, 64.0);
        assert!(relative_eq!(
            expected_shift(m),
            super::wilkinson_shift(m.m11, m.m22, m.m12)
        ));
    }

    #[test]
    fn wilkinson_shift_zero_trace() {
        let m = Matrix2::new(42.0, 20.0, 20.0, -42.0);
        assert!(relative_eq!(
            expected_shift(m),
            super::wilkinson_shift(m.m11, m.m22, m.m12)
        ));
    }

    #[test]
    fn wilkinson_shift_zero_diag_diff_and_zero_off_diagonal() {
        let m = Matrix2::new(42.0, 0.0, 0.0, 42.0);
        assert!(relative_eq!(
            expected_shift(m),
            super::wilkinson_shift(m.m11, m.m22, m.m12)
        ));
    }

    #[test]
    fn wilkinson_shift_zero_det() {
        let m = Matrix2::new(2.0, 4.0, 4.0, 8.0);
        assert!(relative_eq!(
            expected_shift(m),
            super::wilkinson_shift(m.m11, m.m22, m.m12)
        ));
    }
}
