use simba::scalar::ComplexField;

use crate::base::allocator::Allocator;
use crate::base::dimension::Dim;
use crate::base::storage::{Storage, StorageMut};
use crate::base::{DefaultAllocator, OMatrix, SquareMatrix};

use crate::linalg::lu;

impl<T: ComplexField, D: Dim, S: Storage<T, D, D>> SquareMatrix<T, D, S> {
    /// Attempts to invert this square matrix, returning a new matrix.
    ///
    /// # What is Matrix Inversion?
    ///
    /// Matrix inversion is the process of finding a matrix **A⁻¹** such that when you multiply
    /// it by the original matrix **A**, you get the identity matrix: **A × A⁻¹ = I**.
    ///
    /// Think of it like finding the reciprocal of a number: just as `5 × (1/5) = 1`, matrix
    /// inversion finds the "reciprocal" of a matrix. Not all matrices have inverses - only
    /// square matrices with non-zero determinants can be inverted (called "non-singular" or
    /// "invertible" matrices).
    ///
    /// # Common Use Cases
    ///
    /// - **Solving linear equations**: If `A × x = b`, then `x = A⁻¹ × b`
    /// - **Undoing transformations**: Invert a transformation matrix to reverse its effect
    /// - **Coordinate system conversions**: Convert between different coordinate frames
    /// - **Computer graphics**: Invert camera or model matrices for rendering
    ///
    /// # Return Value
    ///
    /// Returns `Some(inverted_matrix)` if the matrix is invertible, or `None` if the matrix
    /// is singular (determinant is zero or numerically close to zero).
    ///
    /// This method consumes the original matrix and returns a new one. If you want to invert
    /// a matrix in-place to save memory, use [`try_inverse_mut`](Self::try_inverse_mut) instead.
    ///
    /// # Examples
    ///
    /// ## Basic 2×2 Matrix Inversion
    ///
    /// ```
    /// use nalgebra::Matrix2;
    ///
    /// let m = Matrix2::new(1.0, 2.0,
    ///                      3.0, 4.0);
    ///
    /// let inv = m.try_inverse().expect("Matrix should be invertible");
    ///
    /// // Verify that m * inv = identity
    /// let identity = m * inv;
    /// let expected_identity = Matrix2::identity();
    ///
    /// assert!((identity - expected_identity).norm() < 1e-10);
    /// ```
    ///
    /// ## Solving a System of Linear Equations
    ///
    /// Given the system: `2x + 3y = 8` and `4x + 5y = 14`, solve for x and y.
    ///
    /// ```
    /// use nalgebra::{Matrix2, Vector2};
    ///
    /// // Coefficient matrix A
    /// let a = Matrix2::new(2.0, 3.0,
    ///                      4.0, 5.0);
    ///
    /// // Right-hand side vector b
    /// let b = Vector2::new(8.0, 14.0);
    ///
    /// // Solve A*x = b using x = A⁻¹ * b
    /// if let Some(a_inv) = a.try_inverse() {
    ///     let x = a_inv * b;
    ///     println!("x = {}, y = {}", x[0], x[1]); // x = 1.0, y = 2.0
    ///
    ///     // Verify the solution
    ///     assert!((a * x - b).norm() < 1e-10);
    /// }
    /// ```
    ///
    /// ## Handling Singular (Non-Invertible) Matrices
    ///
    /// ```
    /// use nalgebra::Matrix2;
    ///
    /// // This matrix has determinant = 0, so it's singular
    /// let singular = Matrix2::new(1.0, 2.0,
    ///                             2.0, 4.0);  // Second row is 2× the first row
    ///
    /// assert!(singular.try_inverse().is_none());
    /// ```
    ///
    /// ## Undoing a Transformation
    ///
    /// ```
    /// use nalgebra::{Matrix3, Vector3};
    ///
    /// // A transformation matrix (rotation + scaling)
    /// let transform = Matrix3::new(2.0,  0.0, 0.0,
    ///                              0.0,  3.0, 0.0,
    ///                              0.0,  0.0, 1.0);
    ///
    /// // Apply transformation to a point
    /// let point = Vector3::new(1.0, 1.0, 1.0);
    /// let transformed = transform * point;
    ///
    /// // Undo the transformation using the inverse
    /// if let Some(inverse) = transform.try_inverse() {
    ///     let original = inverse * transformed;
    ///     assert!((original - point).norm() < 1e-10);
    /// }
    /// ```
    ///
    /// # Performance Notes
    ///
    /// The implementation is optimized for different matrix sizes:
    /// - Dimensions 1-4: Uses specialized, unrolled formulas (very fast)
    /// - Dimension 5+: Uses LU decomposition (general-purpose algorithm)
    ///
    /// # Panics
    ///
    /// Panics if `self` isn't a square matrix.
    ///
    /// # See Also
    ///
    /// * [`try_inverse_mut`](Self::try_inverse_mut) - In-place version that modifies the matrix
    /// * [`pseudo_inverse`](crate::Matrix::pseudo_inverse) - Generalized inverse for non-square matrices
    /// * [`SVD::solve`](crate::linalg::SVD::solve) - Alternative approach for solving linear systems
    #[inline]
    #[must_use = "Did you mean to use try_inverse_mut()?"]
    pub fn try_inverse(self) -> Option<OMatrix<T, D, D>>
    where
        DefaultAllocator: Allocator<D, D>,
    {
        let mut me = self.into_owned();
        if me.try_inverse_mut() { Some(me) } else { None }
    }
}

impl<T: ComplexField, D: Dim, S: StorageMut<T, D, D>> SquareMatrix<T, D, S> {
    /// Attempts to invert this square matrix in-place, modifying the original matrix.
    ///
    /// # What is Matrix Inversion?
    ///
    /// Matrix inversion is the process of finding a matrix **A⁻¹** such that when you multiply
    /// it by the original matrix **A**, you get the identity matrix: **A × A⁻¹ = I**.
    ///
    /// Think of it like finding the reciprocal of a number: just as `5 × (1/5) = 1`, matrix
    /// inversion finds the "reciprocal" of a matrix. This method performs the inversion
    /// in-place, replacing the original matrix with its inverse to save memory.
    ///
    /// # Common Use Cases
    ///
    /// - **Memory-efficient inversion**: When you don't need the original matrix anymore
    /// - **Solving linear equations**: Transform `A × x = b` into `x = A⁻¹ × b`
    /// - **Coordinate transformations**: Reverse coordinate system changes efficiently
    /// - **Real-time applications**: Minimize memory allocations in performance-critical code
    ///
    /// # Return Value
    ///
    /// Returns `true` if the inversion succeeded (matrix was invertible), or `false` if the
    /// matrix is singular (determinant is zero or numerically close to zero).
    ///
    /// **Important**: When the function returns `true`, the matrix has been replaced with its
    /// inverse. When it returns `false`, the matrix content depends on its size (see Behavior below).
    ///
    /// # Behavior
    ///
    /// The behavior when inversion fails depends on the matrix dimension:
    ///
    /// - **For small dimensions (`n < 5`)**: The matrix is left **unchanged** if inversion fails.
    ///   You can safely retry or use the original values.
    ///
    /// - **For dimensions `n >= 5`**: The matrix may be **partially modified** even if inversion
    ///   fails, because LU decomposition is used and it modifies the matrix in-place during
    ///   computation.
    ///
    /// If you need to preserve the original matrix regardless of success or failure,
    /// consider using [`try_inverse`](Self::try_inverse) instead, which returns a new matrix
    /// and leaves the original untouched.
    ///
    /// # Examples
    ///
    /// ## Basic In-Place Matrix Inversion
    ///
    /// ```
    /// use nalgebra::Matrix2;
    ///
    /// let mut m = Matrix2::new(1.0, 2.0,
    ///                          3.0, 4.0);
    ///
    /// let original = m.clone();
    ///
    /// // Invert the matrix in-place
    /// let success = m.try_inverse_mut();
    /// assert!(success);
    ///
    /// // Verify that original * m = identity
    /// let identity = original * m;
    /// let expected = Matrix2::identity();
    /// assert!((identity - expected).norm() < 1e-10);
    /// ```
    ///
    /// ## Memory-Efficient Solving of Linear Systems
    ///
    /// ```
    /// use nalgebra::{Matrix3, Vector3};
    ///
    /// // Solve A*x = b for a 3×3 system
    /// let mut a = Matrix3::new(2.0, 1.0, 1.0,
    ///                          1.0, 3.0, 2.0,
    ///                          1.0, 0.0, 0.0);
    ///
    /// let b = Vector3::new(4.0, 5.0, 6.0);
    ///
    /// // Invert A in-place to save memory
    /// if a.try_inverse_mut() {
    ///     let x = a * b;  // a is now A⁻¹
    ///     println!("Solution: x = {:?}", x);
    /// } else {
    ///     println!("System has no unique solution");
    /// }
    /// ```
    ///
    /// ## Handling Singular Matrices (Small Dimensions)
    ///
    /// ```
    /// use nalgebra::Matrix2;
    ///
    /// // This matrix has determinant = 0, so it's singular
    /// let mut singular = Matrix2::new(1.0, 2.0,
    ///                                 2.0, 4.0);
    /// let original = singular.clone();
    ///
    /// // Try to invert - will fail
    /// let success = singular.try_inverse_mut();
    /// assert!(!success);
    ///
    /// // For small matrices (< 5×5), original is preserved on failure
    /// assert_eq!(singular, original);
    /// ```
    ///
    /// ## Batch Processing Multiple Matrices
    ///
    /// ```
    /// use nalgebra::Matrix3;
    ///
    /// let mut matrices = vec![
    ///     Matrix3::new(1.0, 0.0, 0.0,  0.0, 1.0, 0.0,  0.0, 0.0, 1.0),
    ///     Matrix3::new(2.0, 0.0, 0.0,  0.0, 2.0, 0.0,  0.0, 0.0, 2.0),
    ///     Matrix3::new(1.0, 2.0, 3.0,  0.0, 1.0, 4.0,  0.0, 0.0, 1.0),
    /// ];
    ///
    /// // Invert all matrices in-place
    /// for matrix in matrices.iter_mut() {
    ///     if !matrix.try_inverse_mut() {
    ///         println!("Warning: Found a singular matrix!");
    ///     }
    /// }
    /// ```
    ///
    /// ## Coordinate System Transformation
    ///
    /// ```
    /// use nalgebra::{Matrix4, Vector4};
    ///
    /// // Transform matrix (e.g., from camera space to world space)
    /// let mut camera_to_world = Matrix4::new(
    ///     1.0, 0.0, 0.0, 10.0,
    ///     0.0, 1.0, 0.0, 5.0,
    ///     0.0, 0.0, 1.0, 2.0,
    ///     0.0, 0.0, 0.0, 1.0,
    /// );
    ///
    /// // Invert to get world-to-camera transform in-place
    /// if camera_to_world.try_inverse_mut() {
    ///     // camera_to_world is now world_to_camera
    ///     let world_point = Vector4::new(15.0, 8.0, 4.0, 1.0);
    ///     let camera_point = camera_to_world * world_point;
    ///     println!("Point in camera space: {:?}", camera_point);
    /// }
    /// ```
    ///
    /// # Performance Notes
    ///
    /// The implementation is optimized for different matrix sizes:
    /// - **Dimensions 1-4**: Uses specialized, unrolled formulas (very fast, no allocations)
    /// - **Dimension 5+**: Uses LU decomposition (general-purpose, some internal allocations)
    ///
    /// This method is more memory-efficient than [`try_inverse`](Self::try_inverse) because it
    /// doesn't allocate a new matrix, making it ideal for performance-critical applications.
    ///
    /// # Panics
    ///
    /// Panics if `self` isn't a square matrix.
    ///
    /// # See Also
    ///
    /// * [`try_inverse`](Self::try_inverse) - Non-mutating version that returns a new matrix
    /// * [`pseudo_inverse`](crate::Matrix::pseudo_inverse) - Generalized inverse for non-square matrices
    /// * [`LU::try_inverse`](crate::linalg::LU::try_inverse) - Inverse computation using pre-computed LU decomposition
    /// * [`SVD::solve`](crate::linalg::SVD::solve) - Alternative for solving linear systems
    #[inline]
    pub fn try_inverse_mut(&mut self) -> bool
    where
        DefaultAllocator: Allocator<D, D>,
    {
        assert!(self.is_square(), "Unable to invert a non-square matrix.");

        let dim = self.shape().0;

        unsafe {
            match dim {
                0 => true,
                1 => {
                    let determinant = self.get_unchecked((0, 0)).clone();
                    if determinant.is_zero() {
                        false
                    } else {
                        *self.get_unchecked_mut((0, 0)) = T::one() / determinant;
                        true
                    }
                }
                2 => {
                    let m11 = self.get_unchecked((0, 0)).clone();
                    let m12 = self.get_unchecked((0, 1)).clone();
                    let m21 = self.get_unchecked((1, 0)).clone();
                    let m22 = self.get_unchecked((1, 1)).clone();

                    let determinant = m11.clone() * m22.clone() - m21.clone() * m12.clone();

                    if determinant.is_zero() {
                        false
                    } else {
                        *self.get_unchecked_mut((0, 0)) = m22 / determinant.clone();
                        *self.get_unchecked_mut((0, 1)) = -m12 / determinant.clone();

                        *self.get_unchecked_mut((1, 0)) = -m21 / determinant.clone();
                        *self.get_unchecked_mut((1, 1)) = m11 / determinant;

                        true
                    }
                }
                3 => {
                    let m11 = self.get_unchecked((0, 0)).clone();
                    let m12 = self.get_unchecked((0, 1)).clone();
                    let m13 = self.get_unchecked((0, 2)).clone();

                    let m21 = self.get_unchecked((1, 0)).clone();
                    let m22 = self.get_unchecked((1, 1)).clone();
                    let m23 = self.get_unchecked((1, 2)).clone();

                    let m31 = self.get_unchecked((2, 0)).clone();
                    let m32 = self.get_unchecked((2, 1)).clone();
                    let m33 = self.get_unchecked((2, 2)).clone();

                    let minor_m12_m23 = m22.clone() * m33.clone() - m32.clone() * m23.clone();
                    let minor_m11_m23 = m21.clone() * m33.clone() - m31.clone() * m23.clone();
                    let minor_m11_m22 = m21.clone() * m32.clone() - m31.clone() * m22.clone();

                    let determinant = m11.clone() * minor_m12_m23.clone()
                        - m12.clone() * minor_m11_m23.clone()
                        + m13.clone() * minor_m11_m22.clone();

                    if determinant.is_zero() {
                        false
                    } else {
                        *self.get_unchecked_mut((0, 0)) = minor_m12_m23 / determinant.clone();
                        *self.get_unchecked_mut((0, 1)) = (m13.clone() * m32.clone()
                            - m33.clone() * m12.clone())
                            / determinant.clone();
                        *self.get_unchecked_mut((0, 2)) = (m12.clone() * m23.clone()
                            - m22.clone() * m13.clone())
                            / determinant.clone();

                        *self.get_unchecked_mut((1, 0)) = -minor_m11_m23 / determinant.clone();
                        *self.get_unchecked_mut((1, 1)) =
                            (m11.clone() * m33 - m31.clone() * m13.clone()) / determinant.clone();
                        *self.get_unchecked_mut((1, 2)) =
                            (m13 * m21.clone() - m23 * m11.clone()) / determinant.clone();

                        *self.get_unchecked_mut((2, 0)) = minor_m11_m22 / determinant.clone();
                        *self.get_unchecked_mut((2, 1)) =
                            (m12.clone() * m31 - m32 * m11.clone()) / determinant.clone();
                        *self.get_unchecked_mut((2, 2)) = (m11 * m22 - m21 * m12) / determinant;

                        true
                    }
                }
                4 => {
                    let oself = self.clone_owned();
                    do_inverse4(&oself, self)
                }
                _ => {
                    let oself = self.clone_owned();
                    lu::try_invert_to(oself, self)
                }
            }
        }
    }
}

// NOTE: this is an extremely efficient, loop-unrolled matrix inverse from MESA (MIT licensed).
fn do_inverse4<T: ComplexField, D: Dim, S: StorageMut<T, D, D>>(
    m: &OMatrix<T, D, D>,
    out: &mut SquareMatrix<T, D, S>,
) -> bool
where
    DefaultAllocator: Allocator<D, D>,
{
    let m = m.as_slice();

    let cofactor00 = m[5].clone() * m[10].clone() * m[15].clone()
        - m[5].clone() * m[11].clone() * m[14].clone()
        - m[9].clone() * m[6].clone() * m[15].clone()
        + m[9].clone() * m[7].clone() * m[14].clone()
        + m[13].clone() * m[6].clone() * m[11].clone()
        - m[13].clone() * m[7].clone() * m[10].clone();

    let cofactor01 = -m[4].clone() * m[10].clone() * m[15].clone()
        + m[4].clone() * m[11].clone() * m[14].clone()
        + m[8].clone() * m[6].clone() * m[15].clone()
        - m[8].clone() * m[7].clone() * m[14].clone()
        - m[12].clone() * m[6].clone() * m[11].clone()
        + m[12].clone() * m[7].clone() * m[10].clone();

    let cofactor02 = m[4].clone() * m[9].clone() * m[15].clone()
        - m[4].clone() * m[11].clone() * m[13].clone()
        - m[8].clone() * m[5].clone() * m[15].clone()
        + m[8].clone() * m[7].clone() * m[13].clone()
        + m[12].clone() * m[5].clone() * m[11].clone()
        - m[12].clone() * m[7].clone() * m[9].clone();

    let cofactor03 = -m[4].clone() * m[9].clone() * m[14].clone()
        + m[4].clone() * m[10].clone() * m[13].clone()
        + m[8].clone() * m[5].clone() * m[14].clone()
        - m[8].clone() * m[6].clone() * m[13].clone()
        - m[12].clone() * m[5].clone() * m[10].clone()
        + m[12].clone() * m[6].clone() * m[9].clone();

    let det = m[0].clone() * cofactor00.clone()
        + m[1].clone() * cofactor01.clone()
        + m[2].clone() * cofactor02.clone()
        + m[3].clone() * cofactor03.clone();

    if det.is_zero() {
        return false;
    }
    out[(0, 0)] = cofactor00;

    out[(1, 0)] = -m[1].clone() * m[10].clone() * m[15].clone()
        + m[1].clone() * m[11].clone() * m[14].clone()
        + m[9].clone() * m[2].clone() * m[15].clone()
        - m[9].clone() * m[3].clone() * m[14].clone()
        - m[13].clone() * m[2].clone() * m[11].clone()
        + m[13].clone() * m[3].clone() * m[10].clone();

    out[(2, 0)] = m[1].clone() * m[6].clone() * m[15].clone()
        - m[1].clone() * m[7].clone() * m[14].clone()
        - m[5].clone() * m[2].clone() * m[15].clone()
        + m[5].clone() * m[3].clone() * m[14].clone()
        + m[13].clone() * m[2].clone() * m[7].clone()
        - m[13].clone() * m[3].clone() * m[6].clone();

    out[(3, 0)] = -m[1].clone() * m[6].clone() * m[11].clone()
        + m[1].clone() * m[7].clone() * m[10].clone()
        + m[5].clone() * m[2].clone() * m[11].clone()
        - m[5].clone() * m[3].clone() * m[10].clone()
        - m[9].clone() * m[2].clone() * m[7].clone()
        + m[9].clone() * m[3].clone() * m[6].clone();

    out[(0, 1)] = cofactor01;

    out[(1, 1)] = m[0].clone() * m[10].clone() * m[15].clone()
        - m[0].clone() * m[11].clone() * m[14].clone()
        - m[8].clone() * m[2].clone() * m[15].clone()
        + m[8].clone() * m[3].clone() * m[14].clone()
        + m[12].clone() * m[2].clone() * m[11].clone()
        - m[12].clone() * m[3].clone() * m[10].clone();

    out[(2, 1)] = -m[0].clone() * m[6].clone() * m[15].clone()
        + m[0].clone() * m[7].clone() * m[14].clone()
        + m[4].clone() * m[2].clone() * m[15].clone()
        - m[4].clone() * m[3].clone() * m[14].clone()
        - m[12].clone() * m[2].clone() * m[7].clone()
        + m[12].clone() * m[3].clone() * m[6].clone();

    out[(3, 1)] = m[0].clone() * m[6].clone() * m[11].clone()
        - m[0].clone() * m[7].clone() * m[10].clone()
        - m[4].clone() * m[2].clone() * m[11].clone()
        + m[4].clone() * m[3].clone() * m[10].clone()
        + m[8].clone() * m[2].clone() * m[7].clone()
        - m[8].clone() * m[3].clone() * m[6].clone();

    out[(0, 2)] = cofactor02;

    out[(1, 2)] = -m[0].clone() * m[9].clone() * m[15].clone()
        + m[0].clone() * m[11].clone() * m[13].clone()
        + m[8].clone() * m[1].clone() * m[15].clone()
        - m[8].clone() * m[3].clone() * m[13].clone()
        - m[12].clone() * m[1].clone() * m[11].clone()
        + m[12].clone() * m[3].clone() * m[9].clone();

    out[(2, 2)] = m[0].clone() * m[5].clone() * m[15].clone()
        - m[0].clone() * m[7].clone() * m[13].clone()
        - m[4].clone() * m[1].clone() * m[15].clone()
        + m[4].clone() * m[3].clone() * m[13].clone()
        + m[12].clone() * m[1].clone() * m[7].clone()
        - m[12].clone() * m[3].clone() * m[5].clone();

    out[(0, 3)] = cofactor03;

    out[(3, 2)] = -m[0].clone() * m[5].clone() * m[11].clone()
        + m[0].clone() * m[7].clone() * m[9].clone()
        + m[4].clone() * m[1].clone() * m[11].clone()
        - m[4].clone() * m[3].clone() * m[9].clone()
        - m[8].clone() * m[1].clone() * m[7].clone()
        + m[8].clone() * m[3].clone() * m[5].clone();

    out[(1, 3)] = m[0].clone() * m[9].clone() * m[14].clone()
        - m[0].clone() * m[10].clone() * m[13].clone()
        - m[8].clone() * m[1].clone() * m[14].clone()
        + m[8].clone() * m[2].clone() * m[13].clone()
        + m[12].clone() * m[1].clone() * m[10].clone()
        - m[12].clone() * m[2].clone() * m[9].clone();

    out[(2, 3)] = -m[0].clone() * m[5].clone() * m[14].clone()
        + m[0].clone() * m[6].clone() * m[13].clone()
        + m[4].clone() * m[1].clone() * m[14].clone()
        - m[4].clone() * m[2].clone() * m[13].clone()
        - m[12].clone() * m[1].clone() * m[6].clone()
        + m[12].clone() * m[2].clone() * m[5].clone();

    out[(3, 3)] = m[0].clone() * m[5].clone() * m[10].clone()
        - m[0].clone() * m[6].clone() * m[9].clone()
        - m[4].clone() * m[1].clone() * m[10].clone()
        + m[4].clone() * m[2].clone() * m[9].clone()
        + m[8].clone() * m[1].clone() * m[6].clone()
        - m[8].clone() * m[2].clone() * m[5].clone();

    let inv_det = T::one() / det;

    for j in 0..4 {
        for i in 0..4 {
            out[(i, j)] *= inv_det.clone();
        }
    }
    true
}
