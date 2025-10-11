// Matrix properties checks.
use approx::RelativeEq;
use num::{One, Zero};

use simba::scalar::{ClosedAddAssign, ClosedMulAssign, ComplexField, RealField};

use crate::RawStorage;
use crate::base::allocator::Allocator;
use crate::base::dimension::{Dim, DimMin};
use crate::base::storage::Storage;
use crate::base::{DefaultAllocator, Matrix, SquareMatrix};

impl<T, R: Dim, C: Dim, S: RawStorage<T, R, C>> Matrix<T, R, C, S> {
    /// Returns the total number of elements in this matrix.
    ///
    /// This method calculates the total number of elements by multiplying the number of rows
    /// by the number of columns. For example, a 3×4 matrix has 12 elements, and a 2×2 matrix
    /// has 4 elements.
    ///
    /// This is useful when you need to know the size of the underlying storage, iterate over
    /// all elements, or allocate buffers for matrix operations.
    ///
    /// # Examples
    ///
    /// Basic usage with different matrix sizes:
    ///
    /// ```
    /// # use nalgebra::{Matrix3x4, Matrix2, Matrix3, DMatrix};
    /// // A 3×4 matrix has 12 elements (3 rows × 4 columns)
    /// let mat = Matrix3x4::<f32>::zeros();
    /// assert_eq!(mat.len(), 12);
    ///
    /// // A 2×2 matrix has 4 elements
    /// let square_mat = Matrix2::<f64>::zeros();
    /// assert_eq!(square_mat.len(), 4);
    ///
    /// // A 3×3 matrix has 9 elements
    /// let mat3 = Matrix3::<i32>::zeros();
    /// assert_eq!(mat3.len(), 9);
    /// ```
    ///
    /// With dynamic matrices:
    ///
    /// ```
    /// # use nalgebra::DMatrix;
    /// // Create a dynamic 5×7 matrix
    /// let dmat = DMatrix::<f32>::zeros(5, 7);
    /// assert_eq!(dmat.len(), 35);
    /// ```
    ///
    /// # See Also
    ///
    /// * [`is_empty`](Self::is_empty) - Check if the matrix contains no elements
    /// * [`shape`](Self::shape) - Get the number of rows and columns separately
    /// * [`nrows`](Self::nrows) - Get just the number of rows
    /// * [`ncols`](Self::ncols) - Get just the number of columns
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        let (nrows, ncols) = self.shape();
        nrows * ncols
    }

    /// Returns `true` if the matrix contains no elements.
    ///
    /// A matrix is considered empty if either its number of rows or columns is zero, which
    /// means it has a total size of 0 elements. In practice, empty matrices are rare in
    /// nalgebra since most matrix types have fixed, non-zero dimensions.
    ///
    /// This method is primarily useful when working with dynamically-sized matrices where
    /// dimensions might be determined at runtime.
    ///
    /// # Examples
    ///
    /// Static matrices are typically never empty:
    ///
    /// ```
    /// # use nalgebra::{Matrix3x4, Matrix2, Matrix1};
    /// // A 3×4 matrix is not empty
    /// let mat = Matrix3x4::<f32>::zeros();
    /// assert!(!mat.is_empty());
    ///
    /// // Even a 1×1 matrix is not empty
    /// let small_mat = Matrix1::<f64>::new(5.0);
    /// assert!(!small_mat.is_empty());
    /// ```
    ///
    /// Dynamic matrices can be empty:
    ///
    /// ```
    /// # use nalgebra::DMatrix;
    /// // A dynamic matrix with 0 rows is empty
    /// let empty_mat = DMatrix::<f32>::zeros(0, 5);
    /// assert!(empty_mat.is_empty());
    ///
    /// // A dynamic matrix with 0 columns is also empty
    /// let empty_mat2 = DMatrix::<f32>::zeros(3, 0);
    /// assert!(empty_mat2.is_empty());
    ///
    /// // A non-empty dynamic matrix
    /// let non_empty = DMatrix::<f32>::zeros(3, 4);
    /// assert!(!non_empty.is_empty());
    /// ```
    ///
    /// # See Also
    ///
    /// * [`len`](Self::len) - Get the total number of elements
    /// * [`shape`](Self::shape) - Get the dimensions (rows and columns)
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns `true` if this is a square matrix.
    ///
    /// A square matrix is one where the number of rows equals the number of columns.
    /// For example, 2×2, 3×3, and 4×4 matrices are all square matrices.
    ///
    /// Square matrices are important in linear algebra because they have special properties:
    /// they can have determinants, eigenvalues, and traces. Many matrix operations like
    /// computing inverses, determinants, and matrix exponentials only work on square matrices.
    ///
    /// # Examples
    ///
    /// Square matrices:
    ///
    /// ```
    /// # use nalgebra::{Matrix2, Matrix3, Matrix4, DMatrix};
    /// // A 2×2 matrix is square
    /// let mat2x2 = Matrix2::<f64>::new(
    ///     1.0, 2.0,
    ///     3.0, 4.0,
    /// );
    /// assert!(mat2x2.is_square());
    ///
    /// // A 3×3 matrix is square
    /// let mat3x3 = Matrix3::<f32>::zeros();
    /// assert!(mat3x3.is_square());
    ///
    /// // A dynamic 4×4 matrix is square
    /// let dmat = DMatrix::<f64>::identity(4, 4);
    /// assert!(dmat.is_square());
    /// ```
    ///
    /// Non-square (rectangular) matrices:
    ///
    /// ```
    /// # use nalgebra::{Matrix2x3, Matrix3x4, DMatrix};
    /// // A 2×3 matrix is not square
    /// let mat2x3 = Matrix2x3::<f64>::zeros();
    /// assert!(!mat2x3.is_square());
    ///
    /// // A 3×4 matrix is not square
    /// let mat3x4 = Matrix3x4::<f32>::zeros();
    /// assert!(!mat3x4.is_square());
    ///
    /// // A dynamic 5×3 matrix is not square
    /// let dmat = DMatrix::<f64>::zeros(5, 3);
    /// assert!(!dmat.is_square());
    /// ```
    ///
    /// # See Also
    ///
    /// * [`is_identity`](Self::is_identity) - Check if this is an identity matrix (only valid for square matrices)
    /// * [`trace`](Self::trace) - Compute the trace (sum of diagonal elements) of a square matrix
    /// * [`determinant`](Self::determinant) - Compute the determinant of a square matrix
    #[inline]
    #[must_use]
    pub fn is_square(&self) -> bool {
        let (nrows, ncols) = self.shape();
        nrows == ncols
    }

    // TODO: RelativeEq prevents us from using those methods on integer matrices…
    /// Returns `true` if this matrix is approximately equal to the identity matrix.
    ///
    /// An identity matrix is a matrix where all diagonal elements equal 1 and all off-diagonal
    /// elements equal 0. The identity matrix is denoted as I and has the special property that
    /// multiplying any matrix by I leaves it unchanged: A × I = I × A = A.
    ///
    /// This method checks if the matrix is "close enough" to an identity matrix within a relative
    /// error tolerance of `eps`. This is important because floating-point arithmetic can introduce
    /// small rounding errors, so exact equality checks often fail for computed results.
    ///
    /// # Parameters
    ///
    /// * `eps` - The relative error tolerance. Smaller values require stricter matching.
    ///   Common values are `1.0e-5` for `f32` or `1.0e-12` for `f64`.
    ///
    /// # How It Works
    ///
    /// For each element in the matrix:
    /// - Diagonal elements (where row index equals column index) should be approximately 1
    /// - Off-diagonal elements (where row index differs from column index) should be approximately 0
    ///
    /// For non-square matrices, this only checks elements up to `min(nrows, ncols)` on the diagonal.
    ///
    /// # Examples
    ///
    /// Checking exact identity matrices:
    ///
    /// ```
    /// # use nalgebra::{Matrix2, Matrix3};
    /// // A 2×2 identity matrix
    /// let id2 = Matrix2::<f64>::identity();
    /// assert!(id2.is_identity(1.0e-10));
    ///
    /// // A 3×3 identity matrix
    /// let id3 = Matrix3::<f32>::identity();
    /// assert!(id3.is_identity(1.0e-5));
    /// ```
    ///
    /// Checking matrices that are not identity matrices:
    ///
    /// ```
    /// # use nalgebra::Matrix2;
    /// // A matrix with different diagonal values
    /// let mat = Matrix2::new(
    ///     2.0, 0.0,
    ///     0.0, 2.0,
    /// );
    /// assert!(!mat.is_identity(1.0e-10));
    ///
    /// // A matrix with non-zero off-diagonal elements
    /// let mat2 = Matrix2::new(
    ///     1.0, 0.5,
    ///     0.0, 1.0,
    /// );
    /// assert!(!mat2.is_identity(1.0e-10));
    /// ```
    ///
    /// Handling floating-point errors:
    ///
    /// ```
    /// # use nalgebra::Matrix2;
    /// // A matrix computed through operations might have small errors
    /// let mat = Matrix2::new(
    ///     1.0, 0.0,
    ///     0.0, 1.0,
    /// );
    ///
    /// // Create a slightly perturbed identity matrix (simulating computation errors)
    /// let perturbed = Matrix2::new(
    ///     1.0 + 1.0e-13,  0.0,
    ///     0.0,            1.0 - 1.0e-13,
    /// );
    ///
    /// // With a reasonable tolerance, it's still considered identity
    /// assert!(perturbed.is_identity(1.0e-10));
    ///
    /// // But with a very strict tolerance, it's not
    /// assert!(!perturbed.is_identity(1.0e-14));
    /// ```
    ///
    /// Practical example with matrix operations:
    ///
    /// ```
    /// # use nalgebra::Matrix3;
    /// // Verify that multiplying a matrix by its inverse gives identity
    /// let mat = Matrix3::new(
    ///     1.0, 2.0, 3.0,
    ///     0.0, 1.0, 4.0,
    ///     5.0, 6.0, 0.0,
    /// );
    ///
    /// if let Some(inv) = mat.try_inverse() {
    ///     let product = mat * inv;
    ///     // Due to floating-point errors, we need some tolerance
    ///     assert!(product.is_identity(1.0e-10));
    /// }
    /// ```
    ///
    /// # See Also
    ///
    /// * [`identity`](Self::identity) - Create an identity matrix
    /// * [`is_orthogonal`](Self::is_orthogonal) - Check if a matrix is orthogonal
    /// * [`try_inverse`](Self::try_inverse) - Compute the inverse of a matrix
    #[inline]
    #[must_use]
    pub fn is_identity(&self, eps: T::Epsilon) -> bool
    where
        T: Zero + One + RelativeEq,
        T::Epsilon: Clone,
    {
        let (nrows, ncols) = self.shape();

        for j in 0..ncols {
            for i in 0..nrows {
                let el = unsafe { self.get_unchecked((i, j)) };
                if (i == j && !relative_eq!(*el, T::one(), epsilon = eps.clone()))
                    || (i != j && !relative_eq!(*el, T::zero(), epsilon = eps.clone()))
                {
                    return false;
                }
            }
        }

        true
    }
}

impl<T: ComplexField, R: Dim, C: Dim, S: Storage<T, R, C>> Matrix<T, R, C, S> {
    /// Returns `true` if this matrix is orthogonal within a tolerance of `eps`.
    ///
    /// An orthogonal matrix is a matrix whose columns (and rows) are orthonormal vectors.
    /// This means that when you multiply the matrix by its transpose, you get the identity matrix:
    /// M<sup>T</sup> × M = I (or equivalently, M × M<sup>T</sup> = I for square matrices).
    ///
    /// Orthogonal matrices have several important properties:
    /// - They preserve lengths and angles, making them useful for rotations and reflections
    /// - Their transpose equals their inverse: M<sup>T</sup> = M<sup>-1</sup>
    /// - Their determinant is either +1 or -1
    /// - Multiplying by an orthogonal matrix doesn't change the length of a vector
    ///
    /// This method checks if M<sup>H</sup> × M ≈ I, where M<sup>H</sup> is the conjugate transpose
    /// (adjoint) of M. For real matrices, this is the same as M<sup>T</sup> × M ≈ I.
    ///
    /// # Parameters
    ///
    /// * `eps` - The relative error tolerance for checking approximate equality to the identity matrix.
    ///   Common values are `1.0e-5` for `f32` or `1.0e-12` for `f64`.
    ///
    /// # Examples
    ///
    /// Identity matrices are orthogonal:
    ///
    /// ```
    /// # use nalgebra::{Matrix2, Matrix3};
    /// let id = Matrix2::<f64>::identity();
    /// assert!(id.is_orthogonal(1.0e-10));
    /// ```
    ///
    /// Rotation matrices are orthogonal:
    ///
    /// ```
    /// # use nalgebra::Matrix2;
    /// use std::f64::consts::PI;
    ///
    /// // A 2D rotation matrix (rotating by 45 degrees)
    /// let angle = PI / 4.0;
    /// let cos_a = angle.cos();
    /// let sin_a = angle.sin();
    /// let rotation = Matrix2::new(
    ///     cos_a, -sin_a,
    ///     sin_a,  cos_a,
    /// );
    ///
    /// assert!(rotation.is_orthogonal(1.0e-10));
    /// ```
    ///
    /// Non-orthogonal matrices:
    ///
    /// ```
    /// # use nalgebra::Matrix2;
    /// // A scaling matrix is not orthogonal (it changes lengths)
    /// let scaling = Matrix2::new(
    ///     2.0, 0.0,
    ///     0.0, 2.0,
    /// );
    /// assert!(!scaling.is_orthogonal(1.0e-10));
    ///
    /// // A shearing matrix is not orthogonal
    /// let shear = Matrix2::new(
    ///     1.0, 1.0,
    ///     0.0, 1.0,
    /// );
    /// assert!(!shear.is_orthogonal(1.0e-10));
    /// ```
    ///
    /// 3D rotation matrix:
    ///
    /// ```
    /// # use nalgebra::Matrix3;
    /// use std::f64::consts::PI;
    ///
    /// // A 3D rotation matrix around the Z-axis
    /// let angle = PI / 3.0; // 60 degrees
    /// let cos_a = angle.cos();
    /// let sin_a = angle.sin();
    /// let rotation_z = Matrix3::new(
    ///     cos_a, -sin_a, 0.0,
    ///     sin_a,  cos_a, 0.0,
    ///     0.0,    0.0,   1.0,
    /// );
    ///
    /// assert!(rotation_z.is_orthogonal(1.0e-10));
    /// ```
    ///
    /// Verifying orthogonality after matrix operations:
    ///
    /// ```
    /// # use nalgebra::Matrix3;
    /// use std::f64::consts::PI;
    ///
    /// // Create two rotation matrices
    /// let angle1 = PI / 6.0;
    /// let rot1 = Matrix3::new(
    ///     angle1.cos(), -angle1.sin(), 0.0,
    ///     angle1.sin(),  angle1.cos(), 0.0,
    ///     0.0,           0.0,          1.0,
    /// );
    ///
    /// let angle2 = PI / 4.0;
    /// let rot2 = Matrix3::new(
    ///     angle2.cos(), -angle2.sin(), 0.0,
    ///     angle2.sin(),  angle2.cos(), 0.0,
    ///     0.0,           0.0,          1.0,
    /// );
    ///
    /// // The product of two orthogonal matrices is orthogonal
    /// let combined = rot1 * rot2;
    /// assert!(combined.is_orthogonal(1.0e-10));
    /// ```
    ///
    /// # See Also
    ///
    /// * [`is_special_orthogonal`](Self::is_special_orthogonal) - Check if the matrix is orthogonal with determinant +1 (a proper rotation)
    /// * [`is_identity`](Self::is_identity) - Check if the matrix is the identity matrix
    /// * [`transpose`](Self::transpose) - Compute the transpose of the matrix
    /// * [`adjoint`](Self::adjoint) - Compute the conjugate transpose (Hermitian adjoint)
    #[inline]
    #[must_use]
    pub fn is_orthogonal(&self, eps: T::Epsilon) -> bool
    where
        T: Zero + One + ClosedAddAssign + ClosedMulAssign + RelativeEq,
        S: Storage<T, R, C>,
        T::Epsilon: Clone,
        DefaultAllocator: Allocator<R, C> + Allocator<C, C>,
    {
        (self.ad_mul(self)).is_identity(eps)
    }
}

impl<T: RealField, D: Dim, S: Storage<T, D, D>> SquareMatrix<T, D, S>
where
    DefaultAllocator: Allocator<D, D>,
{
    /// Returns `true` if this matrix is special orthogonal within a tolerance of `eps`.
    ///
    /// A special orthogonal matrix is an orthogonal matrix with a determinant of +1. These matrices
    /// represent proper rotations (rotations without reflections). The "special" in the name refers
    /// to the determinant being exactly +1, as opposed to general orthogonal matrices which can
    /// have a determinant of either +1 or -1.
    ///
    /// Special orthogonal matrices form the SO(n) group in mathematics:
    /// - SO(2): 2D rotations (rotation matrices in the plane)
    /// - SO(3): 3D rotations (rotation matrices in 3D space)
    ///
    /// Key properties of special orthogonal matrices:
    /// - They preserve lengths, angles, and orientation (handedness)
    /// - They represent pure rotations without any reflection or scaling
    /// - The product of two special orthogonal matrices is also special orthogonal
    /// - Their transpose equals their inverse: M<sup>T</sup> = M<sup>-1</sup>
    ///
    /// This method checks three conditions:
    /// 1. The matrix is square
    /// 2. The matrix is orthogonal (M<sup>T</sup> × M = I)
    /// 3. The determinant is positive (which implies det = +1 for orthogonal matrices)
    ///
    /// # Parameters
    ///
    /// * `eps` - The tolerance for checking orthogonality. Common values are `1.0e-5` for `f32`
    ///   or `1.0e-12` for `f64`.
    ///
    /// # Examples
    ///
    /// 2D rotation matrices are special orthogonal:
    ///
    /// ```
    /// # use nalgebra::Matrix2;
    /// use std::f64::consts::PI;
    ///
    /// // A 2D rotation by 30 degrees
    /// let angle = PI / 6.0;
    /// let rotation = Matrix2::new(
    ///     angle.cos(), -angle.sin(),
    ///     angle.sin(),  angle.cos(),
    /// );
    ///
    /// assert!(rotation.is_special_orthogonal(1.0e-10));
    /// ```
    ///
    /// 3D rotation matrices are special orthogonal:
    ///
    /// ```
    /// # use nalgebra::Matrix3;
    /// use std::f64::consts::PI;
    ///
    /// // A 3D rotation around the X-axis
    /// let angle = PI / 4.0; // 45 degrees
    /// let rotation_x = Matrix3::new(
    ///     1.0, 0.0,         0.0,
    ///     0.0, angle.cos(), -angle.sin(),
    ///     0.0, angle.sin(),  angle.cos(),
    /// );
    ///
    /// assert!(rotation_x.is_special_orthogonal(1.0e-10));
    /// ```
    ///
    /// Reflection matrices are orthogonal but NOT special orthogonal (det = -1):
    ///
    /// ```
    /// # use nalgebra::Matrix2;
    /// // A reflection across the x-axis
    /// let reflection = Matrix2::new(
    ///     1.0,  0.0,
    ///     0.0, -1.0,
    /// );
    ///
    /// // It's orthogonal
    /// assert!(reflection.is_orthogonal(1.0e-10));
    ///
    /// // But NOT special orthogonal (determinant is -1)
    /// assert!(!reflection.is_special_orthogonal(1.0e-10));
    /// ```
    ///
    /// Identity matrix is special orthogonal:
    ///
    /// ```
    /// # use nalgebra::Matrix3;
    /// let identity = Matrix3::<f64>::identity();
    /// assert!(identity.is_special_orthogonal(1.0e-10));
    /// ```
    ///
    /// Combining rotations preserves special orthogonality:
    ///
    /// ```
    /// # use nalgebra::Matrix3;
    /// use std::f64::consts::PI;
    ///
    /// // Rotation around X-axis
    /// let angle_x = PI / 3.0;
    /// let rot_x = Matrix3::new(
    ///     1.0, 0.0,            0.0,
    ///     0.0, angle_x.cos(), -angle_x.sin(),
    ///     0.0, angle_x.sin(),  angle_x.cos(),
    /// );
    ///
    /// // Rotation around Y-axis
    /// let angle_y = PI / 4.0;
    /// let rot_y = Matrix3::new(
    ///      angle_y.cos(), 0.0, angle_y.sin(),
    ///      0.0,           1.0, 0.0,
    ///     -angle_y.sin(), 0.0, angle_y.cos(),
    /// );
    ///
    /// // Combined rotation is also special orthogonal
    /// let combined = rot_x * rot_y;
    /// assert!(combined.is_special_orthogonal(1.0e-10));
    /// ```
    ///
    /// # See Also
    ///
    /// * [`is_orthogonal`](Self::is_orthogonal) - Check if the matrix is orthogonal (allows det = ±1)
    /// * [`determinant`](Self::determinant) - Compute the determinant of the matrix
    /// * [`try_inverse`](Self::try_inverse) - Compute the inverse (equals transpose for orthogonal matrices)
    #[inline]
    #[must_use]
    pub fn is_special_orthogonal(&self, eps: T) -> bool
    where
        D: DimMin<D, Output = D>,
        DefaultAllocator: Allocator<D>,
    {
        self.is_square() && self.is_orthogonal(eps) && self.determinant() > T::zero()
    }

    /// Returns `true` if this matrix is invertible (non-singular).
    ///
    /// An invertible matrix (also called a non-singular or non-degenerate matrix) is a square matrix
    /// that has an inverse. If a matrix A is invertible, there exists a matrix A<sup>-1</sup> such that:
    /// A × A<sup>-1</sup> = A<sup>-1</sup> × A = I (the identity matrix).
    ///
    /// A matrix is invertible if and only if:
    /// - Its determinant is non-zero
    /// - Its rows (and columns) are linearly independent
    /// - It has full rank (rank equals the dimension)
    /// - The system of linear equations Ax = b has a unique solution for any b
    ///
    /// Invertible matrices are fundamental in solving linear systems, computing transformations,
    /// and many other linear algebra operations.
    ///
    /// # Implementation Note
    ///
    /// This method actually attempts to compute the inverse. If the inverse can be computed,
    /// the matrix is invertible. This is a definitive test but can be computationally expensive
    /// for large matrices.
    ///
    /// # Examples
    ///
    /// Identity matrices are always invertible:
    ///
    /// ```
    /// # use nalgebra::Matrix2;
    /// let id = Matrix2::<f64>::identity();
    /// assert!(id.is_invertible());
    /// ```
    ///
    /// Checking if a general matrix is invertible:
    ///
    /// ```
    /// # use nalgebra::Matrix2;
    /// // This matrix is invertible
    /// let mat = Matrix2::new(
    ///     1.0, 2.0,
    ///     3.0, 4.0,
    /// );
    /// assert!(mat.is_invertible());
    ///
    /// // We can compute its inverse
    /// let inv = mat.try_inverse().unwrap();
    /// let product = mat * inv;
    /// assert!(product.is_identity(1.0e-10));
    /// ```
    ///
    /// Singular (non-invertible) matrices:
    ///
    /// ```
    /// # use nalgebra::Matrix2;
    /// // A matrix with linearly dependent rows is not invertible
    /// let singular = Matrix2::new(
    ///     1.0, 2.0,
    ///     2.0, 4.0,  // Second row is 2× the first row
    /// );
    /// assert!(!singular.is_invertible());
    ///
    /// // The zero matrix is not invertible
    /// let zero = Matrix2::<f64>::zeros();
    /// assert!(!zero.is_invertible());
    /// ```
    ///
    /// 3×3 matrix examples:
    ///
    /// ```
    /// # use nalgebra::Matrix3;
    /// // An invertible 3×3 matrix
    /// let mat = Matrix3::new(
    ///     1.0, 0.0, 2.0,
    ///     0.0, 1.0, 0.0,
    ///     1.0, 1.0, 1.0,
    /// );
    /// assert!(mat.is_invertible());
    ///
    /// // A singular 3×3 matrix (determinant = 0)
    /// let singular = Matrix3::new(
    ///     1.0, 2.0, 3.0,
    ///     2.0, 4.0, 6.0,  // Second row is 2× the first row
    ///     4.0, 5.0, 6.0,
    /// );
    /// assert!(!singular.is_invertible());
    /// ```
    ///
    /// Rotation matrices are always invertible:
    ///
    /// ```
    /// # use nalgebra::Matrix2;
    /// use std::f64::consts::PI;
    ///
    /// // Create a rotation matrix
    /// let angle = PI / 4.0;
    /// let rotation = Matrix2::new(
    ///     angle.cos(), -angle.sin(),
    ///     angle.sin(),  angle.cos(),
    /// );
    ///
    /// assert!(rotation.is_invertible());
    ///
    /// // For rotation matrices, the inverse equals the transpose
    /// let inverse = rotation.try_inverse().unwrap();
    /// let transpose = rotation.transpose();
    /// assert!((inverse - transpose).norm() < 1.0e-10);
    /// ```
    ///
    /// Practical use case - solving linear systems:
    ///
    /// ```
    /// # use nalgebra::{Matrix3, Vector3};
    /// // Coefficient matrix for a system of linear equations
    /// let a = Matrix3::new(
    ///     2.0, 1.0, 1.0,
    ///     1.0, 3.0, 2.0,
    ///     1.0, 0.0, 0.0,
    /// );
    ///
    /// if a.is_invertible() {
    ///     // We can solve Ax = b for any b
    ///     let b = Vector3::new(4.0, 5.0, 6.0);
    ///     let x = a.try_inverse().unwrap() * b;
    ///
    ///     // Verify the solution
    ///     assert!((a * x - b).norm() < 1.0e-10);
    /// }
    /// ```
    ///
    /// # See Also
    ///
    /// * [`try_inverse`](Self::try_inverse) - Attempt to compute the matrix inverse
    /// * [`determinant`](Self::determinant) - Compute the determinant (zero means not invertible)
    /// * [`pseudo_inverse`](Self::pseudo_inverse) - Compute a pseudo-inverse for singular matrices
    #[inline]
    #[must_use]
    pub fn is_invertible(&self) -> bool {
        // TODO: improve this?
        self.clone_owned().try_inverse().is_some()
    }
}
