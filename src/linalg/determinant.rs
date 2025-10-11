use simba::scalar::ComplexField;

use crate::base::allocator::Allocator;
use crate::base::dimension::DimMin;
use crate::base::storage::Storage;
use crate::base::{DefaultAllocator, SquareMatrix};

use crate::linalg::LU;

impl<T: ComplexField, D: DimMin<D, Output = D>, S: Storage<T, D, D>> SquareMatrix<T, D, S> {
    /// Computes the determinant of a square matrix.
    ///
    /// # What is a Determinant?
    ///
    /// The determinant is a special scalar value computed from a square matrix that provides
    /// important information about the matrix and the linear transformation it represents.
    ///
    /// Think of the determinant as a measure that tells you:
    /// - **If the matrix is invertible**: `det(A) ≠ 0` means invertible, `det(A) = 0` means singular
    /// - **How much the matrix scales volumes**: The absolute value `|det(A)|` tells you how much
    ///   areas (2D) or volumes (3D) are scaled by the transformation
    /// - **Whether orientation is preserved**: If `det(A) < 0`, the transformation flips orientation
    ///   (like a mirror reflection)
    ///
    /// For a 2×2 matrix `[[a, b], [c, d]]`, the determinant is simply: `a*d - b*c`
    ///
    /// # Common Use Cases
    ///
    /// - **Checking invertibility**: Before inverting a matrix, check if `det ≠ 0`
    /// - **Computing volume scaling**: Find how much a transformation scales areas/volumes
    /// - **Testing orientation**: Determine if a transformation preserves or reverses orientation
    /// - **Solving systems**: Determinants appear in Cramer's rule for solving linear equations
    /// - **Cross products**: The determinant is used internally in computing cross products
    ///
    /// # Performance
    ///
    /// The implementation uses optimized algorithms based on matrix size:
    /// - **1×1 matrices**: Returns the single element (instant)
    /// - **2×2 matrices**: Uses direct formula `ad - bc` (very fast)
    /// - **3×3 matrices**: Uses cofactor expansion (fast, unrolled)
    /// - **Larger matrices**: Uses LU decomposition with O(n³) complexity
    ///
    /// # Returns
    ///
    /// The determinant as a scalar value of type `T`.
    ///
    /// # Panics
    ///
    /// Panics if the matrix is not square.
    ///
    /// # Examples
    ///
    /// ## Basic 2×2 Determinant
    ///
    /// ```
    /// use nalgebra::Matrix2;
    ///
    /// let m = Matrix2::new(1.0, 2.0,
    ///                      3.0, 4.0);
    ///
    /// let det: f64 = m.determinant();
    ///
    /// // For [[1, 2], [3, 4]], det = 1*4 - 2*3 = -2
    /// assert!((det - (-2.0)).abs() < 1e-10);
    /// ```
    ///
    /// ## Checking if a Matrix is Invertible
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
    ///     2.0, 4.0, 6.0,  // Second row is 2× the first
    ///     4.0, 5.0, 6.0,
    /// );
    ///
    /// // Non-zero determinant means invertible
    /// let det1: f64 = invertible.determinant();
    /// assert!(det1.abs() > 1e-10);
    /// assert!(invertible.try_inverse().is_some());
    ///
    /// // Zero determinant means singular (not invertible)
    /// let det2: f64 = singular.determinant();
    /// assert!(det2.abs() < 1e-10);
    /// assert!(singular.try_inverse().is_none());
    /// ```
    ///
    /// ## Computing Volume Scaling Factor
    ///
    /// ```
    /// use nalgebra::Matrix3;
    ///
    /// // A 3D transformation that scales by 2 in each direction
    /// let scale = Matrix3::new(
    ///     2.0, 0.0, 0.0,
    ///     0.0, 2.0, 0.0,
    ///     0.0, 0.0, 2.0,
    /// );
    ///
    /// let det: f64 = scale.determinant();
    ///
    /// // Volume is scaled by 2³ = 8
    /// assert!((det - 8.0).abs() < 1e-10);
    /// ```
    ///
    /// ## Checking Orientation Preservation
    ///
    /// ```
    /// use nalgebra::Matrix2;
    ///
    /// // A rotation matrix (preserves orientation)
    /// let angle = std::f64::consts::PI / 4.0;  // 45 degrees
    /// let rotation = Matrix2::new(
    ///     angle.cos(), -angle.sin(),
    ///     angle.sin(),  angle.cos(),
    /// );
    ///
    /// // A reflection matrix (flips orientation)
    /// let reflection = Matrix2::new(
    ///     1.0,  0.0,
    ///     0.0, -1.0,
    /// );
    ///
    /// let det_rotation: f64 = rotation.determinant();
    /// let det_reflection: f64 = reflection.determinant();
    ///
    /// // Rotation has det = +1 (preserves orientation)
    /// assert!((det_rotation - 1.0).abs() < 1e-10);
    ///
    /// // Reflection has det = -1 (reverses orientation)
    /// assert!((det_reflection - (-1.0)).abs() < 1e-10);
    /// ```
    ///
    /// ## Identity and Zero Matrices
    ///
    /// ```
    /// use nalgebra::Matrix3;
    ///
    /// let identity = Matrix3::identity();
    /// let zero = Matrix3::zeros();
    ///
    /// // Identity matrix has determinant = 1
    /// let det_id: f64 = identity.determinant();
    /// assert!((det_id - 1.0).abs() < 1e-10);
    ///
    /// // Zero matrix has determinant = 0 (singular)
    /// let det_zero: f64 = zero.determinant();
    /// assert!(det_zero.abs() < 1e-10);
    /// ```
    ///
    /// ## Using with Different Numeric Types
    ///
    /// ```
    /// use nalgebra::Matrix2;
    ///
    /// // Integer matrix (note: use a float type for the result)
    /// let m_i32 = Matrix2::new(2, 3,
    ///                          4, 5);
    /// // For integers, convert to float first
    /// let m_f64 = m_i32.map(|x| x as f64);
    /// let det: f64 = m_f64.determinant();
    /// assert!((det - (-2.0)).abs() < 1e-10);
    ///
    /// // Complex numbers work too!
    /// use num_complex::Complex;
    /// let m_complex = Matrix2::new(
    ///     Complex::new(1.0, 0.0), Complex::new(2.0, 0.0),
    ///     Complex::new(3.0, 0.0), Complex::new(4.0, 0.0),
    /// );
    /// let det_complex: Complex<f64> = m_complex.determinant();
    /// assert!((det_complex.re - (-2.0)).abs() < 1e-10);
    /// ```
    ///
    /// ## Determinant Properties
    ///
    /// ```
    /// use nalgebra::Matrix2;
    ///
    /// let a = Matrix2::new(1.0, 2.0,
    ///                      3.0, 4.0);
    /// let b = Matrix2::new(5.0, 6.0,
    ///                      7.0, 8.0);
    ///
    /// // Property: det(A * B) = det(A) * det(B)
    /// let det_a: f64 = a.determinant();
    /// let det_b: f64 = b.determinant();
    /// let det_ab: f64 = (a * b).determinant();
    ///
    /// assert!((det_ab - det_a * det_b).abs() < 1e-10);
    ///
    /// // Property: det(A^T) = det(A)
    /// let det_a_transpose: f64 = a.transpose().determinant();
    /// assert!((det_a_transpose - det_a).abs() < 1e-10);
    /// ```
    ///
    /// # Mathematical Properties
    ///
    /// The determinant has several important properties:
    /// - `det(A * B) = det(A) * det(B)` - Multiplicative property
    /// - `det(A^T) = det(A)` - Transpose doesn't change determinant
    /// - `det(A^(-1)) = 1 / det(A)` - Inverse's determinant is reciprocal
    /// - `det(c * A) = c^n * det(A)` - Scaling by scalar (n = dimension)
    /// - `det(I) = 1` - Identity matrix has determinant 1
    ///
    /// # See Also
    ///
    /// - [`try_inverse`](Self::try_inverse) - Compute the matrix inverse (requires non-zero determinant)
    /// - [`try_inverse_mut`](Self::try_inverse_mut) - In-place matrix inversion
    /// - [`LU::determinant`](crate::linalg::LU::determinant) - Determinant via LU decomposition
    /// - [`LU::is_invertible`](crate::linalg::LU::is_invertible) - Check invertibility without computing full determinant
    #[inline]
    #[must_use]
    pub fn determinant(&self) -> T
    where
        DefaultAllocator: Allocator<D, D> + Allocator<D>,
    {
        assert!(
            self.is_square(),
            "Unable to compute the determinant of a non-square matrix."
        );
        let dim = self.shape().0;

        unsafe {
            match dim {
                0 => T::one(),
                1 => self.get_unchecked((0, 0)).clone(),
                2 => {
                    let m11 = self.get_unchecked((0, 0)).clone();
                    let m12 = self.get_unchecked((0, 1)).clone();
                    let m21 = self.get_unchecked((1, 0)).clone();
                    let m22 = self.get_unchecked((1, 1)).clone();

                    m11 * m22 - m21 * m12
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
                    let minor_m11_m23 = m21.clone() * m33 - m31.clone() * m23;
                    let minor_m11_m22 = m21 * m32 - m31 * m22;

                    m11 * minor_m12_m23 - m12 * minor_m11_m23 + m13 * minor_m11_m22
                }
                _ => LU::new(self.clone_owned()).determinant(),
            }
        }
    }
}
