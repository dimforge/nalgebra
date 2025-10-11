use crate::{RawStorage, SimdComplexField};
use num::{One, Zero};
use simba::scalar::{ClosedAddAssign, ClosedMulAssign};

use crate::base::allocator::Allocator;
use crate::base::blas_uninit::{axcpy_uninit, gemm_uninit, gemv_uninit};
use crate::base::constraint::{
    AreMultipliable, DimEq, SameNumberOfColumns, SameNumberOfRows, ShapeConstraint,
};
use crate::base::dimension::{Const, Dim, Dyn, U1, U2, U3, U4};
use crate::base::storage::{Storage, StorageMut};
use crate::base::uninit::Init;
use crate::base::{
    DVectorView, DefaultAllocator, Matrix, Scalar, SquareMatrix, Vector, VectorView,
};

/// # Dot/scalar product
impl<T, R: Dim, C: Dim, S: RawStorage<T, R, C>> Matrix<T, R, C, S>
where
    T: Scalar + Zero + ClosedAddAssign + ClosedMulAssign,
{
    #[inline(always)]
    fn dotx<R2: Dim, C2: Dim, SB>(
        &self,
        rhs: &Matrix<T, R2, C2, SB>,
        conjugate: impl Fn(T) -> T,
    ) -> T
    where
        SB: RawStorage<T, R2, C2>,
        ShapeConstraint: DimEq<R, R2> + DimEq<C, C2>,
    {
        assert!(
            self.nrows() == rhs.nrows(),
            "Dot product dimensions mismatch for shapes {:?} and {:?}: left rows != right rows.",
            self.shape(),
            rhs.shape(),
        );

        assert!(
            self.ncols() == rhs.ncols(),
            "Dot product dimensions mismatch for shapes {:?} and {:?}: left cols != right cols.",
            self.shape(),
            rhs.shape(),
        );

        // So we do some special cases for common fixed-size vectors of dimension lower than 8
        // because the `for` loop below won't be very efficient on those.
        if (R::is::<U2>() || R2::is::<U2>()) && (C::is::<U1>() || C2::is::<U1>()) {
            unsafe {
                let a = conjugate(self.get_unchecked((0, 0)).clone())
                    * rhs.get_unchecked((0, 0)).clone();
                let b = conjugate(self.get_unchecked((1, 0)).clone())
                    * rhs.get_unchecked((1, 0)).clone();

                return a + b;
            }
        }
        if (R::is::<U3>() || R2::is::<U3>()) && (C::is::<U1>() || C2::is::<U1>()) {
            unsafe {
                let a = conjugate(self.get_unchecked((0, 0)).clone())
                    * rhs.get_unchecked((0, 0)).clone();
                let b = conjugate(self.get_unchecked((1, 0)).clone())
                    * rhs.get_unchecked((1, 0)).clone();
                let c = conjugate(self.get_unchecked((2, 0)).clone())
                    * rhs.get_unchecked((2, 0)).clone();

                return a + b + c;
            }
        }
        if (R::is::<U4>() || R2::is::<U4>()) && (C::is::<U1>() || C2::is::<U1>()) {
            unsafe {
                let mut a = conjugate(self.get_unchecked((0, 0)).clone())
                    * rhs.get_unchecked((0, 0)).clone();
                let mut b = conjugate(self.get_unchecked((1, 0)).clone())
                    * rhs.get_unchecked((1, 0)).clone();
                let c = conjugate(self.get_unchecked((2, 0)).clone())
                    * rhs.get_unchecked((2, 0)).clone();
                let d = conjugate(self.get_unchecked((3, 0)).clone())
                    * rhs.get_unchecked((3, 0)).clone();

                a += c;
                b += d;

                return a + b;
            }
        }

        // All this is inspired from the "unrolled version" discussed in:
        // https://blog.theincredibleholk.org/blog/2012/12/10/optimizing-dot-product/
        //
        // And this comment from bluss:
        // https://users.rust-lang.org/t/how-to-zip-two-slices-efficiently/2048/12
        let mut res = T::zero();

        // We have to define them outside of the loop (and not inside at first assignment)
        // otherwise vectorization won't kick in for some reason.
        let mut acc0;
        let mut acc1;
        let mut acc2;
        let mut acc3;
        let mut acc4;
        let mut acc5;
        let mut acc6;
        let mut acc7;

        for j in 0..self.ncols() {
            let mut i = 0;

            acc0 = T::zero();
            acc1 = T::zero();
            acc2 = T::zero();
            acc3 = T::zero();
            acc4 = T::zero();
            acc5 = T::zero();
            acc6 = T::zero();
            acc7 = T::zero();

            while self.nrows() - i >= 8 {
                acc0 += unsafe {
                    conjugate(self.get_unchecked((i, j)).clone())
                        * rhs.get_unchecked((i, j)).clone()
                };
                acc1 += unsafe {
                    conjugate(self.get_unchecked((i + 1, j)).clone())
                        * rhs.get_unchecked((i + 1, j)).clone()
                };
                acc2 += unsafe {
                    conjugate(self.get_unchecked((i + 2, j)).clone())
                        * rhs.get_unchecked((i + 2, j)).clone()
                };
                acc3 += unsafe {
                    conjugate(self.get_unchecked((i + 3, j)).clone())
                        * rhs.get_unchecked((i + 3, j)).clone()
                };
                acc4 += unsafe {
                    conjugate(self.get_unchecked((i + 4, j)).clone())
                        * rhs.get_unchecked((i + 4, j)).clone()
                };
                acc5 += unsafe {
                    conjugate(self.get_unchecked((i + 5, j)).clone())
                        * rhs.get_unchecked((i + 5, j)).clone()
                };
                acc6 += unsafe {
                    conjugate(self.get_unchecked((i + 6, j)).clone())
                        * rhs.get_unchecked((i + 6, j)).clone()
                };
                acc7 += unsafe {
                    conjugate(self.get_unchecked((i + 7, j)).clone())
                        * rhs.get_unchecked((i + 7, j)).clone()
                };
                i += 8;
            }

            res += acc0 + acc4;
            res += acc1 + acc5;
            res += acc2 + acc6;
            res += acc3 + acc7;

            for k in i..self.nrows() {
                res += unsafe {
                    conjugate(self.get_unchecked((k, j)).clone())
                        * rhs.get_unchecked((k, j)).clone()
                }
            }
        }

        res
    }

    /// Computes the dot product (also known as scalar product or inner product) between two vectors or matrices.
    ///
    /// The dot product is a fundamental operation in linear algebra that takes two equal-length
    /// sequences of numbers and returns a single number. For vectors, it's the sum of the products
    /// of corresponding entries: `a · b = a₁b₁ + a₂b₂ + ... + aₙbₙ`.
    ///
    /// When applied to matrices, they are treated as flattened vectors in column-major order.
    ///
    /// # Mathematical Definition
    ///
    /// For vectors: `self.dot(rhs) = self.transpose() * rhs`
    ///
    /// For complex vectors, this uses the standard (not conjugate) dot product.
    /// For the conjugate-linear dot product (sesquilinear form), use [`dotc`](Self::dotc).
    ///
    /// # Important Note
    ///
    /// This is **not** matrix multiplication! For matrix multiplication, use:
    /// - [`gemm`](Self::gemm) for general matrix-matrix multiplication
    /// - [`mul_to`](crate::Matrix::mul_to) for computing `self * rhs` into a pre-allocated result
    /// - [`mul`](core::ops::Mul::mul) or the `*` operator for inline multiplication
    ///
    /// # Panics
    ///
    /// Panics if `self` and `rhs` have different dimensions.
    ///
    /// # Examples
    ///
    /// ## Basic vector dot product
    /// ```
    /// # use nalgebra::Vector3;
    /// let vec1 = Vector3::new(1.0, 2.0, 3.0);
    /// let vec2 = Vector3::new(0.1, 0.2, 0.3);
    ///
    /// // Computes: 1.0*0.1 + 2.0*0.2 + 3.0*0.3 = 0.1 + 0.4 + 0.9 = 1.4
    /// assert_eq!(vec1.dot(&vec2), 1.4);
    /// ```
    ///
    /// ## Using dot product for vector length
    /// ```
    /// # use nalgebra::Vector3;
    /// let vec = Vector3::new(3.0, 4.0, 0.0);
    ///
    /// // The length squared is the dot product of a vector with itself
    /// let length_squared = vec.dot(&vec);
    /// assert_eq!(length_squared, 25.0);
    /// assert_eq!(length_squared.sqrt(), 5.0);
    /// ```
    ///
    /// ## Dot product with matrices (treated as flattened vectors)
    /// ```
    /// # use nalgebra::Matrix2x3;
    /// let mat1 = Matrix2x3::new(1.0, 2.0, 3.0,
    ///                           4.0, 5.0, 6.0);
    /// let mat2 = Matrix2x3::new(0.1, 0.2, 0.3,
    ///                           0.4, 0.5, 0.6);
    /// // Flattened: [1,4,2,5,3,6] · [0.1,0.4,0.2,0.5,0.3,0.6] = 9.1
    /// assert_eq!(mat1.dot(&mat2), 9.1);
    /// ```
    ///
    /// ## Checking orthogonality
    /// ```
    /// # use nalgebra::Vector2;
    /// let vec1 = Vector2::new(1.0, 0.0);
    /// let vec2 = Vector2::new(0.0, 1.0);
    ///
    /// // Orthogonal vectors have a dot product of zero
    /// assert_eq!(vec1.dot(&vec2), 0.0);
    /// ```
    ///
    /// # See Also
    ///
    /// - [`dotc`](Self::dotc) - Conjugate-linear dot product for complex vectors
    /// - [`tr_dot`](Self::tr_dot) - Dot product between the transpose of `self` and `rhs`
    /// - [`norm_squared`](crate::Matrix::norm_squared) - Computes `self.dot(&self)` for vectors
    ///
    #[inline]
    #[must_use]
    pub fn dot<R2: Dim, C2: Dim, SB>(&self, rhs: &Matrix<T, R2, C2, SB>) -> T
    where
        SB: RawStorage<T, R2, C2>,
        ShapeConstraint: DimEq<R, R2> + DimEq<C, C2>,
    {
        self.dotx(rhs, |e| e)
    }

    /// Computes the conjugate-linear dot product (also called sesquilinear form) between two complex vectors or matrices.
    ///
    /// The conjugate dot product is essential for working with complex vectors because it ensures
    /// that the "length squared" of a complex vector is always a real, non-negative number.
    /// This function computes: `conj(self₁)·rhs₁ + conj(self₂)·rhs₂ + ... + conj(selfₙ)·rhsₙ`
    /// where `conj` denotes complex conjugation.
    ///
    /// # Mathematical Definition
    ///
    /// `self.dotc(rhs) = self.adjoint() * rhs`
    ///
    /// For **real** vectors, this is identical to [`dot`](Self::dot) because conjugation has no effect.
    /// For **complex** vectors, the entries of `self` are conjugated before multiplication.
    ///
    /// # Important Properties
    ///
    /// - For complex vectors: `v.dotc(&v)` is always real and non-negative (suitable for norms)
    /// - `v.dotc(&w)` ≠ `w.dotc(&v)` in general (not commutative for complex numbers)
    /// - `v.dotc(&w) = conj(w.dotc(&v))` (conjugate symmetric)
    ///
    /// # Panics
    ///
    /// Panics if `self` and `rhs` have different dimensions.
    ///
    /// # Examples
    ///
    /// ## Complex vector conjugate dot product
    /// ```
    /// # use nalgebra::{Vector2, Complex};
    /// let vec1 = Vector2::new(Complex::new(1.0, 2.0), Complex::new(3.0, 4.0));
    /// let vec2 = Vector2::new(Complex::new(0.4, 0.3), Complex::new(0.2, 0.1));
    ///
    /// // Computes: conj(1+2i)·(0.4+0.3i) + conj(3+4i)·(0.2+0.1i)
    /// //         = (1-2i)·(0.4+0.3i) + (3-4i)·(0.2+0.1i)
    /// //         = (1.0, -0.5) + (1.0, -0.5) = (2.0, -1.0)
    /// assert_eq!(vec1.dotc(&vec2), Complex::new(2.0, -1.0));
    /// ```
    ///
    /// ## Difference between dot and dotc for complex vectors
    /// ```
    /// # use nalgebra::{Vector2, Complex};
    /// let vec1 = Vector2::new(Complex::new(1.0, 2.0), Complex::new(3.0, 4.0));
    /// let vec2 = Vector2::new(Complex::new(0.4, 0.3), Complex::new(0.2, 0.1));
    ///
    /// // For complex vectors: dotc uses conjugation, dot does not
    /// assert_ne!(vec1.dotc(&vec2), vec1.dot(&vec2));
    /// ```
    ///
    /// ## Computing norm squared for complex vectors
    /// ```
    /// # use nalgebra::{Vector2, Complex};
    /// let vec = Vector2::new(Complex::new(3.0, 4.0), Complex::new(0.0, 0.0));
    ///
    /// // Using dotc ensures the result is real and non-negative
    /// let norm_squared = vec.dotc(&vec);
    /// assert_eq!(norm_squared, Complex::new(25.0, 0.0));
    /// // The norm (length) is the square root
    /// assert_eq!(norm_squared.re.sqrt(), 5.0);
    /// ```
    ///
    /// ## Real vectors: dotc equals dot
    /// ```
    /// # use nalgebra::Vector3;
    /// let vec1 = Vector3::new(1.0, 2.0, 3.0);
    /// let vec2 = Vector3::new(4.0, 5.0, 6.0);
    ///
    /// // For real numbers, conjugation does nothing
    /// assert_eq!(vec1.dotc(&vec2), vec1.dot(&vec2));
    /// ```
    ///
    /// # See Also
    ///
    /// - [`dot`](Self::dot) - Standard dot product (without conjugation)
    /// - [`norm_squared`](crate::Matrix::norm_squared) - Uses `dotc` internally for complex vectors
    /// - [`gemv_ad`](Self::gemv_ad) - Matrix-vector multiplication with adjoint
    #[inline]
    #[must_use]
    pub fn dotc<R2: Dim, C2: Dim, SB>(&self, rhs: &Matrix<T, R2, C2, SB>) -> T
    where
        T: SimdComplexField,
        SB: RawStorage<T, R2, C2>,
        ShapeConstraint: DimEq<R, R2> + DimEq<C, C2>,
    {
        self.dotx(rhs, T::simd_conjugate)
    }

    /// Computes the dot product between the transpose of `self` and `rhs`.
    ///
    /// This is a specialized dot product where `self` is implicitly transposed before
    /// computing the dot product. It's equivalent to computing `self.transpose().dot(&rhs)`,
    /// but potentially more efficient as it avoids creating the transposed matrix.
    ///
    /// # Mathematical Definition
    ///
    /// `self.tr_dot(rhs) = self.transpose() * rhs`
    ///
    /// This means the shapes must satisfy: `self` is (M×N) and `rhs` is (N×M), and both
    /// are treated as flattened vectors.
    ///
    /// # Panics
    ///
    /// Panics if the transposed dimensions don't match: requires `self.shape() = (M, N)`
    /// and `rhs.shape() = (N, M)`.
    ///
    /// # Examples
    ///
    /// ## Transposed dot product with vectors
    /// ```
    /// # use nalgebra::{Vector3, RowVector3};
    /// let col_vec = Vector3::new(1.0, 2.0, 3.0);  // 3×1
    /// let row_vec = RowVector3::new(0.1, 0.2, 0.3);  // 1×3
    ///
    /// // After transpose, col_vec becomes 1×3, matching row_vec's shape
    /// assert_eq!(col_vec.tr_dot(&row_vec), 1.4);
    /// ```
    ///
    /// ## Transposed dot product with matrices
    /// ```
    /// # use nalgebra::{Matrix2x3, Matrix3x2};
    /// let mat1 = Matrix2x3::new(1.0, 2.0, 3.0,   // 2×3 matrix
    ///                           4.0, 5.0, 6.0);
    /// let mat2 = Matrix3x2::new(0.1, 0.4,        // 3×2 matrix
    ///                           0.2, 0.5,
    ///                           0.3, 0.6);
    ///
    /// // mat1.transpose() is 3×2, same shape as mat2
    /// assert_eq!(mat1.tr_dot(&mat2), 9.1);
    /// ```
    ///
    /// ## When to use tr_dot vs dot
    /// ```
    /// # use nalgebra::{Matrix2x3, Matrix3x2};
    /// let mat = Matrix2x3::new(1.0, 2.0, 3.0,
    ///                          4.0, 5.0, 6.0);
    /// let mat_t = Matrix3x2::new(1.0, 4.0,
    ///                            2.0, 5.0,
    ///                            3.0, 6.0);
    ///
    /// // These are equivalent, but tr_dot is more efficient:
    /// assert_eq!(mat.tr_dot(&mat_t), mat.transpose().dot(&mat_t));
    /// ```
    ///
    /// # See Also
    ///
    /// - [`dot`](Self::dot) - Standard dot product
    /// - [`dotc`](Self::dotc) - Conjugate-linear dot product
    /// - [`gemv_tr`](Self::gemv_tr) - Matrix-vector multiplication with transpose
    #[inline]
    #[must_use]
    pub fn tr_dot<R2: Dim, C2: Dim, SB>(&self, rhs: &Matrix<T, R2, C2, SB>) -> T
    where
        SB: RawStorage<T, R2, C2>,
        ShapeConstraint: DimEq<C, R2> + DimEq<R, C2>,
    {
        let (nrows, ncols) = self.shape();
        assert_eq!(
            (ncols, nrows),
            rhs.shape(),
            "Transposed dot product dimension mismatch."
        );

        let mut res = T::zero();

        for j in 0..self.nrows() {
            for i in 0..self.ncols() {
                res += unsafe {
                    self.get_unchecked((j, i)).clone() * rhs.get_unchecked((i, j)).clone()
                }
            }
        }

        res
    }
}

/// # BLAS functions
impl<T, D: Dim, S> Vector<T, D, S>
where
    T: Scalar + Zero + ClosedAddAssign + ClosedMulAssign,
    S: StorageMut<T, D>,
{
    /// Computes `self = a * x * c + b * self` (extended BLAS axpy operation).
    ///
    /// This is an extended version of [`axpy`](Self::axpy) that includes an additional scalar
    /// multiplier `c` for the input vector. AXCPY stands for "A times X times C Plus Y".
    /// It performs: scale `x` by `a*c`, scale `self` by `b`, and add them together.
    ///
    /// # Mathematical Definition
    ///
    /// `self = a * x * c + b * self`
    ///
    /// Where:
    /// - `a` is a scalar multiplier
    /// - `x` is the input vector
    /// - `c` is an additional scalar multiplier
    /// - `b` is a scalar multiplier for the current value of `self`
    /// - The result is stored back in `self`
    ///
    /// This is equivalent to `axpy(a*c, x, b)` but may be more convenient in some contexts.
    ///
    /// # Performance Note
    ///
    /// If `b` is zero, `self` is never read from, which can be more efficient.
    ///
    /// # Panics
    ///
    /// Panics if `self` and `x` have different dimensions.
    ///
    /// # Examples
    ///
    /// ## Basic axcpy operation
    /// ```
    /// # use nalgebra::Vector3;
    /// let mut vec1 = Vector3::new(1.0, 2.0, 3.0);
    /// let vec2 = Vector3::new(0.1, 0.2, 0.3);
    ///
    /// // Compute: vec1 = 5.0 * vec2 * 2.0 + 5.0 * vec1
    /// vec1.axcpy(5.0, &vec2, 2.0, 5.0);
    /// assert_eq!(vec1, Vector3::new(6.0, 12.0, 18.0));
    /// ```
    ///
    /// ## Using three separate scalars
    /// ```
    /// # use nalgebra::Vector2;
    /// let mut result = Vector2::new(10.0, 20.0);
    /// let input = Vector2::new(1.0, 2.0);
    ///
    /// // result = 2.0 * input * 3.0 + 1.0 * result
    /// result.axcpy(2.0, &input, 3.0, 1.0);
    /// assert_eq!(result, Vector2::new(16.0, 32.0));
    /// ```
    ///
    /// ## Equivalent to axpy with combined scalars
    /// ```
    /// # use nalgebra::Vector3;
    /// let mut v1 = Vector3::new(1.0, 2.0, 3.0);
    /// let mut v2 = Vector3::new(1.0, 2.0, 3.0);
    /// let x = Vector3::new(0.5, 1.0, 1.5);
    ///
    /// v1.axcpy(2.0, &x, 3.0, 5.0);  // a=2.0, c=3.0, b=5.0
    /// v2.axpy(6.0, &x, 5.0);        // a*c=6.0, b=5.0
    /// assert_eq!(v1, v2);
    /// ```
    ///
    /// # See Also
    ///
    /// - [`axpy`](Self::axpy) - Standard version without the additional scalar `c`
    /// - [`gemv`](Self::gemv) - General matrix-vector multiplication
    #[inline]
    #[allow(clippy::many_single_char_names)]
    pub fn axcpy<D2: Dim, SB>(&mut self, a: T, x: &Vector<T, D2, SB>, c: T, b: T)
    where
        SB: Storage<T, D2>,
        ShapeConstraint: DimEq<D, D2>,
    {
        unsafe { axcpy_uninit(Init, self, a, x, c, b) };
    }

    /// Computes `self = a * x + b * self` (BLAS axpy operation).
    ///
    /// AXPY stands for "A times X Plus Y" and is one of the most fundamental operations in
    /// numerical linear algebra. It performs a scaled vector addition: scales vector `x` by
    /// scalar `a`, scales `self` by scalar `b`, and adds them together.
    ///
    /// # Mathematical Definition
    ///
    /// `self = a * x + b * self`
    ///
    /// Where:
    /// - `a` (alpha) is a scalar multiplier for vector `x`
    /// - `x` is the input vector to be scaled and added
    /// - `b` (beta) is a scalar multiplier for the current value of `self`
    /// - The result is stored back in `self`
    ///
    /// # Performance Note
    ///
    /// If `b` is zero, `self` is never read from, which can be more efficient.
    ///
    /// # Panics
    ///
    /// Panics if `self` and `x` have different dimensions.
    ///
    /// # Examples
    ///
    /// ## Basic axpy operation
    /// ```
    /// # use nalgebra::Vector3;
    /// let mut vec1 = Vector3::new(1.0, 2.0, 3.0);
    /// let vec2 = Vector3::new(0.1, 0.2, 0.3);
    ///
    /// // Compute: vec1 = 10.0 * vec2 + 5.0 * vec1
    /// vec1.axpy(10.0, &vec2, 5.0);
    /// assert_eq!(vec1, Vector3::new(6.0, 12.0, 18.0));
    /// ```
    ///
    /// ## Overwriting with zero beta (b=0)
    /// ```
    /// # use nalgebra::Vector3;
    /// let mut result = Vector3::new(100.0, 200.0, 300.0);  // Initial values don't matter
    /// let vec = Vector3::new(1.0, 2.0, 3.0);
    ///
    /// // When b=0, self is not read, just overwritten with a*x
    /// result.axpy(5.0, &vec, 0.0);
    /// assert_eq!(result, Vector3::new(5.0, 10.0, 15.0));
    /// ```
    ///
    /// ## Simple vector addition (a=1, b=1)
    /// ```
    /// # use nalgebra::Vector3;
    /// let mut v1 = Vector3::new(1.0, 2.0, 3.0);
    /// let v2 = Vector3::new(4.0, 5.0, 6.0);
    ///
    /// // Simple addition: v1 = v2 + v1
    /// v1.axpy(1.0, &v2, 1.0);
    /// assert_eq!(v1, Vector3::new(5.0, 7.0, 9.0));
    /// ```
    ///
    /// ## Accumulating weighted sum
    /// ```
    /// # use nalgebra::Vector2;
    /// let mut average = Vector2::new(0.0, 0.0);
    /// let sample1 = Vector2::new(10.0, 20.0);
    /// let sample2 = Vector2::new(30.0, 40.0);
    ///
    /// // Weighted average: 0.3 * sample1 + 0.7 * sample2
    /// average.axpy(0.3, &sample1, 0.0);  // Start with 0.3 * sample1
    /// average.axpy(0.7, &sample2, 1.0);  // Add 0.7 * sample2
    /// assert_eq!(average, Vector2::new(24.0, 34.0));
    /// ```
    ///
    /// # See Also
    ///
    /// - [`axcpy`](Self::axcpy) - Extended version with additional scaling parameter
    /// - [`gemv`](Self::gemv) - General matrix-vector multiplication (includes axpy as special case)
    /// - [`scale_mut`](crate::Matrix::scale_mut) - Simple scalar multiplication
    #[inline]
    pub fn axpy<D2: Dim, SB>(&mut self, a: T, x: &Vector<T, D2, SB>, b: T)
    where
        T: One,
        SB: Storage<T, D2>,
        ShapeConstraint: DimEq<D, D2>,
    {
        assert_eq!(self.nrows(), x.nrows(), "Axpy: mismatched vector shapes.");
        self.axcpy(a, x, T::one(), b)
    }

    /// Computes `self = alpha * a * x + beta * self` (BLAS gemv operation).
    ///
    /// GEMV stands for "GEneral Matrix-Vector multiplication" and is a core BLAS Level 2 operation.
    /// This function multiplies a matrix `a` by a vector `x`, scales the result by `alpha`,
    /// then adds it to `self` scaled by `beta`.
    ///
    /// # Mathematical Definition
    ///
    /// `self = alpha * a * x + beta * self`
    ///
    /// Where:
    /// - `alpha` is a scalar multiplier for the matrix-vector product
    /// - `a` is an M×N matrix
    /// - `x` is a vector of length N
    /// - `beta` is a scalar multiplier for the current value of `self`
    /// - `self` must be a vector of length M
    ///
    /// # Performance Note
    ///
    /// If `beta` is zero, `self` is never read from, which can be more efficient.
    /// This function is highly optimized and should be preferred over manual implementation.
    ///
    /// # Panics
    ///
    /// Panics if the dimensions are incompatible:
    /// - `self.len()` must equal `a.nrows()`
    /// - `x.len()` must equal `a.ncols()`
    ///
    /// # Examples
    ///
    /// ## Basic matrix-vector multiplication
    /// ```
    /// # use nalgebra::{Matrix2, Vector2};
    /// let mut result = Vector2::new(1.0, 2.0);
    /// let mat = Matrix2::new(1.0, 2.0,
    ///                        3.0, 4.0);
    /// let vec = Vector2::new(0.1, 0.2);
    ///
    /// // Compute: result = 10.0 * mat * vec + 5.0 * result
    /// result.gemv(10.0, &mat, &vec, 5.0);
    /// assert_eq!(result, Vector2::new(10.0, 21.0));
    /// ```
    ///
    /// ## Simple matrix-vector product (alpha=1, beta=0)
    /// ```
    /// # use nalgebra::{Matrix3x2, Vector3, Vector2};
    /// let mut result = Vector3::zeros();
    /// let mat = Matrix3x2::new(1.0, 2.0,
    ///                          3.0, 4.0,
    ///                          5.0, 6.0);
    /// let vec = Vector2::new(10.0, 100.0);
    ///
    /// // Simple product: result = mat * vec
    /// result.gemv(1.0, &mat, &vec, 0.0);
    /// assert_eq!(result, Vector3::new(210.0, 430.0, 650.0));
    /// ```
    ///
    /// ## Accumulating results
    /// ```
    /// # use nalgebra::{Matrix2, Vector2};
    /// let mut accumulated = Vector2::new(1.0, 1.0);
    /// let mat1 = Matrix2::new(1.0, 0.0, 0.0, 1.0);  // Identity
    /// let vec1 = Vector2::new(5.0, 10.0);
    ///
    /// // First accumulation
    /// accumulated.gemv(1.0, &mat1, &vec1, 1.0);
    /// assert_eq!(accumulated, Vector2::new(6.0, 11.0));
    ///
    /// // Second accumulation
    /// let mat2 = Matrix2::new(2.0, 0.0, 0.0, 2.0);  // 2*Identity
    /// accumulated.gemv(1.0, &mat2, &vec1, 1.0);
    /// assert_eq!(accumulated, Vector2::new(16.0, 31.0));
    /// ```
    ///
    /// ## Solving linear systems iteratively
    /// ```
    /// # use nalgebra::{Matrix2, Vector2};
    /// // Jacobi iteration step for solving Ax = b
    /// let mut x = Vector2::new(0.0, 0.0);  // Initial guess
    /// let a = Matrix2::new(4.0, 1.0,
    ///                      1.0, 3.0);
    /// let b = Vector2::new(1.0, 2.0);
    ///
    /// // One iteration: x_new = (b - off_diagonal*x) / diagonal
    /// // Simplified example showing gemv usage in iterative solvers
    /// x.gemv(1.0, &a, &b, 0.0);
    /// ```
    ///
    /// # See Also
    ///
    /// - [`gemv_tr`](Self::gemv_tr) - Matrix-vector multiplication with transposed matrix
    /// - [`gemv_ad`](Self::gemv_ad) - Matrix-vector multiplication with adjoint matrix
    /// - [`sygemv`](Self::sygemv) - Optimized version for symmetric matrices
    /// - [`hegemv`](Self::hegemv) - Optimized version for Hermitian matrices
    /// - [`gemm`](Self::gemm) - General matrix-matrix multiplication
    #[inline]
    pub fn gemv<R2: Dim, C2: Dim, D3: Dim, SB, SC>(
        &mut self,
        alpha: T,
        a: &Matrix<T, R2, C2, SB>,
        x: &Vector<T, D3, SC>,
        beta: T,
    ) where
        T: One,
        SB: Storage<T, R2, C2>,
        SC: Storage<T, D3>,
        ShapeConstraint: DimEq<D, R2> + AreMultipliable<R2, C2, D3, U1>,
    {
        // Safety: this is safe because we are passing Status == Init.
        unsafe { gemv_uninit(Init, self, alpha, a, x, beta) }
    }

    #[inline(always)]
    fn xxgemv<D2: Dim, D3: Dim, SB, SC>(
        &mut self,
        alpha: T,
        a: &SquareMatrix<T, D2, SB>,
        x: &Vector<T, D3, SC>,
        beta: T,
        dot: impl Fn(
            &DVectorView<'_, T, SB::RStride, SB::CStride>,
            &DVectorView<'_, T, SC::RStride, SC::CStride>,
        ) -> T,
    ) where
        T: One,
        SB: Storage<T, D2, D2>,
        SC: Storage<T, D3>,
        ShapeConstraint: DimEq<D, D2> + AreMultipliable<D2, D2, D3, U1>,
    {
        let dim1 = self.nrows();
        let dim2 = a.nrows();
        let dim3 = x.nrows();

        assert!(
            a.is_square(),
            "Symmetric cgemv: the input matrix must be square."
        );
        assert!(
            dim2 == dim3 && dim1 == dim2,
            "Symmetric cgemv: dimensions mismatch."
        );

        if dim2 == 0 {
            return;
        }

        // TODO: avoid bound checks.
        let col2 = a.column(0);
        let val = unsafe { x.vget_unchecked(0).clone() };
        self.axpy(alpha.clone() * val, &col2, beta);
        self[0] += alpha.clone() * dot(&a.view_range(1.., 0), &x.rows_range(1..));

        for j in 1..dim2 {
            let col2 = a.column(j);
            let dot = dot(&col2.rows_range(j..), &x.rows_range(j..));

            let val;
            unsafe {
                val = x.vget_unchecked(j).clone();
                *self.vget_unchecked_mut(j) += alpha.clone() * dot;
            }
            self.rows_range_mut(j + 1..).axpy(
                alpha.clone() * val,
                &col2.rows_range(j + 1..),
                T::one(),
            );
        }
    }

    /// Computes `self = alpha * a * x + beta * self` for a **symmetric** matrix (BLAS symv operation).
    ///
    /// This is an optimized version of [`gemv`](Self::gemv) for symmetric matrices. Because
    /// symmetric matrices have the property that `A = A^T`, only the lower triangular part
    /// (including diagonal) needs to be stored and accessed, making this operation about
    /// twice as fast as general matrix-vector multiplication.
    ///
    /// # Mathematical Definition
    ///
    /// `self = alpha * a * x + beta * self`
    ///
    /// Where:
    /// - `alpha` is a scalar multiplier for the matrix-vector product
    /// - `a` is a symmetric N×N matrix (only lower triangle is read)
    /// - `x` is a vector of length N
    /// - `beta` is a scalar multiplier for the current value of `self`
    /// - `self` must be a vector of length N
    ///
    /// # Important Notes
    ///
    /// - **Only the lower-triangular part** of `a` (including diagonal) is read
    /// - The upper-triangular part of `a` can contain any values (garbage is fine)
    /// - For **Hermitian** complex matrices, use [`hegemv`](Self::hegemv) instead
    ///
    /// # Performance Note
    ///
    /// If `beta` is zero, `self` is never read from, which can be more efficient.
    ///
    /// # Panics
    ///
    /// Panics if `a` is not square, or if dimensions are incompatible.
    ///
    /// # Examples
    ///
    /// ## Basic symmetric matrix-vector multiplication
    /// ```
    /// # use nalgebra::{Matrix2, Vector2};
    /// let mat = Matrix2::new(1.0, 2.0,  // Upper part (2.0) will be ignored
    ///                        2.0, 4.0); // Only lower part is read
    /// let mut vec1 = Vector2::new(1.0, 2.0);
    /// let vec2 = Vector2::new(0.1, 0.2);
    /// vec1.sygemv(10.0, &mat, &vec2, 5.0);
    /// assert_eq!(vec1, Vector2::new(10.0, 20.0));
    /// ```
    ///
    /// ## Upper triangle can be garbage
    /// ```
    /// # use nalgebra::{Matrix2, Vector2};
    /// // Only lower triangle matters for symmetric operations
    /// let mat = Matrix2::new(1.0, 9999999.9999999,  // Upper triangle is ignored
    ///                        2.0, 4.0);
    /// let mut vec1 = Vector2::new(1.0, 2.0);
    /// let vec2 = Vector2::new(0.1, 0.2);
    ///
    /// // Result is the same as if mat[0,1] = 2.0 (symmetric)
    /// vec1.sygemv(10.0, &mat, &vec2, 5.0);
    /// assert_eq!(vec1, Vector2::new(10.0, 20.0));
    /// ```
    ///
    /// ## Symmetric positive definite matrix multiplication
    /// ```
    /// # use nalgebra::{Matrix3, Vector3};
    /// // A symmetric positive definite matrix (e.g., covariance matrix)
    /// let mat = Matrix3::new(4.0, 0.0, 0.0,
    ///                        2.0, 5.0, 0.0,
    ///                        1.0, 3.0, 6.0);
    /// let x = Vector3::new(1.0, 1.0, 1.0);
    /// let mut result = Vector3::zeros();
    ///
    /// // Compute: result = mat * x (treating mat as symmetric)
    /// result.sygemv(1.0, &mat, &x, 0.0);
    /// assert_eq!(result, Vector3::new(7.0, 12.0, 16.0));
    /// ```
    ///
    /// ## Storing only lower triangle saves memory
    /// ```
    /// # use nalgebra::{Matrix2, Vector2};
    /// // In practice, you might only compute/store lower triangle
    /// let mut mat = Matrix2::identity();
    /// mat[(1, 0)] = 3.0;  // Only set lower triangle
    /// // mat[(0, 1)] is never set or used
    ///
    /// let x = Vector2::new(2.0, 1.0);
    /// let mut result = Vector2::zeros();
    /// result.sygemv(1.0, &mat, &x, 0.0);
    /// assert_eq!(result, Vector2::new(2.0, 7.0));  // Uses symmetry: mat[0,1]=mat[1,0]=3.0
    /// ```
    ///
    /// # See Also
    ///
    /// - [`gemv`](Self::gemv) - General matrix-vector multiplication
    /// - [`hegemv`](Self::hegemv) - Hermitian matrix-vector multiplication
    /// - [`syger`](Self::syger) - Symmetric rank-1 update
    #[inline]
    pub fn sygemv<D2: Dim, D3: Dim, SB, SC>(
        &mut self,
        alpha: T,
        a: &SquareMatrix<T, D2, SB>,
        x: &Vector<T, D3, SC>,
        beta: T,
    ) where
        T: One,
        SB: Storage<T, D2, D2>,
        SC: Storage<T, D3>,
        ShapeConstraint: DimEq<D, D2> + AreMultipliable<D2, D2, D3, U1>,
    {
        self.xxgemv(alpha, a, x, beta, |a, b| a.dot(b))
    }

    /// Computes `self = alpha * a * x + beta * self` for a **Hermitian** matrix (BLAS hemv operation).
    ///
    /// This is an optimized version of [`gemv`](Self::gemv) for Hermitian (complex symmetric)
    /// matrices. A Hermitian matrix satisfies `A = A*` (conjugate transpose), which means
    /// only the lower triangular part (including diagonal) needs to be stored. The diagonal
    /// must be real-valued.
    ///
    /// # Mathematical Definition
    ///
    /// `self = alpha * a * x + beta * self`
    ///
    /// Where:
    /// - `alpha` is a scalar multiplier for the matrix-vector product
    /// - `a` is a Hermitian N×N matrix (only lower triangle is read)
    /// - `x` is a complex vector of length N
    /// - `beta` is a scalar multiplier for the current value of `self`
    /// - `self` must be a complex vector of length N
    ///
    /// # Important Notes
    ///
    /// - **Only the lower-triangular part** of `a` (including diagonal) is read
    /// - The diagonal elements should be real (imaginary part is ignored)
    /// - The upper-triangular part of `a` can contain any values (garbage is fine)
    /// - For **real symmetric** matrices, use [`sygemv`](Self::sygemv) instead
    ///
    /// # Performance Note
    ///
    /// If `beta` is zero, `self` is never read from, which can be more efficient.
    ///
    /// # Panics
    ///
    /// Panics if `a` is not square, or if dimensions are incompatible.
    ///
    /// # Examples
    ///
    /// ## Basic Hermitian matrix-vector multiplication
    /// ```
    /// # use nalgebra::{Matrix2, Vector2, Complex};
    /// let mat = Matrix2::new(Complex::new(1.0, 0.0), Complex::new(2.0, -0.1),
    ///                        Complex::new(2.0, 1.0), Complex::new(4.0, 0.0));
    /// let mut vec1 = Vector2::new(Complex::new(1.0, 2.0), Complex::new(3.0, 4.0));
    /// let vec2 = Vector2::new(Complex::new(0.1, 0.2), Complex::new(0.3, 0.4));
    /// vec1.sygemv(Complex::new(10.0, 20.0), &mat, &vec2, Complex::new(5.0, 15.0));
    /// assert_eq!(vec1, Vector2::new(Complex::new(-48.0, 44.0), Complex::new(-75.0, 110.0)));
    /// ```
    ///
    /// ## Upper triangle can be garbage
    /// ```
    /// # use nalgebra::{Matrix2, Vector2, Complex};
    /// // The matrix upper-triangular elements can be garbage because they are never
    /// // read by this method. Only the lower triangle matters.
    ///
    /// let mat = Matrix2::new(Complex::new(1.0, 0.0), Complex::new(99999999.9, 999999999.9),
    ///                        Complex::new(2.0, 1.0), Complex::new(4.0, 0.0));
    /// let mut vec1 = Vector2::new(Complex::new(1.0, 2.0), Complex::new(3.0, 4.0));
    /// let vec2 = Vector2::new(Complex::new(0.1, 0.2), Complex::new(0.3, 0.4));
    /// vec1.sygemv(Complex::new(10.0, 20.0), &mat, &vec2, Complex::new(5.0, 15.0));
    /// assert_eq!(vec1, Vector2::new(Complex::new(-48.0, 44.0), Complex::new(-75.0, 110.0)));
    /// ```
    ///
    /// ## Hermitian matrix properties
    /// ```
    /// # use nalgebra::{Matrix2, Vector2, Complex};
    /// // For Hermitian matrices: A[i,j] = conj(A[j,i])
    /// // Diagonal must be real
    /// let mat = Matrix2::new(
    ///     Complex::new(3.0, 0.0),      // Diagonal: real
    ///     Complex::new(0.0, 0.0),      // Upper triangle (ignored)
    ///     Complex::new(1.0, 2.0),      // Lower triangle
    ///     Complex::new(5.0, 0.0)       // Diagonal: real
    /// );
    /// let x = Vector2::new(Complex::new(1.0, 0.0), Complex::new(0.0, 1.0));
    /// let mut result = Vector2::zeros();
    ///
    /// result.hegemv(Complex::new(1.0, 0.0), &mat, &x, Complex::new(0.0, 0.0));
    /// // The (1,0) element is used as conj(mat[0,1]) = conj(1+2i) = 1-2i
    /// assert_eq!(result, Vector2::new(Complex::new(3.0, -2.0), Complex::new(1.0, 2.0)));
    /// ```
    ///
    /// # See Also
    ///
    /// - [`sygemv`](Self::sygemv) - Symmetric matrix-vector multiplication (real matrices)
    /// - [`gemv`](Self::gemv) - General matrix-vector multiplication
    /// - [`hegerc`](Self::hegerc) - Hermitian rank-1 update
    #[inline]
    pub fn hegemv<D2: Dim, D3: Dim, SB, SC>(
        &mut self,
        alpha: T,
        a: &SquareMatrix<T, D2, SB>,
        x: &Vector<T, D3, SC>,
        beta: T,
    ) where
        T: SimdComplexField,
        SB: Storage<T, D2, D2>,
        SC: Storage<T, D3>,
        ShapeConstraint: DimEq<D, D2> + AreMultipliable<D2, D2, D3, U1>,
    {
        self.xxgemv(alpha, a, x, beta, |a, b| a.dotc(b))
    }

    #[inline(always)]
    fn gemv_xx<R2: Dim, C2: Dim, D3: Dim, SB, SC>(
        &mut self,
        alpha: T,
        a: &Matrix<T, R2, C2, SB>,
        x: &Vector<T, D3, SC>,
        beta: T,
        dot: impl Fn(&VectorView<'_, T, R2, SB::RStride, SB::CStride>, &Vector<T, D3, SC>) -> T,
    ) where
        T: One,
        SB: Storage<T, R2, C2>,
        SC: Storage<T, D3>,
        ShapeConstraint: DimEq<D, C2> + AreMultipliable<C2, R2, D3, U1>,
    {
        let dim1 = self.nrows();
        let (nrows2, ncols2) = a.shape();
        let dim3 = x.nrows();

        assert!(
            nrows2 == dim3 && dim1 == ncols2,
            "Gemv: dimensions mismatch."
        );

        if ncols2 == 0 {
            return;
        }

        if beta.is_zero() {
            for j in 0..ncols2 {
                let val = unsafe { self.vget_unchecked_mut(j) };
                *val = alpha.clone() * dot(&a.column(j), x)
            }
        } else {
            for j in 0..ncols2 {
                let val = unsafe { self.vget_unchecked_mut(j) };
                *val = alpha.clone() * dot(&a.column(j), x) + beta.clone() * val.clone();
            }
        }
    }

    /// Computes `self = alpha * a.transpose() * x + beta * self` (transposed GEMV operation).
    ///
    /// This is the transposed version of [`gemv`](Self::gemv). It multiplies the transpose of
    /// matrix `a` by vector `x`, avoiding the need to explicitly create the transposed matrix.
    /// This is more efficient than calling `a.transpose()` followed by `gemv`.
    ///
    /// # Mathematical Definition
    ///
    /// `self = alpha * a.transpose() * x + beta * self`
    ///
    /// Where:
    /// - `alpha` is a scalar multiplier for the matrix-vector product
    /// - `a` is an M×N matrix (but its transpose N×M is used)
    /// - `x` is a vector of length M
    /// - `beta` is a scalar multiplier for the current value of `self`
    /// - `self` must be a vector of length N
    ///
    /// # Performance Note
    ///
    /// If `beta` is zero, `self` is never read from, which can be more efficient.
    ///
    /// # Panics
    ///
    /// Panics if the dimensions are incompatible:
    /// - `self.len()` must equal `a.ncols()`
    /// - `x.len()` must equal `a.nrows()`
    ///
    /// # Examples
    ///
    /// ## Basic transposed matrix-vector multiplication
    /// ```
    /// # use nalgebra::{Matrix2, Vector2};
    /// let mat = Matrix2::new(1.0, 3.0,
    ///                        2.0, 4.0);
    /// let mut vec1 = Vector2::new(1.0, 2.0);
    /// let vec2 = Vector2::new(0.1, 0.2);
    /// let expected = mat.transpose() * vec2 * 10.0 + vec1 * 5.0;
    ///
    /// vec1.gemv_tr(10.0, &mat, &vec2, 5.0);
    /// assert_eq!(vec1, expected);
    /// ```
    ///
    /// ## Computing with non-square matrices
    /// ```
    /// # use nalgebra::{Matrix3x2, Vector3, Vector2};
    /// let mut result = Vector2::zeros();
    /// let mat = Matrix3x2::new(1.0, 4.0,
    ///                          2.0, 5.0,
    ///                          3.0, 6.0);
    /// let vec = Vector3::new(1.0, 2.0, 3.0);
    ///
    /// // mat is 3×2, mat.transpose() is 2×3, vec is 3×1
    /// // Result: mat.transpose() * vec is 2×1
    /// result.gemv_tr(1.0, &mat, &vec, 0.0);
    /// assert_eq!(result, Vector2::new(14.0, 32.0));
    /// ```
    ///
    /// ## Least squares: computing A^T * b
    /// ```
    /// # use nalgebra::{Matrix3x2, Vector3, Vector2};
    /// // In least squares, we often need to compute A^T * b
    /// let a = Matrix3x2::new(1.0, 1.0,
    ///                        1.0, 2.0,
    ///                        1.0, 3.0);
    /// let b = Vector3::new(1.0, 2.0, 2.0);
    /// let mut atb = Vector2::zeros();
    ///
    /// // Compute A^T * b
    /// atb.gemv_tr(1.0, &a, &b, 0.0);
    /// assert_eq!(atb, Vector2::new(5.0, 11.0));
    /// ```
    ///
    /// ## Efficient alternative to explicit transpose
    /// ```
    /// # use nalgebra::{Matrix2x3, Vector2, Vector3};
    /// let mat = Matrix2x3::new(1.0, 2.0, 3.0,
    ///                          4.0, 5.0, 6.0);
    /// let vec = Vector2::new(1.0, 2.0);
    /// let mut result1 = Vector3::zeros();
    /// let mut result2 = Vector3::zeros();
    ///
    /// // Both produce the same result, but gemv_tr is more efficient
    /// result1.gemv_tr(1.0, &mat, &vec, 0.0);
    /// result2.gemv(1.0, &mat.transpose(), &vec, 0.0);
    /// assert_eq!(result1, result2);
    /// ```
    ///
    /// # See Also
    ///
    /// - [`gemv`](Self::gemv) - Standard matrix-vector multiplication
    /// - [`gemv_ad`](Self::gemv_ad) - Matrix-vector multiplication with adjoint (conjugate transpose)
    /// - [`gemm_tr`](Self::gemm_tr) - Matrix-matrix multiplication with transpose
    #[inline]
    pub fn gemv_tr<R2: Dim, C2: Dim, D3: Dim, SB, SC>(
        &mut self,
        alpha: T,
        a: &Matrix<T, R2, C2, SB>,
        x: &Vector<T, D3, SC>,
        beta: T,
    ) where
        T: One,
        SB: Storage<T, R2, C2>,
        SC: Storage<T, D3>,
        ShapeConstraint: DimEq<D, C2> + AreMultipliable<C2, R2, D3, U1>,
    {
        self.gemv_xx(alpha, a, x, beta, |a, b| a.dot(b))
    }

    /// Computes `self = alpha * a.adjoint() * x + beta * self` (adjoint GEMV operation).
    ///
    /// This is the adjoint (conjugate transpose) version of [`gemv`](Self::gemv). It multiplies
    /// the adjoint (conjugate transpose) of matrix `a` by vector `x`, avoiding the need to
    /// explicitly create the adjoint matrix. For complex matrices, this uses conjugation.
    ///
    /// # Mathematical Definition
    ///
    /// `self = alpha * a.adjoint() * x + beta * self`
    ///
    /// Where:
    /// - `alpha` is a scalar multiplier
    /// - `a` is an M×N matrix (but its adjoint N×M is used)
    /// - `a.adjoint()` is the conjugate transpose: `(a.adjoint())[i,j] = conj(a[j,i])`
    /// - `x` is a vector of length M
    /// - `beta` is a scalar multiplier for the current value of `self`
    /// - `self` must be a vector of length N
    ///
    /// # Important Notes
    ///
    /// - For **real** matrices, this is identical to [`gemv_tr`](Self::gemv_tr) (no conjugation)
    /// - For **complex** matrices, entries are conjugated during the transpose
    ///
    /// # Performance Note
    ///
    /// If `beta` is zero, `self` is never read from, which can be more efficient.
    ///
    /// # Panics
    ///
    /// Panics if dimensions are incompatible.
    ///
    /// # Examples
    ///
    /// ## Complex matrix adjoint multiplication
    /// ```
    /// # use nalgebra::{Matrix2, Vector2, Complex};
    /// let mat = Matrix2::new(Complex::new(1.0, 2.0), Complex::new(3.0, 4.0),
    ///                        Complex::new(5.0, 6.0), Complex::new(7.0, 8.0));
    /// let mut vec1 = Vector2::new(Complex::new(1.0, 2.0), Complex::new(3.0, 4.0));
    /// let vec2 = Vector2::new(Complex::new(0.1, 0.2), Complex::new(0.3, 0.4));
    /// let expected = mat.adjoint() * vec2 * Complex::new(10.0, 20.0) + vec1 * Complex::new(5.0, 15.0);
    ///
    /// vec1.gemv_ad(Complex::new(10.0, 20.0), &mat, &vec2, Complex::new(5.0, 15.0));
    /// assert_eq!(vec1, expected);
    /// ```
    ///
    /// ## Difference between gemv_tr and gemv_ad for complex matrices
    /// ```
    /// # use nalgebra::{Matrix2, Vector2, Complex};
    /// let mat = Matrix2::new(Complex::new(1.0, 1.0), Complex::new(0.0, 0.0),
    ///                        Complex::new(0.0, 0.0), Complex::new(1.0, -1.0));
    /// let vec = Vector2::new(Complex::new(1.0, 0.0), Complex::new(0.0, 1.0));
    /// let mut result_tr = Vector2::zeros();
    /// let mut result_ad = Vector2::zeros();
    ///
    /// // gemv_tr transposes without conjugation
    /// result_tr.gemv_tr(Complex::new(1.0, 0.0), &mat, &vec, Complex::new(0.0, 0.0));
    /// // gemv_ad transposes WITH conjugation
    /// result_ad.gemv_ad(Complex::new(1.0, 0.0), &mat, &vec, Complex::new(0.0, 0.0));
    ///
    /// // Results differ due to conjugation
    /// assert_ne!(result_tr, result_ad);
    /// ```
    ///
    /// ## Real matrices: gemv_ad equals gemv_tr
    /// ```
    /// # use nalgebra::{Matrix2, Vector2};
    /// let mat = Matrix2::new(1.0, 2.0, 3.0, 4.0);
    /// let vec = Vector2::new(1.0, 1.0);
    /// let mut result1 = Vector2::zeros();
    /// let mut result2 = Vector2::zeros();
    ///
    /// result1.gemv_ad(1.0, &mat, &vec, 0.0);
    /// result2.gemv_tr(1.0, &mat, &vec, 0.0);
    /// assert_eq!(result1, result2);  // Same for real matrices
    /// ```
    ///
    /// ## Quantum mechanics: computing ⟨ψ|A|φ⟩
    /// ```
    /// # use nalgebra::{Matrix2, Vector2, Complex};
    /// let operator = Matrix2::new(
    ///     Complex::new(1.0, 0.0), Complex::new(0.0, -1.0),
    ///     Complex::new(0.0, 1.0), Complex::new(-1.0, 0.0)
    /// );
    /// let psi = Vector2::new(Complex::new(1.0, 0.0), Complex::new(0.0, 1.0));
    /// let phi = Vector2::new(Complex::new(1.0, 0.0), Complex::new(1.0, 0.0));
    /// let mut temp = Vector2::zeros();
    ///
    /// // Compute A|φ⟩ first, then take inner product with ⟨ψ|
    /// temp.gemv(Complex::new(1.0, 0.0), &operator, &phi, Complex::new(0.0, 0.0));
    /// let expectation = psi.dotc(&temp);
    /// ```
    ///
    /// # See Also
    ///
    /// - [`gemv`](Self::gemv) - Standard matrix-vector multiplication
    /// - [`gemv_tr`](Self::gemv_tr) - Matrix-vector multiplication with transpose (no conjugation)
    /// - [`gemm_ad`](Self::gemm_ad) - Matrix-matrix multiplication with adjoint
    /// - [`dotc`](Self::dotc) - Conjugate dot product
    #[inline]
    pub fn gemv_ad<R2: Dim, C2: Dim, D3: Dim, SB, SC>(
        &mut self,
        alpha: T,
        a: &Matrix<T, R2, C2, SB>,
        x: &Vector<T, D3, SC>,
        beta: T,
    ) where
        T: SimdComplexField,
        SB: Storage<T, R2, C2>,
        SC: Storage<T, D3>,
        ShapeConstraint: DimEq<D, C2> + AreMultipliable<C2, R2, D3, U1>,
    {
        self.gemv_xx(alpha, a, x, beta, |a, b| a.dotc(b))
    }
}

impl<T, R1: Dim, C1: Dim, S: StorageMut<T, R1, C1>> Matrix<T, R1, C1, S>
where
    T: Scalar + Zero + ClosedAddAssign + ClosedMulAssign,
{
    #[inline(always)]
    fn gerx<D2: Dim, D3: Dim, SB, SC>(
        &mut self,
        alpha: T,
        x: &Vector<T, D2, SB>,
        y: &Vector<T, D3, SC>,
        beta: T,
        conjugate: impl Fn(T) -> T,
    ) where
        T: One,
        SB: Storage<T, D2>,
        SC: Storage<T, D3>,
        ShapeConstraint: DimEq<R1, D2> + DimEq<C1, D3>,
    {
        let (nrows1, ncols1) = self.shape();
        let dim2 = x.nrows();
        let dim3 = y.nrows();

        assert!(
            nrows1 == dim2 && ncols1 == dim3,
            "ger: dimensions mismatch."
        );

        for j in 0..ncols1 {
            // TODO: avoid bound checks.
            let val = unsafe { conjugate(y.vget_unchecked(j).clone()) };
            self.column_mut(j)
                .axpy(alpha.clone() * val, x, beta.clone());
        }
    }

    /// Computes `self = alpha * x * y.transpose() + beta * self` (BLAS ger operation).
    ///
    /// GER stands for "GEneral Rank-1 update" and is a BLAS Level 2 operation that performs
    /// an outer product update on a matrix. The outer product of two vectors produces a matrix,
    /// which is then scaled and added to the current matrix.
    ///
    /// # Mathematical Definition
    ///
    /// `self = alpha * x * y.transpose() + beta * self`
    ///
    /// Where:
    /// - `alpha` is a scalar multiplier for the outer product
    /// - `x` is a vector of length M
    /// - `y` is a vector of length N
    /// - `beta` is a scalar multiplier for the current value of `self`
    /// - `self` must be an M×N matrix
    /// - `x * y.transpose()` produces an M×N matrix (outer product)
    ///
    /// # Performance Note
    ///
    /// If `beta` is zero, `self` is never read from, which can be more efficient.
    ///
    /// # Panics
    ///
    /// Panics if the dimensions are incompatible:
    /// - `self.nrows()` must equal `x.len()`
    /// - `self.ncols()` must equal `y.len()`
    ///
    /// # Examples
    ///
    /// ## Basic rank-1 update
    /// ```
    /// # use nalgebra::{Matrix2x3, Vector2, Vector3};
    /// let mut mat = Matrix2x3::repeat(4.0);
    /// let vec1 = Vector2::new(1.0, 2.0);
    /// let vec2 = Vector3::new(0.1, 0.2, 0.3);
    /// let expected = vec1 * vec2.transpose() * 10.0 + mat * 5.0;
    ///
    /// mat.ger(10.0, &vec1, &vec2, 5.0);
    /// assert_eq!(mat, expected);
    /// ```
    ///
    /// ## Creating outer product matrix
    /// ```
    /// # use nalgebra::{Matrix3x2, Vector3, Vector2};
    /// let mut mat = Matrix3x2::zeros();
    /// let x = Vector3::new(1.0, 2.0, 3.0);
    /// let y = Vector2::new(4.0, 5.0);
    ///
    /// // Create outer product: x * y^T
    /// mat.ger(1.0, &x, &y, 0.0);
    /// assert_eq!(mat, Matrix3x2::new(4.0, 5.0,
    ///                                8.0, 10.0,
    ///                                12.0, 15.0));
    /// ```
    ///
    /// ## Building rank-deficient matrices
    /// ```
    /// # use nalgebra::{Matrix2, Vector2};
    /// // Build a rank-1 matrix from two vectors
    /// let mut matrix = Matrix2::zeros();
    /// let u = Vector2::new(1.0, 2.0);
    /// let v = Vector2::new(3.0, 4.0);
    ///
    /// matrix.ger(1.0, &u, &v, 0.0);
    /// // Result is rank-1: all rows/columns are linearly dependent
    /// assert_eq!(matrix, Matrix2::new(3.0, 4.0,
    ///                                 6.0, 8.0));
    /// ```
    ///
    /// ## Incremental matrix construction
    /// ```
    /// # use nalgebra::{Matrix2, Vector2};
    /// let mut mat = Matrix2::zeros();
    /// let e1 = Vector2::new(1.0, 0.0);
    /// let e2 = Vector2::new(0.0, 1.0);
    ///
    /// // Build identity matrix incrementally
    /// mat.ger(1.0, &e1, &e1, 0.0);  // Add first basis vector
    /// mat.ger(1.0, &e2, &e2, 1.0);  // Add second basis vector
    /// assert_eq!(mat, Matrix2::identity());
    /// ```
    ///
    /// ## Low-rank approximation updates
    /// ```
    /// # use nalgebra::{Matrix2, Vector2};
    /// let mut approx = Matrix2::identity();
    /// let correction = Vector2::new(0.1, 0.2);
    /// let direction = Vector2::new(1.0, -1.0);
    ///
    /// // Add a rank-1 correction to improve approximation
    /// approx.ger(1.0, &correction, &direction, 1.0);
    /// ```
    ///
    /// # See Also
    ///
    /// - [`gerc`](Self::gerc) - Rank-1 update with conjugate transpose (for complex matrices)
    /// - [`syger`](Self::syger) - Symmetric rank-1 update
    /// - [`hegerc`](Self::hegerc) - Hermitian rank-1 update
    /// - [`gemm`](Self::gemm) - General matrix-matrix multiplication (includes rank-k updates)
    #[inline]
    pub fn ger<D2: Dim, D3: Dim, SB, SC>(
        &mut self,
        alpha: T,
        x: &Vector<T, D2, SB>,
        y: &Vector<T, D3, SC>,
        beta: T,
    ) where
        T: One,
        SB: Storage<T, D2>,
        SC: Storage<T, D3>,
        ShapeConstraint: DimEq<R1, D2> + DimEq<C1, D3>,
    {
        self.gerx(alpha, x, y, beta, |e| e)
    }

    /// Computes `self = alpha * x * y.adjoint() + beta * self` (conjugate GER operation).
    ///
    /// GERC stands for "GEneral Rank-1 update Conjugate" and is the complex version of
    /// [`ger`](Self::ger). It performs a rank-1 update using the conjugate (adjoint) of
    /// vector `y`, which is essential for working with complex matrices.
    ///
    /// # Mathematical Definition
    ///
    /// `self = alpha * x * y.adjoint() + beta * self`
    ///
    /// Where:
    /// - `alpha` is a complex scalar multiplier
    /// - `x` is a complex vector of length M
    /// - `y` is a complex vector of length N
    /// - `y.adjoint()` is the conjugate transpose of `y`
    /// - `beta` is a complex scalar multiplier for the current value of `self`
    /// - `self` must be an M×N matrix
    ///
    /// # Important Notes
    ///
    /// - For **real** matrices, this is identical to [`ger`](Self::ger) (no conjugation)
    /// - For **complex** matrices, elements of `y` are conjugated
    /// - This is the proper operation for building certain Hermitian matrices
    ///
    /// # Performance Note
    ///
    /// If `beta` is zero, `self` is never read from, which can be more efficient.
    ///
    /// # Panics
    ///
    /// Panics if dimensions are incompatible.
    ///
    /// # Examples
    ///
    /// ## Basic conjugate rank-1 update
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{Matrix2x3, Vector2, Vector3, Complex};
    /// let mut mat = Matrix2x3::repeat(Complex::new(4.0, 5.0));
    /// let vec1 = Vector2::new(Complex::new(1.0, 2.0), Complex::new(3.0, 4.0));
    /// let vec2 = Vector3::new(Complex::new(0.6, 0.5), Complex::new(0.4, 0.5), Complex::new(0.2, 0.1));
    /// let expected = vec1 * vec2.adjoint() * Complex::new(10.0, 20.0) + mat * Complex::new(5.0, 15.0);
    ///
    /// mat.gerc(Complex::new(10.0, 20.0), &vec1, &vec2, Complex::new(5.0, 15.0));
    /// assert_eq!(mat, expected);
    /// ```
    ///
    /// ## Difference between ger and gerc for complex vectors
    /// ```
    /// # use nalgebra::{Matrix2, Vector2, Complex};
    /// let mut mat_ger = Matrix2::zeros();
    /// let mut mat_gerc = Matrix2::zeros();
    /// let x = Vector2::new(Complex::new(1.0, 0.0), Complex::new(0.0, 1.0));
    /// let y = Vector2::new(Complex::new(1.0, 1.0), Complex::new(1.0, -1.0));
    ///
    /// // ger: no conjugation
    /// mat_ger.ger(Complex::new(1.0, 0.0), &x, &y, Complex::new(0.0, 0.0));
    /// // gerc: conjugates y
    /// mat_gerc.gerc(Complex::new(1.0, 0.0), &x, &y, Complex::new(0.0, 0.0));
    ///
    /// // Results differ due to conjugation of y
    /// assert_ne!(mat_ger, mat_gerc);
    /// ```
    ///
    /// ## Building outer product with conjugation
    /// ```
    /// # use nalgebra::{Matrix2, Vector2, Complex};
    /// let mut mat = Matrix2::zeros();
    /// let u = Vector2::new(Complex::new(1.0, 2.0), Complex::new(3.0, 4.0));
    /// let v = Vector2::new(Complex::new(1.0, 1.0), Complex::new(2.0, -1.0));
    ///
    /// // Compute u * v*
    /// mat.gerc(Complex::new(1.0, 0.0), &u, &v, Complex::new(0.0, 0.0));
    /// // mat[i,j] = u[i] * conj(v[j])
    /// assert_eq!(mat[(0,0)], Complex::new(1.0, 2.0) * Complex::new(1.0, -1.0));
    /// assert_eq!(mat[(0,1)], Complex::new(1.0, 2.0) * Complex::new(2.0, 1.0));
    /// ```
    ///
    /// ## Real matrices: gerc equals ger
    /// ```
    /// # use nalgebra::{Matrix2, Vector2};
    /// let mut mat1 = Matrix2::zeros();
    /// let mut mat2 = Matrix2::zeros();
    /// let x = Vector2::new(1.0, 2.0);
    /// let y = Vector2::new(3.0, 4.0);
    ///
    /// mat1.ger(1.0, &x, &y, 0.0);
    /// mat2.gerc(1.0, &x, &y, 0.0);
    /// assert_eq!(mat1, mat2);  // Same for real numbers
    /// ```
    ///
    /// ## Building matrices for quantum operations
    /// ```
    /// # use nalgebra::{Matrix2, Vector2, Complex};
    /// let mut projector = Matrix2::zeros();
    /// let state = Vector2::new(
    ///     Complex::new(1.0/2.0_f64.sqrt(), 0.0),
    ///     Complex::new(0.0, 1.0/2.0_f64.sqrt())
    /// );
    ///
    /// // Create projector |ψ⟩⟨ψ| = ψ * ψ*
    /// projector.gerc(Complex::new(1.0, 0.0), &state, &state, Complex::new(0.0, 0.0));
    /// // projector is Hermitian and idempotent
    /// ```
    ///
    /// # See Also
    ///
    /// - [`ger`](Self::ger) - General rank-1 update (no conjugation)
    /// - [`hegerc`](Self::hegerc) - Hermitian rank-1 update (symmetric result)
    /// - [`gemv_ad`](Self::gemv_ad) - Matrix-vector multiplication with adjoint
    #[inline]
    pub fn gerc<D2: Dim, D3: Dim, SB, SC>(
        &mut self,
        alpha: T,
        x: &Vector<T, D2, SB>,
        y: &Vector<T, D3, SC>,
        beta: T,
    ) where
        T: SimdComplexField,
        SB: Storage<T, D2>,
        SC: Storage<T, D3>,
        ShapeConstraint: DimEq<R1, D2> + DimEq<C1, D3>,
    {
        self.gerx(alpha, x, y, beta, SimdComplexField::simd_conjugate)
    }

    /// Computes `self = alpha * a * b + beta * self` (BLAS gemm operation).
    ///
    /// GEMM stands for "GEneral Matrix-Matrix multiplication" and is the most important BLAS Level 3
    /// operation. It computes a scaled matrix-matrix product and adds it to a scaled matrix.
    /// This is the workhorse of numerical linear algebra, used in everything from neural networks
    /// to scientific simulations.
    ///
    /// # Mathematical Definition
    ///
    /// `self = alpha * a * b + beta * self`
    ///
    /// Where:
    /// - `alpha` is a scalar multiplier for the matrix product
    /// - `a` is an M×K matrix
    /// - `b` is a K×N matrix
    /// - `beta` is a scalar multiplier for the current value of `self`
    /// - `self` must be an M×N matrix
    /// - `a * b` produces an M×N matrix
    ///
    /// # Performance Note
    ///
    /// If `beta` is zero, `self` is never read from, which can be more efficient.
    /// This operation is highly optimized and should be used instead of manual implementation.
    ///
    /// # Panics
    ///
    /// Panics if the dimensions are incompatible:
    /// - `a.ncols()` must equal `b.nrows()` (for the multiplication)
    /// - `self.nrows()` must equal `a.nrows()`
    /// - `self.ncols()` must equal `b.ncols()`
    ///
    /// # Examples
    ///
    /// ## Basic matrix multiplication
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{Matrix2x3, Matrix3x4, Matrix2x4};
    /// let mut mat1 = Matrix2x4::identity();
    /// let mat2 = Matrix2x3::new(1.0, 2.0, 3.0,
    ///                           4.0, 5.0, 6.0);
    /// let mat3 = Matrix3x4::new(0.1, 0.2, 0.3, 0.4,
    ///                           0.5, 0.6, 0.7, 0.8,
    ///                           0.9, 1.0, 1.1, 1.2);
    /// let expected = mat2 * mat3 * 10.0 + mat1 * 5.0;
    ///
    /// mat1.gemm(10.0, &mat2, &mat3, 5.0);
    /// assert_relative_eq!(mat1, expected);
    /// ```
    ///
    /// ## Simple matrix product (alpha=1, beta=0)
    /// ```
    /// # use nalgebra::{Matrix2, Matrix2x3, Matrix3x2};
    /// let mut result = Matrix2::zeros();
    /// let a = Matrix2x3::new(1.0, 2.0, 3.0,
    ///                        4.0, 5.0, 6.0);
    /// let b = Matrix3x2::new(7.0, 8.0,
    ///                        9.0, 10.0,
    ///                        11.0, 12.0);
    ///
    /// // Simple product: result = a * b
    /// result.gemm(1.0, &a, &b, 0.0);
    /// assert_eq!(result, Matrix2::new(58.0, 64.0,
    ///                                 139.0, 154.0));
    /// ```
    ///
    /// ## Accumulating matrix products
    /// ```
    /// # use nalgebra::Matrix2;
    /// let mut sum = Matrix2::zeros();
    /// let a = Matrix2::new(1.0, 2.0, 3.0, 4.0);
    /// let b = Matrix2::new(5.0, 6.0, 7.0, 8.0);
    /// let c = Matrix2::new(1.0, 0.0, 0.0, 1.0);
    ///
    /// // Accumulate: sum = a*b + c*c
    /// sum.gemm(1.0, &a, &b, 0.0);
    /// sum.gemm(1.0, &c, &c, 1.0);
    /// assert_eq!(sum, Matrix2::new(20.0, 22.0, 44.0, 51.0));
    /// ```
    ///
    /// ## Computing A * A (matrix squared)
    /// ```
    /// # use nalgebra::Matrix2;
    /// let a = Matrix2::new(1.0, 2.0,
    ///                      3.0, 4.0);
    /// let mut a_squared = Matrix2::zeros();
    ///
    /// // Compute A²
    /// a_squared.gemm(1.0, &a, &a, 0.0);
    /// assert_eq!(a_squared, Matrix2::new(7.0, 10.0,
    ///                                    15.0, 22.0));
    /// ```
    ///
    /// ## Linear combination of matrix products
    /// ```
    /// # use nalgebra::Matrix2;
    /// let a = Matrix2::new(1.0, 0.0, 0.0, 2.0);
    /// let b = Matrix2::new(3.0, 4.0, 5.0, 6.0);
    /// let c = Matrix2::new(1.0, 1.0, 1.0, 1.0);
    /// let mut result = Matrix2::from_element(10.0);
    ///
    /// // Compute: result = 2*A*B + 3*C
    /// result.gemm(2.0, &a, &b, 3.0);
    /// assert_eq!(result, Matrix2::new(36.0, 38.0,
    ///                                 40.0, 54.0));
    /// ```
    ///
    /// # See Also
    ///
    /// - [`gemm_tr`](Self::gemm_tr) - Matrix multiplication with transposed left matrix
    /// - [`gemm_ad`](Self::gemm_ad) - Matrix multiplication with adjoint (conjugate transpose)
    /// - [`gemv`](Self::gemv) - Matrix-vector multiplication
    /// - [`mul`](core::ops::Mul::mul) or `*` operator - Simple matrix multiplication
    #[inline]
    pub fn gemm<R2: Dim, C2: Dim, R3: Dim, C3: Dim, SB, SC>(
        &mut self,
        alpha: T,
        a: &Matrix<T, R2, C2, SB>,
        b: &Matrix<T, R3, C3, SC>,
        beta: T,
    ) where
        T: One,
        SB: Storage<T, R2, C2>,
        SC: Storage<T, R3, C3>,
        ShapeConstraint: SameNumberOfRows<R1, R2>
            + SameNumberOfColumns<C1, C3>
            + AreMultipliable<R2, C2, R3, C3>,
    {
        // SAFETY: this is valid because our matrices are initialized and
        // we are using status = Init.
        unsafe { gemm_uninit(Init, self, alpha, a, b, beta) }
    }

    /// Computes `self = alpha * a.transpose() * b + beta * self` (transposed GEMM).
    ///
    /// This is the transposed version of [`gemm`](Self::gemm). It multiplies the transpose of
    /// matrix `a` with matrix `b`, avoiding the need to explicitly create the transposed matrix.
    /// This is more efficient than calling `a.transpose()` followed by `gemm`.
    ///
    /// # Mathematical Definition
    ///
    /// `self = alpha * a.transpose() * b + beta * self`
    ///
    /// Where:
    /// - `alpha` is a scalar multiplier for the matrix product
    /// - `a` is a K×M matrix (but its transpose M×K is used)
    /// - `b` is a K×N matrix
    /// - `beta` is a scalar multiplier for the current value of `self`
    /// - `self` must be an M×N matrix
    ///
    /// # Performance Note
    ///
    /// If `beta` is zero, `self` is never read from, which can be more efficient.
    ///
    /// # Panics
    ///
    /// Panics if the dimensions are incompatible.
    ///
    /// # Examples
    ///
    /// ## Basic transposed matrix multiplication
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{Matrix3x2, Matrix3x4, Matrix2x4};
    /// let mut mat1 = Matrix2x4::identity();
    /// let mat2 = Matrix3x2::new(1.0, 4.0,
    ///                           2.0, 5.0,
    ///                           3.0, 6.0);
    /// let mat3 = Matrix3x4::new(0.1, 0.2, 0.3, 0.4,
    ///                           0.5, 0.6, 0.7, 0.8,
    ///                           0.9, 1.0, 1.1, 1.2);
    /// let expected = mat2.transpose() * mat3 * 10.0 + mat1 * 5.0;
    ///
    /// mat1.gemm_tr(10.0, &mat2, &mat3, 5.0);
    /// assert_eq!(mat1, expected);
    /// ```
    ///
    /// ## Computing A^T * A (Gram matrix)
    /// ```
    /// # use nalgebra::{Matrix3x2, Matrix2};
    /// let a = Matrix3x2::new(1.0, 2.0,
    ///                        3.0, 4.0,
    ///                        5.0, 6.0);
    /// let mut gram = Matrix2::zeros();
    ///
    /// // Compute Gram matrix: A^T * A
    /// gram.gemm_tr(1.0, &a, &a, 0.0);
    /// assert_eq!(gram, Matrix2::new(35.0, 44.0,
    ///                               44.0, 56.0));
    /// ```
    ///
    /// ## Efficient alternative to explicit transpose
    /// ```
    /// # use nalgebra::{Matrix3x2, Matrix3x4, Matrix2x4};
    /// let mat_a = Matrix3x2::new(1.0, 4.0, 2.0, 5.0, 3.0, 6.0);
    /// let mat_b = Matrix3x4::new(0.1, 0.2, 0.3, 0.4,
    ///                            0.5, 0.6, 0.7, 0.8,
    ///                            0.9, 1.0, 1.1, 1.2);
    /// let mut result1 = Matrix2x4::zeros();
    /// let mut result2 = Matrix2x4::zeros();
    ///
    /// // Both are equivalent, but gemm_tr is more efficient
    /// result1.gemm_tr(1.0, &mat_a, &mat_b, 0.0);
    /// result2.gemm(1.0, &mat_a.transpose(), &mat_b, 0.0);
    /// assert_eq!(result1, result2);
    /// ```
    ///
    /// # See Also
    ///
    /// - [`gemm`](Self::gemm) - Standard matrix-matrix multiplication
    /// - [`gemm_ad`](Self::gemm_ad) - Matrix multiplication with adjoint (conjugate transpose)
    /// - [`gemv_tr`](Self::gemv_tr) - Matrix-vector multiplication with transpose
    #[inline]
    pub fn gemm_tr<R2: Dim, C2: Dim, R3: Dim, C3: Dim, SB, SC>(
        &mut self,
        alpha: T,
        a: &Matrix<T, R2, C2, SB>,
        b: &Matrix<T, R3, C3, SC>,
        beta: T,
    ) where
        T: One,
        SB: Storage<T, R2, C2>,
        SC: Storage<T, R3, C3>,
        ShapeConstraint: SameNumberOfRows<R1, C2>
            + SameNumberOfColumns<C1, C3>
            + AreMultipliable<C2, R2, R3, C3>,
    {
        let (nrows1, ncols1) = self.shape();
        let (nrows2, ncols2) = a.shape();
        let (nrows3, ncols3) = b.shape();

        assert_eq!(
            nrows2, nrows3,
            "gemm: dimensions mismatch for multiplication."
        );
        assert_eq!(
            (nrows1, ncols1),
            (ncols2, ncols3),
            "gemm: dimensions mismatch for addition."
        );

        for j1 in 0..ncols1 {
            // TODO: avoid bound checks.
            self.column_mut(j1)
                .gemv_tr(alpha.clone(), a, &b.column(j1), beta.clone());
        }
    }

    /// Computes `self = alpha * a.adjoint() * b + beta * self` (adjoint GEMM operation).
    ///
    /// This is the adjoint (conjugate transpose) version of [`gemm`](Self::gemm). It multiplies
    /// the adjoint of matrix `a` with matrix `b`, avoiding the need to explicitly create the
    /// adjoint matrix. For complex matrices, this includes conjugation of the elements.
    ///
    /// # Mathematical Definition
    ///
    /// `self = alpha * a.adjoint() * b + beta * self`
    ///
    /// Where:
    /// - `alpha` is a scalar multiplier
    /// - `a` is a K×M matrix (but its adjoint M×K is used)
    /// - `a.adjoint()` is the conjugate transpose: `(a.adjoint())[i,j] = conj(a[j,i])`
    /// - `b` is a K×N matrix
    /// - `beta` is a scalar multiplier for the current value of `self`
    /// - `self` must be an M×N matrix
    ///
    /// # Important Notes
    ///
    /// - For **real** matrices, this is identical to [`gemm_tr`](Self::gemm_tr) (no conjugation)
    /// - For **complex** matrices, entries of `a` are conjugated during the transpose
    ///
    /// # Performance Note
    ///
    /// If `beta` is zero, `self` is never read from, which can be more efficient.
    ///
    /// # Panics
    ///
    /// Panics if dimensions are incompatible.
    ///
    /// # Examples
    ///
    /// ## Basic adjoint matrix multiplication
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{Matrix3x2, Matrix3x4, Matrix2x4, Complex};
    /// let mut mat1 = Matrix2x4::identity();
    /// let mat2 = Matrix3x2::new(Complex::new(1.0, 4.0), Complex::new(7.0, 8.0),
    ///                           Complex::new(2.0, 5.0), Complex::new(9.0, 10.0),
    ///                           Complex::new(3.0, 6.0), Complex::new(11.0, 12.0));
    /// let mat3 = Matrix3x4::new(Complex::new(0.1, 1.3), Complex::new(0.2, 1.4), Complex::new(0.3, 1.5), Complex::new(0.4, 1.6),
    ///                           Complex::new(0.5, 1.7), Complex::new(0.6, 1.8), Complex::new(0.7, 1.9), Complex::new(0.8, 2.0),
    ///                           Complex::new(0.9, 2.1), Complex::new(1.0, 2.2), Complex::new(1.1, 2.3), Complex::new(1.2, 2.4));
    /// let expected = mat2.adjoint() * mat3 * Complex::new(10.0, 20.0) + mat1 * Complex::new(5.0, 15.0);
    ///
    /// mat1.gemm_ad(Complex::new(10.0, 20.0), &mat2, &mat3, Complex::new(5.0, 15.0));
    /// assert_eq!(mat1, expected);
    /// ```
    ///
    /// ## Computing A* A (Gram matrix with conjugation)
    /// ```
    /// # use nalgebra::{Matrix3x2, Matrix2, Complex};
    /// let a = Matrix3x2::new(
    ///     Complex::new(1.0, 1.0), Complex::new(2.0, 0.0),
    ///     Complex::new(0.0, 1.0), Complex::new(1.0, 1.0),
    ///     Complex::new(1.0, 0.0), Complex::new(0.0, 1.0)
    /// );
    /// let mut gram = Matrix2::zeros();
    ///
    /// // Compute A* * A (Hermitian Gram matrix)
    /// gram.gemm_ad(Complex::new(1.0, 0.0), &a, &a, Complex::new(0.0, 0.0));
    /// // Result is Hermitian (equal to its adjoint)
    /// ```
    ///
    /// ## Difference between gemm_tr and gemm_ad for complex matrices
    /// ```
    /// # use nalgebra::{Matrix2, Complex};
    /// let a = Matrix2::new(
    ///     Complex::new(1.0, 1.0), Complex::new(0.0, 0.0),
    ///     Complex::new(0.0, 0.0), Complex::new(1.0, -1.0)
    /// );
    /// let b = Matrix2::identity();
    /// let mut result_tr = Matrix2::zeros();
    /// let mut result_ad = Matrix2::zeros();
    ///
    /// // gemm_tr: transpose without conjugation
    /// result_tr.gemm_tr(Complex::new(1.0, 0.0), &a, &b, Complex::new(0.0, 0.0));
    /// // gemm_ad: transpose WITH conjugation
    /// result_ad.gemm_ad(Complex::new(1.0, 0.0), &a, &b, Complex::new(0.0, 0.0));
    ///
    /// // Results differ due to conjugation
    /// assert_ne!(result_tr, result_ad);
    /// ```
    ///
    /// ## Real matrices: gemm_ad equals gemm_tr
    /// ```
    /// # use nalgebra::{Matrix2x3, Matrix3x2, Matrix2};
    /// let a = Matrix3x2::new(1.0, 4.0, 2.0, 5.0, 3.0, 6.0);
    /// let b = Matrix3x2::new(0.1, 0.4, 0.2, 0.5, 0.3, 0.6);
    /// let mut result1 = Matrix2::zeros();
    /// let mut result2 = Matrix2::zeros();
    ///
    /// result1.gemm_ad(1.0, &a, &b, 0.0);
    /// result2.gemm_tr(1.0, &a, &b, 0.0);
    /// assert_eq!(result1, result2);  // Same for real matrices
    /// ```
    ///
    /// ## Efficient alternative to explicit adjoint
    /// ```
    /// # use nalgebra::{Matrix3x2, Matrix3x4, Matrix2x4, Complex};
    /// let mat_a = Matrix3x2::new(
    ///     Complex::new(1.0, 2.0), Complex::new(3.0, 4.0),
    ///     Complex::new(5.0, 6.0), Complex::new(7.0, 8.0),
    ///     Complex::new(9.0, 10.0), Complex::new(11.0, 12.0)
    /// );
    /// let mat_b = Matrix3x4::new(
    ///     Complex::new(1.0, 0.0), Complex::new(0.0, 1.0), Complex::new(1.0, 1.0), Complex::new(0.0, 0.0),
    ///     Complex::new(0.0, 1.0), Complex::new(1.0, 0.0), Complex::new(0.0, 0.0), Complex::new(1.0, 1.0),
    ///     Complex::new(1.0, 1.0), Complex::new(0.0, 0.0), Complex::new(1.0, 0.0), Complex::new(0.0, 1.0)
    /// );
    /// let mut result1 = Matrix2x4::zeros();
    /// let mut result2 = Matrix2x4::zeros();
    ///
    /// // Both produce the same result, but gemm_ad is more efficient
    /// result1.gemm_ad(Complex::new(1.0, 0.0), &mat_a, &mat_b, Complex::new(0.0, 0.0));
    /// result2.gemm(Complex::new(1.0, 0.0), &mat_a.adjoint(), &mat_b, Complex::new(0.0, 0.0));
    /// assert_eq!(result1, result2);
    /// ```
    ///
    /// # See Also
    ///
    /// - [`gemm`](Self::gemm) - Standard matrix-matrix multiplication
    /// - [`gemm_tr`](Self::gemm_tr) - Matrix multiplication with transpose (no conjugation)
    /// - [`gemv_ad`](Self::gemv_ad) - Matrix-vector multiplication with adjoint
    /// - [`gerc`](Self::gerc) - Rank-1 update with conjugate
    #[inline]
    pub fn gemm_ad<R2: Dim, C2: Dim, R3: Dim, C3: Dim, SB, SC>(
        &mut self,
        alpha: T,
        a: &Matrix<T, R2, C2, SB>,
        b: &Matrix<T, R3, C3, SC>,
        beta: T,
    ) where
        T: SimdComplexField,
        SB: Storage<T, R2, C2>,
        SC: Storage<T, R3, C3>,
        ShapeConstraint: SameNumberOfRows<R1, C2>
            + SameNumberOfColumns<C1, C3>
            + AreMultipliable<C2, R2, R3, C3>,
    {
        let (nrows1, ncols1) = self.shape();
        let (nrows2, ncols2) = a.shape();
        let (nrows3, ncols3) = b.shape();

        assert_eq!(
            nrows2, nrows3,
            "gemm: dimensions mismatch for multiplication."
        );
        assert_eq!(
            (nrows1, ncols1),
            (ncols2, ncols3),
            "gemm: dimensions mismatch for addition."
        );

        for j1 in 0..ncols1 {
            // TODO: avoid bound checks.
            self.column_mut(j1)
                .gemv_ad(alpha.clone(), a, &b.column(j1), beta.clone());
        }
    }
}

impl<T, R1: Dim, C1: Dim, S: StorageMut<T, R1, C1>> Matrix<T, R1, C1, S>
where
    T: Scalar + Zero + ClosedAddAssign + ClosedMulAssign,
{
    #[inline(always)]
    fn xxgerx<D2: Dim, D3: Dim, SB, SC>(
        &mut self,
        alpha: T,
        x: &Vector<T, D2, SB>,
        y: &Vector<T, D3, SC>,
        beta: T,
        conjugate: impl Fn(T) -> T,
    ) where
        T: One,
        SB: Storage<T, D2>,
        SC: Storage<T, D3>,
        ShapeConstraint: DimEq<R1, D2> + DimEq<C1, D3>,
    {
        let dim1 = self.nrows();
        let dim2 = x.nrows();
        let dim3 = y.nrows();

        assert!(
            self.is_square(),
            "Symmetric ger: the input matrix must be square."
        );
        assert!(dim1 == dim2 && dim1 == dim3, "ger: dimensions mismatch.");

        for j in 0..dim1 {
            let val = unsafe { conjugate(y.vget_unchecked(j).clone()) };
            let subdim = Dyn(dim1 - j);
            // TODO: avoid bound checks.
            self.generic_view_mut((j, j), (subdim, Const::<1>)).axpy(
                alpha.clone() * val,
                &x.rows_range(j..),
                beta.clone(),
            );
        }
    }

    /// Computes `self = alpha * x * y.transpose() + beta * self`, where `self` is a **symmetric**
    /// matrix.
    ///
    /// If `beta` is zero, `self` is never read. The result is symmetric. Only the lower-triangular
    /// (including the diagonal) part of `self` is read/written.
    ///
    /// # Example
    /// ```
    /// # use nalgebra::{Matrix2, Vector2};
    /// let mut mat = Matrix2::identity();
    /// let vec1 = Vector2::new(1.0, 2.0);
    /// let vec2 = Vector2::new(0.1, 0.2);
    /// let expected = vec1 * vec2.transpose() * 10.0 + mat * 5.0;
    /// mat.m12 = 99999.99999; // This component is on the upper-triangular part and will not be read/written.
    ///
    /// mat.ger_symm(10.0, &vec1, &vec2, 5.0);
    /// assert_eq!(mat.lower_triangle(), expected.lower_triangle());
    /// assert_eq!(mat.m12, 99999.99999); // This was untouched.
    /// ```
    #[inline]
    #[deprecated(note = "This is renamed `syger` to match the original BLAS terminology.")]
    pub fn ger_symm<D2: Dim, D3: Dim, SB, SC>(
        &mut self,
        alpha: T,
        x: &Vector<T, D2, SB>,
        y: &Vector<T, D3, SC>,
        beta: T,
    ) where
        T: One,
        SB: Storage<T, D2>,
        SC: Storage<T, D3>,
        ShapeConstraint: DimEq<R1, D2> + DimEq<C1, D3>,
    {
        self.syger(alpha, x, y, beta)
    }

    /// Computes `self = alpha * x * y.transpose() + beta * self` for a **symmetric** matrix (BLAS syr operation).
    ///
    /// This is an optimized rank-1 update for symmetric matrices. Because the result is symmetric,
    /// only the lower triangular part (including diagonal) is computed and stored, making this
    /// about twice as fast as the general [`ger`](Self::ger) operation.
    ///
    /// # Mathematical Definition
    ///
    /// `self = alpha * x * y.transpose() + beta * self`
    ///
    /// Where:
    /// - `alpha` is a scalar multiplier for the outer product
    /// - `x` and `y` are vectors of length N (typically the same vector for symmetric updates)
    /// - `beta` is a scalar multiplier for the current value of `self`
    /// - `self` must be an N×N symmetric matrix
    /// - Only the **lower triangle** of the result is computed
    ///
    /// # Important Notes
    ///
    /// - **Only the lower-triangular part** (including diagonal) is read/written
    /// - The upper-triangular part is never touched and can contain garbage
    /// - The result will be symmetric even if only lower triangle is stored
    /// - For **Hermitian** complex matrices, use [`hegerc`](Self::hegerc) instead
    ///
    /// # Performance Note
    ///
    /// If `beta` is zero, `self` is never read from, which can be more efficient.
    ///
    /// # Panics
    ///
    /// Panics if `self` is not square, or if dimensions are incompatible.
    ///
    /// # Examples
    ///
    /// ## Basic symmetric rank-1 update
    /// ```
    /// # use nalgebra::{Matrix2, Vector2};
    /// let mut mat = Matrix2::identity();
    /// let vec1 = Vector2::new(1.0, 2.0);
    /// let vec2 = Vector2::new(0.1, 0.2);
    /// let expected = vec1 * vec2.transpose() * 10.0 + mat * 5.0;
    /// mat.m12 = 99999.99999; // This component is on the upper-triangular part and will not be read/written.
    ///
    /// mat.syger(10.0, &vec1, &vec2, 5.0);
    /// assert_eq!(mat.lower_triangle(), expected.lower_triangle());
    /// assert_eq!(mat.m12, 99999.99999); // This was untouched.
    /// ```
    ///
    /// ## Symmetric rank-1 update with same vector
    /// ```
    /// # use nalgebra::{Matrix3, Vector3};
    /// let mut mat = Matrix3::zeros();
    /// let vec = Vector3::new(1.0, 2.0, 3.0);
    ///
    /// // Common pattern: A = A + v*v^T (symmetric)
    /// mat.syger(1.0, &vec, &vec, 0.0);
    /// // Only lower triangle is computed
    /// assert_eq!(mat[(0,0)], 1.0);
    /// assert_eq!(mat[(1,0)], 2.0);
    /// assert_eq!(mat[(1,1)], 4.0);
    /// assert_eq!(mat[(2,0)], 3.0);
    /// assert_eq!(mat[(2,1)], 6.0);
    /// assert_eq!(mat[(2,2)], 9.0);
    /// ```
    ///
    /// ## Building covariance matrices
    /// ```
    /// # use nalgebra::{Matrix2, Vector2};
    /// let mut covariance = Matrix2::zeros();
    /// let sample1 = Vector2::new(1.0, 2.0);
    /// let sample2 = Vector2::new(3.0, 1.0);
    ///
    /// // Accumulate outer products: Cov = Σ(x_i * x_i^T)
    /// covariance.syger(1.0, &sample1, &sample1, 0.0);
    /// covariance.syger(1.0, &sample2, &sample2, 1.0);
    /// // Result stored in lower triangle
    /// assert_eq!(covariance[(0,0)], 10.0);
    /// assert_eq!(covariance[(1,0)], 5.0);
    /// assert_eq!(covariance[(1,1)], 5.0);
    /// ```
    ///
    /// ## Upper triangle remains unchanged
    /// ```
    /// # use nalgebra::{Matrix2, Vector2};
    /// let mut mat = Matrix2::new(1.0, 999.0,  // Upper triangle has garbage
    ///                            0.0, 1.0);
    /// let v = Vector2::new(2.0, 3.0);
    ///
    /// mat.syger(1.0, &v, &v, 1.0);
    /// // Lower triangle updated, upper triangle unchanged
    /// assert_eq!(mat[(0,0)], 5.0);   // 1 + 2*2
    /// assert_eq!(mat[(1,0)], 6.0);   // 0 + 2*3
    /// assert_eq!(mat[(1,1)], 10.0);  // 1 + 3*3
    /// assert_eq!(mat[(0,1)], 999.0); // Unchanged!
    /// ```
    ///
    /// # See Also
    ///
    /// - [`ger`](Self::ger) - General rank-1 update
    /// - [`hegerc`](Self::hegerc) - Hermitian rank-1 update (complex matrices)
    /// - [`sygemv`](Self::sygemv) - Symmetric matrix-vector multiplication
    #[inline]
    pub fn syger<D2: Dim, D3: Dim, SB, SC>(
        &mut self,
        alpha: T,
        x: &Vector<T, D2, SB>,
        y: &Vector<T, D3, SC>,
        beta: T,
    ) where
        T: One,
        SB: Storage<T, D2>,
        SC: Storage<T, D3>,
        ShapeConstraint: DimEq<R1, D2> + DimEq<C1, D3>,
    {
        self.xxgerx(alpha, x, y, beta, |e| e)
    }

    /// Computes `self = alpha * x * y.adjoint() + beta * self` for a **Hermitian** matrix (BLAS her operation).
    ///
    /// This is an optimized rank-1 update for Hermitian (complex symmetric) matrices. A Hermitian
    /// matrix satisfies `A = A*` (conjugate transpose), so only the lower triangular part (including
    /// diagonal) needs to be computed and stored. The diagonal will always be real-valued after the update.
    ///
    /// # Mathematical Definition
    ///
    /// `self = alpha * x * y.adjoint() + beta * self`
    ///
    /// Where:
    /// - `alpha` is a complex scalar multiplier for the outer product
    /// - `x` and `y` are complex vectors of length N (often the same vector)
    /// - `y.adjoint()` is the conjugate transpose of `y`
    /// - `beta` is a complex scalar multiplier for the current value of `self`
    /// - `self` must be an N×N Hermitian matrix
    /// - Only the **lower triangle** of the result is computed
    ///
    /// # Important Notes
    ///
    /// - **Only the lower-triangular part** (including diagonal) is read/written
    /// - The diagonal will be real-valued (imaginary parts are ignored/zeroed)
    /// - The upper-triangular part is never touched and can contain garbage
    /// - For **real symmetric** matrices, use [`syger`](Self::syger) instead
    ///
    /// # Performance Note
    ///
    /// If `beta` is zero, `self` is never read from, which can be more efficient.
    ///
    /// # Panics
    ///
    /// Panics if `self` is not square, or if dimensions are incompatible.
    ///
    /// # Examples
    ///
    /// ## Basic Hermitian rank-1 update
    /// ```
    /// # use nalgebra::{Matrix2, Vector2, Complex};
    /// let mut mat = Matrix2::identity();
    /// let vec1 = Vector2::new(Complex::new(1.0, 3.0), Complex::new(2.0, 4.0));
    /// let vec2 = Vector2::new(Complex::new(0.2, 0.4), Complex::new(0.1, 0.3));
    /// let expected = vec1 * vec2.adjoint() * Complex::new(10.0, 20.0) + mat * Complex::new(5.0, 15.0);
    /// mat.m12 = Complex::new(99999.99999, 88888.88888); // This component is on the upper-triangular part and will not be read/written.
    ///
    /// mat.hegerc(Complex::new(10.0, 20.0), &vec1, &vec2, Complex::new(5.0, 15.0));
    /// assert_eq!(mat.lower_triangle(), expected.lower_triangle());
    /// assert_eq!(mat.m12, Complex::new(99999.99999, 88888.88888)); // This was untouched.
    /// ```
    ///
    /// ## Hermitian rank-1 update with same vector
    /// ```
    /// # use nalgebra::{Matrix2, Vector2, Complex};
    /// let mut mat = Matrix2::zeros();
    /// let v = Vector2::new(Complex::new(1.0, 2.0), Complex::new(3.0, 4.0));
    ///
    /// // Common pattern: A = A + v*v* (Hermitian)
    /// mat.hegerc(Complex::new(1.0, 0.0), &v, &v, Complex::new(0.0, 0.0));
    /// // Diagonal is real, lower triangle is complex
    /// assert_eq!(mat[(0,0)], Complex::new(5.0, 0.0));    // |1+2i|² = 5
    /// assert_eq!(mat[(1,0)], Complex::new(11.0, 2.0));   // (3+4i)·conj(1+2i)
    /// assert_eq!(mat[(1,1)], Complex::new(25.0, 0.0));   // |3+4i|² = 25
    /// ```
    ///
    /// ## Building Hermitian matrices (e.g., density matrices)
    /// ```
    /// # use nalgebra::{Matrix2, Vector2, Complex};
    /// let mut density = Matrix2::zeros();
    /// let state1 = Vector2::new(Complex::new(0.6, 0.0), Complex::new(0.8, 0.0));
    /// let state2 = Vector2::new(Complex::new(0.0, 0.6), Complex::new(0.0, -0.8));
    ///
    /// // Density matrix: ρ = 0.5*|ψ₁⟩⟨ψ₁| + 0.5*|ψ₂⟩⟨ψ₂|
    /// density.hegerc(Complex::new(0.5, 0.0), &state1, &state1, Complex::new(0.0, 0.0));
    /// density.hegerc(Complex::new(0.5, 0.0), &state2, &state2, Complex::new(1.0, 0.0));
    /// // Result is Hermitian with real diagonal
    /// assert_eq!(density[(0,0)].im, 0.0);  // Diagonal is real
    /// assert_eq!(density[(1,1)].im, 0.0);  // Diagonal is real
    /// ```
    ///
    /// ## Diagonal remains real
    /// ```
    /// # use nalgebra::{Matrix2, Vector2, Complex};
    /// let mut mat = Matrix2::from_diagonal(&Vector2::new(
    ///     Complex::new(1.0, 0.0),
    ///     Complex::new(2.0, 0.0)
    /// ));
    /// let v = Vector2::new(Complex::new(1.0, 1.0), Complex::new(1.0, -1.0));
    ///
    /// // After Hermitian update, diagonal stays real
    /// mat.hegerc(Complex::new(1.0, 0.0), &v, &v, Complex::new(1.0, 0.0));
    /// assert_eq!(mat[(0,0)].im, 0.0);  // Diagonal is always real
    /// assert_eq!(mat[(1,1)].im, 0.0);  // Diagonal is always real
    /// ```
    ///
    /// # See Also
    ///
    /// - [`syger`](Self::syger) - Symmetric rank-1 update (real matrices)
    /// - [`ger`](Self::ger) - General rank-1 update
    /// - [`gerc`](Self::gerc) - General rank-1 update with conjugate
    /// - [`hegemv`](Self::hegemv) - Hermitian matrix-vector multiplication
    #[inline]
    pub fn hegerc<D2: Dim, D3: Dim, SB, SC>(
        &mut self,
        alpha: T,
        x: &Vector<T, D2, SB>,
        y: &Vector<T, D3, SC>,
        beta: T,
    ) where
        T: SimdComplexField,
        SB: Storage<T, D2>,
        SC: Storage<T, D3>,
        ShapeConstraint: DimEq<R1, D2> + DimEq<C1, D3>,
    {
        self.xxgerx(alpha, x, y, beta, SimdComplexField::simd_conjugate)
    }
}

impl<T, D1: Dim, S: StorageMut<T, D1, D1>> SquareMatrix<T, D1, S>
where
    T: Scalar + Zero + One + ClosedAddAssign + ClosedMulAssign,
{
    /// Computes the quadratic form `self = alpha * lhs * mid * lhs.transpose() + beta * self` with workspace.
    ///
    /// This function computes a quadratic form, which appears frequently in statistics,
    /// optimization, and numerical methods. The operation is: multiply `lhs` by `mid`, then
    /// by the transpose of `lhs`, creating a symmetric result matrix.
    ///
    /// # Mathematical Definition
    ///
    /// `self = alpha * lhs * mid * lhs.transpose() + beta * self`
    ///
    /// Where:
    /// - `alpha` is a scalar multiplier
    /// - `lhs` is an M×K matrix (the "left-hand side")
    /// - `mid` is a K×K matrix (typically symmetric/positive definite)
    /// - `lhs.transpose()` is K×M
    /// - `beta` is a scalar multiplier for the current value of `self`
    /// - `self` must be an M×M matrix
    /// - The result is always an M×M symmetric matrix
    ///
    /// # Workspace Parameter
    ///
    /// This version uses a provided workspace vector to avoid allocations for intermediate
    /// results. The workspace `work` must be a vector of length M. Its initial values don't
    /// matter as they will be overwritten.
    ///
    /// # Performance Notes
    ///
    /// - If `beta` is zero, `self` is never read from
    /// - For dynamic matrices, reusing the workspace across multiple calls avoids repeated allocations
    /// - For static matrices, use [`quadform_tr`](Self::quadform_tr) for convenience
    ///
    /// # Panics
    ///
    /// Panics if dimensions are incompatible.
    ///
    /// # Examples
    ///
    /// ## Basic quadratic form with workspace
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{DMatrix, DVector};
    /// // Note that all those would also work with statically-sized matrices.
    /// // We use DMatrix/DVector since that's the only case where pre-allocating the
    /// // workspace is actually useful (assuming the same workspace is re-used for
    /// // several computations) because it avoids repeated dynamic allocations.
    /// let mut mat = DMatrix::identity(2, 2);
    /// let lhs = DMatrix::from_row_slice(2, 3, &[1.0, 2.0, 3.0,
    ///                                           4.0, 5.0, 6.0]);
    /// let mid = DMatrix::from_row_slice(3, 3, &[0.1, 0.2, 0.3,
    ///                                           0.5, 0.6, 0.7,
    ///                                           0.9, 1.0, 1.1]);
    /// // The random shows that values on the workspace do not
    /// // matter as they will be overwritten.
    /// let mut workspace = DVector::new_random(2);
    /// let expected = &lhs * &mid * lhs.transpose() * 10.0 + &mat * 5.0;
    ///
    /// mat.quadform_tr_with_workspace(&mut workspace, 10.0, &lhs, &mid, 5.0);
    /// assert_relative_eq!(mat, expected);
    /// ```
    ///
    /// ## Reusing workspace for multiple computations
    /// ```
    /// # use nalgebra::{DMatrix, DVector};
    /// let mut result = DMatrix::zeros(2, 2);
    /// let lhs1 = DMatrix::from_row_slice(2, 3, &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
    /// let lhs2 = DMatrix::from_row_slice(2, 3, &[0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
    /// let mid = DMatrix::identity(3, 3);
    /// let mut workspace = DVector::zeros(2);
    ///
    /// // Reuse workspace to avoid allocations
    /// result.quadform_tr_with_workspace(&mut workspace, 1.0, &lhs1, &mid, 0.0);
    /// result.quadform_tr_with_workspace(&mut workspace, 1.0, &lhs2, &mid, 1.0);
    /// ```
    ///
    /// ## Computing covariance: Σ = (1/n) * X * X^T
    /// ```
    /// # use nalgebra::{DMatrix, DVector};
    /// let data = DMatrix::from_row_slice(2, 3, &[1.0, 2.0, 3.0,
    ///                                            4.0, 5.0, 6.0]);
    /// let mut covariance = DMatrix::zeros(2, 2);
    /// let identity = DMatrix::identity(3, 3);
    /// let mut work = DVector::zeros(2);
    ///
    /// // Covariance: (1/n) * data * I * data^T
    /// let n = 3.0;
    /// covariance.quadform_tr_with_workspace(&mut work, 1.0/n, &data, &identity, 0.0);
    /// ```
    ///
    /// # See Also
    ///
    /// - [`quadform_tr`](Self::quadform_tr) - Same operation without explicit workspace
    /// - [`quadform_with_workspace`](Self::quadform_with_workspace) - Quadratic form with rhs.transpose()
    /// - [`gemm`](Self::gemm) - General matrix-matrix multiplication
    pub fn quadform_tr_with_workspace<D2, S2, R3, C3, S3, D4, S4>(
        &mut self,
        work: &mut Vector<T, D2, S2>,
        alpha: T,
        lhs: &Matrix<T, R3, C3, S3>,
        mid: &SquareMatrix<T, D4, S4>,
        beta: T,
    ) where
        D2: Dim,
        R3: Dim,
        C3: Dim,
        D4: Dim,
        S2: StorageMut<T, D2>,
        S3: Storage<T, R3, C3>,
        S4: Storage<T, D4, D4>,
        ShapeConstraint: DimEq<D1, D2> + DimEq<D1, R3> + DimEq<D2, R3> + DimEq<C3, D4>,
    {
        work.gemv(T::one(), lhs, &mid.column(0), T::zero());
        self.ger(alpha.clone(), work, &lhs.column(0), beta);

        for j in 1..mid.ncols() {
            work.gemv(T::one(), lhs, &mid.column(j), T::zero());
            self.ger(alpha.clone(), work, &lhs.column(j), T::one());
        }
    }

    /// Computes the quadratic form `self = alpha * lhs * mid * lhs.transpose() + beta * self`.
    ///
    /// This is a convenience wrapper around [`quadform_tr_with_workspace`](Self::quadform_tr_with_workspace)
    /// that automatically allocates a workspace vector. For static-sized matrices, the workspace
    /// is allocated on the stack. For dynamic matrices, consider using the workspace version
    /// if you need to call this function multiple times.
    ///
    /// # Mathematical Definition
    ///
    /// `self = alpha * lhs * mid * lhs.transpose() + beta * self`
    ///
    /// Where:
    /// - `alpha` is a scalar multiplier
    /// - `lhs` is an M×K matrix
    /// - `mid` is a K×K matrix (typically symmetric)
    /// - `beta` is a scalar multiplier for the current value of `self`
    /// - `self` must be an M×M matrix
    /// - The result is always symmetric
    ///
    /// # Workspace Allocation
    ///
    /// This method allocates a workspace vector of dimension M:
    /// - For static sizes (e.g., `Matrix2`), allocation is on the stack (very fast)
    /// - For dynamic sizes (e.g., `DMatrix`), allocation is on the heap
    /// - To avoid repeated allocations, use [`quadform_tr_with_workspace`](Self::quadform_tr_with_workspace)
    ///
    /// # Performance Note
    ///
    /// If `beta` is zero, `self` is never read from, which can be more efficient.
    ///
    /// # Panics
    ///
    /// Panics if dimensions are incompatible.
    ///
    /// # Examples
    ///
    /// ## Basic quadratic form
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{Matrix2, Matrix3, Matrix2x3};
    /// let mut mat = Matrix2::identity();
    /// let lhs = Matrix2x3::new(1.0, 2.0, 3.0,
    ///                          4.0, 5.0, 6.0);
    /// let mid = Matrix3::new(0.1, 0.2, 0.3,
    ///                        0.5, 0.6, 0.7,
    ///                        0.9, 1.0, 1.1);
    /// let expected = lhs * mid * lhs.transpose() * 10.0 + mat * 5.0;
    ///
    /// mat.quadform_tr(10.0, &lhs, &mid, 5.0);
    /// assert_relative_eq!(mat, expected);
    /// ```
    ///
    /// ## Computing X * Σ * X^T (covariance transformation)
    /// ```
    /// # use nalgebra::{Matrix2, Matrix3, Matrix2x3};
    /// let transformation = Matrix2x3::new(1.0, 0.0, 0.0,
    ///                                     0.0, 1.0, 1.0);
    /// let covariance = Matrix3::new(1.0, 0.0, 0.0,
    ///                               0.0, 2.0, 0.5,
    ///                               0.0, 0.5, 3.0);
    /// let mut result = Matrix2::zeros();
    ///
    /// // Transform covariance: result = transformation * covariance * transformation^T
    /// result.quadform_tr(1.0, &transformation, &covariance, 0.0);
    /// ```
    ///
    /// ## Weighted outer product
    /// ```
    /// # use nalgebra::{Matrix2, Matrix3, Matrix2x3};
    /// let vectors = Matrix2x3::new(1.0, 2.0, 3.0,
    ///                              4.0, 5.0, 6.0);
    /// let weights = Matrix3::from_diagonal(&nalgebra::Vector3::new(0.5, 1.0, 0.5));
    /// let mut gram = Matrix2::zeros();
    ///
    /// // Weighted Gram matrix
    /// gram.quadform_tr(1.0, &vectors, &weights, 0.0);
    /// ```
    ///
    /// ## Accumulating quadratic forms
    /// ```
    /// # use nalgebra::{Matrix2, Matrix2x3, Matrix3};
    /// let mut sum = Matrix2::zeros();
    /// let data1 = Matrix2x3::new(1.0, 0.0, 0.0, 0.0, 1.0, 0.0);
    /// let data2 = Matrix2x3::new(0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
    /// let mid = Matrix3::identity();
    ///
    /// sum.quadform_tr(1.0, &data1, &mid, 0.0);
    /// sum.quadform_tr(1.0, &data2, &mid, 1.0);  // Accumulate
    /// ```
    ///
    /// # See Also
    ///
    /// - [`quadform_tr_with_workspace`](Self::quadform_tr_with_workspace) - Same with explicit workspace
    /// - [`quadform`](Self::quadform) - Quadratic form: rhs.transpose() * mid * rhs
    /// - [`gemm`](Self::gemm) - General matrix-matrix multiplication
    pub fn quadform_tr<R3, C3, S3, D4, S4>(
        &mut self,
        alpha: T,
        lhs: &Matrix<T, R3, C3, S3>,
        mid: &SquareMatrix<T, D4, S4>,
        beta: T,
    ) where
        R3: Dim,
        C3: Dim,
        D4: Dim,
        S3: Storage<T, R3, C3>,
        S4: Storage<T, D4, D4>,
        ShapeConstraint: DimEq<D1, D1> + DimEq<D1, R3> + DimEq<C3, D4>,
        DefaultAllocator: Allocator<D1>,
    {
        // TODO: would it be useful to avoid the zero-initialization of the workspace data?
        let mut work = Matrix::zeros_generic(self.shape_generic().0, Const::<1>);
        self.quadform_tr_with_workspace(&mut work, alpha, lhs, mid, beta)
    }

    /// Computes the quadratic form `self = alpha * rhs.transpose() * mid * rhs + beta * self` with workspace.
    ///
    /// This is the "transposed" version of [`quadform_tr_with_workspace`](Self::quadform_tr_with_workspace),
    /// computing `rhs.transpose() * mid * rhs` instead of `lhs * mid * lhs.transpose()`.
    /// The result is always symmetric.
    ///
    /// # Mathematical Definition
    ///
    /// `self = alpha * rhs.transpose() * mid * rhs + beta * self`
    ///
    /// Where:
    /// - `alpha` is a scalar multiplier
    /// - `rhs` is a K×N matrix (the "right-hand side")
    /// - `mid` is a K×K matrix (typically symmetric/positive definite)
    /// - `rhs.transpose()` is N×K
    /// - `beta` is a scalar multiplier for the current value of `self`
    /// - `self` must be an N×N matrix
    /// - The result is always an N×N symmetric matrix
    ///
    /// # Workspace Parameter
    ///
    /// This version uses a provided workspace vector to avoid allocations. The workspace `work`
    /// must be a vector of length K. Its initial values don't matter as they will be overwritten.
    ///
    /// # Performance Notes
    ///
    /// - If `beta` is zero, `self` is never read from
    /// - For dynamic matrices, reusing the workspace avoids repeated allocations
    /// - For static matrices, use [`quadform`](Self::quadform) for convenience
    ///
    /// # Panics
    ///
    /// Panics if dimensions are incompatible.
    ///
    /// # Examples
    ///
    /// ## Basic quadratic form with workspace
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{DMatrix, DVector};
    /// // Note that all those would also work with statically-sized matrices.
    /// // We use DMatrix/DVector since that's the only case where pre-allocating the
    /// // workspace is actually useful (assuming the same workspace is re-used for
    /// // several computations) because it avoids repeated dynamic allocations.
    /// let mut mat = DMatrix::identity(2, 2);
    /// let rhs = DMatrix::from_row_slice(3, 2, &[1.0, 2.0,
    ///                                           3.0, 4.0,
    ///                                           5.0, 6.0]);
    /// let mid = DMatrix::from_row_slice(3, 3, &[0.1, 0.2, 0.3,
    ///                                           0.5, 0.6, 0.7,
    ///                                           0.9, 1.0, 1.1]);
    /// // The random shows that values on the workspace do not
    /// // matter as they will be overwritten.
    /// let mut workspace = DVector::new_random(3);
    /// let expected = rhs.transpose() * &mid * &rhs * 10.0 + &mat * 5.0;
    ///
    /// mat.quadform_with_workspace(&mut workspace, 10.0, &mid, &rhs, 5.0);
    /// assert_relative_eq!(mat, expected);
    /// ```
    ///
    /// ## Computing X^T * Σ * X (sandwich form)
    /// ```
    /// # use nalgebra::{DMatrix, DVector};
    /// let design_matrix = DMatrix::from_row_slice(4, 2, &[
    ///     1.0, 1.0,
    ///     1.0, 2.0,
    ///     1.0, 3.0,
    ///     1.0, 4.0,
    /// ]);
    /// let weights = DMatrix::from_diagonal(&DVector::from_vec(vec![1.0, 1.0, 1.0, 1.0]));
    /// let mut xtx = DMatrix::zeros(2, 2);
    /// let mut work = DVector::zeros(4);
    ///
    /// // Compute X^T * W * X (weighted least squares normal equations)
    /// xtx.quadform_with_workspace(&mut work, 1.0, &weights, &design_matrix, 0.0);
    /// ```
    ///
    /// ## Reusing workspace for efficiency
    /// ```
    /// # use nalgebra::{DMatrix, DVector};
    /// let mut result = DMatrix::zeros(2, 2);
    /// let data1 = DMatrix::identity(3, 2);
    /// let data2 = DMatrix::from_row_slice(3, 2, &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
    /// let mid = DMatrix::identity(3, 3);
    /// let mut work = DVector::zeros(3);
    ///
    /// // Efficiently compute sum of quadratic forms
    /// result.quadform_with_workspace(&mut work, 1.0, &mid, &data1, 0.0);
    /// result.quadform_with_workspace(&mut work, 1.0, &mid, &data2, 1.0);
    /// ```
    ///
    /// # See Also
    ///
    /// - [`quadform`](Self::quadform) - Same operation without explicit workspace
    /// - [`quadform_tr_with_workspace`](Self::quadform_tr_with_workspace) - Quadratic form with lhs transpose
    /// - [`gemm`](Self::gemm) - General matrix-matrix multiplication
    pub fn quadform_with_workspace<D2, S2, D3, S3, R4, C4, S4>(
        &mut self,
        work: &mut Vector<T, D2, S2>,
        alpha: T,
        mid: &SquareMatrix<T, D3, S3>,
        rhs: &Matrix<T, R4, C4, S4>,
        beta: T,
    ) where
        D2: Dim,
        D3: Dim,
        R4: Dim,
        C4: Dim,
        S2: StorageMut<T, D2>,
        S3: Storage<T, D3, D3>,
        S4: Storage<T, R4, C4>,
        ShapeConstraint:
            DimEq<D3, R4> + DimEq<D1, C4> + DimEq<D2, D3> + AreMultipliable<C4, R4, D2, U1>,
    {
        work.gemv(T::one(), mid, &rhs.column(0), T::zero());
        self.column_mut(0)
            .gemv_tr(alpha.clone(), rhs, work, beta.clone());

        for j in 1..rhs.ncols() {
            work.gemv(T::one(), mid, &rhs.column(j), T::zero());
            self.column_mut(j)
                .gemv_tr(alpha.clone(), rhs, work, beta.clone());
        }
    }

    /// Computes the quadratic form `self = alpha * rhs.transpose() * mid * rhs + beta * self`.
    ///
    /// This is a convenience wrapper around [`quadform_with_workspace`](Self::quadform_with_workspace)
    /// that automatically allocates a workspace vector. For static-sized matrices, the workspace
    /// is allocated on the stack. For dynamic matrices, consider using the workspace version
    /// if you need to call this function multiple times.
    ///
    /// # Mathematical Definition
    ///
    /// `self = alpha * rhs.transpose() * mid * rhs + beta * self`
    ///
    /// Where:
    /// - `alpha` is a scalar multiplier
    /// - `rhs` is a K×N matrix
    /// - `mid` is a K×K matrix (typically symmetric)
    /// - `beta` is a scalar multiplier for the current value of `self`
    /// - `self` must be an N×N matrix
    /// - The result is always symmetric
    ///
    /// # Workspace Allocation
    ///
    /// This method allocates a workspace vector of dimension K:
    /// - For static sizes (e.g., `Matrix2`), allocation is on the stack (very fast)
    /// - For dynamic sizes (e.g., `DMatrix`), allocation is on the heap
    /// - To avoid repeated allocations, use [`quadform_with_workspace`](Self::quadform_with_workspace)
    ///
    /// # Performance Note
    ///
    /// If `beta` is zero, `self` is never read from, which can be more efficient.
    ///
    /// # Panics
    ///
    /// Panics if dimensions are incompatible.
    ///
    /// # Examples
    ///
    /// ## Basic quadratic form
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{Matrix2, Matrix3x2, Matrix3};
    /// let mut mat = Matrix2::identity();
    /// let rhs = Matrix3x2::new(1.0, 2.0,
    ///                          3.0, 4.0,
    ///                          5.0, 6.0);
    /// let mid = Matrix3::new(0.1, 0.2, 0.3,
    ///                        0.5, 0.6, 0.7,
    ///                        0.9, 1.0, 1.1);
    /// let expected = rhs.transpose() * mid * rhs * 10.0 + mat * 5.0;
    ///
    /// mat.quadform(10.0, &mid, &rhs, 5.0);
    /// assert_relative_eq!(mat, expected);
    /// ```
    ///
    /// ## Weighted least squares normal equations: X^T W X
    /// ```
    /// # use nalgebra::{Matrix4x2, Matrix4, Matrix2};
    /// // Design matrix for linear regression
    /// let x = Matrix4x2::new(
    ///     1.0, 1.0,   // Intercept and feature for point 1
    ///     1.0, 2.0,   // Intercept and feature for point 2
    ///     1.0, 3.0,   // Intercept and feature for point 3
    ///     1.0, 4.0,   // Intercept and feature for point 4
    /// );
    /// let weights = Matrix4::from_diagonal(&nalgebra::Vector4::new(1.0, 1.0, 1.0, 1.0));
    /// let mut normal_eq = Matrix2::zeros();
    ///
    /// // Compute X^T * W * X
    /// normal_eq.quadform(1.0, &weights, &x, 0.0);
    /// ```
    ///
    /// ## Information matrix in statistics
    /// ```
    /// # use nalgebra::{Matrix3x2, Matrix3, Matrix2};
    /// let jacobian = Matrix3x2::new(
    ///     1.0, 0.5,
    ///     0.8, 1.2,
    ///     0.6, 0.9,
    /// );
    /// let precision = Matrix3::identity();  // Inverse covariance
    /// let mut fisher_info = Matrix2::zeros();
    ///
    /// // Fisher information: J^T * Precision * J
    /// fisher_info.quadform(1.0, &precision, &jacobian, 0.0);
    /// ```
    ///
    /// ## Building Gram matrix
    /// ```
    /// # use nalgebra::{Matrix3x2, Matrix3, Matrix2};
    /// let vectors = Matrix3x2::new(
    ///     1.0, 2.0,
    ///     3.0, 4.0,
    ///     5.0, 6.0,
    /// );
    /// let identity = Matrix3::identity();
    /// let mut gram = Matrix2::zeros();
    ///
    /// // Gram matrix: V^T * V
    /// gram.quadform(1.0, &identity, &vectors, 0.0);
    /// // gram now contains the inner products of column vectors
    /// ```
    ///
    /// ## Accumulating quadratic forms
    /// ```
    /// # use nalgebra::{Matrix3x2, Matrix3, Matrix2};
    /// let mut sum = Matrix2::zeros();
    /// let data1 = Matrix3x2::new(1.0, 0.0, 0.0, 1.0, 0.0, 0.0);
    /// let data2 = Matrix3x2::new(0.0, 1.0, 1.0, 0.0, 0.0, 1.0);
    /// let mid = Matrix3::identity();
    ///
    /// sum.quadform(1.0, &mid, &data1, 0.0);
    /// sum.quadform(1.0, &mid, &data2, 1.0);  // Accumulate
    /// ```
    ///
    /// # See Also
    ///
    /// - [`quadform_with_workspace`](Self::quadform_with_workspace) - Same with explicit workspace
    /// - [`quadform_tr`](Self::quadform_tr) - Quadratic form: lhs * mid * lhs.transpose()
    /// - [`gemm`](Self::gemm) - General matrix-matrix multiplication
    pub fn quadform<D2, S2, R3, C3, S3>(
        &mut self,
        alpha: T,
        mid: &SquareMatrix<T, D2, S2>,
        rhs: &Matrix<T, R3, C3, S3>,
        beta: T,
    ) where
        D2: Dim,
        R3: Dim,
        C3: Dim,
        S2: Storage<T, D2, D2>,
        S3: Storage<T, R3, C3>,
        ShapeConstraint: DimEq<D2, R3> + DimEq<D1, C3> + AreMultipliable<C3, R3, D2, U1>,
        DefaultAllocator: Allocator<D2>,
    {
        // TODO: would it be useful to avoid the zero-initialization of the workspace data?
        let mut work = Vector::zeros_generic(mid.shape_generic().0, Const::<1>);
        self.quadform_with_workspace(&mut work, alpha, mid, rhs, beta)
    }
}
