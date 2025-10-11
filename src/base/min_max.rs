use crate::storage::RawStorage;
use crate::{ComplexField, Dim, Matrix, Scalar, SimdComplexField, SimdPartialOrd, Vector};
use num::{Signed, Zero};
use simba::simd::SimdSigned;

/// # Find the min and max components
impl<T: Scalar, R: Dim, C: Dim, S: RawStorage<T, R, C>> Matrix<T, R, C, S> {
    /// Returns the absolute value of the component with the largest absolute value.
    ///
    /// This function searches through all elements in the matrix (or vector) and finds
    /// the one whose absolute value is the largest. It then returns that absolute value.
    /// This is useful when you want to know the magnitude of the most extreme element,
    /// regardless of whether it's positive or negative.
    ///
    /// The name "amax" stands for "absolute maximum".
    ///
    /// # Examples
    ///
    /// Basic usage with a vector:
    ///
    /// ```
    /// # use nalgebra::Vector3;
    /// let v = Vector3::new(-1.0, 2.0, 3.0);
    /// assert_eq!(v.amax(), 3.0);  // 3.0 has the largest absolute value
    ///
    /// let v = Vector3::new(-1.0, -2.0, -3.0);
    /// assert_eq!(v.amax(), 3.0);  // |-3.0| = 3.0 is the largest absolute value
    /// ```
    ///
    /// Usage with a matrix:
    ///
    /// ```
    /// # use nalgebra::Matrix2x3;
    /// let m = Matrix2x3::new(
    ///     1.0, -5.0, 2.0,
    ///     3.0, -4.0, 0.0
    /// );
    /// assert_eq!(m.amax(), 5.0);  // |-5.0| = 5.0 is the largest
    /// ```
    ///
    /// Comparing positive and negative values:
    ///
    /// ```
    /// # use nalgebra::Vector4;
    /// let v = Vector4::new(10.0, -20.0, 5.0, 15.0);
    /// assert_eq!(v.amax(), 20.0);  // -20.0 has the largest absolute value
    /// ```
    ///
    /// # See Also
    ///
    /// - [`amin`](Self::amin): Find the smallest absolute value
    /// - [`max`](Self::max): Find the largest value (without taking absolute value)
    /// - [`min`](Self::min): Find the smallest value
    /// - [`iamax_full`](Self::iamax_full): Find the index of the component with largest absolute value (matrix)
    /// - [`iamax`](crate::base::Vector::iamax): Find the index of the component with largest absolute value (vector)
    #[inline]
    #[must_use]
    pub fn amax(&self) -> T
    where
        T: Zero + SimdSigned + SimdPartialOrd,
    {
        self.fold_with(
            |e| e.unwrap_or(&T::zero()).simd_abs(),
            |a, b| a.simd_max(b.simd_abs()),
        )
    }

    /// Returns the 1-norm of the complex component with the largest 1-norm.
    ///
    /// This function is specifically designed for matrices/vectors containing complex numbers.
    /// It computes the 1-norm (also called the Manhattan norm or taxicab norm) for each
    /// complex number, which is the sum of the absolute values of the real and imaginary parts:
    /// `|a + bi| = |a| + |b|`. The function then returns the largest of these norms.
    ///
    /// The name "camax" stands for "complex absolute maximum".
    ///
    /// # Examples
    ///
    /// Basic usage with complex numbers:
    ///
    /// ```
    /// # use nalgebra::{Vector3, Complex};
    /// let v = Vector3::new(
    ///     Complex::new(-3.0, -2.0),  // 1-norm: |-3| + |-2| = 5
    ///     Complex::new(1.0, 2.0),    // 1-norm: |1| + |2| = 3
    ///     Complex::new(1.0, 3.0)     // 1-norm: |1| + |3| = 4
    /// );
    /// assert_eq!(v.camax(), 5.0);  // 5 is the largest 1-norm
    /// ```
    ///
    /// All real numbers (imaginary part is zero):
    ///
    /// ```
    /// # use nalgebra::{Vector3, Complex};
    /// let v = Vector3::new(
    ///     Complex::new(2.0, 0.0),
    ///     Complex::new(-5.0, 0.0),
    ///     Complex::new(3.0, 0.0)
    /// );
    /// assert_eq!(v.camax(), 5.0);  // For real numbers, 1-norm equals absolute value
    /// ```
    ///
    /// Working with a matrix:
    ///
    /// ```
    /// # use nalgebra::{Matrix2x2, Complex};
    /// let m = Matrix2x2::new(
    ///     Complex::new(1.0, 1.0),   // 1-norm: 2
    ///     Complex::new(2.0, 3.0),   // 1-norm: 5
    ///     Complex::new(0.0, 4.0),   // 1-norm: 4
    ///     Complex::new(-1.0, -1.0)  // 1-norm: 2
    /// );
    /// assert_eq!(m.camax(), 5.0);
    /// ```
    ///
    /// # See Also
    ///
    /// - [`camin`](Self::camin): Find the smallest complex 1-norm
    /// - [`amax`](Self::amax): Find the largest absolute value (for real numbers)
    /// - [`icamax_full`](Self::icamax_full): Find the index of the component with largest complex 1-norm (matrix)
    /// - [`icamax`](crate::base::Vector::icamax): Find the index of the component with largest complex 1-norm (vector)
    #[inline]
    #[must_use]
    pub fn camax(&self) -> T::SimdRealField
    where
        T: SimdComplexField,
    {
        self.fold_with(
            |e| e.unwrap_or(&T::zero()).clone().simd_norm1(),
            |a, b| a.simd_max(b.clone().simd_norm1()),
        )
    }

    /// Returns the component with the largest value.
    ///
    /// This function finds and returns the maximum element in the matrix or vector,
    /// comparing the actual values (not their absolute values). Negative numbers are
    /// considered smaller than positive numbers in the comparison.
    ///
    /// # Examples
    ///
    /// Basic usage with floating point numbers:
    ///
    /// ```
    /// # use nalgebra::Vector3;
    /// let v = Vector3::new(-1.0, 2.0, 3.0);
    /// assert_eq!(v.max(), 3.0);
    ///
    /// let v = Vector3::new(-1.0, -2.0, -3.0);
    /// assert_eq!(v.max(), -1.0);  // -1 is larger than -2 and -3
    /// ```
    ///
    /// Usage with integers:
    ///
    /// ```
    /// # use nalgebra::Vector3;
    /// let v = Vector3::new(5u32, 2, 3);
    /// assert_eq!(v.max(), 5);
    /// ```
    ///
    /// Working with a matrix:
    ///
    /// ```
    /// # use nalgebra::Matrix2x3;
    /// let m = Matrix2x3::new(
    ///     1.0, -5.0, 2.0,
    ///     8.0, -4.0, 0.0
    /// );
    /// assert_eq!(m.max(), 8.0);
    /// ```
    ///
    /// Comparing with absolute maximum:
    ///
    /// ```
    /// # use nalgebra::Vector3;
    /// let v = Vector3::new(-10.0, 5.0, 3.0);
    /// assert_eq!(v.max(), 5.0);   // largest value
    /// assert_eq!(v.amax(), 10.0); // largest absolute value
    /// ```
    ///
    /// # See Also
    ///
    /// - [`min`](Self::min): Find the smallest value
    /// - [`amax`](Self::amax): Find the largest absolute value
    /// - [`argmax`](crate::base::Vector::argmax): Find both the index and value of the maximum (vector)
    /// - [`imax`](crate::base::Vector::imax): Find the index of the maximum value (vector)
    #[inline]
    #[must_use]
    pub fn max(&self) -> T
    where
        T: SimdPartialOrd + Zero,
    {
        self.fold_with(
            |e| e.cloned().unwrap_or_else(T::zero),
            |a, b| a.simd_max(b.clone()),
        )
    }

    /// Returns the absolute value of the component with the smallest absolute value.
    ///
    /// This function searches through all elements in the matrix (or vector) and finds
    /// the one whose absolute value is the smallest. It then returns that absolute value.
    /// This is useful when you want to find the element closest to zero in magnitude,
    /// regardless of whether it's positive or negative.
    ///
    /// The name "amin" stands for "absolute minimum".
    ///
    /// # Examples
    ///
    /// Basic usage with a vector:
    ///
    /// ```
    /// # use nalgebra::Vector3;
    /// let v = Vector3::new(-1.0, 2.0, -3.0);
    /// assert_eq!(v.amin(), 1.0);  // |-1.0| = 1.0 is the smallest absolute value
    ///
    /// let v = Vector3::new(10.0, 2.0, 30.0);
    /// assert_eq!(v.amin(), 2.0);  // 2.0 has the smallest absolute value
    /// ```
    ///
    /// Working with negative numbers:
    ///
    /// ```
    /// # use nalgebra::Vector4;
    /// let v = Vector4::new(-10.0, -20.0, -5.0, -15.0);
    /// assert_eq!(v.amin(), 5.0);  // |-5.0| = 5.0 is the smallest
    /// ```
    ///
    /// Usage with a matrix:
    ///
    /// ```
    /// # use nalgebra::Matrix2x3;
    /// let m = Matrix2x3::new(
    ///     100.0, -50.0, 25.0,
    ///     -3.0, 40.0, 0.5
    /// );
    /// assert_eq!(m.amin(), 0.5);  // 0.5 is closest to zero
    /// ```
    ///
    /// Finding elements close to zero:
    ///
    /// ```
    /// # use nalgebra::Vector3;
    /// let v = Vector3::new(1000.0, 0.01, -500.0);
    /// assert_eq!(v.amin(), 0.01);  // 0.01 is the element closest to zero
    /// ```
    ///
    /// # See Also
    ///
    /// - [`amax`](Self::amax): Find the largest absolute value
    /// - [`min`](Self::min): Find the smallest value (without taking absolute value)
    /// - [`max`](Self::max): Find the largest value
    /// - [`iamin`](crate::base::Vector::iamin): Find the index of the component with smallest absolute value (vector)
    #[inline]
    #[must_use]
    pub fn amin(&self) -> T
    where
        T: Zero + SimdPartialOrd + SimdSigned,
    {
        self.fold_with(
            |e| e.map(|e| e.simd_abs()).unwrap_or_else(T::zero),
            |a, b| a.simd_min(b.simd_abs()),
        )
    }

    /// Returns the 1-norm of the complex component with the smallest 1-norm.
    ///
    /// This function is specifically designed for matrices/vectors containing complex numbers.
    /// It computes the 1-norm (Manhattan norm) for each complex number, which is the sum of
    /// the absolute values of the real and imaginary parts: `|a + bi| = |a| + |b|`.
    /// The function then returns the smallest of these norms.
    ///
    /// The name "camin" stands for "complex absolute minimum".
    ///
    /// # Examples
    ///
    /// Basic usage with complex numbers:
    ///
    /// ```
    /// # use nalgebra::{Vector3, Complex};
    /// let v = Vector3::new(
    ///     Complex::new(-3.0, -2.0),  // 1-norm: |-3| + |-2| = 5
    ///     Complex::new(1.0, 2.0),    // 1-norm: |1| + |2| = 3
    ///     Complex::new(1.0, 3.0)     // 1-norm: |1| + |3| = 4
    /// );
    /// assert_eq!(v.camin(), 3.0);  // 3 is the smallest 1-norm
    /// ```
    ///
    /// Finding the complex number closest to zero:
    ///
    /// ```
    /// # use nalgebra::{Vector3, Complex};
    /// let v = Vector3::new(
    ///     Complex::new(10.0, 5.0),   // 1-norm: 15
    ///     Complex::new(0.5, 0.3),    // 1-norm: 0.8
    ///     Complex::new(-2.0, 1.0)    // 1-norm: 3
    /// );
    /// assert_eq!(v.camin(), 0.8);  // Complex::new(0.5, 0.3) is closest to zero
    /// ```
    ///
    /// All real numbers (imaginary part is zero):
    ///
    /// ```
    /// # use nalgebra::{Vector3, Complex};
    /// let v = Vector3::new(
    ///     Complex::new(5.0, 0.0),
    ///     Complex::new(-2.0, 0.0),
    ///     Complex::new(10.0, 0.0)
    /// );
    /// assert_eq!(v.camin(), 2.0);  // For real numbers, 1-norm equals absolute value
    /// ```
    ///
    /// # See Also
    ///
    /// - [`camax`](Self::camax): Find the largest complex 1-norm
    /// - [`amin`](Self::amin): Find the smallest absolute value (for real numbers)
    /// - [`min`](Self::min): Find the smallest value
    #[inline]
    #[must_use]
    pub fn camin(&self) -> T::SimdRealField
    where
        T: SimdComplexField,
    {
        self.fold_with(
            |e| {
                e.map(|e| e.clone().simd_norm1())
                    .unwrap_or_else(T::SimdRealField::zero)
            },
            |a, b| a.simd_min(b.clone().simd_norm1()),
        )
    }

    /// Returns the component with the smallest value.
    ///
    /// This function finds and returns the minimum element in the matrix or vector,
    /// comparing the actual values (not their absolute values). Negative numbers are
    /// considered smaller than positive numbers in the comparison.
    ///
    /// # Examples
    ///
    /// Basic usage with floating point numbers:
    ///
    /// ```
    /// # use nalgebra::Vector3;
    /// let v = Vector3::new(-1.0, 2.0, 3.0);
    /// assert_eq!(v.min(), -1.0);  // -1.0 is the smallest value
    ///
    /// let v = Vector3::new(1.0, 2.0, 3.0);
    /// assert_eq!(v.min(), 1.0);  // 1.0 is the smallest value
    /// ```
    ///
    /// Usage with integers:
    ///
    /// ```
    /// # use nalgebra::Vector3;
    /// let v = Vector3::new(5u32, 2, 3);
    /// assert_eq!(v.min(), 2);
    /// ```
    ///
    /// Working with a matrix:
    ///
    /// ```
    /// # use nalgebra::Matrix2x3;
    /// let m = Matrix2x3::new(
    ///     1.0, -5.0, 2.0,
    ///     8.0, -4.0, 0.0
    /// );
    /// assert_eq!(m.min(), -5.0);
    /// ```
    ///
    /// Comparing with absolute minimum:
    ///
    /// ```
    /// # use nalgebra::Vector3;
    /// let v = Vector3::new(-10.0, 5.0, 3.0);
    /// assert_eq!(v.min(), -10.0);  // smallest value
    /// assert_eq!(v.amin(), 3.0);   // smallest absolute value
    /// ```
    ///
    /// With all negative numbers:
    ///
    /// ```
    /// # use nalgebra::Vector4;
    /// let v = Vector4::new(-1.0, -5.0, -2.0, -10.0);
    /// assert_eq!(v.min(), -10.0);  // -10.0 is the smallest (most negative)
    /// ```
    ///
    /// # See Also
    ///
    /// - [`max`](Self::max): Find the largest value
    /// - [`amin`](Self::amin): Find the smallest absolute value
    /// - [`argmin`](crate::base::Vector::argmin): Find both the index and value of the minimum (vector)
    /// - [`imin`](crate::base::Vector::imin): Find the index of the minimum value (vector)
    #[inline]
    #[must_use]
    pub fn min(&self) -> T
    where
        T: SimdPartialOrd + Zero,
    {
        self.fold_with(
            |e| e.cloned().unwrap_or_else(T::zero),
            |a, b| a.simd_min(b.clone()),
        )
    }

    /// Computes the index of the matrix component with the largest complex 1-norm.
    ///
    /// This function finds the position (row, column) of the complex number in the matrix
    /// that has the largest 1-norm. The 1-norm of a complex number `a + bi` is `|a| + |b|`.
    /// The returned tuple contains `(row_index, column_index)`, both starting from 0.
    ///
    /// This is particularly useful when working with complex matrices and you need to identify
    /// which element has the largest magnitude in the 1-norm sense.
    ///
    /// # Panics
    ///
    /// Panics if the matrix is empty.
    ///
    /// # Examples
    ///
    /// Basic usage with complex numbers:
    ///
    /// ```
    /// # use nalgebra::{Matrix2x3, Complex};
    /// let mat = Matrix2x3::new(
    ///     Complex::new(11.0, 1.0),   // 1-norm: 12
    ///     Complex::new(-12.0, 2.0),  // 1-norm: 14
    ///     Complex::new(13.0, 3.0),   // 1-norm: 16
    ///     Complex::new(21.0, 43.0),  // 1-norm: 64 (largest!)
    ///     Complex::new(22.0, 5.0),   // 1-norm: 27
    ///     Complex::new(-23.0, 0.0)   // 1-norm: 23
    /// );
    /// assert_eq!(mat.icamax_full(), (1, 0));  // Element at row 1, column 0
    /// ```
    ///
    /// Finding the dominant element:
    ///
    /// ```
    /// # use nalgebra::{Matrix3x2, Complex};
    /// let mat = Matrix3x2::new(
    ///     Complex::new(1.0, 1.0),    // 1-norm: 2
    ///     Complex::new(2.0, 2.0),    // 1-norm: 4
    ///     Complex::new(3.0, 3.0),    // 1-norm: 6
    ///     Complex::new(4.0, 4.0),    // 1-norm: 8
    ///     Complex::new(5.0, 10.0),   // 1-norm: 15 (largest!)
    ///     Complex::new(1.0, 2.0)     // 1-norm: 3
    /// );
    /// let (row, col) = mat.icamax_full();
    /// assert_eq!((row, col), (2, 0));
    /// assert_eq!(mat[(row, col)], Complex::new(5.0, 10.0));
    /// ```
    ///
    /// With real numbers (imaginary part is zero):
    ///
    /// ```
    /// # use nalgebra::{Matrix2x2, Complex};
    /// let mat = Matrix2x2::new(
    ///     Complex::new(1.0, 0.0),
    ///     Complex::new(-5.0, 0.0),   // 1-norm: 5 (largest)
    ///     Complex::new(2.0, 0.0),
    ///     Complex::new(3.0, 0.0)
    /// );
    /// assert_eq!(mat.icamax_full(), (0, 1));
    /// ```
    ///
    /// # See Also
    ///
    /// - [`camax`](Self::camax): Get the value of the largest complex 1-norm
    /// - [`icamax`](crate::base::Vector::icamax): Find the index in a vector (returns single index)
    /// - [`iamax_full`](Self::iamax_full): Find the index of largest absolute value (for real numbers)
    #[inline]
    #[must_use]
    pub fn icamax_full(&self) -> (usize, usize)
    where
        T: ComplexField,
    {
        assert!(!self.is_empty(), "The input matrix must not be empty.");

        let mut the_max = unsafe { self.get_unchecked((0, 0)).clone().norm1() };
        let mut the_ij = (0, 0);

        for j in 0..self.ncols() {
            for i in 0..self.nrows() {
                let val = unsafe { self.get_unchecked((i, j)).clone().norm1() };

                if val > the_max {
                    the_max = val;
                    the_ij = (i, j);
                }
            }
        }

        the_ij
    }
}

impl<T: Scalar + PartialOrd + Signed, R: Dim, C: Dim, S: RawStorage<T, R, C>> Matrix<T, R, C, S> {
    /// Computes the index of the matrix component with the largest absolute value.
    ///
    /// This function finds the position (row, column) of the element in the matrix
    /// whose absolute value is the largest. The returned tuple contains
    /// `(row_index, column_index)`, both starting from 0.
    ///
    /// This is useful for finding the most extreme value in a matrix when you need to
    /// know both its location and you don't care about its sign.
    ///
    /// # Panics
    ///
    /// Panics if the matrix is empty.
    ///
    /// # Examples
    ///
    /// Basic usage with integers:
    ///
    /// ```
    /// # use nalgebra::Matrix2x3;
    /// let mat = Matrix2x3::new(
    ///     11, -12, 13,
    ///     21,  22, -23   // |-23| = 23 is the largest absolute value
    /// );
    /// assert_eq!(mat.iamax_full(), (1, 2));
    /// ```
    ///
    /// Working with floating point numbers:
    ///
    /// ```
    /// # use nalgebra::Matrix3x3;
    /// let mat = Matrix3x3::new(
    ///     1.0,  2.0,  3.0,
    ///     4.0, -50.0, 6.0,  // |-50.0| = 50.0 is the largest
    ///     7.0,  8.0,  9.0
    /// );
    /// let (row, col) = mat.iamax_full();
    /// assert_eq!((row, col), (1, 1));
    /// assert_eq!(mat[(row, col)], -50.0);
    /// ```
    ///
    /// Finding the dominant element for numerical stability checks:
    ///
    /// ```
    /// # use nalgebra::Matrix2x2;
    /// let mat = Matrix2x2::new(
    ///     0.001, -0.002,
    ///     0.003,  0.005  // |0.005| is the largest
    /// );
    /// let (row, col) = mat.iamax_full();
    /// assert_eq!((row, col), (1, 1));
    /// ```
    ///
    /// Negative values are compared by absolute value:
    ///
    /// ```
    /// # use nalgebra::Matrix2x2;
    /// let mat = Matrix2x2::new(
    ///     -100, 50,
    ///      30,  20
    /// );
    /// assert_eq!(mat.iamax_full(), (0, 0));  // |-100| = 100 is largest
    /// ```
    ///
    /// # See Also
    ///
    /// - [`amax`](Self::amax): Get the value of the largest absolute value
    /// - [`iamax`](crate::base::Vector::iamax): Find the index in a vector (returns single index)
    /// - [`icamax_full`](Self::icamax_full): Find the index of largest complex 1-norm
    #[inline]
    #[must_use]
    pub fn iamax_full(&self) -> (usize, usize) {
        assert!(!self.is_empty(), "The input matrix must not be empty.");

        let mut the_max = unsafe { self.get_unchecked((0, 0)).abs() };
        let mut the_ij = (0, 0);

        for j in 0..self.ncols() {
            for i in 0..self.nrows() {
                let val = unsafe { self.get_unchecked((i, j)).abs() };

                if val > the_max {
                    the_max = val;
                    the_ij = (i, j);
                }
            }
        }

        the_ij
    }
}

// TODO: find a way to avoid code duplication just for complex number support.
/// # Find the min and max components (vector-specific methods)
impl<T: Scalar, D: Dim, S: RawStorage<T, D>> Vector<T, D, S> {
    /// Computes the index of the vector component with the largest complex 1-norm.
    ///
    /// This function finds the index of the complex number in the vector that has the
    /// largest 1-norm. The 1-norm of a complex number `a + bi` is `|a| + |b|`.
    /// The index starts from 0.
    ///
    /// This is the vector-specific version that returns a single index, unlike
    /// [`icamax_full`](crate::base::Matrix::icamax_full) which returns `(row, column)` for matrices.
    ///
    /// # Panics
    ///
    /// Panics if the vector is empty.
    ///
    /// # Examples
    ///
    /// Basic usage with complex numbers:
    ///
    /// ```
    /// # use nalgebra::{Vector3, Complex};
    /// let vec = Vector3::new(
    ///     Complex::new(11.0, 3.0),   // 1-norm: 14
    ///     Complex::new(-15.0, 0.0),  // 1-norm: 15
    ///     Complex::new(13.0, 5.0)    // 1-norm: 18 (largest!)
    /// );
    /// assert_eq!(vec.icamax(), 2);
    /// ```
    ///
    /// Finding the dominant component:
    ///
    /// ```
    /// # use nalgebra::{Vector4, Complex};
    /// let vec = Vector4::new(
    ///     Complex::new(1.0, 1.0),    // 1-norm: 2
    ///     Complex::new(10.0, 5.0),   // 1-norm: 15 (largest!)
    ///     Complex::new(2.0, 3.0),    // 1-norm: 5
    ///     Complex::new(-8.0, 2.0)    // 1-norm: 10
    /// );
    /// let index = vec.icamax();
    /// assert_eq!(index, 1);
    /// assert_eq!(vec[index], Complex::new(10.0, 5.0));
    /// ```
    ///
    /// With real numbers (imaginary part is zero):
    ///
    /// ```
    /// # use nalgebra::{Vector3, Complex};
    /// let vec = Vector3::new(
    ///     Complex::new(5.0, 0.0),
    ///     Complex::new(-10.0, 0.0),  // 1-norm: 10 (largest)
    ///     Complex::new(3.0, 0.0)
    /// );
    /// assert_eq!(vec.icamax(), 1);
    /// ```
    ///
    /// # See Also
    ///
    /// - [`camax`](crate::base::Matrix::camax): Get the value of the largest complex 1-norm
    /// - [`icamax_full`](crate::base::Matrix::icamax_full): Find the index in a matrix (returns row and column)
    /// - [`iamax`](Self::iamax): Find the index of largest absolute value (for real numbers)
    #[inline]
    #[must_use]
    pub fn icamax(&self) -> usize
    where
        T: ComplexField,
    {
        assert!(!self.is_empty(), "The input vector must not be empty.");

        let mut the_max = unsafe { self.vget_unchecked(0).clone().norm1() };
        let mut the_i = 0;

        for i in 1..self.nrows() {
            let val = unsafe { self.vget_unchecked(i).clone().norm1() };

            if val > the_max {
                the_max = val;
                the_i = i;
            }
        }

        the_i
    }

    /// Computes the index and value of the vector component with the largest value.
    ///
    /// This function returns both the position and the value of the maximum element in the
    /// vector as a tuple `(index, value)`. The index starts from 0. This is useful when you
    /// need to know not just what the maximum value is, but also where it's located.
    ///
    /// # Panics
    ///
    /// Panics if the vector is empty.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use nalgebra::Vector3;
    /// let vec = Vector3::new(11, -15, 13);
    /// let (index, value) = vec.argmax();
    /// assert_eq!(index, 2);
    /// assert_eq!(value, 13);
    /// ```
    ///
    /// Working with floating point numbers:
    ///
    /// ```
    /// # use nalgebra::Vector4;
    /// let vec = Vector4::new(3.5, 1.2, 7.8, 2.1);
    /// let (index, max_value) = vec.argmax();
    /// assert_eq!(index, 2);
    /// assert_eq!(max_value, 7.8);
    /// ```
    ///
    /// When all values are negative:
    ///
    /// ```
    /// # use nalgebra::Vector3;
    /// let vec = Vector3::new(-5, -2, -10);
    /// assert_eq!(vec.argmax(), (1, -2));  // -2 is the largest (least negative)
    /// ```
    ///
    /// Using the result to access the element:
    ///
    /// ```
    /// # use nalgebra::Vector5;
    /// let vec = Vector5::new(10, 20, 50, 30, 40);
    /// let (max_idx, max_val) = vec.argmax();
    /// assert_eq!(vec[max_idx], max_val);
    /// assert_eq!(vec[max_idx], 50);
    /// ```
    ///
    /// # See Also
    ///
    /// - [`imax`](Self::imax): Get only the index of the maximum value
    /// - [`max`](crate::base::Matrix::max): Get only the maximum value
    /// - [`argmin`](Self::argmin): Find the index and value of the minimum
    #[inline]
    #[must_use]
    pub fn argmax(&self) -> (usize, T)
    where
        T: PartialOrd,
    {
        assert!(!self.is_empty(), "The input vector must not be empty.");

        let mut the_max = unsafe { self.vget_unchecked(0) };
        let mut the_i = 0;

        for i in 1..self.nrows() {
            let val = unsafe { self.vget_unchecked(i) };

            if val > the_max {
                the_max = val;
                the_i = i;
            }
        }

        (the_i, the_max.clone())
    }

    /// Computes the index of the vector component with the largest value.
    ///
    /// This function finds and returns the index (starting from 0) of the maximum element
    /// in the vector. If you also need the value itself, use [`argmax`](Self::argmax) instead.
    ///
    /// # Panics
    ///
    /// Panics if the vector is empty.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use nalgebra::Vector3;
    /// let vec = Vector3::new(11, -15, 13);
    /// assert_eq!(vec.imax(), 2);  // 13 at index 2 is the largest
    /// ```
    ///
    /// With floating point numbers:
    ///
    /// ```
    /// # use nalgebra::Vector4;
    /// let vec = Vector4::new(1.5, 8.2, 3.7, 2.1);
    /// assert_eq!(vec.imax(), 1);  // 8.2 at index 1 is the largest
    /// ```
    ///
    /// When the maximum is negative:
    ///
    /// ```
    /// # use nalgebra::Vector3;
    /// let vec = Vector3::new(-10, -5, -20);
    /// assert_eq!(vec.imax(), 1);  // -5 is the largest (least negative)
    /// ```
    ///
    /// Accessing the maximum value using the index:
    ///
    /// ```
    /// # use nalgebra::Vector5;
    /// let vec = Vector5::new(10, 30, 50, 20, 40);
    /// let max_index = vec.imax();
    /// assert_eq!(vec[max_index], 50);
    /// ```
    ///
    /// # See Also
    ///
    /// - [`argmax`](Self::argmax): Get both index and value of the maximum
    /// - [`max`](crate::base::Matrix::max): Get only the maximum value
    /// - [`imin`](Self::imin): Find the index of the minimum value
    /// - [`iamax`](Self::iamax): Find the index of the largest absolute value
    #[inline]
    #[must_use]
    pub fn imax(&self) -> usize
    where
        T: PartialOrd,
    {
        self.argmax().0
    }

    /// Computes the index of the vector component with the largest absolute value.
    ///
    /// This function finds and returns the index (starting from 0) of the element whose
    /// absolute value is the largest. This is useful when you want to find the most extreme
    /// value regardless of its sign.
    ///
    /// This is the vector-specific version that returns a single index, unlike
    /// [`iamax_full`](crate::base::Matrix::iamax_full) which returns `(row, column)` for matrices.
    ///
    /// # Panics
    ///
    /// Panics if the vector is empty.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use nalgebra::Vector3;
    /// let vec = Vector3::new(11, -15, 13);
    /// assert_eq!(vec.iamax(), 1);  // |-15| = 15 is the largest absolute value
    /// ```
    ///
    /// With floating point numbers:
    ///
    /// ```
    /// # use nalgebra::Vector4;
    /// let vec = Vector4::new(3.5, -8.2, 7.1, 2.3);
    /// assert_eq!(vec.iamax(), 1);  // |-8.2| = 8.2 is the largest
    /// ```
    ///
    /// Finding the element farthest from zero:
    ///
    /// ```
    /// # use nalgebra::Vector5;
    /// let vec = Vector5::new(1.0, -100.0, 50.0, -25.0, 10.0);
    /// let index = vec.iamax();
    /// assert_eq!(index, 1);
    /// assert_eq!(vec[index], -100.0);  // -100 has the largest magnitude
    /// ```
    ///
    /// Difference from imax:
    ///
    /// ```
    /// # use nalgebra::Vector3;
    /// let vec = Vector3::new(5, -20, 10);
    /// assert_eq!(vec.imax(), 2);   // 10 is the largest value
    /// assert_eq!(vec.iamax(), 1);  // |-20| = 20 is the largest absolute value
    /// ```
    ///
    /// # See Also
    ///
    /// - [`amax`](crate::base::Matrix::amax): Get the value of the largest absolute value
    /// - [`iamax_full`](crate::base::Matrix::iamax_full): Find the index in a matrix (returns row and column)
    /// - [`imax`](Self::imax): Find the index of the maximum value (without absolute value)
    /// - [`iamin`](Self::iamin): Find the index of the smallest absolute value
    #[inline]
    #[must_use]
    pub fn iamax(&self) -> usize
    where
        T: PartialOrd + Signed,
    {
        assert!(!self.is_empty(), "The input vector must not be empty.");

        let mut the_max = unsafe { self.vget_unchecked(0).abs() };
        let mut the_i = 0;

        for i in 1..self.nrows() {
            let val = unsafe { self.vget_unchecked(i).abs() };

            if val > the_max {
                the_max = val;
                the_i = i;
            }
        }

        the_i
    }

    /// Computes the index and value of the vector component with the smallest value.
    ///
    /// This function returns both the position and the value of the minimum element in the
    /// vector as a tuple `(index, value)`. The index starts from 0. This is useful when you
    /// need to know not just what the minimum value is, but also where it's located.
    ///
    /// # Panics
    ///
    /// Panics if the vector is empty.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use nalgebra::Vector3;
    /// let vec = Vector3::new(11, -15, 13);
    /// let (index, value) = vec.argmin();
    /// assert_eq!(index, 1);
    /// assert_eq!(value, -15);
    /// ```
    ///
    /// With floating point numbers:
    ///
    /// ```
    /// # use nalgebra::Vector4;
    /// let vec = Vector4::new(3.5, 1.2, 7.8, 2.1);
    /// let (index, min_value) = vec.argmin();
    /// assert_eq!(index, 1);
    /// assert_eq!(min_value, 1.2);
    /// ```
    ///
    /// When all values are positive:
    ///
    /// ```
    /// # use nalgebra::Vector3;
    /// let vec = Vector3::new(10, 5, 20);
    /// assert_eq!(vec.argmin(), (1, 5));
    /// ```
    ///
    /// Using the result to access the element:
    ///
    /// ```
    /// # use nalgebra::Vector5;
    /// let vec = Vector5::new(50, 20, 10, 30, 40);
    /// let (min_idx, min_val) = vec.argmin();
    /// assert_eq!(vec[min_idx], min_val);
    /// assert_eq!(vec[min_idx], 10);
    /// ```
    ///
    /// Finding the most negative value:
    ///
    /// ```
    /// # use nalgebra::Vector4;
    /// let vec = Vector4::new(-5.0, 10.0, -20.0, 15.0);
    /// assert_eq!(vec.argmin(), (2, -20.0));  // -20.0 is the smallest (most negative)
    /// ```
    ///
    /// # See Also
    ///
    /// - [`imin`](Self::imin): Get only the index of the minimum value
    /// - [`min`](crate::base::Matrix::min): Get only the minimum value
    /// - [`argmax`](Self::argmax): Find the index and value of the maximum
    #[inline]
    #[must_use]
    pub fn argmin(&self) -> (usize, T)
    where
        T: PartialOrd,
    {
        assert!(!self.is_empty(), "The input vector must not be empty.");

        let mut the_min = unsafe { self.vget_unchecked(0) };
        let mut the_i = 0;

        for i in 1..self.nrows() {
            let val = unsafe { self.vget_unchecked(i) };

            if val < the_min {
                the_min = val;
                the_i = i;
            }
        }

        (the_i, the_min.clone())
    }

    /// Computes the index of the vector component with the smallest value.
    ///
    /// This function finds and returns the index (starting from 0) of the minimum element
    /// in the vector. If you also need the value itself, use [`argmin`](Self::argmin) instead.
    ///
    /// # Panics
    ///
    /// Panics if the vector is empty.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use nalgebra::Vector3;
    /// let vec = Vector3::new(11, -15, 13);
    /// assert_eq!(vec.imin(), 1);  // -15 at index 1 is the smallest
    /// ```
    ///
    /// With floating point numbers:
    ///
    /// ```
    /// # use nalgebra::Vector4;
    /// let vec = Vector4::new(8.5, 2.3, 5.7, 1.2);
    /// assert_eq!(vec.imin(), 3);  // 1.2 at index 3 is the smallest
    /// ```
    ///
    /// When the minimum is negative:
    ///
    /// ```
    /// # use nalgebra::Vector3;
    /// let vec = Vector3::new(10, 5, -3);
    /// assert_eq!(vec.imin(), 2);  // -3 is the smallest
    /// ```
    ///
    /// Accessing the minimum value using the index:
    ///
    /// ```
    /// # use nalgebra::Vector5;
    /// let vec = Vector5::new(50, 30, 10, 20, 40);
    /// let min_index = vec.imin();
    /// assert_eq!(vec[min_index], 10);
    /// ```
    ///
    /// # See Also
    ///
    /// - [`argmin`](Self::argmin): Get both index and value of the minimum
    /// - [`min`](crate::base::Matrix::min): Get only the minimum value
    /// - [`imax`](Self::imax): Find the index of the maximum value
    /// - [`iamin`](Self::iamin): Find the index of the smallest absolute value
    #[inline]
    #[must_use]
    pub fn imin(&self) -> usize
    where
        T: PartialOrd,
    {
        self.argmin().0
    }

    /// Computes the index of the vector component with the smallest absolute value.
    ///
    /// This function finds and returns the index (starting from 0) of the element whose
    /// absolute value is the smallest. This is useful for finding the value closest to zero
    /// in magnitude, regardless of its sign.
    ///
    /// # Panics
    ///
    /// Panics if the vector is empty.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use nalgebra::Vector3;
    /// let vec = Vector3::new(11, -15, 13);
    /// assert_eq!(vec.iamin(), 0);  // |11| = 11 is the smallest absolute value
    /// ```
    ///
    /// Finding the element closest to zero:
    ///
    /// ```
    /// # use nalgebra::Vector4;
    /// let vec = Vector4::new(100.0, -0.5, 50.0, 25.0);
    /// assert_eq!(vec.iamin(), 1);  // |-0.5| = 0.5 is closest to zero
    /// ```
    ///
    /// With negative numbers:
    ///
    /// ```
    /// # use nalgebra::Vector5;
    /// let vec = Vector5::new(-10, -100, -5, -50, -25);
    /// assert_eq!(vec.iamin(), 2);  // |-5| = 5 is the smallest absolute value
    /// ```
    ///
    /// Accessing the element using the index:
    ///
    /// ```
    /// # use nalgebra::Vector4;
    /// let vec = Vector4::new(10, -2, 20, 5);
    /// let min_abs_index = vec.iamin();
    /// assert_eq!(vec[min_abs_index], -2);  // -2 has the smallest absolute value
    /// ```
    ///
    /// Difference from imin:
    ///
    /// ```
    /// # use nalgebra::Vector3;
    /// let vec = Vector3::new(5, -20, 10);
    /// assert_eq!(vec.imin(), 1);   // -20 is the smallest value
    /// assert_eq!(vec.iamin(), 0);  // |5| = 5 is the smallest absolute value
    /// ```
    ///
    /// # See Also
    ///
    /// - [`amin`](crate::base::Matrix::amin): Get the value of the smallest absolute value
    /// - [`imin`](Self::imin): Find the index of the minimum value (without absolute value)
    /// - [`iamax`](Self::iamax): Find the index of the largest absolute value
    #[inline]
    #[must_use]
    pub fn iamin(&self) -> usize
    where
        T: PartialOrd + Signed,
    {
        assert!(!self.is_empty(), "The input vector must not be empty.");

        let mut the_min = unsafe { self.vget_unchecked(0).abs() };
        let mut the_i = 0;

        for i in 1..self.nrows() {
            let val = unsafe { self.vget_unchecked(i).abs() };

            if val < the_min {
                the_min = val;
                the_i = i;
            }
        }

        the_i
    }
}
