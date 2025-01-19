use crate::allocator::Allocator;
use crate::storage::RawStorage;
use crate::{Const, DefaultAllocator, Dim, Matrix, OVector, RowOVector, Scalar, VectorView, U1};
use num::{One, Zero};
use simba::scalar::{ClosedAddAssign, ClosedMulAssign, Field, SupersetOf};
use std::mem::MaybeUninit;

/// # Folding on columns and rows
impl<T: Scalar, R: Dim, C: Dim, S: RawStorage<T, R, C>> Matrix<T, R, C, S> {
    /// Returns a row vector where each element is the result of the application of `f` on the
    /// corresponding column of the original matrix.
    #[inline]
    #[must_use]
    pub fn compress_rows(
        &self,
        f: impl Fn(VectorView<'_, T, R, S::RStride, S::CStride>) -> T,
    ) -> RowOVector<T, C>
    where
        DefaultAllocator: Allocator<U1, C>,
    {
        let ncols = self.shape_generic().1;
        let mut res = Matrix::uninit(Const::<1>, ncols);

        for i in 0..ncols.value() {
            // TODO: avoid bound checking of column.
            // Safety: all indices are in range.
            unsafe {
                *res.get_unchecked_mut((0, i)) = MaybeUninit::new(f(self.column(i)));
            }
        }

        // Safety: res is now fully initialized.
        unsafe { res.assume_init() }
    }

    /// Returns a column vector where each element is the result of the application of `f` on the
    /// corresponding column of the original matrix.
    ///
    /// This is the same as `self.compress_rows(f).transpose()`.
    #[inline]
    #[must_use]
    pub fn compress_rows_tr(
        &self,
        f: impl Fn(VectorView<'_, T, R, S::RStride, S::CStride>) -> T,
    ) -> OVector<T, C>
    where
        DefaultAllocator: Allocator<C>,
    {
        let ncols = self.shape_generic().1;
        let mut res = Matrix::uninit(ncols, Const::<1>);

        for i in 0..ncols.value() {
            // TODO: avoid bound checking of column.
            // Safety: all indices are in range.
            unsafe {
                *res.vget_unchecked_mut(i) = MaybeUninit::new(f(self.column(i)));
            }
        }

        // Safety: res is now fully initialized.
        unsafe { res.assume_init() }
    }

    /// Returns a column vector resulting from the folding of `f` on each column of this matrix.
    #[inline]
    #[must_use]
    pub fn compress_columns(
        &self,
        init: OVector<T, R>,
        f: impl Fn(&mut OVector<T, R>, VectorView<'_, T, R, S::RStride, S::CStride>),
    ) -> OVector<T, R>
    where
        DefaultAllocator: Allocator<R>,
    {
        let mut res = init;

        for i in 0..self.ncols() {
            f(&mut res, self.column(i))
        }

        res
    }
}

/// # Common statistics operations
impl<T: Scalar, R: Dim, C: Dim, S: RawStorage<T, R, C>> Matrix<T, R, C, S> {
    /*
     *
     * Sum computation.
     *
     */
    /// The sum of all the elements of this matrix.
    ///
    /// # Example
    ///
    /// ```
    /// # use nalgebra::Matrix2x3;
    ///
    /// let m = Matrix2x3::new(1.0, 2.0, 3.0,
    ///                        4.0, 5.0, 6.0);
    /// assert_eq!(m.sum(), 21.0);
    /// ```
    #[inline]
    #[must_use]
    pub fn sum(&self) -> T
    where
        T: ClosedAddAssign + Zero,
    {
        self.iter().cloned().fold(T::zero(), |a, b| a + b)
    }

    /// The sum of all the rows of this matrix.
    ///
    /// Use `.row_sum_tr` if you need the result in a column vector instead.
    ///
    /// # Example
    ///
    /// ```
    /// # use nalgebra::{Matrix2x3, Matrix3x2};
    /// # use nalgebra::{RowVector2, RowVector3};
    ///
    /// let m = Matrix2x3::new(1.0, 2.0, 3.0,
    ///                        4.0, 5.0, 6.0);
    /// assert_eq!(m.row_sum(), RowVector3::new(5.0, 7.0, 9.0));
    ///
    /// let mint = Matrix3x2::new(1, 2,
    ///                           3, 4,
    ///                           5, 6);
    /// assert_eq!(mint.row_sum(), RowVector2::new(9,12));
    /// ```
    #[inline]
    #[must_use]
    pub fn row_sum(&self) -> RowOVector<T, C>
    where
        T: ClosedAddAssign + Zero,
        DefaultAllocator: Allocator<U1, C>,
    {
        self.compress_rows(|col| col.sum())
    }

    /// The sum of all the rows of this matrix. The result is transposed and returned as a column vector.
    ///
    /// # Example
    ///
    /// ```
    /// # use nalgebra::{Matrix2x3, Matrix3x2};
    /// # use nalgebra::{Vector2, Vector3};
    ///
    /// let m = Matrix2x3::new(1.0, 2.0, 3.0,
    ///                        4.0, 5.0, 6.0);
    /// assert_eq!(m.row_sum_tr(), Vector3::new(5.0, 7.0, 9.0));
    ///
    /// let mint = Matrix3x2::new(1, 2,
    ///                           3, 4,
    ///                           5, 6);
    /// assert_eq!(mint.row_sum_tr(), Vector2::new(9, 12));
    /// ```
    #[inline]
    #[must_use]
    pub fn row_sum_tr(&self) -> OVector<T, C>
    where
        T: ClosedAddAssign + Zero,
        DefaultAllocator: Allocator<C>,
    {
        self.compress_rows_tr(|col| col.sum())
    }

    /// The sum of all the columns of this matrix.
    ///
    /// # Example
    ///
    /// ```
    /// # use nalgebra::{Matrix2x3, Matrix3x2};
    /// # use nalgebra::{Vector2, Vector3};
    ///
    /// let m = Matrix2x3::new(1.0, 2.0, 3.0,
    ///                        4.0, 5.0, 6.0);
    /// assert_eq!(m.column_sum(), Vector2::new(6.0, 15.0));
    ///
    /// let mint = Matrix3x2::new(1, 2,
    ///                           3, 4,
    ///                           5, 6);
    /// assert_eq!(mint.column_sum(), Vector3::new(3, 7, 11));
    /// ```
    #[inline]
    #[must_use]
    pub fn column_sum(&self) -> OVector<T, R>
    where
        T: ClosedAddAssign + Zero,
        DefaultAllocator: Allocator<R>,
    {
        let nrows = self.shape_generic().0;
        self.compress_columns(OVector::zeros_generic(nrows, Const::<1>), |out, col| {
            *out += col;
        })
    }

    /*
     *
     * Product computation.
     *
     */
    /// The product of all the elements of this matrix.
    ///
    /// # Example
    ///
    /// ```
    /// # use nalgebra::Matrix2x3;
    ///
    /// let m = Matrix2x3::new(1.0, 2.0, 3.0,
    ///                        4.0, 5.0, 6.0);
    /// assert_eq!(m.product(), 720.0);
    /// ```
    #[inline]
    #[must_use]
    pub fn product(&self) -> T
    where
        T: ClosedMulAssign + One,
    {
        self.iter().cloned().fold(T::one(), |a, b| a * b)
    }

    /// The product of all the rows of this matrix.
    ///
    /// Use `.row_sum_tr` if you need the result in a column vector instead.
    ///
    /// # Example
    ///
    /// ```
    /// # use nalgebra::{Matrix2x3, Matrix3x2};
    /// # use nalgebra::{RowVector2, RowVector3};
    ///
    /// let m = Matrix2x3::new(1.0, 2.0, 3.0,
    ///                        4.0, 5.0, 6.0);
    /// assert_eq!(m.row_product(), RowVector3::new(4.0, 10.0, 18.0));
    ///
    /// let mint = Matrix3x2::new(1, 2,
    ///                           3, 4,
    ///                           5, 6);
    /// assert_eq!(mint.row_product(), RowVector2::new(15, 48));
    /// ```
    #[inline]
    #[must_use]
    pub fn row_product(&self) -> RowOVector<T, C>
    where
        T: ClosedMulAssign + One,
        DefaultAllocator: Allocator<U1, C>,
    {
        self.compress_rows(|col| col.product())
    }

    /// The product of all the rows of this matrix. The result is transposed and returned as a column vector.
    ///
    /// # Example
    ///
    /// ```
    /// # use nalgebra::{Matrix2x3, Matrix3x2};
    /// # use nalgebra::{Vector2, Vector3};
    ///
    /// let m = Matrix2x3::new(1.0, 2.0, 3.0,
    ///                        4.0, 5.0, 6.0);
    /// assert_eq!(m.row_product_tr(), Vector3::new(4.0, 10.0, 18.0));
    ///
    /// let mint = Matrix3x2::new(1, 2,
    ///                           3, 4,
    ///                           5, 6);
    /// assert_eq!(mint.row_product_tr(), Vector2::new(15, 48));
    /// ```
    #[inline]
    #[must_use]
    pub fn row_product_tr(&self) -> OVector<T, C>
    where
        T: ClosedMulAssign + One,
        DefaultAllocator: Allocator<C>,
    {
        self.compress_rows_tr(|col| col.product())
    }

    /// The product of all the columns of this matrix.
    ///
    /// # Example
    ///
    /// ```
    /// # use nalgebra::{Matrix2x3, Matrix3x2};
    /// # use nalgebra::{Vector2, Vector3};
    ///
    /// let m = Matrix2x3::new(1.0, 2.0, 3.0,
    ///                        4.0, 5.0, 6.0);
    /// assert_eq!(m.column_product(), Vector2::new(6.0, 120.0));
    ///
    /// let mint = Matrix3x2::new(1, 2,
    ///                           3, 4,
    ///                           5, 6);
    /// assert_eq!(mint.column_product(), Vector3::new(2, 12, 30));
    /// ```
    #[inline]
    #[must_use]
    pub fn column_product(&self) -> OVector<T, R>
    where
        T: ClosedMulAssign + One,
        DefaultAllocator: Allocator<R>,
    {
        let nrows = self.shape_generic().0;
        self.compress_columns(
            OVector::repeat_generic(nrows, Const::<1>, T::one()),
            |out, col| {
                out.component_mul_assign(&col);
            },
        )
    }

    /*
     *
     * Variance computation.
     *
     */
    /// The variance of all the elements of this matrix.
    ///
    /// # Example
    ///
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::Matrix2x3;
    ///
    /// let m = Matrix2x3::new(1.0, 2.0, 3.0,
    ///                        4.0, 5.0, 6.0);
    /// assert_relative_eq!(m.variance(), 35.0 / 12.0, epsilon = 1.0e-8);
    /// ```
    #[inline]
    #[must_use]
    pub fn variance(&self) -> T
    where
        T: Field + SupersetOf<f64>,
    {
        if self.is_empty() {
            T::zero()
        } else {
            let n_elements: T = crate::convert(self.len() as f64);
            let mean = self.mean();

            self.iter().cloned().fold(T::zero(), |acc, x| {
                acc + (x.clone() - mean.clone()) * (x - mean.clone())
            }) / n_elements
        }
    }

    /// The variance of all the rows of this matrix.
    ///
    /// Use `.row_variance_tr` if you need the result in a column vector instead.
    /// # Example
    ///
    /// ```
    /// # use nalgebra::{Matrix2x3, RowVector3};
    ///
    /// let m = Matrix2x3::new(1.0, 2.0, 3.0,
    ///                        4.0, 5.0, 6.0);
    /// assert_eq!(m.row_variance(), RowVector3::new(2.25, 2.25, 2.25));
    /// ```
    #[inline]
    #[must_use]
    pub fn row_variance(&self) -> RowOVector<T, C>
    where
        T: Field + SupersetOf<f64>,
        DefaultAllocator: Allocator<U1, C>,
    {
        self.compress_rows(|col| col.variance())
    }

    /// The variance of all the rows of this matrix. The result is transposed and returned as a column vector.
    ///
    /// # Example
    ///
    /// ```
    /// # use nalgebra::{Matrix2x3, Vector3};
    ///
    /// let m = Matrix2x3::new(1.0, 2.0, 3.0,
    ///                        4.0, 5.0, 6.0);
    /// assert_eq!(m.row_variance_tr(), Vector3::new(2.25, 2.25, 2.25));
    /// ```
    #[inline]
    #[must_use]
    pub fn row_variance_tr(&self) -> OVector<T, C>
    where
        T: Field + SupersetOf<f64>,
        DefaultAllocator: Allocator<C>,
    {
        self.compress_rows_tr(|col| col.variance())
    }

    /// The variance of all the columns of this matrix.
    ///
    /// # Example
    ///
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{Matrix2x3, Vector2};
    ///
    /// let m = Matrix2x3::new(1.0, 2.0, 3.0,
    ///                        4.0, 5.0, 6.0);
    /// assert_relative_eq!(m.column_variance(), Vector2::new(2.0 / 3.0, 2.0 / 3.0), epsilon = 1.0e-8);
    /// ```
    #[inline]
    #[must_use]
    pub fn column_variance(&self) -> OVector<T, R>
    where
        T: Field + SupersetOf<f64>,
        DefaultAllocator: Allocator<R>,
    {
        let (nrows, ncols) = self.shape_generic();

        let mut mean = self.column_mean();
        mean.apply(|e| *e = -(e.clone() * e.clone()));

        let denom = T::one() / crate::convert::<_, T>(ncols.value() as f64);
        self.compress_columns(mean, |out, col| {
            for i in 0..nrows.value() {
                unsafe {
                    let val = col.vget_unchecked(i);
                    *out.vget_unchecked_mut(i) += denom.clone() * val.clone() * val.clone()
                }
            }
        })
    }

    /*
     *
     * Mean computation.
     *
     */
    /// The mean of all the elements of this matrix.
    ///
    /// # Example
    ///
    /// ```
    /// # use nalgebra::Matrix2x3;
    ///
    /// let m = Matrix2x3::new(1.0, 2.0, 3.0,
    ///                        4.0, 5.0, 6.0);
    /// assert_eq!(m.mean(), 3.5);
    /// ```
    #[inline]
    #[must_use]
    pub fn mean(&self) -> T
    where
        T: Field + SupersetOf<f64>,
    {
        if self.is_empty() {
            T::zero()
        } else {
            self.sum() / crate::convert(self.len() as f64)
        }
    }

    /// The mean of all the rows of this matrix.
    ///
    /// Use `.row_mean_tr` if you need the result in a column vector instead.
    ///
    /// # Example
    ///
    /// ```
    /// # use nalgebra::{Matrix2x3, RowVector3};
    ///
    /// let m = Matrix2x3::new(1.0, 2.0, 3.0,
    ///                        4.0, 5.0, 6.0);
    /// assert_eq!(m.row_mean(), RowVector3::new(2.5, 3.5, 4.5));
    /// ```
    #[inline]
    #[must_use]
    pub fn row_mean(&self) -> RowOVector<T, C>
    where
        T: Field + SupersetOf<f64>,
        DefaultAllocator: Allocator<U1, C>,
    {
        self.compress_rows(|col| col.mean())
    }

    /// The mean of all the rows of this matrix. The result is transposed and returned as a column vector.
    ///
    /// # Example
    ///
    /// ```
    /// # use nalgebra::{Matrix2x3, Vector3};
    ///
    /// let m = Matrix2x3::new(1.0, 2.0, 3.0,
    ///                        4.0, 5.0, 6.0);
    /// assert_eq!(m.row_mean_tr(), Vector3::new(2.5, 3.5, 4.5));
    /// ```
    #[inline]
    #[must_use]
    pub fn row_mean_tr(&self) -> OVector<T, C>
    where
        T: Field + SupersetOf<f64>,
        DefaultAllocator: Allocator<C>,
    {
        self.compress_rows_tr(|col| col.mean())
    }

    /// The mean of all the columns of this matrix.
    ///
    /// # Example
    ///
    /// ```
    /// # use nalgebra::{Matrix2x3, Vector2};
    ///
    /// let m = Matrix2x3::new(1.0, 2.0, 3.0,
    ///                        4.0, 5.0, 6.0);
    /// assert_eq!(m.column_mean(), Vector2::new(2.0, 5.0));
    /// ```
    #[inline]
    #[must_use]
    pub fn column_mean(&self) -> OVector<T, R>
    where
        T: Field + SupersetOf<f64>,
        DefaultAllocator: Allocator<R>,
    {
        let (nrows, ncols) = self.shape_generic();
        let denom = T::one() / crate::convert::<_, T>(ncols.value() as f64);
        self.compress_columns(OVector::zeros_generic(nrows, Const::<1>), |out, col| {
            out.axpy(denom.clone(), &col, T::one())
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::*;

    #[test]
    fn test_sum() {
        let square_float = Matrix2::new(1.0, 2.0, 3.0, 4.0);
        assert_eq!(square_float.sum(), 10.0);
        let rect41_float = Matrix4x1::new(1.0, 2.0, 3.0, 4.0);
        assert_eq!(rect41_float.sum(), 10.0);
        let rect14_float = Matrix1x4::new(1.0, 2.0, 3.0, 4.0);
        assert_eq!(rect14_float.sum(), 10.0);

        let square_int = Matrix2::new(1, 2, 3, 4);
        assert_eq!(square_int.sum(), 10);
        let rect41_int = Matrix4x1::new(1, 2, 3, 4);
        assert_eq!(rect41_int.sum(), 10);
        let rect14_int = Matrix1x4::new(1, 2, 3, 4);
        assert_eq!(rect14_int.sum(), 10);
    }
    #[test]
    fn test_sum_edge() {
        let edge00_float = DMatrix::<f32>::zeros(0, 0);
        assert_eq!(edge00_float.sum(), 0.0);
        let edge40_float = DMatrix::<f32>::zeros(4, 0);
        assert_eq!(edge40_float.sum(), 0.0);
        let edge04_float = DMatrix::<f32>::zeros(0, 4);
        assert_eq!(edge04_float.sum(), 0.0);

        let edge00_int = DMatrix::<i32>::zeros(0, 0);
        assert_eq!(edge00_int.sum(), 0);
        let edge40_int = DMatrix::<i32>::zeros(4, 0);
        assert_eq!(edge40_int.sum(), 0);
        let edge04_int = DMatrix::<i32>::zeros(0, 4);
        assert_eq!(edge04_int.sum(), 0);
    }
    #[test]
    fn test_row_sum() {
        let square_float = Matrix2::new(1.0, 2.0, 3.0, 4.0);
        assert_eq!(square_float.row_sum(), RowVector2::new(4.0, 6.0));
        let rect41_float = Matrix4x1::new(1.0, 2.0, 3.0, 4.0);
        assert_eq!(rect41_float.row_sum(), RowVector1::new(10.0));
        let rect14_float = Matrix1x4::new(1.0, 2.0, 3.0, 4.0);
        assert_eq!(rect14_float.row_sum(), RowVector4::new(1.0, 2.0, 3.0, 4.0));

        let square_int = Matrix2::new(1, 2, 3, 4);
        assert_eq!(square_int.row_sum(), RowVector2::new(4, 6));
        let rect41_int = Matrix4x1::new(1, 2, 3, 4);
        assert_eq!(rect41_int.row_sum(), RowVector1::new(10));
        let rect14_int = Matrix1x4::new(1, 2, 3, 4);
        assert_eq!(rect14_int.row_sum(), RowVector4::new(1, 2, 3, 4));
    }
    #[test]
    fn test_row_sum_edge() {
        let edge00_float = DMatrix::<f32>::zeros(0, 0);
        assert_eq!(edge00_float.row_sum(), RowDVector::<f32>::zeros(0));
        let edge40_float = DMatrix::<f32>::zeros(4, 0);
        assert_eq!(edge40_float.row_sum(), RowDVector::<f32>::zeros(0));
        let edge04_float = DMatrix::<f32>::zeros(0, 4);
        assert_eq!(edge04_float.row_sum(), RowDVector::<f32>::zeros(4));

        let edge00_int = DMatrix::<i32>::zeros(0, 0);
        assert_eq!(edge00_int.row_sum(), RowDVector::<i32>::zeros(0));
        let edge40_int = DMatrix::<i32>::zeros(4, 0);
        assert_eq!(edge40_int.row_sum(), RowDVector::<i32>::zeros(0));
        let edge04_int = DMatrix::<i32>::zeros(0, 4);
        assert_eq!(edge04_int.row_sum(), RowDVector::<i32>::zeros(4));
    }
    #[test]
    fn test_row_sum_tr() {
        let square_float = Matrix2::new(1.0, 2.0, 3.0, 4.0);
        assert_eq!(square_float.row_sum_tr(), Vector2::new(4.0, 6.0));
        let rect41_float = Matrix4x1::new(1.0, 2.0, 3.0, 4.0);
        assert_eq!(rect41_float.row_sum_tr(), Vector1::new(10.0));
        let rect14_float = Matrix1x4::new(1.0, 2.0, 3.0, 4.0);
        assert_eq!(rect14_float.row_sum_tr(), Vector4::new(1.0, 2.0, 3.0, 4.0));

        let square_int = Matrix2::new(1, 2, 3, 4);
        assert_eq!(square_int.row_sum_tr(), Vector2::new(4, 6));
        let rect41_int = Matrix4x1::new(1, 2, 3, 4);
        assert_eq!(rect41_int.row_sum_tr(), Vector1::new(10));
        let rect14_int = Matrix1x4::new(1, 2, 3, 4);
        assert_eq!(rect14_int.row_sum_tr(), Vector4::new(1, 2, 3, 4));
    }
    #[test]
    fn test_row_sum_tr_edge() {
        let edge00_float = DMatrix::<f32>::zeros(0, 0);
        assert_eq!(edge00_float.row_sum_tr(), DVector::<f32>::zeros(0));
        let edge40_float = DMatrix::<f32>::zeros(4, 0);
        assert_eq!(edge40_float.row_sum_tr(), DVector::<f32>::zeros(0));
        let edge04_float = DMatrix::<f32>::zeros(0, 4);
        assert_eq!(edge04_float.row_sum_tr(), DVector::<f32>::zeros(4));

        let edge00_int = DMatrix::<i32>::zeros(0, 0);
        assert_eq!(edge00_int.row_sum_tr(), DVector::<i32>::zeros(0));
        let edge40_int = DMatrix::<i32>::zeros(4, 0);
        assert_eq!(edge40_int.row_sum_tr(), DVector::<i32>::zeros(0));
        let edge04_int = DMatrix::<i32>::zeros(0, 4);
        assert_eq!(edge04_int.row_sum_tr(), DVector::<i32>::zeros(4));
    }
    #[test]
    fn test_column_sum() {
        let square_float = Matrix2::new(1.0, 2.0, 3.0, 4.0);
        assert_eq!(square_float.column_sum(), Vector2::new(3.0, 7.0));
        let rect41_float = Matrix4x1::new(1.0, 2.0, 3.0, 4.0);
        assert_eq!(rect41_float.column_sum(), Vector4::new(1.0, 2.0, 3.0, 4.0));
        let rect14_float = Matrix1x4::new(1.0, 2.0, 3.0, 4.0);
        assert_eq!(rect14_float.column_sum(), Vector1::new(10.0));

        let square_int = Matrix2::new(1, 2, 3, 4);
        assert_eq!(square_int.column_sum(), Vector2::new(3, 7));
        let rect41_int = Matrix4x1::new(1, 2, 3, 4);
        assert_eq!(rect41_int.column_sum(), Vector4::new(1, 2, 3, 4));
        let rect14_int = Matrix1x4::new(1, 2, 3, 4);
        assert_eq!(rect14_int.column_sum(), Vector1::new(10));
    }
    #[test]
    fn test_column_sum_edge() {
        let edge00_float = DMatrix::<f32>::zeros(0, 0);
        assert_eq!(edge00_float.column_sum(), DVector::<f32>::zeros(0));
        let edge40_float = DMatrix::<f32>::zeros(4, 0);
        assert_eq!(edge40_float.column_sum(), DVector::<f32>::zeros(4));
        let edge04_float = DMatrix::<f32>::zeros(0, 4);
        assert_eq!(edge04_float.column_sum(), DVector::<f32>::zeros(0));

        let edge00_int = DMatrix::<i32>::zeros(0, 0);
        assert_eq!(edge00_int.column_sum(), DVector::<i32>::zeros(0));
        let edge40_int = DMatrix::<i32>::zeros(4, 0);
        assert_eq!(edge40_int.column_sum(), DVector::<i32>::zeros(4));
        let edge04_int = DMatrix::<i32>::zeros(0, 4);
        assert_eq!(edge04_int.column_sum(), DVector::<i32>::zeros(0));
    }
    #[test]
    fn test_product() {
        let square_float = Matrix2::new(1.0, 2.0, 3.0, 4.0);
        assert_eq!(square_float.product(), 24.0);
        let rect41_float = Matrix4x1::new(1.0, 2.0, 3.0, 4.0);
        assert_eq!(rect41_float.product(), 24.0);
        let rect14_float = Matrix1x4::new(1.0, 2.0, 3.0, 4.0);
        assert_eq!(rect14_float.product(), 24.0);

        let square_int = Matrix2::new(1, 2, 3, 4);
        assert_eq!(square_int.product(), 24);
        let rect41_int = Matrix4x1::new(1, 2, 3, 4);
        assert_eq!(rect41_int.product(), 24);
        let rect14_int = Matrix1x4::new(1, 2, 3, 4);
        assert_eq!(rect14_int.product(), 24);
    }
    #[test]
    fn test_product_edge() {
        let edge00_float = DMatrix::<f32>::zeros(0, 0);
        assert_eq!(edge00_float.product(), 1.0);
        let edge40_float = DMatrix::<f32>::zeros(4, 0);
        assert_eq!(edge40_float.product(), 1.0);
        let edge04_float = DMatrix::<f32>::zeros(0, 4);
        assert_eq!(edge04_float.product(), 1.0);

        let edge00_int = DMatrix::<i32>::zeros(0, 0);
        assert_eq!(edge00_int.product(), 1);
        let edge40_int = DMatrix::<i32>::zeros(4, 0);
        assert_eq!(edge40_int.product(), 1);
        let edge04_int = DMatrix::<i32>::zeros(0, 4);
        assert_eq!(edge04_int.product(), 1);
    }
    #[test]
    fn test_row_product() {
        let square_float = Matrix2::new(1.0, 2.0, 3.0, 4.0);
        assert_eq!(square_float.row_product(), RowVector2::new(3.0, 8.0));
        let rect41_float = Matrix4x1::new(1.0, 2.0, 3.0, 4.0);
        assert_eq!(rect41_float.row_product(), RowVector1::new(24.0));
        let rect14_float = Matrix1x4::new(1.0, 2.0, 3.0, 4.0);
        assert_eq!(
            rect14_float.row_product(),
            RowVector4::new(1.0, 2.0, 3.0, 4.0)
        );

        let square_int = Matrix2::new(1, 2, 3, 4);
        assert_eq!(square_int.row_product(), RowVector2::new(3, 8));
        let rect41_int = Matrix4x1::new(1, 2, 3, 4);
        assert_eq!(rect41_int.row_product(), RowVector1::new(24));
        let rect14_int = Matrix1x4::new(1, 2, 3, 4);
        assert_eq!(rect14_int.row_product(), RowVector4::new(1, 2, 3, 4));
    }
    #[test]
    fn test_row_product_edge() {
        let edge00_float = DMatrix::<f32>::zeros(0, 0);
        assert_eq!(edge00_float.row_product(), RowDVector::<f32>::zeros(0));
        let edge40_float = DMatrix::<f32>::zeros(4, 0);
        assert_eq!(edge40_float.row_product(), RowDVector::<f32>::zeros(0));
        let edge04_float = DMatrix::<f32>::zeros(0, 4);
        assert_eq!(
            edge04_float.row_product(),
            RowDVector::from_row_slice(&[1.0, 1.0, 1.0, 1.0])
        );

        let edge00_int = DMatrix::<i32>::zeros(0, 0);
        assert_eq!(edge00_int.row_product(), RowDVector::<i32>::zeros(0));
        let edge40_int = DMatrix::<i32>::zeros(4, 0);
        assert_eq!(edge40_int.row_product(), RowDVector::<i32>::zeros(0));
        let edge04_int = DMatrix::<i32>::zeros(0, 4);
        assert_eq!(
            edge04_int.row_product(),
            RowDVector::from_row_slice(&[1, 1, 1, 1])
        );
    }
    #[test]
    fn test_row_product_tr() {
        let square_float = Matrix2::new(1.0, 2.0, 3.0, 4.0);
        assert_eq!(square_float.row_product_tr(), Vector2::new(3.0, 8.0));
        let rect41_float = Matrix4x1::new(1.0, 2.0, 3.0, 4.0);
        assert_eq!(rect41_float.row_product_tr(), Vector1::new(24.0));
        let rect14_float = Matrix1x4::new(1.0, 2.0, 3.0, 4.0);
        assert_eq!(
            rect14_float.row_product_tr(),
            Vector4::new(1.0, 2.0, 3.0, 4.0)
        );

        let square_int = Matrix2::new(1, 2, 3, 4);
        assert_eq!(square_int.row_product_tr(), Vector2::new(3, 8));
        let rect41_int = Matrix4x1::new(1, 2, 3, 4);
        assert_eq!(rect41_int.row_product_tr(), Vector1::new(24));
        let rect14_int = Matrix1x4::new(1, 2, 3, 4);
        assert_eq!(rect14_int.row_product_tr(), Vector4::new(1, 2, 3, 4));
    }
    #[test]
    fn test_row_product_tr_edge() {
        let edge00_float = DMatrix::<f32>::zeros(0, 0);
        assert_eq!(edge00_float.row_product_tr(), DVector::<f32>::zeros(0));
        let edge40_float = DMatrix::<f32>::zeros(4, 0);
        assert_eq!(edge40_float.row_product_tr(), DVector::<f32>::zeros(0));
        let edge04_float = DMatrix::<f32>::zeros(0, 4);
        assert_eq!(
            edge04_float.row_product_tr(),
            DVector::from_row_slice(&[1.0, 1.0, 1.0, 1.0])
        );

        let edge00_int = DMatrix::<i32>::zeros(0, 0);
        assert_eq!(edge00_int.row_product_tr(), DVector::<i32>::zeros(0));
        let edge40_int = DMatrix::<i32>::zeros(4, 0);
        assert_eq!(edge40_int.row_product_tr(), DVector::<i32>::zeros(0));
        let edge04_int = DMatrix::<i32>::zeros(0, 4);
        assert_eq!(
            edge04_int.row_product_tr(),
            DVector::from_row_slice(&[1, 1, 1, 1])
        );
    }
    #[test]
    fn test_column_product() {
        let square_float = Matrix2::new(1.0, 2.0, 3.0, 4.0);
        assert_eq!(square_float.column_product(), Vector2::new(2.0, 12.0));
        let rect41_float = Matrix4x1::new(1.0, 2.0, 3.0, 4.0);
        assert_eq!(
            rect41_float.column_product(),
            Vector4::new(1.0, 2.0, 3.0, 4.0)
        );
        let rect14_float = Matrix1x4::new(1.0, 2.0, 3.0, 4.0);
        assert_eq!(rect14_float.column_product(), Vector1::new(24.0));

        let square_int = Matrix2::new(1, 2, 3, 4);
        assert_eq!(square_int.column_product(), Vector2::new(2, 12));
        let rect41_int = Matrix4x1::new(1, 2, 3, 4);
        assert_eq!(rect41_int.column_product(), Vector4::new(1, 2, 3, 4));
        let rect14_int = Matrix1x4::new(1, 2, 3, 4);
        assert_eq!(rect14_int.column_product(), Vector1::new(24));
    }
    #[test]
    fn test_column_product_edge() {
        let edge00_float = DMatrix::<f32>::zeros(0, 0);
        assert_eq!(edge00_float.column_product(), DVector::<f32>::zeros(0));
        let edge40_float = DMatrix::<f32>::zeros(4, 0);
        assert_eq!(
            edge40_float.column_product(),
            DVector::from_row_slice(&[1.0, 1.0, 1.0, 1.0])
        );
        let edge04_float = DMatrix::<f32>::zeros(0, 4);
        assert_eq!(edge04_float.column_product(), DVector::<f32>::zeros(0));

        let edge00_int = DMatrix::<i32>::zeros(0, 0);
        assert_eq!(edge00_int.column_product(), DVector::<i32>::zeros(0));
        let edge40_int = DMatrix::<i32>::zeros(4, 0);
        assert_eq!(
            edge40_int.column_product(),
            DVector::from_row_slice(&[1, 1, 1, 1])
        );
        let edge04_int = DMatrix::<i32>::zeros(0, 4);
        assert_eq!(edge04_int.column_product(), DVector::<i32>::zeros(0));
    }
    #[test]
    fn test_variance() {
        let square_float = Matrix2::new(1.0, 2.0, 3.0, 4.0);
        assert_eq!(square_float.variance(), 1.25);
        let rect41_float = Matrix4x1::new(1.0, 2.0, 3.0, 4.0);
        assert_eq!(rect41_float.variance(), 1.25);
        let rect14_float = Matrix1x4::new(1.0, 2.0, 3.0, 4.0);
        assert_eq!(rect14_float.variance(), 1.25);
    }
    #[test]
    fn test_variance_edge() {
        let edge00_float = DMatrix::<f32>::zeros(0, 0);
        assert_eq!(edge00_float.variance(), 0.0);
        let edge40_float = DMatrix::<f32>::zeros(4, 0);
        assert_eq!(edge40_float.variance(), 0.0);
        let edge04_float = DMatrix::<f32>::zeros(0, 4);
        assert_eq!(edge04_float.variance(), 0.0);
    }
    #[test]
    fn test_row_variance() {
        let square_float = Matrix2::new(1.0, 2.0, 3.0, 4.0);
        assert_eq!(square_float.row_variance(), RowVector2::new(1.0, 1.0));
        let rect41_float = Matrix4x1::new(1.0, 2.0, 3.0, 4.0);
        assert_eq!(rect41_float.row_variance(), RowVector1::new(1.25));
        let rect14_float = Matrix1x4::new(1.0, 2.0, 3.0, 4.0);
        assert_eq!(
            rect14_float.row_variance(),
            RowVector4::new(0.0, 0.0, 0.0, 0.0)
        );
    }
    #[test]
    fn test_row_variance_edge() {
        let edge00_float = DMatrix::<f32>::zeros(0, 0);
        assert_eq!(edge00_float.row_variance(), RowDVector::<f32>::zeros(0));
        let edge40_float = DMatrix::<f32>::zeros(4, 0);
        assert_eq!(edge40_float.row_variance(), RowDVector::<f32>::zeros(0));
        let edge04_float = DMatrix::<f32>::zeros(0, 4);
        assert_eq!(edge04_float.row_variance(), RowDVector::<f32>::zeros(4));
    }
    #[test]
    fn test_row_variance_tr() {
        let square_float = Matrix2::new(1.0, 2.0, 3.0, 4.0);
        assert_eq!(square_float.row_variance_tr(), Vector2::new(1.0, 1.0));
        let rect41_float = Matrix4x1::new(1.0, 2.0, 3.0, 4.0);
        assert_eq!(rect41_float.row_variance_tr(), Vector1::new(1.25));
        let rect14_float = Matrix1x4::new(1.0, 2.0, 3.0, 4.0);
        assert_eq!(
            rect14_float.row_variance_tr(),
            Vector4::new(0.0, 0.0, 0.0, 0.0)
        );
    }
    #[test]
    fn test_row_variance_tr_edge() {
        let edge00_float = DMatrix::<f32>::zeros(0, 0);
        assert_eq!(edge00_float.row_variance_tr(), DVector::<f32>::zeros(0));
        let edge40_float = DMatrix::<f32>::zeros(4, 0);
        assert_eq!(edge40_float.row_variance_tr(), DVector::<f32>::zeros(0));
        let edge04_float = DMatrix::<f32>::zeros(0, 4);
        assert_eq!(edge04_float.row_variance_tr(), DVector::<f32>::zeros(4));
    }
    #[test]
    fn test_column_variance() {
        let square_float = Matrix2::new(1.0, 2.0, 3.0, 4.0);
        assert_eq!(square_float.column_variance(), Vector2::new(0.25, 0.25));
        let rect41_float = Matrix4x1::new(1.0, 2.0, 3.0, 4.0);
        assert_eq!(
            rect41_float.column_variance(),
            Vector4::new(0.0, 0.0, 0.0, 0.0)
        );
        let rect14_float = Matrix1x4::new(1.0, 2.0, 3.0, 4.0);
        assert_eq!(rect14_float.column_variance(), Vector1::new(1.25));
    }
    #[test]
    fn test_column_variance_edge() {
        let edge00_float = DMatrix::<f32>::zeros(0, 0);
        assert_eq!(edge00_float.column_variance(), DVector::<f32>::zeros(0));
        let edge40_float = DMatrix::<f32>::zeros(4, 0);
        assert_eq!(edge40_float.column_variance(), DVector::<f32>::zeros(4));
        let edge04_float = DMatrix::<f32>::zeros(0, 4);
        assert_eq!(edge04_float.column_variance(), DVector::<f32>::zeros(0));
    }
    #[test]
    fn test_mean() {
        let square_float = Matrix2::new(1.0, 2.0, 3.0, 4.0);
        assert_eq!(square_float.mean(), 2.5);
        let rect41_float = Matrix4x1::new(1.0, 2.0, 3.0, 4.0);
        assert_eq!(rect41_float.mean(), 2.5);
        let rect14_float = Matrix1x4::new(1.0, 2.0, 3.0, 4.0);
        assert_eq!(rect14_float.mean(), 2.5);
    }
    #[test]
    fn test_mean_edge() {
        let edge00_float = DMatrix::<f32>::zeros(0, 0);
        assert_eq!(edge00_float.mean(), 0.0);
        let edge40_float = DMatrix::<f32>::zeros(4, 0);
        assert_eq!(edge40_float.mean(), 0.0);
        let edge04_float = DMatrix::<f32>::zeros(0, 4);
        assert_eq!(edge04_float.mean(), 0.0);
    }
    #[test]
    fn test_row_mean() {
        let square_float = Matrix2::new(1.0, 2.0, 3.0, 4.0);
        assert_eq!(square_float.row_mean(), RowVector2::new(2.0, 3.0));
        let rect41_float = Matrix4x1::new(1.0, 2.0, 3.0, 4.0);
        assert_eq!(rect41_float.row_mean(), RowVector1::new(2.5));
        let rect14_float = Matrix1x4::new(1.0, 2.0, 3.0, 4.0);
        assert_eq!(rect14_float.row_mean(), RowVector4::new(1.0, 2.0, 3.0, 4.0));
    }
    #[test]
    fn test_row_mean_edge() {
        let edge00_float = DMatrix::<f32>::zeros(0, 0);
        assert_eq!(edge00_float.row_mean(), RowDVector::<f32>::zeros(0));
        let edge40_float = DMatrix::<f32>::zeros(4, 0);
        assert_eq!(edge40_float.row_mean(), RowDVector::<f32>::zeros(0));
        let edge04_float = DMatrix::<f32>::zeros(0, 4);
        assert_eq!(edge04_float.row_mean(), RowDVector::<f32>::zeros(4));
    }
    #[test]
    fn test_row_mean_tr() {
        let square_float = Matrix2::new(1.0, 2.0, 3.0, 4.0);
        assert_eq!(square_float.row_mean_tr(), Vector2::new(2.0, 3.0));
        let rect41_float = Matrix4x1::new(1.0, 2.0, 3.0, 4.0);
        assert_eq!(rect41_float.row_mean_tr(), Vector1::new(2.5));
        let rect14_float = Matrix1x4::new(1.0, 2.0, 3.0, 4.0);
        assert_eq!(rect14_float.row_mean_tr(), Vector4::new(1.0, 2.0, 3.0, 4.0));
    }
    #[test]
    fn test_row_mean_tr_edge() {
        let edge00_float = DMatrix::<f32>::zeros(0, 0);
        assert_eq!(edge00_float.row_mean_tr(), DVector::<f32>::zeros(0));
        let edge40_float = DMatrix::<f32>::zeros(4, 0);
        assert_eq!(edge40_float.row_mean_tr(), DVector::<f32>::zeros(0));
        let edge04_float = DMatrix::<f32>::zeros(0, 4);
        assert_eq!(edge04_float.row_mean_tr(), DVector::<f32>::zeros(4));
    }
    #[test]
    fn test_column_mean() {
        let square_float = Matrix2::new(1.0, 2.0, 3.0, 4.0);
        assert_eq!(square_float.column_mean(), Vector2::new(1.5, 3.5));
        let rect41_float = Matrix4x1::new(1.0, 2.0, 3.0, 4.0);
        assert_eq!(rect41_float.column_mean(), Vector4::new(1.0, 2.0, 3.0, 4.0));
        let rect14_float = Matrix1x4::new(1.0, 2.0, 3.0, 4.0);
        assert_eq!(rect14_float.column_mean(), Vector1::new(2.5));
    }
    #[test]
    fn test_column_mean_edge() {
        let edge00_float = DMatrix::<f32>::zeros(0, 0);
        assert_eq!(edge00_float.column_mean(), DVector::<f32>::zeros(0));
        let edge40_float = DMatrix::<f32>::zeros(4, 0);
        assert_eq!(edge40_float.column_mean(), DVector::<f32>::zeros(4));
        let edge04_float = DMatrix::<f32>::zeros(0, 4);
        assert_eq!(edge04_float.column_mean(), DVector::<f32>::zeros(0));
    }
}
