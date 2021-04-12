use crate::base::dimension::{Dynamic, U1, U2, U3, U4, U5, U6};
use crate::base::matrix_slice::{SliceStorage, SliceStorageMut};
use crate::base::{Const, Matrix};

/*
 *
 *
 * Matrix slice aliases.
 *
 *
 */
// NOTE: we can't provide defaults for the strides because it's not supported yet by min_const_generics.
/// A column-major matrix slice with dimensions known at compile-time.
///
/// **Because this is an alias, not all its methods are listed here. See the [`Matrix`](crate::base::Matrix) type too.**
pub type SMatrixSlice<'a, T, const R: usize, const C: usize> =
    Matrix<T, Const<R>, Const<C>, SliceStorage<'a, T, Const<R>, Const<C>, Const<1>, Const<R>>>;

/// A column-major matrix slice dynamic numbers of rows and columns.
///
/// **Because this is an alias, not all its methods are listed here. See the [`Matrix`](crate::base::Matrix) type too.**
pub type DMatrixSlice<'a, T, RStride = U1, CStride = Dynamic> =
    Matrix<T, Dynamic, Dynamic, SliceStorage<'a, T, Dynamic, Dynamic, RStride, CStride>>;

/// A column-major 1x1 matrix slice.
///
/// **Because this is an alias, not all its methods are listed here. See the [`Matrix`](crate::base::Matrix) type too.**
pub type MatrixSlice1<'a, T, RStride = U1, CStride = U1> =
    Matrix<T, U1, U1, SliceStorage<'a, T, U1, U1, RStride, CStride>>;
/// A column-major 2x2 matrix slice.
///
/// **Because this is an alias, not all its methods are listed here. See the [`Matrix`](crate::base::Matrix) type too.**
pub type MatrixSlice2<'a, T, RStride = U1, CStride = U2> =
    Matrix<T, U2, U2, SliceStorage<'a, T, U2, U2, RStride, CStride>>;
/// A column-major 3x3 matrix slice.
///
/// **Because this is an alias, not all its methods are listed here. See the [`Matrix`](crate::base::Matrix) type too.**
pub type MatrixSlice3<'a, T, RStride = U1, CStride = U3> =
    Matrix<T, U3, U3, SliceStorage<'a, T, U3, U3, RStride, CStride>>;
/// A column-major 4x4 matrix slice.
///
/// **Because this is an alias, not all its methods are listed here. See the [`Matrix`](crate::base::Matrix) type too.**
pub type MatrixSlice4<'a, T, RStride = U1, CStride = U4> =
    Matrix<T, U4, U4, SliceStorage<'a, T, U4, U4, RStride, CStride>>;
/// A column-major 5x5 matrix slice.
///
/// **Because this is an alias, not all its methods are listed here. See the [`Matrix`](crate::base::Matrix) type too.**
pub type MatrixSlice5<'a, T, RStride = U1, CStride = U5> =
    Matrix<T, U5, U5, SliceStorage<'a, T, U5, U5, RStride, CStride>>;
/// A column-major 6x6 matrix slice.
///
/// **Because this is an alias, not all its methods are listed here. See the [`Matrix`](crate::base::Matrix) type too.**
pub type MatrixSlice6<'a, T, RStride = U1, CStride = U6> =
    Matrix<T, U6, U6, SliceStorage<'a, T, U6, U6, RStride, CStride>>;

/// A column-major 1x2 matrix slice.
///
/// **Because this is an alias, not all its methods are listed here. See the [`Matrix`](crate::base::Matrix) type too.**
pub type MatrixSlice1x2<'a, T, RStride = U1, CStride = U1> =
    Matrix<T, U1, U2, SliceStorage<'a, T, U1, U2, RStride, CStride>>;
/// A column-major 1x3 matrix slice.
///
/// **Because this is an alias, not all its methods are listed here. See the [`Matrix`](crate::base::Matrix) type too.**
pub type MatrixSlice1x3<'a, T, RStride = U1, CStride = U1> =
    Matrix<T, U1, U3, SliceStorage<'a, T, U1, U3, RStride, CStride>>;
/// A column-major 1x4 matrix slice.
///
/// **Because this is an alias, not all its methods are listed here. See the [`Matrix`](crate::base::Matrix) type too.**
pub type MatrixSlice1x4<'a, T, RStride = U1, CStride = U1> =
    Matrix<T, U1, U4, SliceStorage<'a, T, U1, U4, RStride, CStride>>;
/// A column-major 1x5 matrix slice.
///
/// **Because this is an alias, not all its methods are listed here. See the [`Matrix`](crate::base::Matrix) type too.**
pub type MatrixSlice1x5<'a, T, RStride = U1, CStride = U1> =
    Matrix<T, U1, U5, SliceStorage<'a, T, U1, U5, RStride, CStride>>;
/// A column-major 1x6 matrix slice.
///
/// **Because this is an alias, not all its methods are listed here. See the [`Matrix`](crate::base::Matrix) type too.**
pub type MatrixSlice1x6<'a, T, RStride = U1, CStride = U1> =
    Matrix<T, U1, U6, SliceStorage<'a, T, U1, U6, RStride, CStride>>;

/// A column-major 2x1 matrix slice.
///
/// **Because this is an alias, not all its methods are listed here. See the [`Matrix`](crate::base::Matrix) type too.**
pub type MatrixSlice2x1<'a, T, RStride = U1, CStride = U2> =
    Matrix<T, U2, U1, SliceStorage<'a, T, U2, U1, RStride, CStride>>;
/// A column-major 2x3 matrix slice.
///
/// **Because this is an alias, not all its methods are listed here. See the [`Matrix`](crate::base::Matrix) type too.**
pub type MatrixSlice2x3<'a, T, RStride = U1, CStride = U2> =
    Matrix<T, U2, U3, SliceStorage<'a, T, U2, U3, RStride, CStride>>;
/// A column-major 2x4 matrix slice.
///
/// **Because this is an alias, not all its methods are listed here. See the [`Matrix`](crate::base::Matrix) type too.**
pub type MatrixSlice2x4<'a, T, RStride = U1, CStride = U2> =
    Matrix<T, U2, U4, SliceStorage<'a, T, U2, U4, RStride, CStride>>;
/// A column-major 2x5 matrix slice.
///
/// **Because this is an alias, not all its methods are listed here. See the [`Matrix`](crate::base::Matrix) type too.**
pub type MatrixSlice2x5<'a, T, RStride = U1, CStride = U2> =
    Matrix<T, U2, U5, SliceStorage<'a, T, U2, U5, RStride, CStride>>;
/// A column-major 2x6 matrix slice.
///
/// **Because this is an alias, not all its methods are listed here. See the [`Matrix`](crate::base::Matrix) type too.**
pub type MatrixSlice2x6<'a, T, RStride = U1, CStride = U2> =
    Matrix<T, U2, U6, SliceStorage<'a, T, U2, U6, RStride, CStride>>;

/// A column-major 3x1 matrix slice.
///
/// **Because this is an alias, not all its methods are listed here. See the [`Matrix`](crate::base::Matrix) type too.**
pub type MatrixSlice3x1<'a, T, RStride = U1, CStride = U3> =
    Matrix<T, U3, U1, SliceStorage<'a, T, U3, U1, RStride, CStride>>;
/// A column-major 3x2 matrix slice.
///
/// **Because this is an alias, not all its methods are listed here. See the [`Matrix`](crate::base::Matrix) type too.**
pub type MatrixSlice3x2<'a, T, RStride = U1, CStride = U3> =
    Matrix<T, U3, U2, SliceStorage<'a, T, U3, U2, RStride, CStride>>;
/// A column-major 3x4 matrix slice.
///
/// **Because this is an alias, not all its methods are listed here. See the [`Matrix`](crate::base::Matrix) type too.**
pub type MatrixSlice3x4<'a, T, RStride = U1, CStride = U3> =
    Matrix<T, U3, U4, SliceStorage<'a, T, U3, U4, RStride, CStride>>;
/// A column-major 3x5 matrix slice.
///
/// **Because this is an alias, not all its methods are listed here. See the [`Matrix`](crate::base::Matrix) type too.**
pub type MatrixSlice3x5<'a, T, RStride = U1, CStride = U3> =
    Matrix<T, U3, U5, SliceStorage<'a, T, U3, U5, RStride, CStride>>;
/// A column-major 3x6 matrix slice.
///
/// **Because this is an alias, not all its methods are listed here. See the [`Matrix`](crate::base::Matrix) type too.**
pub type MatrixSlice3x6<'a, T, RStride = U1, CStride = U3> =
    Matrix<T, U3, U6, SliceStorage<'a, T, U3, U6, RStride, CStride>>;

/// A column-major 4x1 matrix slice.
///
/// **Because this is an alias, not all its methods are listed here. See the [`Matrix`](crate::base::Matrix) type too.**
pub type MatrixSlice4x1<'a, T, RStride = U1, CStride = U4> =
    Matrix<T, U4, U1, SliceStorage<'a, T, U4, U1, RStride, CStride>>;
/// A column-major 4x2 matrix slice.
///
/// **Because this is an alias, not all its methods are listed here. See the [`Matrix`](crate::base::Matrix) type too.**
pub type MatrixSlice4x2<'a, T, RStride = U1, CStride = U4> =
    Matrix<T, U4, U2, SliceStorage<'a, T, U4, U2, RStride, CStride>>;
/// A column-major 4x3 matrix slice.
///
/// **Because this is an alias, not all its methods are listed here. See the [`Matrix`](crate::base::Matrix) type too.**
pub type MatrixSlice4x3<'a, T, RStride = U1, CStride = U4> =
    Matrix<T, U4, U3, SliceStorage<'a, T, U4, U3, RStride, CStride>>;
/// A column-major 4x5 matrix slice.
///
/// **Because this is an alias, not all its methods are listed here. See the [`Matrix`](crate::base::Matrix) type too.**
pub type MatrixSlice4x5<'a, T, RStride = U1, CStride = U4> =
    Matrix<T, U4, U5, SliceStorage<'a, T, U4, U5, RStride, CStride>>;
/// A column-major 4x6 matrix slice.
///
/// **Because this is an alias, not all its methods are listed here. See the [`Matrix`](crate::base::Matrix) type too.**
pub type MatrixSlice4x6<'a, T, RStride = U1, CStride = U4> =
    Matrix<T, U4, U6, SliceStorage<'a, T, U4, U6, RStride, CStride>>;

/// A column-major 5x1 matrix slice.
///
/// **Because this is an alias, not all its methods are listed here. See the [`Matrix`](crate::base::Matrix) type too.**
pub type MatrixSlice5x1<'a, T, RStride = U1, CStride = U5> =
    Matrix<T, U5, U1, SliceStorage<'a, T, U5, U1, RStride, CStride>>;
/// A column-major 5x2 matrix slice.
///
/// **Because this is an alias, not all its methods are listed here. See the [`Matrix`](crate::base::Matrix) type too.**
pub type MatrixSlice5x2<'a, T, RStride = U1, CStride = U5> =
    Matrix<T, U5, U2, SliceStorage<'a, T, U5, U2, RStride, CStride>>;
/// A column-major 5x3 matrix slice.
///
/// **Because this is an alias, not all its methods are listed here. See the [`Matrix`](crate::base::Matrix) type too.**
pub type MatrixSlice5x3<'a, T, RStride = U1, CStride = U5> =
    Matrix<T, U5, U3, SliceStorage<'a, T, U5, U3, RStride, CStride>>;
/// A column-major 5x4 matrix slice.
///
/// **Because this is an alias, not all its methods are listed here. See the [`Matrix`](crate::base::Matrix) type too.**
pub type MatrixSlice5x4<'a, T, RStride = U1, CStride = U5> =
    Matrix<T, U5, U4, SliceStorage<'a, T, U5, U4, RStride, CStride>>;
/// A column-major 5x6 matrix slice.
///
/// **Because this is an alias, not all its methods are listed here. See the [`Matrix`](crate::base::Matrix) type too.**
pub type MatrixSlice5x6<'a, T, RStride = U1, CStride = U5> =
    Matrix<T, U5, U6, SliceStorage<'a, T, U5, U6, RStride, CStride>>;

/// A column-major 6x1 matrix slice.
///
/// **Because this is an alias, not all its methods are listed here. See the [`Matrix`](crate::base::Matrix) type too.**
pub type MatrixSlice6x1<'a, T, RStride = U1, CStride = U6> =
    Matrix<T, U6, U1, SliceStorage<'a, T, U6, U1, RStride, CStride>>;
/// A column-major 6x2 matrix slice.
///
/// **Because this is an alias, not all its methods are listed here. See the [`Matrix`](crate::base::Matrix) type too.**
pub type MatrixSlice6x2<'a, T, RStride = U1, CStride = U6> =
    Matrix<T, U6, U2, SliceStorage<'a, T, U6, U2, RStride, CStride>>;
/// A column-major 6x3 matrix slice.
///
/// **Because this is an alias, not all its methods are listed here. See the [`Matrix`](crate::base::Matrix) type too.**
pub type MatrixSlice6x3<'a, T, RStride = U1, CStride = U6> =
    Matrix<T, U6, U3, SliceStorage<'a, T, U6, U3, RStride, CStride>>;
/// A column-major 6x4 matrix slice.
///
/// **Because this is an alias, not all its methods are listed here. See the [`Matrix`](crate::base::Matrix) type too.**
pub type MatrixSlice6x4<'a, T, RStride = U1, CStride = U6> =
    Matrix<T, U6, U4, SliceStorage<'a, T, U6, U4, RStride, CStride>>;
/// A column-major 6x5 matrix slice.
///
/// **Because this is an alias, not all its methods are listed here. See the [`Matrix`](crate::base::Matrix) type too.**
pub type MatrixSlice6x5<'a, T, RStride = U1, CStride = U6> =
    Matrix<T, U6, U5, SliceStorage<'a, T, U6, U5, RStride, CStride>>;

/// A column-major matrix slice with 1 row and a number of columns chosen at runtime.
pub type MatrixSlice1xX<'a, T, RStride = U1, CStride = U1> =
    Matrix<T, U1, Dynamic, SliceStorage<'a, T, U1, Dynamic, RStride, CStride>>;
/// A column-major matrix slice with 2 rows and a number of columns chosen at runtime.
pub type MatrixSlice2xX<'a, T, RStride = U1, CStride = U2> =
    Matrix<T, U2, Dynamic, SliceStorage<'a, T, U2, Dynamic, RStride, CStride>>;
/// A column-major matrix slice with 3 rows and a number of columns chosen at runtime.
pub type MatrixSlice3xX<'a, T, RStride = U1, CStride = U3> =
    Matrix<T, U3, Dynamic, SliceStorage<'a, T, U3, Dynamic, RStride, CStride>>;
/// A column-major matrix slice with 4 rows and a number of columns chosen at runtime.
pub type MatrixSlice4xX<'a, T, RStride = U1, CStride = U4> =
    Matrix<T, U4, Dynamic, SliceStorage<'a, T, U4, Dynamic, RStride, CStride>>;
/// A column-major matrix slice with 5 rows and a number of columns chosen at runtime.
pub type MatrixSlice5xX<'a, T, RStride = U1, CStride = U5> =
    Matrix<T, U5, Dynamic, SliceStorage<'a, T, U5, Dynamic, RStride, CStride>>;
/// A column-major matrix slice with 6 rows and a number of columns chosen at runtime.
pub type MatrixSlice6xX<'a, T, RStride = U1, CStride = U6> =
    Matrix<T, U6, Dynamic, SliceStorage<'a, T, U6, Dynamic, RStride, CStride>>;

/// A column-major matrix slice with a number of rows chosen at runtime and 1 column.
pub type MatrixSliceXx1<'a, T, RStride = U1, CStride = Dynamic> =
    Matrix<T, Dynamic, U1, SliceStorage<'a, T, Dynamic, U1, RStride, CStride>>;
/// A column-major matrix slice with a number of rows chosen at runtime and 2 columns.
pub type MatrixSliceXx2<'a, T, RStride = U1, CStride = Dynamic> =
    Matrix<T, Dynamic, U2, SliceStorage<'a, T, Dynamic, U2, RStride, CStride>>;
/// A column-major matrix slice with a number of rows chosen at runtime and 3 columns.
pub type MatrixSliceXx3<'a, T, RStride = U1, CStride = Dynamic> =
    Matrix<T, Dynamic, U3, SliceStorage<'a, T, Dynamic, U3, RStride, CStride>>;
/// A column-major matrix slice with a number of rows chosen at runtime and 4 columns.
pub type MatrixSliceXx4<'a, T, RStride = U1, CStride = Dynamic> =
    Matrix<T, Dynamic, U4, SliceStorage<'a, T, Dynamic, U4, RStride, CStride>>;
/// A column-major matrix slice with a number of rows chosen at runtime and 5 columns.
pub type MatrixSliceXx5<'a, T, RStride = U1, CStride = Dynamic> =
    Matrix<T, Dynamic, U5, SliceStorage<'a, T, Dynamic, U5, RStride, CStride>>;
/// A column-major matrix slice with a number of rows chosen at runtime and 6 columns.
pub type MatrixSliceXx6<'a, T, RStride = U1, CStride = Dynamic> =
    Matrix<T, Dynamic, U6, SliceStorage<'a, T, Dynamic, U6, RStride, CStride>>;

/// A column vector slice with dimensions known at compile-time.
///
/// **Because this is an alias, not all its methods are listed here. See the [`Matrix`](crate::base::Matrix) type too.**
pub type VectorSlice<'a, T, D, RStride = U1, CStride = D> =
    Matrix<T, D, U1, SliceStorage<'a, T, D, U1, RStride, CStride>>;

/// A column vector slice with dimensions known at compile-time.
///
/// **Because this is an alias, not all its methods are listed here. See the [`Matrix`](crate::base::Matrix) type too.**
pub type SVectorSlice<'a, T, const D: usize> =
    Matrix<T, Const<D>, Const<1>, SliceStorage<'a, T, Const<D>, Const<1>, Const<1>, Const<D>>>;

/// A column vector slice dynamic numbers of rows and columns.
pub type DVectorSlice<'a, T, RStride = U1, CStride = Dynamic> =
    Matrix<T, Dynamic, U1, SliceStorage<'a, T, Dynamic, U1, RStride, CStride>>;

/// A 1D column vector slice.
///
/// **Because this is an alias, not all its methods are listed here. See the [`Matrix`](crate::base::Matrix) type too.**
pub type VectorSlice1<'a, T, RStride = U1, CStride = U1> =
    Matrix<T, U1, U1, SliceStorage<'a, T, U1, U1, RStride, CStride>>;
/// A 2D column vector slice.
///
/// **Because this is an alias, not all its methods are listed here. See the [`Matrix`](crate::base::Matrix) type too.**
pub type VectorSlice2<'a, T, RStride = U1, CStride = U2> =
    Matrix<T, U2, U1, SliceStorage<'a, T, U2, U1, RStride, CStride>>;
/// A 3D column vector slice.
///
/// **Because this is an alias, not all its methods are listed here. See the [`Matrix`](crate::base::Matrix) type too.**
pub type VectorSlice3<'a, T, RStride = U1, CStride = U3> =
    Matrix<T, U3, U1, SliceStorage<'a, T, U3, U1, RStride, CStride>>;
/// A 4D column vector slice.
///
/// **Because this is an alias, not all its methods are listed here. See the [`Matrix`](crate::base::Matrix) type too.**
pub type VectorSlice4<'a, T, RStride = U1, CStride = U4> =
    Matrix<T, U4, U1, SliceStorage<'a, T, U4, U1, RStride, CStride>>;
/// A 5D column vector slice.
///
/// **Because this is an alias, not all its methods are listed here. See the [`Matrix`](crate::base::Matrix) type too.**
pub type VectorSlice5<'a, T, RStride = U1, CStride = U5> =
    Matrix<T, U5, U1, SliceStorage<'a, T, U5, U1, RStride, CStride>>;
/// A 6D column vector slice.
///
/// **Because this is an alias, not all its methods are listed here. See the [`Matrix`](crate::base::Matrix) type too.**
pub type VectorSlice6<'a, T, RStride = U1, CStride = U6> =
    Matrix<T, U6, U1, SliceStorage<'a, T, U6, U1, RStride, CStride>>;

/*
 *
 *
 * Same thing, but for mutable slices.
 *
 *
 */
/// A column-major matrix slice with `R` rows and `C` columns.
///
/// **Because this is an alias, not all its methods are listed here. See the [`Matrix`](crate::base::Matrix) type too.**
pub type MatrixSliceMutMN<'a, T, R, C, RStride = U1, CStride = R> =
    Matrix<T, R, C, SliceStorageMut<'a, T, R, C, RStride, CStride>>;

/// A column-major matrix slice with `D` rows and columns.
///
/// **Because this is an alias, not all its methods are listed here. See the [`Matrix`](crate::base::Matrix) type too.**
pub type MatrixSliceMutN<'a, T, D, RStride = U1, CStride = D> =
    Matrix<T, D, D, SliceStorageMut<'a, T, D, D, RStride, CStride>>;

/// A column-major matrix slice with dimensions known at compile-time.
///
/// **Because this is an alias, not all its methods are listed here. See the [`Matrix`](crate::base::Matrix) type too.**
pub type SMatrixSliceMut<'a, T, const R: usize, const C: usize> =
    Matrix<T, Const<R>, Const<C>, SliceStorageMut<'a, T, Const<R>, Const<C>, Const<1>, Const<R>>>;

/// A column-major matrix slice dynamic numbers of rows and columns.
///
/// **Because this is an alias, not all its methods are listed here. See the [`Matrix`](crate::base::Matrix) type too.**
pub type DMatrixSliceMut<'a, T, RStride = U1, CStride = Dynamic> =
    Matrix<T, Dynamic, Dynamic, SliceStorageMut<'a, T, Dynamic, Dynamic, RStride, CStride>>;

/// A column-major 1x1 matrix slice.
///
/// **Because this is an alias, not all its methods are listed here. See the [`Matrix`](crate::base::Matrix) type too.**
pub type MatrixSliceMut1<'a, T, RStride = U1, CStride = U1> =
    Matrix<T, U1, U1, SliceStorageMut<'a, T, U1, U1, RStride, CStride>>;
/// A column-major 2x2 matrix slice.
///
/// **Because this is an alias, not all its methods are listed here. See the [`Matrix`](crate::base::Matrix) type too.**
pub type MatrixSliceMut2<'a, T, RStride = U1, CStride = U2> =
    Matrix<T, U2, U2, SliceStorageMut<'a, T, U2, U2, RStride, CStride>>;
/// A column-major 3x3 matrix slice.
///
/// **Because this is an alias, not all its methods are listed here. See the [`Matrix`](crate::base::Matrix) type too.**
pub type MatrixSliceMut3<'a, T, RStride = U1, CStride = U3> =
    Matrix<T, U3, U3, SliceStorageMut<'a, T, U3, U3, RStride, CStride>>;
/// A column-major 4x4 matrix slice.
///
/// **Because this is an alias, not all its methods are listed here. See the [`Matrix`](crate::base::Matrix) type too.**
pub type MatrixSliceMut4<'a, T, RStride = U1, CStride = U4> =
    Matrix<T, U4, U4, SliceStorageMut<'a, T, U4, U4, RStride, CStride>>;
/// A column-major 5x5 matrix slice.
///
/// **Because this is an alias, not all its methods are listed here. See the [`Matrix`](crate::base::Matrix) type too.**
pub type MatrixSliceMut5<'a, T, RStride = U1, CStride = U5> =
    Matrix<T, U5, U5, SliceStorageMut<'a, T, U5, U5, RStride, CStride>>;
/// A column-major 6x6 matrix slice.
///
/// **Because this is an alias, not all its methods are listed here. See the [`Matrix`](crate::base::Matrix) type too.**
pub type MatrixSliceMut6<'a, T, RStride = U1, CStride = U6> =
    Matrix<T, U6, U6, SliceStorageMut<'a, T, U6, U6, RStride, CStride>>;

/// A column-major 1x2 matrix slice.
///
/// **Because this is an alias, not all its methods are listed here. See the [`Matrix`](crate::base::Matrix) type too.**
pub type MatrixSliceMut1x2<'a, T, RStride = U1, CStride = U1> =
    Matrix<T, U1, U2, SliceStorageMut<'a, T, U1, U2, RStride, CStride>>;
/// A column-major 1x3 matrix slice.
///
/// **Because this is an alias, not all its methods are listed here. See the [`Matrix`](crate::base::Matrix) type too.**
pub type MatrixSliceMut1x3<'a, T, RStride = U1, CStride = U1> =
    Matrix<T, U1, U3, SliceStorageMut<'a, T, U1, U3, RStride, CStride>>;
/// A column-major 1x4 matrix slice.
///
/// **Because this is an alias, not all its methods are listed here. See the [`Matrix`](crate::base::Matrix) type too.**
pub type MatrixSliceMut1x4<'a, T, RStride = U1, CStride = U1> =
    Matrix<T, U1, U4, SliceStorageMut<'a, T, U1, U4, RStride, CStride>>;
/// A column-major 1x5 matrix slice.
///
/// **Because this is an alias, not all its methods are listed here. See the [`Matrix`](crate::base::Matrix) type too.**
pub type MatrixSliceMut1x5<'a, T, RStride = U1, CStride = U1> =
    Matrix<T, U1, U5, SliceStorageMut<'a, T, U1, U5, RStride, CStride>>;
/// A column-major 1x6 matrix slice.
///
/// **Because this is an alias, not all its methods are listed here. See the [`Matrix`](crate::base::Matrix) type too.**
pub type MatrixSliceMut1x6<'a, T, RStride = U1, CStride = U1> =
    Matrix<T, U1, U6, SliceStorageMut<'a, T, U1, U6, RStride, CStride>>;

/// A column-major 2x1 matrix slice.
///
/// **Because this is an alias, not all its methods are listed here. See the [`Matrix`](crate::base::Matrix) type too.**
pub type MatrixSliceMut2x1<'a, T, RStride = U1, CStride = U2> =
    Matrix<T, U2, U1, SliceStorageMut<'a, T, U2, U1, RStride, CStride>>;
/// A column-major 2x3 matrix slice.
///
/// **Because this is an alias, not all its methods are listed here. See the [`Matrix`](crate::base::Matrix) type too.**
pub type MatrixSliceMut2x3<'a, T, RStride = U1, CStride = U2> =
    Matrix<T, U2, U3, SliceStorageMut<'a, T, U2, U3, RStride, CStride>>;
/// A column-major 2x4 matrix slice.
///
/// **Because this is an alias, not all its methods are listed here. See the [`Matrix`](crate::base::Matrix) type too.**
pub type MatrixSliceMut2x4<'a, T, RStride = U1, CStride = U2> =
    Matrix<T, U2, U4, SliceStorageMut<'a, T, U2, U4, RStride, CStride>>;
/// A column-major 2x5 matrix slice.
///
/// **Because this is an alias, not all its methods are listed here. See the [`Matrix`](crate::base::Matrix) type too.**
pub type MatrixSliceMut2x5<'a, T, RStride = U1, CStride = U2> =
    Matrix<T, U2, U5, SliceStorageMut<'a, T, U2, U5, RStride, CStride>>;
/// A column-major 2x6 matrix slice.
///
/// **Because this is an alias, not all its methods are listed here. See the [`Matrix`](crate::base::Matrix) type too.**
pub type MatrixSliceMut2x6<'a, T, RStride = U1, CStride = U2> =
    Matrix<T, U2, U6, SliceStorageMut<'a, T, U2, U6, RStride, CStride>>;

/// A column-major 3x1 matrix slice.
///
/// **Because this is an alias, not all its methods are listed here. See the [`Matrix`](crate::base::Matrix) type too.**
pub type MatrixSliceMut3x1<'a, T, RStride = U1, CStride = U3> =
    Matrix<T, U3, U1, SliceStorageMut<'a, T, U3, U1, RStride, CStride>>;
/// A column-major 3x2 matrix slice.
///
/// **Because this is an alias, not all its methods are listed here. See the [`Matrix`](crate::base::Matrix) type too.**
pub type MatrixSliceMut3x2<'a, T, RStride = U1, CStride = U3> =
    Matrix<T, U3, U2, SliceStorageMut<'a, T, U3, U2, RStride, CStride>>;
/// A column-major 3x4 matrix slice.
///
/// **Because this is an alias, not all its methods are listed here. See the [`Matrix`](crate::base::Matrix) type too.**
pub type MatrixSliceMut3x4<'a, T, RStride = U1, CStride = U3> =
    Matrix<T, U3, U4, SliceStorageMut<'a, T, U3, U4, RStride, CStride>>;
/// A column-major 3x5 matrix slice.
///
/// **Because this is an alias, not all its methods are listed here. See the [`Matrix`](crate::base::Matrix) type too.**
pub type MatrixSliceMut3x5<'a, T, RStride = U1, CStride = U3> =
    Matrix<T, U3, U5, SliceStorageMut<'a, T, U3, U5, RStride, CStride>>;
/// A column-major 3x6 matrix slice.
///
/// **Because this is an alias, not all its methods are listed here. See the [`Matrix`](crate::base::Matrix) type too.**
pub type MatrixSliceMut3x6<'a, T, RStride = U1, CStride = U3> =
    Matrix<T, U3, U6, SliceStorageMut<'a, T, U3, U6, RStride, CStride>>;

/// A column-major 4x1 matrix slice.
///
/// **Because this is an alias, not all its methods are listed here. See the [`Matrix`](crate::base::Matrix) type too.**
pub type MatrixSliceMut4x1<'a, T, RStride = U1, CStride = U4> =
    Matrix<T, U4, U1, SliceStorageMut<'a, T, U4, U1, RStride, CStride>>;
/// A column-major 4x2 matrix slice.
///
/// **Because this is an alias, not all its methods are listed here. See the [`Matrix`](crate::base::Matrix) type too.**
pub type MatrixSliceMut4x2<'a, T, RStride = U1, CStride = U4> =
    Matrix<T, U4, U2, SliceStorageMut<'a, T, U4, U2, RStride, CStride>>;
/// A column-major 4x3 matrix slice.
///
/// **Because this is an alias, not all its methods are listed here. See the [`Matrix`](crate::base::Matrix) type too.**
pub type MatrixSliceMut4x3<'a, T, RStride = U1, CStride = U4> =
    Matrix<T, U4, U3, SliceStorageMut<'a, T, U4, U3, RStride, CStride>>;
/// A column-major 4x5 matrix slice.
///
/// **Because this is an alias, not all its methods are listed here. See the [`Matrix`](crate::base::Matrix) type too.**
pub type MatrixSliceMut4x5<'a, T, RStride = U1, CStride = U4> =
    Matrix<T, U4, U5, SliceStorageMut<'a, T, U4, U5, RStride, CStride>>;
/// A column-major 4x6 matrix slice.
///
/// **Because this is an alias, not all its methods are listed here. See the [`Matrix`](crate::base::Matrix) type too.**
pub type MatrixSliceMut4x6<'a, T, RStride = U1, CStride = U4> =
    Matrix<T, U4, U6, SliceStorageMut<'a, T, U4, U6, RStride, CStride>>;

/// A column-major 5x1 matrix slice.
///
/// **Because this is an alias, not all its methods are listed here. See the [`Matrix`](crate::base::Matrix) type too.**
pub type MatrixSliceMut5x1<'a, T, RStride = U1, CStride = U5> =
    Matrix<T, U5, U1, SliceStorageMut<'a, T, U5, U1, RStride, CStride>>;
/// A column-major 5x2 matrix slice.
///
/// **Because this is an alias, not all its methods are listed here. See the [`Matrix`](crate::base::Matrix) type too.**
pub type MatrixSliceMut5x2<'a, T, RStride = U1, CStride = U5> =
    Matrix<T, U5, U2, SliceStorageMut<'a, T, U5, U2, RStride, CStride>>;
/// A column-major 5x3 matrix slice.
///
/// **Because this is an alias, not all its methods are listed here. See the [`Matrix`](crate::base::Matrix) type too.**
pub type MatrixSliceMut5x3<'a, T, RStride = U1, CStride = U5> =
    Matrix<T, U5, U3, SliceStorageMut<'a, T, U5, U3, RStride, CStride>>;
/// A column-major 5x4 matrix slice.
///
/// **Because this is an alias, not all its methods are listed here. See the [`Matrix`](crate::base::Matrix) type too.**
pub type MatrixSliceMut5x4<'a, T, RStride = U1, CStride = U5> =
    Matrix<T, U5, U4, SliceStorageMut<'a, T, U5, U4, RStride, CStride>>;
/// A column-major 5x6 matrix slice.
///
/// **Because this is an alias, not all its methods are listed here. See the [`Matrix`](crate::base::Matrix) type too.**
pub type MatrixSliceMut5x6<'a, T, RStride = U1, CStride = U5> =
    Matrix<T, U5, U6, SliceStorageMut<'a, T, U5, U6, RStride, CStride>>;

/// A column-major 6x1 matrix slice.
///
/// **Because this is an alias, not all its methods are listed here. See the [`Matrix`](crate::base::Matrix) type too.**
pub type MatrixSliceMut6x1<'a, T, RStride = U1, CStride = U6> =
    Matrix<T, U6, U1, SliceStorageMut<'a, T, U6, U1, RStride, CStride>>;
/// A column-major 6x2 matrix slice.
///
/// **Because this is an alias, not all its methods are listed here. See the [`Matrix`](crate::base::Matrix) type too.**
pub type MatrixSliceMut6x2<'a, T, RStride = U1, CStride = U6> =
    Matrix<T, U6, U2, SliceStorageMut<'a, T, U6, U2, RStride, CStride>>;
/// A column-major 6x3 matrix slice.
///
/// **Because this is an alias, not all its methods are listed here. See the [`Matrix`](crate::base::Matrix) type too.**
pub type MatrixSliceMut6x3<'a, T, RStride = U1, CStride = U6> =
    Matrix<T, U6, U3, SliceStorageMut<'a, T, U6, U3, RStride, CStride>>;
/// A column-major 6x4 matrix slice.
///
/// **Because this is an alias, not all its methods are listed here. See the [`Matrix`](crate::base::Matrix) type too.**
pub type MatrixSliceMut6x4<'a, T, RStride = U1, CStride = U6> =
    Matrix<T, U6, U4, SliceStorageMut<'a, T, U6, U4, RStride, CStride>>;
/// A column-major 6x5 matrix slice.
///
/// **Because this is an alias, not all its methods are listed here. See the [`Matrix`](crate::base::Matrix) type too.**
pub type MatrixSliceMut6x5<'a, T, RStride = U1, CStride = U6> =
    Matrix<T, U6, U5, SliceStorageMut<'a, T, U6, U5, RStride, CStride>>;

/// A column-major matrix slice with 1 row and a number of columns chosen at runtime.
pub type MatrixSliceMut1xX<'a, T, RStride = U1, CStride = U1> =
    Matrix<T, U1, Dynamic, SliceStorageMut<'a, T, U1, Dynamic, RStride, CStride>>;
/// A column-major matrix slice with 2 rows and a number of columns chosen at runtime.
pub type MatrixSliceMut2xX<'a, T, RStride = U1, CStride = U2> =
    Matrix<T, U2, Dynamic, SliceStorageMut<'a, T, U2, Dynamic, RStride, CStride>>;
/// A column-major matrix slice with 3 rows and a number of columns chosen at runtime.
pub type MatrixSliceMut3xX<'a, T, RStride = U1, CStride = U3> =
    Matrix<T, U3, Dynamic, SliceStorageMut<'a, T, U3, Dynamic, RStride, CStride>>;
/// A column-major matrix slice with 4 rows and a number of columns chosen at runtime.
pub type MatrixSliceMut4xX<'a, T, RStride = U1, CStride = U4> =
    Matrix<T, U4, Dynamic, SliceStorageMut<'a, T, U4, Dynamic, RStride, CStride>>;
/// A column-major matrix slice with 5 rows and a number of columns chosen at runtime.
pub type MatrixSliceMut5xX<'a, T, RStride = U1, CStride = U5> =
    Matrix<T, U5, Dynamic, SliceStorageMut<'a, T, U5, Dynamic, RStride, CStride>>;
/// A column-major matrix slice with 6 rows and a number of columns chosen at runtime.
pub type MatrixSliceMut6xX<'a, T, RStride = U1, CStride = U6> =
    Matrix<T, U6, Dynamic, SliceStorageMut<'a, T, U6, Dynamic, RStride, CStride>>;

/// A column-major matrix slice with a number of rows chosen at runtime and 1 column.
pub type MatrixSliceMutXx1<'a, T, RStride = U1, CStride = Dynamic> =
    Matrix<T, Dynamic, U1, SliceStorageMut<'a, T, Dynamic, U1, RStride, CStride>>;
/// A column-major matrix slice with a number of rows chosen at runtime and 2 columns.
pub type MatrixSliceMutXx2<'a, T, RStride = U1, CStride = Dynamic> =
    Matrix<T, Dynamic, U2, SliceStorageMut<'a, T, Dynamic, U2, RStride, CStride>>;
/// A column-major matrix slice with a number of rows chosen at runtime and 3 columns.
pub type MatrixSliceMutXx3<'a, T, RStride = U1, CStride = Dynamic> =
    Matrix<T, Dynamic, U3, SliceStorageMut<'a, T, Dynamic, U3, RStride, CStride>>;
/// A column-major matrix slice with a number of rows chosen at runtime and 4 columns.
pub type MatrixSliceMutXx4<'a, T, RStride = U1, CStride = Dynamic> =
    Matrix<T, Dynamic, U4, SliceStorageMut<'a, T, Dynamic, U4, RStride, CStride>>;
/// A column-major matrix slice with a number of rows chosen at runtime and 5 columns.
pub type MatrixSliceMutXx5<'a, T, RStride = U1, CStride = Dynamic> =
    Matrix<T, Dynamic, U5, SliceStorageMut<'a, T, Dynamic, U5, RStride, CStride>>;
/// A column-major matrix slice with a number of rows chosen at runtime and 6 columns.
pub type MatrixSliceMutXx6<'a, T, RStride = U1, CStride = Dynamic> =
    Matrix<T, Dynamic, U6, SliceStorageMut<'a, T, Dynamic, U6, RStride, CStride>>;

/// A column vector slice with dimensions known at compile-time.
///
/// **Because this is an alias, not all its methods are listed here. See the [`Matrix`](crate::base::Matrix) type too.**
pub type VectorSliceMut<'a, T, D, RStride = U1, CStride = D> =
    Matrix<T, D, U1, SliceStorageMut<'a, T, D, U1, RStride, CStride>>;

/// A column vector slice with dimensions known at compile-time.
///
/// **Because this is an alias, not all its methods are listed here. See the [`Matrix`](crate::base::Matrix) type too.**
pub type SVectorSliceMut<'a, T, const D: usize> =
    Matrix<T, Const<D>, Const<1>, SliceStorageMut<'a, T, Const<D>, Const<1>, Const<1>, Const<D>>>;

/// A column vector slice dynamic numbers of rows and columns.
///
/// **Because this is an alias, not all its methods are listed here. See the [`Matrix`](crate::base::Matrix) type too.**
pub type DVectorSliceMut<'a, T, RStride = U1, CStride = Dynamic> =
    Matrix<T, Dynamic, U1, SliceStorageMut<'a, T, Dynamic, U1, RStride, CStride>>;

/// A 1D column vector slice.
///
/// **Because this is an alias, not all its methods are listed here. See the [`Matrix`](crate::base::Matrix) type too.**
pub type VectorSliceMut1<'a, T, RStride = U1, CStride = U1> =
    Matrix<T, U1, U1, SliceStorageMut<'a, T, U1, U1, RStride, CStride>>;
/// A 2D column vector slice.
///
/// **Because this is an alias, not all its methods are listed here. See the [`Matrix`](crate::base::Matrix) type too.**
pub type VectorSliceMut2<'a, T, RStride = U1, CStride = U2> =
    Matrix<T, U2, U1, SliceStorageMut<'a, T, U2, U1, RStride, CStride>>;
/// A 3D column vector slice.
///
/// **Because this is an alias, not all its methods are listed here. See the [`Matrix`](crate::base::Matrix) type too.**
pub type VectorSliceMut3<'a, T, RStride = U1, CStride = U3> =
    Matrix<T, U3, U1, SliceStorageMut<'a, T, U3, U1, RStride, CStride>>;
/// A 4D column vector slice.
///
/// **Because this is an alias, not all its methods are listed here. See the [`Matrix`](crate::base::Matrix) type too.**
pub type VectorSliceMut4<'a, T, RStride = U1, CStride = U4> =
    Matrix<T, U4, U1, SliceStorageMut<'a, T, U4, U1, RStride, CStride>>;
/// A 5D column vector slice.
///
/// **Because this is an alias, not all its methods are listed here. See the [`Matrix`](crate::base::Matrix) type too.**
pub type VectorSliceMut5<'a, T, RStride = U1, CStride = U5> =
    Matrix<T, U5, U1, SliceStorageMut<'a, T, U5, U1, RStride, CStride>>;
/// A 6D column vector slice.
///
/// **Because this is an alias, not all its methods are listed here. See the [`Matrix`](crate::base::Matrix) type too.**
pub type VectorSliceMut6<'a, T, RStride = U1, CStride = U6> =
    Matrix<T, U6, U1, SliceStorageMut<'a, T, U6, U1, RStride, CStride>>;
