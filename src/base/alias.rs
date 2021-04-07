#[cfg(any(feature = "alloc", feature = "std"))]
use crate::base::dimension::Dynamic;
use crate::base::dimension::{U1, U2, U3, U4, U5, U6};
use crate::base::storage::Owned;
#[cfg(any(feature = "std", feature = "alloc"))]
use crate::base::vec_storage::VecStorage;
use crate::base::{Const, Matrix, Unit};

/*
 *
 *
 * Column-major matrices.
 *
 *
 */
/// A statically sized column-major matrix with `R` rows and `C` columns.
///
/// **Because this is an alias, not all its methods are listed here. See the [`Matrix`](crate::base::Matrix) type too.**
#[deprecated(note = "This matrix name contains a typo. Use MatrixMN instead.")]
pub type MatrixNM<N, R, C> = Matrix<N, R, C, Owned<N, R, C>>;

/// A statically sized column-major matrix with `R` rows and `C` columns.
///
/// **Because this is an alias, not all its methods are listed here. See the [`Matrix`](crate::base::Matrix) type too.**
pub type MatrixMN<N, R, C> = Matrix<N, R, C, Owned<N, R, C>>;
pub type CMatrixMN<N, const R: usize, const C: usize> =
    Matrix<N, Const<R>, Const<C>, Owned<N, Const<R>, Const<C>>>;

/// A statically sized column-major square matrix with `D` rows and columns.
///
/// **Because this is an alias, not all its methods are listed here. See the [`Matrix`](crate::base::Matrix) type too.**
pub type MatrixN<N, D> = Matrix<N, D, D, Owned<N, D, D>>;
pub type CMatrixN<N, const D: usize> = Matrix<N, Const<D>, Const<D>, Owned<N, Const<D>, Const<D>>>;

/// A dynamically sized column-major matrix.
///
/// **Because this is an alias, not all its methods are listed here. See the [`Matrix`](crate::base::Matrix) type too.**
#[cfg(any(feature = "std", feature = "alloc"))]
pub type DMatrix<N> = Matrix<N, Dynamic, Dynamic, Owned<N, Dynamic, Dynamic>>;

/// A heap-allocated, column-major, matrix with a dynamic number of rows and 1 columns.
///
/// **Because this is an alias, not all its methods are listed here. See the [`Matrix`](crate::base::Matrix) type too.**
#[cfg(any(feature = "std", feature = "alloc"))]
pub type MatrixXx1<N> = Matrix<N, Dynamic, U1, Owned<N, Dynamic, U1>>;
/// A heap-allocated, column-major, matrix with a dynamic number of rows and 2 columns.
///
/// **Because this is an alias, not all its methods are listed here. See the [`Matrix`](crate::base::Matrix) type too.**
#[cfg(any(feature = "std", feature = "alloc"))]
pub type MatrixXx2<N> = Matrix<N, Dynamic, U2, Owned<N, Dynamic, U2>>;
/// A heap-allocated, column-major, matrix with a dynamic number of rows and 3 columns.
///
/// **Because this is an alias, not all its methods are listed here. See the [`Matrix`](crate::base::Matrix) type too.**
#[cfg(any(feature = "std", feature = "alloc"))]
pub type MatrixXx3<N> = Matrix<N, Dynamic, U3, Owned<N, Dynamic, U3>>;
/// A heap-allocated, column-major, matrix with a dynamic number of rows and 4 columns.
///
/// **Because this is an alias, not all its methods are listed here. See the [`Matrix`](crate::base::Matrix) type too.**
#[cfg(any(feature = "std", feature = "alloc"))]
pub type MatrixXx4<N> = Matrix<N, Dynamic, U4, Owned<N, Dynamic, U4>>;
/// A heap-allocated, column-major, matrix with a dynamic number of rows and 5 columns.
///
/// **Because this is an alias, not all its methods are listed here. See the [`Matrix`](crate::base::Matrix) type too.**
#[cfg(any(feature = "std", feature = "alloc"))]
pub type MatrixXx5<N> = Matrix<N, Dynamic, U5, Owned<N, Dynamic, U5>>;
/// A heap-allocated, column-major, matrix with a dynamic number of rows and 6 columns.
///
/// **Because this is an alias, not all its methods are listed here. See the [`Matrix`](crate::base::Matrix) type too.**
#[cfg(any(feature = "std", feature = "alloc"))]
pub type MatrixXx6<N> = Matrix<N, Dynamic, U6, Owned<N, Dynamic, U6>>;

/// A heap-allocated, row-major, matrix with 1 rows and a dynamic number of columns.
///
/// **Because this is an alias, not all its methods are listed here. See the [`Matrix`](crate::base::Matrix) type too.**
#[cfg(any(feature = "std", feature = "alloc"))]
pub type Matrix1xX<N> = Matrix<N, U1, Dynamic, Owned<N, U1, Dynamic>>;
/// A heap-allocated, row-major, matrix with 2 rows and a dynamic number of columns.
///
/// **Because this is an alias, not all its methods are listed here. See the [`Matrix`](crate::base::Matrix) type too.**
#[cfg(any(feature = "std", feature = "alloc"))]
pub type Matrix2xX<N> = Matrix<N, U2, Dynamic, Owned<N, U2, Dynamic>>;
/// A heap-allocated, row-major, matrix with 3 rows and a dynamic number of columns.
///
/// **Because this is an alias, not all its methods are listed here. See the [`Matrix`](crate::base::Matrix) type too.**
#[cfg(any(feature = "std", feature = "alloc"))]
pub type Matrix3xX<N> = Matrix<N, U3, Dynamic, Owned<N, U3, Dynamic>>;
/// A heap-allocated, row-major, matrix with 4 rows and a dynamic number of columns.
///
/// **Because this is an alias, not all its methods are listed here. See the [`Matrix`](crate::base::Matrix) type too.**
#[cfg(any(feature = "std", feature = "alloc"))]
pub type Matrix4xX<N> = Matrix<N, U4, Dynamic, Owned<N, U4, Dynamic>>;
/// A heap-allocated, row-major, matrix with 5 rows and a dynamic number of columns.
///
/// **Because this is an alias, not all its methods are listed here. See the [`Matrix`](crate::base::Matrix) type too.**
#[cfg(any(feature = "std", feature = "alloc"))]
pub type Matrix5xX<N> = Matrix<N, U5, Dynamic, Owned<N, U5, Dynamic>>;
/// A heap-allocated, row-major, matrix with 6 rows and a dynamic number of columns.
///
/// **Because this is an alias, not all its methods are listed here. See the [`Matrix`](crate::base::Matrix) type too.**
#[cfg(any(feature = "std", feature = "alloc"))]
pub type Matrix6xX<N> = Matrix<N, U6, Dynamic, Owned<N, U6, Dynamic>>;

/// A stack-allocated, column-major, 1x1 square matrix.
///
/// **Because this is an alias, not all its methods are listed here. See the [`Matrix`](crate::base::Matrix) type too.**
pub type Matrix1<N> = Matrix<N, U1, U1, Owned<N, U1, U1>>;
/// A stack-allocated, column-major, 2x2 square matrix.
///
/// **Because this is an alias, not all its methods are listed here. See the [`Matrix`](crate::base::Matrix) type too.**
pub type Matrix2<N> = Matrix<N, U2, U2, Owned<N, U2, U2>>;
/// A stack-allocated, column-major, 3x3 square matrix.
///
/// **Because this is an alias, not all its methods are listed here. See the [`Matrix`](crate::base::Matrix) type too.**
pub type Matrix3<N> = Matrix<N, U3, U3, Owned<N, U3, U3>>;
/// A stack-allocated, column-major, 4x4 square matrix.
///
/// **Because this is an alias, not all its methods are listed here. See the [`Matrix`](crate::base::Matrix) type too.**
pub type Matrix4<N> = Matrix<N, U4, U4, Owned<N, U4, U4>>;
/// A stack-allocated, column-major, 5x5 square matrix.
///
/// **Because this is an alias, not all its methods are listed here. See the [`Matrix`](crate::base::Matrix) type too.**
pub type Matrix5<N> = Matrix<N, U5, U5, Owned<N, U5, U5>>;
/// A stack-allocated, column-major, 6x6 square matrix.
///
/// **Because this is an alias, not all its methods are listed here. See the [`Matrix`](crate::base::Matrix) type too.**
pub type Matrix6<N> = Matrix<N, U6, U6, Owned<N, U6, U6>>;

/// A stack-allocated, column-major, 1x2 matrix.
///
/// **Because this is an alias, not all its methods are listed here. See the [`Matrix`](crate::base::Matrix) type too.**
pub type Matrix1x2<N> = Matrix<N, U1, U2, Owned<N, U1, U2>>;
/// A stack-allocated, column-major, 1x3 matrix.
///
/// **Because this is an alias, not all its methods are listed here. See the [`Matrix`](crate::base::Matrix) type too.**
pub type Matrix1x3<N> = Matrix<N, U1, U3, Owned<N, U1, U3>>;
/// A stack-allocated, column-major, 1x4 matrix.
///
/// **Because this is an alias, not all its methods are listed here. See the [`Matrix`](crate::base::Matrix) type too.**
pub type Matrix1x4<N> = Matrix<N, U1, U4, Owned<N, U1, U4>>;
/// A stack-allocated, column-major, 1x5 matrix.
///
/// **Because this is an alias, not all its methods are listed here. See the [`Matrix`](crate::base::Matrix) type too.**
pub type Matrix1x5<N> = Matrix<N, U1, U5, Owned<N, U1, U5>>;
/// A stack-allocated, column-major, 1x6 matrix.
///
/// **Because this is an alias, not all its methods are listed here. See the [`Matrix`](crate::base::Matrix) type too.**
pub type Matrix1x6<N> = Matrix<N, U1, U6, Owned<N, U1, U6>>;

/// A stack-allocated, column-major, 2x3 matrix.
///
/// **Because this is an alias, not all its methods are listed here. See the [`Matrix`](crate::base::Matrix) type too.**
pub type Matrix2x3<N> = Matrix<N, U2, U3, Owned<N, U2, U3>>;
/// A stack-allocated, column-major, 2x4 matrix.
///
/// **Because this is an alias, not all its methods are listed here. See the [`Matrix`](crate::base::Matrix) type too.**
pub type Matrix2x4<N> = Matrix<N, U2, U4, Owned<N, U2, U4>>;
/// A stack-allocated, column-major, 2x5 matrix.
///
/// **Because this is an alias, not all its methods are listed here. See the [`Matrix`](crate::base::Matrix) type too.**
pub type Matrix2x5<N> = Matrix<N, U2, U5, Owned<N, U2, U5>>;
/// A stack-allocated, column-major, 2x6 matrix.
///
/// **Because this is an alias, not all its methods are listed here. See the [`Matrix`](crate::base::Matrix) type too.**
pub type Matrix2x6<N> = Matrix<N, U2, U6, Owned<N, U2, U6>>;

/// A stack-allocated, column-major, 3x4 matrix.
///
/// **Because this is an alias, not all its methods are listed here. See the [`Matrix`](crate::base::Matrix) type too.**
pub type Matrix3x4<N> = Matrix<N, U3, U4, Owned<N, U3, U4>>;
/// A stack-allocated, column-major, 3x5 matrix.
///
/// **Because this is an alias, not all its methods are listed here. See the [`Matrix`](crate::base::Matrix) type too.**
pub type Matrix3x5<N> = Matrix<N, U3, U5, Owned<N, U3, U5>>;
/// A stack-allocated, column-major, 3x6 matrix.
///
/// **Because this is an alias, not all its methods are listed here. See the [`Matrix`](crate::base::Matrix) type too.**
pub type Matrix3x6<N> = Matrix<N, U3, U6, Owned<N, U3, U6>>;

/// A stack-allocated, column-major, 4x5 matrix.
///
/// **Because this is an alias, not all its methods are listed here. See the [`Matrix`](crate::base::Matrix) type too.**
pub type Matrix4x5<N> = Matrix<N, U4, U5, Owned<N, U4, U5>>;
/// A stack-allocated, column-major, 4x6 matrix.
///
/// **Because this is an alias, not all its methods are listed here. See the [`Matrix`](crate::base::Matrix) type too.**
pub type Matrix4x6<N> = Matrix<N, U4, U6, Owned<N, U4, U6>>;

/// A stack-allocated, column-major, 5x6 matrix.
///
/// **Because this is an alias, not all its methods are listed here. See the [`Matrix`](crate::base::Matrix) type too.**
pub type Matrix5x6<N> = Matrix<N, U5, U6, Owned<N, U5, U6>>;

/// A stack-allocated, column-major, 2x1 matrix.
///
/// **Because this is an alias, not all its methods are listed here. See the [`Matrix`](crate::base::Matrix) type too.**
pub type Matrix2x1<N> = Matrix<N, U2, U1, Owned<N, U2, U1>>;
/// A stack-allocated, column-major, 3x1 matrix.
///
/// **Because this is an alias, not all its methods are listed here. See the [`Matrix`](crate::base::Matrix) type too.**
pub type Matrix3x1<N> = Matrix<N, U3, U1, Owned<N, U3, U1>>;
/// A stack-allocated, column-major, 4x1 matrix.
///
/// **Because this is an alias, not all its methods are listed here. See the [`Matrix`](crate::base::Matrix) type too.**
pub type Matrix4x1<N> = Matrix<N, U4, U1, Owned<N, U4, U1>>;
/// A stack-allocated, column-major, 5x1 matrix.
///
/// **Because this is an alias, not all its methods are listed here. See the [`Matrix`](crate::base::Matrix) type too.**
pub type Matrix5x1<N> = Matrix<N, U5, U1, Owned<N, U5, U1>>;
/// A stack-allocated, column-major, 6x1 matrix.
///
/// **Because this is an alias, not all its methods are listed here. See the [`Matrix`](crate::base::Matrix) type too.**
pub type Matrix6x1<N> = Matrix<N, U6, U1, Owned<N, U6, U1>>;

/// A stack-allocated, column-major, 3x2 matrix.
///
/// **Because this is an alias, not all its methods are listed here. See the [`Matrix`](crate::base::Matrix) type too.**
pub type Matrix3x2<N> = Matrix<N, U3, U2, Owned<N, U3, U2>>;
/// A stack-allocated, column-major, 4x2 matrix.
///
/// **Because this is an alias, not all its methods are listed here. See the [`Matrix`](crate::base::Matrix) type too.**
pub type Matrix4x2<N> = Matrix<N, U4, U2, Owned<N, U4, U2>>;
/// A stack-allocated, column-major, 5x2 matrix.
///
/// **Because this is an alias, not all its methods are listed here. See the [`Matrix`](crate::base::Matrix) type too.**
pub type Matrix5x2<N> = Matrix<N, U5, U2, Owned<N, U5, U2>>;
/// A stack-allocated, column-major, 6x2 matrix.
///
/// **Because this is an alias, not all its methods are listed here. See the [`Matrix`](crate::base::Matrix) type too.**
pub type Matrix6x2<N> = Matrix<N, U6, U2, Owned<N, U6, U2>>;

/// A stack-allocated, column-major, 4x3 matrix.
///
/// **Because this is an alias, not all its methods are listed here. See the [`Matrix`](crate::base::Matrix) type too.**
pub type Matrix4x3<N> = Matrix<N, U4, U3, Owned<N, U4, U3>>;
/// A stack-allocated, column-major, 5x3 matrix.
///
/// **Because this is an alias, not all its methods are listed here. See the [`Matrix`](crate::base::Matrix) type too.**
pub type Matrix5x3<N> = Matrix<N, U5, U3, Owned<N, U5, U3>>;
/// A stack-allocated, column-major, 6x3 matrix.
///
/// **Because this is an alias, not all its methods are listed here. See the [`Matrix`](crate::base::Matrix) type too.**
pub type Matrix6x3<N> = Matrix<N, U6, U3, Owned<N, U6, U3>>;

/// A stack-allocated, column-major, 5x4 matrix.
///
/// **Because this is an alias, not all its methods are listed here. See the [`Matrix`](crate::base::Matrix) type too.**
pub type Matrix5x4<N> = Matrix<N, U5, U4, Owned<N, U5, U4>>;
/// A stack-allocated, column-major, 6x4 matrix.
///
/// **Because this is an alias, not all its methods are listed here. See the [`Matrix`](crate::base::Matrix) type too.**
pub type Matrix6x4<N> = Matrix<N, U6, U4, Owned<N, U6, U4>>;

/// A stack-allocated, column-major, 6x5 matrix.
///
/// **Because this is an alias, not all its methods are listed here. See the [`Matrix`](crate::base::Matrix) type too.**
pub type Matrix6x5<N> = Matrix<N, U6, U5, Owned<N, U6, U5>>;

/*
 *
 *
 * Column vectors.
 *
 *
 */
/// A dynamically sized column vector.
#[cfg(any(feature = "std", feature = "alloc"))]
pub type DVector<N> = Matrix<N, Dynamic, U1, VecStorage<N, Dynamic, U1>>;

/// A statically sized D-dimensional column vector.
pub type VectorN<N, D> = Matrix<N, D, U1, Owned<N, D, U1>>;
pub type CVectorN<N, const D: usize> = Matrix<N, Const<D>, U1, Owned<N, Const<D>, U1>>;

/// A stack-allocated, 1-dimensional column vector.
pub type Vector1<N> = Matrix<N, U1, U1, Owned<N, U1, U1>>;
/// A stack-allocated, 2-dimensional column vector.
pub type Vector2<N> = Matrix<N, U2, U1, Owned<N, U2, U1>>;
/// A stack-allocated, 3-dimensional column vector.
pub type Vector3<N> = Matrix<N, U3, U1, Owned<N, U3, U1>>;
/// A stack-allocated, 4-dimensional column vector.
pub type Vector4<N> = Matrix<N, U4, U1, Owned<N, U4, U1>>;
/// A stack-allocated, 5-dimensional column vector.
pub type Vector5<N> = Matrix<N, U5, U1, Owned<N, U5, U1>>;
/// A stack-allocated, 6-dimensional column vector.
pub type Vector6<N> = Matrix<N, U6, U1, Owned<N, U6, U1>>;

/*
 *
 *
 * Row vectors.
 *
 *
 */
/// A dynamically sized row vector.
#[cfg(any(feature = "std", feature = "alloc"))]
pub type RowDVector<N> = Matrix<N, U1, Dynamic, VecStorage<N, U1, Dynamic>>;

/// A statically sized D-dimensional row vector.
pub type RowVectorN<N, D> = Matrix<N, U1, D, Owned<N, U1, D>>;

/// A stack-allocated, 1-dimensional row vector.
pub type RowVector1<N> = Matrix<N, U1, U1, Owned<N, U1, U1>>;
/// A stack-allocated, 2-dimensional row vector.
pub type RowVector2<N> = Matrix<N, U1, U2, Owned<N, U1, U2>>;
/// A stack-allocated, 3-dimensional row vector.
pub type RowVector3<N> = Matrix<N, U1, U3, Owned<N, U1, U3>>;
/// A stack-allocated, 4-dimensional row vector.
pub type RowVector4<N> = Matrix<N, U1, U4, Owned<N, U1, U4>>;
/// A stack-allocated, 5-dimensional row vector.
pub type RowVector5<N> = Matrix<N, U1, U5, Owned<N, U1, U5>>;
/// A stack-allocated, 6-dimensional row vector.
pub type RowVector6<N> = Matrix<N, U1, U6, Owned<N, U1, U6>>;

/*
 *
 *
 * Unit Vector.
 *
 *
 */
/// A stack-allocated, 1-dimensional unit vector.
pub type UnitVector1<N> = Unit<Matrix<N, U1, U1, Owned<N, U1, U1>>>;
/// A stack-allocated, 2-dimensional unit vector.
pub type UnitVector2<N> = Unit<Matrix<N, U2, U1, Owned<N, U2, U1>>>;
/// A stack-allocated, 3-dimensional unit vector.
pub type UnitVector3<N> = Unit<Matrix<N, U3, U1, Owned<N, U3, U1>>>;
/// A stack-allocated, 4-dimensional unit vector.
pub type UnitVector4<N> = Unit<Matrix<N, U4, U1, Owned<N, U4, U1>>>;
/// A stack-allocated, 5-dimensional unit vector.
pub type UnitVector5<N> = Unit<Matrix<N, U5, U1, Owned<N, U5, U1>>>;
/// A stack-allocated, 6-dimensional unit vector.
pub type UnitVector6<N> = Unit<Matrix<N, U6, U1, Owned<N, U6, U1>>>;
