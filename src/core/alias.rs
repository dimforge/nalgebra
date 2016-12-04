use core::Matrix;
use core::dimension::{Dynamic, U1, U2, U3, U4, U5, U6};
use core::matrix_array::MatrixArray;
use core::matrix_vec::MatrixVec;

/*
 *
 *
 * Column-major matrices.
 *
 *
 */
/// A dynamically sized column-major matrix.
pub type DMatrix<N> = Matrix<N, Dynamic, Dynamic, MatrixVec<N, Dynamic, Dynamic>>;

/// A staticaly sized column-major matrix with `R` rows and `C` columns.
pub type MatrixNM<N, R, C> = Matrix<N, R, C, MatrixArray<N, R, C>>;

/// A staticaly sized column-major square matrix with `D` rows and columns.
pub type MatrixN<N, D> = MatrixNM<N, D, D>;

pub type Matrix1<N> = MatrixN<N, U1>;
pub type Matrix2<N> = MatrixN<N, U2>;
pub type Matrix3<N> = MatrixN<N, U3>;
pub type Matrix4<N> = MatrixN<N, U4>;
pub type Matrix5<N> = MatrixN<N, U5>;
pub type Matrix6<N> = MatrixN<N, U6>;

pub type Matrix1x2<N> = MatrixNM<N, U1, U2>;
pub type Matrix1x3<N> = MatrixNM<N, U1, U3>;
pub type Matrix1x4<N> = MatrixNM<N, U1, U4>;
pub type Matrix1x5<N> = MatrixNM<N, U1, U5>;
pub type Matrix1x6<N> = MatrixNM<N, U1, U6>;

pub type Matrix2x3<N> = MatrixNM<N, U2, U3>;
pub type Matrix2x4<N> = MatrixNM<N, U2, U4>;
pub type Matrix2x5<N> = MatrixNM<N, U2, U5>;
pub type Matrix2x6<N> = MatrixNM<N, U2, U6>;

pub type Matrix3x4<N> = MatrixNM<N, U3, U4>;
pub type Matrix3x5<N> = MatrixNM<N, U3, U5>;
pub type Matrix3x6<N> = MatrixNM<N, U3, U6>;

pub type Matrix4x5<N> = MatrixNM<N, U4, U5>;
pub type Matrix4x6<N> = MatrixNM<N, U4, U6>;

pub type Matrix5x6<N> = MatrixNM<N, U5, U6>;


pub type Matrix2x1<N> = MatrixNM<N, U2, U1>;
pub type Matrix3x1<N> = MatrixNM<N, U3, U1>;
pub type Matrix4x1<N> = MatrixNM<N, U4, U1>;
pub type Matrix5x1<N> = MatrixNM<N, U5, U1>;
pub type Matrix6x1<N> = MatrixNM<N, U6, U1>;

pub type Matrix3x2<N> = MatrixNM<N, U3, U2>;
pub type Matrix4x2<N> = MatrixNM<N, U4, U2>;
pub type Matrix5x2<N> = MatrixNM<N, U5, U2>;
pub type Matrix6x2<N> = MatrixNM<N, U6, U2>;

pub type Matrix4x3<N> = MatrixNM<N, U4, U3>;
pub type Matrix5x3<N> = MatrixNM<N, U5, U3>;
pub type Matrix6x3<N> = MatrixNM<N, U6, U3>;

pub type Matrix5x4<N> = MatrixNM<N, U5, U4>;
pub type Matrix6x4<N> = MatrixNM<N, U6, U4>;

pub type Matrix6x5<N> = MatrixNM<N, U6, U5>;


/*
 *
 *
 * Column vectors.
 *
 *
 */
/// A dynamically sized column vector.
pub type DVector<N> = Matrix<N, Dynamic, U1, MatrixVec<N, Dynamic, U1>>;

/// A statically sized D-dimensional column vector.
pub type VectorN<N, D> = MatrixNM<N, D, U1>;

pub type Vector1<N> = VectorN<N, U1>;
pub type Vector2<N> = VectorN<N, U2>;
pub type Vector3<N> = VectorN<N, U3>;
pub type Vector4<N> = VectorN<N, U4>;
pub type Vector5<N> = VectorN<N, U5>;
pub type Vector6<N> = VectorN<N, U6>;


/*
 *
 *
 * Row vectors.
 *
 *
 */
/// A dynamically sized row vector.
pub type RowDVector<N> = Matrix<N, U1, Dynamic, MatrixVec<N, U1, Dynamic>>;

/// A statically sized D-dimensional row vector.
pub type RowVectorN<N, D> = MatrixNM<N, U1, D>;

pub type RowVector1<N> = RowVectorN<N, U1>;
pub type RowVector2<N> = RowVectorN<N, U2>;
pub type RowVector3<N> = RowVectorN<N, U3>;
pub type RowVector4<N> = RowVectorN<N, U4>;
pub type RowVector5<N> = RowVectorN<N, U5>;
pub type RowVector6<N> = RowVectorN<N, U6>;
