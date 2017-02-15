use core::MatrixArray;
use core::dimension::{U1, U2, U3};

use geometry::{Rotation, IsometryBase, UnitQuaternion, UnitComplex};

/// A D-dimensional isometry.
pub type Isometry<N, D> = IsometryBase<N, D, MatrixArray<N, D, U1>, Rotation<N, D>>;

/// A 2-dimensional isometry using a unit complex number for its rotational part.
pub type Isometry2<N> = IsometryBase<N, U2, MatrixArray<N, U2, U1>, UnitComplex<N>>;

/// A 3-dimensional isometry using a unit quaternion for its rotational part.
pub type Isometry3<N> = IsometryBase<N, U3, MatrixArray<N, U3, U1>, UnitQuaternion<N>>;

/// A 2-dimensional isometry using a rotation matrix for its rotation part.
pub type IsometryMatrix2<N> = Isometry<N, U2>;

/// A 3-dimensional isometry using a rotation matrix for its rotation part.
pub type IsometryMatrix3<N> = Isometry<N, U3>;
