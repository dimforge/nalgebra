use core::MatrixArray;
use core::dimension::{U1, U4};

use geometry::{QuaternionBase, UnitQuaternionBase};

/// A statically-allocated quaternion.
pub type Quaternion<N> = QuaternionBase<N, MatrixArray<N, U4, U1>>;

/// A statically-allocated unit quaternion.
pub type UnitQuaternion<N> = UnitQuaternionBase<N, MatrixArray<N, U4, U1>>;
