use core::MatrixArray;
use core::dimension::{U2, U3};

use geometry::RotationBase;

/// A D-dimensional rotation matrix.
pub type Rotation<N, D> = RotationBase<N, D, MatrixArray<N, D, D>>;

/// A 2-dimensional rotation matrix.
pub type Rotation2<N> = Rotation<N, U2>;

/// A 3-dimensional rotation matrix.
pub type Rotation3<N> = Rotation<N, U3>;
