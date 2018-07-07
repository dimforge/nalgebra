use base::dimension::{U2, U3};

use geometry::{Isometry, Rotation2, Rotation3, UnitComplex, UnitQuaternion};

/// A 2-dimensional direct isometry using a unit complex number for its rotational part.
pub type Isometry2<N> = Isometry<N, U2, UnitComplex<N>>;

/// A 3-dimensional direct isometry using a unit quaternion for its rotational part.
pub type Isometry3<N> = Isometry<N, U3, UnitQuaternion<N>>;

/// A 2-dimensional direct isometry using a rotation matrix for its rotational part.
pub type IsometryMatrix2<N> = Isometry<N, U2, Rotation2<N>>;

/// A 3-dimensional direct isometry using a rotation matrix for its rotational part.
pub type IsometryMatrix3<N> = Isometry<N, U3, Rotation3<N>>;
