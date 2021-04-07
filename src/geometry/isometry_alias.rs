use crate::geometry::{Isometry, Rotation2, Rotation3, UnitComplex, UnitQuaternion};

/// A 2-dimensional direct isometry using a unit complex number for its rotational part.
///
/// **Because this is an alias, not all its methods are listed here. See the [`Isometry`](crate::Isometry) type too.**
///
/// Also known as a 2D rigid-body motion, or as an element of SE(2).

pub type Isometry2<N> = Isometry<N, UnitComplex<N>, 2>;

/// A 3-dimensional direct isometry using a unit quaternion for its rotational part.
///
/// **Because this is an alias, not all its methods are listed here. See the [`Isometry`](crate::Isometry) type too.**
///
/// Also known as a rigid-body motion, or as an element of SE(3).
pub type Isometry3<N> = Isometry<N, UnitQuaternion<N>, 3>;

/// A 2-dimensional direct isometry using a rotation matrix for its rotational part.
///
/// **Because this is an alias, not all its methods are listed here. See the [`Isometry`](crate::Isometry) type too.**
///
/// Also known as a rigid-body motion, or as an element of SE(2).
pub type IsometryMatrix2<N> = Isometry<N, Rotation2<N>, 2>;

/// A 3-dimensional direct isometry using a rotation matrix for its rotational part.
///
/// **Because this is an alias, not all its methods are listed here. See the [`Isometry`](crate::Isometry) type too.**
///
/// Also known as a rigid-body motion, or as an element of SE(3).
pub type IsometryMatrix3<N> = Isometry<N, Rotation3<N>, 3>;

// This tests that the types correctly implement `Copy`, without having to run tests
// (when targeting no-std for example).
#[allow(dead_code)]
fn ensure_copy() {
    fn is_copy<T: Copy>() {}

    is_copy::<IsometryMatrix2<f32>>();
    is_copy::<IsometryMatrix3<f32>>();
    is_copy::<Isometry2<f32>>();
    is_copy::<Isometry3<f32>>();
}
