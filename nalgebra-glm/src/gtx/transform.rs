use na::{Rotation2, Rotation3, Unit};

use crate::aliases::{TMat3, TMat4, TVec2, TVec3};
use crate::traits::{Number, RealNumber};

/// A rotation 4 * 4 matrix created from an axis of 3 scalars and an angle expressed in radians.
///
/// # See also:
///
/// * [`scaling()`]
/// * [`translation()`]
/// * [`rotation2d()`]
/// * [`scaling2d()`]
/// * [`translation2d()`]
pub fn rotation<T: RealNumber>(angle: T, v: &TVec3<T>) -> TMat4<T> {
    Rotation3::from_axis_angle(&Unit::new_normalize(*v), angle).to_homogeneous()
}

/// A 4 * 4 scale matrix created from a vector of 3 components.
///
/// # See also:
///
/// * [`rotation()`]
/// * [`translation()`]
/// * [`rotation2d()`]
/// * [`scaling2d()`]
/// * [`translation2d()`]
pub fn scaling<T: Number>(v: &TVec3<T>) -> TMat4<T> {
    TMat4::new_nonuniform_scaling(v)
}

/// A 4 * 4 translation matrix created from the scaling factor on each axis.
///
/// # See also:
///
/// * [`rotation()`]
/// * [`scaling()`]
/// * [`rotation2d()`]
/// * [`scaling2d()`]
/// * [`translation2d()`]
pub fn translation<T: Number>(v: &TVec3<T>) -> TMat4<T> {
    TMat4::new_translation(v)
}

/// A rotation 3 * 3 matrix created from an angle expressed in radians.
///
/// # See also:
///
/// * [`rotation()`]
/// * [`scaling()`]
/// * [`translation()`]
/// * [`scaling2d()`]
/// * [`translation2d()`]
pub fn rotation2d<T: RealNumber>(angle: T) -> TMat3<T> {
    Rotation2::new(angle).to_homogeneous()
}

/// A 3 * 3 scale matrix created from a vector of 2 components.
///
/// # See also:
///
/// * [`rotation()`]
/// * [`scaling()`]
/// * [`translation()`]
/// * [`rotation2d()`]
/// * [`translation2d()`]
pub fn scaling2d<T: Number>(v: &TVec2<T>) -> TMat3<T> {
    TMat3::new_nonuniform_scaling(v)
}

/// A 3 * 3 translation matrix created from the scaling factor on each axis.
///
/// # See also:
///
/// * [`rotation()`]
/// * [`scaling()`]
/// * [`translation()`]
/// * [`rotation2d()`]
/// * [`scaling2d()`]
pub fn translation2d<T: Number>(v: &TVec2<T>) -> TMat3<T> {
    TMat3::new_translation(v)
}
