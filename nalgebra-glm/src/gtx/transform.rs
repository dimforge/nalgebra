use na::{RealField, Rotation2, Rotation3, Unit};

use crate::aliases::{TMat3, TMat4, TVec2, TVec3};
use crate::traits::Number;

/// A rotation 4 * 4 matrix created from an axis of 3 scalars and an angle expressed in radians.
///
/// # See also:
///
/// * [`scaling`](fn.scaling.html)
/// * [`translation`](fn.translation.html)
/// * [`rotation2d`](fn.rotation2d.html)
/// * [`scaling2d`](fn.scaling2d.html)
/// * [`translation2d`](fn.translation2d.html)
pub fn rotation<N: RealField>(angle: N, v: &TVec3<N>) -> TMat4<N> {
    Rotation3::from_axis_angle(&Unit::new_normalize(*v), angle).to_homogeneous()
}

/// A 4 * 4 scale matrix created from a vector of 3 components.
///
/// # See also:
///
/// * [`rotation`](fn.rotation.html)
/// * [`translation`](fn.translation.html)
/// * [`rotation2d`](fn.rotation2d.html)
/// * [`scaling2d`](fn.scaling2d.html)
/// * [`translation2d`](fn.translation2d.html)
pub fn scaling<N: Number>(v: &TVec3<N>) -> TMat4<N> {
    TMat4::new_nonuniform_scaling(v)
}

/// A 4 * 4 translation matrix created from the scaling factor on each axis.
///
/// # See also:
///
/// * [`rotation`](fn.rotation.html)
/// * [`scaling`](fn.scaling.html)
/// * [`rotation2d`](fn.rotation2d.html)
/// * [`scaling2d`](fn.scaling2d.html)
/// * [`translation2d`](fn.translation2d.html)
pub fn translation<N: Number>(v: &TVec3<N>) -> TMat4<N> {
    TMat4::new_translation(v)
}

/// A rotation 3 * 3 matrix created from an angle expressed in radians.
///
/// # See also:
///
/// * [`rotation`](fn.rotation.html)
/// * [`scaling`](fn.scaling.html)
/// * [`translation`](fn.translation.html)
/// * [`scaling2d`](fn.scaling2d.html)
/// * [`translation2d`](fn.translation2d.html)
pub fn rotation2d<N: RealField>(angle: N) -> TMat3<N> {
    Rotation2::new(angle).to_homogeneous()
}

/// A 3 * 3 scale matrix created from a vector of 2 components.
///
/// # See also:
///
/// * [`rotation`](fn.rotation.html)
/// * [`scaling`](fn.scaling.html)
/// * [`translation`](fn.translation.html)
/// * [`rotation2d`](fn.rotation2d.html)
/// * [`translation2d`](fn.translation2d.html)
pub fn scaling2d<N: Number>(v: &TVec2<N>) -> TMat3<N> {
    TMat3::new_nonuniform_scaling(v)
}

/// A 3 * 3 translation matrix created from the scaling factor on each axis.
///
/// # See also:
///
/// * [`rotation`](fn.rotation.html)
/// * [`scaling`](fn.scaling.html)
/// * [`translation`](fn.translation.html)
/// * [`rotation2d`](fn.rotation2d.html)
/// * [`scaling2d`](fn.scaling2d.html)
pub fn translation2d<N: Number>(v: &TVec2<N>) -> TMat3<N> {
    TMat3::new_translation(v)
}
