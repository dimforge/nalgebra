use na::{Real, Unit, Rotation2, Rotation3};

use traits::Number;
use aliases::{TVec3, TVec2, TMat3, TMat4};

/// A rotation 4 * 4 matrix created from an axis of 3 scalars and an angle expressed in radians.
pub fn rotation<N: Real>(angle: N, v: &TVec3<N>) -> TMat4<N> {
    Rotation3::from_axis_angle(&Unit::new_normalize(*v), angle).to_homogeneous()
}

/// A 4 * 4 scale matrix created from a vector of 3 components.
pub fn scaling<N: Number>(v: &TVec3<N>) -> TMat4<N> {
    TMat4::new_nonuniform_scaling(v)
}

/// A 4 * 4 translation matrix created from the scaling factor on each axis.
pub fn translation<N: Number>(v: &TVec3<N>) -> TMat4<N> {
    TMat4::new_translation(v)
}


/// A rotation 3 * 3 matrix created from an angle expressed in radians.
pub fn rotation2d<N: Real>(angle: N) -> TMat3<N> {
    Rotation2::new(angle).to_homogeneous()
}

/// A 3 * 3 scale matrix created from a vector of 2 components.
pub fn scaling2d<N: Number>(v: &TVec2<N>) -> TMat3<N> {
    TMat3::new_nonuniform_scaling(v)
}

/// A 3 * 3 translation matrix created from the scaling factor on each axis.
pub fn translation2d<N: Number>(v: &TVec2<N>) -> TMat3<N> {
    TMat3::new_translation(v)
}
