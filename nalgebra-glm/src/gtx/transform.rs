use na::{Real, Unit, Rotation2, Rotation3, Matrix3, Matrix4, U2, U3, U4};

use traits::Number;
use aliases::{Vec, Mat};

/// A rotation 4 * 4 matrix created from an axis of 3 scalars and an angle expressed in radians.
pub fn rotation<N: Real>(angle: N, v: &Vec<N, U3>) -> Mat<N, U4, U4> {
    Rotation3::from_axis_angle(&Unit::new_normalize(*v), angle).to_homogeneous()
}

/// A 4 * 4 scale matrix created from a vector of 3 components.
pub fn scaling<N: Number>(v: &Vec<N, U3>) -> Mat<N, U4, U4> {
    Matrix4::new_nonuniform_scaling(v)
}

/// A 4 * 4 translation matrix created from the scaling factor on each axis.
pub fn translation<N: Number>(v: &Vec<N, U3>) -> Mat<N, U4, U4> {
    Matrix4::new_translation(v)
}


/// A rotation 3 * 3 matrix created from an angle expressed in radians.
pub fn rotation2d<N: Real>(angle: N) -> Mat<N, U3, U3> {
    Rotation2::new(angle).to_homogeneous()
}

/// A 3 * 3 scale matrix created from a vector of 2 components.
pub fn scaling2d<N: Number>(v: &Vec<N, U2>) -> Mat<N, U3, U3> {
    Matrix3::new_nonuniform_scaling(v)
}

/// A 3 * 3 translation matrix created from the scaling factor on each axis.
pub fn translation2d<N: Number>(v: &Vec<N, U2>) -> Mat<N, U3, U3> {
    Matrix3::new_translation(v)
}
