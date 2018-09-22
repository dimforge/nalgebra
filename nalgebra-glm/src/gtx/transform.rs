use na::{Real, Unit, Rotation3, Matrix4, U3, U4};

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
