use na::{Real, Unit, Rotation3, Matrix4, U3, U4};

use traits::Number;
use aliases::{Vec, Mat};

/// Builds a rotation 4 * 4 matrix created from an axis of 3 scalars and an angle expressed in radians.
pub fn rotate<N: Real>(angle: N, v: &Vec<N, U3>) -> Mat<N, U4, U4> {
    Rotation3::from_axis_angle(&Unit::new_normalize(*v), angle).to_homogeneous()
}

/// Transforms a matrix with a scale 4 * 4 matrix created from a vector of 3 components.
pub fn scale<N: Number>(v: &Vec<N, U3>) -> Mat<N, U4, U4> {
    Matrix4::new_nonuniform_scaling(v)
}

/// Transforms a matrix with a translation 4 * 4 matrix created from 3 scalars.
pub fn translate<N: Number>(v: &Vec<N, U3>) -> Mat<N, U4, U4> {
    Matrix4::new_translation(v)
}
