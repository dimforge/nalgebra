use na::{DefaultAllocator, Real, U3, U4, Unit, Rotation3, Point3};

use traits::{Dimension, Number, Alloc};
use aliases::{Mat, Vec};

/// The identity matrix.
pub fn identity<N: Number, D: Dimension>() -> Mat<N, D, D>
    where DefaultAllocator: Alloc<N, D, D> {
    Mat::<N, D, D>::identity()
}

/// Build a look at view matrix based on the right handedness.
///
/// # Parameters
///    * `eye`   − Position of the camera
///    * `center` − Position where the camera is looking at
///    * `u` − Normalized up vector, how the camera is oriented. Typically `(0, 1, 0)`
pub fn look_at<N: Real>(eye: &Vec<N, U3>, center: &Vec<N, U3>, up: &Vec<N, U3>) -> Mat<N, U4, U4> {
    look_at_rh(eye, center, up)
}

/// Build a left handed look at view matrix.
///
/// # Parameters
///    * `eye`   − Position of the camera
///    * `center` − Position where the camera is looking at
///    * `u` − Normalized up vector, how the camera is oriented. Typically `(0, 1, 0)`
pub fn look_at_lh<N: Real>(eye: &Vec<N, U3>, center: &Vec<N, U3>, up: &Vec<N, U3>) -> Mat<N, U4, U4> {
    Mat::look_at_lh(&Point3::from_coordinates(*eye), &Point3::from_coordinates(*center), up)
}

/// Build a right handed look at view matrix.
///
/// # Parameters
///    * `eye`   − Position of the camera
///    * `center` − Position where the camera is looking at
///    * `u` − Normalized up vector, how the camera is oriented. Typically `(0, 1, 0)`
pub fn look_at_rh<N: Real>(eye: &Vec<N, U3>, center: &Vec<N, U3>, up: &Vec<N, U3>) -> Mat<N, U4, U4> {
    Mat::look_at_rh(&Point3::from_coordinates(*eye), &Point3::from_coordinates(*center), up)
}

/// Builds a rotation 4 * 4 matrix created from an axis vector and an angle.
///
/// # Parameters
///    * m − Input matrix multiplied by this rotation matrix.
///    * angle − Rotation angle expressed in radians.
///    * axis  − Rotation axis, recommended to be normalized.
pub fn rotate<N: Real>(m: &Mat<N, U4, U4>, angle: N, axis: &Vec<N, U3>) -> Mat<N, U4, U4> {
    m * Rotation3::from_axis_angle(&Unit::new_normalize(*axis), angle).to_homogeneous()
}

/// Builds a scale 4 * 4 matrix created from 3 scalars.
///
/// # Parameters
///    * m − Input matrix multiplied by this scale matrix.
///    * v − Ratio of scaling for each axis.
pub fn scale<N: Number>(m: &Mat<N, U4, U4>, v: &Vec<N, U3>) -> Mat<N, U4, U4> {
    m.prepend_nonuniform_scaling(v)
}

/// Builds a translation 4 * 4 matrix created from a vector of 3 components.
///
/// # Parameters
///    * m − Input matrix multiplied by this translation matrix.
///    * v − Coordinates of a translation vector.
pub fn translate<N: Number>(m: &Mat<N, U4, U4>, v: &Vec<N, U3>) -> Mat<N, U4, U4> {
    m.prepend_translation(v)
}
