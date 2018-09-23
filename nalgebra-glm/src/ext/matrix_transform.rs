use na::{DefaultAllocator, Real, Unit, Rotation3, Point3};

use traits::{Dimension, Number, Alloc};
use aliases::{TMat, TVec, TVec3, TMat4};

/// The identity matrix.
pub fn identity<N: Number, D: Dimension>() -> TMat<N, D, D>
    where DefaultAllocator: Alloc<N, D, D> {
    TMat::<N, D, D>::identity()
}

/// Build a look at view matrix based on the right handedness.
///
/// # Parameters
///    * `eye`   − Position of the camera
///    * `center` − Position where the camera is looking at
///    * `u` − Normalized up vector, how the camera is oriented. Typically `(0, 1, 0)`
pub fn look_at<N: Real>(eye: &TVec3<N>, center: &TVec3<N>, up: &TVec3<N>) -> TMat4<N> {
    look_at_rh(eye, center, up)
}

/// Build a left handed look at view matrix.
///
/// # Parameters
///    * `eye`   − Position of the camera
///    * `center` − Position where the camera is looking at
///    * `u` − Normalized up vector, how the camera is oriented. Typically `(0, 1, 0)`
pub fn look_at_lh<N: Real>(eye: &TVec3<N>, center: &TVec3<N>, up: &TVec3<N>) -> TMat4<N> {
    TMat::look_at_lh(&Point3::from_coordinates(*eye), &Point3::from_coordinates(*center), up)
}

/// Build a right handed look at view matrix.
///
/// # Parameters
///    * `eye`   − Position of the camera
///    * `center` − Position where the camera is looking at
///    * `u` − Normalized up vector, how the camera is oriented. Typically `(0, 1, 0)`
pub fn look_at_rh<N: Real>(eye: &TVec3<N>, center: &TVec3<N>, up: &TVec3<N>) -> TMat4<N> {
    TMat::look_at_rh(&Point3::from_coordinates(*eye), &Point3::from_coordinates(*center), up)
}

/// Builds a rotation 4 * 4 matrix created from an axis vector and an angle and right-multiply it to `m`.
///
/// # Parameters
///    * m − Input matrix multiplied by this rotation matrix.
///    * angle − Rotation angle expressed in radians.
///    * axis  − Rotation axis, recommended to be normalized.
pub fn rotate<N: Real>(m: &TMat4<N>, angle: N, axis: &TVec3<N>) -> TMat4<N> {
    m * Rotation3::from_axis_angle(&Unit::new_normalize(*axis), angle).to_homogeneous()
}

/// Builds a rotation 4 * 4 matrix around the X axis and right-multiply it to `m`.
///
/// # Parameters
///    * m − Input matrix multiplied by this rotation matrix.
///    * angle − Rotation angle expressed in radians.
pub fn rotate_x<N: Real>(m: &TMat4<N>, angle: N) -> TMat4<N> {
    rotate(m, angle, &TVec::x())
}

/// Builds a rotation 4 * 4 matrix around the Y axis and right-multiply it to `m`.
///
/// # Parameters
///    * m − Input matrix multiplied by this rotation matrix.
///    * angle − Rotation angle expressed in radians.
pub fn rotate_y<N: Real>(m: &TMat4<N>, angle: N) -> TMat4<N> {
    rotate(m, angle, &TVec::y())
}

/// Builds a rotation 4 * 4 matrix around the Z axis and right-multiply it to `m`.
///
/// # Parameters
///    * m − Input matrix multiplied by this rotation matrix.
///    * angle − Rotation angle expressed in radians.
pub fn rotate_z<N: Real>(m: &TMat4<N>, angle: N) -> TMat4<N> {
    rotate(m, angle, &TVec::z())
}

/// Builds a scale 4 * 4 matrix created from 3 scalars and right-multiply it to `m`.
///
/// # Parameters
///    * m − Input matrix multiplied by this scale matrix.
///    * v − Ratio of scaling for each axis.
pub fn scale<N: Number>(m: &TMat4<N>, v: &TVec3<N>) -> TMat4<N> {
    m.prepend_nonuniform_scaling(v)
}

/// Builds a translation 4 * 4 matrix created from a vector of 3 components and right-multiply it to `m`.
///
/// # Parameters
///    * m − Input matrix multiplied by this translation matrix.
///    * v − Coordinates of a translation vector.
pub fn translate<N: Number>(m: &TMat4<N>, v: &TVec3<N>) -> TMat4<N> {
    m.prepend_translation(v)
}
