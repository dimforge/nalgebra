use na::{Point3, Rotation3, Unit};

use crate::aliases::{TMat, TMat4, TVec, TVec3};
use crate::traits::{Number, RealNumber};

/// The identity matrix.
pub fn identity<T: Number, const D: usize>() -> TMat<T, D, D> {
    TMat::<T, D, D>::identity()
}

/// Build a look at view matrix based on the right handedness.
///
/// # Parameters:
///
/// * `eye` − Position of the camera.
/// * `center` − Position where the camera is looking at.
/// * `u` − Normalized up vector, how the camera is oriented. Typically `(0, 1, 0)`.
///
/// # See also:
///
/// * [`look_at_lh`](fn.look_at_lh.html)
/// * [`look_at_rh`](fn.look_at_rh.html)
pub fn look_at<T: RealNumber>(eye: &TVec3<T>, center: &TVec3<T>, up: &TVec3<T>) -> TMat4<T> {
    look_at_rh(eye, center, up)
}

/// Build a left handed look at view matrix.
///
/// # Parameters:
///
/// * `eye` − Position of the camera.
/// * `center` − Position where the camera is looking at.
/// * `u` − Normalized up vector, how the camera is oriented. Typically `(0, 1, 0)`.
///
/// # See also:
///
/// * [`look_at`](fn.look_at.html)
/// * [`look_at_rh`](fn.look_at_rh.html)
pub fn look_at_lh<T: RealNumber>(eye: &TVec3<T>, center: &TVec3<T>, up: &TVec3<T>) -> TMat4<T> {
    TMat::look_at_lh(&Point3::from(*eye), &Point3::from(*center), up)
}

/// Build a right handed look at view matrix.
///
/// # Parameters:
///
/// * `eye` − Position of the camera.
/// * `center` − Position where the camera is looking at.
/// * `u` − Normalized up vector, how the camera is oriented. Typically `(0, 1, 0)`.
///
/// # See also:
///
/// * [`look_at`](fn.look_at.html)
/// * [`look_at_lh`](fn.look_at_lh.html)
pub fn look_at_rh<T: RealNumber>(eye: &TVec3<T>, center: &TVec3<T>, up: &TVec3<T>) -> TMat4<T> {
    TMat::look_at_rh(&Point3::from(*eye), &Point3::from(*center), up)
}

/// Builds a rotation 4 * 4 matrix created from an axis vector and an angle and right-multiply it to `m`.
///
/// # Parameters:
///
/// * `m` − Input matrix multiplied by this rotation matrix.
/// * `angle` − Rotation angle expressed in radians.
/// * `axis` − Rotation axis, recommended to be normalized.
///
/// # See also:
///
/// * [`rotate_x`](fn.rotate_x.html)
/// * [`rotate_y`](fn.rotate_y.html)
/// * [`rotate_z`](fn.rotate_z.html)
/// * [`scale`](fn.scale.html)
/// * [`translate`](fn.translate.html)
pub fn rotate<T: RealNumber>(m: &TMat4<T>, angle: T, axis: &TVec3<T>) -> TMat4<T> {
    m * Rotation3::from_axis_angle(&Unit::new_normalize(*axis), angle).to_homogeneous()
}

/// Builds a rotation 4 * 4 matrix around the X axis and right-multiply it to `m`.
///
/// # Parameters:
///
/// * `m` − Input matrix multiplied by this rotation matrix.
/// * `angle` − Rotation angle expressed in radians.
///
/// # See also:
///
/// * [`rotate`](fn.rotate.html)
/// * [`rotate_y`](fn.rotate_y.html)
/// * [`rotate_z`](fn.rotate_z.html)
/// * [`scale`](fn.scale.html)
/// * [`translate`](fn.translate.html)
pub fn rotate_x<T: RealNumber>(m: &TMat4<T>, angle: T) -> TMat4<T> {
    rotate(m, angle, &TVec::x())
}

/// Builds a rotation 4 * 4 matrix around the Y axis and right-multiply it to `m`.
///
/// # Parameters:
///
/// * `m` − Input matrix multiplied by this rotation matrix.
/// * `angle` − Rotation angle expressed in radians.
///
/// # See also:
///
/// * [`rotate`](fn.rotate.html)
/// * [`rotate_x`](fn.rotate_x.html)
/// * [`rotate_z`](fn.rotate_z.html)
/// * [`scale`](fn.scale.html)
/// * [`translate`](fn.translate.html)
pub fn rotate_y<T: RealNumber>(m: &TMat4<T>, angle: T) -> TMat4<T> {
    rotate(m, angle, &TVec::y())
}

/// Builds a rotation 4 * 4 matrix around the Z axis and right-multiply it to `m`.
///
/// # Parameters:
///
/// * `m` − Input matrix multiplied by this rotation matrix.
/// * `angle` − Rotation angle expressed in radians.
///
/// # See also:
///
/// * [`rotate`](fn.rotate.html)
/// * [`rotate_x`](fn.rotate_x.html)
/// * [`rotate_y`](fn.rotate_y.html)
/// * [`scale`](fn.scale.html)
/// * [`translate`](fn.translate.html)
pub fn rotate_z<T: RealNumber>(m: &TMat4<T>, angle: T) -> TMat4<T> {
    rotate(m, angle, &TVec::z())
}

/// Builds a scale 4 * 4 matrix created from 3 scalars and right-multiply it to `m`.
///
/// # Parameters:
///
/// * `m` − Input matrix multiplied by this scale matrix.
/// * `v` − Ratio of scaling for each axis.
///
/// # See also:
///
/// * [`rotate`](fn.rotate.html)
/// * [`rotate_x`](fn.rotate_x.html)
/// * [`rotate_y`](fn.rotate_y.html)
/// * [`rotate_z`](fn.rotate_z.html)
/// * [`translate`](fn.translate.html)
pub fn scale<T: Number>(m: &TMat4<T>, v: &TVec3<T>) -> TMat4<T> {
    m.prepend_nonuniform_scaling(v)
}

/// Builds a translation 4 * 4 matrix created from a vector of 3 components and right-multiply it to `m`.
///
/// # Parameters:
///
/// * `m` − Input matrix multiplied by this translation matrix.
/// * `v` − Coordinates of a translation vector.
///
/// # See also:
///
/// * [`rotate`](fn.rotate.html)
/// * [`rotate_x`](fn.rotate_x.html)
/// * [`rotate_y`](fn.rotate_y.html)
/// * [`rotate_z`](fn.rotate_z.html)
/// * [`scale`](fn.scale.html)
pub fn translate<T: Number>(m: &TMat4<T>, v: &TVec3<T>) -> TMat4<T> {
    m.prepend_translation(v)
}
