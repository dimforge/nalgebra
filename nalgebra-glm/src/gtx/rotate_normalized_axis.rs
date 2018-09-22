use na::{Real, Rotation3, Unit, U3, U4};

use aliases::{Vec, Mat};

/// Builds a rotation 4 * 4 matrix created from a normalized axis and an angle.
///
/// # Parameters
///     * `m` - Input matrix multiplied by this rotation matrix.
///     * `angle` - Rotation angle expressed in radians.
///     * `axis` - Rotation axis, must be normalized.
pub fn rotate_normalized_axis<N: Real>(m: &Mat<N, U4, U4>, angle: N, axis: &Vec<N, U3>) -> Mat<N, U4, U4> {
    m * Rotation3::from_axis_angle(&Unit::new_unchecked(*axis), angle).to_homogeneous()
}