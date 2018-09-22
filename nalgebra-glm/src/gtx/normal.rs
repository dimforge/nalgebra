use na::{Real, U3};

use aliases::Vec;

/// The normal vector of the given triangle.
///
/// The normal is computed as the normalized vector `cross(p2 - p1, p3 - p1)`.
pub fn triangle_normal<N: Real>(p1: &Vec<N, U3>, p2: &Vec<N, U3>, p3: &Vec<N, U3>) -> Vec<N, U3> {
    (p2 - p1).cross(&(p3 - p1)).normalize()
}