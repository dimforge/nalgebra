use na::Real;

use aliases::TVec3;

/// The normal vector of the given triangle.
///
/// The normal is computed as the normalized vector `cross(p2 - p1, p3 - p1)`.
pub fn triangle_normal<N: Real>(p1: &TVec3<N>, p2: &TVec3<N>, p3: &TVec3<N>) -> TVec3<N> {
    (p2 - p1).cross(&(p3 - p1)).normalize()
}
