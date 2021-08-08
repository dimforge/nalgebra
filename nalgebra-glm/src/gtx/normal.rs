use crate::RealNumber;

use crate::aliases::TVec3;

/// The normal vector of the given triangle.
///
/// The normal is computed as the normalized vector `cross(p2 - p1, p3 - p1)`.
pub fn triangle_normal<T: RealNumber>(p1: &TVec3<T>, p2: &TVec3<T>, p3: &TVec3<T>) -> TVec3<T> {
    (p2 - p1).cross(&(p3 - p1)).normalize()
}
