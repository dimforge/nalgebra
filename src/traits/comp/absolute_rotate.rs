/// Trait of matrices having the following operation:
///
///     self.absolute_rotate(v) = self.rotation_matrix().absolute().rmul(v)
///
/// The operation is accessible using the `RotationMatrix`, `Absolute`, and `RMul` traits, but
/// doing so is not easy in generic code as it can be a cause of type over-parametrization.
///
/// # Known use case:
///     * to compute efficiently the AABB of a rotated AABB.
pub trait AbsoluteRotate<V> {

    /// This is the same as:
    ///
    ///     self.absolute_rotate(v) = self.rotation_matrix().absolute().rmul(v)
    fn absolute_rotate(&self, &V) -> V;
}
