/**
 * Trait of elements having a cross product.
 */
pub trait Cross<V> {
    /// Computes the cross product between two elements (usually vectors).
    fn cross(&self, other: &Self) -> V;
}

/**
 * Trait of elements having a cross product operation which can be expressed as a matrix.
 */
pub trait CrossMatrix<M> {
    /// The matrix associated to any cross product with this vector. I.e. `v.cross(anything)` =
    /// `v.cross_matrix().rmul(anything)`.
    fn cross_matrix(&self) -> M;
}
