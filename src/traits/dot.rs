/// Traits of objects having a dot product.
pub trait Dot<N> {
    /// Computes the dot (inner) product of two vectors.
    #[inline]
    fn dot(&self, &Self) -> N;

    /**
     * Short-cut to compute the projection of a point on a vector, but without
     * computing intermediate vectors.
     * This must be equivalent to:
     *
     *   (a - b).dot(c)
     *
     */
    #[inline]
    fn sub_dot(&self, b: &Self, c: &Self) -> N;
}
