/// Traits of objects having an euclidian norm.
pub trait Norm<N: Algebraic> {
    /// Computes the norm a an object.
    #[inline]
    fn norm(&self) -> N {
        self.sqnorm().sqrt()
    }

    /**
     * Computes the squared norm of an object. Usually faster than computing the
     * norm itself.
     */
    #[inline]
    fn sqnorm(&self) -> N;

    /// Gets the normalized version of the argument.
    #[inline]
    fn normalized(&self) -> Self;

    /// In-place version of `normalized`.
    #[inline]
    fn normalize(&mut self) -> N;
}
