/**
 * Trait of objects having a L² norm and which can be normalized.
 */
pub trait Norm<N> {
    /// Computes the norm a an object.
    fn norm(&self) -> N;

    /**
     * Computes the squared norm of an object. Usually faster than computing the
     * norm itself.
     */
    fn sqnorm(&self) -> N;

    /// Gets the normalized version of the argument.
    fn normalized(&self) -> Self;

    /// In-place version of `normalized`.
    fn normalize(&mut self) -> N;
}
