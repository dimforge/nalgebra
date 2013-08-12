/// Traits of objects having an outer product.
pub trait Outer<V, M> {
    /// Computes the outer product `self * other`
    fn outer(&self, other: &V) -> M;
}
