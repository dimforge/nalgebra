/**
 * Trait of objects having a right multiplication with another element.
 */
pub trait RMul<V>
{
    /// Computes self * v
    fn rmul(&self, v : &V) -> V;
}

/**
 * Trait of objects having a left multiplication with another element.
 */
pub trait LMul<V>
{
    /// Computes v * self
    fn lmul(&self, &V) -> V;
}
