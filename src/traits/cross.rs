/**
 * Trait of elements having a cross product.
 */
pub trait Cross<Result>
{
    /// Computes the cross product between two elements (usually vectors).
    fn cross(&self, other : &Self) -> Result;
}
