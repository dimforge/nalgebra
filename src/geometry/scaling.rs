use crate::{SVector, Scalar};

/// A scaling represents a non-uniform scale transformation
pub struct Scaling<T: Scalar, const D: usize>
{
    /// The inner vector
    pub vector: SVector<T, D>
}

impl<T, const D: usize> From<SVector<T, D>> for Scaling<T, D>
    where T: Scalar
{
    fn from(other: SVector<T, D>) -> Self
    {
        return Scaling {
            vector: other
        };
    }
}

impl<T, const D: usize> Into<SVector<T, D>> for Scaling<T, D>
    where T: Scalar
{
    fn into(self) -> SVector<T, D>
    {
        return self.vector;
    }
}
