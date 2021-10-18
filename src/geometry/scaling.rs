use crate::{DefaultAllocator, DimName, OVector, allocator::Allocator, Scalar, Const};

/// A scaling
pub type Scaling<T, const D: usize> = OScaling<T, Const<D>>;

/// An owned scaling represents a non-uniform scale transformation
pub struct OScaling<T: Scalar, D: DimName>(pub OVector<T, D>) where DefaultAllocator: Allocator<T, D>;

impl<T, D: DimName> From<OVector<T, D>> for OScaling<T, D>
    where T: Scalar, DefaultAllocator: Allocator<T, D>
{
    fn from(other: OVector<T, D>) -> Self
    {
        return OScaling::<T, D>(other);
    }
}

impl<T, D: DimName> Into<OVector<T, D>> for OScaling<T, D>
    where T: Scalar, DefaultAllocator: Allocator<T, D>
{
    fn into(self) -> OVector<T, D>
    {
        return self.0;
    }
}
