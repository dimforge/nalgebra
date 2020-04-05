use simba::simd::SimdValue;

use crate::base::allocator::Allocator;
use crate::base::dimension::DimName;
use crate::base::{DefaultAllocator, VectorN};
use crate::Scalar;

use crate::geometry::Translation;

impl<N: Scalar + SimdValue, D: DimName> SimdValue for Translation<N, D>
where
    N::Element: Scalar,
    DefaultAllocator: Allocator<N, D> + Allocator<N::Element, D>,
{
    type Element = Translation<N::Element, D>;
    type SimdBool = N::SimdBool;

    #[inline]
    fn lanes() -> usize {
        N::lanes()
    }

    #[inline]
    fn splat(val: Self::Element) -> Self {
        VectorN::splat(val.vector).into()
    }

    #[inline]
    fn extract(&self, i: usize) -> Self::Element {
        self.vector.extract(i).into()
    }

    #[inline]
    unsafe fn extract_unchecked(&self, i: usize) -> Self::Element {
        self.vector.extract_unchecked(i).into()
    }

    #[inline]
    fn replace(&mut self, i: usize, val: Self::Element) {
        self.vector.replace(i, val.vector)
    }

    #[inline]
    unsafe fn replace_unchecked(&mut self, i: usize, val: Self::Element) {
        self.vector.replace_unchecked(i, val.vector)
    }

    #[inline]
    fn select(self, cond: Self::SimdBool, other: Self) -> Self {
        self.vector.select(cond, other.vector).into()
    }
}
