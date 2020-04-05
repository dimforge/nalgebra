use simba::simd::{SimdRealField, SimdValue};

use crate::base::allocator::Allocator;
use crate::base::dimension::DimName;
use crate::base::DefaultAllocator;

use crate::geometry::{AbstractRotation, Isometry, Similarity};

impl<N: SimdRealField, D: DimName, R> SimdValue for Similarity<N, D, R>
where
    N::Element: SimdRealField,
    R: SimdValue<SimdBool = N::SimdBool> + AbstractRotation<N, D>,
    R::Element: AbstractRotation<N::Element, D>,
    DefaultAllocator: Allocator<N, D> + Allocator<N::Element, D>,
{
    type Element = Similarity<N::Element, D, R::Element>;
    type SimdBool = N::SimdBool;

    #[inline]
    fn lanes() -> usize {
        N::lanes()
    }

    #[inline]
    fn splat(val: Self::Element) -> Self {
        let scaling = N::splat(val.scaling());
        Similarity::from_isometry(Isometry::splat(val.isometry), scaling)
    }

    #[inline]
    fn extract(&self, i: usize) -> Self::Element {
        Similarity::from_isometry(self.isometry.extract(i), self.scaling().extract(i))
    }

    #[inline]
    unsafe fn extract_unchecked(&self, i: usize) -> Self::Element {
        Similarity::from_isometry(
            self.isometry.extract_unchecked(i),
            self.scaling().extract_unchecked(i),
        )
    }

    #[inline]
    fn replace(&mut self, i: usize, val: Self::Element) {
        let mut s = self.scaling();
        s.replace(i, val.scaling());
        self.set_scaling(s);
        self.isometry.replace(i, val.isometry);
    }

    #[inline]
    unsafe fn replace_unchecked(&mut self, i: usize, val: Self::Element) {
        let mut s = self.scaling();
        s.replace_unchecked(i, val.scaling());
        self.set_scaling(s);
        self.isometry.replace_unchecked(i, val.isometry);
    }

    #[inline]
    fn select(self, cond: Self::SimdBool, other: Self) -> Self {
        let scaling = self.scaling().select(cond, other.scaling());
        Similarity::from_isometry(self.isometry.select(cond, other.isometry), scaling)
    }
}
