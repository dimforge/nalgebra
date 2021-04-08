use simba::simd::SimdValue;

use crate::base::{Scalar, VectorN};

use crate::geometry::Point;

impl<N: Scalar + SimdValue, const D: usize> SimdValue for Point<N, D>
where
    N::Element: Scalar,
{
    type Element = Point<N::Element, D>;
    type SimdBool = N::SimdBool;

    #[inline]
    fn lanes() -> usize {
        N::lanes()
    }

    #[inline]
    fn splat(val: Self::Element) -> Self {
        VectorN::splat(val.coords).into()
    }

    #[inline]
    fn extract(&self, i: usize) -> Self::Element {
        self.coords.extract(i).into()
    }

    #[inline]
    unsafe fn extract_unchecked(&self, i: usize) -> Self::Element {
        self.coords.extract_unchecked(i).into()
    }

    #[inline]
    fn replace(&mut self, i: usize, val: Self::Element) {
        self.coords.replace(i, val.coords)
    }

    #[inline]
    unsafe fn replace_unchecked(&mut self, i: usize, val: Self::Element) {
        self.coords.replace_unchecked(i, val.coords)
    }

    #[inline]
    fn select(self, cond: Self::SimdBool, other: Self) -> Self {
        self.coords.select(cond, other.coords).into()
    }
}
