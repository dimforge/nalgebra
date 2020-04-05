use simba::simd::SimdValue;

use crate::base::allocator::Allocator;
use crate::base::dimension::DimName;
use crate::base::{DefaultAllocator, MatrixN, Scalar};

use crate::geometry::Rotation;

impl<N, D> SimdValue for Rotation<N, D>
where
    N: Scalar + SimdValue,
    D: DimName,
    N::Element: Scalar,
    DefaultAllocator: Allocator<N, D, D> + Allocator<N::Element, D, D>,
{
    type Element = Rotation<N::Element, D>;
    type SimdBool = N::SimdBool;

    #[inline]
    fn lanes() -> usize {
        N::lanes()
    }

    #[inline]
    fn splat(val: Self::Element) -> Self {
        Rotation::from_matrix_unchecked(MatrixN::splat(val.into_inner()))
    }

    #[inline]
    fn extract(&self, i: usize) -> Self::Element {
        Rotation::from_matrix_unchecked(self.matrix().extract(i))
    }

    #[inline]
    unsafe fn extract_unchecked(&self, i: usize) -> Self::Element {
        Rotation::from_matrix_unchecked(self.matrix().extract_unchecked(i))
    }

    #[inline]
    fn replace(&mut self, i: usize, val: Self::Element) {
        self.matrix_mut_unchecked().replace(i, val.into_inner())
    }

    #[inline]
    unsafe fn replace_unchecked(&mut self, i: usize, val: Self::Element) {
        self.matrix_mut_unchecked()
            .replace_unchecked(i, val.into_inner())
    }

    #[inline]
    fn select(self, cond: Self::SimdBool, other: Self) -> Self {
        Rotation::from_matrix_unchecked(self.into_inner().select(cond, other.into_inner()))
    }
}
