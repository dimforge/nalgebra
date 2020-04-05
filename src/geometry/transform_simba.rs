use simba::simd::SimdValue;

use crate::base::allocator::Allocator;
use crate::base::dimension::{DimNameAdd, DimNameSum, U1};
use crate::base::{DefaultAllocator, MatrixN, Scalar};
use crate::RealField;

use crate::geometry::{TCategory, Transform};

impl<N: RealField, D: DimNameAdd<U1>, C> SimdValue for Transform<N, D, C>
where
    N::Element: Scalar,
    C: TCategory,
    DefaultAllocator: Allocator<N, DimNameSum<D, U1>, DimNameSum<D, U1>>
        + Allocator<N::Element, DimNameSum<D, U1>, DimNameSum<D, U1>>,
{
    type Element = Transform<N::Element, D, C>;
    type SimdBool = N::SimdBool;

    #[inline]
    fn lanes() -> usize {
        N::lanes()
    }

    #[inline]
    fn splat(val: Self::Element) -> Self {
        Transform::from_matrix_unchecked(MatrixN::splat(val.into_inner()))
    }

    #[inline]
    fn extract(&self, i: usize) -> Self::Element {
        Transform::from_matrix_unchecked(self.matrix().extract(i))
    }

    #[inline]
    unsafe fn extract_unchecked(&self, i: usize) -> Self::Element {
        Transform::from_matrix_unchecked(self.matrix().extract_unchecked(i))
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
        Transform::from_matrix_unchecked(self.into_inner().select(cond, other.into_inner()))
    }
}
