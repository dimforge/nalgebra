#[cfg(all(feature = "alloc", not(feature = "std")))]
use alloc::vec::Vec;

use simba::simd::SimdValue;

use crate::base::allocator::Allocator;
use crate::base::dimension::Dim;
use crate::base::{DefaultAllocator, MatrixMN, Scalar};

/*
 *
 * Simd structures.
 *
 */
impl<N, R, C> SimdValue for MatrixMN<N, R, C>
where
    N: Scalar + SimdValue,
    R: Dim,
    C: Dim,
    N::Element: Scalar,
    DefaultAllocator: Allocator<N, R, C> + Allocator<N::Element, R, C>,
{
    type Element = MatrixMN<N::Element, R, C>;
    type SimdBool = N::SimdBool;

    #[inline]
    fn lanes() -> usize {
        N::lanes()
    }

    #[inline]
    fn splat(val: Self::Element) -> Self {
        val.map(N::splat)
    }

    #[inline]
    fn extract(&self, i: usize) -> Self::Element {
        self.map(|e| e.extract(i))
    }

    #[inline]
    unsafe fn extract_unchecked(&self, i: usize) -> Self::Element {
        self.map(|e| e.extract_unchecked(i))
    }

    #[inline]
    fn replace(&mut self, i: usize, val: Self::Element) {
        self.zip_apply(&val, |mut a, b| {
            a.replace(i, b);
            a
        })
    }

    #[inline]
    unsafe fn replace_unchecked(&mut self, i: usize, val: Self::Element) {
        self.zip_apply(&val, |mut a, b| {
            a.replace_unchecked(i, b);
            a
        })
    }

    fn select(self, cond: Self::SimdBool, other: Self) -> Self {
        self.zip_map(&other, |a, b| a.select(cond, b))
    }
}
