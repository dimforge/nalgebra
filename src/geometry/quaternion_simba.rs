use simba::simd::SimdValue;

use crate::base::Vector4;
use crate::geometry::{Quaternion, UnitQuaternion};
use crate::Scalar;

impl<N: Scalar + SimdValue> SimdValue for Quaternion<N>
where
    N::Element: Scalar,
{
    type Element = Quaternion<N::Element>;
    type SimdBool = N::SimdBool;

    #[inline]
    fn lanes() -> usize {
        N::lanes()
    }

    #[inline]
    fn splat(val: Self::Element) -> Self {
        Vector4::splat(val.coords).into()
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

impl<N: Scalar + SimdValue> SimdValue for UnitQuaternion<N>
where
    N::Element: Scalar,
{
    type Element = UnitQuaternion<N::Element>;
    type SimdBool = N::SimdBool;

    #[inline]
    fn lanes() -> usize {
        N::lanes()
    }

    #[inline]
    fn splat(val: Self::Element) -> Self {
        UnitQuaternion::new_unchecked(Quaternion::splat(val.into_inner()))
    }

    #[inline]
    fn extract(&self, i: usize) -> Self::Element {
        UnitQuaternion::new_unchecked(self.as_ref().extract(i))
    }

    #[inline]
    unsafe fn extract_unchecked(&self, i: usize) -> Self::Element {
        UnitQuaternion::new_unchecked(self.as_ref().extract_unchecked(i))
    }

    #[inline]
    fn replace(&mut self, i: usize, val: Self::Element) {
        self.as_mut_unchecked().replace(i, val.into_inner())
    }

    #[inline]
    unsafe fn replace_unchecked(&mut self, i: usize, val: Self::Element) {
        self.as_mut_unchecked()
            .replace_unchecked(i, val.into_inner())
    }

    #[inline]
    fn select(self, cond: Self::SimdBool, other: Self) -> Self {
        UnitQuaternion::new_unchecked(self.into_inner().select(cond, other.into_inner()))
    }
}
