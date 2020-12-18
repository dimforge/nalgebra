/*
 * This file provides:
 *
 * NOTE: Work in progress https://github.com/dimforge/nalgebra/issues/487
 *
 * (Dual Quaternion)
 *
 * Index<usize>
 * IndexMut<usize>
 *
 * (Assignment Operators)
 *
 * DualQuaternion × Scalar
 * DualQuaternion × DualQuaternion
 * DualQuaternion + DualQuaternion
 * DualQuaternion - DualQuaternion
 *
 * ---
 *
 * References:
 *   Multiplication:
 *   - https://cs.gmu.edu/~jmlien/teaching/cs451/uploads/Main/dual-quaternion.pdf
 */

use crate::base::allocator::Allocator;
use crate::{DefaultAllocator, DualQuaternion, SimdRealField, U1, U4};
use simba::simd::SimdValue;
use std::mem;
use std::ops::{Add, Index, IndexMut, Mul, Sub};

impl<N: SimdRealField> AsRef<[N; 8]> for DualQuaternion<N> {
    #[inline]
    fn as_ref(&self) -> &[N; 8] {
        unsafe { mem::transmute(self) }
    }
}

impl<N: SimdRealField> AsMut<[N; 8]> for DualQuaternion<N> {
    #[inline]
    fn as_mut(&mut self) -> &mut [N; 8] {
        unsafe { mem::transmute(self) }
    }
}

impl<N: SimdRealField> Index<usize> for DualQuaternion<N> {
    type Output = N;

    #[inline]
    fn index(&self, i: usize) -> &Self::Output {
        &self.as_ref()[i]
    }
}

impl<N: SimdRealField> IndexMut<usize> for DualQuaternion<N> {
    #[inline]
    fn index_mut(&mut self, i: usize) -> &mut N {
        &mut self.as_mut()[i]
    }
}

impl<N: SimdRealField> Mul<DualQuaternion<N>> for DualQuaternion<N>
where
    N::Element: SimdRealField,
    DefaultAllocator: Allocator<N, U4, U1> + Allocator<N, U4, U1>,
{
    type Output = DualQuaternion<N>;

    fn mul(self, rhs: Self) -> Self::Output {
        Self::from_real_and_dual(
            self.real * rhs.real,
            self.real * rhs.dual + self.dual * rhs.real,
        )
    }
}

impl<N: SimdRealField> Mul<N> for DualQuaternion<N>
where
    N::Element: SimdRealField + SimdValue,
    DefaultAllocator: Allocator<N, U4, U1> + Allocator<N, U4, U1>,
{
    type Output = DualQuaternion<N>;

    fn mul(self, rhs: N) -> Self::Output {
        Self::from_real_and_dual(self.real * rhs, self.dual * rhs)
    }
}

impl<N: SimdRealField> Add<DualQuaternion<N>> for DualQuaternion<N>
where
    N::Element: SimdRealField,
    DefaultAllocator: Allocator<N, U4, U1> + Allocator<N, U4, U1>,
{
    type Output = DualQuaternion<N>;

    fn add(self, rhs: DualQuaternion<N>) -> Self::Output {
        Self::from_real_and_dual(self.real + rhs.real, self.dual + rhs.dual)
    }
}

impl<N: SimdRealField> Sub<DualQuaternion<N>> for DualQuaternion<N>
where
    N::Element: SimdRealField,
    DefaultAllocator: Allocator<N, U4, U1> + Allocator<N, U4, U1>,
{
    type Output = DualQuaternion<N>;

    fn sub(self, rhs: DualQuaternion<N>) -> Self::Output {
        Self::from_real_and_dual(self.real - rhs.real, self.dual - rhs.dual)
    }
}
