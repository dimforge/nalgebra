use num::{One, Zero};
use simba::scalar::{ClosedDiv, SubsetOf, SupersetOf};
use simba::simd::PrimitiveSimdValue;

use crate::base::allocator::Allocator;
use crate::base::dimension::{DimNameAdd, DimNameSum, U1};
use crate::base::{Const, DefaultAllocator, Matrix, OVector, Scalar};

use crate::geometry::Point;

/*
 * This file provides the following conversions:
 * =============================================
 *
 * Point -> Point
 * Point -> Vector (homogeneous)
 */

impl<T1, T2, const D: usize> SubsetOf<Point<T2, D>> for Point<T1, D>
where
    T1: Scalar,
    T2: Scalar + SupersetOf<T1>,
{
    #[inline]
    fn to_superset(&self) -> Point<T2, D> {
        Point::from(self.coords.to_superset())
    }

    #[inline]
    fn is_in_subset(m: &Point<T2, D>) -> bool {
        // TODO: is there a way to reuse the `.is_in_subset` from the matrix implementation of
        // SubsetOf?
        m.iter().all(|e| e.is_in_subset())
    }

    #[inline]
    fn from_superset_unchecked(m: &Point<T2, D>) -> Self {
        Self::from(Matrix::from_superset_unchecked(&m.coords))
    }
}

impl<T1, T2, const D: usize> SubsetOf<OVector<T2, DimNameSum<Const<D>, U1>>> for Point<T1, D>
where
    Const<D>: DimNameAdd<U1>,
    T1: Scalar,
    T2: Scalar + Zero + One + ClosedDiv + SupersetOf<T1>,
    DefaultAllocator:
        Allocator<T1, DimNameSum<Const<D>, U1>> + Allocator<T2, DimNameSum<Const<D>, U1>>,
    // + Allocator<T1, D>
    // + Allocator<T2, D>,
{
    #[inline]
    fn to_superset(&self) -> OVector<T2, DimNameSum<Const<D>, U1>> {
        let p: Point<T2, D> = self.to_superset();
        p.to_homogeneous()
    }

    #[inline]
    fn is_in_subset(v: &OVector<T2, DimNameSum<Const<D>, U1>>) -> bool {
        crate::is_convertible::<_, OVector<T1, DimNameSum<Const<D>, U1>>>(v) && !v[D].is_zero()
    }

    #[inline]
    fn from_superset_unchecked(v: &OVector<T2, DimNameSum<Const<D>, U1>>) -> Self {
        let coords = v.fixed_slice::<D, 1>(0, 0) / v[D].inlined_clone();
        Self {
            coords: crate::convert_unchecked(coords),
        }
    }
}

impl<T: Scalar + Zero + One, const D: usize> From<Point<T, D>>
    for OVector<T, DimNameSum<Const<D>, U1>>
where
    Const<D>: DimNameAdd<U1>,
    DefaultAllocator: Allocator<T, DimNameSum<Const<D>, U1>>,
{
    #[inline]
    fn from(t: Point<T, D>) -> Self {
        t.to_homogeneous()
    }
}

impl<T: Scalar, const D: usize> From<[T; D]> for Point<T, D> {
    #[inline]
    fn from(coords: [T; D]) -> Self {
        Point {
            coords: coords.into(),
        }
    }
}

impl<T: Scalar, const D: usize> Into<[T; D]> for Point<T, D> {
    #[inline]
    fn into(self) -> [T; D] {
        self.coords.into()
    }
}

impl<T: Scalar, const D: usize> From<OVector<T, Const<D>>> for Point<T, D> {
    #[inline]
    fn from(coords: OVector<T, Const<D>>) -> Self {
        Point { coords }
    }
}

impl<T: Scalar + Copy + PrimitiveSimdValue, const D: usize> From<[Point<T::Element, D>; 2]>
    for Point<T, D>
where
    T: From<[<T as simba::simd::SimdValue>::Element; 2]>,
    T::Element: Scalar + Copy,
    <DefaultAllocator as Allocator<T::Element, Const<D>>>::Buffer: Copy,
{
    #[inline]
    fn from(arr: [Point<T::Element, D>; 2]) -> Self {
        Self::from(OVector::from([arr[0].coords, arr[1].coords]))
    }
}

impl<T: Scalar + Copy + PrimitiveSimdValue, const D: usize> From<[Point<T::Element, D>; 4]>
    for Point<T, D>
where
    T: From<[<T as simba::simd::SimdValue>::Element; 4]>,
    T::Element: Scalar + Copy,
    <DefaultAllocator as Allocator<T::Element, Const<D>>>::Buffer: Copy,
{
    #[inline]
    fn from(arr: [Point<T::Element, D>; 4]) -> Self {
        Self::from(OVector::from([
            arr[0].coords,
            arr[1].coords,
            arr[2].coords,
            arr[3].coords,
        ]))
    }
}

impl<T: Scalar + Copy + PrimitiveSimdValue, const D: usize> From<[Point<T::Element, D>; 8]>
    for Point<T, D>
where
    T: From<[<T as simba::simd::SimdValue>::Element; 8]>,
    T::Element: Scalar + Copy,
    <DefaultAllocator as Allocator<T::Element, Const<D>>>::Buffer: Copy,
{
    #[inline]
    fn from(arr: [Point<T::Element, D>; 8]) -> Self {
        Self::from(OVector::from([
            arr[0].coords,
            arr[1].coords,
            arr[2].coords,
            arr[3].coords,
            arr[4].coords,
            arr[5].coords,
            arr[6].coords,
            arr[7].coords,
        ]))
    }
}

impl<T: Scalar + Copy + PrimitiveSimdValue, const D: usize> From<[Point<T::Element, D>; 16]>
    for Point<T, D>
where
    T: From<[<T as simba::simd::SimdValue>::Element; 16]>,
    T::Element: Scalar + Copy,
    <DefaultAllocator as Allocator<T::Element, Const<D>>>::Buffer: Copy,
{
    #[inline]
    fn from(arr: [Point<T::Element, D>; 16]) -> Self {
        Self::from(OVector::from([
            arr[0].coords,
            arr[1].coords,
            arr[2].coords,
            arr[3].coords,
            arr[4].coords,
            arr[5].coords,
            arr[6].coords,
            arr[7].coords,
            arr[8].coords,
            arr[9].coords,
            arr[10].coords,
            arr[11].coords,
            arr[12].coords,
            arr[13].coords,
            arr[14].coords,
            arr[15].coords,
        ]))
    }
}
