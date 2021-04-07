use num::{One, Zero};
use simba::scalar::{ClosedDiv, SubsetOf, SupersetOf};
use simba::simd::PrimitiveSimdValue;

use crate::base::allocator::Allocator;
use crate::base::dimension::{DimNameAdd, DimNameSum, U1};
use crate::base::{Const, DefaultAllocator, Matrix, Scalar, VectorN};

use crate::geometry::Point;

/*
 * This file provides the following conversions:
 * =============================================
 *
 * Point -> Point
 * Point -> Vector (homogeneous)
 */

impl<N1, N2, const D: usize> SubsetOf<Point<N2, D>> for Point<N1, D>
where
    N1: Scalar,
    N2: Scalar + SupersetOf<N1>,
    // DefaultAllocator: Allocator<N2, D> + Allocator<N1, D>,
{
    #[inline]
    fn to_superset(&self) -> Point<N2, D> {
        Point::from(self.coords.to_superset())
    }

    #[inline]
    fn is_in_subset(m: &Point<N2, D>) -> bool {
        // TODO: is there a way to reuse the `.is_in_subset` from the matrix implementation of
        // SubsetOf?
        m.iter().all(|e| e.is_in_subset())
    }

    #[inline]
    fn from_superset_unchecked(m: &Point<N2, D>) -> Self {
        Self::from(Matrix::from_superset_unchecked(&m.coords))
    }
}

impl<N1, N2, const D: usize> SubsetOf<VectorN<N2, DimNameSum<Const<D>, U1>>> for Point<N1, D>
where
    Const<D>: DimNameAdd<U1>,
    N1: Scalar,
    N2: Scalar + Zero + One + ClosedDiv + SupersetOf<N1>,
    DefaultAllocator:
        Allocator<N1, DimNameSum<Const<D>, U1>> + Allocator<N2, DimNameSum<Const<D>, U1>>,
    // + Allocator<N1, D>
    // + Allocator<N2, D>,
{
    #[inline]
    fn to_superset(&self) -> VectorN<N2, DimNameSum<Const<D>, U1>> {
        let p: Point<N2, D> = self.to_superset();
        p.to_homogeneous()
    }

    #[inline]
    fn is_in_subset(v: &VectorN<N2, DimNameSum<Const<D>, U1>>) -> bool {
        crate::is_convertible::<_, VectorN<N1, DimNameSum<D, U1>>>(v) && !v[D].is_zero()
    }

    #[inline]
    fn from_superset_unchecked(v: &VectorN<N2, DimNameSum<Const<D>, U1>>) -> Self {
        let coords = v.fixed_slice::<D, U1>(0, 0) / v[D].inlined_clone();
        Self {
            coords: crate::convert_unchecked(coords),
        }
    }
}

impl<N: Scalar + Zero + One, const D: usize> From<Point<N, D>>
    for VectorN<N, DimNameSum<Const<D>, U1>>
where
    Const<D>: DimNameAdd<U1>,
    DefaultAllocator: Allocator<N, DimNameSum<Const<D>, U1>>,
{
    #[inline]
    fn from(t: Point<N, D>) -> Self {
        t.to_homogeneous()
    }
}

impl<N: Scalar, const D: usize> From<VectorN<N, Const<D>>> for Point<N, D>
// where
//     DefaultAllocator: Allocator<N, D>,
{
    #[inline]
    fn from(coords: VectorN<N, Const<D>>) -> Self {
        Point { coords }
    }
}

impl<N: Scalar + Copy + PrimitiveSimdValue, const D: usize> From<[Point<N::Element, D>; 2]>
    for Point<N, D>
where
    N: From<[<N as simba::simd::SimdValue>::Element; 2]>,
    N::Element: Scalar + Copy,
    // DefaultAllocator: Allocator<N, D> + Allocator<N::Element, D>,
    <DefaultAllocator as Allocator<N::Element, Const<D>>>::Buffer: Copy,
{
    #[inline]
    fn from(arr: [Point<N::Element, D>; 2]) -> Self {
        Self::from(VectorN::from([arr[0].coords, arr[1].coords]))
    }
}

impl<N: Scalar + Copy + PrimitiveSimdValue, const D: usize> From<[Point<N::Element, D>; 4]>
    for Point<N, D>
where
    N: From<[<N as simba::simd::SimdValue>::Element; 4]>,
    N::Element: Scalar + Copy,
    // DefaultAllocator: Allocator<N, D> + Allocator<N::Element, D>,
    <DefaultAllocator as Allocator<N::Element, Const<D>>>::Buffer: Copy,
{
    #[inline]
    fn from(arr: [Point<N::Element, D>; 4]) -> Self {
        Self::from(VectorN::from([
            arr[0].coords,
            arr[1].coords,
            arr[2].coords,
            arr[3].coords,
        ]))
    }
}

impl<N: Scalar + Copy + PrimitiveSimdValue, const D: usize> From<[Point<N::Element, D>; 8]>
    for Point<N, D>
where
    N: From<[<N as simba::simd::SimdValue>::Element; 8]>,
    N::Element: Scalar + Copy,
    // DefaultAllocator: Allocator<N, D> + Allocator<N::Element, D>,
    <DefaultAllocator as Allocator<N::Element, Const<D>>>::Buffer: Copy,
{
    #[inline]
    fn from(arr: [Point<N::Element, D>; 8]) -> Self {
        Self::from(VectorN::from([
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

impl<N: Scalar + Copy + PrimitiveSimdValue, const D: usize> From<[Point<N::Element, D>; 16]>
    for Point<N, D>
where
    N: From<[<N as simba::simd::SimdValue>::Element; 16]>,
    N::Element: Scalar + Copy,
    // DefaultAllocator: Allocator<N, D> + Allocator<N::Element, D>,
    <DefaultAllocator as Allocator<N::Element, Const<D>>>::Buffer: Copy,
{
    #[inline]
    fn from(arr: [Point<N::Element, D>; 16]) -> Self {
        Self::from(VectorN::from([
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
