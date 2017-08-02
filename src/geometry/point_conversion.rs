use num::{One, Zero};
use alga::general::{SubsetOf, SupersetOf, ClosedDiv};

use core::{DefaultAllocator, Scalar, Matrix, VectorN};
use core::dimension::{DimName, DimNameSum, DimNameAdd, U1};
use core::allocator::Allocator;

use geometry::Point;

/*
 * This file provides the following conversions:
 * =============================================
 *
 * Point -> Point
 * Point -> Vector (homogeneous)
 */

impl<N1, N2, D> SubsetOf<Point<N2, D>> for Point<N1, D>
    where D: DimName,
          N1: Scalar,
          N2: Scalar + SupersetOf<N1>,
          DefaultAllocator: Allocator<N2, D> +
                            Allocator<N1, D> {
    #[inline]
    fn to_superset(&self) -> Point<N2, D> {
        Point::from_coordinates(self.coords.to_superset())
    }

    #[inline]
    fn is_in_subset(m: &Point<N2, D>) -> bool {
        // FIXME: is there a way to reuse the `.is_in_subset` from the matrix implementation of
        // SubsetOf?
        m.iter().all(|e| e.is_in_subset())
    }

    #[inline]
    unsafe fn from_superset_unchecked(m: &Point<N2, D>) -> Self {
        Point::from_coordinates(Matrix::from_superset_unchecked(&m.coords))
    }
}


impl<N1, N2, D> SubsetOf<VectorN<N2, DimNameSum<D, U1>>> for Point<N1, D>
    where D:  DimNameAdd<U1>,
          N1: Scalar,
          N2: Scalar + Zero + One + ClosedDiv + SupersetOf<N1>,
          DefaultAllocator: Allocator<N1, D>                 +
                            Allocator<N1, DimNameSum<D, U1>> +
                            Allocator<N2, DimNameSum<D, U1>> +
                            Allocator<N2, D> {
    #[inline]
    fn to_superset(&self) -> VectorN<N2, DimNameSum<D, U1>> {
        let p: Point<N2, D> = self.to_superset();
        p.to_homogeneous()
    }

    #[inline]
    fn is_in_subset(v: &VectorN<N2, DimNameSum<D, U1>>) -> bool {
        ::is_convertible::<_, VectorN<N1, DimNameSum<D, U1>>>(v) &&
        !v[D::dim()].is_zero()
    }

    #[inline]
    unsafe fn from_superset_unchecked(v: &VectorN<N2, DimNameSum<D, U1>>) -> Self {
        let coords =  v.fixed_slice::<D, U1>(0, 0) / v[D::dim()];
        Self::from_coordinates(::convert_unchecked(coords))
    }
}
