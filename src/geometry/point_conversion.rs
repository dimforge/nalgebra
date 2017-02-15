use num::{One, Zero};
use alga::general::{SubsetOf, SupersetOf, ClosedDiv};

use core::{Scalar, Matrix, ColumnVector, OwnedColumnVector};
use core::dimension::{DimName, DimNameSum, DimNameAdd, U1};
use core::storage::OwnedStorage;
use core::allocator::{Allocator, OwnedAllocator};

use geometry::{PointBase, OwnedPoint};

/*
 * This file provides the following conversions:
 * =============================================
 *
 * PointBase -> PointBase
 * PointBase -> ColumnVector (homogeneous)
 */

impl<N1, N2, D, SA, SB> SubsetOf<PointBase<N2, D, SB>> for PointBase<N1, D, SA>
    where D: DimName,
          N1: Scalar,
          N2: Scalar + SupersetOf<N1>,
          SA: OwnedStorage<N1, D, U1>,
          SB: OwnedStorage<N2, D, U1>,
          SB::Alloc: OwnedAllocator<N2, D, U1, SB>,
          SA::Alloc: OwnedAllocator<N1, D, U1, SA> {
    #[inline]
    fn to_superset(&self) -> PointBase<N2, D, SB> {
        PointBase::from_coordinates(self.coords.to_superset())
    }

    #[inline]
    fn is_in_subset(m: &PointBase<N2, D, SB>) -> bool {
        // FIXME: is there a way to reuse the `.is_in_subset` from the matrix implementation of
        // SubsetOf?
        m.iter().all(|e| e.is_in_subset())
    }

    #[inline]
    unsafe fn from_superset_unchecked(m: &PointBase<N2, D, SB>) -> Self {
        PointBase::from_coordinates(Matrix::from_superset_unchecked(&m.coords))
    }
}


impl<N1, N2, D, SA, SB> SubsetOf<ColumnVector<N2, DimNameSum<D, U1>, SB>> for PointBase<N1, D, SA>
    where D:  DimNameAdd<U1>,
          N1: Scalar,
          N2: Scalar + Zero + One + ClosedDiv + SupersetOf<N1>,
          SA: OwnedStorage<N1, D, U1>,
          SB: OwnedStorage<N2, DimNameSum<D, U1>, U1>,
          SA::Alloc: OwnedAllocator<N1, D, U1, SA> +
                     Allocator<N1, DimNameSum<D, U1>, U1>,
          SB::Alloc: OwnedAllocator<N2, DimNameSum<D, U1>, U1, SB> +
                     Allocator<N2, D, U1> {
    #[inline]
    fn to_superset(&self) -> ColumnVector<N2, DimNameSum<D, U1>, SB> {
        let p: OwnedPoint<N2, D, SB::Alloc> = self.to_superset();
        p.to_homogeneous()
    }

    #[inline]
    fn is_in_subset(v: &ColumnVector<N2, DimNameSum<D, U1>, SB>) -> bool {
        ::is_convertible::<_, OwnedColumnVector<N1, DimNameSum<D, U1>, SA::Alloc>>(v) &&
        !v[D::dim()].is_zero()
    }

    #[inline]
    unsafe fn from_superset_unchecked(v: &ColumnVector<N2, DimNameSum<D, U1>, SB>) -> Self {
        let coords =  v.fixed_slice::<D, U1>(0, 0) / v[D::dim()];
        Self::from_coordinates(::convert_unchecked(coords))
    }
}
