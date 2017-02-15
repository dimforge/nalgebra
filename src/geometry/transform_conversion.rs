use approx::ApproxEq;

use alga::general::{SubsetOf, Field};

use core::{Scalar, SquareMatrix};
use core::dimension::{DimName, DimNameAdd, DimNameSum, U1};
use core::storage::OwnedStorage;
use core::allocator::OwnedAllocator;

use geometry::{TransformBase, TCategory, SuperTCategoryOf};


impl<N1, N2, D: DimName, SA, SB, C1, C2> SubsetOf<TransformBase<N2, D, SB, C2>> for TransformBase<N1, D, SA, C1>
    where N1: Scalar + Field + ApproxEq + SubsetOf<N2>,
          N2: Scalar + Field + ApproxEq,
          C1: TCategory,
          C2: SuperTCategoryOf<C1>,
          D: DimNameAdd<U1>,
          SA: OwnedStorage<N1, DimNameSum<D, U1>, DimNameSum<D, U1>>,
          SB: OwnedStorage<N2, DimNameSum<D, U1>, DimNameSum<D, U1>>,
          SA::Alloc: OwnedAllocator<N1, DimNameSum<D, U1>, DimNameSum<D, U1>, SA>,
          SB::Alloc: OwnedAllocator<N2, DimNameSum<D, U1>, DimNameSum<D, U1>, SB>,
          N1::Epsilon: Copy,
          N2::Epsilon: Copy {
    #[inline]
    fn to_superset(&self) -> TransformBase<N2, D, SB, C2> {
        TransformBase::from_matrix_unchecked(self.to_homogeneous().to_superset())
    }

    #[inline]
    fn is_in_subset(t: &TransformBase<N2, D, SB, C2>) -> bool {
        <Self as SubsetOf<_>>::is_in_subset(t.matrix())
    }

    #[inline]
    unsafe fn from_superset_unchecked(t: &TransformBase<N2, D, SB, C2>) -> Self {
        Self::from_superset_unchecked(t.matrix())
    }
}


impl<N1, N2, D: DimName, SA, SB, C> SubsetOf<SquareMatrix<N2, DimNameSum<D, U1>, SB>> for TransformBase<N1, D, SA, C>
    where N1: Scalar + Field + ApproxEq + SubsetOf<N2>,
          N2: Scalar + Field + ApproxEq,
          C: TCategory,
          D: DimNameAdd<U1>,
          SA: OwnedStorage<N1, DimNameSum<D, U1>, DimNameSum<D, U1>>,
          SB: OwnedStorage<N2, DimNameSum<D, U1>, DimNameSum<D, U1>>,
          SA::Alloc: OwnedAllocator<N1, DimNameSum<D, U1>, DimNameSum<D, U1>, SA>,
          SB::Alloc: OwnedAllocator<N2, DimNameSum<D, U1>, DimNameSum<D, U1>, SB>,
          N1::Epsilon: Copy,
          N2::Epsilon: Copy {
    #[inline]
    fn to_superset(&self) -> SquareMatrix<N2, DimNameSum<D, U1>, SB> {
        self.matrix().to_superset()
    }

    #[inline]
    fn is_in_subset(m: &SquareMatrix<N2, DimNameSum<D, U1>, SB>) -> bool {
        C::check_homogeneous_invariants(m)
    }

    #[inline]
    unsafe fn from_superset_unchecked(m: &SquareMatrix<N2, DimNameSum<D, U1>, SB>) -> Self {
        TransformBase::from_matrix_unchecked(::convert_ref_unchecked(m))
    }
}
