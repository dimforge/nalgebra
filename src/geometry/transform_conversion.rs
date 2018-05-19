use alga::general::{Real, SubsetOf};

use base::{DefaultAllocator, MatrixN};
use base::dimension::{DimName, DimNameAdd, DimNameSum, U1};
use base::allocator::Allocator;

use geometry::{SuperTCategoryOf, TCategory, Transform};

impl<N1, N2, D: DimName, C1, C2> SubsetOf<Transform<N2, D, C2>> for Transform<N1, D, C1>
where
    N1: Real + SubsetOf<N2>,
    N2: Real,
    C1: TCategory,
    C2: SuperTCategoryOf<C1>,
    D: DimNameAdd<U1>,
    DefaultAllocator: Allocator<N1, DimNameSum<D, U1>, DimNameSum<D, U1>>
        + Allocator<N2, DimNameSum<D, U1>, DimNameSum<D, U1>>,
    N1::Epsilon: Copy,
    N2::Epsilon: Copy,
{
    #[inline]
    fn to_superset(&self) -> Transform<N2, D, C2> {
        Transform::from_matrix_unchecked(self.to_homogeneous().to_superset())
    }

    #[inline]
    fn is_in_subset(t: &Transform<N2, D, C2>) -> bool {
        <Self as SubsetOf<_>>::is_in_subset(t.matrix())
    }

    #[inline]
    unsafe fn from_superset_unchecked(t: &Transform<N2, D, C2>) -> Self {
        Self::from_superset_unchecked(t.matrix())
    }
}

impl<N1, N2, D: DimName, C> SubsetOf<MatrixN<N2, DimNameSum<D, U1>>> for Transform<N1, D, C>
where
    N1: Real + SubsetOf<N2>,
    N2: Real,
    C: TCategory,
    D: DimNameAdd<U1>,
    DefaultAllocator: Allocator<N1, DimNameSum<D, U1>, DimNameSum<D, U1>>
        + Allocator<N2, DimNameSum<D, U1>, DimNameSum<D, U1>>,
    N1::Epsilon: Copy,
    N2::Epsilon: Copy,
{
    #[inline]
    fn to_superset(&self) -> MatrixN<N2, DimNameSum<D, U1>> {
        self.matrix().to_superset()
    }

    #[inline]
    fn is_in_subset(m: &MatrixN<N2, DimNameSum<D, U1>>) -> bool {
        C::check_homogeneous_invariants(m)
    }

    #[inline]
    unsafe fn from_superset_unchecked(m: &MatrixN<N2, DimNameSum<D, U1>>) -> Self {
        Transform::from_matrix_unchecked(::convert_ref_unchecked(m))
    }
}
