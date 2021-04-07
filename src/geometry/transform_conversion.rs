use simba::scalar::{RealField, SubsetOf};

use crate::base::allocator::Allocator;
use crate::base::dimension::{DimNameAdd, DimNameSum, U1};
use crate::base::{Const, DefaultAllocator, MatrixN};

use crate::geometry::{SuperTCategoryOf, TCategory, Transform};

impl<N1, N2, C1, C2, const D: usize> SubsetOf<Transform<N2, C2, D>> for Transform<N1, C1, D>
where
    N1: RealField + SubsetOf<N2>,
    N2: RealField,
    C1: TCategory,
    C2: SuperTCategoryOf<C1>,
    Const<D>: DimNameAdd<U1>,
    DefaultAllocator: Allocator<N1, DimNameSum<Const<D>, U1>, DimNameSum<Const<D>, U1>>
        + Allocator<N2, DimNameSum<Const<D>, U1>, DimNameSum<Const<D>, U1>>,
    N1::Epsilon: Copy,
    N2::Epsilon: Copy,
{
    #[inline]
    fn to_superset(&self) -> Transform<N2, C2, D> {
        Transform::from_matrix_unchecked(self.to_homogeneous().to_superset())
    }

    #[inline]
    fn is_in_subset(t: &Transform<N2, C2, D>) -> bool {
        <Self as SubsetOf<_>>::is_in_subset(t.matrix())
    }

    #[inline]
    fn from_superset_unchecked(t: &Transform<N2, C2, D>) -> Self {
        Self::from_superset_unchecked(t.matrix())
    }
}

impl<N1, N2, C, const D: usize> SubsetOf<MatrixN<N2, DimNameSum<Const<D>, U1>>>
    for Transform<N1, C, D>
where
    N1: RealField + SubsetOf<N2>,
    N2: RealField,
    C: TCategory,
    Const<D>: DimNameAdd<U1>,
    DefaultAllocator: Allocator<N1, DimNameSum<Const<D>, U1>, DimNameSum<Const<D>, U1>>
        + Allocator<N2, DimNameSum<Const<D>, U1>, DimNameSum<Const<D>, U1>>,
    N1::Epsilon: Copy,
    N2::Epsilon: Copy,
{
    #[inline]
    fn to_superset(&self) -> MatrixN<N2, DimNameSum<Const<D>, U1>> {
        self.matrix().to_superset()
    }

    #[inline]
    fn is_in_subset(m: &MatrixN<N2, DimNameSum<Const<D>, U1>>) -> bool {
        C::check_homogeneous_invariants(m)
    }

    #[inline]
    fn from_superset_unchecked(m: &MatrixN<N2, DimNameSum<Const<D>, U1>>) -> Self {
        Self::from_matrix_unchecked(crate::convert_ref_unchecked(m))
    }
}

impl<N: RealField, C, const D: usize> From<Transform<N, C, D>>
    for MatrixN<N, DimNameSum<Const<D>, U1>>
where
    Const<D>: DimNameAdd<U1>,
    C: TCategory,
    DefaultAllocator: Allocator<N, DimNameSum<Const<D>, U1>, DimNameSum<Const<D>, U1>>,
{
    #[inline]
    fn from(t: Transform<N, C, D>) -> Self {
        t.to_homogeneous()
    }
}
