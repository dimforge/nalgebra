use alga::general::{SubsetOf, SupersetOf, Real};
use alga::linear::Rotation;

use core::{Scalar, ColumnVector, SquareMatrix};
use core::dimension::{DimName, DimNameAdd, DimNameSum, U1};
use core::storage::OwnedStorage;
use core::allocator::{Allocator, OwnedAllocator};

use geometry::{PointBase, TranslationBase, IsometryBase, SimilarityBase, TransformBase, SuperTCategoryOf, TAffine};

/*
 * This file provides the following conversions:
 * =============================================
 *
 * TranslationBase -> TranslationBase
 * TranslationBase -> IsometryBase
 * TranslationBase -> SimilarityBase
 * TranslationBase -> TransformBase
 * TranslationBase -> Matrix (homogeneous)
 */

impl<N1, N2, D: DimName, SA, SB> SubsetOf<TranslationBase<N2, D, SB>> for TranslationBase<N1, D, SA>
    where N1: Scalar,
          N2: Scalar + SupersetOf<N1>,
          SA: OwnedStorage<N1, D, U1>,
          SB: OwnedStorage<N2, D, U1>,
          SA::Alloc: OwnedAllocator<N1, D, U1, SA>,
          SB::Alloc: OwnedAllocator<N2, D, U1, SB> {
    #[inline]
    fn to_superset(&self) -> TranslationBase<N2, D, SB> {
        TranslationBase::from_vector(self.vector.to_superset())
    }

    #[inline]
    fn is_in_subset(rot: &TranslationBase<N2, D, SB>) -> bool {
        ::is_convertible::<_, ColumnVector<N1, D, SA>>(&rot.vector)
    }

    #[inline]
    unsafe fn from_superset_unchecked(rot: &TranslationBase<N2, D, SB>) -> Self {
        TranslationBase::from_vector(rot.vector.to_subset_unchecked())
    }
}


impl<N1, N2, D: DimName, SA, SB, R> SubsetOf<IsometryBase<N2, D, SB, R>> for TranslationBase<N1, D, SA>
    where N1: Real,
          N2: Real + SupersetOf<N1>,
          SA: OwnedStorage<N1, D, U1>,
          SB: OwnedStorage<N2, D, U1>,
          R:  Rotation<PointBase<N2, D, SB>>,
          SA::Alloc: OwnedAllocator<N1, D, U1, SA>,
          SB::Alloc: OwnedAllocator<N2, D, U1, SB> {
    #[inline]
    fn to_superset(&self) -> IsometryBase<N2, D, SB, R> {
        IsometryBase::from_parts(self.to_superset(), R::identity())
    }

    #[inline]
    fn is_in_subset(iso: &IsometryBase<N2, D, SB, R>) -> bool {
        iso.rotation == R::identity()
    }

    #[inline]
    unsafe fn from_superset_unchecked(iso: &IsometryBase<N2, D, SB, R>) -> Self {
        Self::from_superset_unchecked(&iso.translation)
    }
}


impl<N1, N2, D: DimName, SA, SB, R> SubsetOf<SimilarityBase<N2, D, SB, R>> for TranslationBase<N1, D, SA>
    where N1: Real,
          N2: Real + SupersetOf<N1>,
          SA: OwnedStorage<N1, D, U1>,
          SB: OwnedStorage<N2, D, U1>,
          R:  Rotation<PointBase<N2, D, SB>>,
          SA::Alloc: OwnedAllocator<N1, D, U1, SA>,
          SB::Alloc: OwnedAllocator<N2, D, U1, SB> {
    #[inline]
    fn to_superset(&self) -> SimilarityBase<N2, D, SB, R> {
        SimilarityBase::from_parts(self.to_superset(), R::identity(), N2::one())
    }

    #[inline]
    fn is_in_subset(sim: &SimilarityBase<N2, D, SB, R>) -> bool {
        sim.isometry.rotation == R::identity() &&
        sim.scaling() == N2::one()
    }

    #[inline]
    unsafe fn from_superset_unchecked(sim: &SimilarityBase<N2, D, SB, R>) -> Self {
        Self::from_superset_unchecked(&sim.isometry.translation)
    }
}


impl<N1, N2, D, SA, SB, C> SubsetOf<TransformBase<N2, D, SB, C>> for TranslationBase<N1, D, SA>
    where N1: Real,
          N2: Real + SupersetOf<N1>,
          SA: OwnedStorage<N1, D, U1>,
          SB: OwnedStorage<N2, DimNameSum<D, U1>, DimNameSum<D, U1>>,
          C:  SuperTCategoryOf<TAffine>,
          D:  DimNameAdd<U1>,
          SA::Alloc: OwnedAllocator<N1, D, U1, SA> +
                     Allocator<N1, DimNameSum<D, U1>, DimNameSum<D, U1>>,
          SB::Alloc: OwnedAllocator<N2, DimNameSum<D, U1>, DimNameSum<D, U1>, SB> +
                     Allocator<N2, D, U1> +
                     Allocator<N2, DimNameSum<D, U1>, D> {
    #[inline]
    fn to_superset(&self) -> TransformBase<N2, D, SB, C> {
        TransformBase::from_matrix_unchecked(self.to_homogeneous().to_superset())
    }

    #[inline]
    fn is_in_subset(t: &TransformBase<N2, D, SB, C>) -> bool {
        <Self as SubsetOf<_>>::is_in_subset(t.matrix())
    }

    #[inline]
    unsafe fn from_superset_unchecked(t: &TransformBase<N2, D, SB, C>) -> Self {
        Self::from_superset_unchecked(t.matrix())
    }
}


impl<N1, N2, D, SA, SB> SubsetOf<SquareMatrix<N2, DimNameSum<D, U1>, SB>> for TranslationBase<N1, D, SA>
    where N1: Real,
          N2: Real + SupersetOf<N1>,
          SA: OwnedStorage<N1, D, U1>,
          SB: OwnedStorage<N2, DimNameSum<D, U1>, DimNameSum<D, U1>>,
          D:  DimNameAdd<U1>,
          SA::Alloc: OwnedAllocator<N1, D, U1, SA> +
                     Allocator<N1, DimNameSum<D, U1>, DimNameSum<D, U1>>,
          SB::Alloc: OwnedAllocator<N2, DimNameSum<D, U1>, DimNameSum<D, U1>, SB> +
                     Allocator<N2, D, U1> +
                     Allocator<N2, DimNameSum<D, U1>, D> {
    #[inline]
    fn to_superset(&self) -> SquareMatrix<N2, DimNameSum<D, U1>, SB> {
        self.to_homogeneous().to_superset()
    }

    #[inline]
    fn is_in_subset(m: &SquareMatrix<N2, DimNameSum<D, U1>, SB>) -> bool {
        let id = m.fixed_slice::<DimNameSum<D, U1>, D>(0, 0);

        // Scalar types agree.
        m.iter().all(|e| SupersetOf::<N1>::is_in_subset(e)) &&
        // The block part does nothing.
        id.is_identity(N2::zero()) &&
        // The normalization factor is one.
        m[(D::dim(), D::dim())] == N2::one()
    }

    #[inline]
    unsafe fn from_superset_unchecked(m: &SquareMatrix<N2, DimNameSum<D, U1>, SB>) -> Self {
        let t = m.fixed_slice::<D, U1>(0, D::dim());
        Self::from_vector(::convert_unchecked(t.into_owned()))
    }
}
