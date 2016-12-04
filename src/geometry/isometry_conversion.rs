use alga::general::{Real, SubsetOf, SupersetOf};
use alga::linear::Rotation;

use core::{SquareMatrix, OwnedSquareMatrix};
use core::dimension::{DimName, DimNameAdd, DimNameSum, U1};
use core::storage::OwnedStorage;
use core::allocator::{Allocator, OwnedAllocator};

use geometry::{PointBase, TranslationBase, IsometryBase, SimilarityBase, TransformBase, SuperTCategoryOf, TAffine};

/*
 * This file provides the following conversions:
 * =============================================
 *
 * IsometryBase -> IsometryBase
 * IsometryBase -> SimilarityBase
 * IsometryBase -> TransformBase
 * IsometryBase -> Matrix (homogeneous)
 */


impl<N1, N2, D: DimName, SA, SB, R1, R2> SubsetOf<IsometryBase<N2, D, SB, R2>> for IsometryBase<N1, D, SA, R1>
    where N1: Real,
          N2: Real + SupersetOf<N1>,
          R1: Rotation<PointBase<N1, D, SA>> + SubsetOf<R2>,
          R2: Rotation<PointBase<N2, D, SB>>,
          SA: OwnedStorage<N1, D, U1>,
          SB: OwnedStorage<N2, D, U1>,
          SA::Alloc: OwnedAllocator<N1, D, U1, SA>,
          SB::Alloc: OwnedAllocator<N2, D, U1, SB> {
    #[inline]
    fn to_superset(&self) -> IsometryBase<N2, D, SB, R2> {
        IsometryBase::from_parts(
            self.translation.to_superset(),
            self.rotation.to_superset()
        )
    }

    #[inline]
    fn is_in_subset(iso: &IsometryBase<N2, D, SB, R2>) -> bool {
        ::is_convertible::<_, TranslationBase<N1, D, SA>>(&iso.translation) &&
        ::is_convertible::<_, R1>(&iso.rotation)
    }

    #[inline]
    unsafe fn from_superset_unchecked(iso: &IsometryBase<N2, D, SB, R2>) -> Self {
        IsometryBase::from_parts(
            iso.translation.to_subset_unchecked(),
            iso.rotation.to_subset_unchecked()
        )
    }
}


impl<N1, N2, D: DimName, SA, SB, R1, R2> SubsetOf<SimilarityBase<N2, D, SB, R2>> for IsometryBase<N1, D, SA, R1>
    where N1: Real,
          N2: Real + SupersetOf<N1>,
          R1: Rotation<PointBase<N1, D, SA>> + SubsetOf<R2>,
          R2: Rotation<PointBase<N2, D, SB>>,
          SA: OwnedStorage<N1, D, U1>,
          SB: OwnedStorage<N2, D, U1>,
          SA::Alloc: OwnedAllocator<N1, D, U1, SA>,
          SB::Alloc: OwnedAllocator<N2, D, U1, SB> {
    #[inline]
    fn to_superset(&self) -> SimilarityBase<N2, D, SB, R2> {
        SimilarityBase::from_isometry(
            self.to_superset(),
            N2::one()
        )
    }

    #[inline]
    fn is_in_subset(sim: &SimilarityBase<N2, D, SB, R2>) -> bool {
        ::is_convertible::<_, IsometryBase<N1, D, SA, R1>>(&sim.isometry) &&
        sim.scaling() == N2::one()
    }

    #[inline]
    unsafe fn from_superset_unchecked(sim: &SimilarityBase<N2, D, SB, R2>) -> Self {
        ::convert_ref_unchecked(&sim.isometry)
    }
}


impl<N1, N2, D, SA, SB, R, C> SubsetOf<TransformBase<N2, D, SB, C>> for IsometryBase<N1, D, SA, R>
    where N1: Real,
          N2: Real + SupersetOf<N1>,
          SA: OwnedStorage<N1, D, U1>,
          SB: OwnedStorage<N2, DimNameSum<D, U1>, DimNameSum<D, U1>>,
          C:  SuperTCategoryOf<TAffine>,
          R: Rotation<PointBase<N1, D, SA>> +
             SubsetOf<OwnedSquareMatrix<N1, DimNameSum<D, U1>, SA::Alloc>> + // needed by: .to_homogeneous()
             SubsetOf<SquareMatrix<N2, DimNameSum<D, U1>, SB>>,              // needed by: ::convert_unchecked(mm)
          D:  DimNameAdd<U1>,
          SA::Alloc: OwnedAllocator<N1, D, U1, SA> +
                     Allocator<N1, D, D> +                                 // needed by R
                     Allocator<N1, DimNameSum<D, U1>, DimNameSum<D, U1>> + // needed by: .to_homogeneous()
                     Allocator<N2, DimNameSum<D, U1>, DimNameSum<D, U1>>,  // needed by R
          SB::Alloc: OwnedAllocator<N2, DimNameSum<D, U1>, DimNameSum<D, U1>, SB> +
                     Allocator<N2, D, D> +  // needed by: mm.fixed_slice_mut
                     Allocator<N2, D, U1> + // needed by: m.fixed_slice
                     Allocator<N2, U1, D> { // needed by: m.fixed_slice
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


impl<N1, N2, D, SA, SB, R> SubsetOf<SquareMatrix<N2, DimNameSum<D, U1>, SB>> for IsometryBase<N1, D, SA, R>
    where N1: Real,
          N2: Real + SupersetOf<N1>,
          SA: OwnedStorage<N1, D, U1>,
          SB: OwnedStorage<N2, DimNameSum<D, U1>, DimNameSum<D, U1>>,
          R: Rotation<PointBase<N1, D, SA>> +
             SubsetOf<OwnedSquareMatrix<N1, DimNameSum<D, U1>, SA::Alloc>> + // needed by: .to_homogeneous()
             SubsetOf<SquareMatrix<N2, DimNameSum<D, U1>, SB>>,              // needed by: ::convert_unchecked(mm)
          D:  DimNameAdd<U1>,
          SA::Alloc: OwnedAllocator<N1, D, U1, SA> +
                     Allocator<N1, D, D> +                                 // needed by R
                     Allocator<N1, DimNameSum<D, U1>, DimNameSum<D, U1>> + // needed by: .to_homogeneous()
                     Allocator<N2, DimNameSum<D, U1>, DimNameSum<D, U1>>,  // needed by R
          SB::Alloc: OwnedAllocator<N2, DimNameSum<D, U1>, DimNameSum<D, U1>, SB> +
                     Allocator<N2, D, D> +  // needed by: mm.fixed_slice_mut
                     Allocator<N2, D, U1> + // needed by: m.fixed_slice
                     Allocator<N2, U1, D> { // needed by: m.fixed_slice
    #[inline]
    fn to_superset(&self) -> SquareMatrix<N2, DimNameSum<D, U1>, SB> {
        self.to_homogeneous().to_superset()
    }

    #[inline]
    fn is_in_subset(m: &SquareMatrix<N2, DimNameSum<D, U1>, SB>) -> bool {
        let rot    = m.fixed_slice::<D, D>(0, 0);
        let bottom = m.fixed_slice::<U1, D>(D::dim(), 0);

        // Scalar types agree.
        m.iter().all(|e| SupersetOf::<N1>::is_in_subset(e)) &&
        // The block part is a rotation.
        rot.is_special_orthogonal(N2::default_epsilon() * ::convert(100.0)) &&
        // The bottom row is (0, 0, ..., 1)
        bottom.iter().all(|e| e.is_zero()) &&
        m[(D::dim(), D::dim())] == N2::one()
    }

    #[inline]
    unsafe fn from_superset_unchecked(m: &SquareMatrix<N2, DimNameSum<D, U1>, SB>) -> Self {
        let t = m.fixed_slice::<D, U1>(0, D::dim()).into_owned();
        let t = TranslationBase::from_vector(::convert_unchecked(t));

        Self::from_parts(t, ::convert_unchecked(m.clone_owned()))
    }
}
