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
 * SimilarityBase -> SimilarityBase
 * SimilarityBase -> TransformBase
 * SimilarityBase -> Matrix (homogeneous)
 */


impl<N1, N2, D: DimName, SA, SB, R1, R2> SubsetOf<SimilarityBase<N2, D, SB, R2>> for SimilarityBase<N1, D, SA, R1>
    where N1: Real + SubsetOf<N2>,
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
            self.isometry.to_superset(),
            self.scaling().to_superset()
        )
    }

    #[inline]
    fn is_in_subset(sim: &SimilarityBase<N2, D, SB, R2>) -> bool {
        ::is_convertible::<_, IsometryBase<N1, D, SA, R1>>(&sim.isometry) &&
        ::is_convertible::<_, N1>(&sim.scaling())
    }

    #[inline]
    unsafe fn from_superset_unchecked(sim: &SimilarityBase<N2, D, SB, R2>) -> Self {
        SimilarityBase::from_isometry(
            sim.isometry.to_subset_unchecked(),
            sim.scaling().to_subset_unchecked()
        )
    }
}


impl<N1, N2, D, SA, SB, R, C> SubsetOf<TransformBase<N2, D, SB, C>> for SimilarityBase<N1, D, SA, R>
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


impl<N1, N2, D, SA, SB, R> SubsetOf<SquareMatrix<N2, DimNameSum<D, U1>, SB>> for SimilarityBase<N1, D, SA, R>
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
                     Allocator<N2, D, D>  + // needed by: mm.fixed_slice_mut
                     Allocator<N2, D, U1> + // needed by: m.fixed_slice
                     Allocator<N2, U1, D> { // needed by: m.fixed_slice
    #[inline]
    fn to_superset(&self) -> SquareMatrix<N2, DimNameSum<D, U1>, SB> {
        self.to_homogeneous().to_superset()
    }

    #[inline]
    fn is_in_subset(m: &SquareMatrix<N2, DimNameSum<D, U1>, SB>) -> bool {
        let mut rot = m.fixed_slice::<D, D>(0, 0).clone_owned();
        if rot.fixed_columns_mut::<U1>(0).try_normalize_mut(N2::zero()).is_some() &&
           rot.fixed_columns_mut::<U1>(1).try_normalize_mut(N2::zero()).is_some() &&
           rot.fixed_columns_mut::<U1>(2).try_normalize_mut(N2::zero()).is_some() {

            // FIXME: could we avoid explicit the computation of the determinant?
            // (its sign is needed to see if the scaling factor is negative).
            if rot.determinant() < N2::zero() {
                rot.fixed_columns_mut::<U1>(0).neg_mut();
                rot.fixed_columns_mut::<U1>(1).neg_mut();
                rot.fixed_columns_mut::<U1>(2).neg_mut();
            }

            let bottom = m.fixed_slice::<U1, D>(D::dim(), 0);
            // Scalar types agree.
            m.iter().all(|e| SupersetOf::<N1>::is_in_subset(e)) &&
            // The normalized block part is a rotation.
            // rot.is_special_orthogonal(N2::default_epsilon().sqrt()) &&
            // The bottom row is (0, 0, ..., 1)
            bottom.iter().all(|e| e.is_zero()) &&
            m[(D::dim(), D::dim())] == N2::one()
        }
        else {
            false
        }
    }

    #[inline]
    unsafe fn from_superset_unchecked(m: &SquareMatrix<N2, DimNameSum<D, U1>, SB>) -> Self {
        let mut mm = m.clone_owned();
        let na = mm.fixed_slice_mut::<D, U1>(0, 0).normalize_mut();
        let nb = mm.fixed_slice_mut::<D, U1>(0, 1).normalize_mut();
        let nc = mm.fixed_slice_mut::<D, U1>(0, 2).normalize_mut();

        let mut scale = (na + nb + nc) / ::convert(3.0); // We take the mean, for robustness.

        // FIXME: could we avoid the explicit computation of the determinant?
        // (its sign is needed to see if the scaling factor is negative).
        if mm.fixed_slice::<D, D>(0, 0).determinant() < N2::zero() {
            mm.fixed_slice_mut::<D, U1>(0, 0).neg_mut();
            mm.fixed_slice_mut::<D, U1>(0, 1).neg_mut();
            mm.fixed_slice_mut::<D, U1>(0, 2).neg_mut();
            scale = -scale;
        }

        let t = m.fixed_slice::<D, U1>(0, D::dim()).into_owned();
        let t = TranslationBase::from_vector(::convert_unchecked(t));

        Self::from_parts(t, ::convert_unchecked(mm), ::convert_unchecked(scale))
    }
}
