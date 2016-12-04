use num::Zero;

use alga::general::{Real, SubsetOf, SupersetOf};
use alga::linear::Rotation as AlgaRotation;

use core::{Matrix, SquareMatrix};
use core::dimension::{DimName, DimNameSum, DimNameAdd, U1, U3, U4};
use core::storage::OwnedStorage;
use core::allocator::{OwnedAllocator, Allocator};

use geometry::{PointBase, TranslationBase, RotationBase, UnitQuaternionBase, OwnedUnitQuaternionBase, IsometryBase,
               SimilarityBase, TransformBase, SuperTCategoryOf, TAffine};

/*
 * This file provides the following conversions:
 * =============================================
 *
 * RotationBase  -> RotationBase
 * Rotation3 -> UnitQuaternion
 * RotationBase  -> IsometryBase
 * RotationBase  -> SimilarityBase
 * RotationBase  -> TransformBase
 * RotationBase  -> Matrix (homogeneous)
 */


impl<N1, N2, D: DimName, SA, SB> SubsetOf<RotationBase<N2, D, SB>> for RotationBase<N1, D, SA>
    where N1: Real,
          N2: Real + SupersetOf<N1>,
          SA: OwnedStorage<N1, D, D>,
          SB: OwnedStorage<N2, D, D>,
          SA::Alloc: OwnedAllocator<N1, D, D, SA>,
          SB::Alloc: OwnedAllocator<N2, D, D, SB> {
    #[inline]
    fn to_superset(&self) -> RotationBase<N2, D, SB> {
        RotationBase::from_matrix_unchecked(self.matrix().to_superset())
    }

    #[inline]
    fn is_in_subset(rot: &RotationBase<N2, D, SB>) -> bool {
        ::is_convertible::<_, Matrix<N1, D, D, SA>>(rot.matrix())
    }

    #[inline]
    unsafe fn from_superset_unchecked(rot: &RotationBase<N2, D, SB>) -> Self {
        RotationBase::from_matrix_unchecked(rot.matrix().to_subset_unchecked())
    }
}


impl<N1, N2, SA, SB> SubsetOf<UnitQuaternionBase<N2, SB>> for RotationBase<N1, U3, SA>
    where N1: Real,
          N2: Real + SupersetOf<N1>,
          SA: OwnedStorage<N1, U3, U3>,
          SB: OwnedStorage<N2, U4, U1>,
          SA::Alloc: OwnedAllocator<N1, U3, U3, SA> +
                     Allocator<N1, U4, U1> +
                     Allocator<N1, U3, U1>,
          SB::Alloc: OwnedAllocator<N2, U4, U1, SB> +
                     Allocator<N2, U3, U3> {
    #[inline]
    fn to_superset(&self) -> UnitQuaternionBase<N2, SB> {
        let q = OwnedUnitQuaternionBase::<N1, SA::Alloc>::from_rotation_matrix(self);
        q.to_superset()
    }

    #[inline]
    fn is_in_subset(q: &UnitQuaternionBase<N2, SB>) -> bool {
        ::is_convertible::<_, OwnedUnitQuaternionBase<N1, SA::Alloc>>(q)
    }

    #[inline]
    unsafe fn from_superset_unchecked(q: &UnitQuaternionBase<N2, SB>) -> Self {
        let q: OwnedUnitQuaternionBase<N1, SA::Alloc> = ::convert_ref_unchecked(q);
        q.to_rotation_matrix()
    }
}


impl<N1, N2, D: DimName, SA, SB, R> SubsetOf<IsometryBase<N2, D, SB, R>> for RotationBase<N1, D, SA>
    where N1: Real,
          N2: Real + SupersetOf<N1>,
          SA: OwnedStorage<N1, D, D>,
          SB: OwnedStorage<N2, D, U1>,
          R:  AlgaRotation<PointBase<N2, D, SB>> + SupersetOf<RotationBase<N1, D, SA>>,
          SA::Alloc: OwnedAllocator<N1, D, D, SA>,
          SB::Alloc: OwnedAllocator<N2, D, U1, SB> {
    #[inline]
    fn to_superset(&self) -> IsometryBase<N2, D, SB, R> {
        IsometryBase::from_parts(TranslationBase::identity(), ::convert_ref(self))
    }

    #[inline]
    fn is_in_subset(iso: &IsometryBase<N2, D, SB, R>) -> bool {
        iso.translation.vector.is_zero()
    }

    #[inline]
    unsafe fn from_superset_unchecked(iso: &IsometryBase<N2, D, SB, R>) -> Self {
        ::convert_ref_unchecked(&iso.rotation)
    }
}


impl<N1, N2, D: DimName, SA, SB, R> SubsetOf<SimilarityBase<N2, D, SB, R>> for RotationBase<N1, D, SA>
    where N1: Real,
          N2: Real + SupersetOf<N1>,
          SA: OwnedStorage<N1, D, D>,
          SB: OwnedStorage<N2, D, U1>,
          R:  AlgaRotation<PointBase<N2, D, SB>> + SupersetOf<RotationBase<N1, D, SA>>,
          SA::Alloc: OwnedAllocator<N1, D, D, SA>,
          SB::Alloc: OwnedAllocator<N2, D, U1, SB> {
    #[inline]
    fn to_superset(&self) -> SimilarityBase<N2, D, SB, R> {
        SimilarityBase::from_parts(TranslationBase::identity(), ::convert_ref(self), N2::one())
    }

    #[inline]
    fn is_in_subset(sim: &SimilarityBase<N2, D, SB, R>) -> bool {
        sim.isometry.translation.vector.is_zero() &&
        sim.scaling() == N2::one()
    }

    #[inline]
    unsafe fn from_superset_unchecked(sim: &SimilarityBase<N2, D, SB, R>) -> Self {
        ::convert_ref_unchecked(&sim.isometry.rotation)
    }
}


impl<N1, N2, D, SA, SB, C> SubsetOf<TransformBase<N2, D, SB, C>> for RotationBase<N1, D, SA>
    where N1: Real,
          N2: Real + SupersetOf<N1>,
          SA: OwnedStorage<N1, D, D>,
          SB: OwnedStorage<N2, DimNameSum<D, U1>, DimNameSum<D, U1>>,
          C:  SuperTCategoryOf<TAffine>,
          D:  DimNameAdd<U1>,
          SA::Alloc: OwnedAllocator<N1, D, D, SA> +
                     Allocator<N1, DimNameSum<D, U1>, DimNameSum<D, U1>>,
          SB::Alloc: OwnedAllocator<N2, DimNameSum<D, U1>, DimNameSum<D, U1>, SB> +
                     Allocator<N2, D, D> +
                     Allocator<N2, U1, D> {
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


impl<N1, N2, D, SA, SB> SubsetOf<SquareMatrix<N2, DimNameSum<D, U1>, SB>> for RotationBase<N1, D, SA>
    where N1: Real,
          N2: Real + SupersetOf<N1>,
          SA: OwnedStorage<N1, D, D>,
          SB: OwnedStorage<N2, DimNameSum<D, U1>, DimNameSum<D, U1>>,
          D:  DimNameAdd<U1>,
          SA::Alloc: OwnedAllocator<N1, D, D, SA> +
                     Allocator<N1, DimNameSum<D, U1>, DimNameSum<D, U1>>,
          SB::Alloc: OwnedAllocator<N2, DimNameSum<D, U1>, DimNameSum<D, U1>, SB> +
                     Allocator<N2, D, D> +
                     Allocator<N2, U1, D> {
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
        let r = m.fixed_slice::<D, D>(0, 0);
        Self::from_matrix_unchecked(::convert_unchecked(r.into_owned()))
    }
}
