use num::Zero;

use alga::general::{SubsetOf, SupersetOf, Real};
use alga::linear::Rotation as AlgaRotation;

use core::{ColumnVector, SquareMatrix};
use core::dimension::{U1, U3, U4};
use core::storage::OwnedStorage;
use core::allocator::{Allocator, OwnedAllocator};
use geometry::{PointBase, QuaternionBase, UnitQuaternionBase, OwnedUnitQuaternionBase, RotationBase,
               OwnedRotation, IsometryBase, SimilarityBase, TransformBase, SuperTCategoryOf, TAffine, TranslationBase};

/*
 * This file provides the following conversions:
 * =============================================
 *
 * Quaternion     -> Quaternion
 * UnitQuaternion -> UnitQuaternion
 * UnitQuaternion -> RotationBase<U3>
 * UnitQuaternion -> IsometryBase<U3>
 * UnitQuaternion -> SimilarityBase<U3>
 * UnitQuaternion -> TransformBase<U3>
 * UnitQuaternion -> Matrix<U4> (homogeneous)
 *
 * NOTE:
 * UnitQuaternion -> Quaternion is already provided by: Unit<T> -> T
 */

impl<N1, N2, SA, SB> SubsetOf<QuaternionBase<N2, SB>> for QuaternionBase<N1, SA>
    where N1: Real,
          N2: Real + SupersetOf<N1>,
          SA: OwnedStorage<N1, U4, U1>,
          SB: OwnedStorage<N2, U4, U1>,
          SA::Alloc: OwnedAllocator<N1, U4, U1, SA>,
          SB::Alloc: OwnedAllocator<N2, U4, U1, SB> {
    #[inline]
    fn to_superset(&self) -> QuaternionBase<N2, SB> {
        QuaternionBase::from_vector(self.coords.to_superset())
    }

    #[inline]
    fn is_in_subset(q: &QuaternionBase<N2, SB>) -> bool {
        ::is_convertible::<_, ColumnVector<N1, U4, SA>>(&q.coords)
    }

    #[inline]
    unsafe fn from_superset_unchecked(q: &QuaternionBase<N2, SB>) -> Self {
        Self::from_vector(q.coords.to_subset_unchecked())
    }
}

impl<N1, N2, SA, SB> SubsetOf<UnitQuaternionBase<N2, SB>> for UnitQuaternionBase<N1, SA>
    where N1: Real,
          N2: Real + SupersetOf<N1>,
          SA: OwnedStorage<N1, U4, U1>,
          SB: OwnedStorage<N2, U4, U1>,
          SA::Alloc: OwnedAllocator<N1, U4, U1, SA>,
          SB::Alloc: OwnedAllocator<N2, U4, U1, SB> {
    #[inline]
    fn to_superset(&self) -> UnitQuaternionBase<N2, SB> {
        UnitQuaternionBase::new_unchecked(self.as_ref().to_superset())
    }

    #[inline]
    fn is_in_subset(uq: &UnitQuaternionBase<N2, SB>) -> bool {
        ::is_convertible::<_, QuaternionBase<N1, SA>>(uq.as_ref())
    }

    #[inline]
    unsafe fn from_superset_unchecked(uq: &UnitQuaternionBase<N2, SB>) -> Self {
        Self::new_unchecked(::convert_ref_unchecked(uq.as_ref()))
    }
}

impl<N1, N2, SA, SB> SubsetOf<RotationBase<N2, U3, SB>> for UnitQuaternionBase<N1, SA>
    where N1: Real,
          N2: Real + SupersetOf<N1>,
          SA: OwnedStorage<N1, U4, U1>,
          SB: OwnedStorage<N2, U3, U3>,
          SA::Alloc: OwnedAllocator<N1, U4, U1, SA> +
                     Allocator<N1, U3, U3>,
          SB::Alloc: OwnedAllocator<N2, U3, U3, SB> +
                     Allocator<N2, U4, U1> +
                     Allocator<N2, U3, U1> {
    #[inline]
    fn to_superset(&self) -> RotationBase<N2, U3, SB> {
        let q: OwnedUnitQuaternionBase<N2, SB::Alloc> = self.to_superset();
        q.to_rotation_matrix()
    }

    #[inline]
    fn is_in_subset(rot: &RotationBase<N2, U3, SB>) -> bool {
        ::is_convertible::<_, OwnedRotation<N1, U3, SA::Alloc>>(rot)
    }

    #[inline]
    unsafe fn from_superset_unchecked(rot: &RotationBase<N2, U3, SB>) -> Self {
        let q = OwnedUnitQuaternionBase::<N2, SB::Alloc>::from_rotation_matrix(rot);
        ::convert_unchecked(q)
    }
}


impl<N1, N2, SA, SB, R> SubsetOf<IsometryBase<N2, U3, SB, R>> for UnitQuaternionBase<N1, SA>
    where N1: Real,
          N2: Real + SupersetOf<N1>,
          SA: OwnedStorage<N1, U4, U1>,
          SB: OwnedStorage<N2, U3, U1>,
          R:  AlgaRotation<PointBase<N2, U3, SB>> + SupersetOf<UnitQuaternionBase<N1, SA>>,
          SA::Alloc: OwnedAllocator<N1, U4, U1, SA>,
          SB::Alloc: OwnedAllocator<N2, U3, U1, SB> {
    #[inline]
    fn to_superset(&self) -> IsometryBase<N2, U3, SB, R> {
        IsometryBase::from_parts(TranslationBase::identity(), ::convert_ref(self))
    }

    #[inline]
    fn is_in_subset(iso: &IsometryBase<N2, U3, SB, R>) -> bool {
        iso.translation.vector.is_zero()
    }

    #[inline]
    unsafe fn from_superset_unchecked(iso: &IsometryBase<N2, U3, SB, R>) -> Self {
        ::convert_ref_unchecked(&iso.rotation)
    }
}


impl<N1, N2, SA, SB, R> SubsetOf<SimilarityBase<N2, U3, SB, R>> for UnitQuaternionBase<N1, SA>
    where N1: Real,
          N2: Real + SupersetOf<N1>,
          SA: OwnedStorage<N1, U4, U1>,
          SB: OwnedStorage<N2, U3, U1>,
          R:  AlgaRotation<PointBase<N2, U3, SB>> + SupersetOf<UnitQuaternionBase<N1, SA>>,
          SA::Alloc: OwnedAllocator<N1, U4, U1, SA>,
          SB::Alloc: OwnedAllocator<N2, U3, U1, SB> {
    #[inline]
    fn to_superset(&self) -> SimilarityBase<N2, U3, SB, R> {
        SimilarityBase::from_isometry(::convert_ref(self), N2::one())
    }

    #[inline]
    fn is_in_subset(sim: &SimilarityBase<N2, U3, SB, R>) -> bool {
        sim.isometry.translation.vector.is_zero() &&
        sim.scaling() == N2::one()
    }

    #[inline]
    unsafe fn from_superset_unchecked(sim: &SimilarityBase<N2, U3, SB, R>) -> Self {
        ::convert_ref_unchecked(&sim.isometry)
    }
}


impl<N1, N2, SA, SB, C> SubsetOf<TransformBase<N2, U3, SB, C>> for UnitQuaternionBase<N1, SA>
    where N1: Real,
          N2: Real + SupersetOf<N1>,
          SA: OwnedStorage<N1, U4, U1>,
          SB: OwnedStorage<N2, U4, U4>,
          C:  SuperTCategoryOf<TAffine>,
          SA::Alloc: OwnedAllocator<N1, U4, U1, SA> +
                     Allocator<N1, U3, U3> +
                     Allocator<N1, U3, U1> +
                     Allocator<N1, U4, U4>,
          SB::Alloc: OwnedAllocator<N2, U4, U4, SB> +
                     Allocator<N2, U3, U3> +
                     Allocator<N2, U1, U3> {
    #[inline]
    fn to_superset(&self) -> TransformBase<N2, U3, SB, C> {
        TransformBase::from_matrix_unchecked(self.to_homogeneous().to_superset())
    }

    #[inline]
    fn is_in_subset(t: &TransformBase<N2, U3, SB, C>) -> bool {
        <Self as SubsetOf<_>>::is_in_subset(t.matrix())
    }

    #[inline]
    unsafe fn from_superset_unchecked(t: &TransformBase<N2, U3, SB, C>) -> Self {
        Self::from_superset_unchecked(t.matrix())
    }
}


impl<N1, N2, SA, SB> SubsetOf<SquareMatrix<N2, U4, SB>> for UnitQuaternionBase<N1, SA>
    where N1: Real,
          N2: Real + SupersetOf<N1>,
          SA: OwnedStorage<N1, U4, U1>,
          SB: OwnedStorage<N2, U4, U4>,
          SA::Alloc: OwnedAllocator<N1, U4, U1, SA> +
                     Allocator<N1, U3, U3> +
                     Allocator<N1, U3, U1> +
                     Allocator<N1, U4, U4>,
          SB::Alloc: OwnedAllocator<N2, U4, U4, SB> +
                     Allocator<N2, U3, U3> +
                     Allocator<N2, U1, U3> {
    #[inline]
    fn to_superset(&self) -> SquareMatrix<N2, U4, SB> {
        self.to_homogeneous().to_superset()
    }

    #[inline]
    fn is_in_subset(m: &SquareMatrix<N2, U4, SB>) -> bool {
        ::is_convertible::<_, OwnedRotation<N1, U3, SA::Alloc>>(m)
    }

    #[inline]
    unsafe fn from_superset_unchecked(m: &SquareMatrix<N2, U4, SB>) -> Self {
        let rot: OwnedRotation<N1, U3, SA::Alloc> = ::convert_ref_unchecked(m);
        Self::from_rotation_matrix(&rot)
    }
}
