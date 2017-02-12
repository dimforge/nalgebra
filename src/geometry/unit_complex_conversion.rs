use num::Zero;
use num_complex::Complex;

use alga::general::{SubsetOf, SupersetOf, Real};
use alga::linear::Rotation as AlgaRotation;

use core::SquareMatrix;
use core::dimension::{U1, U2, U3};
use core::storage::OwnedStorage;
use core::allocator::{Allocator, OwnedAllocator};
use geometry::{PointBase, UnitComplex, RotationBase, OwnedRotation, IsometryBase,
               SimilarityBase, TransformBase, SuperTCategoryOf, TAffine, TranslationBase};

/*
 * This file provides the following conversions:
 * =============================================
 *
 * UnitComplex -> UnitComplex
 * UnitComplex -> RotationBase<U1>
 * UnitComplex -> IsometryBase<U2>
 * UnitComplex -> SimilarityBase<U2>
 * UnitComplex -> TransformBase<U2>
 * UnitComplex -> Matrix<U3> (homogeneous)
 *
 * NOTE:
 * UnitComplex -> Complex is already provided by: Unit<T> -> T
 */

impl<N1, N2> SubsetOf<UnitComplex<N2>> for UnitComplex<N1>
    where N1: Real,
          N2: Real + SupersetOf<N1> {
    #[inline]
    fn to_superset(&self) -> UnitComplex<N2> {
        UnitComplex::new_unchecked(self.as_ref().to_superset())
    }

    #[inline]
    fn is_in_subset(uq: &UnitComplex<N2>) -> bool {
        ::is_convertible::<_, Complex<N1>>(uq.as_ref())
    }

    #[inline]
    unsafe fn from_superset_unchecked(uq: &UnitComplex<N2>) -> Self {
        Self::new_unchecked(::convert_ref_unchecked(uq.as_ref()))
    }
}

impl<N1, N2, S> SubsetOf<RotationBase<N2, U2, S>> for UnitComplex<N1>
    where N1: Real,
          N2: Real + SupersetOf<N1>,
          S: OwnedStorage<N2, U2, U2>,
          S::Alloc: OwnedAllocator<N2, U2, U2, S> +
                    Allocator<N2, U3, U1> +
                    Allocator<N2, U2, U1> +
                    Allocator<N1, U2, U2> {
    #[inline]
    fn to_superset(&self) -> RotationBase<N2, U2, S> {
        let q: UnitComplex<N2> = self.to_superset();
        q.to_rotation_matrix().to_superset()
    }

    #[inline]
    fn is_in_subset(rot: &RotationBase<N2, U2, S>) -> bool {
        ::is_convertible::<_, OwnedRotation<N1, U2, S::Alloc>>(rot)
    }

    #[inline]
    unsafe fn from_superset_unchecked(rot: &RotationBase<N2, U2, S>) -> Self {
        let q = UnitComplex::<N2>::from_rotation_matrix(rot);
        ::convert_unchecked(q)
    }
}


impl<N1, N2, S, R> SubsetOf<IsometryBase<N2, U2, S, R>> for UnitComplex<N1>
    where N1: Real,
          N2: Real + SupersetOf<N1>,
          S: OwnedStorage<N2, U2, U1>,
          R: AlgaRotation<PointBase<N2, U2, S>> + SupersetOf<UnitComplex<N1>>,
          S::Alloc: OwnedAllocator<N2, U2, U1, S> {
    #[inline]
    fn to_superset(&self) -> IsometryBase<N2, U2, S, R> {
        IsometryBase::from_parts(TranslationBase::identity(), ::convert_ref(self))
    }

    #[inline]
    fn is_in_subset(iso: &IsometryBase<N2, U2, S, R>) -> bool {
        iso.translation.vector.is_zero()
    }

    #[inline]
    unsafe fn from_superset_unchecked(iso: &IsometryBase<N2, U2, S, R>) -> Self {
        ::convert_ref_unchecked(&iso.rotation)
    }
}


impl<N1, N2, S, R> SubsetOf<SimilarityBase<N2, U2, S, R>> for UnitComplex<N1>
    where N1: Real,
          N2: Real + SupersetOf<N1>,
          S: OwnedStorage<N2, U2, U1>,
          R: AlgaRotation<PointBase<N2, U2, S>> + SupersetOf<UnitComplex<N1>>,
          S::Alloc: OwnedAllocator<N2, U2, U1, S> {
    #[inline]
    fn to_superset(&self) -> SimilarityBase<N2, U2, S, R> {
        SimilarityBase::from_isometry(::convert_ref(self), N2::one())
    }

    #[inline]
    fn is_in_subset(sim: &SimilarityBase<N2, U2, S, R>) -> bool {
        sim.isometry.translation.vector.is_zero() &&
        sim.scaling() == N2::one()
    }

    #[inline]
    unsafe fn from_superset_unchecked(sim: &SimilarityBase<N2, U2, S, R>) -> Self {
        ::convert_ref_unchecked(&sim.isometry)
    }
}


impl<N1, N2, S, C> SubsetOf<TransformBase<N2, U2, S, C>> for UnitComplex<N1>
    where N1: Real,
          N2: Real + SupersetOf<N1>,
          S: OwnedStorage<N2, U3, U3>,
          C: SuperTCategoryOf<TAffine>,
          S::Alloc: OwnedAllocator<N2, U3, U3, S> +
                    Allocator<N2, U2, U2>         +
                    Allocator<N2, U1, U2>         +
                    Allocator<N1, U2, U2>         +
                    Allocator<N1, U3, U3> {
    #[inline]
    fn to_superset(&self) -> TransformBase<N2, U2, S, C> {
        TransformBase::from_matrix_unchecked(self.to_homogeneous().to_superset())
    }

    #[inline]
    fn is_in_subset(t: &TransformBase<N2, U2, S, C>) -> bool {
        <Self as SubsetOf<_>>::is_in_subset(t.matrix())
    }

    #[inline]
    unsafe fn from_superset_unchecked(t: &TransformBase<N2, U2, S, C>) -> Self {
        Self::from_superset_unchecked(t.matrix())
    }
}


impl<N1, N2, S> SubsetOf<SquareMatrix<N2, U3, S>> for UnitComplex<N1>
    where N1: Real,
          N2: Real + SupersetOf<N1>,
          S: OwnedStorage<N2, U3, U3>,
          S::Alloc: OwnedAllocator<N2, U3, U3, S> +
                    Allocator<N2, U2, U2>         +
                    Allocator<N2, U1, U2>         +
                    Allocator<N1, U2, U2>         +
                    Allocator<N1, U3, U3> {
    #[inline]
    fn to_superset(&self) -> SquareMatrix<N2, U3, S> {
        self.to_homogeneous().to_superset()
    }

    #[inline]
    fn is_in_subset(m: &SquareMatrix<N2, U3, S>) -> bool {
        ::is_convertible::<_, OwnedRotation<N1, U2, S::Alloc>>(m)
    }

    #[inline]
    unsafe fn from_superset_unchecked(m: &SquareMatrix<N2, U3, S>) -> Self {
        let rot: OwnedRotation<N1, U2, S::Alloc> = ::convert_ref_unchecked(m);
        Self::from_rotation_matrix(&rot)
    }
}
