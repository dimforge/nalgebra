use approx::ApproxEq;

use alga::general::{AbstractMagma, AbstractGroup, AbstractLoop, AbstractMonoid, AbstractQuasigroup,
                    AbstractSemigroup, Field, Real, Inverse, Multiplicative, Identity};
use alga::linear::{Transformation, ProjectiveTransformation};

use core::{Scalar, ColumnVector};
use core::dimension::{DimNameSum, DimNameAdd, U1};
use core::storage::OwnedStorage;
use core::allocator::{Allocator, OwnedAllocator};

use geometry::{PointBase, TransformBase, TCategory, SubTCategoryOf, TProjective};


/*
 *
 * Algebraic structures.
 *
 */
impl<N, D: DimNameAdd<U1>, S, C> Identity<Multiplicative> for TransformBase<N, D, S, C>
    where N: Scalar + Field,
          S: OwnedStorage<N, DimNameSum<D, U1>, DimNameSum<D, U1>>,
          C: TCategory,
          S::Alloc: OwnedAllocator<N, DimNameSum<D, U1>, DimNameSum<D, U1>, S> {
    #[inline]
    fn identity() -> Self {
        Self::identity()
    }
}

impl<N, D: DimNameAdd<U1>, S, C> Inverse<Multiplicative> for TransformBase<N, D, S, C>
    where N: Scalar + Field + ApproxEq,
          S: OwnedStorage<N, DimNameSum<D, U1>, DimNameSum<D, U1>>,
          C: SubTCategoryOf<TProjective>,
          S::Alloc: OwnedAllocator<N, DimNameSum<D, U1>, DimNameSum<D, U1>, S> {
    #[inline]
    fn inverse(&self) -> Self {
        self.clone().inverse()
    }

    #[inline]
    fn inverse_mut(&mut self) {
        self.inverse_mut()
    }
}

impl<N, D: DimNameAdd<U1>, S, C> AbstractMagma<Multiplicative> for TransformBase<N, D, S, C>
    where N: Scalar + Field,
          S: OwnedStorage<N, DimNameSum<D, U1>, DimNameSum<D, U1>>,
          C: TCategory,
          S::Alloc: OwnedAllocator<N, DimNameSum<D, U1>, DimNameSum<D, U1>, S> {
    #[inline]
    fn operate(&self, rhs: &Self) -> Self {
        self * rhs
    }
}

macro_rules! impl_multiplicative_structures(
    ($($marker: ident<$operator: ident>),* $(,)*) => {$(
        impl<N, D: DimNameAdd<U1>, S, C> $marker<$operator> for TransformBase<N, D, S, C>
            where N: Scalar + Field,
                  S: OwnedStorage<N, DimNameSum<D, U1>, DimNameSum<D, U1>>,
                  C: TCategory,
                  S::Alloc: OwnedAllocator<N, DimNameSum<D, U1>, DimNameSum<D, U1>, S> { }
    )*}
);

macro_rules! impl_inversible_multiplicative_structures(
    ($($marker: ident<$operator: ident>),* $(,)*) => {$(
        impl<N, D: DimNameAdd<U1>, S, C> $marker<$operator> for TransformBase<N, D, S, C>
            where N: Scalar + Field + ApproxEq,
                  S: OwnedStorage<N, DimNameSum<D, U1>, DimNameSum<D, U1>>,
                  C: SubTCategoryOf<TProjective>,
                  S::Alloc: OwnedAllocator<N, DimNameSum<D, U1>, DimNameSum<D, U1>, S> { }
    )*}
);

impl_multiplicative_structures!(
    AbstractSemigroup<Multiplicative>,
    AbstractMonoid<Multiplicative>,
);

impl_inversible_multiplicative_structures!(
    AbstractQuasigroup<Multiplicative>,
    AbstractLoop<Multiplicative>,
    AbstractGroup<Multiplicative>
);

/*
 *
 * Transformation groups.
 *
 */
impl<N, D: DimNameAdd<U1>, SA, SB, C> Transformation<PointBase<N, D, SB>> for TransformBase<N, D, SA, C>
    where N:  Real,
          SA: OwnedStorage<N, DimNameSum<D, U1>, DimNameSum<D, U1>>,
          SB: OwnedStorage<N, D, U1, Alloc = SA::Alloc>,
          C:  TCategory,
          SA::Alloc: OwnedAllocator<N, DimNameSum<D, U1>, DimNameSum<D, U1>, SA> +
                     Allocator<N, D, D>  +
                     Allocator<N, D, U1> +
                     Allocator<N, U1, D>,
          SB::Alloc: OwnedAllocator<N, D, U1, SB> {
    #[inline]
    fn transform_point(&self, pt: &PointBase<N, D, SB>) -> PointBase<N, D, SB> {
        self * pt
    }

    #[inline]
    fn transform_vector(&self, v: &ColumnVector<N, D, SB>) -> ColumnVector<N, D, SB> {
        self * v
    }
}

impl<N, D: DimNameAdd<U1>, SA, SB, C> ProjectiveTransformation<PointBase<N, D, SB>> for TransformBase<N, D, SA, C>
    where N:  Real,
          SA: OwnedStorage<N, DimNameSum<D, U1>, DimNameSum<D, U1>>,
          SB: OwnedStorage<N, D, U1, Alloc = SA::Alloc>,
          C:  SubTCategoryOf<TProjective>,
          SA::Alloc: OwnedAllocator<N, DimNameSum<D, U1>, DimNameSum<D, U1>, SA> +
                     Allocator<N, D, D>  +
                     Allocator<N, D, U1> +
                     Allocator<N, U1, D>,
          SB::Alloc: OwnedAllocator<N, D, U1, SB> {
    #[inline]
    fn inverse_transform_point(&self, pt: &PointBase<N, D, SB>) -> PointBase<N, D, SB> {
        self.inverse() * pt
    }

    #[inline]
    fn inverse_transform_vector(&self, v: &ColumnVector<N, D, SB>) -> ColumnVector<N, D, SB> {
        self.inverse() * v
    }
}

// FIXME: we need to implement an SVD for this.
//
// impl<N, D: DimNameAdd<U1>, SA, SB, C> AffineTransformation<PointBase<N, D, SB>> for TransformBase<N, D, SA, C>
//     where N:  Real,
//           SA: OwnedStorage<N, DimNameSum<D, U1>, DimNameSum<D, U1>>,
//           SB: OwnedStorage<N, D, U1, Alloc = SA::Alloc>,
//           C: SubTCategoryOf<TAffine>,
//           SA::Alloc: OwnedAllocator<N, DimNameSum<D, U1>, DimNameSum<D, U1>, SA> +
//                      Allocator<N, D, D>  +
//                      Allocator<N, D, U1> +
//                      Allocator<N, U1, D>,
//           SB::Alloc: OwnedAllocator<N, D, U1, SB> {
//     type PreRotation       = OwnedRotation<N, D, SA::Alloc>;
//     type NonUniformScaling = OwnedColumnVector<N, D, SA::Alloc>;
//     type PostRotation      = OwnedRotation<N, D, SA::Alloc>;
//     type Translation       = OwnedTranslation<N, D, SA::Alloc>;
// 
//     #[inline]
//     fn decompose(&self) -> (Self::Translation, Self::PostRotation, Self::NonUniformScaling, Self::PreRotation) {
//         unimplemented!()
//     }
// }
