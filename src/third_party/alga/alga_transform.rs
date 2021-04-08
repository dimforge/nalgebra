use alga::general::{
    AbstractGroup, AbstractLoop, AbstractMagma, AbstractMonoid, AbstractQuasigroup,
    AbstractSemigroup, Identity, Multiplicative, RealField, TwoSidedInverse,
};
use alga::linear::{ProjectiveTransformation, Transformation};

use crate::base::allocator::Allocator;
use crate::base::dimension::{DimNameAdd, DimNameSum, U1};
use crate::base::{CVectorN, Const, DefaultAllocator};

use crate::geometry::{Point, SubTCategoryOf, TCategory, TProjective, Transform};

/*
 *
 * Algebraic structures.
 *
 */
impl<N: RealField + simba::scalar::RealField, C, const D: usize> Identity<Multiplicative>
    for Transform<N, C, D>
where
    Const<D>: DimNameAdd<U1>,
    C: TCategory,
    DefaultAllocator: Allocator<N, DimNameSum<Const<D>, U1>, DimNameSum<Const<D>, U1>>,
{
    #[inline]
    fn identity() -> Self {
        Self::identity()
    }
}

impl<N: RealField + simba::scalar::RealField, C, const D: usize> TwoSidedInverse<Multiplicative>
    for Transform<N, C, D>
where
    Const<D>: DimNameAdd<U1>,
    C: SubTCategoryOf<TProjective>,
    DefaultAllocator: Allocator<N, DimNameSum<Const<D>, U1>, DimNameSum<Const<D>, U1>>,
{
    #[inline]
    #[must_use = "Did you mean to use two_sided_inverse_mut()?"]
    fn two_sided_inverse(&self) -> Self {
        self.clone().inverse()
    }

    #[inline]
    fn two_sided_inverse_mut(&mut self) {
        self.inverse_mut()
    }
}

impl<N: RealField + simba::scalar::RealField, C, const D: usize> AbstractMagma<Multiplicative>
    for Transform<N, C, D>
where
    Const<D>: DimNameAdd<U1>,
    C: TCategory,
    DefaultAllocator: Allocator<N, DimNameSum<Const<D>, U1>, DimNameSum<Const<D>, U1>>,
{
    #[inline]
    fn operate(&self, rhs: &Self) -> Self {
        self * rhs
    }
}

macro_rules! impl_multiplicative_structures(
    ($($marker: ident<$operator: ident>),* $(,)*) => {$(
        impl<N: RealField + simba::scalar::RealField, C, const D: usize> $marker<$operator> for Transform<N, C, D>
            where
                  Const<D>: DimNameAdd<U1>,
                  C: TCategory,
                  DefaultAllocator: Allocator<N, DimNameSum<Const<D>, U1>, DimNameSum<Const<D>, U1>> { }
    )*}
);

macro_rules! impl_inversible_multiplicative_structures(
    ($($marker: ident<$operator: ident>),* $(,)*) => {$(
        impl<N: RealField + simba::scalar::RealField, C, const D: usize> $marker<$operator> for Transform<N, C, D>
            where
                  Const<D>: DimNameAdd<U1>,
                  C: SubTCategoryOf<TProjective>,
                  DefaultAllocator: Allocator<N, DimNameSum<Const<D>, U1>, DimNameSum<Const<D>, U1>> { }
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
impl<N, C, const D: usize> Transformation<Point<N, D>> for Transform<N, C, D>
where
    Const<D>: DimNameAdd<U1>,
    N: RealField + simba::scalar::RealField,
    C: TCategory,
    DefaultAllocator: Allocator<N, DimNameSum<Const<D>, U1>, DimNameSum<Const<D>, U1>>
        + Allocator<N, DimNameSum<Const<D>, U1>>,
{
    #[inline]
    fn transform_point(&self, pt: &Point<N, D>) -> Point<N, D> {
        self.transform_point(pt)
    }

    #[inline]
    fn transform_vector(&self, v: &CVectorN<N, D>) -> CVectorN<N, D> {
        self.transform_vector(v)
    }
}

impl<N, C, const D: usize> ProjectiveTransformation<Point<N, D>> for Transform<N, C, D>
where
    Const<D>: DimNameAdd<U1>,
    N: RealField + simba::scalar::RealField,
    C: SubTCategoryOf<TProjective>,
    DefaultAllocator: Allocator<N, DimNameSum<Const<D>, U1>, DimNameSum<Const<D>, U1>>
        + Allocator<N, DimNameSum<Const<D>, U1>>,
{
    #[inline]
    fn inverse_transform_point(&self, pt: &Point<N, D>) -> Point<N, D> {
        self.inverse_transform_point(pt)
    }

    #[inline]
    fn inverse_transform_vector(&self, v: &CVectorN<N, D>) -> CVectorN<N, D> {
        self.inverse_transform_vector(v)
    }
}

// TODO: we need to implement an SVD for this.
//
// impl<N, C, const D: usize> AffineTransformation<Point<N, D>> for Transform<N, C, D>
//     where N:  RealField,
//           C: SubTCategoryOf<TAffine>,
//           DefaultAllocator: Allocator<N, DimNameSum<Const<D>, U1>, DimNameSum<Const<D>, U1>> +
//                             Allocator<N, D, D> +
//                             Allocator<N, D> {
//     type PreRotation       = Rotation<N, D>;
//     type NonUniformScaling = VectorN<N, D>;
//     type PostRotation      = Rotation<N, D>;
//     type Translation       = Translation<N, D>;
//
//     #[inline]
//     fn decompose(&self) -> (Self::Translation, Self::PostRotation, Self::NonUniformScaling, Self::PreRotation) {
//         unimplemented!()
//     }
// }
