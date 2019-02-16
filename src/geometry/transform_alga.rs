use alga::general::{
    AbstractGroup, AbstractLoop, AbstractMagma, AbstractMonoid, AbstractQuasigroup,
    AbstractSemigroup, Identity, TwoSidedInverse, Multiplicative, Real,
};
use alga::linear::{ProjectiveTransformation, Transformation};

use base::allocator::Allocator;
use base::dimension::{DimNameAdd, DimNameSum, U1};
use base::{DefaultAllocator, VectorN};

use geometry::{Point, SubTCategoryOf, TCategory, TProjective, Transform};

/*
 *
 * Algebraic structures.
 *
 */
impl<N: Real, D: DimNameAdd<U1>, C> Identity<Multiplicative> for Transform<N, D, C>
where
    C: TCategory,
    DefaultAllocator: Allocator<N, DimNameSum<D, U1>, DimNameSum<D, U1>>,
{
    #[inline]
    fn identity() -> Self {
        Self::identity()
    }
}

impl<N: Real, D: DimNameAdd<U1>, C> TwoSidedInverse<Multiplicative> for Transform<N, D, C>
where
    C: SubTCategoryOf<TProjective>,
    DefaultAllocator: Allocator<N, DimNameSum<D, U1>, DimNameSum<D, U1>>,
{
    #[inline]
    fn two_sided_inverse(&self) -> Self {
        self.clone().inverse()
    }

    #[inline]
    fn two_sided_inverse_mut(&mut self) {
        self.inverse_mut()
    }
}

impl<N: Real, D: DimNameAdd<U1>, C> AbstractMagma<Multiplicative> for Transform<N, D, C>
where
    C: TCategory,
    DefaultAllocator: Allocator<N, DimNameSum<D, U1>, DimNameSum<D, U1>>,
{
    #[inline]
    fn operate(&self, rhs: &Self) -> Self {
        self * rhs
    }
}

macro_rules! impl_multiplicative_structures(
    ($($marker: ident<$operator: ident>),* $(,)*) => {$(
        impl<N: Real, D: DimNameAdd<U1>, C> $marker<$operator> for Transform<N, D, C>
            where C: TCategory,
                  DefaultAllocator: Allocator<N, DimNameSum<D, U1>, DimNameSum<D, U1>> { }
    )*}
);

macro_rules! impl_inversible_multiplicative_structures(
    ($($marker: ident<$operator: ident>),* $(,)*) => {$(
        impl<N: Real, D: DimNameAdd<U1>, C> $marker<$operator> for Transform<N, D, C>
            where C: SubTCategoryOf<TProjective>,
                  DefaultAllocator: Allocator<N, DimNameSum<D, U1>, DimNameSum<D, U1>> { }
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
impl<N, D: DimNameAdd<U1>, C> Transformation<Point<N, D>> for Transform<N, D, C>
where
    N: Real,
    C: TCategory,
    DefaultAllocator: Allocator<N, DimNameSum<D, U1>, DimNameSum<D, U1>>
        + Allocator<N, DimNameSum<D, U1>>
        + Allocator<N, D, D>
        + Allocator<N, D>,
{
    #[inline]
    fn transform_point(&self, pt: &Point<N, D>) -> Point<N, D> {
        self * pt
    }

    #[inline]
    fn transform_vector(&self, v: &VectorN<N, D>) -> VectorN<N, D> {
        self * v
    }
}

impl<N, D: DimNameAdd<U1>, C> ProjectiveTransformation<Point<N, D>> for Transform<N, D, C>
where
    N: Real,
    C: SubTCategoryOf<TProjective>,
    DefaultAllocator: Allocator<N, DimNameSum<D, U1>, DimNameSum<D, U1>>
        + Allocator<N, DimNameSum<D, U1>>
        + Allocator<N, D, D>
        + Allocator<N, D>,
{
    #[inline]
    fn inverse_transform_point(&self, pt: &Point<N, D>) -> Point<N, D> {
        self.two_sided_inverse() * pt
    }

    #[inline]
    fn inverse_transform_vector(&self, v: &VectorN<N, D>) -> VectorN<N, D> {
        self.two_sided_inverse() * v
    }
}

// FIXME: we need to implement an SVD for this.
//
// impl<N, D: DimNameAdd<U1>, C> AffineTransformation<Point<N, D>> for Transform<N, D, C>
//     where N:  Real,
//           C: SubTCategoryOf<TAffine>,
//           DefaultAllocator: Allocator<N, DimNameSum<D, U1>, DimNameSum<D, U1>> +
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
