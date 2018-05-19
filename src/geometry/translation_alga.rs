use alga::general::{AbstractGroup, AbstractLoop, AbstractMagma, AbstractMonoid,
                    AbstractQuasigroup, AbstractSemigroup, Id, Identity, Inverse, Multiplicative,
                    Real};
use alga::linear::{AffineTransformation, DirectIsometry, Isometry, ProjectiveTransformation,
                   Similarity, Transformation};
use alga::linear::Translation as AlgaTranslation;

use base::{DefaultAllocator, VectorN};
use base::dimension::DimName;
use base::allocator::Allocator;

use geometry::{Point, Translation};

/*
 *
 * Algebraic structures.
 *
 */
impl<N: Real, D: DimName> Identity<Multiplicative> for Translation<N, D>
where
    DefaultAllocator: Allocator<N, D>,
{
    #[inline]
    fn identity() -> Self {
        Self::identity()
    }
}

impl<N: Real, D: DimName> Inverse<Multiplicative> for Translation<N, D>
where
    DefaultAllocator: Allocator<N, D>,
{
    #[inline]
    fn inverse(&self) -> Self {
        self.inverse()
    }

    #[inline]
    fn inverse_mut(&mut self) {
        self.inverse_mut()
    }
}

impl<N: Real, D: DimName> AbstractMagma<Multiplicative> for Translation<N, D>
where
    DefaultAllocator: Allocator<N, D>,
{
    #[inline]
    fn operate(&self, rhs: &Self) -> Self {
        self * rhs
    }
}

macro_rules! impl_multiplicative_structures(
    ($($marker: ident<$operator: ident>),* $(,)*) => {$(
        impl<N: Real, D: DimName> $marker<$operator> for Translation<N, D>
            where DefaultAllocator: Allocator<N, D> { }
    )*}
);

impl_multiplicative_structures!(
    AbstractSemigroup<Multiplicative>,
    AbstractMonoid<Multiplicative>,
    AbstractQuasigroup<Multiplicative>,
    AbstractLoop<Multiplicative>,
    AbstractGroup<Multiplicative>
);

/*
 *
 * Transformation groups.
 *
 */
impl<N: Real, D: DimName> Transformation<Point<N, D>> for Translation<N, D>
where
    DefaultAllocator: Allocator<N, D>,
{
    #[inline]
    fn transform_point(&self, pt: &Point<N, D>) -> Point<N, D> {
        pt + &self.vector
    }

    #[inline]
    fn transform_vector(&self, v: &VectorN<N, D>) -> VectorN<N, D> {
        v.clone()
    }
}

impl<N: Real, D: DimName> ProjectiveTransformation<Point<N, D>> for Translation<N, D>
where
    DefaultAllocator: Allocator<N, D>,
{
    #[inline]
    fn inverse_transform_point(&self, pt: &Point<N, D>) -> Point<N, D> {
        pt - &self.vector
    }

    #[inline]
    fn inverse_transform_vector(&self, v: &VectorN<N, D>) -> VectorN<N, D> {
        v.clone()
    }
}

impl<N: Real, D: DimName> AffineTransformation<Point<N, D>> for Translation<N, D>
where
    DefaultAllocator: Allocator<N, D>,
{
    type Rotation = Id;
    type NonUniformScaling = Id;
    type Translation = Self;

    #[inline]
    fn decompose(&self) -> (Self, Id, Id, Id) {
        (self.clone(), Id::new(), Id::new(), Id::new())
    }

    #[inline]
    fn append_translation(&self, t: &Self::Translation) -> Self {
        t * self
    }

    #[inline]
    fn prepend_translation(&self, t: &Self::Translation) -> Self {
        self * t
    }

    #[inline]
    fn append_rotation(&self, _: &Self::Rotation) -> Self {
        self.clone()
    }

    #[inline]
    fn prepend_rotation(&self, _: &Self::Rotation) -> Self {
        self.clone()
    }

    #[inline]
    fn append_scaling(&self, _: &Self::NonUniformScaling) -> Self {
        self.clone()
    }

    #[inline]
    fn prepend_scaling(&self, _: &Self::NonUniformScaling) -> Self {
        self.clone()
    }
}

impl<N: Real, D: DimName> Similarity<Point<N, D>> for Translation<N, D>
where
    DefaultAllocator: Allocator<N, D>,
{
    type Scaling = Id;

    #[inline]
    fn translation(&self) -> Self {
        self.clone()
    }

    #[inline]
    fn rotation(&self) -> Id {
        Id::new()
    }

    #[inline]
    fn scaling(&self) -> Id {
        Id::new()
    }
}

macro_rules! marker_impl(
    ($($Trait: ident),*) => {$(
        impl<N: Real, D: DimName> $Trait<Point<N, D>> for Translation<N, D>
            where DefaultAllocator: Allocator<N, D> { }
    )*}
);

marker_impl!(Isometry, DirectIsometry);

/// Subgroups of the n-dimensional translation group `T(n)`.
impl<N: Real, D: DimName> AlgaTranslation<Point<N, D>> for Translation<N, D>
where
    DefaultAllocator: Allocator<N, D>,
{
    #[inline]
    fn to_vector(&self) -> VectorN<N, D> {
        self.vector.clone()
    }

    #[inline]
    fn from_vector(v: VectorN<N, D>) -> Option<Self> {
        Some(Self::from_vector(v))
    }

    #[inline]
    fn powf(&self, n: N) -> Option<Self> {
        Some(Self::from_vector(&self.vector * n))
    }

    #[inline]
    fn translation_between(a: &Point<N, D>, b: &Point<N, D>) -> Option<Self> {
        Some(Self::from_vector(b - a))
    }
}
