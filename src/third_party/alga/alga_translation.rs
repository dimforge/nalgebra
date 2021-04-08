use alga::general::{
    AbstractGroup, AbstractLoop, AbstractMagma, AbstractMonoid, AbstractQuasigroup,
    AbstractSemigroup, Id, Identity, Multiplicative, RealField, TwoSidedInverse,
};
use alga::linear::Translation as AlgaTranslation;
use alga::linear::{
    AffineTransformation, DirectIsometry, Isometry, ProjectiveTransformation, Similarity,
    Transformation,
};

use crate::base::CVectorN;
use crate::geometry::{Point, Translation};

/*
 *
 * Algebraic structures.
 *
 */
impl<N: RealField + simba::scalar::RealField, const D: usize> Identity<Multiplicative>
    for Translation<N, D>
{
    #[inline]
    fn identity() -> Self {
        Self::identity()
    }
}

impl<N: RealField + simba::scalar::RealField, const D: usize> TwoSidedInverse<Multiplicative>
    for Translation<N, D>
{
    #[inline]
    #[must_use = "Did you mean to use two_sided_inverse_mut()?"]
    fn two_sided_inverse(&self) -> Self {
        self.inverse()
    }

    #[inline]
    fn two_sided_inverse_mut(&mut self) {
        self.inverse_mut()
    }
}

impl<N: RealField + simba::scalar::RealField, const D: usize> AbstractMagma<Multiplicative>
    for Translation<N, D>
{
    #[inline]
    fn operate(&self, rhs: &Self) -> Self {
        self * rhs
    }
}

macro_rules! impl_multiplicative_structures(
    ($($marker: ident<$operator: ident>),* $(,)*) => {$(
        impl<N: RealField + simba::scalar::RealField, const D: usize> $marker<$operator> for Translation<N, D>
            { }
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
impl<N: RealField + simba::scalar::RealField, const D: usize> Transformation<Point<N, D>>
    for Translation<N, D>
{
    #[inline]
    fn transform_point(&self, pt: &Point<N, D>) -> Point<N, D> {
        self.transform_point(pt)
    }

    #[inline]
    fn transform_vector(&self, v: &CVectorN<N, D>) -> CVectorN<N, D> {
        v.clone()
    }
}

impl<N: RealField + simba::scalar::RealField, const D: usize> ProjectiveTransformation<Point<N, D>>
    for Translation<N, D>
{
    #[inline]
    fn inverse_transform_point(&self, pt: &Point<N, D>) -> Point<N, D> {
        self.inverse_transform_point(pt)
    }

    #[inline]
    fn inverse_transform_vector(&self, v: &CVectorN<N, D>) -> CVectorN<N, D> {
        v.clone()
    }
}

impl<N: RealField + simba::scalar::RealField, const D: usize> AffineTransformation<Point<N, D>>
    for Translation<N, D>
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

impl<N: RealField + simba::scalar::RealField, const D: usize> Similarity<Point<N, D>>
    for Translation<N, D>
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
        impl<N: RealField + simba::scalar::RealField, const D: usize> $Trait<Point<N, D>> for Translation<N, D>
            { }
    )*}
);

marker_impl!(Isometry, DirectIsometry);

/// Subgroups of the n-dimensional translation group `T(n)`.
impl<N: RealField + simba::scalar::RealField, const D: usize> AlgaTranslation<Point<N, D>>
    for Translation<N, D>
{
    #[inline]
    fn to_vector(&self) -> CVectorN<N, D> {
        self.vector.clone()
    }

    #[inline]
    fn from_vector(v: CVectorN<N, D>) -> Option<Self> {
        Some(Self::from(v))
    }

    #[inline]
    fn powf(&self, n: N) -> Option<Self> {
        Some(Self::from(&self.vector * n))
    }

    #[inline]
    fn translation_between(a: &Point<N, D>, b: &Point<N, D>) -> Option<Self> {
        Some(Self::from(b - a))
    }
}
