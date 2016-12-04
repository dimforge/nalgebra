use alga::general::{AbstractMagma, AbstractGroup, AbstractLoop, AbstractMonoid, AbstractQuasigroup,
                    AbstractSemigroup, Real, Inverse, Multiplicative, Identity, Id};
use alga::linear::{Transformation, Similarity, AffineTransformation, DirectIsometry, Isometry,
                   Rotation, ProjectiveTransformation};

use core::ColumnVector;
use core::dimension::{DimName, U1};
use core::storage::OwnedStorage;
use core::allocator::OwnedAllocator;

use geometry::{IsometryBase, TranslationBase, PointBase};


/*
 *
 * Algebraic structures.
 *
 */
impl<N, D: DimName, S, R> Identity<Multiplicative> for IsometryBase<N, D, S, R>
    where N: Real,
          S: OwnedStorage<N, D, U1>,
          R: Rotation<PointBase<N, D, S>>,
          S::Alloc: OwnedAllocator<N, D, U1, S> {
    #[inline]
    fn identity() -> Self {
        Self::identity()
    }
}

impl<N, D: DimName, S, R> Inverse<Multiplicative> for IsometryBase<N, D, S, R>
    where N: Real,
          S: OwnedStorage<N, D, U1>,
          R: Rotation<PointBase<N, D, S>>,
          S::Alloc: OwnedAllocator<N, D, U1, S> {
    #[inline]
    fn inverse(&self) -> Self {
        self.inverse()
    }

    #[inline]
    fn inverse_mut(&mut self) {
        self.inverse_mut()
    }
}

impl<N, D: DimName, S, R> AbstractMagma<Multiplicative> for IsometryBase<N, D, S, R>
    where N: Real,
          S: OwnedStorage<N, D, U1>,
          R: Rotation<PointBase<N, D, S>>,
          S::Alloc: OwnedAllocator<N, D, U1, S> {
    #[inline]
    fn operate(&self, rhs: &Self) -> Self {
        self * rhs
    }
}

macro_rules! impl_multiplicative_structures(
    ($($marker: ident<$operator: ident>),* $(,)*) => {$(
        impl<N, D: DimName, S, R> $marker<$operator> for IsometryBase<N, D, S, R>
            where N: Real,
                  S: OwnedStorage<N, D, U1>,
                  R: Rotation<PointBase<N, D, S>>,
                  S::Alloc: OwnedAllocator<N, D, U1, S> { }
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
impl<N, D: DimName, S, R> Transformation<PointBase<N, D, S>> for IsometryBase<N, D, S, R>
    where N: Real,
          S: OwnedStorage<N, D, U1>,
          R: Rotation<PointBase<N, D, S>>,
          S::Alloc: OwnedAllocator<N, D, U1, S> {
    #[inline]
    fn transform_point(&self, pt: &PointBase<N, D, S>) -> PointBase<N, D, S> {
        self * pt
    }

    #[inline]
    fn transform_vector(&self, v: &ColumnVector<N, D, S>) -> ColumnVector<N, D, S> {
        self * v
    }
}

impl<N, D: DimName, S, R> ProjectiveTransformation<PointBase<N, D, S>> for IsometryBase<N, D, S, R>
    where N: Real,
          S: OwnedStorage<N, D, U1>,
          R: Rotation<PointBase<N, D, S>>,
          S::Alloc: OwnedAllocator<N, D, U1, S> {
    #[inline]
    fn inverse_transform_point(&self, pt: &PointBase<N, D, S>) -> PointBase<N, D, S> {
        self.rotation.inverse_transform_point(&(pt - &self.translation.vector))
    }

    #[inline]
    fn inverse_transform_vector(&self, v: &ColumnVector<N, D, S>) -> ColumnVector<N, D, S> {
        self.rotation.inverse_transform_vector(v)
    }
}

impl<N, D: DimName, S, R> AffineTransformation<PointBase<N, D, S>> for IsometryBase<N, D, S, R>
    where N: Real,
          S: OwnedStorage<N, D, U1>,
          R: Rotation<PointBase<N, D, S>>,
          S::Alloc: OwnedAllocator<N, D, U1, S> {
    type Rotation          = R;
    type NonUniformScaling = Id;
    type Translation       = TranslationBase<N, D, S>;

    #[inline]
    fn decompose(&self) -> (TranslationBase<N, D, S>, R, Id, R) {
        (self.translation.clone(), self.rotation.clone(), Id::new(), R::identity())
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
    fn append_rotation(&self, r: &Self::Rotation) -> Self {
        let shift = r.transform_vector(&self.translation.vector);
        IsometryBase::from_parts(TranslationBase::from_vector(shift), r.clone() * self.rotation.clone())
    }

    #[inline]
    fn prepend_rotation(&self, r: &Self::Rotation) -> Self {
        self * r
    }

    #[inline]
    fn append_scaling(&self, _: &Self::NonUniformScaling) -> Self {
        self.clone()
    }

    #[inline]
    fn prepend_scaling(&self, _: &Self::NonUniformScaling) -> Self {
        self.clone()
    }

    #[inline]
    fn append_rotation_wrt_point(&self, r: &Self::Rotation, p: &PointBase<N, D, S>) -> Option<Self> {
        let mut res = self.clone();
        res.append_rotation_wrt_point_mut(r, p);
        Some(res)
    }
}

impl<N, D: DimName, S, R> Similarity<PointBase<N, D, S>> for IsometryBase<N, D, S, R>
    where N: Real,
          S: OwnedStorage<N, D, U1>,
          R: Rotation<PointBase<N, D, S>>,
          S::Alloc: OwnedAllocator<N, D, U1, S> {
    type Scaling = Id;

    #[inline]
    fn translation(&self) -> TranslationBase<N, D, S> {
        self.translation.clone()
    }

    #[inline]
    fn rotation(&self) -> R {
        self.rotation.clone()
    }

    #[inline]
    fn scaling(&self) -> Id {
        Id::new()
    }
}

macro_rules! marker_impl(
    ($($Trait: ident),*) => {$(
        impl<N, D: DimName, S, R> $Trait<PointBase<N, D, S>> for IsometryBase<N, D, S, R>
        where N: Real,
              S: OwnedStorage<N, D, U1>,
              R: Rotation<PointBase<N, D, S>>,
              S::Alloc: OwnedAllocator<N, D, U1, S> { }
    )*}
);

marker_impl!(Isometry, DirectIsometry);
