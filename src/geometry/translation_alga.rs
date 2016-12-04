use alga::general::{AbstractMagma, AbstractGroup, AbstractLoop, AbstractMonoid, AbstractQuasigroup,
                    AbstractSemigroup, Real, Inverse, Multiplicative, Identity, Id};
use alga::linear::{Transformation, ProjectiveTransformation, Similarity, AffineTransformation,
                   Isometry, DirectIsometry, Translation};

use core::ColumnVector;
use core::dimension::{DimName, U1};
use core::storage::OwnedStorage;
use core::allocator::OwnedAllocator;

use geometry::{TranslationBase, PointBase};


/*
 *
 * Algebraic structures.
 *
 */
impl<N, D: DimName, S> Identity<Multiplicative> for TranslationBase<N, D, S>
    where N: Real,
          S: OwnedStorage<N, D, U1>,
          S::Alloc: OwnedAllocator<N, D, U1, S> {
    #[inline]
    fn identity() -> Self {
        Self::identity()
    }
}

impl<N, D: DimName, S> Inverse<Multiplicative> for TranslationBase<N, D, S>
    where N: Real,
          S: OwnedStorage<N, D, U1>,
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

impl<N, D: DimName, S> AbstractMagma<Multiplicative> for TranslationBase<N, D, S>
    where N: Real,
          S: OwnedStorage<N, D, U1>,
          S::Alloc: OwnedAllocator<N, D, U1, S> {
    #[inline]
    fn operate(&self, rhs: &Self) -> Self {
        self * rhs
    }
}

macro_rules! impl_multiplicative_structures(
    ($($marker: ident<$operator: ident>),* $(,)*) => {$(
        impl<N, D: DimName, S> $marker<$operator> for TranslationBase<N, D, S>
            where N: Real,
                  S: OwnedStorage<N, D, U1>,
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
impl<N, D: DimName, S> Transformation<PointBase<N, D, S>> for TranslationBase<N, D, S>
    where N: Real,
          S: OwnedStorage<N, D, U1>,
          S::Alloc: OwnedAllocator<N, D, U1, S> {
    #[inline]
    fn transform_point(&self, pt: &PointBase<N, D, S>) -> PointBase<N, D, S> {
        pt + &self.vector
    }

    #[inline]
    fn transform_vector(&self, v: &ColumnVector<N, D, S>) -> ColumnVector<N, D, S> {
        v.clone()
    }
}

impl<N, D: DimName, S> ProjectiveTransformation<PointBase<N, D, S>> for TranslationBase<N, D, S>
    where N: Real,
          S: OwnedStorage<N, D, U1>,
          S::Alloc: OwnedAllocator<N, D, U1, S> {
    #[inline]
    fn inverse_transform_point(&self, pt: &PointBase<N, D, S>) -> PointBase<N, D, S> {
        pt - &self.vector
    }

    #[inline]
    fn inverse_transform_vector(&self, v: &ColumnVector<N, D, S>) -> ColumnVector<N, D, S> {
        v.clone()
    }
}

impl<N, D: DimName, S> AffineTransformation<PointBase<N, D, S>> for TranslationBase<N, D, S>
    where N: Real,
          S: OwnedStorage<N, D, U1>,
          S::Alloc: OwnedAllocator<N, D, U1, S> {
    type Rotation          = Id;
    type NonUniformScaling = Id;
    type Translation       = Self;

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


impl<N, D: DimName, S> Similarity<PointBase<N, D, S>> for TranslationBase<N, D, S>
    where N: Real,
          S: OwnedStorage<N, D, U1>,
          S::Alloc: OwnedAllocator<N, D, U1, S> {
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
        impl<N, D: DimName, S> $Trait<PointBase<N, D, S>> for TranslationBase<N, D, S>
    where N: Real,
          S: OwnedStorage<N, D, U1>,
          S::Alloc: OwnedAllocator<N, D, U1, S> { }
    )*}
);

marker_impl!(Isometry, DirectIsometry);


/// Subgroups of the n-dimensional translation group `T(n)`.
impl<N, D: DimName, S> Translation<PointBase<N, D, S>> for TranslationBase<N, D, S>
    where N: Real,
          S: OwnedStorage<N, D, U1>,
          S::Alloc: OwnedAllocator<N, D, U1, S> {
    #[inline]
    fn to_vector(&self) -> ColumnVector<N, D, S> {
        self.vector.clone()
    }

    #[inline]
    fn from_vector(v: ColumnVector<N, D, S>) -> Option<Self> {
        Some(Self::from_vector(v))
    }

    #[inline]
    fn powf(&self, n: N) -> Option<Self> {
        Some(Self::from_vector(&self.vector * n))
    }

    #[inline]
    fn translation_between(a: &PointBase<N, D, S>, b: &PointBase<N, D, S>) -> Option<Self> {
        Some(Self::from_vector(b - a))
    }
}
