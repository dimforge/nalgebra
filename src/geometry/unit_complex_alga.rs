use alga::general::{AbstractMagma, AbstractGroup, AbstractLoop, AbstractMonoid, AbstractQuasigroup,
                    AbstractSemigroup, Real, Inverse, Multiplicative, Identity, Id};
use alga::linear::{Transformation, AffineTransformation, Similarity, Isometry, DirectIsometry,
                   OrthogonalTransformation, Rotation, ProjectiveTransformation};

use core::ColumnVector;
use core::storage::OwnedStorage;
use core::allocator::OwnedAllocator;
use core::dimension::{U1, U2};
use geometry::{PointBase, UnitComplex};

/*
 *
 * Implementations for UnitComplex.
 *
 */
impl<N: Real> Identity<Multiplicative> for UnitComplex<N> {
    #[inline]
    fn identity() -> Self {
        Self::identity()
    }
}

impl<N: Real> AbstractMagma<Multiplicative> for UnitComplex<N> {
    #[inline]
    fn operate(&self, rhs: &Self) -> Self {
        self * rhs
    }
}

impl<N: Real> Inverse<Multiplicative> for UnitComplex<N> {
    #[inline]
    fn inverse(&self) -> Self {
        self.inverse()
    }

    #[inline]
    fn inverse_mut(&mut self) {
        self.inverse_mut()
    }
}

macro_rules! impl_structures(
    ($($marker: ident<$operator: ident>),* $(,)*) => {$(
        impl<N: Real> $marker<$operator> for UnitComplex<N> {
        }
    )*}
);


impl_structures!(
    AbstractSemigroup<Multiplicative>,
    AbstractQuasigroup<Multiplicative>,
    AbstractMonoid<Multiplicative>,
    AbstractLoop<Multiplicative>,
    AbstractGroup<Multiplicative>
);

impl<N, S> Transformation<PointBase<N, U2, S>> for UnitComplex<N>
    where N: Real,
          S: OwnedStorage<N, U2, U1>,
          S::Alloc: OwnedAllocator<N, U2, U1, S> {
    #[inline]
    fn transform_point(&self, pt: &PointBase<N, U2, S>) -> PointBase<N, U2, S> {
        self * pt
    }

    #[inline]
    fn transform_vector(&self, v: &ColumnVector<N, U2, S>) -> ColumnVector<N, U2, S> {
        self * v
    }
}

impl<N, S> ProjectiveTransformation<PointBase<N, U2, S>> for UnitComplex<N>
    where N: Real,
          S: OwnedStorage<N, U2, U1>,
          S::Alloc: OwnedAllocator<N, U2, U1, S> {
    #[inline]
    fn inverse_transform_point(&self, pt: &PointBase<N, U2, S>) -> PointBase<N, U2, S> {
        // FIXME: would it be useful performancewise not to call inverse explicitly (i-e. implement
        // the inverse transformation explicitly here) ?
        self.inverse() * pt
    }

    #[inline]
    fn inverse_transform_vector(&self, v: &ColumnVector<N, U2, S>) -> ColumnVector<N, U2, S> {
        self.inverse() * v
    }
}

impl<N, S> AffineTransformation<PointBase<N, U2, S>> for UnitComplex<N>
    where N: Real,
          S: OwnedStorage<N, U2, U1>,
          S::Alloc: OwnedAllocator<N, U2, U1, S> {
    type Rotation          = Self;
    type NonUniformScaling = Id;
    type Translation       = Id;

    #[inline]
    fn decompose(&self) -> (Id, Self, Id, Self) {
        (Id::new(), self.clone(), Id::new(), Self::identity())
    }

    #[inline]
    fn append_translation(&self, _: &Self::Translation) -> Self {
        self.clone()
    }

    #[inline]
    fn prepend_translation(&self, _: &Self::Translation) -> Self {
        self.clone()
    }

    #[inline]
    fn append_rotation(&self, r: &Self::Rotation) -> Self {
        r * self
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
}

impl<N, S> Similarity<PointBase<N, U2, S>> for UnitComplex<N>
    where N: Real,
          S: OwnedStorage<N, U2, U1>,
          S::Alloc: OwnedAllocator<N, U2, U1, S> {
    type Scaling  = Id;

    #[inline]
    fn translation(&self) -> Id {
        Id::new()
    }

    #[inline]
    fn rotation(&self) -> Self {
        self.clone()
    }

    #[inline]
    fn scaling(&self) -> Id {
        Id::new()
    }
}

macro_rules! marker_impl(
    ($($Trait: ident),*) => {$(
        impl<N, S> $Trait<PointBase<N, U2, S>> for UnitComplex<N>
        where N: Real,
              S: OwnedStorage<N, U2, U1>,
              S::Alloc: OwnedAllocator<N, U2, U1, S> { }
    )*}
);

marker_impl!(Isometry, DirectIsometry, OrthogonalTransformation);



impl<N, S> Rotation<PointBase<N, U2, S>> for UnitComplex<N>
    where N: Real,
          S: OwnedStorage<N, U2, U1>,
          S::Alloc: OwnedAllocator<N, U2, U1, S> {
    #[inline]
    fn powf(&self, n: N) -> Option<Self> {
        Some(self.powf(n))
    }

    #[inline]
    fn rotation_between(a: &ColumnVector<N, U2, S>, b: &ColumnVector<N, U2, S>) -> Option<Self> {
        Some(Self::rotation_between(a, b))
    }

    #[inline]
    fn scaled_rotation_between(a: &ColumnVector<N, U2, S>, b: &ColumnVector<N, U2, S>, s: N) -> Option<Self> {
        Some(Self::scaled_rotation_between(a, b, s))
    }
}
