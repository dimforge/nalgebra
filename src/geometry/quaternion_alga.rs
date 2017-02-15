use num::Zero;

use alga::general::{AbstractMagma, AbstractGroup, AbstractGroupAbelian, AbstractLoop,
                    AbstractMonoid, AbstractQuasigroup, AbstractSemigroup, AbstractModule,
                    Module, Real, Inverse, Multiplicative, Additive, Identity, Id};
use alga::linear::{Transformation, AffineTransformation, Similarity, Isometry, DirectIsometry,
                   OrthogonalTransformation, VectorSpace, FiniteDimVectorSpace, NormedSpace,
                   Rotation, ProjectiveTransformation};

use core::ColumnVector;
use core::storage::OwnedStorage;
use core::allocator::{Allocator, OwnedAllocator};
use core::dimension::{U1, U3, U4};
use geometry::{PointBase, QuaternionBase, UnitQuaternionBase};


impl<N, S> Identity<Multiplicative> for QuaternionBase<N, S>
    where N: Real,
          S: OwnedStorage<N, U4, U1>,
          S::Alloc: OwnedAllocator<N, U4, U1, S> {
    #[inline]
    fn identity() -> Self {
        Self::identity()
    }
}

impl<N, S> Identity<Additive> for QuaternionBase<N, S>
    where N: Real,
          S: OwnedStorage<N, U4, U1>,
          S::Alloc: OwnedAllocator<N, U4, U1, S> {
    #[inline]
    fn identity() -> Self {
        Self::zero()
    }
}

impl<N, S> AbstractMagma<Multiplicative> for QuaternionBase<N, S>
    where N: Real,
          S: OwnedStorage<N, U4, U1>,
          S::Alloc: OwnedAllocator<N, U4, U1, S> {
    #[inline]
    fn operate(&self, rhs: &Self) -> Self {
        self * rhs
    }
}

impl<N, S> AbstractMagma<Additive> for QuaternionBase<N, S>
    where N: Real,
          S: OwnedStorage<N, U4, U1>,
          S::Alloc: OwnedAllocator<N, U4, U1, S> {
    #[inline]
    fn operate(&self, rhs: &Self) -> Self {
        self + rhs
    }
}

impl<N, S> Inverse<Additive> for QuaternionBase<N, S>
    where N: Real,
          S: OwnedStorage<N, U4, U1>,
          S::Alloc: OwnedAllocator<N, U4, U1, S> {
    #[inline]
    fn inverse(&self) -> Self {
        -self
    }
}

macro_rules! impl_structures(
    ($Quaternion: ident; $($marker: ident<$operator: ident>),* $(,)*) => {$(
        impl<N, S> $marker<$operator> for $Quaternion<N, S>
            where N: Real,
                  S: OwnedStorage<N, U4, U1>,
                  S::Alloc: OwnedAllocator<N, U4, U1, S> { }
    )*}
);

impl_structures!(
    QuaternionBase;
    AbstractSemigroup<Multiplicative>,
    AbstractMonoid<Multiplicative>,

    AbstractSemigroup<Additive>,
    AbstractQuasigroup<Additive>,
    AbstractMonoid<Additive>,
    AbstractLoop<Additive>,
    AbstractGroup<Additive>,
    AbstractGroupAbelian<Additive>
);


/*
 *
 * Vector space.
 *
 */
impl<N, S> AbstractModule for QuaternionBase<N, S>
    where N: Real,
          S: OwnedStorage<N, U4, U1>,
          S::Alloc: OwnedAllocator<N, U4, U1, S> {
    type AbstractRing = N;

    #[inline]
    fn multiply_by(&self, n: N) -> Self {
        self * n
    }
}

impl<N, S> Module for QuaternionBase<N, S>
    where N: Real,
          S: OwnedStorage<N, U4, U1>,
          S::Alloc: OwnedAllocator<N, U4, U1, S> {
    type Ring = N;
}

impl<N, S> VectorSpace for QuaternionBase<N, S>
    where N: Real,
          S: OwnedStorage<N, U4, U1>,
          S::Alloc: OwnedAllocator<N, U4, U1, S> {
    type Field = N;
}

impl<N, S> FiniteDimVectorSpace for QuaternionBase<N, S>
    where N: Real,
          S: OwnedStorage<N, U4, U1>,
          S::Alloc: OwnedAllocator<N, U4, U1, S> {
    #[inline]
    fn dimension() -> usize {
        4
    }

    #[inline]
    fn canonical_basis_element(i: usize) -> Self {
        Self::from_vector(ColumnVector::canonical_basis_element(i))
    }

    #[inline]
    fn dot(&self, other: &Self) -> N {
        self.coords.dot(&other.coords)
    }

    #[inline]
    unsafe fn component_unchecked(&self, i: usize) -> &N {
        self.coords.component_unchecked(i)
    }

    #[inline]
    unsafe fn component_unchecked_mut(&mut self, i: usize) -> &mut N {
        self.coords.component_unchecked_mut(i)
    }
}

impl<N, S> NormedSpace for QuaternionBase<N, S>
    where N: Real,
          S: OwnedStorage<N, U4, U1>,
          S::Alloc: OwnedAllocator<N, U4, U1, S> {
    #[inline]
    fn norm_squared(&self) -> N {
        self.coords.norm_squared()
    }

    #[inline]
    fn norm(&self) -> N {
        self.as_vector().norm()
    }

    #[inline]
    fn normalize(&self) -> Self {
        let v = self.coords.normalize();
        Self::from_vector(v)
    }

    #[inline]
    fn normalize_mut(&mut self) -> N {
        self.coords.normalize_mut()
    }

    #[inline]
    fn try_normalize(&self, min_norm: N) -> Option<Self> {
        if let Some(v) = self.coords.try_normalize(min_norm) {
            Some(Self::from_vector(v))
        }
        else {
            None
        }
    }

    #[inline]
    fn try_normalize_mut(&mut self, min_norm: N) -> Option<N> {
        self.coords.try_normalize_mut(min_norm)
    }
}

/*
 *
 * Implementations for UnitQuaternionBase.
 *
 */
impl<N, S> Identity<Multiplicative> for UnitQuaternionBase<N, S>
    where N: Real,
          S: OwnedStorage<N, U4, U1>,
          S::Alloc: OwnedAllocator<N, U4, U1, S> {
    #[inline]
    fn identity() -> Self {
        Self::identity()
    }
}

impl<N, S> AbstractMagma<Multiplicative> for UnitQuaternionBase<N, S>
    where N: Real,
          S: OwnedStorage<N, U4, U1>,
          S::Alloc: OwnedAllocator<N, U4, U1, S> {
    #[inline]
    fn operate(&self, rhs: &Self) -> Self {
        self * rhs
    }
}

impl<N, S> Inverse<Multiplicative> for UnitQuaternionBase<N, S>
    where N: Real,
          S: OwnedStorage<N, U4, U1>,
          S::Alloc: OwnedAllocator<N, U4, U1, S> {
    #[inline]
    fn inverse(&self) -> Self {
        self.inverse()
    }

    #[inline]
    fn inverse_mut(&mut self) {
        self.inverse_mut()
    }
}

impl_structures!(
    UnitQuaternionBase;
    AbstractSemigroup<Multiplicative>,
    AbstractQuasigroup<Multiplicative>,
    AbstractMonoid<Multiplicative>,
    AbstractLoop<Multiplicative>,
    AbstractGroup<Multiplicative>
);

impl<N, SA, SB> Transformation<PointBase<N, U3, SB>> for UnitQuaternionBase<N, SA>
    where N: Real,
          SA: OwnedStorage<N, U4, U1>,
          SB: OwnedStorage<N, U3, U1, Alloc = SA::Alloc>,
          SA::Alloc: OwnedAllocator<N, U4, U1, SA>,
          SB::Alloc: OwnedAllocator<N, U3, U1, SB> {
    #[inline]
    fn transform_point(&self, pt: &PointBase<N, U3, SB>) -> PointBase<N, U3, SB> {
        self * pt
    }

    #[inline]
    fn transform_vector(&self, v: &ColumnVector<N, U3, SB>) -> ColumnVector<N, U3, SB> {
        self * v
    }
}

impl<N, SA, SB> ProjectiveTransformation<PointBase<N, U3, SB>> for UnitQuaternionBase<N, SA>
    where N: Real,
          SA: OwnedStorage<N, U4, U1>,
          SB: OwnedStorage<N, U3, U1, Alloc = SA::Alloc>,
          SA::Alloc: OwnedAllocator<N, U4, U1, SA>,
          SB::Alloc: OwnedAllocator<N, U3, U1, SB> {
    #[inline]
    fn inverse_transform_point(&self, pt: &PointBase<N, U3, SB>) -> PointBase<N, U3, SB> {
        // FIXME: would it be useful performancewise not to call inverse explicitly (i-e. implement
        // the inverse transformation explicitly here) ?
        self.inverse() * pt
    }

    #[inline]
    fn inverse_transform_vector(&self, v: &ColumnVector<N, U3, SB>) -> ColumnVector<N, U3, SB> {
        self.inverse() * v
    }
}

impl<N, SA, SB> AffineTransformation<PointBase<N, U3, SB>> for UnitQuaternionBase<N, SA>
    where N: Real,
          SA: OwnedStorage<N, U4, U1>,
          SB: OwnedStorage<N, U3, U1, Alloc = SA::Alloc>,
          SA::Alloc: OwnedAllocator<N, U4, U1, SA>,
          SB::Alloc: OwnedAllocator<N, U3, U1, SB> {
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

impl<N, SA, SB> Similarity<PointBase<N, U3, SB>> for UnitQuaternionBase<N, SA>
    where N: Real,
          SA: OwnedStorage<N, U4, U1>,
          SB: OwnedStorage<N, U3, U1, Alloc = SA::Alloc>,
          SA::Alloc: OwnedAllocator<N, U4, U1, SA>,
          SB::Alloc: OwnedAllocator<N, U3, U1, SB> {
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
        impl<N, SA, SB> $Trait<PointBase<N, U3, SB>> for UnitQuaternionBase<N, SA>
        where N: Real,
              SA: OwnedStorage<N, U4, U1>,
              SB: OwnedStorage<N, U3, U1, Alloc = SA::Alloc>,
              SA::Alloc: OwnedAllocator<N, U4, U1, SA>,
              SB::Alloc: OwnedAllocator<N, U3, U1, SB> { }
    )*}
);

marker_impl!(Isometry, DirectIsometry, OrthogonalTransformation);



impl<N, SA, SB> Rotation<PointBase<N, U3, SB>> for UnitQuaternionBase<N, SA>
    where N:  Real,
          SA: OwnedStorage<N, U4, U1>,
          SB: OwnedStorage<N, U3, U1, Alloc = SA::Alloc>,
          SA::Alloc: OwnedAllocator<N, U4, U1, SA> + Allocator<N, U3, U1>,
          SB::Alloc: OwnedAllocator<N, U3, U1, SB> {
    #[inline]
    fn powf(&self, n: N) -> Option<Self> {
        Some(self.powf(n))
    }

    #[inline]
    fn rotation_between(a: &ColumnVector<N, U3, SB>, b: &ColumnVector<N, U3, SB>) -> Option<Self> {
        Self::rotation_between(a, b)
    }

    #[inline]
    fn scaled_rotation_between(a: &ColumnVector<N, U3, SB>, b: &ColumnVector<N, U3, SB>, s: N) -> Option<Self> {
        Self::scaled_rotation_between(a, b, s)
    }
}
