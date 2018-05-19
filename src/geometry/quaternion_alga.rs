use num::Zero;

use alga::general::{AbstractGroup, AbstractGroupAbelian, AbstractLoop, AbstractMagma,
                    AbstractModule, AbstractMonoid, AbstractQuasigroup, AbstractSemigroup,
                    Additive, Id, Identity, Inverse, Module, Multiplicative, Real};
use alga::linear::{AffineTransformation, DirectIsometry, FiniteDimVectorSpace, Isometry,
                   NormedSpace, OrthogonalTransformation, ProjectiveTransformation, Rotation,
                   Similarity, Transformation, VectorSpace};

use base::{Vector3, Vector4};
use geometry::{Point3, Quaternion, UnitQuaternion};

impl<N: Real> Identity<Multiplicative> for Quaternion<N> {
    #[inline]
    fn identity() -> Self {
        Self::identity()
    }
}

impl<N: Real> Identity<Additive> for Quaternion<N> {
    #[inline]
    fn identity() -> Self {
        Self::zero()
    }
}

impl<N: Real> AbstractMagma<Multiplicative> for Quaternion<N> {
    #[inline]
    fn operate(&self, rhs: &Self) -> Self {
        self * rhs
    }
}

impl<N: Real> AbstractMagma<Additive> for Quaternion<N> {
    #[inline]
    fn operate(&self, rhs: &Self) -> Self {
        self + rhs
    }
}

impl<N: Real> Inverse<Additive> for Quaternion<N> {
    #[inline]
    fn inverse(&self) -> Self {
        -self
    }
}

macro_rules! impl_structures(
    ($Quaternion: ident; $($marker: ident<$operator: ident>),* $(,)*) => {$(
        impl<N: Real> $marker<$operator> for $Quaternion<N> { }
    )*}
);

impl_structures!(
    Quaternion;
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
impl<N: Real> AbstractModule for Quaternion<N> {
    type AbstractRing = N;

    #[inline]
    fn multiply_by(&self, n: N) -> Self {
        self * n
    }
}

impl<N: Real> Module for Quaternion<N> {
    type Ring = N;
}

impl<N: Real> VectorSpace for Quaternion<N> {
    type Field = N;
}

impl<N: Real> FiniteDimVectorSpace for Quaternion<N> {
    #[inline]
    fn dimension() -> usize {
        4
    }

    #[inline]
    fn canonical_basis_element(i: usize) -> Self {
        Self::from_vector(Vector4::canonical_basis_element(i))
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

impl<N: Real> NormedSpace for Quaternion<N> {
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
        } else {
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
 * Implementations for UnitQuaternion.
 *
 */
impl<N: Real> Identity<Multiplicative> for UnitQuaternion<N> {
    #[inline]
    fn identity() -> Self {
        Self::identity()
    }
}

impl<N: Real> AbstractMagma<Multiplicative> for UnitQuaternion<N> {
    #[inline]
    fn operate(&self, rhs: &Self) -> Self {
        self * rhs
    }
}

impl<N: Real> Inverse<Multiplicative> for UnitQuaternion<N> {
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
    UnitQuaternion;
    AbstractSemigroup<Multiplicative>,
    AbstractQuasigroup<Multiplicative>,
    AbstractMonoid<Multiplicative>,
    AbstractLoop<Multiplicative>,
    AbstractGroup<Multiplicative>
);

impl<N: Real> Transformation<Point3<N>> for UnitQuaternion<N> {
    #[inline]
    fn transform_point(&self, pt: &Point3<N>) -> Point3<N> {
        self * pt
    }

    #[inline]
    fn transform_vector(&self, v: &Vector3<N>) -> Vector3<N> {
        self * v
    }
}

impl<N: Real> ProjectiveTransformation<Point3<N>> for UnitQuaternion<N> {
    #[inline]
    fn inverse_transform_point(&self, pt: &Point3<N>) -> Point3<N> {
        // FIXME: would it be useful performancewise not to call inverse explicitly (i-e. implement
        // the inverse transformation explicitly here) ?
        self.inverse() * pt
    }

    #[inline]
    fn inverse_transform_vector(&self, v: &Vector3<N>) -> Vector3<N> {
        self.inverse() * v
    }
}

impl<N: Real> AffineTransformation<Point3<N>> for UnitQuaternion<N> {
    type Rotation = Self;
    type NonUniformScaling = Id;
    type Translation = Id;

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

impl<N: Real> Similarity<Point3<N>> for UnitQuaternion<N> {
    type Scaling = Id;

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
        impl<N: Real> $Trait<Point3<N>> for UnitQuaternion<N> { }
    )*}
);

marker_impl!(Isometry, DirectIsometry, OrthogonalTransformation);

impl<N: Real> Rotation<Point3<N>> for UnitQuaternion<N> {
    #[inline]
    fn powf(&self, n: N) -> Option<Self> {
        Some(self.powf(n))
    }

    #[inline]
    fn rotation_between(a: &Vector3<N>, b: &Vector3<N>) -> Option<Self> {
        Self::rotation_between(a, b)
    }

    #[inline]
    fn scaled_rotation_between(a: &Vector3<N>, b: &Vector3<N>, s: N) -> Option<Self> {
        Self::scaled_rotation_between(a, b, s)
    }
}
