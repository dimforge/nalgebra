use num::Zero;

use alga::general::{
    AbstractGroup, AbstractGroupAbelian, AbstractLoop, AbstractMagma, AbstractModule,
    AbstractMonoid, AbstractQuasigroup, AbstractSemigroup, Additive, Id, Identity, Module,
    Multiplicative, RealField, TwoSidedInverse,
};
use alga::linear::{
    AffineTransformation, DirectIsometry, FiniteDimVectorSpace, Isometry, NormedSpace,
    ProjectiveTransformation, Similarity, Transformation, VectorSpace,
};

use crate::base::Vector3;
use crate::geometry::{
    DualQuaternion, Point3, Quaternion, Translation3, UnitDualQuaternion, UnitQuaternion,
};

impl<N: RealField + simba::scalar::RealField> Identity<Multiplicative> for DualQuaternion<N> {
    #[inline]
    fn identity() -> Self {
        Self::identity()
    }
}

impl<N: RealField + simba::scalar::RealField> Identity<Additive> for DualQuaternion<N> {
    #[inline]
    fn identity() -> Self {
        Self::zero()
    }
}

impl<N: RealField + simba::scalar::RealField> AbstractMagma<Multiplicative> for DualQuaternion<N> {
    #[inline]
    fn operate(&self, rhs: &Self) -> Self {
        self * rhs
    }
}

impl<N: RealField + simba::scalar::RealField> AbstractMagma<Additive> for DualQuaternion<N> {
    #[inline]
    fn operate(&self, rhs: &Self) -> Self {
        self + rhs
    }
}

impl<N: RealField + simba::scalar::RealField> TwoSidedInverse<Additive> for DualQuaternion<N> {
    #[inline]
    fn two_sided_inverse(&self) -> Self {
        -self
    }
}

macro_rules! impl_structures(
    ($DualQuaternion: ident; $($marker: ident<$operator: ident>),* $(,)*) => {$(
        impl<N: RealField + simba::scalar::RealField> $marker<$operator> for $DualQuaternion<N> { }
    )*}
);

impl_structures!(
    DualQuaternion;
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
impl<N: RealField + simba::scalar::RealField> AbstractModule for DualQuaternion<N> {
    type AbstractRing = N;

    #[inline]
    fn multiply_by(&self, n: N) -> Self {
        self * n
    }
}

impl<N: RealField + simba::scalar::RealField> Module for DualQuaternion<N> {
    type Ring = N;
}

impl<N: RealField + simba::scalar::RealField> VectorSpace for DualQuaternion<N> {
    type Field = N;
}

impl<N: RealField + simba::scalar::RealField> FiniteDimVectorSpace for DualQuaternion<N> {
    #[inline]
    fn dimension() -> usize {
        8
    }

    #[inline]
    fn canonical_basis_element(i: usize) -> Self {
        if i < 4 {
            DualQuaternion::from_real_and_dual(
                Quaternion::canonical_basis_element(i),
                Quaternion::zero(),
            )
        } else {
            DualQuaternion::from_real_and_dual(
                Quaternion::zero(),
                Quaternion::canonical_basis_element(i - 4),
            )
        }
    }

    #[inline]
    fn dot(&self, other: &Self) -> N {
        self.real.dot(&other.real) + self.dual.dot(&other.dual)
    }

    #[inline]
    unsafe fn component_unchecked(&self, i: usize) -> &N {
        self.as_ref().get_unchecked(i)
    }

    #[inline]
    unsafe fn component_unchecked_mut(&mut self, i: usize) -> &mut N {
        self.as_mut().get_unchecked_mut(i)
    }
}

impl<N: RealField + simba::scalar::RealField> NormedSpace for DualQuaternion<N> {
    type RealField = N;
    type ComplexField = N;

    #[inline]
    fn norm_squared(&self) -> N {
        self.real.norm_squared()
    }

    #[inline]
    fn norm(&self) -> N {
        self.real.norm()
    }

    #[inline]
    fn normalize(&self) -> Self {
        self.normalize()
    }

    #[inline]
    fn normalize_mut(&mut self) -> N {
        self.normalize_mut()
    }

    #[inline]
    fn try_normalize(&self, min_norm: N) -> Option<Self> {
        let real_norm = self.real.norm();
        if real_norm > min_norm {
            Some(Self::from_real_and_dual(
                self.real / real_norm,
                self.dual / real_norm,
            ))
        } else {
            None
        }
    }

    #[inline]
    fn try_normalize_mut(&mut self, min_norm: N) -> Option<N> {
        let real_norm = self.real.norm();
        if real_norm > min_norm {
            self.real /= real_norm;
            self.dual /= real_norm;
            Some(real_norm)
        } else {
            None
        }
    }
}

/*
 *
 * Implementations for UnitDualQuaternion.
 *
 */
impl<N: RealField + simba::scalar::RealField> Identity<Multiplicative> for UnitDualQuaternion<N> {
    #[inline]
    fn identity() -> Self {
        Self::identity()
    }
}

impl<N: RealField + simba::scalar::RealField> AbstractMagma<Multiplicative>
    for UnitDualQuaternion<N>
{
    #[inline]
    fn operate(&self, rhs: &Self) -> Self {
        self * rhs
    }
}

impl<N: RealField + simba::scalar::RealField> TwoSidedInverse<Multiplicative>
    for UnitDualQuaternion<N>
{
    #[inline]
    fn two_sided_inverse(&self) -> Self {
        self.inverse()
    }

    #[inline]
    fn two_sided_inverse_mut(&mut self) {
        self.inverse_mut()
    }
}

impl_structures!(
    UnitDualQuaternion;
    AbstractSemigroup<Multiplicative>,
    AbstractQuasigroup<Multiplicative>,
    AbstractMonoid<Multiplicative>,
    AbstractLoop<Multiplicative>,
    AbstractGroup<Multiplicative>
);

impl<N: RealField + simba::scalar::RealField> Transformation<Point3<N>> for UnitDualQuaternion<N> {
    #[inline]
    fn transform_point(&self, pt: &Point3<N>) -> Point3<N> {
        self.transform_point(pt)
    }

    #[inline]
    fn transform_vector(&self, v: &Vector3<N>) -> Vector3<N> {
        self.transform_vector(v)
    }
}

impl<N: RealField + simba::scalar::RealField> ProjectiveTransformation<Point3<N>>
    for UnitDualQuaternion<N>
{
    #[inline]
    fn inverse_transform_point(&self, pt: &Point3<N>) -> Point3<N> {
        self.inverse_transform_point(pt)
    }

    #[inline]
    fn inverse_transform_vector(&self, v: &Vector3<N>) -> Vector3<N> {
        self.inverse_transform_vector(v)
    }
}

impl<N: RealField + simba::scalar::RealField> AffineTransformation<Point3<N>>
    for UnitDualQuaternion<N>
{
    type Rotation = UnitQuaternion<N>;
    type NonUniformScaling = Id;
    type Translation = Translation3<N>;

    #[inline]
    fn decompose(&self) -> (Self::Translation, Self::Rotation, Id, Self::Rotation) {
        (
            self.translation(),
            self.rotation(),
            Id::new(),
            UnitQuaternion::identity(),
        )
    }

    #[inline]
    fn append_translation(&self, translation: &Self::Translation) -> Self {
        self * Self::from_parts(translation.clone(), UnitQuaternion::identity())
    }

    #[inline]
    fn prepend_translation(&self, translation: &Self::Translation) -> Self {
        Self::from_parts(translation.clone(), UnitQuaternion::identity()) * self
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

impl<N: RealField + simba::scalar::RealField> Similarity<Point3<N>> for UnitDualQuaternion<N> {
    type Scaling = Id;

    #[inline]
    fn translation(&self) -> Translation3<N> {
        self.translation()
    }

    #[inline]
    fn rotation(&self) -> UnitQuaternion<N> {
        self.rotation()
    }

    #[inline]
    fn scaling(&self) -> Id {
        Id::new()
    }
}

macro_rules! marker_impl(
    ($($Trait: ident),*) => {$(
        impl<N: RealField + simba::scalar::RealField> $Trait<Point3<N>> for UnitDualQuaternion<N> { }
    )*}
);

marker_impl!(Isometry, DirectIsometry);
