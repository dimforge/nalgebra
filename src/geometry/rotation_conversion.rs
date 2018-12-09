use num::Zero;

use alga::general::{Real, SubsetOf, SupersetOf};
use alga::linear::Rotation as AlgaRotation;

#[cfg(feature = "mint")]
use mint;

use base::allocator::Allocator;
use base::dimension::{DimMin, DimName, DimNameAdd, DimNameSum, U1};
use base::{DefaultAllocator, Matrix2, Matrix3, Matrix4, MatrixN};

use geometry::{
    Isometry, Point, Rotation, Rotation2, Rotation3, Similarity, SuperTCategoryOf, TAffine,
    Transform, Translation, UnitComplex, UnitQuaternion,
};

/*
 * This file provides the following conversions:
 * =============================================
 *
 * Rotation  -> Rotation
 * Rotation3 -> UnitQuaternion
 * Rotation2 -> UnitComplex
 * Rotation  -> Isometry
 * Rotation  -> Similarity
 * Rotation  -> Transform
 * Rotation  -> Matrix (homogeneous)
 * mint::EulerAngles -> Rotation

*/

impl<N1, N2, D: DimName> SubsetOf<Rotation<N2, D>> for Rotation<N1, D>
where
    N1: Real,
    N2: Real + SupersetOf<N1>,
    DefaultAllocator: Allocator<N1, D, D> + Allocator<N2, D, D>,
{
    #[inline]
    fn to_superset(&self) -> Rotation<N2, D> {
        Rotation::from_matrix_unchecked(self.matrix().to_superset())
    }

    #[inline]
    fn is_in_subset(rot: &Rotation<N2, D>) -> bool {
        ::is_convertible::<_, MatrixN<N1, D>>(rot.matrix())
    }

    #[inline]
    unsafe fn from_superset_unchecked(rot: &Rotation<N2, D>) -> Self {
        Rotation::from_matrix_unchecked(rot.matrix().to_subset_unchecked())
    }
}

impl<N1, N2> SubsetOf<UnitQuaternion<N2>> for Rotation3<N1>
where
    N1: Real,
    N2: Real + SupersetOf<N1>,
{
    #[inline]
    fn to_superset(&self) -> UnitQuaternion<N2> {
        let q = UnitQuaternion::<N1>::from_rotation_matrix(self);
        q.to_superset()
    }

    #[inline]
    fn is_in_subset(q: &UnitQuaternion<N2>) -> bool {
        ::is_convertible::<_, UnitQuaternion<N1>>(q)
    }

    #[inline]
    unsafe fn from_superset_unchecked(q: &UnitQuaternion<N2>) -> Self {
        let q: UnitQuaternion<N1> = ::convert_ref_unchecked(q);
        q.to_rotation_matrix()
    }
}

impl<N1, N2> SubsetOf<UnitComplex<N2>> for Rotation2<N1>
where
    N1: Real,
    N2: Real + SupersetOf<N1>,
{
    #[inline]
    fn to_superset(&self) -> UnitComplex<N2> {
        let q = UnitComplex::<N1>::from_rotation_matrix(self);
        q.to_superset()
    }

    #[inline]
    fn is_in_subset(q: &UnitComplex<N2>) -> bool {
        ::is_convertible::<_, UnitComplex<N1>>(q)
    }

    #[inline]
    unsafe fn from_superset_unchecked(q: &UnitComplex<N2>) -> Self {
        let q: UnitComplex<N1> = ::convert_ref_unchecked(q);
        q.to_rotation_matrix()
    }
}

impl<N1, N2, D: DimName, R> SubsetOf<Isometry<N2, D, R>> for Rotation<N1, D>
where
    N1: Real,
    N2: Real + SupersetOf<N1>,
    R: AlgaRotation<Point<N2, D>> + SupersetOf<Rotation<N1, D>>,
    DefaultAllocator: Allocator<N1, D, D> + Allocator<N2, D>,
{
    #[inline]
    fn to_superset(&self) -> Isometry<N2, D, R> {
        Isometry::from_parts(Translation::identity(), ::convert_ref(self))
    }

    #[inline]
    fn is_in_subset(iso: &Isometry<N2, D, R>) -> bool {
        iso.translation.vector.is_zero()
    }

    #[inline]
    unsafe fn from_superset_unchecked(iso: &Isometry<N2, D, R>) -> Self {
        ::convert_ref_unchecked(&iso.rotation)
    }
}

impl<N1, N2, D: DimName, R> SubsetOf<Similarity<N2, D, R>> for Rotation<N1, D>
where
    N1: Real,
    N2: Real + SupersetOf<N1>,
    R: AlgaRotation<Point<N2, D>> + SupersetOf<Rotation<N1, D>>,
    DefaultAllocator: Allocator<N1, D, D> + Allocator<N2, D>,
{
    #[inline]
    fn to_superset(&self) -> Similarity<N2, D, R> {
        Similarity::from_parts(Translation::identity(), ::convert_ref(self), N2::one())
    }

    #[inline]
    fn is_in_subset(sim: &Similarity<N2, D, R>) -> bool {
        sim.isometry.translation.vector.is_zero() && sim.scaling() == N2::one()
    }

    #[inline]
    unsafe fn from_superset_unchecked(sim: &Similarity<N2, D, R>) -> Self {
        ::convert_ref_unchecked(&sim.isometry.rotation)
    }
}

impl<N1, N2, D, C> SubsetOf<Transform<N2, D, C>> for Rotation<N1, D>
where
    N1: Real,
    N2: Real + SupersetOf<N1>,
    C: SuperTCategoryOf<TAffine>,
    D: DimNameAdd<U1> + DimMin<D, Output = D>, // needed by .is_special_orthogonal()
    DefaultAllocator: Allocator<N1, D, D>
        + Allocator<N2, D, D>
        + Allocator<N1, DimNameSum<D, U1>, DimNameSum<D, U1>>
        + Allocator<N2, DimNameSum<D, U1>, DimNameSum<D, U1>>
        + Allocator<(usize, usize), D>,
{
    // needed by .is_special_orthogonal()
    #[inline]
    fn to_superset(&self) -> Transform<N2, D, C> {
        Transform::from_matrix_unchecked(self.to_homogeneous().to_superset())
    }

    #[inline]
    fn is_in_subset(t: &Transform<N2, D, C>) -> bool {
        <Self as SubsetOf<_>>::is_in_subset(t.matrix())
    }

    #[inline]
    unsafe fn from_superset_unchecked(t: &Transform<N2, D, C>) -> Self {
        Self::from_superset_unchecked(t.matrix())
    }
}

impl<N1, N2, D> SubsetOf<MatrixN<N2, DimNameSum<D, U1>>> for Rotation<N1, D>
where
    N1: Real,
    N2: Real + SupersetOf<N1>,
    D: DimNameAdd<U1> + DimMin<D, Output = D>, // needed by .is_special_orthogonal()
    DefaultAllocator: Allocator<N1, D, D>
        + Allocator<N2, D, D>
        + Allocator<N1, DimNameSum<D, U1>, DimNameSum<D, U1>>
        + Allocator<N2, DimNameSum<D, U1>, DimNameSum<D, U1>>
        + Allocator<(usize, usize), D>,
{
    // needed by .is_special_orthogonal()
    #[inline]
    fn to_superset(&self) -> MatrixN<N2, DimNameSum<D, U1>> {
        self.to_homogeneous().to_superset()
    }

    #[inline]
    fn is_in_subset(m: &MatrixN<N2, DimNameSum<D, U1>>) -> bool {
        let rot = m.fixed_slice::<D, D>(0, 0);
        let bottom = m.fixed_slice::<U1, D>(D::dim(), 0);

        // Scalar types agree.
        m.iter().all(|e| SupersetOf::<N1>::is_in_subset(e)) &&
        // The block part is a rotation.
        rot.is_special_orthogonal(N2::default_epsilon() * ::convert(100.0)) &&
        // The bottom row is (0, 0, ..., 1)
        bottom.iter().all(|e| e.is_zero()) && m[(D::dim(), D::dim())] == N2::one()
    }

    #[inline]
    unsafe fn from_superset_unchecked(m: &MatrixN<N2, DimNameSum<D, U1>>) -> Self {
        let r = m.fixed_slice::<D, D>(0, 0);
        Self::from_matrix_unchecked(::convert_unchecked(r.into_owned()))
    }
}

#[cfg(feature = "mint")]
impl<N: Real> From<mint::EulerAngles<N, mint::IntraXYZ>> for Rotation3<N> {
    fn from(euler: mint::EulerAngles<N, mint::IntraXYZ>) -> Self {
        Self::from_euler_angles(euler.a, euler.b, euler.c)
    }
}

impl<N: Real> From<Rotation2<N>> for Matrix3<N> {
    #[inline]
    fn from(q: Rotation2<N>) -> Matrix3<N> {
        q.to_homogeneous()
    }
}

impl<N: Real> From<Rotation2<N>> for Matrix2<N> {
    #[inline]
    fn from(q: Rotation2<N>) -> Matrix2<N> {
        q.into_inner()
    }
}

impl<N: Real> From<Rotation3<N>> for Matrix4<N> {
    #[inline]
    fn from(q: Rotation3<N>) -> Matrix4<N> {
        q.to_homogeneous()
    }
}

impl<N: Real> From<Rotation3<N>> for Matrix3<N> {
    #[inline]
    fn from(q: Rotation3<N>) -> Matrix3<N> {
        q.into_inner()
    }
}
