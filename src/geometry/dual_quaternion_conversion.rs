use simba::scalar::{RealField, SubsetOf, SupersetOf};
use simba::simd::SimdRealField;

use crate::base::dimension::U3;
use crate::base::{Matrix4, Vector4};
use crate::geometry::{
    DualQuaternion, Isometry3, Similarity3, SuperTCategoryOf, TAffine, Transform, Translation3,
    UnitDualQuaternion, UnitQuaternion,
};

/*
 * This file provides the following conversions:
 * =============================================
 *
 * DualQuaternion     -> DualQuaternion
 * UnitDualQuaternion -> UnitDualQuaternion
 * UnitDualQuaternion -> Isometry<U3>
 * UnitDualQuaternion -> Similarity<U3>
 * UnitDualQuaternion -> Transform<U3>
 * UnitDualQuaternion -> Matrix<U4> (homogeneous)
 *
 * NOTE:
 * UnitDualQuaternion -> DualQuaternion is already provided by: Unit<T> -> T
 */

impl<N1, N2> SubsetOf<DualQuaternion<N2>> for DualQuaternion<N1>
where
    N1: SimdRealField,
    N2: SimdRealField + SupersetOf<N1>,
{
    #[inline]
    fn to_superset(&self) -> DualQuaternion<N2> {
        DualQuaternion::from_real_and_dual(self.real.to_superset(), self.dual.to_superset())
    }

    #[inline]
    fn is_in_subset(dq: &DualQuaternion<N2>) -> bool {
        crate::is_convertible::<_, Vector4<N1>>(&dq.real.coords)
            && crate::is_convertible::<_, Vector4<N1>>(&dq.dual.coords)
    }

    #[inline]
    fn from_superset_unchecked(dq: &DualQuaternion<N2>) -> Self {
        DualQuaternion::from_real_and_dual(
            dq.real.to_subset_unchecked(),
            dq.dual.to_subset_unchecked(),
        )
    }
}

impl<N1, N2> SubsetOf<UnitDualQuaternion<N2>> for UnitDualQuaternion<N1>
where
    N1: SimdRealField,
    N2: SimdRealField + SupersetOf<N1>,
{
    #[inline]
    fn to_superset(&self) -> UnitDualQuaternion<N2> {
        UnitDualQuaternion::new_unchecked(self.as_ref().to_superset())
    }

    #[inline]
    fn is_in_subset(dq: &UnitDualQuaternion<N2>) -> bool {
        crate::is_convertible::<_, DualQuaternion<N1>>(dq.as_ref())
    }

    #[inline]
    fn from_superset_unchecked(dq: &UnitDualQuaternion<N2>) -> Self {
        Self::new_unchecked(crate::convert_ref_unchecked(dq.as_ref()))
    }
}

impl<N1, N2> SubsetOf<Isometry3<N2>> for UnitDualQuaternion<N1>
where
    N1: RealField,
    N2: RealField + SupersetOf<N1>,
{
    #[inline]
    fn to_superset(&self) -> Isometry3<N2> {
        let dq: UnitDualQuaternion<N2> = self.to_superset();
        let iso = dq.to_isometry();
        crate::convert_unchecked(iso)
    }

    #[inline]
    fn is_in_subset(iso: &Isometry3<N2>) -> bool {
        crate::is_convertible::<_, UnitQuaternion<N1>>(&iso.rotation)
            && crate::is_convertible::<_, Translation3<N1>>(&iso.translation)
    }

    #[inline]
    fn from_superset_unchecked(iso: &Isometry3<N2>) -> Self {
        let dq = UnitDualQuaternion::<N2>::from_isometry(iso);
        crate::convert_unchecked(dq)
    }
}

impl<N1, N2> SubsetOf<Similarity3<N2>> for UnitDualQuaternion<N1>
where
    N1: RealField,
    N2: RealField + SupersetOf<N1>,
{
    #[inline]
    fn to_superset(&self) -> Similarity3<N2> {
        Similarity3::from_isometry(crate::convert_ref(self), N2::one())
    }

    #[inline]
    fn is_in_subset(sim: &Similarity3<N2>) -> bool {
        sim.scaling() == N2::one()
    }

    #[inline]
    fn from_superset_unchecked(sim: &Similarity3<N2>) -> Self {
        crate::convert_ref_unchecked(&sim.isometry)
    }
}

impl<N1, N2, C> SubsetOf<Transform<N2, U3, C>> for UnitDualQuaternion<N1>
where
    N1: RealField,
    N2: RealField + SupersetOf<N1>,
    C: SuperTCategoryOf<TAffine>,
{
    #[inline]
    fn to_superset(&self) -> Transform<N2, U3, C> {
        Transform::from_matrix_unchecked(self.to_homogeneous().to_superset())
    }

    #[inline]
    fn is_in_subset(t: &Transform<N2, U3, C>) -> bool {
        <Self as SubsetOf<_>>::is_in_subset(t.matrix())
    }

    #[inline]
    fn from_superset_unchecked(t: &Transform<N2, U3, C>) -> Self {
        Self::from_superset_unchecked(t.matrix())
    }
}

impl<N1: RealField, N2: RealField + SupersetOf<N1>> SubsetOf<Matrix4<N2>>
    for UnitDualQuaternion<N1>
{
    #[inline]
    fn to_superset(&self) -> Matrix4<N2> {
        self.to_homogeneous().to_superset()
    }

    #[inline]
    fn is_in_subset(m: &Matrix4<N2>) -> bool {
        crate::is_convertible::<_, Isometry3<N1>>(m)
    }

    #[inline]
    fn from_superset_unchecked(m: &Matrix4<N2>) -> Self {
        let iso: Isometry3<N1> = crate::convert_ref_unchecked(m);
        Self::from_isometry(&iso)
    }
}

impl<N: SimdRealField + RealField> From<UnitDualQuaternion<N>> for Matrix4<N>
where
    N::Element: SimdRealField,
{
    #[inline]
    fn from(dq: UnitDualQuaternion<N>) -> Self {
        dq.to_homogeneous()
    }
}

impl<N: SimdRealField> From<UnitDualQuaternion<N>> for Isometry3<N>
where
    N::Element: SimdRealField,
{
    #[inline]
    fn from(dq: UnitDualQuaternion<N>) -> Self {
        dq.to_isometry()
    }
}

impl<N: SimdRealField> From<Isometry3<N>> for UnitDualQuaternion<N>
where
    N::Element: SimdRealField,
{
    #[inline]
    fn from(iso: Isometry3<N>) -> Self {
        Self::from_isometry(&iso)
    }
}
