use num::Zero;
use num_complex::Complex;

use simba::scalar::{RealField, SubsetOf, SupersetOf};
use simba::simd::{PrimitiveSimdValue, SimdRealField};

use crate::base::dimension::U2;
use crate::base::{Matrix2, Matrix3, Scalar};
use crate::geometry::{
    AbstractRotation, Isometry, Rotation2, Similarity, SuperTCategoryOf, TAffine, Transform,
    Translation, UnitComplex,
};

/*
 * This file provides the following conversions:
 * =============================================
 *
 * UnitComplex -> UnitComplex
 * UnitComplex -> Rotation<U1>
 * UnitComplex -> Isometry<U2>
 * UnitComplex -> Similarity<U2>
 * UnitComplex -> Transform<U2>
 * UnitComplex -> Matrix<U3> (homogeneous)
 *
 * NOTE:
 * UnitComplex -> Complex is already provided by: Unit<T> -> T
 */

impl<N1, N2> SubsetOf<UnitComplex<N2>> for UnitComplex<N1>
where
    N1: RealField,
    N2: RealField + SupersetOf<N1>,
{
    #[inline]
    fn to_superset(&self) -> UnitComplex<N2> {
        UnitComplex::new_unchecked(self.as_ref().to_superset())
    }

    #[inline]
    fn is_in_subset(uq: &UnitComplex<N2>) -> bool {
        crate::is_convertible::<_, Complex<N1>>(uq.as_ref())
    }

    #[inline]
    fn from_superset_unchecked(uq: &UnitComplex<N2>) -> Self {
        Self::new_unchecked(crate::convert_ref_unchecked(uq.as_ref()))
    }
}

impl<N1, N2> SubsetOf<Rotation2<N2>> for UnitComplex<N1>
where
    N1: RealField,
    N2: RealField + SupersetOf<N1>,
{
    #[inline]
    fn to_superset(&self) -> Rotation2<N2> {
        let q: UnitComplex<N2> = self.to_superset();
        q.to_rotation_matrix().to_superset()
    }

    #[inline]
    fn is_in_subset(rot: &Rotation2<N2>) -> bool {
        crate::is_convertible::<_, Rotation2<N1>>(rot)
    }

    #[inline]
    fn from_superset_unchecked(rot: &Rotation2<N2>) -> Self {
        let q = UnitComplex::<N2>::from_rotation_matrix(rot);
        crate::convert_unchecked(q)
    }
}

impl<N1, N2, R> SubsetOf<Isometry<N2, U2, R>> for UnitComplex<N1>
where
    N1: RealField,
    N2: RealField + SupersetOf<N1>,
    R: AbstractRotation<N2, U2> + SupersetOf<Self>,
{
    #[inline]
    fn to_superset(&self) -> Isometry<N2, U2, R> {
        Isometry::from_parts(Translation::identity(), crate::convert_ref(self))
    }

    #[inline]
    fn is_in_subset(iso: &Isometry<N2, U2, R>) -> bool {
        iso.translation.vector.is_zero()
    }

    #[inline]
    fn from_superset_unchecked(iso: &Isometry<N2, U2, R>) -> Self {
        crate::convert_ref_unchecked(&iso.rotation)
    }
}

impl<N1, N2, R> SubsetOf<Similarity<N2, U2, R>> for UnitComplex<N1>
where
    N1: RealField,
    N2: RealField + SupersetOf<N1>,
    R: AbstractRotation<N2, U2> + SupersetOf<Self>,
{
    #[inline]
    fn to_superset(&self) -> Similarity<N2, U2, R> {
        Similarity::from_isometry(crate::convert_ref(self), N2::one())
    }

    #[inline]
    fn is_in_subset(sim: &Similarity<N2, U2, R>) -> bool {
        sim.isometry.translation.vector.is_zero() && sim.scaling() == N2::one()
    }

    #[inline]
    fn from_superset_unchecked(sim: &Similarity<N2, U2, R>) -> Self {
        crate::convert_ref_unchecked(&sim.isometry)
    }
}

impl<N1, N2, C> SubsetOf<Transform<N2, U2, C>> for UnitComplex<N1>
where
    N1: RealField,
    N2: RealField + SupersetOf<N1>,
    C: SuperTCategoryOf<TAffine>,
{
    #[inline]
    fn to_superset(&self) -> Transform<N2, U2, C> {
        Transform::from_matrix_unchecked(self.to_homogeneous().to_superset())
    }

    #[inline]
    fn is_in_subset(t: &Transform<N2, U2, C>) -> bool {
        <Self as SubsetOf<_>>::is_in_subset(t.matrix())
    }

    #[inline]
    fn from_superset_unchecked(t: &Transform<N2, U2, C>) -> Self {
        Self::from_superset_unchecked(t.matrix())
    }
}

impl<N1: RealField, N2: RealField + SupersetOf<N1>> SubsetOf<Matrix3<N2>> for UnitComplex<N1> {
    #[inline]
    fn to_superset(&self) -> Matrix3<N2> {
        self.to_homogeneous().to_superset()
    }

    #[inline]
    fn is_in_subset(m: &Matrix3<N2>) -> bool {
        crate::is_convertible::<_, Rotation2<N1>>(m)
    }

    #[inline]
    fn from_superset_unchecked(m: &Matrix3<N2>) -> Self {
        let rot: Rotation2<N1> = crate::convert_ref_unchecked(m);
        Self::from_rotation_matrix(&rot)
    }
}

impl<N: SimdRealField> From<UnitComplex<N>> for Rotation2<N>
where
    N::Element: SimdRealField,
{
    #[inline]
    fn from(q: UnitComplex<N>) -> Self {
        q.to_rotation_matrix()
    }
}

impl<N: SimdRealField> From<Rotation2<N>> for UnitComplex<N>
where
    N::Element: SimdRealField,
{
    #[inline]
    fn from(q: Rotation2<N>) -> Self {
        Self::from_rotation_matrix(&q)
    }
}

impl<N: SimdRealField> From<UnitComplex<N>> for Matrix3<N>
where
    N::Element: SimdRealField,
{
    #[inline]
    fn from(q: UnitComplex<N>) -> Matrix3<N> {
        q.to_homogeneous()
    }
}

impl<N: SimdRealField> From<UnitComplex<N>> for Matrix2<N>
where
    N::Element: SimdRealField,
{
    #[inline]
    fn from(q: UnitComplex<N>) -> Self {
        q.to_rotation_matrix().into_inner()
    }
}

impl<N: Scalar + Copy + PrimitiveSimdValue> From<[UnitComplex<N::Element>; 2]> for UnitComplex<N>
where
    N: From<[<N as simba::simd::SimdValue>::Element; 2]>,
    N::Element: Scalar + Copy,
{
    #[inline]
    fn from(arr: [UnitComplex<N::Element>; 2]) -> Self {
        Self::new_unchecked(Complex {
            re: N::from([arr[0].re, arr[1].re]),
            im: N::from([arr[0].im, arr[1].im]),
        })
    }
}

impl<N: Scalar + Copy + PrimitiveSimdValue> From<[UnitComplex<N::Element>; 4]> for UnitComplex<N>
where
    N: From<[<N as simba::simd::SimdValue>::Element; 4]>,
    N::Element: Scalar + Copy,
{
    #[inline]
    fn from(arr: [UnitComplex<N::Element>; 4]) -> Self {
        Self::new_unchecked(Complex {
            re: N::from([arr[0].re, arr[1].re, arr[2].re, arr[3].re]),
            im: N::from([arr[0].im, arr[1].im, arr[2].im, arr[3].im]),
        })
    }
}

impl<N: Scalar + Copy + PrimitiveSimdValue> From<[UnitComplex<N::Element>; 8]> for UnitComplex<N>
where
    N: From<[<N as simba::simd::SimdValue>::Element; 8]>,
    N::Element: Scalar + Copy,
{
    #[inline]
    fn from(arr: [UnitComplex<N::Element>; 8]) -> Self {
        Self::new_unchecked(Complex {
            re: N::from([
                arr[0].re, arr[1].re, arr[2].re, arr[3].re, arr[4].re, arr[5].re, arr[6].re,
                arr[7].re,
            ]),
            im: N::from([
                arr[0].im, arr[1].im, arr[2].im, arr[3].im, arr[4].im, arr[5].im, arr[6].im,
                arr[7].im,
            ]),
        })
    }
}

impl<N: Scalar + Copy + PrimitiveSimdValue> From<[UnitComplex<N::Element>; 16]> for UnitComplex<N>
where
    N: From<[<N as simba::simd::SimdValue>::Element; 16]>,
    N::Element: Scalar + Copy,
{
    #[inline]
    fn from(arr: [UnitComplex<N::Element>; 16]) -> Self {
        Self::new_unchecked(Complex {
            re: N::from([
                arr[0].re, arr[1].re, arr[2].re, arr[3].re, arr[4].re, arr[5].re, arr[6].re,
                arr[7].re, arr[8].re, arr[9].re, arr[10].re, arr[11].re, arr[12].re, arr[13].re,
                arr[14].re, arr[15].re,
            ]),
            im: N::from([
                arr[0].im, arr[1].im, arr[2].im, arr[3].im, arr[4].im, arr[5].im, arr[6].im,
                arr[7].im, arr[8].im, arr[9].im, arr[10].im, arr[11].im, arr[12].im, arr[13].im,
                arr[14].im, arr[15].im,
            ]),
        })
    }
}
