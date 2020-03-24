use simba::scalar::{RealField, SubsetOf, SupersetOf};
use simba::simd::{PrimitiveSimdValue, SimdRealField, SimdValue};

use crate::base::allocator::Allocator;
use crate::base::dimension::{DimMin, DimName, DimNameAdd, DimNameSum, U1};
use crate::base::{DefaultAllocator, MatrixN, Scalar};

use crate::geometry::{
    AbstractRotation, Isometry, Similarity, SuperTCategoryOf, TAffine, Transform, Translation,
};

/*
 * This file provides the following conversions:
 * =============================================
 *
 * Isometry -> Isometry
 * Isometry -> Similarity
 * Isometry -> Transform
 * Isometry -> Matrix (homogeneous)
 */

impl<N1, N2, D: DimName, R1, R2> SubsetOf<Isometry<N2, D, R2>> for Isometry<N1, D, R1>
where
    N1: RealField,
    N2: RealField + SupersetOf<N1>,
    R1: AbstractRotation<N1, D> + SubsetOf<R2>,
    R2: AbstractRotation<N2, D>,
    DefaultAllocator: Allocator<N1, D> + Allocator<N2, D>,
{
    #[inline]
    fn to_superset(&self) -> Isometry<N2, D, R2> {
        Isometry::from_parts(self.translation.to_superset(), self.rotation.to_superset())
    }

    #[inline]
    fn is_in_subset(iso: &Isometry<N2, D, R2>) -> bool {
        crate::is_convertible::<_, Translation<N1, D>>(&iso.translation)
            && crate::is_convertible::<_, R1>(&iso.rotation)
    }

    #[inline]
    fn from_superset_unchecked(iso: &Isometry<N2, D, R2>) -> Self {
        Isometry::from_parts(
            iso.translation.to_subset_unchecked(),
            iso.rotation.to_subset_unchecked(),
        )
    }
}

impl<N1, N2, D: DimName, R1, R2> SubsetOf<Similarity<N2, D, R2>> for Isometry<N1, D, R1>
where
    N1: RealField,
    N2: RealField + SupersetOf<N1>,
    R1: AbstractRotation<N1, D> + SubsetOf<R2>,
    R2: AbstractRotation<N2, D>,
    DefaultAllocator: Allocator<N1, D> + Allocator<N2, D>,
{
    #[inline]
    fn to_superset(&self) -> Similarity<N2, D, R2> {
        Similarity::from_isometry(self.to_superset(), N2::one())
    }

    #[inline]
    fn is_in_subset(sim: &Similarity<N2, D, R2>) -> bool {
        crate::is_convertible::<_, Isometry<N1, D, R1>>(&sim.isometry) && sim.scaling() == N2::one()
    }

    #[inline]
    fn from_superset_unchecked(sim: &Similarity<N2, D, R2>) -> Self {
        crate::convert_ref_unchecked(&sim.isometry)
    }
}

impl<N1, N2, D, R, C> SubsetOf<Transform<N2, D, C>> for Isometry<N1, D, R>
where
    N1: RealField,
    N2: RealField + SupersetOf<N1>,
    C: SuperTCategoryOf<TAffine>,
    R: AbstractRotation<N1, D>
        + SubsetOf<MatrixN<N1, DimNameSum<D, U1>>>
        + SubsetOf<MatrixN<N2, DimNameSum<D, U1>>>,
    D: DimNameAdd<U1> + DimMin<D, Output = D>, // needed by .is_special_orthogonal()
    DefaultAllocator: Allocator<N1, D>
        + Allocator<N1, D, D>
        + Allocator<N1, DimNameSum<D, U1>, DimNameSum<D, U1>>
        + Allocator<N2, DimNameSum<D, U1>, DimNameSum<D, U1>>
        + Allocator<N2, DimNameSum<D, U1>, DimNameSum<D, U1>>
        + Allocator<(usize, usize), D>
        + Allocator<N2, D, D>
        + Allocator<N2, D>,
{
    #[inline]
    fn to_superset(&self) -> Transform<N2, D, C> {
        Transform::from_matrix_unchecked(self.to_homogeneous().to_superset())
    }

    #[inline]
    fn is_in_subset(t: &Transform<N2, D, C>) -> bool {
        <Self as SubsetOf<_>>::is_in_subset(t.matrix())
    }

    #[inline]
    fn from_superset_unchecked(t: &Transform<N2, D, C>) -> Self {
        Self::from_superset_unchecked(t.matrix())
    }
}

impl<N1, N2, D, R> SubsetOf<MatrixN<N2, DimNameSum<D, U1>>> for Isometry<N1, D, R>
where
    N1: RealField,
    N2: RealField + SupersetOf<N1>,
    R: AbstractRotation<N1, D>
        + SubsetOf<MatrixN<N1, DimNameSum<D, U1>>>
        + SubsetOf<MatrixN<N2, DimNameSum<D, U1>>>,
    D: DimNameAdd<U1> + DimMin<D, Output = D>, // needed by .is_special_orthogonal()
    DefaultAllocator: Allocator<N1, D>
        + Allocator<N1, D, D>
        + Allocator<N1, DimNameSum<D, U1>, DimNameSum<D, U1>>
        + Allocator<N2, DimNameSum<D, U1>, DimNameSum<D, U1>>
        + Allocator<N2, DimNameSum<D, U1>, DimNameSum<D, U1>>
        + Allocator<(usize, usize), D>
        + Allocator<N2, D, D>
        + Allocator<N2, D>,
{
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
        rot.is_special_orthogonal(N2::default_epsilon() * crate::convert(100.0)) &&
        // The bottom row is (0, 0, ..., 1)
        bottom.iter().all(|e| e.is_zero()) && m[(D::dim(), D::dim())] == N2::one()
    }

    #[inline]
    fn from_superset_unchecked(m: &MatrixN<N2, DimNameSum<D, U1>>) -> Self {
        let t = m.fixed_slice::<D, U1>(0, D::dim()).into_owned();
        let t = Translation {
            vector: crate::convert_unchecked(t),
        };

        Self::from_parts(t, crate::convert_unchecked(m.clone_owned()))
    }
}

impl<N: SimdRealField, D: DimName, R> From<Isometry<N, D, R>> for MatrixN<N, DimNameSum<D, U1>>
where
    D: DimNameAdd<U1>,
    R: SubsetOf<MatrixN<N, DimNameSum<D, U1>>>,
    DefaultAllocator: Allocator<N, DimNameSum<D, U1>, DimNameSum<D, U1>> + Allocator<N, D>,
{
    #[inline]
    fn from(iso: Isometry<N, D, R>) -> Self {
        iso.to_homogeneous()
    }
}

impl<N: Scalar + PrimitiveSimdValue, D: DimName, R> From<[Isometry<N::Element, D, R::Element>; 2]>
    for Isometry<N, D, R>
where
    N: From<[<N as SimdValue>::Element; 2]>,
    R: SimdValue + AbstractRotation<N, D> + From<[<R as SimdValue>::Element; 2]>,
    R::Element: AbstractRotation<N::Element, D>,
    N::Element: Scalar + Copy,
    R::Element: Scalar + Copy,
    DefaultAllocator: Allocator<N, D> + Allocator<N::Element, D>,
{
    #[inline]
    fn from(arr: [Isometry<N::Element, D, R::Element>; 2]) -> Self {
        let tra = Translation::from([arr[0].translation.clone(), arr[1].translation.clone()]);
        let rot = R::from([arr[0].rotation.clone(), arr[0].rotation.clone()]);

        Self::from_parts(tra, rot)
    }
}

impl<N: Scalar + PrimitiveSimdValue, D: DimName, R> From<[Isometry<N::Element, D, R::Element>; 4]>
    for Isometry<N, D, R>
where
    N: From<[<N as SimdValue>::Element; 4]>,
    R: SimdValue + AbstractRotation<N, D> + From<[<R as SimdValue>::Element; 4]>,
    R::Element: AbstractRotation<N::Element, D>,
    N::Element: Scalar + Copy,
    R::Element: Scalar + Copy,
    DefaultAllocator: Allocator<N, D> + Allocator<N::Element, D>,
{
    #[inline]
    fn from(arr: [Isometry<N::Element, D, R::Element>; 4]) -> Self {
        let tra = Translation::from([
            arr[0].translation.clone(),
            arr[1].translation.clone(),
            arr[2].translation.clone(),
            arr[3].translation.clone(),
        ]);
        let rot = R::from([
            arr[0].rotation.clone(),
            arr[1].rotation.clone(),
            arr[2].rotation.clone(),
            arr[3].rotation.clone(),
        ]);

        Self::from_parts(tra, rot)
    }
}

impl<N: Scalar + PrimitiveSimdValue, D: DimName, R> From<[Isometry<N::Element, D, R::Element>; 8]>
    for Isometry<N, D, R>
where
    N: From<[<N as SimdValue>::Element; 8]>,
    R: SimdValue + AbstractRotation<N, D> + From<[<R as SimdValue>::Element; 8]>,
    R::Element: AbstractRotation<N::Element, D>,
    N::Element: Scalar + Copy,
    R::Element: Scalar + Copy,
    DefaultAllocator: Allocator<N, D> + Allocator<N::Element, D>,
{
    #[inline]
    fn from(arr: [Isometry<N::Element, D, R::Element>; 8]) -> Self {
        let tra = Translation::from([
            arr[0].translation.clone(),
            arr[1].translation.clone(),
            arr[2].translation.clone(),
            arr[3].translation.clone(),
            arr[4].translation.clone(),
            arr[5].translation.clone(),
            arr[6].translation.clone(),
            arr[7].translation.clone(),
        ]);
        let rot = R::from([
            arr[0].rotation.clone(),
            arr[1].rotation.clone(),
            arr[2].rotation.clone(),
            arr[3].rotation.clone(),
            arr[4].rotation.clone(),
            arr[5].rotation.clone(),
            arr[6].rotation.clone(),
            arr[7].rotation.clone(),
        ]);

        Self::from_parts(tra, rot)
    }
}

impl<N: Scalar + PrimitiveSimdValue, D: DimName, R> From<[Isometry<N::Element, D, R::Element>; 16]>
    for Isometry<N, D, R>
where
    N: From<[<N as SimdValue>::Element; 16]>,
    R: SimdValue + AbstractRotation<N, D> + From<[<R as SimdValue>::Element; 16]>,
    R::Element: AbstractRotation<N::Element, D>,
    N::Element: Scalar + Copy,
    R::Element: Scalar + Copy,
    DefaultAllocator: Allocator<N, D> + Allocator<N::Element, D>,
{
    #[inline]
    fn from(arr: [Isometry<N::Element, D, R::Element>; 16]) -> Self {
        let tra = Translation::from([
            arr[0].translation.clone(),
            arr[1].translation.clone(),
            arr[2].translation.clone(),
            arr[3].translation.clone(),
            arr[4].translation.clone(),
            arr[5].translation.clone(),
            arr[6].translation.clone(),
            arr[7].translation.clone(),
            arr[8].translation.clone(),
            arr[9].translation.clone(),
            arr[10].translation.clone(),
            arr[11].translation.clone(),
            arr[12].translation.clone(),
            arr[13].translation.clone(),
            arr[14].translation.clone(),
            arr[15].translation.clone(),
        ]);
        let rot = R::from([
            arr[0].rotation.clone(),
            arr[1].rotation.clone(),
            arr[2].rotation.clone(),
            arr[3].rotation.clone(),
            arr[4].rotation.clone(),
            arr[5].rotation.clone(),
            arr[6].rotation.clone(),
            arr[7].rotation.clone(),
            arr[8].rotation.clone(),
            arr[9].rotation.clone(),
            arr[10].rotation.clone(),
            arr[11].rotation.clone(),
            arr[12].rotation.clone(),
            arr[13].rotation.clone(),
            arr[14].rotation.clone(),
            arr[15].rotation.clone(),
        ]);

        Self::from_parts(tra, rot)
    }
}
