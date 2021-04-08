use simba::scalar::{RealField, SubsetOf, SupersetOf};
use simba::simd::{PrimitiveSimdValue, SimdRealField, SimdValue};

use crate::base::allocator::Allocator;
use crate::base::dimension::{DimMin, DimNameAdd, DimNameSum, U1};
use crate::base::{Const, DefaultAllocator, MatrixN, Scalar};

use crate::geometry::{
    AbstractRotation, Isometry, Isometry3, Similarity, SuperTCategoryOf, TAffine, Transform,
    Translation, UnitDualQuaternion, UnitQuaternion,
};

/*
 * This file provides the following conversions:
 * =============================================
 *
 * Isometry -> Isometry
 * Isometry3 -> UnitDualQuaternion
 * Isometry -> Similarity
 * Isometry -> Transform
 * Isometry -> Matrix (homogeneous)
 */

impl<N1, N2, R1, R2, const D: usize> SubsetOf<Isometry<N2, R2, D>> for Isometry<N1, R1, D>
where
    N1: RealField,
    N2: RealField + SupersetOf<N1>,
    R1: AbstractRotation<N1, D> + SubsetOf<R2>,
    R2: AbstractRotation<N2, D>,
{
    #[inline]
    fn to_superset(&self) -> Isometry<N2, R2, D> {
        Isometry::from_parts(self.translation.to_superset(), self.rotation.to_superset())
    }

    #[inline]
    fn is_in_subset(iso: &Isometry<N2, R2, D>) -> bool {
        crate::is_convertible::<_, Translation<N1, D>>(&iso.translation)
            && crate::is_convertible::<_, R1>(&iso.rotation)
    }

    #[inline]
    fn from_superset_unchecked(iso: &Isometry<N2, R2, D>) -> Self {
        Isometry::from_parts(
            iso.translation.to_subset_unchecked(),
            iso.rotation.to_subset_unchecked(),
        )
    }
}

impl<N1, N2> SubsetOf<UnitDualQuaternion<N2>> for Isometry3<N1>
where
    N1: RealField,
    N2: RealField + SupersetOf<N1>,
{
    #[inline]
    fn to_superset(&self) -> UnitDualQuaternion<N2> {
        let dq = UnitDualQuaternion::<N1>::from_isometry(self);
        dq.to_superset()
    }

    #[inline]
    fn is_in_subset(dq: &UnitDualQuaternion<N2>) -> bool {
        crate::is_convertible::<_, UnitQuaternion<N1>>(&dq.rotation())
            && crate::is_convertible::<_, Translation<N1, 3>>(&dq.translation())
    }

    #[inline]
    fn from_superset_unchecked(dq: &UnitDualQuaternion<N2>) -> Self {
        let dq: UnitDualQuaternion<N1> = crate::convert_ref_unchecked(dq);
        dq.to_isometry()
    }
}

impl<N1, N2, R1, R2, const D: usize> SubsetOf<Similarity<N2, R2, D>> for Isometry<N1, R1, D>
where
    N1: RealField,
    N2: RealField + SupersetOf<N1>,
    R1: AbstractRotation<N1, D> + SubsetOf<R2>,
    R2: AbstractRotation<N2, D>,
{
    #[inline]
    fn to_superset(&self) -> Similarity<N2, R2, D> {
        Similarity::from_isometry(self.to_superset(), N2::one())
    }

    #[inline]
    fn is_in_subset(sim: &Similarity<N2, R2, D>) -> bool {
        crate::is_convertible::<_, Isometry<N1, R1, D>>(&sim.isometry) && sim.scaling() == N2::one()
    }

    #[inline]
    fn from_superset_unchecked(sim: &Similarity<N2, R2, D>) -> Self {
        crate::convert_ref_unchecked(&sim.isometry)
    }
}

impl<N1, N2, R, C, const D: usize> SubsetOf<Transform<N2, C, D>> for Isometry<N1, R, D>
where
    N1: RealField,
    N2: RealField + SupersetOf<N1>,
    C: SuperTCategoryOf<TAffine>,
    R: AbstractRotation<N1, D>
        + SubsetOf<MatrixN<N1, DimNameSum<Const<D>, U1>>>
        + SubsetOf<MatrixN<N2, DimNameSum<Const<D>, U1>>>,
    Const<D>: DimNameAdd<U1> + DimMin<Const<D>, Output = Const<D>>, // needed by .is_special_orthogonal()
    DefaultAllocator: Allocator<N1, DimNameSum<Const<D>, U1>, DimNameSum<Const<D>, U1>>
        + Allocator<N2, DimNameSum<Const<D>, U1>, DimNameSum<Const<D>, U1>>
        + Allocator<N2, DimNameSum<Const<D>, U1>, DimNameSum<Const<D>, U1>>,
    // + Allocator<N1, D>
    // + Allocator<(usize, usize), D>
    // + Allocator<N2, D, D>
    // + Allocator<N2, D>
{
    #[inline]
    fn to_superset(&self) -> Transform<N2, C, D> {
        Transform::from_matrix_unchecked(self.to_homogeneous().to_superset())
    }

    #[inline]
    fn is_in_subset(t: &Transform<N2, C, D>) -> bool {
        <Self as SubsetOf<_>>::is_in_subset(t.matrix())
    }

    #[inline]
    fn from_superset_unchecked(t: &Transform<N2, C, D>) -> Self {
        Self::from_superset_unchecked(t.matrix())
    }
}

impl<N1, N2, R, const D: usize> SubsetOf<MatrixN<N2, DimNameSum<Const<D>, U1>>>
    for Isometry<N1, R, D>
where
    N1: RealField,
    N2: RealField + SupersetOf<N1>,
    R: AbstractRotation<N1, D>
        + SubsetOf<MatrixN<N1, DimNameSum<Const<D>, U1>>>
        + SubsetOf<MatrixN<N2, DimNameSum<Const<D>, U1>>>,
    Const<D>: DimNameAdd<U1> + DimMin<Const<D>, Output = Const<D>>, // needed by .is_special_orthogonal()
    DefaultAllocator: Allocator<N1, DimNameSum<Const<D>, U1>, DimNameSum<Const<D>, U1>>
        + Allocator<N2, DimNameSum<Const<D>, U1>, DimNameSum<Const<D>, U1>>
        + Allocator<N2, DimNameSum<Const<D>, U1>, DimNameSum<Const<D>, U1>>, // + Allocator<(usize, usize), D>
                                                                             // + Allocator<N2, D, D>
                                                                             // + Allocator<N2, D>
                                                                             // + Allocator<N1, D>
                                                                             // + Allocator<N1, D, D>
{
    #[inline]
    fn to_superset(&self) -> MatrixN<N2, DimNameSum<Const<D>, U1>> {
        self.to_homogeneous().to_superset()
    }

    #[inline]
    fn is_in_subset(m: &MatrixN<N2, DimNameSum<Const<D>, U1>>) -> bool {
        let rot = m.fixed_slice::<Const<D>, Const<D>>(0, 0);
        let bottom = m.fixed_slice::<U1, Const<D>>(D, 0);

        // Scalar types agree.
        m.iter().all(|e| SupersetOf::<N1>::is_in_subset(e)) &&
        // The block part is a rotation.
        rot.is_special_orthogonal(N2::default_epsilon() * crate::convert(100.0)) &&
        // The bottom row is (0, 0, ..., 1)
        bottom.iter().all(|e| e.is_zero()) && m[(D, D)] == N2::one()
    }

    #[inline]
    fn from_superset_unchecked(m: &MatrixN<N2, DimNameSum<Const<D>, U1>>) -> Self {
        let t = m.fixed_slice::<Const<D>, U1>(0, D).into_owned();
        let t = Translation {
            vector: crate::convert_unchecked(t),
        };

        Self::from_parts(t, crate::convert_unchecked(m.clone_owned()))
    }
}

impl<N: SimdRealField, R: AbstractRotation<N, D>, const D: usize> From<Translation<N, D>>
    for Isometry<N, R, D>
// where
//     DefaultAllocator: Allocator<N, D>,
{
    #[inline]
    fn from(tra: Translation<N, D>) -> Self {
        Self::from_parts(tra, R::identity())
    }
}

impl<N: SimdRealField, R, const D: usize> From<Isometry<N, R, D>>
    for MatrixN<N, DimNameSum<Const<D>, U1>>
where
    Const<D>: DimNameAdd<U1>,
    R: SubsetOf<MatrixN<N, DimNameSum<Const<D>, U1>>>,
    DefaultAllocator: Allocator<N, DimNameSum<Const<D>, U1>, DimNameSum<Const<D>, U1>>, // + Allocator<N, D>,
{
    #[inline]
    fn from(iso: Isometry<N, R, D>) -> Self {
        iso.to_homogeneous()
    }
}

impl<N: Scalar + PrimitiveSimdValue, R, const D: usize>
    From<[Isometry<N::Element, R::Element, D>; 2]> for Isometry<N, R, D>
where
    N: From<[<N as SimdValue>::Element; 2]>,
    R: SimdValue + AbstractRotation<N, D> + From<[<R as SimdValue>::Element; 2]>,
    R::Element: AbstractRotation<N::Element, D>,
    N::Element: Scalar + Copy,
    R::Element: Scalar + Copy,
{
    #[inline]
    fn from(arr: [Isometry<N::Element, R::Element, D>; 2]) -> Self {
        let tra = Translation::from([arr[0].translation.clone(), arr[1].translation.clone()]);
        let rot = R::from([arr[0].rotation, arr[0].rotation]);

        Self::from_parts(tra, rot)
    }
}

impl<N: Scalar + PrimitiveSimdValue, R, const D: usize>
    From<[Isometry<N::Element, R::Element, D>; 4]> for Isometry<N, R, D>
where
    N: From<[<N as SimdValue>::Element; 4]>,
    R: SimdValue + AbstractRotation<N, D> + From<[<R as SimdValue>::Element; 4]>,
    R::Element: AbstractRotation<N::Element, D>,
    N::Element: Scalar + Copy,
    R::Element: Scalar + Copy,
{
    #[inline]
    fn from(arr: [Isometry<N::Element, R::Element, D>; 4]) -> Self {
        let tra = Translation::from([
            arr[0].translation.clone(),
            arr[1].translation.clone(),
            arr[2].translation.clone(),
            arr[3].translation.clone(),
        ]);
        let rot = R::from([
            arr[0].rotation,
            arr[1].rotation,
            arr[2].rotation,
            arr[3].rotation,
        ]);

        Self::from_parts(tra, rot)
    }
}

impl<N: Scalar + PrimitiveSimdValue, R, const D: usize>
    From<[Isometry<N::Element, R::Element, D>; 8]> for Isometry<N, R, D>
where
    N: From<[<N as SimdValue>::Element; 8]>,
    R: SimdValue + AbstractRotation<N, D> + From<[<R as SimdValue>::Element; 8]>,
    R::Element: AbstractRotation<N::Element, D>,
    N::Element: Scalar + Copy,
    R::Element: Scalar + Copy,
{
    #[inline]
    fn from(arr: [Isometry<N::Element, R::Element, D>; 8]) -> Self {
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
            arr[0].rotation,
            arr[1].rotation,
            arr[2].rotation,
            arr[3].rotation,
            arr[4].rotation,
            arr[5].rotation,
            arr[6].rotation,
            arr[7].rotation,
        ]);

        Self::from_parts(tra, rot)
    }
}

impl<N: Scalar + PrimitiveSimdValue, R, const D: usize>
    From<[Isometry<N::Element, R::Element, D>; 16]> for Isometry<N, R, D>
where
    N: From<[<N as SimdValue>::Element; 16]>,
    R: SimdValue + AbstractRotation<N, D> + From<[<R as SimdValue>::Element; 16]>,
    R::Element: AbstractRotation<N::Element, D>,
    N::Element: Scalar + Copy,
    R::Element: Scalar + Copy,
{
    #[inline]
    fn from(arr: [Isometry<N::Element, R::Element, D>; 16]) -> Self {
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
            arr[0].rotation,
            arr[1].rotation,
            arr[2].rotation,
            arr[3].rotation,
            arr[4].rotation,
            arr[5].rotation,
            arr[6].rotation,
            arr[7].rotation,
            arr[8].rotation,
            arr[9].rotation,
            arr[10].rotation,
            arr[11].rotation,
            arr[12].rotation,
            arr[13].rotation,
            arr[14].rotation,
            arr[15].rotation,
        ]);

        Self::from_parts(tra, rot)
    }
}
