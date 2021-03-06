use num::{One, Zero};

use simba::scalar::{RealField, SubsetOf, SupersetOf};
use simba::simd::PrimitiveSimdValue;

use crate::base::allocator::Allocator;
use crate::base::dimension::{DimName, DimNameAdd, DimNameSum, U1};
use crate::base::{DefaultAllocator, MatrixN, Scalar, VectorN};

use crate::geometry::{
    AbstractRotation, Isometry, Similarity, SuperTCategoryOf, TAffine, Transform, Translation,
    Translation3, UnitDualQuaternion, UnitQuaternion,
};

/*
 * This file provides the following conversions:
 * =============================================
 *
 * Translation -> Translation
 * Translation -> Isometry
 * Translation3 -> UnitDualQuaternion
 * Translation -> Similarity
 * Translation -> Transform
 * Translation -> Matrix (homogeneous)
 */

impl<N1, N2, D: DimName> SubsetOf<Translation<N2, D>> for Translation<N1, D>
where
    N1: Scalar,
    N2: Scalar + SupersetOf<N1>,
    DefaultAllocator: Allocator<N1, D> + Allocator<N2, D>,
{
    #[inline]
    fn to_superset(&self) -> Translation<N2, D> {
        Translation::from(self.vector.to_superset())
    }

    #[inline]
    fn is_in_subset(rot: &Translation<N2, D>) -> bool {
        crate::is_convertible::<_, VectorN<N1, D>>(&rot.vector)
    }

    #[inline]
    fn from_superset_unchecked(rot: &Translation<N2, D>) -> Self {
        Translation {
            vector: rot.vector.to_subset_unchecked(),
        }
    }
}

impl<N1, N2, D: DimName, R> SubsetOf<Isometry<N2, D, R>> for Translation<N1, D>
where
    N1: RealField,
    N2: RealField + SupersetOf<N1>,
    R: AbstractRotation<N2, D>,
    DefaultAllocator: Allocator<N1, D> + Allocator<N2, D>,
{
    #[inline]
    fn to_superset(&self) -> Isometry<N2, D, R> {
        Isometry::from_parts(self.to_superset(), R::identity())
    }

    #[inline]
    fn is_in_subset(iso: &Isometry<N2, D, R>) -> bool {
        iso.rotation == R::identity()
    }

    #[inline]
    fn from_superset_unchecked(iso: &Isometry<N2, D, R>) -> Self {
        Self::from_superset_unchecked(&iso.translation)
    }
}

impl<N1, N2> SubsetOf<UnitDualQuaternion<N2>> for Translation3<N1>
where
    N1: RealField,
    N2: RealField + SupersetOf<N1>,
{
    #[inline]
    fn to_superset(&self) -> UnitDualQuaternion<N2> {
        let dq = UnitDualQuaternion::<N1>::from_parts(self.clone(), UnitQuaternion::identity());
        dq.to_superset()
    }

    #[inline]
    fn is_in_subset(dq: &UnitDualQuaternion<N2>) -> bool {
        crate::is_convertible::<_, Translation<N1, _>>(&dq.translation())
            && dq.rotation() == UnitQuaternion::identity()
    }

    #[inline]
    fn from_superset_unchecked(dq: &UnitDualQuaternion<N2>) -> Self {
        let dq: UnitDualQuaternion<N1> = crate::convert_ref_unchecked(dq);
        dq.translation()
    }
}

impl<N1, N2, D: DimName, R> SubsetOf<Similarity<N2, D, R>> for Translation<N1, D>
where
    N1: RealField,
    N2: RealField + SupersetOf<N1>,
    R: AbstractRotation<N2, D>,
    DefaultAllocator: Allocator<N1, D> + Allocator<N2, D>,
{
    #[inline]
    fn to_superset(&self) -> Similarity<N2, D, R> {
        Similarity::from_parts(self.to_superset(), R::identity(), N2::one())
    }

    #[inline]
    fn is_in_subset(sim: &Similarity<N2, D, R>) -> bool {
        sim.isometry.rotation == R::identity() && sim.scaling() == N2::one()
    }

    #[inline]
    fn from_superset_unchecked(sim: &Similarity<N2, D, R>) -> Self {
        Self::from_superset_unchecked(&sim.isometry.translation)
    }
}

impl<N1, N2, D, C> SubsetOf<Transform<N2, D, C>> for Translation<N1, D>
where
    N1: RealField,
    N2: RealField + SupersetOf<N1>,
    C: SuperTCategoryOf<TAffine>,
    D: DimNameAdd<U1>,
    DefaultAllocator: Allocator<N1, D>
        + Allocator<N2, D>
        + Allocator<N1, DimNameSum<D, U1>, DimNameSum<D, U1>>
        + Allocator<N2, DimNameSum<D, U1>, DimNameSum<D, U1>>,
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

impl<N1, N2, D> SubsetOf<MatrixN<N2, DimNameSum<D, U1>>> for Translation<N1, D>
where
    N1: RealField,
    N2: RealField + SupersetOf<N1>,
    D: DimNameAdd<U1>,
    DefaultAllocator: Allocator<N1, D>
        + Allocator<N2, D>
        + Allocator<N1, DimNameSum<D, U1>, DimNameSum<D, U1>>
        + Allocator<N2, DimNameSum<D, U1>, DimNameSum<D, U1>>,
{
    #[inline]
    fn to_superset(&self) -> MatrixN<N2, DimNameSum<D, U1>> {
        self.to_homogeneous().to_superset()
    }

    #[inline]
    fn is_in_subset(m: &MatrixN<N2, DimNameSum<D, U1>>) -> bool {
        let id = m.fixed_slice::<DimNameSum<D, U1>, D>(0, 0);

        // Scalar types agree.
        m.iter().all(|e| SupersetOf::<N1>::is_in_subset(e)) &&
        // The block part does nothing.
        id.is_identity(N2::zero()) &&
        // The normalization factor is one.
        m[(D::dim(), D::dim())] == N2::one()
    }

    #[inline]
    fn from_superset_unchecked(m: &MatrixN<N2, DimNameSum<D, U1>>) -> Self {
        let t = m.fixed_slice::<D, U1>(0, D::dim());
        Self {
            vector: crate::convert_unchecked(t.into_owned()),
        }
    }
}

impl<N: Scalar + Zero + One, D: DimName> From<Translation<N, D>> for MatrixN<N, DimNameSum<D, U1>>
where
    D: DimNameAdd<U1>,
    DefaultAllocator: Allocator<N, D> + Allocator<N, DimNameSum<D, U1>, DimNameSum<D, U1>>,
{
    #[inline]
    fn from(t: Translation<N, D>) -> Self {
        t.to_homogeneous()
    }
}

impl<N: Scalar, D: DimName> From<VectorN<N, D>> for Translation<N, D>
where
    DefaultAllocator: Allocator<N, D>,
{
    #[inline]
    fn from(vector: VectorN<N, D>) -> Self {
        Translation { vector }
    }
}

impl<N: Scalar + PrimitiveSimdValue, D: DimName> From<[Translation<N::Element, D>; 2]>
    for Translation<N, D>
where
    N: From<[<N as simba::simd::SimdValue>::Element; 2]>,
    N::Element: Scalar,
    DefaultAllocator: Allocator<N, D> + Allocator<N::Element, D>,
{
    #[inline]
    fn from(arr: [Translation<N::Element, D>; 2]) -> Self {
        Self::from(VectorN::from([
            arr[0].vector.clone(),
            arr[1].vector.clone(),
        ]))
    }
}

impl<N: Scalar + PrimitiveSimdValue, D: DimName> From<[Translation<N::Element, D>; 4]>
    for Translation<N, D>
where
    N: From<[<N as simba::simd::SimdValue>::Element; 4]>,
    N::Element: Scalar,
    DefaultAllocator: Allocator<N, D> + Allocator<N::Element, D>,
{
    #[inline]
    fn from(arr: [Translation<N::Element, D>; 4]) -> Self {
        Self::from(VectorN::from([
            arr[0].vector.clone(),
            arr[1].vector.clone(),
            arr[2].vector.clone(),
            arr[3].vector.clone(),
        ]))
    }
}

impl<N: Scalar + PrimitiveSimdValue, D: DimName> From<[Translation<N::Element, D>; 8]>
    for Translation<N, D>
where
    N: From<[<N as simba::simd::SimdValue>::Element; 8]>,
    N::Element: Scalar,
    DefaultAllocator: Allocator<N, D> + Allocator<N::Element, D>,
{
    #[inline]
    fn from(arr: [Translation<N::Element, D>; 8]) -> Self {
        Self::from(VectorN::from([
            arr[0].vector.clone(),
            arr[1].vector.clone(),
            arr[2].vector.clone(),
            arr[3].vector.clone(),
            arr[4].vector.clone(),
            arr[5].vector.clone(),
            arr[6].vector.clone(),
            arr[7].vector.clone(),
        ]))
    }
}

impl<N: Scalar + PrimitiveSimdValue, D: DimName> From<[Translation<N::Element, D>; 16]>
    for Translation<N, D>
where
    N: From<[<N as simba::simd::SimdValue>::Element; 16]>,
    N::Element: Scalar,
    DefaultAllocator: Allocator<N, D> + Allocator<N::Element, D>,
{
    #[inline]
    fn from(arr: [Translation<N::Element, D>; 16]) -> Self {
        Self::from(VectorN::from([
            arr[0].vector.clone(),
            arr[1].vector.clone(),
            arr[2].vector.clone(),
            arr[3].vector.clone(),
            arr[4].vector.clone(),
            arr[5].vector.clone(),
            arr[6].vector.clone(),
            arr[7].vector.clone(),
            arr[8].vector.clone(),
            arr[9].vector.clone(),
            arr[10].vector.clone(),
            arr[11].vector.clone(),
            arr[12].vector.clone(),
            arr[13].vector.clone(),
            arr[14].vector.clone(),
            arr[15].vector.clone(),
        ]))
    }
}
