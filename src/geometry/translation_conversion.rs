use num::{One, Zero};

use simba::scalar::{RealField, SubsetOf, SupersetOf};
use simba::simd::PrimitiveSimdValue;

use crate::base::allocator::Allocator;
use crate::base::dimension::{DimNameAdd, DimNameSum, U1};
use crate::base::{Const, DefaultAllocator, DimName, OMatrix, OVector, SVector, Scalar};

use crate::geometry::{
    AbstractRotation, Isometry, Similarity, SuperTCategoryOf, TAffine, Transform, Translation,
    Translation3, UnitDualQuaternion, UnitQuaternion,
};
use crate::{ArrayStorage, Point};

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

impl<T1, T2, const D: usize> SubsetOf<Translation<T2, D>> for Translation<T1, D>
where
    T1: Scalar,
    T2: Scalar + SupersetOf<T1>,
{
    #[inline]
    fn to_superset(&self) -> Translation<T2, D> {
        Translation::from(self.vector.to_superset())
    }

    #[inline]
    fn is_in_subset(rot: &Translation<T2, D>) -> bool {
        crate::is_convertible::<_, SVector<T1, D>>(&rot.vector)
    }

    #[inline]
    fn from_superset_unchecked(rot: &Translation<T2, D>) -> Self {
        Translation {
            vector: rot.vector.to_subset_unchecked(),
        }
    }
}

impl<T1, T2, R, const D: usize> SubsetOf<Isometry<T2, R, D>> for Translation<T1, D>
where
    T1: RealField,
    T2: RealField + SupersetOf<T1>,
    R: AbstractRotation<T2, D>,
{
    #[inline]
    fn to_superset(&self) -> Isometry<T2, R, D> {
        Isometry::from_parts(self.to_superset(), R::identity())
    }

    #[inline]
    fn is_in_subset(iso: &Isometry<T2, R, D>) -> bool {
        iso.rotation == R::identity()
    }

    #[inline]
    fn from_superset_unchecked(iso: &Isometry<T2, R, D>) -> Self {
        Self::from_superset_unchecked(&iso.translation)
    }
}

impl<T1, T2> SubsetOf<UnitDualQuaternion<T2>> for Translation3<T1>
where
    T1: RealField,
    T2: RealField + SupersetOf<T1>,
{
    #[inline]
    fn to_superset(&self) -> UnitDualQuaternion<T2> {
        let dq = UnitDualQuaternion::<T1>::from_parts(self.clone(), UnitQuaternion::identity());
        dq.to_superset()
    }

    #[inline]
    fn is_in_subset(dq: &UnitDualQuaternion<T2>) -> bool {
        crate::is_convertible::<_, Translation<T1, 3>>(&dq.translation())
            && dq.rotation() == UnitQuaternion::identity()
    }

    #[inline]
    fn from_superset_unchecked(dq: &UnitDualQuaternion<T2>) -> Self {
        let dq: UnitDualQuaternion<T1> = crate::convert_ref_unchecked(dq);
        dq.translation()
    }
}

impl<T1, T2, R, const D: usize> SubsetOf<Similarity<T2, R, D>> for Translation<T1, D>
where
    T1: RealField,
    T2: RealField + SupersetOf<T1>,
    R: AbstractRotation<T2, D>,
{
    #[inline]
    fn to_superset(&self) -> Similarity<T2, R, D> {
        Similarity::from_parts(self.to_superset(), R::identity(), T2::one())
    }

    #[inline]
    fn is_in_subset(sim: &Similarity<T2, R, D>) -> bool {
        sim.isometry.rotation == R::identity() && sim.scaling() == T2::one()
    }

    #[inline]
    fn from_superset_unchecked(sim: &Similarity<T2, R, D>) -> Self {
        Self::from_superset_unchecked(&sim.isometry.translation)
    }
}

impl<T1, T2, C, const D: usize> SubsetOf<Transform<T2, C, D>> for Translation<T1, D>
where
    T1: RealField,
    T2: RealField + SupersetOf<T1>,
    C: SuperTCategoryOf<TAffine>,
    Const<D>: DimNameAdd<U1>,
    DefaultAllocator: Allocator<DimNameSum<Const<D>, U1>, DimNameSum<Const<D>, U1>>,
{
    #[inline]
    fn to_superset(&self) -> Transform<T2, C, D> {
        Transform::from_matrix_unchecked(self.to_homogeneous().to_superset())
    }

    #[inline]
    fn is_in_subset(t: &Transform<T2, C, D>) -> bool {
        <Self as SubsetOf<_>>::is_in_subset(t.matrix())
    }

    #[inline]
    fn from_superset_unchecked(t: &Transform<T2, C, D>) -> Self {
        Self::from_superset_unchecked(t.matrix())
    }
}

impl<T1, T2, const D: usize>
    SubsetOf<OMatrix<T2, DimNameSum<Const<D>, U1>, DimNameSum<Const<D>, U1>>> for Translation<T1, D>
where
    T1: RealField,
    T2: RealField + SupersetOf<T1>,
    Const<D>: DimNameAdd<U1>,
    DefaultAllocator: Allocator<Const<D>, Buffer<T1> = ArrayStorage<T1, D, 1>>
        + Allocator<DimNameSum<Const<D>, U1>, DimNameSum<Const<D>, U1>>,
{
    #[inline]
    fn to_superset(&self) -> OMatrix<T2, DimNameSum<Const<D>, U1>, DimNameSum<Const<D>, U1>> {
        self.to_homogeneous().to_superset()
    }

    #[inline]
    fn is_in_subset(m: &OMatrix<T2, DimNameSum<Const<D>, U1>, DimNameSum<Const<D>, U1>>) -> bool {
        let id = m.generic_view((0, 0), (DimNameSum::<Const<D>, U1>::name(), Const::<D>));

        // Scalar types agree.
        m.iter().all(|e| SupersetOf::<T1>::is_in_subset(e)) &&
        // The block part does nothing.
        id.is_identity(T2::zero()) &&
        // The normalization factor is one.
        m[(D, D)] == T2::one()
    }

    #[inline]
    fn from_superset_unchecked(
        m: &OMatrix<T2, DimNameSum<Const<D>, U1>, DimNameSum<Const<D>, U1>>,
    ) -> Self {
        let t: OVector<T2, Const<D>> = m.fixed_view::<D, 1>(0, D).into_owned();
        Self {
            vector: crate::convert_unchecked(t),
        }
    }
}

/// Converts a translation into its homogeneous matrix representation.
///
/// This implementation allows implicit conversion from a `Translation<T, D>` to a
/// homogeneous transformation matrix of size (D+1)×(D+1). The resulting matrix can
/// be multiplied with other transformation matrices.
///
/// # Examples
///
/// ```
/// # use nalgebra::{Translation3, Matrix4};
/// let t = Translation3::new(10.0, 20.0, 30.0);
///
/// // Implicit conversion to matrix
/// let matrix: Matrix4<f64> = t.into();
///
/// let expected = Matrix4::new(
///     1.0, 0.0, 0.0, 10.0,
///     0.0, 1.0, 0.0, 20.0,
///     0.0, 0.0, 1.0, 30.0,
///     0.0, 0.0, 0.0, 1.0
/// );
/// assert_eq!(matrix, expected);
/// ```
///
/// # See Also
///
/// * [`Translation::to_homogeneous`] - Explicit method for the same conversion
impl<T: Scalar + Zero + One, const D: usize> From<Translation<T, D>>
    for OMatrix<T, DimNameSum<Const<D>, U1>, DimNameSum<Const<D>, U1>>
where
    Const<D>: DimNameAdd<U1>,
    DefaultAllocator:
        Allocator<DimNameSum<Const<D>, U1>, DimNameSum<Const<D>, U1>> + Allocator<Const<D>>,
{
    #[inline]
    fn from(t: Translation<T, D>) -> Self {
        t.to_homogeneous()
    }
}

/// Creates a translation from a displacement vector.
///
/// This implementation allows you to convert a vector (which represents the displacement
/// in each dimension) directly into a translation. This is useful when you already have
/// the displacement calculated as a vector.
///
/// # Examples
///
/// ```
/// # use nalgebra::{Translation3, Vector3, Point3};
/// let displacement = Vector3::new(5.0, 10.0, 15.0);
///
/// // Convert vector to translation
/// let t: Translation3<f64> = displacement.into();
///
/// // Use it to move a point
/// let point = Point3::origin();
/// let moved = t * point;
/// assert_eq!(moved, Point3::new(5.0, 10.0, 15.0));
/// ```
///
/// Practical use - velocity to translation:
/// ```
/// # use nalgebra::{Translation2, Vector2, Point2};
/// let velocity = Vector2::new(2.0, 3.0);
/// let delta_time = 0.5;
///
/// // Calculate displacement
/// let displacement = velocity * delta_time;
///
/// // Convert to translation
/// let movement: Translation2<f64> = displacement.into();
///
/// let position = Point2::new(10.0, 20.0);
/// let new_position = movement * position;
/// assert_eq!(new_position, Point2::new(11.0, 21.5));
/// ```
///
/// # See Also
///
/// * [`Translation::new`] - Create from individual components
impl<T: Scalar, const D: usize> From<OVector<T, Const<D>>> for Translation<T, D> {
    #[inline]
    fn from(vector: OVector<T, Const<D>>) -> Self {
        Translation { vector }
    }
}

/// Creates a translation from an array of coordinates.
///
/// This implementation provides a convenient way to create a translation from a simple
/// array of values. The array length must match the dimensionality of the translation.
///
/// # Examples
///
/// Creating a 2D translation:
/// ```
/// # use nalgebra::{Translation2, Point2};
/// let coords = [5.0, 10.0];
/// let t: Translation2<f64> = coords.into();
///
/// assert_eq!(t, Translation2::new(5.0, 10.0));
///
/// let point = Point2::origin();
/// assert_eq!(t * point, Point2::new(5.0, 10.0));
/// ```
///
/// Creating a 3D translation:
/// ```
/// # use nalgebra::{Translation3, Point3};
/// let coords = [1.0, 2.0, 3.0];
/// let t: Translation3<f64> = coords.into();
///
/// assert_eq!(t, Translation3::new(1.0, 2.0, 3.0));
/// ```
///
/// Using in a function:
/// ```
/// # use nalgebra::{Translation2, Point2};
/// fn offset_position(pos: Point2<f64>, offset_array: [f64; 2]) -> Point2<f64> {
///     let translation: Translation2<f64> = offset_array.into();
///     translation * pos
/// }
///
/// let result = offset_position(Point2::new(10.0, 20.0), [5.0, -5.0]);
/// assert_eq!(result, Point2::new(15.0, 15.0));
/// ```
///
/// # See Also
///
/// * [`Translation::new`] - Explicit constructor with named parameters
/// * [`From<Vector>`] - Create from a vector type
impl<T: Scalar, const D: usize> From<[T; D]> for Translation<T, D> {
    #[inline]
    fn from(coords: [T; D]) -> Self {
        Translation {
            vector: coords.into(),
        }
    }
}

/// Creates a translation from a point, treating the point's coordinates as a displacement vector.
///
/// This implementation allows you to convert a `Point` into a `Translation` by using the
/// point's coordinates as the translation's displacement. This is useful when you want to
/// treat a position as an offset from the origin.
///
/// # Examples
///
/// Basic conversion:
/// ```
/// # use nalgebra::{Translation2, Point2};
/// let point = Point2::new(5.0, 10.0);
/// let t: Translation2<f64> = point.into();
///
/// assert_eq!(t, Translation2::new(5.0, 10.0));
/// ```
///
/// Using as a displacement:
/// ```
/// # use nalgebra::{Translation3, Point3};
/// // Treat a point as a displacement from origin
/// let target_position = Point3::new(100.0, 200.0, 300.0);
/// let translation: Translation3<f64> = target_position.into();
///
/// // Apply this displacement to the origin
/// let result = translation * Point3::origin();
/// assert_eq!(result, target_position);
/// ```
///
/// Practical use - converting between representations:
/// ```
/// # use nalgebra::{Translation2, Point2};
/// // Get offset between two coordinate systems
/// let origin_a = Point2::<f64>::origin();
/// let origin_b = Point2::new(50.0, 100.0);
///
/// // Create translation from second origin
/// let offset: Translation2<f64> = origin_b.into();
///
/// // Transform point from system A to system B
/// let point_in_a = Point2::new(10.0, 20.0);
/// let point_in_b = offset * point_in_a;
/// assert_eq!(point_in_b, Point2::new(60.0, 120.0));
/// ```
///
/// # See Also
///
/// * [`Translation::new`] - Create from individual components
/// * [`From<Vector>`] - Create from a displacement vector
impl<T: Scalar, const D: usize> From<Point<T, D>> for Translation<T, D> {
    #[inline]
    fn from(pt: Point<T, D>) -> Self {
        Translation { vector: pt.coords }
    }
}

/// Converts a translation into an array of its displacement components.
///
/// This implementation extracts the translation's displacement values into a simple array.
/// This is useful when you need to pass translation data to APIs that work with raw arrays,
/// or when you need to serialize or store the translation components.
///
/// # Examples
///
/// Basic conversion:
/// ```
/// # use nalgebra::Translation2;
/// let t = Translation2::new(5.0, 10.0);
/// let array: [f64; 2] = t.into();
/// assert_eq!(array, [5.0, 10.0]);
/// ```
///
/// Converting 3D translation:
/// ```
/// # use nalgebra::Translation3;
/// let t = Translation3::new(1.0, 2.0, 3.0);
/// let array: [f64; 3] = t.into();
/// assert_eq!(array, [1.0, 2.0, 3.0]);
/// ```
///
/// Using for serialization or storage:
/// ```
/// # use nalgebra::Translation2;
/// fn store_translation_data(t: Translation2<f32>) -> [f32; 2] {
///     t.into()
/// }
///
/// let translation = Translation2::new(100.0, 200.0);
/// let data = store_translation_data(translation);
/// assert_eq!(data, [100.0, 200.0]);
/// ```
///
/// # See Also
///
/// * [`From<[T; D]>`] for `Translation` - Reverse conversion from array to translation
impl<T: Scalar, const D: usize> From<Translation<T, D>> for [T; D] {
    #[inline]
    fn from(t: Translation<T, D>) -> Self {
        t.vector.into()
    }
}

/// Converts an array of 2 translations into a SIMD translation.
///
/// This is an advanced conversion for SIMD (Single Instruction, Multiple Data) operations,
/// which allow performing the same operation on multiple translations simultaneously for
/// better performance. This is typically used in high-performance computing scenarios.
///
/// # Examples
///
/// ```
/// # use nalgebra::Translation2;
/// # #[cfg(feature = "simd")]
/// # {
/// use simba::simd::f32x2;
///
/// let t1 = Translation2::new(1.0f32, 2.0);
/// let t2 = Translation2::new(3.0f32, 4.0);
///
/// let simd_t: Translation2<f32x2> = [t1, t2].into();
/// # }
/// ```
impl<T: Scalar + PrimitiveSimdValue, const D: usize> From<[Translation<T::Element, D>; 2]>
    for Translation<T, D>
where
    T: From<[<T as simba::simd::SimdValue>::Element; 2]>,
    T::Element: Scalar,
{
    #[inline]
    fn from(arr: [Translation<T::Element, D>; 2]) -> Self {
        Self::from(OVector::from([
            arr[0].vector.clone(),
            arr[1].vector.clone(),
        ]))
    }
}

/// Converts an array of 4 translations into a SIMD translation.
///
/// This is an advanced conversion for SIMD operations that processes 4 translations
/// simultaneously. This can provide significant performance improvements when applying
/// the same transformation to multiple objects.
///
/// # Examples
///
/// ```
/// # use nalgebra::Translation2;
/// # #[cfg(feature = "simd")]
/// # {
/// use simba::simd::f32x4;
///
/// let translations = [
///     Translation2::new(1.0f32, 2.0),
///     Translation2::new(3.0f32, 4.0),
///     Translation2::new(5.0f32, 6.0),
///     Translation2::new(7.0f32, 8.0),
/// ];
///
/// let simd_t: Translation2<f32x4> = translations.into();
/// # }
/// ```
impl<T: Scalar + PrimitiveSimdValue, const D: usize> From<[Translation<T::Element, D>; 4]>
    for Translation<T, D>
where
    T: From<[<T as simba::simd::SimdValue>::Element; 4]>,
    T::Element: Scalar,
{
    #[inline]
    fn from(arr: [Translation<T::Element, D>; 4]) -> Self {
        Self::from(OVector::from([
            arr[0].vector.clone(),
            arr[1].vector.clone(),
            arr[2].vector.clone(),
            arr[3].vector.clone(),
        ]))
    }
}

/// Converts an array of 8 translations into a SIMD translation.
///
/// This is an advanced conversion for wide SIMD operations that processes 8 translations
/// simultaneously. This provides maximum throughput for batch transformation operations
/// on modern CPUs with AVX-512 or similar instruction sets.
///
/// # Examples
///
/// ```
/// # use nalgebra::Translation2;
/// # #[cfg(feature = "simd")]
/// # {
/// use simba::simd::f32x8;
///
/// let translations = [
///     Translation2::new(1.0f32, 2.0),
///     Translation2::new(3.0f32, 4.0),
///     Translation2::new(5.0f32, 6.0),
///     Translation2::new(7.0f32, 8.0),
///     Translation2::new(9.0f32, 10.0),
///     Translation2::new(11.0f32, 12.0),
///     Translation2::new(13.0f32, 14.0),
///     Translation2::new(15.0f32, 16.0),
/// ];
///
/// let simd_t: Translation2<f32x8> = translations.into();
/// # }
/// ```
impl<T: Scalar + PrimitiveSimdValue, const D: usize> From<[Translation<T::Element, D>; 8]>
    for Translation<T, D>
where
    T: From<[<T as simba::simd::SimdValue>::Element; 8]>,
    T::Element: Scalar,
{
    #[inline]
    fn from(arr: [Translation<T::Element, D>; 8]) -> Self {
        Self::from(OVector::from([
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

/// Converts an array of 16 translations into a SIMD translation.
///
/// This is an advanced conversion for very wide SIMD operations that processes 16 translations
/// simultaneously. This is useful for maximum throughput on specialized hardware or when
/// processing large batches of transformations in parallel.
///
/// # Examples
///
/// ```
/// # use nalgebra::Translation2;
/// # #[cfg(feature = "simd")]
/// # {
/// use simba::simd::f32x16;
///
/// // Create 16 translations for batch processing
/// let mut translations = [Translation2::new(0.0f32, 0.0); 16];
/// for (i, t) in translations.iter_mut().enumerate() {
///     *t = Translation2::new(i as f32, (i * 2) as f32);
/// }
///
/// let simd_t: Translation2<f32x16> = translations.into();
/// # }
/// ```
impl<T: Scalar + PrimitiveSimdValue, const D: usize> From<[Translation<T::Element, D>; 16]>
    for Translation<T, D>
where
    T: From<[<T as simba::simd::SimdValue>::Element; 16]>,
    T::Element: Scalar,
{
    #[inline]
    fn from(arr: [Translation<T::Element, D>; 16]) -> Self {
        Self::from(OVector::from([
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
