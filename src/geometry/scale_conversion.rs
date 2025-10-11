use num::{One, Zero};

use simba::scalar::{RealField, SubsetOf, SupersetOf};
use simba::simd::PrimitiveSimdValue;

use crate::base::allocator::Allocator;
use crate::base::dimension::{DimNameAdd, DimNameSum, U1};
use crate::base::{Const, DefaultAllocator, OMatrix, OVector, SVector, Scalar};

use crate::Point;
use crate::geometry::{Scale, SuperTCategoryOf, TAffine, Transform};

/*
 * This file provides the following conversions:
 * =============================================
 *
 * Scale -> Scale
 * Scale -> Transform
 * Scale -> Matrix (homogeneous)
 */

impl<T1, T2, const D: usize> SubsetOf<Scale<T2, D>> for Scale<T1, D>
where
    T1: Scalar,
    T2: Scalar + SupersetOf<T1>,
{
    #[inline]
    fn to_superset(&self) -> Scale<T2, D> {
        Scale::from(self.vector.to_superset())
    }

    #[inline]
    fn is_in_subset(rot: &Scale<T2, D>) -> bool {
        crate::is_convertible::<_, SVector<T1, D>>(&rot.vector)
    }

    #[inline]
    fn from_superset_unchecked(rot: &Scale<T2, D>) -> Self {
        Scale {
            vector: rot.vector.to_subset_unchecked(),
        }
    }
}

impl<T1, T2, C, const D: usize> SubsetOf<Transform<T2, C, D>> for Scale<T1, D>
where
    T1: RealField,
    T2: RealField + SupersetOf<T1>,
    C: SuperTCategoryOf<TAffine>,
    Const<D>: DimNameAdd<U1>,
    DefaultAllocator: Allocator<DimNameSum<Const<D>, U1>, DimNameSum<Const<D>, U1>>
        + Allocator<DimNameSum<Const<D>, U1>, U1>,
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
    SubsetOf<OMatrix<T2, DimNameSum<Const<D>, U1>, DimNameSum<Const<D>, U1>>> for Scale<T1, D>
where
    T1: RealField,
    T2: RealField + SupersetOf<T1>,
    Const<D>: DimNameAdd<U1>,
    DefaultAllocator: Allocator<DimNameSum<Const<D>, U1>, DimNameSum<Const<D>, U1>>
        + Allocator<DimNameSum<Const<D>, U1>, U1>,
{
    #[inline]
    fn to_superset(&self) -> OMatrix<T2, DimNameSum<Const<D>, U1>, DimNameSum<Const<D>, U1>> {
        self.to_homogeneous().to_superset()
    }

    #[inline]
    fn is_in_subset(m: &OMatrix<T2, DimNameSum<Const<D>, U1>, DimNameSum<Const<D>, U1>>) -> bool {
        if m[(D, D)] != T2::one() {
            return false;
        }
        for i in 0..D + 1 {
            for j in 0..D + 1 {
                if i != j && m[(i, j)] != T2::zero() {
                    return false;
                }
            }
        }
        true
    }

    #[inline]
    fn from_superset_unchecked(
        m: &OMatrix<T2, DimNameSum<Const<D>, U1>, DimNameSum<Const<D>, U1>>,
    ) -> Self {
        let v = m.fixed_view::<D, D>(0, 0).diagonal();
        Self {
            vector: crate::convert_unchecked(v),
        }
    }
}

/// Converts a scale transformation into a homogeneous transformation matrix.
///
/// This is the same as calling [`to_homogeneous()`](Scale::to_homogeneous), but allows
/// using the convenient `.into()` syntax.
///
/// # Examples
///
/// ```
/// # use nalgebra::{Scale3, Matrix4};
/// let scale = Scale3::new(2.0, 3.0, 4.0);
///
/// // Using .into()
/// let matrix: Matrix4<f64> = scale.into();
///
/// // Same as
/// let matrix2 = scale.to_homogeneous();
/// assert_eq!(matrix, matrix2);
/// ```
///
/// # See Also
///
/// - [`to_homogeneous`](Scale::to_homogeneous) - Explicit conversion method with more documentation
impl<T: Scalar + Zero + One, const D: usize> From<Scale<T, D>>
    for OMatrix<T, DimNameSum<Const<D>, U1>, DimNameSum<Const<D>, U1>>
where
    Const<D>: DimNameAdd<U1>,
    DefaultAllocator: Allocator<DimNameSum<Const<D>, U1>, DimNameSum<Const<D>, U1>>
        + Allocator<DimNameSum<Const<D>, U1>, U1>
        + Allocator<Const<D>>,
{
    #[inline]
    fn from(t: Scale<T, D>) -> Self {
        t.to_homogeneous()
    }
}

/// Creates a scale transformation from a vector.
///
/// The vector's components become the scale factors for each axis.
///
/// # Examples
///
/// ```
/// # use nalgebra::{Scale3, Vector3};
/// let vec = Vector3::new(2.0, 3.0, 4.0);
/// let scale: Scale3<f64> = vec.into();
///
/// assert_eq!(scale, Scale3::new(2.0, 3.0, 4.0));
/// ```
///
/// # See Also
///
/// - [`new`](Scale::new) - Direct constructor
impl<T: Scalar, const D: usize> From<OVector<T, Const<D>>> for Scale<T, D> {
    #[inline]
    fn from(vector: OVector<T, Const<D>>) -> Self {
        Scale { vector }
    }
}

/// Creates a scale transformation from an array of scale factors.
///
/// This is a convenient way to create a scale when you have an array of values.
/// The array elements become the scale factors for each axis in order.
///
/// # Examples
///
/// ## 2D scale from array
/// ```
/// # use nalgebra::Scale2;
/// let factors = [2.0, 3.0];
/// let scale: Scale2<f64> = factors.into();
///
/// assert_eq!(scale, Scale2::new(2.0, 3.0));
/// ```
///
/// ## 3D scale from array
/// ```
/// # use nalgebra::Scale3;
/// let scale = Scale3::from([1.5, 2.5, 3.5]);
/// assert_eq!(scale, Scale3::new(1.5, 2.5, 3.5));
/// ```
///
/// ## Use case: loading from configuration
/// ```
/// # use nalgebra::{Scale2, Point2};
/// // Imagine loading scale factors from a config file
/// fn get_sprite_scale_from_config() -> [f32; 2] {
///     [1.5, 2.0]  // width and height multipliers
/// }
///
/// let scale: Scale2<f32> = get_sprite_scale_from_config().into();
/// let sprite_size = Point2::new(32.0, 32.0);
/// let scaled_size = scale.transform_point(&sprite_size);
///
/// assert_eq!(scaled_size, Point2::new(48.0, 64.0));
/// ```
///
/// # See Also
///
/// - [`new`](Scale::new) - Direct constructor with individual parameters
impl<T: Scalar, const D: usize> From<[T; D]> for Scale<T, D> {
    #[inline]
    fn from(coords: [T; D]) -> Self {
        Scale {
            vector: coords.into(),
        }
    }
}

/// Creates a scale transformation from a point's coordinates.
///
/// The point's coordinates become the scale factors for each axis. This can be useful
/// when you want to use point coordinates directly as scale factors.
///
/// # Examples
///
/// ```
/// # use nalgebra::{Scale3, Point3};
/// let point = Point3::new(2.0, 3.0, 4.0);
/// let scale: Scale3<f64> = point.into();
///
/// assert_eq!(scale, Scale3::new(2.0, 3.0, 4.0));
/// ```
///
/// ## Use case: proportional scaling
/// ```
/// # use nalgebra::{Scale2, Point2};
/// // Use one point's coordinates to scale another
/// let reference_size = Point2::new(2.0, 1.5);  // 2x width, 1.5x height
/// let scale = Scale2::from(reference_size);
///
/// let original = Point2::new(10.0, 20.0);
/// let scaled = scale.transform_point(&original);
///
/// assert_eq!(scaled, Point2::new(20.0, 30.0));
/// ```
///
/// # See Also
///
/// - [`new`](Scale::new) - Direct constructor
impl<T: Scalar, const D: usize> From<Point<T, D>> for Scale<T, D> {
    #[inline]
    fn from(pt: Point<T, D>) -> Self {
        Scale { vector: pt.coords }
    }
}

/// Converts a scale transformation into an array of scale factors.
///
/// This extracts the scale factors as an array, which can be useful for serialization,
/// storage, or interfacing with APIs that expect arrays.
///
/// # Examples
///
/// ```
/// # use nalgebra::Scale3;
/// let scale = Scale3::new(2.0, 3.0, 4.0);
/// let factors: [f64; 3] = scale.into();
///
/// assert_eq!(factors, [2.0, 3.0, 4.0]);
/// ```
///
/// ## Use case: saving to configuration
/// ```
/// # use nalgebra::Scale2;
/// let scale = Scale2::new(1.5, 2.0);
///
/// // Convert to array for saving
/// let config_values: [f32; 2] = scale.into();
///
/// // Later, reconstruct the scale
/// let loaded_scale = Scale2::from(config_values);
/// assert_eq!(loaded_scale, scale);
/// ```
///
/// # See Also
///
/// - [`From<[T; D]>`](From) - Create a scale from an array
impl<T: Scalar, const D: usize> From<Scale<T, D>> for [T; D] {
    #[inline]
    fn from(t: Scale<T, D>) -> Self {
        t.vector.into()
    }
}

/// Converts an array of 2 scales into a SIMD scale (for vectorized operations).
///
/// This is used internally for SIMD (Single Instruction, Multiple Data) optimizations,
/// allowing multiple scale operations to be performed in parallel.
///
/// # Advanced Usage
///
/// This is typically used with SIMD types like `f32x2` from the `simba` crate for
/// performance-critical code. Most users won't need to use this directly.
///
/// # See Also
///
/// - [SIMD documentation](https://docs.rs/simba) for more on vectorized operations
impl<T: Scalar + PrimitiveSimdValue, const D: usize> From<[Scale<T::Element, D>; 2]> for Scale<T, D>
where
    T: From<[<T as simba::simd::SimdValue>::Element; 2]>,
    T::Element: Scalar,
{
    #[inline]
    fn from(arr: [Scale<T::Element, D>; 2]) -> Self {
        Self::from(OVector::from([
            arr[0].vector.clone(),
            arr[1].vector.clone(),
        ]))
    }
}

/// Converts an array of 4 scales into a SIMD scale (for vectorized operations).
///
/// This is used for SIMD optimizations with 4-wide vector types like `f32x4`.
/// Most users won't need this directly; it's used internally for performance.
///
/// # See Also
///
/// - [SIMD documentation](https://docs.rs/simba) for more on vectorized operations
impl<T: Scalar + PrimitiveSimdValue, const D: usize> From<[Scale<T::Element, D>; 4]> for Scale<T, D>
where
    T: From<[<T as simba::simd::SimdValue>::Element; 4]>,
    T::Element: Scalar,
{
    #[inline]
    fn from(arr: [Scale<T::Element, D>; 4]) -> Self {
        Self::from(OVector::from([
            arr[0].vector.clone(),
            arr[1].vector.clone(),
            arr[2].vector.clone(),
            arr[3].vector.clone(),
        ]))
    }
}

/// Converts an array of 8 scales into a SIMD scale (for vectorized operations).
///
/// This is used for SIMD optimizations with 8-wide vector types like `f32x8`.
/// Most users won't need this directly; it's used internally for performance.
///
/// # See Also
///
/// - [SIMD documentation](https://docs.rs/simba) for more on vectorized operations
impl<T: Scalar + PrimitiveSimdValue, const D: usize> From<[Scale<T::Element, D>; 8]> for Scale<T, D>
where
    T: From<[<T as simba::simd::SimdValue>::Element; 8]>,
    T::Element: Scalar,
{
    #[inline]
    fn from(arr: [Scale<T::Element, D>; 8]) -> Self {
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

/// Converts an array of 16 scales into a SIMD scale (for vectorized operations).
///
/// This is used for SIMD optimizations with 16-wide vector types like `f32x16`.
/// Most users won't need this directly; it's used internally for performance.
///
/// # See Also
///
/// - [SIMD documentation](https://docs.rs/simba) for more on vectorized operations
impl<T: Scalar + PrimitiveSimdValue, const D: usize> From<[Scale<T::Element, D>; 16]>
    for Scale<T, D>
where
    T: From<[<T as simba::simd::SimdValue>::Element; 16]>,
    T::Element: Scalar,
{
    #[inline]
    fn from(arr: [Scale<T::Element, D>; 16]) -> Self {
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
