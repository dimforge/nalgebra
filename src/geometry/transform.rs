use approx::{AbsDiffEq, RelativeEq, UlpsEq};
use std::any::Any;
use std::fmt::{self, Debug};
use std::hash;
use std::marker::PhantomData;

#[cfg(feature = "serde-serialize-no-std")]
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use simba::scalar::RealField;

use crate::base::allocator::Allocator;
use crate::base::dimension::{DimNameAdd, DimNameSum, U1};
use crate::base::storage::Owned;
use crate::base::{Const, DefaultAllocator, DimName, OMatrix, SVector};

use crate::geometry::Point;

/// Trait implemented by phantom types identifying the projective transformation type.
///
/// NOTE: this trait is not intended to be implemented outside of the `nalgebra` crate.
pub trait TCategory: Any + Debug + Copy + PartialEq + Send {
    /// Indicates whether a `Transform` with the category `Self` has a bottom-row different from
    /// `0 0 .. 1`.
    #[inline]
    fn has_normalizer() -> bool {
        true
    }

    /// Checks that the given matrix is a valid homogeneous representation of an element of the
    /// category `Self`.
    fn check_homogeneous_invariants<T: RealField, D: DimName>(mat: &OMatrix<T, D, D>) -> bool
    where
        T::Epsilon: Clone,
        DefaultAllocator: Allocator<D, D>;
}

/// Traits that gives the `Transform` category that is compatible with the result of the
/// multiplication of transformations with categories `Self` and `Other`.
pub trait TCategoryMul<Other: TCategory>: TCategory {
    /// The transform category that results from the multiplication of a `Transform<Self>` to a
    /// `Transform<Other>`. This is usually equal to `Self` or `Other`, whichever is the most
    /// general category.
    type Representative: TCategory;
}

/// Indicates that `Self` is a more general `Transform` category than `Other`.
pub trait SuperTCategoryOf<Other: TCategory>: TCategory {}

/// Indicates that `Self` is a more specific `Transform` category than `Other`.
///
/// Automatically implemented based on `SuperTCategoryOf`.
pub trait SubTCategoryOf<Other: TCategory>: TCategory {}
impl<T1, T2> SubTCategoryOf<T2> for T1
where
    T1: TCategory,
    T2: SuperTCategoryOf<T1>,
{
}

/// Tag representing the most general (not necessarily inversible) `Transform` type.
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum TGeneral {}

/// Tag representing the most general inversible `Transform` type.
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum TProjective {}

/// Tag representing an affine `Transform`. Its bottom-row is equal to `(0, 0 ... 0, 1)`.
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum TAffine {}

impl TCategory for TGeneral {
    #[inline]
    fn check_homogeneous_invariants<T: RealField, D: DimName>(_: &OMatrix<T, D, D>) -> bool
    where
        T::Epsilon: Clone,
        DefaultAllocator: Allocator<D, D>,
    {
        true
    }
}

impl TCategory for TProjective {
    #[inline]
    fn check_homogeneous_invariants<T: RealField, D: DimName>(mat: &OMatrix<T, D, D>) -> bool
    where
        T::Epsilon: Clone,
        DefaultAllocator: Allocator<D, D>,
    {
        mat.is_invertible()
    }
}

impl TCategory for TAffine {
    #[inline]
    fn has_normalizer() -> bool {
        false
    }

    #[inline]
    fn check_homogeneous_invariants<T: RealField, D: DimName>(mat: &OMatrix<T, D, D>) -> bool
    where
        T::Epsilon: Clone,
        DefaultAllocator: Allocator<D, D>,
    {
        let last = D::DIM - 1;
        mat.is_invertible()
            && mat[(last, last)] == T::one()
            && (0..last).all(|i| mat[(last, i)].is_zero())
    }
}

macro_rules! category_mul_impl(
    ($($a: ident * $b: ident => $c: ty);* $(;)*) => {$(
        impl TCategoryMul<$a> for $b {
            type Representative = $c;
        }
    )*}
);

// We require stability upon multiplication.
impl<T: TCategory> TCategoryMul<T> for T {
    type Representative = T;
}

category_mul_impl!(
//  TGeneral * TGeneral    => TGeneral;
    TGeneral * TProjective => TGeneral;
    TGeneral * TAffine     => TGeneral;

    TProjective * TGeneral    => TGeneral;
//  TProjective * TProjective => TProjective;
    TProjective * TAffine     => TProjective;

    TAffine * TGeneral    => TGeneral;
    TAffine * TProjective => TProjective;
//  TAffine * TAffine     => TAffine;
);

macro_rules! super_tcategory_impl(
    ($($a: ident >= $b: ident);* $(;)*) => {$(
        impl SuperTCategoryOf<$b> for $a { }
    )*}
);

impl<T: TCategory> SuperTCategoryOf<T> for T {}

super_tcategory_impl!(
    TGeneral    >= TProjective;
    TGeneral    >= TAffine;
    TProjective >= TAffine;
);

/// A general transformation in homogeneous coordinates.
///
/// A `Transform` represents the most general type of transformation in N-dimensional space,
/// stored as a homogeneous matrix with dimensions `(D + 1, D + 1)`. For example, a 3D transform
/// uses a 4×4 matrix, and a 2D transform uses a 3×3 matrix.
///
/// # What is a General Transformation?
///
/// Unlike more restrictive transformation types like [`Isometry`] (rotation + translation) or
/// [`Similarity`] (rotation + translation + uniform scaling), a `Transform` can represent
/// **any** linear transformation, including:
///
/// - **Affine transformations**: Translation, rotation, scaling, shearing, and their combinations
/// - **Projective transformations**: Perspective projections used in 3D graphics
/// - **Arbitrary deformations**: Non-uniform scaling, skewing, or custom matrix transformations
///
/// # When to Use Transform
///
/// Use `Transform` when you need:
/// - **Perspective projections** for 3D cameras
/// - **Non-uniform scaling** (different scale factors on each axis)
/// - **Shearing or skewing** operations
/// - **Custom matrix transformations** that don't fit other categories
/// - **Composing different transformation types** without losing generality
///
/// For simpler use cases, prefer:
/// - [`Isometry`] for rigid-body motion (rotation + translation, preserves distances)
/// - [`Similarity`] for uniform scaling with rotation and translation (preserves angles)
///
/// # Transform Categories
///
/// The `Transform` type is parameterized by a category `C` that encodes mathematical properties:
///
/// - [`TGeneral`]: The most general transformation (no guarantees about invertibility)
/// - [`TProjective`]: Invertible transformations (guaranteed to have an inverse)
/// - [`TAffine`]: Affine transformations (bottom row is `[0, 0, ..., 0, 1]`)
///
/// Categories form a hierarchy: `TAffine ⊂ TProjective ⊂ TGeneral`. You can convert to a
/// more general category using [`set_category`](Self::set_category), but not to a more specific one.
///
/// # Type Aliases
///
/// Common type aliases are provided for convenience:
/// - [`Transform2`](crate::Transform2), [`Transform3`](crate::Transform3) - General 2D/3D transforms
/// - [`Projective2`](crate::Projective2), [`Projective3`](crate::Projective3) - Invertible 2D/3D transforms
/// - [`Affine2`](crate::Affine2), [`Affine3`](crate::Affine3) - Affine 2D/3D transforms
///
/// # Example: Perspective Projection
///
/// ```
/// # use nalgebra::{Projective3, Point3, Matrix4};
/// # use approx::assert_relative_eq;
/// // Create a simple perspective projection matrix
/// let fov = 1.0f32; // Field of view
/// let aspect = 16.0 / 9.0;
/// let near = 0.1;
/// let far = 100.0;
///
/// let f = 1.0 / (fov / 2.0).tan();
/// let matrix = Matrix4::new(
///     f / aspect, 0.0, 0.0, 0.0,
///     0.0, f, 0.0, 0.0,
///     0.0, 0.0, (far + near) / (near - far), (2.0 * far * near) / (near - far),
///     0.0, 0.0, -1.0, 0.0,
/// );
///
/// let projection = Projective3::from_matrix_unchecked(matrix);
///
/// // Project a 3D point
/// let point = Point3::new(1.0, 1.0, -5.0);
/// let projected = projection.transform_point(&point);
/// ```
///
/// # Example: Non-Uniform Scaling
///
/// ```
/// # use nalgebra::{Transform2, Point2, Matrix3};
/// # use approx::assert_relative_eq;
/// // Scale x by 2.0 and y by 3.0
/// let matrix = Matrix3::new(
///     2.0, 0.0, 0.0,
///     0.0, 3.0, 0.0,
///     0.0, 0.0, 1.0,
/// );
/// let transform = Transform2::from_matrix_unchecked(matrix);
///
/// let point = Point2::new(1.0, 1.0);
/// let scaled = transform.transform_point(&point);
/// assert_relative_eq!(scaled, Point2::new(2.0, 3.0));
/// ```
///
/// # Example: Shearing Transformation
///
/// ```
/// # use nalgebra::{Affine2, Point2, Matrix3};
/// # use approx::assert_relative_eq;
/// // Shear in the x direction based on y
/// let matrix = Matrix3::new(
///     1.0, 0.5, 0.0,  // x' = x + 0.5 * y
///     0.0, 1.0, 0.0,  // y' = y
///     0.0, 0.0, 1.0,  // Affine bottom row
/// );
/// let shear = Affine2::from_matrix_unchecked(matrix);
///
/// let point = Point2::new(1.0, 2.0);
/// let sheared = shear.transform_point(&point);
/// assert_relative_eq!(sheared, Point2::new(2.0, 2.0)); // x becomes 1.0 + 0.5*2.0 = 2.0
/// ```
///
/// # See Also
///
/// * [`Isometry`] - Rigid-body transformations (rotation + translation)
/// * [`Similarity`] - Uniform scaling + rotation + translation
/// * [`from_matrix_unchecked`](Self::from_matrix_unchecked) - Create from a homogeneous matrix
#[repr(C)]
pub struct Transform<T: RealField, C: TCategory, const D: usize>
where
    Const<D>: DimNameAdd<U1>,
    DefaultAllocator: Allocator<DimNameSum<Const<D>, U1>, DimNameSum<Const<D>, U1>>,
{
    matrix: OMatrix<T, DimNameSum<Const<D>, U1>, DimNameSum<Const<D>, U1>>,
    _phantom: PhantomData<C>,
}

impl<T: RealField + Debug, C: TCategory, const D: usize> Debug for Transform<T, C, D>
where
    Const<D>: DimNameAdd<U1>,
    DefaultAllocator: Allocator<DimNameSum<Const<D>, U1>, DimNameSum<Const<D>, U1>>,
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        self.matrix.fmt(formatter)
    }
}

impl<T: RealField + hash::Hash, C: TCategory, const D: usize> hash::Hash for Transform<T, C, D>
where
    Const<D>: DimNameAdd<U1>,
    DefaultAllocator: Allocator<DimNameSum<Const<D>, U1>, DimNameSum<Const<D>, U1>>,
    Owned<T, DimNameSum<Const<D>, U1>, DimNameSum<Const<D>, U1>>: hash::Hash,
{
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        self.matrix.hash(state);
    }
}

impl<T: RealField + Copy, C: TCategory, const D: usize> Copy for Transform<T, C, D>
where
    Const<D>: DimNameAdd<U1>,
    DefaultAllocator: Allocator<DimNameSum<Const<D>, U1>, DimNameSum<Const<D>, U1>>,
    Owned<T, DimNameSum<Const<D>, U1>, DimNameSum<Const<D>, U1>>: Copy,
{
}

impl<T: RealField, C: TCategory, const D: usize> Clone for Transform<T, C, D>
where
    Const<D>: DimNameAdd<U1>,
    DefaultAllocator: Allocator<DimNameSum<Const<D>, U1>, DimNameSum<Const<D>, U1>>,
{
    #[inline]
    fn clone(&self) -> Self {
        Transform::from_matrix_unchecked(self.matrix.clone())
    }
}

#[cfg(feature = "bytemuck")]
unsafe impl<T, C: TCategory, const D: usize> bytemuck::Zeroable for Transform<T, C, D>
where
    T: RealField + bytemuck::Zeroable,
    Const<D>: DimNameAdd<U1>,
    DefaultAllocator: Allocator<DimNameSum<Const<D>, U1>, DimNameSum<Const<D>, U1>>,
    OMatrix<T, DimNameSum<Const<D>, U1>, DimNameSum<Const<D>, U1>>: bytemuck::Zeroable,
{
}

#[cfg(feature = "bytemuck")]
unsafe impl<T, C: TCategory, const D: usize> bytemuck::Pod for Transform<T, C, D>
where
    T: RealField + bytemuck::Pod,
    Const<D>: DimNameAdd<U1>,
    DefaultAllocator: Allocator<DimNameSum<Const<D>, U1>, DimNameSum<Const<D>, U1>>,
    OMatrix<T, DimNameSum<Const<D>, U1>, DimNameSum<Const<D>, U1>>: bytemuck::Pod,
    Owned<T, DimNameSum<Const<D>, U1>, DimNameSum<Const<D>, U1>>: Copy,
{
}

#[cfg(feature = "serde-serialize-no-std")]
impl<T: RealField, C: TCategory, const D: usize> Serialize for Transform<T, C, D>
where
    Const<D>: DimNameAdd<U1>,
    DefaultAllocator: Allocator<DimNameSum<Const<D>, U1>, DimNameSum<Const<D>, U1>>,
    Owned<T, DimNameSum<Const<D>, U1>, DimNameSum<Const<D>, U1>>: Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.matrix.serialize(serializer)
    }
}

#[cfg(feature = "serde-serialize-no-std")]
impl<'a, T: RealField, C: TCategory, const D: usize> Deserialize<'a> for Transform<T, C, D>
where
    Const<D>: DimNameAdd<U1>,
    DefaultAllocator: Allocator<DimNameSum<Const<D>, U1>, DimNameSum<Const<D>, U1>>,
    Owned<T, DimNameSum<Const<D>, U1>, DimNameSum<Const<D>, U1>>: Deserialize<'a>,
{
    fn deserialize<Des>(deserializer: Des) -> Result<Self, Des::Error>
    where
        Des: Deserializer<'a>,
    {
        let matrix = OMatrix::<T, DimNameSum<Const<D>, U1>, DimNameSum<Const<D>, U1>>::deserialize(
            deserializer,
        )?;

        Ok(Transform::from_matrix_unchecked(matrix))
    }
}

impl<T: RealField + Eq, C: TCategory, const D: usize> Eq for Transform<T, C, D>
where
    Const<D>: DimNameAdd<U1>,
    DefaultAllocator: Allocator<DimNameSum<Const<D>, U1>, DimNameSum<Const<D>, U1>>,
{
}

impl<T: RealField, C: TCategory, const D: usize> PartialEq for Transform<T, C, D>
where
    Const<D>: DimNameAdd<U1>,
    DefaultAllocator: Allocator<DimNameSum<Const<D>, U1>, DimNameSum<Const<D>, U1>>,
{
    #[inline]
    fn eq(&self, right: &Self) -> bool {
        self.matrix == right.matrix
    }
}

impl<T: RealField, C: TCategory, const D: usize> Transform<T, C, D>
where
    Const<D>: DimNameAdd<U1>,
    DefaultAllocator: Allocator<DimNameSum<Const<D>, U1>, DimNameSum<Const<D>, U1>>,
{
    /// Creates a new transformation from the given homogeneous matrix without checking invariants.
    ///
    /// This constructor does not verify that the matrix satisfies the invariants required by
    /// the transformation category. For example, when creating a [`TProjective`] transform,
    /// it won't check that the matrix is invertible. When creating a [`TAffine`] transform,
    /// it won't check that the bottom row is `[0, 0, ..., 0, 1]`.
    ///
    /// # Safety
    ///
    /// While this function is not marked `unsafe`, incorrect usage can lead to undefined behavior
    /// when methods assume category invariants hold. Only use this when you're certain the matrix
    /// satisfies the category requirements, or when working with [`TGeneral`] which has no invariants.
    ///
    /// # Parameters
    ///
    /// * `matrix` - A homogeneous transformation matrix of size `(D+1) × (D+1)`
    ///
    /// # Example: Creating a 2D Transform
    ///
    /// ```
    /// # use nalgebra::{Transform2, Matrix3};
    /// // A simple 2D translation by (5, 10)
    /// let matrix = Matrix3::new(
    ///     1.0, 0.0, 5.0,
    ///     0.0, 1.0, 10.0,
    ///     0.0, 0.0, 1.0,
    /// );
    /// let transform = Transform2::from_matrix_unchecked(matrix);
    /// ```
    ///
    /// # Example: Creating a 3D Projective Transform
    ///
    /// ```
    /// # use nalgebra::{Projective3, Matrix4};
    /// // A 3D perspective projection matrix (simplified)
    /// let matrix = Matrix4::new(
    ///     1.0, 0.0, 0.0, 0.0,
    ///     0.0, 1.0, 0.0, 0.0,
    ///     0.0, 0.0, 1.0, 0.0,
    ///     0.0, 0.0, -0.1, 1.0,
    /// );
    /// let projection = Projective3::from_matrix_unchecked(matrix);
    /// ```
    ///
    /// # Example: Non-uniform Scaling
    ///
    /// ```
    /// # use nalgebra::{Affine3, Matrix4};
    /// // Scale x by 2, y by 3, z by 0.5
    /// let matrix = Matrix4::new(
    ///     2.0, 0.0, 0.0, 0.0,
    ///     0.0, 3.0, 0.0, 0.0,
    ///     0.0, 0.0, 0.5, 0.0,
    ///     0.0, 0.0, 0.0, 1.0,
    /// );
    /// let scale = Affine3::from_matrix_unchecked(matrix);
    /// ```
    ///
    /// # See Also
    ///
    /// * [`into_inner`](Self::into_inner) - Extract the underlying matrix
    /// * [`matrix`](Self::matrix) - Get a reference to the matrix
    /// * [`to_homogeneous`](Self::to_homogeneous) - Convert to homogeneous matrix
    #[inline]
    pub const fn from_matrix_unchecked(
        matrix: OMatrix<T, DimNameSum<Const<D>, U1>, DimNameSum<Const<D>, U1>>,
    ) -> Self {
        Transform {
            matrix,
            _phantom: PhantomData,
        }
    }

    /// Consumes the transform and returns the underlying homogeneous matrix.
    ///
    /// This method takes ownership of the transform and returns the raw matrix representation.
    /// Use this when you need to work with the matrix directly or interface with APIs that
    /// expect matrix types.
    ///
    /// # Returns
    ///
    /// The underlying `(D+1) × (D+1)` homogeneous transformation matrix
    ///
    /// # Example: 2D Transform
    ///
    /// ```
    /// # use nalgebra::{Matrix3, Transform2};
    /// let m = Matrix3::new(
    ///     1.0, 2.0, 0.0,
    ///     3.0, 4.0, 0.0,
    ///     0.0, 0.0, 1.0
    /// );
    /// let t = Transform2::from_matrix_unchecked(m);
    /// assert_eq!(t.into_inner(), m);
    /// ```
    ///
    /// # Example: 3D Affine Transform
    ///
    /// ```
    /// # use nalgebra::{Affine3, Matrix4};
    /// # use approx::assert_relative_eq;
    /// let matrix = Matrix4::new(
    ///     2.0, 0.0, 0.0, 5.0,
    ///     0.0, 2.0, 0.0, 10.0,
    ///     0.0, 0.0, 2.0, 15.0,
    ///     0.0, 0.0, 0.0, 1.0,
    /// );
    /// let transform = Affine3::from_matrix_unchecked(matrix);
    /// let extracted = transform.into_inner();
    /// assert_eq!(extracted, matrix);
    /// ```
    ///
    /// # Example: Using with Graphics APIs
    ///
    /// ```
    /// # use nalgebra::{Projective3, Matrix4};
    /// let projection = Projective3::from_matrix_unchecked(Matrix4::identity());
    ///
    /// // Extract matrix for use with graphics APIs (OpenGL, WebGPU, etc.)
    /// let matrix: Matrix4<f32> = projection.into_inner();
    /// // Now you can pass `matrix.as_slice()` to the graphics API
    /// ```
    ///
    /// # See Also
    ///
    /// * [`matrix`](Self::matrix) - Get a reference without consuming
    /// * [`to_homogeneous`](Self::to_homogeneous) - Clone the matrix
    /// * [`from_matrix_unchecked`](Self::from_matrix_unchecked) - Create from matrix
    #[inline]
    pub fn into_inner(self) -> OMatrix<T, DimNameSum<Const<D>, U1>, DimNameSum<Const<D>, U1>> {
        self.matrix
    }

    /// Retrieves the underlying matrix.
    ///
    /// **Deprecated:** Use [`into_inner`](Self::into_inner) instead. This method has been renamed
    /// for clarity and consistency with Rust naming conventions. The functionality is identical.
    ///
    /// # Returns
    ///
    /// The underlying `(D+1) × (D+1)` homogeneous transformation matrix
    ///
    /// # Example
    ///
    /// ```
    /// # use nalgebra::{Transform2, Matrix3};
    /// let matrix = Matrix3::<f32>::identity();
    /// let transform = Transform2::from_matrix_unchecked(matrix);
    ///
    /// // Don't use this - it's deprecated
    /// // let m = transform.unwrap();
    ///
    /// // Use this instead:
    /// let m = transform.into_inner();
    /// ```
    ///
    /// # See Also
    ///
    /// * [`into_inner`](Self::into_inner) - The preferred replacement for this method
    /// * [`matrix`](Self::matrix) - Get a reference without consuming
    #[deprecated(note = "use `.into_inner()` instead")]
    #[inline]
    pub fn unwrap(self) -> OMatrix<T, DimNameSum<Const<D>, U1>, DimNameSum<Const<D>, U1>> {
        self.matrix
    }

    /// Returns a reference to the underlying homogeneous matrix.
    ///
    /// This provides immutable access to the transformation matrix without consuming the
    /// transform. Use this when you need to read the matrix values, pass them to other
    /// functions, or perform matrix operations.
    ///
    /// # Returns
    ///
    /// An immutable reference to the `(D+1) × (D+1)` homogeneous transformation matrix
    ///
    /// # Example: Reading Matrix Values
    ///
    /// ```
    /// # use nalgebra::{Matrix3, Transform2};
    /// let m = Matrix3::new(
    ///     1.0, 2.0, 0.0,
    ///     3.0, 4.0, 0.0,
    ///     0.0, 0.0, 1.0
    /// );
    /// let t = Transform2::from_matrix_unchecked(m);
    /// assert_eq!(*t.matrix(), m);
    /// ```
    ///
    /// # Example: Inspecting Transform Components
    ///
    /// ```
    /// # use nalgebra::{Affine3, Matrix4};
    /// let matrix = Matrix4::new(
    ///     2.0, 0.0, 0.0, 5.0,
    ///     0.0, 3.0, 0.0, 10.0,
    ///     0.0, 0.0, 1.0, 15.0,
    ///     0.0, 0.0, 0.0, 1.0,
    /// );
    /// let transform = Affine3::from_matrix_unchecked(matrix);
    ///
    /// // Access the translation component (last column, first 3 rows)
    /// let tx = transform.matrix()[(0, 3)];
    /// let ty = transform.matrix()[(1, 3)];
    /// let tz = transform.matrix()[(2, 3)];
    /// assert_eq!((tx, ty, tz), (5.0, 10.0, 15.0));
    /// ```
    ///
    /// # Example: Matrix Composition
    ///
    /// ```
    /// # use nalgebra::{Transform2, Matrix3};
    /// # use approx::assert_relative_eq;
    /// let t1 = Transform2::from_matrix_unchecked(Matrix3::<f32>::identity());
    /// let t2 = Transform2::from_matrix_unchecked(Matrix3::<f32>::identity());
    ///
    /// // Compose by multiplying matrices
    /// let composed_matrix = t1.matrix() * t2.matrix();
    /// let composed = Transform2::from_matrix_unchecked(composed_matrix);
    /// ```
    ///
    /// # See Also
    ///
    /// * [`into_inner`](Self::into_inner) - Consume and get the matrix
    /// * [`matrix_mut_unchecked`](Self::matrix_mut_unchecked) - Get mutable reference
    /// * [`to_homogeneous`](Self::to_homogeneous) - Clone the matrix
    #[inline]
    #[must_use]
    pub const fn matrix(&self) -> &OMatrix<T, DimNameSum<Const<D>, U1>, DimNameSum<Const<D>, U1>> {
        &self.matrix
    }

    /// Returns a mutable reference to the underlying matrix without checking invariants.
    ///
    /// This method provides mutable access to the transformation matrix. It's marked
    /// `_unchecked` because directly modifying the matrix can violate the invariants
    /// required by the transformation category. For example, modifying an [`TAffine`]
    /// transform's bottom row would break its affine property.
    ///
    /// # Safety Considerations
    ///
    /// While not marked `unsafe`, using this method incorrectly can lead to inconsistent
    /// state. Only modify the matrix if you understand the category invariants:
    ///
    /// - [`TGeneral`]: No restrictions - modify freely
    /// - [`TProjective`]: Matrix must remain invertible
    /// - [`TAffine`]: Bottom row must stay `[0, 0, ..., 0, 1]`
    ///
    /// # Returns
    ///
    /// A mutable reference to the `(D+1) × (D+1)` homogeneous transformation matrix
    ///
    /// # Example: Modifying Translation (Safe for All Categories)
    ///
    /// ```
    /// # use nalgebra::{Matrix3, Transform2};
    /// let m = Matrix3::new(
    ///     1.0, 2.0, 0.0,
    ///     3.0, 4.0, 0.0,
    ///     0.0, 0.0, 1.0
    /// );
    /// let mut t = Transform2::from_matrix_unchecked(m);
    ///
    /// // Modify the translation components (last column)
    /// t.matrix_mut_unchecked().m13 = 42.0;
    /// t.matrix_mut_unchecked().m23 = 90.0;
    ///
    /// let expected = Matrix3::new(
    ///     1.0, 2.0, 42.0,
    ///     3.0, 4.0, 90.0,
    ///     0.0, 0.0, 1.0
    /// );
    /// assert_eq!(*t.matrix(), expected);
    /// ```
    ///
    /// # Example: Scaling a Transform (General Category)
    ///
    /// ```
    /// # use nalgebra::{Transform2, Matrix3};
    /// let mut t = Transform2::from_matrix_unchecked(Matrix3::identity());
    ///
    /// // Scale all transformation components by 2
    /// *t.matrix_mut_unchecked() *= 2.0;
    /// ```
    ///
    /// # Example: Custom Matrix Modification
    ///
    /// ```
    /// # use nalgebra::{Transform3, Matrix4};
    /// let mut transform = Transform3::from_matrix_unchecked(Matrix4::identity());
    ///
    /// // Apply custom matrix modifications
    /// let matrix = transform.matrix_mut_unchecked();
    /// matrix[(0, 0)] = 2.0;  // Scale X
    /// matrix[(1, 1)] = 2.0;  // Scale Y
    /// matrix[(2, 2)] = 2.0;  // Scale Z
    /// ```
    ///
    /// # See Also
    ///
    /// * [`matrix`](Self::matrix) - Get immutable reference
    /// * [`matrix_mut`](Self::matrix_mut) - Available only for [`TGeneral`] category
    /// * [`from_matrix_unchecked`](Self::from_matrix_unchecked) - Create new transform
    #[inline]
    pub const fn matrix_mut_unchecked(
        &mut self,
    ) -> &mut OMatrix<T, DimNameSum<Const<D>, U1>, DimNameSum<Const<D>, U1>> {
        &mut self.matrix
    }

    /// Converts this transform to a more general category.
    ///
    /// Transform categories form a hierarchy: `TAffine ⊂ TProjective ⊂ TGeneral`. This method
    /// allows you to "widen" the category to a more general one. The transformation matrix
    /// itself is not modified, only its type-level guarantees are relaxed.
    ///
    /// You **cannot** narrow to a more specific category using this method. For example, you
    /// cannot convert `TProjective` to `TAffine` because not all projective transforms are affine.
    ///
    /// # Type Parameters
    ///
    /// * `CNew` - The new category, which must be a superset of the current category
    ///
    /// # Returns
    ///
    /// The same transformation with a more general category type
    ///
    /// # Example: Affine to Projective
    ///
    /// ```
    /// # use nalgebra::{Affine2, Projective2, Matrix3};
    /// let affine_matrix = Matrix3::new(
    ///     2.0, 0.0, 5.0,
    ///     0.0, 2.0, 10.0,
    ///     0.0, 0.0, 1.0,  // Affine: bottom row is [0, 0, 1]
    /// );
    /// let affine = Affine2::from_matrix_unchecked(affine_matrix);
    ///
    /// // Convert to more general projective category
    /// let projective: Projective2<f32> = affine.set_category();
    /// ```
    ///
    /// # Example: Projective to General
    ///
    /// ```
    /// # use nalgebra::{Projective3, Transform3, Matrix4};
    /// let projective = Projective3::from_matrix_unchecked(Matrix4::identity());
    ///
    /// // Widen to the most general category
    /// let general: Transform3<f32> = projective.set_category();
    /// ```
    ///
    /// # Example: Composing Different Transform Types
    ///
    /// ```
    /// # use nalgebra::{Affine2, Projective2, Matrix3};
    /// # use approx::assert_relative_eq;
    /// // Sometimes you need to widen categories to compose transforms
    /// let affine = Affine2::from_matrix_unchecked(Matrix3::identity());
    /// let projective = Projective2::from_matrix_unchecked(Matrix3::identity());
    ///
    /// // To multiply them, convert affine to projective
    /// let affine_as_proj: Projective2<f32> = affine.set_category();
    /// let composed = projective * affine_as_proj;
    /// ```
    ///
    /// # See Also
    ///
    /// * [`from_matrix_unchecked`](Self::from_matrix_unchecked) - Create with specific category
    /// * [`TGeneral`] - Most general category
    /// * [`TProjective`] - Invertible transformations
    /// * [`TAffine`] - Affine transformations
    #[inline]
    pub fn set_category<CNew: SuperTCategoryOf<C>>(self) -> Transform<T, CNew, D> {
        Transform::from_matrix_unchecked(self.matrix)
    }

    /// Clones this transform into one that owns its data.
    ///
    /// **Deprecated:** This method is redundant with automatic `Copy` and the `.clone()` method
    /// and will be removed in a future release. For types that implement `Copy`, the transform
    /// is automatically copied. For types that don't, use `.clone()` instead.
    ///
    /// # Returns
    ///
    /// A cloned copy of this transformation
    ///
    /// # Example
    ///
    /// ```
    /// # use nalgebra::{Transform2, Matrix3};
    /// let transform = Transform2::from_matrix_unchecked(Matrix3::<f32>::identity());
    ///
    /// // Don't use this - it's deprecated
    /// // let cloned = transform.clone_owned();
    ///
    /// // For Copy types, just assign directly:
    /// let copied = transform;
    ///
    /// // For non-Copy types, use .clone():
    /// let cloned = transform.clone();
    /// ```
    ///
    /// # See Also
    ///
    /// * `Clone` trait - Use `.clone()` instead
    /// * `Copy` trait - For automatic copying on assignment
    #[inline]
    #[deprecated(
        note = "This method is redundant with automatic `Copy` and the `.clone()` method and will be removed in a future release."
    )]
    pub fn clone_owned(&self) -> Transform<T, C, D> {
        Transform::from_matrix_unchecked(self.matrix.clone_owned())
    }

    /// Clones and returns the underlying homogeneous transformation matrix.
    ///
    /// This creates a copy of the internal matrix representation. Use this when you need to
    /// pass the matrix to functions that take ownership, or when you want to keep both the
    /// transform and its matrix representation.
    ///
    /// # Returns
    ///
    /// A cloned copy of the `(D+1) × (D+1)` homogeneous transformation matrix
    ///
    /// # Example: 2D Transform to Matrix
    ///
    /// ```
    /// # use nalgebra::{Matrix3, Transform2};
    /// let m = Matrix3::new(
    ///     1.0, 2.0, 0.0,
    ///     3.0, 4.0, 0.0,
    ///     0.0, 0.0, 1.0
    /// );
    /// let t = Transform2::from_matrix_unchecked(m);
    /// assert_eq!(t.to_homogeneous(), m);
    /// ```
    ///
    /// # Example: Exporting for Graphics APIs
    ///
    /// ```
    /// # use nalgebra::{Affine3, Matrix4};
    /// let transform = Affine3::from_matrix_unchecked(Matrix4::identity());
    ///
    /// // Get matrix for OpenGL/WebGPU shader uniform
    /// let matrix = transform.to_homogeneous();
    /// let data: &[f32] = matrix.as_slice();
    /// // Send `data` to GPU...
    /// ```
    ///
    /// # Example: Matrix Arithmetic
    ///
    /// ```
    /// # use nalgebra::{Projective2, Matrix3};
    /// # use approx::assert_relative_eq;
    /// let t1 = Projective2::from_matrix_unchecked(Matrix3::<f32>::identity());
    /// let t2 = Projective2::from_matrix_unchecked(Matrix3::<f32>::identity());
    ///
    /// // Extract matrices for custom operations
    /// let m1 = t1.to_homogeneous();
    /// let m2 = t2.to_homogeneous();
    /// let product = m1 * m2;
    /// ```
    ///
    /// # See Also
    ///
    /// * [`into_inner`](Self::into_inner) - Consume and get the matrix (no clone)
    /// * [`matrix`](Self::matrix) - Get a reference to the matrix
    /// * [`from_matrix_unchecked`](Self::from_matrix_unchecked) - Create from matrix
    #[inline]
    #[must_use]
    pub fn to_homogeneous(&self) -> OMatrix<T, DimNameSum<Const<D>, U1>, DimNameSum<Const<D>, U1>> {
        self.matrix().clone_owned()
    }

    /// Attempts to compute the inverse of this transformation.
    ///
    /// This method tries to invert the transformation matrix. It returns `None` if the matrix
    /// is not invertible (singular). For transformations in the [`TProjective`] or [`TAffine`]
    /// categories (which guarantee invertibility), use [`inverse`](Self::inverse) instead.
    ///
    /// The inverse transformation "undoes" the original: applying a transform and then its
    /// inverse returns you to the original state.
    ///
    /// # Returns
    ///
    /// * `Some(inverse)` if the matrix is invertible
    /// * `None` if the matrix is singular (determinant is zero)
    ///
    /// # Example: Invertible Transform
    ///
    /// ```
    /// # use nalgebra::{Matrix3, Transform2};
    /// # use approx::assert_relative_eq;
    /// // Create an invertible 2D transform
    /// let m = Matrix3::new(
    ///     2.0, 2.0, -0.3,
    ///     3.0, 4.0, 0.1,
    ///     0.0, 0.0, 1.0
    /// );
    /// let t = Transform2::from_matrix_unchecked(m);
    /// let inv_t = t.try_inverse().unwrap();
    ///
    /// // Verify: t * t^-1 = identity
    /// assert_relative_eq!(t * inv_t, Transform2::identity());
    /// assert_relative_eq!(inv_t * t, Transform2::identity());
    /// ```
    ///
    /// # Example: Non-Invertible Transform
    ///
    /// ```
    /// # use nalgebra::{Matrix3, Transform2};
    /// // This matrix is singular (zero determinant)
    /// let m = Matrix3::new(
    ///     0.0, 2.0, 1.0,
    ///     3.0, 0.0, 5.0,
    ///     0.0, 0.0, 0.0  // Bottom row all zeros
    /// );
    /// let t = Transform2::from_matrix_unchecked(m);
    /// assert!(t.try_inverse().is_none());
    /// ```
    ///
    /// # Example: Undoing a Transformation
    ///
    /// ```
    /// # use nalgebra::{Transform3, Point3, Matrix4};
    /// # use approx::assert_relative_eq;
    /// // Create a non-uniform scale
    /// let scale = Matrix4::new(
    ///     2.0, 0.0, 0.0, 5.0,
    ///     0.0, 3.0, 0.0, 10.0,
    ///     0.0, 0.0, 0.5, 15.0,
    ///     0.0, 0.0, 0.0, 1.0,
    /// );
    /// let transform = Transform3::from_matrix_unchecked(scale);
    ///
    /// let point = Point3::new(1.0, 2.0, 3.0);
    /// let transformed = transform.transform_point(&point);
    ///
    /// // Get back original point
    /// if let Some(inv) = transform.try_inverse() {
    ///     let back = inv.transform_point(&transformed);
    ///     assert_relative_eq!(back, point, epsilon = 1.0e-6);
    /// }
    /// ```
    ///
    /// # See Also
    ///
    /// * [`inverse`](Self::inverse) - Unwrapping version for [`TProjective`] transforms
    /// * [`try_inverse_mut`](Self::try_inverse_mut) - In-place version
    /// * [`inverse_transform_point`](Self::inverse_transform_point) - Transform by inverse directly
    #[inline]
    #[must_use = "Did you mean to use try_inverse_mut()?"]
    pub fn try_inverse(self) -> Option<Transform<T, C, D>> {
        self.matrix
            .try_inverse()
            .map(Transform::from_matrix_unchecked)
    }

    /// Computes the inverse of this transformation (guaranteed to succeed for projective transforms).
    ///
    /// This method is available only for transformations in the [`TProjective`] or [`TAffine`]
    /// categories, which are guaranteed to be invertible. Unlike [`try_inverse`](Self::try_inverse),
    /// this method unwraps the result and will panic if inversion somehow fails (which should
    /// never happen for valid projective/affine transforms).
    ///
    /// For general transforms ([`TGeneral`]) that might not be invertible, use
    /// [`try_inverse`](Self::try_inverse) instead.
    ///
    /// # Returns
    ///
    /// The inverse transformation
    ///
    /// # Panics
    ///
    /// Should never panic for valid projective/affine transforms, but will panic if the
    /// matrix is unexpectedly singular.
    ///
    /// # Example: 2D Projective Transform
    ///
    /// ```
    /// # use nalgebra::{Matrix3, Projective2};
    /// # use approx::assert_relative_eq;
    /// let m = Matrix3::new(
    ///     2.0, 2.0, -0.3,
    ///     3.0, 4.0, 0.1,
    ///     0.0, 0.0, 1.0
    /// );
    /// let proj = Projective2::from_matrix_unchecked(m);
    /// let inv = proj.inverse();
    ///
    /// // Verify: proj * proj^-1 = identity
    /// assert_relative_eq!(proj * inv, Projective2::identity());
    /// assert_relative_eq!(inv * proj, Projective2::identity());
    /// ```
    ///
    /// # Example: 3D Affine Transform
    ///
    /// ```
    /// # use nalgebra::{Affine3, Matrix4, Point3};
    /// # use approx::assert_relative_eq;
    /// // Translation + rotation + non-uniform scale
    /// let matrix = Matrix4::new(
    ///     2.0, 0.0, 0.0, 10.0,
    ///     0.0, 3.0, 0.0, 20.0,
    ///     0.0, 0.0, 0.5, 30.0,
    ///     0.0, 0.0, 0.0, 1.0,
    /// );
    /// let transform = Affine3::from_matrix_unchecked(matrix);
    /// let inverse = transform.inverse();
    ///
    /// // Transform and inverse cancel out
    /// let point = Point3::new(5.0, 10.0, 15.0);
    /// let result = inverse.transform_point(&transform.transform_point(&point));
    /// assert_relative_eq!(result, point, epsilon = 1.0e-6);
    /// ```
    ///
    /// # Example: Camera View Matrix Inversion
    ///
    /// ```
    /// # use nalgebra::{Projective3, Matrix4, Point3};
    /// // World-to-camera transformation
    /// let view_matrix = Projective3::from_matrix_unchecked(Matrix4::<f32>::identity());
    ///
    /// // Camera-to-world is the inverse
    /// let inv_view = view_matrix.inverse();
    /// ```
    ///
    /// # See Also
    ///
    /// * [`try_inverse`](Self::try_inverse) - Fallible version for general transforms
    /// * [`inverse_mut`](Self::inverse_mut) - In-place inversion
    /// * [`inverse_transform_point`](Self::inverse_transform_point) - Apply inverse to a point
    #[inline]
    #[must_use = "Did you mean to use inverse_mut()?"]
    pub fn inverse(self) -> Transform<T, C, D>
    where
        C: SubTCategoryOf<TProjective>,
    {
        // TODO: specialize for TAffine?
        Transform::from_matrix_unchecked(self.matrix.try_inverse().unwrap())
    }

    /// Attempts to invert this transformation in-place.
    ///
    /// This modifies the transform to become its own inverse. Returns `true` if the inversion
    /// succeeded, or `false` if the matrix is not invertible (singular). This is more efficient
    /// than [`try_inverse`](Self::try_inverse) as it doesn't allocate a new transform.
    ///
    /// For [`TProjective`] and [`TAffine`] transforms (which are guaranteed invertible),
    /// use [`inverse_mut`](Self::inverse_mut) instead.
    ///
    /// # Returns
    ///
    /// * `true` if the transform was successfully inverted
    /// * `false` if the matrix is singular (leaves the transform unchanged)
    ///
    /// # Example: In-Place Inversion
    ///
    /// ```
    /// # use nalgebra::{Matrix3, Transform2};
    /// # use approx::assert_relative_eq;
    /// let m = Matrix3::new(
    ///     2.0, 2.0, -0.3,
    ///     3.0, 4.0, 0.1,
    ///     0.0, 0.0, 1.0
    /// );
    /// let t = Transform2::from_matrix_unchecked(m);
    /// let mut inv_t = t;
    /// assert!(inv_t.try_inverse_mut());
    /// assert_relative_eq!(t * inv_t, Transform2::identity());
    /// assert_relative_eq!(inv_t * t, Transform2::identity());
    /// ```
    ///
    /// # Example: Non-Invertible Case
    ///
    /// ```
    /// # use nalgebra::{Matrix3, Transform2};
    /// // Singular matrix
    /// let m = Matrix3::new(
    ///     0.0, 2.0, 1.0,
    ///     3.0, 0.0, 5.0,
    ///     0.0, 0.0, 0.0
    /// );
    /// let mut t = Transform2::from_matrix_unchecked(m);
    /// assert!(!t.try_inverse_mut()); // Inversion fails
    /// ```
    ///
    /// # See Also
    ///
    /// * [`try_inverse`](Self::try_inverse) - Non-mutating version
    /// * [`inverse_mut`](Self::inverse_mut) - For projective transforms (guaranteed to work)
    #[inline]
    pub fn try_inverse_mut(&mut self) -> bool {
        self.matrix.try_inverse_mut()
    }

    /// Inverts this transformation in-place (guaranteed to succeed for projective transforms).
    ///
    /// This modifies the transform to become its own inverse. This method is only available
    /// for [`TProjective`] and [`TAffine`] categories, which are guaranteed to be invertible.
    /// It's more efficient than [`inverse`](Self::inverse) as it doesn't allocate a new transform.
    ///
    /// For general transforms that might not be invertible, use [`try_inverse_mut`](Self::try_inverse_mut).
    ///
    /// # Example: 2D Projective Transform
    ///
    /// ```
    /// # use nalgebra::{Matrix3, Projective2};
    /// # use approx::assert_relative_eq;
    /// let m = Matrix3::new(
    ///     2.0, 2.0, -0.3,
    ///     3.0, 4.0, 0.1,
    ///     0.0, 0.0, 1.0
    /// );
    /// let proj = Projective2::from_matrix_unchecked(m);
    /// let mut inv_t = proj;
    /// inv_t.inverse_mut();
    /// assert_relative_eq!(proj * inv_t, Projective2::identity());
    /// assert_relative_eq!(inv_t * proj, Projective2::identity());
    /// ```
    ///
    /// # Example: Efficient Inversion Loop
    ///
    /// ```
    /// # use nalgebra::{Affine3, Matrix4, Point3};
    /// # use approx::assert_relative_eq;
    /// // Process multiple transforms efficiently
    /// let mut transforms: Vec<Affine3<f32>> = vec![
    ///     Affine3::from_matrix_unchecked(Matrix4::identity()),
    ///     // ... more transforms
    /// ];
    ///
    /// // Invert all in-place without allocating
    /// for t in &mut transforms {
    ///     t.inverse_mut();
    /// }
    /// ```
    ///
    /// # See Also
    ///
    /// * [`inverse`](Self::inverse) - Non-mutating version
    /// * [`try_inverse_mut`](Self::try_inverse_mut) - Fallible version for general transforms
    #[inline]
    pub fn inverse_mut(&mut self)
    where
        C: SubTCategoryOf<TProjective>,
    {
        let _ = self.matrix.try_inverse_mut();
    }
}

impl<T, C, const D: usize> Transform<T, C, D>
where
    T: RealField,
    C: TCategory,
    Const<D>: DimNameAdd<U1>,
    DefaultAllocator: Allocator<DimNameSum<Const<D>, U1>, DimNameSum<Const<D>, U1>>
        + Allocator<DimNameSum<Const<D>, U1>>, // + Allocator<D, D>
                                               // + Allocator<D>
{
    /// Transforms a point by applying this transformation.
    ///
    /// This applies the full transformation to a point, including translation, rotation,
    /// scaling, shearing, perspective, and any other operations encoded in the matrix.
    /// This method is equivalent to using the `*` operator: `self * pt`.
    ///
    /// # Parameters
    ///
    /// * `pt` - The point to transform
    ///
    /// # Returns
    ///
    /// The transformed point
    ///
    /// # Example: 2D Point Transformation
    ///
    /// ```
    /// # use nalgebra::{Transform2, Point2, Matrix3};
    /// # use approx::assert_relative_eq;
    /// // Non-uniform scale: x by 2, y by 3
    /// let matrix = Matrix3::new(
    ///     2.0, 0.0, 5.0,
    ///     0.0, 3.0, 10.0,
    ///     0.0, 0.0, 1.0,
    /// );
    /// let transform = Transform2::from_matrix_unchecked(matrix);
    ///
    /// let point = Point2::new(1.0, 2.0);
    /// let result = transform.transform_point(&point);
    /// // x: 1*2 + 5 = 7, y: 2*3 + 10 = 16
    /// assert_relative_eq!(result, Point2::new(7.0, 16.0));
    /// ```
    ///
    /// # Example: 3D Shearing
    ///
    /// ```
    /// # use nalgebra::{Affine3, Point3, Matrix4};
    /// # use approx::assert_relative_eq;
    /// // Shear x based on y
    /// let matrix = Matrix4::new(
    ///     1.0, 0.5, 0.0, 0.0,
    ///     0.0, 1.0, 0.0, 0.0,
    ///     0.0, 0.0, 1.0, 0.0,
    ///     0.0, 0.0, 0.0, 1.0,
    /// );
    /// let shear = Affine3::from_matrix_unchecked(matrix);
    ///
    /// let point = Point3::new(1.0, 2.0, 3.0);
    /// let sheared = shear.transform_point(&point);
    /// // x becomes 1 + 0.5*2 = 2.0
    /// assert_relative_eq!(sheared, Point3::new(2.0, 2.0, 3.0));
    /// ```
    ///
    /// # See Also
    ///
    /// * [`transform_vector`](Self::transform_vector) - Transform a vector (ignores translation)
    /// * [`inverse_transform_point`](Self::inverse_transform_point) - Apply inverse transformation
    #[inline]
    #[must_use]
    pub fn transform_point(&self, pt: &Point<T, D>) -> Point<T, D> {
        self * pt
    }

    /// Transforms a vector by this transformation, ignoring translation.
    ///
    /// Vectors represent directions or displacements, not positions, so they are not
    /// affected by translation. This method applies all transformation components except
    /// translation (e.g., rotation, scaling, shearing, etc.).
    ///
    /// This method is equivalent to using the `*` operator: `self * v`.
    ///
    /// # Parameters
    ///
    /// * `v` - The vector to transform
    ///
    /// # Returns
    ///
    /// The transformed vector (without translation)
    ///
    /// # Example: 2D Vector with Non-Uniform Scale
    ///
    /// ```
    /// # use nalgebra::{Transform2, Vector2, Matrix3};
    /// # use approx::assert_relative_eq;
    /// // Scale x by 2, y by 3, translate by (100, 200)
    /// let matrix = Matrix3::new(
    ///     2.0, 0.0, 100.0,  // Translation ignored for vectors
    ///     0.0, 3.0, 200.0,
    ///     0.0, 0.0, 1.0,
    /// );
    /// let transform = Transform2::from_matrix_unchecked(matrix);
    ///
    /// let vector = Vector2::new(1.0, 2.0);
    /// let scaled = transform.transform_vector(&vector);
    /// // Only scaling applied: x*2=2, y*3=6
    /// assert_relative_eq!(scaled, Vector2::new(2.0, 6.0));
    /// ```
    ///
    /// # Example: 3D Direction Vector
    ///
    /// ```
    /// # use nalgebra::{Affine3, Vector3, Matrix4};
    /// # use approx::assert_relative_eq;
    /// // Rotation around Y axis by 90 degrees
    /// use std::f32::consts::PI;
    /// let cos = (PI / 2.0).cos();
    /// let sin = (PI / 2.0).sin();
    /// let matrix = Matrix4::new(
    ///     cos, 0.0, sin, 10.0,  // Translation component
    ///     0.0, 1.0, 0.0, 20.0,
    ///     -sin, 0.0, cos, 30.0,
    ///     0.0, 0.0, 0.0, 1.0,
    /// );
    /// let transform = Affine3::from_matrix_unchecked(matrix);
    ///
    /// let direction = Vector3::new(1.0, 0.0, 0.0);
    /// let rotated = transform.transform_vector(&direction);
    /// // Only rotation applied, translation ignored
    /// assert_relative_eq!(rotated, Vector3::new(0.0, 0.0, -1.0), epsilon = 1.0e-6);
    /// ```
    ///
    /// # See Also
    ///
    /// * [`transform_point`](Self::transform_point) - Transform a point (includes translation)
    /// * [`inverse_transform_vector`](Self::inverse_transform_vector) - Apply inverse transformation
    #[inline]
    #[must_use]
    pub fn transform_vector(&self, v: &SVector<T, D>) -> SVector<T, D> {
        self * v
    }
}

impl<T: RealField, C: TCategory, const D: usize> Transform<T, C, D>
where
    Const<D>: DimNameAdd<U1>,
    C: SubTCategoryOf<TProjective>,
    DefaultAllocator: Allocator<DimNameSum<Const<D>, U1>, DimNameSum<Const<D>, U1>>
        + Allocator<DimNameSum<Const<D>, U1>>, // + Allocator<D, D>
                                               // + Allocator<D>
{
    /// Transforms a point by the inverse of this transformation.
    ///
    /// This applies the inverse transformation to a point without explicitly computing
    /// the full inverse transform. This is available only for [`TProjective`] and [`TAffine`]
    /// categories (which are guaranteed invertible).
    ///
    /// This is equivalent to `self.inverse() * pt` but may be more efficient in some cases.
    ///
    /// # Parameters
    ///
    /// * `pt` - The point to transform
    ///
    /// # Returns
    ///
    /// The point transformed by the inverse of this transformation
    ///
    /// # Example: Undoing a Transformation
    ///
    /// ```
    /// # use nalgebra::{Projective2, Point2, Matrix3};
    /// # use approx::assert_relative_eq;
    /// // Non-uniform scale and translation
    /// let matrix = Matrix3::new(
    ///     2.0, 0.0, 10.0,
    ///     0.0, 3.0, 20.0,
    ///     0.0, 0.0, 1.0,
    /// );
    /// let transform = Projective2::from_matrix_unchecked(matrix);
    ///
    /// let original = Point2::new(1.0, 2.0);
    /// let transformed = transform.transform_point(&original);
    ///
    /// // Get back to original using inverse transform
    /// let back = transform.inverse_transform_point(&transformed);
    /// assert_relative_eq!(back, original, epsilon = 1.0e-6);
    /// ```
    ///
    /// # Example: World to Local Space Conversion
    ///
    /// ```
    /// # use nalgebra::{Affine3, Point3, Matrix4};
    /// # use approx::assert_relative_eq;
    /// // Object's world transform
    /// let world_transform = Affine3::from_matrix_unchecked(Matrix4::new(
    ///     2.0, 0.0, 0.0, 10.0,
    ///     0.0, 2.0, 0.0, 5.0,
    ///     0.0, 0.0, 2.0, 0.0,
    ///     0.0, 0.0, 0.0, 1.0,
    /// ));
    ///
    /// // Point in world space
    /// let world_point = Point3::new(12.0, 9.0, 4.0);
    ///
    /// // Convert to object's local space
    /// let local_point = world_transform.inverse_transform_point(&world_point);
    /// // local_point is now (1, 2, 2) in object space
    /// assert_relative_eq!(local_point, Point3::new(1.0, 2.0, 2.0), epsilon = 1.0e-6);
    /// ```
    ///
    /// # See Also
    ///
    /// * [`transform_point`](Self::transform_point) - Apply forward transformation
    /// * [`inverse`](Self::inverse) - Compute full inverse transformation
    /// * [`inverse_transform_vector`](Self::inverse_transform_vector) - Inverse transform a vector
    #[inline]
    #[must_use]
    pub fn inverse_transform_point(&self, pt: &Point<T, D>) -> Point<T, D> {
        self.clone().inverse() * pt
    }

    /// Transforms a vector by the inverse of this transformation, ignoring translation.
    ///
    /// This applies the inverse transformation to a vector without explicitly computing
    /// the full inverse transform. Since vectors represent directions, translation is ignored.
    /// This is available only for [`TProjective`] and [`TAffine`] categories.
    ///
    /// This is equivalent to `self.inverse() * v` but may be more efficient in some cases.
    ///
    /// # Parameters
    ///
    /// * `v` - The vector to transform
    ///
    /// # Returns
    ///
    /// The vector transformed by the inverse (without translation)
    ///
    /// # Example: Reversing a Vector Transformation
    ///
    /// ```
    /// # use nalgebra::{Projective2, Vector2, Matrix3};
    /// # use approx::assert_relative_eq;
    /// // Scale and rotation
    /// use std::f32::consts::PI;
    /// let cos = (PI / 4.0).cos();
    /// let sin = (PI / 4.0).sin();
    /// let matrix = Matrix3::new(
    ///     2.0 * cos, -2.0 * sin, 10.0,  // Translation component
    ///     2.0 * sin,  2.0 * cos, 20.0,
    ///     0.0,        0.0,       1.0,
    /// );
    /// let transform = Projective2::from_matrix_unchecked(matrix);
    ///
    /// let original = Vector2::new(1.0, 0.0);
    /// let transformed = transform.transform_vector(&original);
    ///
    /// // Reverse the transformation
    /// let back = transform.inverse_transform_vector(&transformed);
    /// assert_relative_eq!(back, original, epsilon = 1.0e-6);
    /// ```
    ///
    /// # Example: World to Local Direction
    ///
    /// ```
    /// # use nalgebra::{Affine3, Vector3, Matrix4};
    /// # use approx::assert_relative_eq;
    /// // Object transform: scaled 2x
    /// let object_transform = Affine3::from_matrix_unchecked(Matrix4::new(
    ///     2.0, 0.0, 0.0, 10.0,
    ///     0.0, 2.0, 0.0, 5.0,
    ///     0.0, 0.0, 2.0, 0.0,
    ///     0.0, 0.0, 0.0, 1.0,
    /// ));
    ///
    /// // Direction in world space
    /// let world_direction = Vector3::new(4.0, 0.0, 0.0);
    ///
    /// // Convert to object's local space (will be scaled down)
    /// let local_direction = object_transform.inverse_transform_vector(&world_direction);
    /// assert_relative_eq!(local_direction, Vector3::new(2.0, 0.0, 0.0), epsilon = 1.0e-6);
    /// ```
    ///
    /// # See Also
    ///
    /// * [`transform_vector`](Self::transform_vector) - Apply forward transformation
    /// * [`inverse`](Self::inverse) - Compute full inverse transformation
    /// * [`inverse_transform_point`](Self::inverse_transform_point) - Inverse transform a point
    #[inline]
    #[must_use]
    pub fn inverse_transform_vector(&self, v: &SVector<T, D>) -> SVector<T, D> {
        self.clone().inverse() * v
    }
}

impl<T: RealField, const D: usize> Transform<T, TGeneral, D>
where
    Const<D>: DimNameAdd<U1>,
    DefaultAllocator: Allocator<DimNameSum<Const<D>, U1>, DimNameSum<Const<D>, U1>>,
{
    /// Returns a mutable reference to the underlying matrix (only for [`TGeneral`] category).
    ///
    /// This method is only available for transforms with the [`TGeneral`] category, which has
    /// no invariants to maintain. For other categories ([`TProjective`], [`TAffine`]), use
    /// [`matrix_mut_unchecked`](Self::matrix_mut_unchecked) if you know what you're doing.
    ///
    /// # Returns
    ///
    /// A mutable reference to the `(D+1) × (D+1)` homogeneous transformation matrix
    ///
    /// # Example: Modifying a General Transform
    ///
    /// ```
    /// # use nalgebra::{Transform2, Matrix3, Point2};
    /// # use approx::assert_relative_eq;
    /// let mut transform = Transform2::from_matrix_unchecked(Matrix3::identity());
    ///
    /// // Freely modify the matrix for general transforms
    /// transform.matrix_mut()[(0, 0)] = 2.0;  // Scale X
    /// transform.matrix_mut()[(1, 1)] = 3.0;  // Scale Y
    /// transform.matrix_mut()[(0, 2)] = 5.0;  // Translate X
    /// transform.matrix_mut()[(1, 2)] = 10.0; // Translate Y
    ///
    /// let point = Point2::new(1.0, 1.0);
    /// let result = transform.transform_point(&point);
    /// assert_relative_eq!(result, Point2::new(7.0, 13.0)); // (1*2+5, 1*3+10)
    /// ```
    ///
    /// # Example: Building a Transform Incrementally
    ///
    /// ```
    /// # use nalgebra::{Transform3, Matrix4};
    /// let mut transform = Transform3::from_matrix_unchecked(Matrix4::identity());
    ///
    /// // Build transformation step by step
    /// let matrix = transform.matrix_mut();
    /// matrix[(0, 0)] = 2.0;  // Scale X by 2
    /// matrix[(1, 1)] = 2.0;  // Scale Y by 2
    /// matrix[(2, 2)] = 2.0;  // Scale Z by 2
    /// matrix[(0, 3)] = 10.0; // Translate X by 10
    /// matrix[(1, 3)] = 20.0; // Translate Y by 20
    /// matrix[(2, 3)] = 30.0; // Translate Z by 30
    /// ```
    ///
    /// # See Also
    ///
    /// * [`matrix_mut_unchecked`](Self::matrix_mut_unchecked) - Unsafe version for other categories
    /// * [`matrix`](Self::matrix) - Get immutable reference
    /// * [`from_matrix_unchecked`](Self::from_matrix_unchecked) - Create from matrix
    #[inline]
    pub const fn matrix_mut(
        &mut self,
    ) -> &mut OMatrix<T, DimNameSum<Const<D>, U1>, DimNameSum<Const<D>, U1>> {
        self.matrix_mut_unchecked()
    }
}

impl<T: RealField, C: TCategory, const D: usize> AbsDiffEq for Transform<T, C, D>
where
    Const<D>: DimNameAdd<U1>,
    T::Epsilon: Clone,
    DefaultAllocator: Allocator<DimNameSum<Const<D>, U1>, DimNameSum<Const<D>, U1>>,
{
    type Epsilon = T::Epsilon;

    #[inline]
    fn default_epsilon() -> Self::Epsilon {
        T::default_epsilon()
    }

    #[inline]
    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        self.matrix.abs_diff_eq(&other.matrix, epsilon)
    }
}

impl<T: RealField, C: TCategory, const D: usize> RelativeEq for Transform<T, C, D>
where
    Const<D>: DimNameAdd<U1>,
    T::Epsilon: Clone,
    DefaultAllocator: Allocator<DimNameSum<Const<D>, U1>, DimNameSum<Const<D>, U1>>,
{
    #[inline]
    fn default_max_relative() -> Self::Epsilon {
        T::default_max_relative()
    }

    #[inline]
    fn relative_eq(
        &self,
        other: &Self,
        epsilon: Self::Epsilon,
        max_relative: Self::Epsilon,
    ) -> bool {
        self.matrix
            .relative_eq(&other.matrix, epsilon, max_relative)
    }
}

impl<T: RealField, C: TCategory, const D: usize> UlpsEq for Transform<T, C, D>
where
    Const<D>: DimNameAdd<U1>,
    T::Epsilon: Clone,
    DefaultAllocator: Allocator<DimNameSum<Const<D>, U1>, DimNameSum<Const<D>, U1>>,
{
    #[inline]
    fn default_max_ulps() -> u32 {
        T::default_max_ulps()
    }

    #[inline]
    fn ulps_eq(&self, other: &Self, epsilon: Self::Epsilon, max_ulps: u32) -> bool {
        self.matrix.ulps_eq(&other.matrix, epsilon, max_ulps)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::base::Matrix4;

    #[test]
    fn checks_homogeneous_invariants_of_square_identity_matrix() {
        assert!(TAffine::check_homogeneous_invariants(
            &Matrix4::<f32>::identity()
        ));
    }
}
