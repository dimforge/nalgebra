// Needed otherwise the rkyv macros generate code incompatible with rust-2024
#![cfg_attr(feature = "rkyv-serialize", allow(unsafe_op_in_unsafe_fn))]

use approx::{AbsDiffEq, RelativeEq, UlpsEq};
use num::{One, Zero};
use std::fmt;
use std::hash;

#[cfg(feature = "serde-serialize-no-std")]
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use simba::scalar::{ClosedAddAssign, ClosedNeg, ClosedSubAssign};

use crate::base::allocator::Allocator;
use crate::base::dimension::{DimNameAdd, DimNameSum, U1};
use crate::base::storage::Owned;
use crate::base::{Const, DefaultAllocator, OMatrix, SVector, Scalar};

use crate::geometry::Point;

#[cfg(feature = "rkyv-serialize")]
use rkyv::bytecheck;

/// A translation in a D-dimensional space.
///
/// A translation is a geometric transformation that moves every point by a fixed distance
/// in a specific direction. Unlike rotation or scaling, translations preserve the shape
/// and orientation of objects - they simply shift their position.
///
/// # Type Parameters
///
/// * `T` - The scalar type (usually `f32` or `f64`)
/// * `D` - The dimension of the space (e.g., 2 for 2D, 3 for 3D)
///
/// # Common Use Cases
///
/// - Moving game objects in space
/// - Positioning UI elements with offsets
/// - Camera movement and positioning
/// - Applying coordinate system shifts
///
/// # Examples
///
/// Creating and using a 2D translation:
/// ```
/// # use nalgebra::{Translation2, Point2};
/// // Create a translation that moves points 5 units right and 3 units up
/// let t = Translation2::new(5.0, 3.0);
///
/// // Apply it to a point
/// let point = Point2::new(1.0, 2.0);
/// let moved = t * point;
/// assert_eq!(moved, Point2::new(6.0, 5.0));
/// ```
///
/// Using in 3D space:
/// ```
/// # use nalgebra::{Translation3, Point3};
/// // Moving an object in 3D
/// let t = Translation3::new(10.0, 0.0, -5.0);
/// let position = Point3::new(0.0, 0.0, 0.0);
/// let new_position = t * position;
/// assert_eq!(new_position, Point3::new(10.0, 0.0, -5.0));
/// ```
///
/// # See Also
///
/// * [`Translation::identity`] - Create a translation that doesn't move anything
/// * [`Translation::inverse`] - Get the opposite translation
/// * [`Translation::new`] - Create a translation from coordinates
#[repr(C)]
#[cfg_attr(
    feature = "rkyv-serialize-no-std",
    derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize),
    archive(
        as = "Translation<T::Archived, D>",
        bound(archive = "
        T: rkyv::Archive,
        SVector<T, D>: rkyv::Archive<Archived = SVector<T::Archived, D>>
    ")
    )
)]
#[cfg_attr(feature = "rkyv-serialize", derive(bytecheck::CheckBytes))]
#[cfg_attr(feature = "defmt", derive(defmt::Format))]
#[derive(Copy, Clone)]
pub struct Translation<T, const D: usize> {
    /// The translation coordinates, i.e., how much is added to a point's coordinates when it is
    /// translated.
    pub vector: SVector<T, D>,
}

impl<T: fmt::Debug, const D: usize> fmt::Debug for Translation<T, D> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        self.vector.as_slice().fmt(formatter)
    }
}

impl<T: Scalar + hash::Hash, const D: usize> hash::Hash for Translation<T, D>
where
    Owned<T, Const<D>>: hash::Hash,
{
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        self.vector.hash(state)
    }
}

#[cfg(feature = "bytemuck")]
unsafe impl<T, const D: usize> bytemuck::Zeroable for Translation<T, D>
where
    T: Scalar + bytemuck::Zeroable,
    SVector<T, D>: bytemuck::Zeroable,
{
}

#[cfg(feature = "bytemuck")]
unsafe impl<T, const D: usize> bytemuck::Pod for Translation<T, D>
where
    T: Scalar + bytemuck::Pod,
    SVector<T, D>: bytemuck::Pod,
{
}

#[cfg(feature = "serde-serialize-no-std")]
impl<T: Scalar, const D: usize> Serialize for Translation<T, D>
where
    Owned<T, Const<D>>: Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.vector.serialize(serializer)
    }
}

#[cfg(feature = "serde-serialize-no-std")]
impl<'a, T: Scalar, const D: usize> Deserialize<'a> for Translation<T, D>
where
    Owned<T, Const<D>>: Deserialize<'a>,
{
    fn deserialize<Des>(deserializer: Des) -> Result<Self, Des::Error>
    where
        Des: Deserializer<'a>,
    {
        let matrix = SVector::<T, D>::deserialize(deserializer)?;

        Ok(Translation::from(matrix))
    }
}

impl<T: Scalar, const D: usize> Translation<T, D> {
    /// Creates a new translation from the given vector.
    ///
    /// **Deprecated:** Use `Translation::from(vector)` instead.
    ///
    /// This method constructs a translation from a vector representing the displacement
    /// in each dimension.
    ///
    /// # Arguments
    ///
    /// * `vector` - The displacement vector for the translation
    ///
    /// # Examples
    ///
    /// ```
    /// # use nalgebra::{Translation3, Vector3};
    /// # #[allow(deprecated)]
    /// # {
    /// let vec = Vector3::new(1.0, 2.0, 3.0);
    /// let t = Translation3::from_vector(vec);
    /// # }
    /// ```
    ///
    /// # See Also
    ///
    /// * [`From::from`] - The preferred way to create translations from vectors
    /// * [`Translation::new`] - Create from individual components
    #[inline]
    #[deprecated(note = "Use `::from` instead.")]
    pub const fn from_vector(vector: SVector<T, D>) -> Translation<T, D> {
        Translation { vector }
    }

    /// Computes the inverse of this translation.
    ///
    /// The inverse of a translation is a translation that moves points in the opposite
    /// direction by the same distance. When you compose a translation with its inverse,
    /// you get the identity translation (which doesn't move anything).
    ///
    /// This is useful when you want to "undo" a translation or move back to an original position.
    ///
    /// # Returns
    ///
    /// A new translation that moves points in the opposite direction.
    ///
    /// # Examples
    ///
    /// Basic usage in 3D:
    /// ```
    /// # use nalgebra::{Translation3, Point3};
    /// let t = Translation3::new(1.0, 2.0, 3.0);
    /// let t_inv = t.inverse();
    ///
    /// // Composing a translation with its inverse gives identity
    /// assert_eq!(t * t_inv, Translation3::identity());
    /// assert_eq!(t_inv * t, Translation3::identity());
    ///
    /// // The inverse moves in the opposite direction
    /// let point = Point3::new(5.0, 6.0, 7.0);
    /// let moved = t * point;
    /// let back = t_inv * moved;
    /// assert_eq!(back, point);
    /// ```
    ///
    /// Practical use case - returning to origin:
    /// ```
    /// # use nalgebra::{Translation2, Point2};
    /// // Move a camera position
    /// let camera_offset = Translation2::new(100.0, 50.0);
    /// let screen_pos = Point2::new(0.0, 0.0);
    /// let world_pos = camera_offset * screen_pos;
    ///
    /// // Convert back to screen coordinates
    /// let back_to_screen = camera_offset.inverse() * world_pos;
    /// assert_eq!(back_to_screen, screen_pos);
    /// ```
    ///
    /// Works in all dimensions:
    /// ```
    /// # use nalgebra::Translation2;
    /// let t = Translation2::new(1.0, 2.0);
    /// assert_eq!(t * t.inverse(), Translation2::identity());
    /// assert_eq!(t.inverse() * t, Translation2::identity());
    /// ```
    ///
    /// # See Also
    ///
    /// * [`inverse_mut`](Self::inverse_mut) - In-place version of this method
    /// * [`Translation::identity`] - The neutral translation that doesn't move anything
    #[inline]
    #[must_use = "Did you mean to use inverse_mut()?"]
    pub fn inverse(&self) -> Translation<T, D>
    where
        T: ClosedNeg,
    {
        Translation::from(-&self.vector)
    }

    /// Converts this translation into its equivalent homogeneous transformation matrix.
    ///
    /// A homogeneous matrix is a square matrix representation of the translation that can
    /// be used in computer graphics pipelines. For a D-dimensional translation, this produces
    /// a (D+1)×(D+1) matrix. The translation values appear in the last column.
    ///
    /// Homogeneous matrices are useful because they allow translations to be combined with
    /// other transformations (like rotations and scaling) using matrix multiplication.
    ///
    /// # Returns
    ///
    /// A (D+1)×(D+1) homogeneous transformation matrix representing this translation.
    ///
    /// # Examples
    ///
    /// Converting a 3D translation:
    /// ```
    /// # use nalgebra::{Translation3, Matrix4};
    /// let t = Translation3::new(10.0, 20.0, 30.0);
    /// let matrix = t.to_homogeneous();
    ///
    /// // The result is a 4x4 matrix with translation in the last column
    /// let expected = Matrix4::new(
    ///     1.0, 0.0, 0.0, 10.0,
    ///     0.0, 1.0, 0.0, 20.0,
    ///     0.0, 0.0, 1.0, 30.0,
    ///     0.0, 0.0, 0.0, 1.0
    /// );
    /// assert_eq!(matrix, expected);
    /// ```
    ///
    /// Converting a 2D translation:
    /// ```
    /// # use nalgebra::{Translation2, Matrix3};
    /// let t = Translation2::new(10.0, 20.0);
    /// let matrix = t.to_homogeneous();
    ///
    /// // The result is a 3x3 matrix
    /// let expected = Matrix3::new(
    ///     1.0, 0.0, 10.0,
    ///     0.0, 1.0, 20.0,
    ///     0.0, 0.0, 1.0
    /// );
    /// assert_eq!(matrix, expected);
    /// ```
    ///
    /// Using in a graphics pipeline:
    /// ```
    /// # use nalgebra::{Translation3, Point3, Point4};
    /// let t = Translation3::new(5.0, 0.0, 0.0);
    /// let matrix = t.to_homogeneous();
    ///
    /// // Apply to a homogeneous point
    /// let point = Point4::new(1.0, 2.0, 3.0, 1.0);
    /// let result = matrix * point;
    /// assert_eq!(result, Point4::new(6.0, 2.0, 3.0, 1.0));
    /// ```
    ///
    /// # See Also
    ///
    /// * [`From<Translation>`] for `Matrix` - Implicit conversion to homogeneous matrix
    #[inline]
    #[must_use]
    pub fn to_homogeneous(&self) -> OMatrix<T, DimNameSum<Const<D>, U1>, DimNameSum<Const<D>, U1>>
    where
        T: Zero + One,
        Const<D>: DimNameAdd<U1>,
        DefaultAllocator: Allocator<DimNameSum<Const<D>, U1>, DimNameSum<Const<D>, U1>>,
    {
        let mut res = OMatrix::<T, DimNameSum<Const<D>, U1>, DimNameSum<Const<D>, U1>>::identity();
        res.fixed_view_mut::<D, 1>(0, D).copy_from(&self.vector);

        res
    }

    /// Inverts this translation in-place, modifying it directly.
    ///
    /// This method mutates the translation to become its inverse, which moves points in
    /// the opposite direction. This is more efficient than calling [`inverse`](Self::inverse)
    /// when you don't need to keep the original translation.
    ///
    /// After calling this method, the translation will move points in the opposite direction
    /// by the same distance.
    ///
    /// # Examples
    ///
    /// Basic usage in 3D:
    /// ```
    /// # use nalgebra::{Translation3, Point3};
    /// let original = Translation3::new(1.0, 2.0, 3.0);
    /// let mut t = Translation3::new(1.0, 2.0, 3.0);
    /// t.inverse_mut();
    ///
    /// // t is now the inverse
    /// assert_eq!(original * t, Translation3::identity());
    /// assert_eq!(t * original, Translation3::identity());
    ///
    /// // It moves in the opposite direction
    /// let point = Point3::new(5.0, 6.0, 7.0);
    /// assert_eq!(t * point, Point3::new(4.0, 4.0, 4.0));
    /// ```
    ///
    /// Practical use - toggling a camera offset:
    /// ```
    /// # use nalgebra::{Translation2, Point2};
    /// let mut camera_offset = Translation2::new(100.0, 50.0);
    /// let point = Point2::new(10.0, 20.0);
    ///
    /// // Apply offset
    /// let world_pos = camera_offset * point;
    /// assert_eq!(world_pos, Point2::new(110.0, 70.0));
    ///
    /// // Flip the offset direction
    /// camera_offset.inverse_mut();
    /// let back = camera_offset * world_pos;
    /// assert_eq!(back, point);
    /// ```
    ///
    /// Works in all dimensions:
    /// ```
    /// # use nalgebra::Translation2;
    /// let t = Translation2::new(1.0, 2.0);
    /// let mut inv_t = Translation2::new(1.0, 2.0);
    /// inv_t.inverse_mut();
    /// assert_eq!(t * inv_t, Translation2::identity());
    /// assert_eq!(inv_t * t, Translation2::identity());
    /// ```
    ///
    /// # See Also
    ///
    /// * [`inverse`](Self::inverse) - Non-mutating version that returns a new translation
    /// * [`Translation::identity`] - The neutral translation
    #[inline]
    pub fn inverse_mut(&mut self)
    where
        T: ClosedNeg,
    {
        self.vector.neg_mut()
    }
}

impl<T: Scalar + ClosedAddAssign, const D: usize> Translation<T, D> {
    /// Applies this translation to the given point, moving it by the translation's offset.
    ///
    /// This method transforms a point by adding the translation's displacement vector to it.
    /// It's equivalent to the multiplication `self * pt`, but can be more explicit in intent.
    ///
    /// Translations preserve distances and angles - they simply shift the position of the point
    /// without changing any other properties.
    ///
    /// # Arguments
    ///
    /// * `pt` - The point to translate
    ///
    /// # Returns
    ///
    /// A new point that has been moved by this translation's offset.
    ///
    /// # Examples
    ///
    /// Basic usage in 3D:
    /// ```
    /// # use nalgebra::{Translation3, Point3};
    /// let t = Translation3::new(1.0, 2.0, 3.0);
    /// let point = Point3::new(4.0, 5.0, 6.0);
    /// let transformed = t.transform_point(&point);
    /// assert_eq!(transformed, Point3::new(5.0, 7.0, 9.0));
    /// ```
    ///
    /// Moving a game object:
    /// ```
    /// # use nalgebra::{Translation2, Point2};
    /// // Player position
    /// let player_pos = Point2::new(100.0, 200.0);
    ///
    /// // Movement offset (5 units right, 10 units down)
    /// let movement = Translation2::new(5.0, -10.0);
    ///
    /// // New position after movement
    /// let new_pos = movement.transform_point(&player_pos);
    /// assert_eq!(new_pos, Point2::new(105.0, 190.0));
    /// ```
    ///
    /// Equivalent to multiplication:
    /// ```
    /// # use nalgebra::{Translation2, Point2};
    /// let t = Translation2::new(3.0, 4.0);
    /// let p = Point2::new(1.0, 2.0);
    ///
    /// // These are equivalent
    /// assert_eq!(t.transform_point(&p), t * p);
    /// ```
    ///
    /// Chaining multiple translations:
    /// ```
    /// # use nalgebra::{Translation3, Point3};
    /// let t1 = Translation3::new(1.0, 0.0, 0.0);
    /// let t2 = Translation3::new(0.0, 2.0, 0.0);
    /// let p = Point3::origin();
    ///
    /// let result = t2.transform_point(&t1.transform_point(&p));
    /// assert_eq!(result, Point3::new(1.0, 2.0, 0.0));
    /// ```
    ///
    /// # See Also
    ///
    /// * [`inverse_transform_point`](Self::inverse_transform_point) - Apply the inverse translation
    /// * Multiplication operator (`*`) - Alternative syntax for the same operation
    #[inline]
    #[must_use]
    pub fn transform_point(&self, pt: &Point<T, D>) -> Point<T, D> {
        pt + &self.vector
    }
}

impl<T: Scalar + ClosedSubAssign, const D: usize> Translation<T, D> {
    /// Applies the inverse of this translation to the given point, moving it in the opposite direction.
    ///
    /// This method transforms a point by subtracting the translation's displacement vector from it.
    /// It's useful when you want to "undo" a translation or convert from a translated coordinate
    /// system back to the original one.
    ///
    /// This is more efficient than computing `self.inverse().transform_point(pt)` because it
    /// doesn't create a new translation object.
    ///
    /// # Arguments
    ///
    /// * `pt` - The point to transform
    ///
    /// # Returns
    ///
    /// A new point moved in the opposite direction of this translation.
    ///
    /// # Examples
    ///
    /// Basic usage in 3D:
    /// ```
    /// # use nalgebra::{Translation3, Point3};
    /// let t = Translation3::new(1.0, 2.0, 3.0);
    /// let point = Point3::new(4.0, 5.0, 6.0);
    /// let transformed = t.inverse_transform_point(&point);
    /// assert_eq!(transformed, Point3::new(3.0, 3.0, 3.0));
    /// ```
    ///
    /// Undoing a transformation:
    /// ```
    /// # use nalgebra::{Translation2, Point2};
    /// let t = Translation2::new(10.0, 20.0);
    /// let original = Point2::new(5.0, 5.0);
    ///
    /// // Apply translation
    /// let moved = t.transform_point(&original);
    /// assert_eq!(moved, Point2::new(15.0, 25.0));
    ///
    /// // Undo translation
    /// let back = t.inverse_transform_point(&moved);
    /// assert_eq!(back, original);
    /// ```
    ///
    /// Converting between coordinate systems:
    /// ```
    /// # use nalgebra::{Translation2, Point2};
    /// // Offset from world origin to UI origin
    /// let ui_offset = Translation2::new(100.0, 50.0);
    ///
    /// // Convert UI coordinates to world coordinates
    /// let ui_pos = Point2::new(150.0, 75.0);
    /// let world_pos = ui_offset.inverse_transform_point(&ui_pos);
    /// assert_eq!(world_pos, Point2::new(50.0, 25.0));
    /// ```
    ///
    /// More efficient than creating an inverse:
    /// ```
    /// # use nalgebra::{Translation3, Point3};
    /// let t = Translation3::new(1.0, 2.0, 3.0);
    /// let p = Point3::new(10.0, 20.0, 30.0);
    ///
    /// // These give the same result
    /// let result1 = t.inverse_transform_point(&p);
    /// let result2 = t.inverse().transform_point(&p);
    /// assert_eq!(result1, result2);
    ///
    /// // But inverse_transform_point is more efficient
    /// ```
    ///
    /// # See Also
    ///
    /// * [`transform_point`](Self::transform_point) - Apply the translation in the forward direction
    /// * [`inverse`](Self::inverse) - Get the inverse translation object
    #[inline]
    #[must_use]
    pub fn inverse_transform_point(&self, pt: &Point<T, D>) -> Point<T, D> {
        pt - &self.vector
    }
}

impl<T: Scalar + Eq, const D: usize> Eq for Translation<T, D> {}

impl<T: Scalar + PartialEq, const D: usize> PartialEq for Translation<T, D> {
    #[inline]
    fn eq(&self, right: &Translation<T, D>) -> bool {
        self.vector == right.vector
    }
}

impl<T: Scalar + AbsDiffEq, const D: usize> AbsDiffEq for Translation<T, D>
where
    T::Epsilon: Clone,
{
    type Epsilon = T::Epsilon;

    #[inline]
    fn default_epsilon() -> Self::Epsilon {
        T::default_epsilon()
    }

    #[inline]
    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        self.vector.abs_diff_eq(&other.vector, epsilon)
    }
}

impl<T: Scalar + RelativeEq, const D: usize> RelativeEq for Translation<T, D>
where
    T::Epsilon: Clone,
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
        self.vector
            .relative_eq(&other.vector, epsilon, max_relative)
    }
}

impl<T: Scalar + UlpsEq, const D: usize> UlpsEq for Translation<T, D>
where
    T::Epsilon: Clone,
{
    #[inline]
    fn default_max_ulps() -> u32 {
        T::default_max_ulps()
    }

    #[inline]
    fn ulps_eq(&self, other: &Self, epsilon: Self::Epsilon, max_ulps: u32) -> bool {
        self.vector.ulps_eq(&other.vector, epsilon, max_ulps)
    }
}

/*
 *
 * Display
 *
 */
impl<T: Scalar + fmt::Display, const D: usize> fmt::Display for Translation<T, D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let precision = f.precision().unwrap_or(3);

        writeln!(f, "Translation {{")?;
        write!(f, "{:.*}", precision, self.vector)?;
        writeln!(f, "}}")
    }
}
