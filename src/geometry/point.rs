// Needed otherwise the rkyv macros generate code incompatible with rust-2024
#![cfg_attr(feature = "rkyv-serialize", allow(unsafe_op_in_unsafe_fn))]

use approx::{AbsDiffEq, RelativeEq, UlpsEq};
use num::{One, Zero};
use std::cmp::Ordering;
use std::fmt;
use std::hash;

#[cfg(feature = "rkyv-serialize")]
use rkyv::bytecheck;
#[cfg(feature = "serde-serialize-no-std")]
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use simba::simd::SimdPartialOrd;

use crate::base::allocator::Allocator;
use crate::base::dimension::{DimName, DimNameAdd, DimNameSum, U1};
use crate::base::iter::{MatrixIter, MatrixIterMut};
use crate::base::{Const, DefaultAllocator, OVector, Scalar};
use simba::scalar::{ClosedAddAssign, ClosedMulAssign, ClosedSubAssign};
use std::mem::MaybeUninit;

/// A point in an euclidean space.
///
/// The difference between a point and a vector is only semantic. See [the user guide](https://www.nalgebra.rs/docs/user_guide/points_and_transformations)
/// for details on the distinction. The most notable difference that vectors ignore translations.
/// In particular, an [`Isometry2`](crate::Isometry2) or [`Isometry3`](crate::Isometry3) will
/// transform points by applying a rotation and a translation on them. However, these isometries
/// will only apply rotations to vectors (when doing `isometry * vector`, the translation part of
/// the isometry is ignored).
///
/// # Construction
/// * [From individual components <span style="float:right;">`new`…</span>](#construction-from-individual-components)
/// * [Swizzling <span style="float:right;">`xx`, `yxz`…</span>](#swizzling)
/// * [Other construction methods <span style="float:right;">`origin`, `from_slice`, `from_homogeneous`…</span>](#other-construction-methods)
///
/// # Transformation
/// Transforming a point by an [Isometry](crate::Isometry), [rotation](crate::Rotation), etc. can be
/// achieved by multiplication, e.g., `isometry * point` or `rotation * point`. Some of these transformation
/// may have some other methods, e.g., `isometry.inverse_transform_point(&point)`. See the documentation
/// of said transformations for details.
#[repr(C)]
#[derive(Clone)]
#[cfg_attr(feature = "rkyv-serialize", derive(bytecheck::CheckBytes))]
#[cfg_attr(
    feature = "rkyv-serialize-no-std",
    derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize),
    archive(
        as = "OPoint<T::Archived, D>",
        bound(archive = "
        T: rkyv::Archive,
        T::Archived: Scalar,
        OVector<T, D>: rkyv::Archive<Archived = OVector<T::Archived, D>>,
        DefaultAllocator: Allocator<D>,
    ")
    )
)]
#[cfg_attr(feature = "defmt", derive(defmt::Format))]
pub struct OPoint<T: Scalar, D: DimName>
where
    DefaultAllocator: Allocator<D>,
{
    /// The coordinates of this point, i.e., the shift from the origin.
    ///
    /// # Understanding Points vs Vectors
    ///
    /// While points and vectors both have coordinates, they represent different concepts:
    /// - A **point** represents a location in space
    /// - A **vector** represents a displacement or direction
    ///
    /// The `coords` field stores the vector from the origin to this point's location.
    ///
    /// # Examples
    ///
    /// ```
    /// # use nalgebra::{Point2, Point3, Vector2, Vector3};
    /// // Access coordinates directly
    /// let p = Point2::new(3.0, 4.0);
    /// assert_eq!(p.coords, Vector2::new(3.0, 4.0));
    ///
    /// // Modify coordinates
    /// let mut p = Point3::new(1.0, 2.0, 3.0);
    /// p.coords.x = 10.0;
    /// assert_eq!(p.coords, Vector3::new(10.0, 2.0, 3.0));
    ///
    /// // Convert to a vector when you need displacement semantics
    /// let origin = Point2::new(0.0, 0.0);
    /// let position = Point2::new(5.0, 12.0);
    /// let direction = position.coords - origin.coords;
    /// assert_eq!(direction, Vector2::new(5.0, 12.0));
    /// ```
    ///
    /// # See Also
    ///
    /// - [`into()`](Self#impl-From<OPoint<T,+D>>-for-OVector<T,+DimNameSum<D,+U1>>) - Convert point to homogeneous vector
    /// - [`from()`](Self#impl-From<OVector<T,+D>>-for-OPoint<T,+D>) - Create point from vector
    pub coords: OVector<T, D>,
}

impl<T: Scalar + fmt::Debug, D: DimName> fmt::Debug for OPoint<T, D>
where
    DefaultAllocator: Allocator<D>,
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        self.coords.as_slice().fmt(formatter)
    }
}

impl<T: Scalar + hash::Hash, D: DimName> hash::Hash for OPoint<T, D>
where
    DefaultAllocator: Allocator<D>,
{
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        self.coords.hash(state)
    }
}

impl<T: Scalar + Copy, D: DimName> Copy for OPoint<T, D>
where
    DefaultAllocator: Allocator<D>,
    OVector<T, D>: Copy,
{
}

#[cfg(feature = "bytemuck")]
unsafe impl<T: Scalar, D: DimName> bytemuck::Zeroable for OPoint<T, D>
where
    OVector<T, D>: bytemuck::Zeroable,
    DefaultAllocator: Allocator<D>,
{
}

#[cfg(feature = "bytemuck")]
unsafe impl<T: Scalar, D: DimName> bytemuck::Pod for OPoint<T, D>
where
    T: Copy,
    OVector<T, D>: bytemuck::Pod,
    DefaultAllocator: Allocator<D>,
{
}

#[cfg(feature = "serde-serialize-no-std")]
impl<T: Scalar, D: DimName> Serialize for OPoint<T, D>
where
    DefaultAllocator: Allocator<D>,
    <DefaultAllocator as Allocator<D>>::Buffer<T>: Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.coords.serialize(serializer)
    }
}

#[cfg(feature = "serde-serialize-no-std")]
impl<'a, T: Scalar, D: DimName> Deserialize<'a> for OPoint<T, D>
where
    DefaultAllocator: Allocator<D>,
    <DefaultAllocator as Allocator<D>>::Buffer<T>: Deserialize<'a>,
{
    fn deserialize<Des>(deserializer: Des) -> Result<Self, Des::Error>
    where
        Des: Deserializer<'a>,
    {
        let coords = OVector::<T, D>::deserialize(deserializer)?;

        Ok(Self::from(coords))
    }
}

impl<T: Scalar, D: DimName> OPoint<T, D>
where
    DefaultAllocator: Allocator<D>,
{
    /// Returns a new point by applying a function to each coordinate.
    ///
    /// This method creates a new point where each coordinate is the result of applying
    /// the function `f` to the corresponding coordinate of `self`. The function can
    /// change the type of the coordinates (e.g., from `f64` to `i32`).
    ///
    /// # Arguments
    ///
    /// * `f` - A closure that takes each coordinate value and returns a new value
    ///
    /// # Examples
    ///
    /// ## Scaling coordinates
    /// ```
    /// # use nalgebra::Point2;
    /// let p = Point2::new(1.0, 2.0);
    /// let scaled = p.map(|e| e * 10.0);
    /// assert_eq!(scaled, Point2::new(10.0, 20.0));
    /// ```
    ///
    /// ## Converting coordinate types
    /// ```
    /// # use nalgebra::Point3;
    /// let p = Point3::new(1.7f32, 2.3f32, 3.9f32);
    /// let rounded = p.map(|e| e.round() as i32);
    /// assert_eq!(rounded, Point3::new(2, 2, 4));
    /// ```
    ///
    /// ## Negating all coordinates
    /// ```
    /// # use nalgebra::Point2;
    /// let p = Point2::new(-5.0, 3.0);
    /// let negated = p.map(|e| -e);
    /// assert_eq!(negated, Point2::new(5.0, -3.0));
    /// ```
    ///
    /// ## Squaring each coordinate
    /// ```
    /// # use nalgebra::Point3;
    /// let p = Point3::new(1.0, 2.0, 3.0);
    /// let squared = p.map(|e| e * e);
    /// assert_eq!(squared, Point3::new(1.0, 4.0, 9.0));
    /// ```
    ///
    /// # See Also
    ///
    /// * [`apply`](Self::apply) - Modify coordinates in place with a closure
    /// * [`cast`](Self::cast) - Convert coordinate types using type conversion
    #[inline]
    #[must_use]
    pub fn map<T2: Scalar, F: FnMut(T) -> T2>(&self, f: F) -> OPoint<T2, D>
    where
        DefaultAllocator: Allocator<D>,
    {
        self.coords.map(f).into()
    }

    /// Modifies each coordinate of the point in place using a closure.
    ///
    /// This method applies the function `f` to each coordinate of the point, modifying
    /// the point directly. Unlike [`map`](Self::map), this method doesn't create a new
    /// point and doesn't change the coordinate type.
    ///
    /// # Arguments
    ///
    /// * `f` - A closure that receives a mutable reference to each coordinate
    ///
    /// # Examples
    ///
    /// ## Scaling coordinates in place
    /// ```
    /// # use nalgebra::Point2;
    /// let mut p = Point2::new(1.0, 2.0);
    /// p.apply(|e| *e *= 10.0);
    /// assert_eq!(p, Point2::new(10.0, 20.0));
    /// ```
    ///
    /// ## Doubling each coordinate
    /// ```
    /// # use nalgebra::Point3;
    /// let mut p = Point3::new(1.0, 2.0, 3.0);
    /// p.apply(|e| *e = *e * 2.0);
    /// assert_eq!(p, Point3::new(2.0, 4.0, 6.0));
    /// ```
    ///
    /// ## Adding a constant to all coordinates
    /// ```
    /// # use nalgebra::Point2;
    /// let mut p = Point2::new(1.0, 2.0);
    /// p.apply(|e| *e = *e + 5.0);
    /// assert_eq!(p, Point2::new(6.0, 7.0));
    /// ```
    ///
    /// ## Normalizing screen coordinates
    /// ```
    /// # use nalgebra::Point2;
    /// // Convert from [0, 800] to [-1, 1] range
    /// let mut screen_pos = Point2::new(400.0, 600.0);
    /// screen_pos.apply(|e| *e = (*e / 400.0) - 1.0);
    /// assert_eq!(screen_pos, Point2::new(0.0, 0.5));
    /// ```
    ///
    /// # See Also
    ///
    /// * [`map`](Self::map) - Create a new point by applying a function to coordinates
    /// * [`iter_mut`](Self::iter_mut) - Iterate over mutable references to coordinates
    #[inline]
    pub fn apply<F: FnMut(&mut T)>(&mut self, f: F) {
        self.coords.apply(f)
    }

    /// Converts this point to homogeneous coordinates by appending a `1`.
    ///
    /// Homogeneous coordinates are commonly used in computer graphics and geometric
    /// transformations. A point in N-dimensional space becomes a vector in (N+1)-dimensional
    /// space with the extra coordinate set to `1`. This allows representing translations
    /// as matrix multiplications.
    ///
    /// This is the same as calling `.into()` with the appropriate type annotation.
    ///
    /// # What are Homogeneous Coordinates?
    ///
    /// Homogeneous coordinates add an extra dimension to make certain transformations
    /// easier to express:
    /// - A 2D point `(x, y)` becomes the 3D vector `(x, y, 1)`
    /// - A 3D point `(x, y, z)` becomes the 4D vector `(x, y, z, 1)`
    ///
    /// This representation is essential for:
    /// - Applying affine transformations (rotation + translation) with matrix multiplication
    /// - Perspective projection in 3D graphics
    /// - Combining multiple transformations efficiently
    ///
    /// # Examples
    ///
    /// ## Basic conversion to homogeneous coordinates
    /// ```
    /// # use nalgebra::{Point2, Vector3};
    /// let p = Point2::new(10.0, 20.0);
    /// let homogeneous = p.to_homogeneous();
    /// assert_eq!(homogeneous, Vector3::new(10.0, 20.0, 1.0));
    /// ```
    ///
    /// ## Works in any dimension
    /// ```
    /// # use nalgebra::{Point3, Vector4};
    /// let p = Point3::new(10.0, 20.0, 30.0);
    /// let homogeneous = p.to_homogeneous();
    /// assert_eq!(homogeneous, Vector4::new(10.0, 20.0, 30.0, 1.0));
    /// ```
    ///
    /// ## Useful for transformations in graphics
    /// ```
    /// # use nalgebra::{Point2, Vector3, Matrix3};
    /// // Create a 2D point
    /// let point = Point2::new(5.0, 3.0);
    ///
    /// // Translation matrix (move by (10, 20))
    /// let translation = Matrix3::new(
    ///     1.0, 0.0, 10.0,
    ///     0.0, 1.0, 20.0,
    ///     0.0, 0.0,  1.0,
    /// );
    ///
    /// // Apply transformation using homogeneous coordinates
    /// let transformed = translation * point.to_homogeneous();
    /// assert_eq!(transformed, Vector3::new(15.0, 23.0, 1.0));
    /// ```
    ///
    /// # See Also
    ///
    /// * [`from_homogeneous`](Self::from_homogeneous) - Convert from homogeneous coordinates back to a point
    /// * [`coords`](Self::coords) - Access the underlying coordinate vector
    #[inline]
    #[must_use]
    pub fn to_homogeneous(&self) -> OVector<T, DimNameSum<D, U1>>
    where
        T: One,
        D: DimNameAdd<U1>,
        DefaultAllocator: Allocator<DimNameSum<D, U1>>,
    {
        // TODO: this is mostly a copy-past from Vector::push.
        //       But we can’t use Vector::push because of the DimAdd bound
        //       (which we don’t use because we use DimNameAdd).
        //       We should find a way to re-use Vector::push.
        let len = self.len();
        let mut res = crate::Matrix::uninit(DimNameSum::<D, U1>::name(), Const::<1>);
        // This is basically a copy_from except that we warp the copied
        // values into MaybeUninit.
        res.generic_view_mut((0, 0), self.coords.shape_generic())
            .zip_apply(&self.coords, |out, e| *out = MaybeUninit::new(e));
        res[(len, 0)] = MaybeUninit::new(T::one());

        // Safety: res has been fully initialized.
        unsafe { res.assume_init() }
    }

    /// Performs linear interpolation (lerp) between two points.
    ///
    /// Computes a point along the line segment between `self` and `rhs`. The parameter `t`
    /// controls the position:
    /// - When `t = 0.0`, returns `self`
    /// - When `t = 1.0`, returns `rhs`
    /// - When `t = 0.5`, returns the midpoint
    ///
    /// The formula used is: `self * (1.0 - t) + rhs * t`
    ///
    /// # Arguments
    ///
    /// * `rhs` - The target point to interpolate towards
    /// * `t` - The interpolation parameter (not restricted to [0, 1])
    ///
    /// # Note
    ///
    /// The parameter `t` is not clamped to [0, 1]. Values outside this range will
    /// extrapolate beyond the two points:
    /// - `t < 0.0` extrapolates beyond `self`
    /// - `t > 1.0` extrapolates beyond `rhs`
    ///
    /// # Examples
    ///
    /// ## Basic interpolation
    /// ```
    /// # use nalgebra::Point3;
    /// let start = Point3::new(0.0, 0.0, 0.0);
    /// let end = Point3::new(10.0, 20.0, 30.0);
    ///
    /// // At t=0, we get the start point
    /// assert_eq!(start.lerp(&end, 0.0), start);
    ///
    /// // At t=1, we get the end point
    /// assert_eq!(start.lerp(&end, 1.0), end);
    ///
    /// // At t=0.5, we get the midpoint
    /// assert_eq!(start.lerp(&end, 0.5), Point3::new(5.0, 10.0, 15.0));
    /// ```
    ///
    /// ## Animation and smooth movement
    /// ```
    /// # use nalgebra::Point2;
    /// let start_pos = Point2::new(0.0, 0.0);
    /// let target_pos = Point2::new(100.0, 50.0);
    ///
    /// // Simulate 10% progress in an animation
    /// let current_pos = start_pos.lerp(&target_pos, 0.1);
    /// assert_eq!(current_pos, Point2::new(10.0, 5.0));
    ///
    /// // 90% complete
    /// let almost_there = start_pos.lerp(&target_pos, 0.9);
    /// assert_eq!(almost_there, Point2::new(90.0, 45.0));
    /// ```
    ///
    /// ## Extrapolation (t outside [0, 1])
    /// ```
    /// # use nalgebra::Point2;
    /// let a = Point2::new(0.0, 0.0);
    /// let b = Point2::new(10.0, 10.0);
    ///
    /// // Extrapolate beyond b
    /// let beyond = a.lerp(&b, 2.0);
    /// assert_eq!(beyond, Point2::new(20.0, 20.0));
    ///
    /// // Extrapolate before a
    /// let before = a.lerp(&b, -1.0);
    /// assert_eq!(before, Point2::new(-10.0, -10.0));
    /// ```
    ///
    /// ## Finding points along a path
    /// ```
    /// # use nalgebra::Point3;
    /// let waypoint1 = Point3::new(1.0, 2.0, 3.0);
    /// let waypoint2 = Point3::new(10.0, 20.0, 30.0);
    ///
    /// // Find a point 10% of the way from waypoint1 to waypoint2
    /// let position = waypoint1.lerp(&waypoint2, 0.1);
    /// assert_eq!(position, Point3::new(1.9, 3.8, 5.7));
    /// ```
    ///
    /// # See Also
    ///
    /// * [`coords`](Self::coords) - Access the underlying coordinate vector
    #[must_use]
    pub fn lerp(&self, rhs: &OPoint<T, D>, t: T) -> OPoint<T, D>
    where
        T: Scalar + Zero + One + ClosedAddAssign + ClosedSubAssign + ClosedMulAssign,
    {
        OPoint {
            coords: self.coords.lerp(&rhs.coords, t),
        }
    }

    /// Creates a new point with the given coordinates.
    #[deprecated(note = "Use Point::from(vector) instead.")]
    #[inline]
    pub const fn from_coordinates(coords: OVector<T, D>) -> Self {
        Self { coords }
    }

    /// Returns the number of coordinates (dimensions) of this point.
    ///
    /// This method returns the dimensionality of the space in which the point exists.
    /// For example, a 2D point has 2 coordinates (x, y), while a 3D point has 3
    /// coordinates (x, y, z).
    ///
    /// # Returns
    ///
    /// The number of coordinates in this point.
    ///
    /// # Examples
    ///
    /// ## Getting dimensions of different points
    /// ```
    /// # use nalgebra::{Point1, Point2, Point3, Point4};
    /// let p1 = Point1::new(1.0);
    /// assert_eq!(p1.len(), 1);
    ///
    /// let p2 = Point2::new(1.0, 2.0);
    /// assert_eq!(p2.len(), 2);
    ///
    /// let p3 = Point3::new(1.0, 2.0, 3.0);
    /// assert_eq!(p3.len(), 3);
    ///
    /// let p4 = Point4::new(1.0, 2.0, 3.0, 4.0);
    /// assert_eq!(p4.len(), 4);
    /// ```
    ///
    /// ## Using len() for dynamic operations
    /// ```
    /// # use nalgebra::Point3;
    /// let point = Point3::new(1.0, 2.0, 3.0);
    ///
    /// // Iterate over all coordinates by index
    /// for i in 0..point.len() {
    ///     println!("Coordinate {}: {}", i, point[i]);
    /// }
    /// ```
    ///
    /// # Note
    ///
    /// This method returns the *dimension* of the point, not the magnitude or
    /// Euclidean length. To compute the distance from the origin, use the vector
    /// norm methods on [`coords`](Self::coords):
    ///
    /// ```
    /// # use nalgebra::Point3;
    /// let point = Point3::new(3.0, 4.0, 0.0);
    /// let dimension = point.len();  // Returns 3 (number of coordinates)
    /// let distance = point.coords.norm();  // Returns 5.0 (distance from origin)
    /// assert_eq!(dimension, 3);
    /// assert_eq!(distance, 5.0);
    /// ```
    ///
    /// # See Also
    ///
    /// * [`is_empty`](Self::is_empty) - Check if the point has zero dimensions
    /// * [`iter`](Self::iter) - Iterate over the coordinates
    /// * [`coords`](Self::coords) - Access the coordinate vector
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.coords.len()
    }

    /// Returns `true` if the point has zero dimensions.
    ///
    /// In practice, this method always returns `false` for standard point types
    /// like `Point1`, `Point2`, `Point3`, etc., since they all have at least one
    /// dimension. This method exists for API completeness and consistency with
    /// collection-like types.
    ///
    /// # Returns
    ///
    /// `true` if the point has 0 dimensions, `false` otherwise.
    ///
    /// # Examples
    ///
    /// ## Standard points are never empty
    /// ```
    /// # use nalgebra::{Point1, Point2, Point3};
    /// let p1 = Point1::new(1.0);
    /// assert!(!p1.is_empty());
    ///
    /// let p2 = Point2::new(1.0, 2.0);
    /// assert!(!p2.is_empty());
    ///
    /// let p3 = Point3::new(1.0, 2.0, 3.0);
    /// assert!(!p3.is_empty());
    /// ```
    ///
    /// ## Checking before iteration
    /// ```
    /// # use nalgebra::Point3;
    /// let point = Point3::new(1.0, 2.0, 3.0);
    ///
    /// if !point.is_empty() {
    ///     // Safe to access coordinates
    ///     println!("First coordinate: {}", point[0]);
    /// }
    /// ```
    ///
    /// # See Also
    ///
    /// * [`len`](Self::len) - Get the number of coordinates
    /// * [`iter`](Self::iter) - Iterate over the coordinates
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The stride of this point. This is the number of buffer element separating each component of
    /// this point.
    #[inline]
    #[deprecated(note = "This methods is no longer significant and will always return 1.")]
    pub fn stride(&self) -> usize {
        self.coords.strides().0
    }

    /// Returns an iterator over immutable references to the point's coordinates.
    ///
    /// This method provides read-only access to each coordinate in order. Use
    /// [`iter_mut`](Self::iter_mut) if you need to modify the coordinates during iteration.
    ///
    /// # Returns
    ///
    /// An iterator yielding references to each coordinate.
    ///
    /// # Examples
    ///
    /// ## Basic iteration
    /// ```
    /// # use nalgebra::Point3;
    /// let p = Point3::new(1.0, 2.0, 3.0);
    /// let mut it = p.iter().cloned();
    ///
    /// assert_eq!(it.next(), Some(1.0));
    /// assert_eq!(it.next(), Some(2.0));
    /// assert_eq!(it.next(), Some(3.0));
    /// assert_eq!(it.next(), None);
    /// ```
    ///
    /// ## Collecting coordinates into a vector
    /// ```
    /// # use nalgebra::Point3;
    /// let point = Point3::new(10.0, 20.0, 30.0);
    /// let coords: Vec<f64> = point.iter().cloned().collect();
    /// assert_eq!(coords, vec![10.0, 20.0, 30.0]);
    /// ```
    ///
    /// ## Finding maximum coordinate
    /// ```
    /// # use nalgebra::Point3;
    /// let point = Point3::new(5.0, 12.0, 3.0);
    /// let max_coord = point.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    /// assert_eq!(max_coord, 12.0);
    /// ```
    ///
    /// ## Computing sum of coordinates
    /// ```
    /// # use nalgebra::Point2;
    /// let point = Point2::new(3.0, 7.0);
    /// let sum: f64 = point.iter().sum();
    /// assert_eq!(sum, 10.0);
    /// ```
    ///
    /// ## Checking if all coordinates are positive
    /// ```
    /// # use nalgebra::Point3;
    /// let point1 = Point3::new(1.0, 2.0, 3.0);
    /// assert!(point1.iter().all(|&x| x > 0.0));
    ///
    /// let point2 = Point3::new(1.0, -2.0, 3.0);
    /// assert!(!point2.iter().all(|&x| x > 0.0));
    /// ```
    ///
    /// # See Also
    ///
    /// * [`iter_mut`](Self::iter_mut) - Iterate with mutable references
    /// * [`len`](Self::len) - Get the number of coordinates
    /// * [`apply`](Self::apply) - Modify coordinates with a closure
    #[inline]
    pub fn iter(
        &self,
    ) -> MatrixIter<'_, T, D, Const<1>, <DefaultAllocator as Allocator<D>>::Buffer<T>> {
        self.coords.iter()
    }

    /// Gets a reference to i-th element of this point without bound-checking.
    ///
    /// # Safety
    ///
    /// `i` must be less than `self.len()`.
    #[inline]
    #[must_use]
    pub unsafe fn get_unchecked(&self, i: usize) -> &T {
        unsafe { self.coords.vget_unchecked(i) }
    }

    /// Returns an iterator over mutable references to the point's coordinates.
    ///
    /// This method allows you to modify each coordinate in place during iteration.
    /// For read-only access, use [`iter`](Self::iter) instead.
    ///
    /// # Returns
    ///
    /// An iterator yielding mutable references to each coordinate.
    ///
    /// # Examples
    ///
    /// ## Scaling all coordinates
    /// ```
    /// # use nalgebra::Point3;
    /// let mut p = Point3::new(1.0, 2.0, 3.0);
    ///
    /// for coord in p.iter_mut() {
    ///     *coord *= 10.0;
    /// }
    ///
    /// assert_eq!(p, Point3::new(10.0, 20.0, 30.0));
    /// ```
    ///
    /// ## Doubling coordinates
    /// ```
    /// # use nalgebra::Point3;
    /// let mut point = Point3::new(1.0, 2.0, 3.0);
    ///
    /// for coord in point.iter_mut() {
    ///     *coord = *coord * 2.0;
    /// }
    ///
    /// assert_eq!(point, Point3::new(2.0, 4.0, 6.0));
    /// ```
    ///
    /// ## Adding a constant to each coordinate
    /// ```
    /// # use nalgebra::Point2;
    /// let mut point = Point2::new(1.0, 4.0);
    ///
    /// for coord in point.iter_mut() {
    ///     *coord = *coord + 10.0;
    /// }
    ///
    /// assert_eq!(point, Point2::new(11.0, 14.0));
    /// ```
    ///
    /// ## Conditionally modifying coordinates
    /// ```
    /// # use nalgebra::Point3;
    /// let mut point = Point3::new(1.0, 5.0, 3.0);
    ///
    /// for coord in point.iter_mut() {
    ///     if *coord > 4.0 {
    ///         *coord = 4.0;
    ///     }
    /// }
    ///
    /// assert_eq!(point, Point3::new(1.0, 4.0, 3.0));
    /// ```
    ///
    /// ## Normalizing coordinates to [0, 1] range
    /// ```
    /// # use nalgebra::Point2;
    /// let mut screen_pos = Point2::new(400.0, 300.0);
    /// let screen_size = Point2::new(800.0, 600.0);
    ///
    /// for (coord, size) in screen_pos.iter_mut().zip(screen_size.iter()) {
    ///     *coord /= size;
    /// }
    ///
    /// assert_eq!(screen_pos, Point2::new(0.5, 0.5));
    /// ```
    ///
    /// # See Also
    ///
    /// * [`iter`](Self::iter) - Iterate with immutable references
    /// * [`apply`](Self::apply) - Modify coordinates with a closure
    /// * [`map`](Self::map) - Create a new point by transforming coordinates
    #[inline]
    pub fn iter_mut(
        &mut self,
    ) -> MatrixIterMut<'_, T, D, Const<1>, <DefaultAllocator as Allocator<D>>::Buffer<T>> {
        self.coords.iter_mut()
    }

    /// Gets a mutable reference to i-th element of this point without bound-checking.
    ///
    /// # Safety
    ///
    /// `i` must be less than `self.len()`.
    #[inline]
    #[must_use]
    pub unsafe fn get_unchecked_mut(&mut self, i: usize) -> &mut T {
        unsafe { self.coords.vget_unchecked_mut(i) }
    }

    /// Swaps two entries without bound-checking.
    ///
    /// # Safety
    ///
    /// `i1` and `i2` must be less than `self.len()`.
    #[inline]
    pub unsafe fn swap_unchecked(&mut self, i1: usize, i2: usize) {
        unsafe { self.coords.swap_unchecked((i1, 0), (i2, 0)) }
    }
}

impl<T: Scalar + AbsDiffEq, D: DimName> AbsDiffEq for OPoint<T, D>
where
    T::Epsilon: Clone,
    DefaultAllocator: Allocator<D>,
{
    type Epsilon = T::Epsilon;

    #[inline]
    fn default_epsilon() -> Self::Epsilon {
        T::default_epsilon()
    }

    #[inline]
    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        self.coords.abs_diff_eq(&other.coords, epsilon)
    }
}

impl<T: Scalar + RelativeEq, D: DimName> RelativeEq for OPoint<T, D>
where
    T::Epsilon: Clone,
    DefaultAllocator: Allocator<D>,
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
        self.coords
            .relative_eq(&other.coords, epsilon, max_relative)
    }
}

impl<T: Scalar + UlpsEq, D: DimName> UlpsEq for OPoint<T, D>
where
    T::Epsilon: Clone,
    DefaultAllocator: Allocator<D>,
{
    #[inline]
    fn default_max_ulps() -> u32 {
        T::default_max_ulps()
    }

    #[inline]
    fn ulps_eq(&self, other: &Self, epsilon: Self::Epsilon, max_ulps: u32) -> bool {
        self.coords.ulps_eq(&other.coords, epsilon, max_ulps)
    }
}

impl<T: Scalar + Eq, D: DimName> Eq for OPoint<T, D> where DefaultAllocator: Allocator<D> {}

impl<T: Scalar, D: DimName> PartialEq for OPoint<T, D>
where
    DefaultAllocator: Allocator<D>,
{
    #[inline]
    fn eq(&self, right: &Self) -> bool {
        self.coords == right.coords
    }
}

impl<T: Scalar + PartialOrd, D: DimName> PartialOrd for OPoint<T, D>
where
    DefaultAllocator: Allocator<D>,
{
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.coords.partial_cmp(&other.coords)
    }

    #[inline]
    fn lt(&self, right: &Self) -> bool {
        self.coords.lt(&right.coords)
    }

    #[inline]
    fn le(&self, right: &Self) -> bool {
        self.coords.le(&right.coords)
    }

    #[inline]
    fn gt(&self, right: &Self) -> bool {
        self.coords.gt(&right.coords)
    }

    #[inline]
    fn ge(&self, right: &Self) -> bool {
        self.coords.ge(&right.coords)
    }
}

/*
 * inf/sup
 */
impl<T: Scalar + SimdPartialOrd, D: DimName> OPoint<T, D>
where
    DefaultAllocator: Allocator<D>,
{
    /// Computes the component-wise minimum of two points.
    ///
    /// Returns a new point where each coordinate is the minimum of the corresponding
    /// coordinates from `self` and `other`. This is also known as the infimum or
    /// componentwise min operation.
    ///
    /// # Arguments
    ///
    /// * `other` - The other point to compare with
    ///
    /// # Returns
    ///
    /// A new point with the minimum coordinate values from both points.
    ///
    /// # Examples
    ///
    /// ## Basic component-wise minimum
    /// ```
    /// # use nalgebra::Point2;
    /// let p1 = Point2::new(1.0, 5.0);
    /// let p2 = Point2::new(3.0, 2.0);
    ///
    /// let min_point = p1.inf(&p2);
    /// assert_eq!(min_point, Point2::new(1.0, 2.0));
    /// ```
    ///
    /// ## Finding bounding box minimum corner
    /// ```
    /// # use nalgebra::Point3;
    /// // Define some points in 3D space
    /// let p1 = Point3::new(2.0, -1.0, 5.0);
    /// let p2 = Point3::new(-3.0, 4.0, 1.0);
    /// let p3 = Point3::new(0.0, 2.0, -2.0);
    ///
    /// // Find the minimum corner of the bounding box
    /// let min_corner = p1.inf(&p2).inf(&p3);
    /// assert_eq!(min_corner, Point3::new(-3.0, -1.0, -2.0));
    /// ```
    ///
    /// ## Clamping a point to a minimum bound
    /// ```
    /// # use nalgebra::Point2;
    /// let position = Point2::new(-5.0, 10.0);
    /// let min_bound = Point2::new(0.0, 0.0);
    ///
    /// // Ensure position doesn't go below min_bound
    /// // (You'd also need sup() for the upper bound)
    /// let clamped = position.sup(&min_bound);
    /// assert_eq!(clamped, Point2::new(0.0, 10.0));
    /// ```
    ///
    /// ## Works with any comparable type
    /// ```
    /// # use nalgebra::Point3;
    /// let p1 = Point3::new(10, 20, 30);
    /// let p2 = Point3::new(15, 5, 25);
    ///
    /// let min = p1.inf(&p2);
    /// assert_eq!(min, Point3::new(10, 5, 25));
    /// ```
    ///
    /// # See Also
    ///
    /// * [`sup`](Self::sup) - Compute component-wise maximum
    /// * [`inf_sup`](Self::inf_sup) - Compute both minimum and maximum at once
    #[inline]
    #[must_use]
    pub fn inf(&self, other: &Self) -> OPoint<T, D> {
        self.coords.inf(&other.coords).into()
    }

    /// Computes the component-wise maximum of two points.
    ///
    /// Returns a new point where each coordinate is the maximum of the corresponding
    /// coordinates from `self` and `other`. This is also known as the supremum or
    /// componentwise max operation.
    ///
    /// # Arguments
    ///
    /// * `other` - The other point to compare with
    ///
    /// # Returns
    ///
    /// A new point with the maximum coordinate values from both points.
    ///
    /// # Examples
    ///
    /// ## Basic component-wise maximum
    /// ```
    /// # use nalgebra::Point2;
    /// let p1 = Point2::new(1.0, 5.0);
    /// let p2 = Point2::new(3.0, 2.0);
    ///
    /// let max_point = p1.sup(&p2);
    /// assert_eq!(max_point, Point2::new(3.0, 5.0));
    /// ```
    ///
    /// ## Finding bounding box maximum corner
    /// ```
    /// # use nalgebra::Point3;
    /// // Define some points in 3D space
    /// let p1 = Point3::new(2.0, -1.0, 5.0);
    /// let p2 = Point3::new(-3.0, 4.0, 1.0);
    /// let p3 = Point3::new(0.0, 2.0, 7.0);
    ///
    /// // Find the maximum corner of the bounding box
    /// let max_corner = p1.sup(&p2).sup(&p3);
    /// assert_eq!(max_corner, Point3::new(2.0, 4.0, 7.0));
    /// ```
    ///
    /// ## Creating an axis-aligned bounding box
    /// ```
    /// # use nalgebra::Point2;
    /// let objects = vec![
    ///     Point2::new(1.0, 2.0),
    ///     Point2::new(5.0, 1.0),
    ///     Point2::new(3.0, 6.0),
    /// ];
    ///
    /// // Find bounding box for all objects
    /// let mut min = objects[0];
    /// let mut max = objects[0];
    ///
    /// for point in &objects[1..] {
    ///     min = min.inf(point);
    ///     max = max.sup(point);
    /// }
    ///
    /// assert_eq!(min, Point2::new(1.0, 1.0));
    /// assert_eq!(max, Point2::new(5.0, 6.0));
    /// ```
    ///
    /// ## Clamping a point to upper bounds
    /// ```
    /// # use nalgebra::Point2;
    /// let position = Point2::new(150.0, 80.0);
    /// let max_bound = Point2::new(100.0, 100.0);
    ///
    /// // Ensure position doesn't exceed max_bound
    /// let clamped = position.inf(&max_bound);
    /// assert_eq!(clamped, Point2::new(100.0, 80.0));
    /// ```
    ///
    /// ## Works with any comparable type
    /// ```
    /// # use nalgebra::Point3;
    /// let p1 = Point3::new(10, 20, 30);
    /// let p2 = Point3::new(15, 5, 25);
    ///
    /// let max = p1.sup(&p2);
    /// assert_eq!(max, Point3::new(15, 20, 30));
    /// ```
    ///
    /// # See Also
    ///
    /// * [`inf`](Self::inf) - Compute component-wise minimum
    /// * [`inf_sup`](Self::inf_sup) - Compute both minimum and maximum at once
    #[inline]
    #[must_use]
    pub fn sup(&self, other: &Self) -> OPoint<T, D> {
        self.coords.sup(&other.coords).into()
    }

    /// Computes both the component-wise minimum and maximum of two points simultaneously.
    ///
    /// This method is more efficient than calling [`inf`](Self::inf) and [`sup`](Self::sup)
    /// separately, as it computes both in a single pass.
    ///
    /// # Arguments
    ///
    /// * `other` - The other point to compare with
    ///
    /// # Returns
    ///
    /// A tuple `(min_point, max_point)` where:
    /// - `min_point` contains the minimum of each coordinate pair
    /// - `max_point` contains the maximum of each coordinate pair
    ///
    /// # Examples
    ///
    /// ## Basic usage
    /// ```
    /// # use nalgebra::Point2;
    /// let p1 = Point2::new(1.0, 5.0);
    /// let p2 = Point2::new(3.0, 2.0);
    ///
    /// let (min, max) = p1.inf_sup(&p2);
    /// assert_eq!(min, Point2::new(1.0, 2.0));
    /// assert_eq!(max, Point2::new(3.0, 5.0));
    /// ```
    ///
    /// ## Computing axis-aligned bounding box
    /// ```
    /// # use nalgebra::Point3;
    /// let corner1 = Point3::new(2.0, -1.0, 5.0);
    /// let corner2 = Point3::new(-3.0, 4.0, 1.0);
    ///
    /// let (min_corner, max_corner) = corner1.inf_sup(&corner2);
    /// assert_eq!(min_corner, Point3::new(-3.0, -1.0, 1.0));
    /// assert_eq!(max_corner, Point3::new(2.0, 4.0, 5.0));
    ///
    /// // The bounding box spans from min_corner to max_corner
    /// let size = max_corner - min_corner;
    /// ```
    ///
    /// ## Finding bounds for collision detection
    /// ```
    /// # use nalgebra::Point2;
    /// // Two objects with their positions
    /// let obj1_pos = Point2::new(10.0, 20.0);
    /// let obj2_pos = Point2::new(15.0, 18.0);
    ///
    /// // Find the encompassing bounding region
    /// let (min_bound, max_bound) = obj1_pos.inf_sup(&obj2_pos);
    ///
    /// assert_eq!(min_bound, Point2::new(10.0, 18.0));
    /// assert_eq!(max_bound, Point2::new(15.0, 20.0));
    /// ```
    ///
    /// ## More efficient than separate calls
    /// ```
    /// # use nalgebra::Point3;
    /// let p1 = Point3::new(1.0, 2.0, 3.0);
    /// let p2 = Point3::new(4.0, 0.0, 2.0);
    ///
    /// // This is more efficient:
    /// let (min, max) = p1.inf_sup(&p2);
    ///
    /// // Than this:
    /// let min_slow = p1.inf(&p2);
    /// let max_slow = p1.sup(&p2);
    ///
    /// assert_eq!(min, min_slow);
    /// assert_eq!(max, max_slow);
    /// ```
    ///
    /// # See Also
    ///
    /// * [`inf`](Self::inf) - Compute component-wise minimum only
    /// * [`sup`](Self::sup) - Compute component-wise maximum only
    #[inline]
    #[must_use]
    pub fn inf_sup(&self, other: &Self) -> (OPoint<T, D>, OPoint<T, D>) {
        let (inf, sup) = self.coords.inf_sup(&other.coords);
        (inf.into(), sup.into())
    }
}

/*
 *
 * Display
 *
 */
impl<T: Scalar + fmt::Display, D: DimName> fmt::Display for OPoint<T, D>
where
    DefaultAllocator: Allocator<D>,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{{")?;

        let mut it = self.coords.iter();

        <T as fmt::Display>::fmt(it.next().unwrap(), f)?;

        for comp in it {
            write!(f, ", ")?;
            <T as fmt::Display>::fmt(comp, f)?;
        }

        write!(f, "}}")
    }
}
