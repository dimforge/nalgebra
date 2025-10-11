#[cfg(feature = "arbitrary")]
use quickcheck::{Arbitrary, Gen};

use num::{Bounded, One, Zero};
#[cfg(feature = "rand-no-std")]
use rand::{
    Rng,
    distr::{Distribution, StandardUniform},
};

use crate::base::allocator::Allocator;
use crate::base::dimension::{DimNameAdd, DimNameSum, U1};
use crate::base::{DefaultAllocator, Scalar};
use crate::{
    Const, DimName, OPoint, OVector, Point1, Point2, Point3, Point4, Point5, Point6, Vector1,
    Vector2, Vector3, Vector4, Vector5, Vector6,
};
use simba::scalar::{ClosedDivAssign, SupersetOf};

use crate::geometry::Point;

impl<T: Scalar + Zero, D: DimName> Default for OPoint<T, D>
where
    DefaultAllocator: Allocator<D>,
{
    fn default() -> Self {
        Self::origin()
    }
}

/// # Other construction methods
impl<T: Scalar, D: DimName> OPoint<T, D>
where
    DefaultAllocator: Allocator<D>,
{
    /// Creates a new point at the origin with all coordinates set to zero.
    ///
    /// The origin is the reference point in a coordinate system, typically representing
    /// the center or starting position (0, 0) in 2D or (0, 0, 0) in 3D space.
    ///
    /// This is equivalent to the `Default` implementation for points.
    ///
    /// # Returns
    ///
    /// A point with all coordinates equal to zero.
    ///
    /// # Examples
    ///
    /// ## Creating origin points in different dimensions
    /// ```
    /// # use nalgebra::{Point1, Point2, Point3};
    /// // 1D origin
    /// let origin_1d = Point1::<f32>::origin();
    /// assert_eq!(origin_1d.x, 0.0);
    ///
    /// // 2D origin (0, 0)
    /// let origin_2d = Point2::<f32>::origin();
    /// assert_eq!(origin_2d.x, 0.0);
    /// assert_eq!(origin_2d.y, 0.0);
    ///
    /// // 3D origin (0, 0, 0)
    /// let origin_3d = Point3::<f32>::origin();
    /// assert_eq!(origin_3d.x, 0.0);
    /// assert_eq!(origin_3d.y, 0.0);
    /// assert_eq!(origin_3d.z, 0.0);
    /// ```
    ///
    /// ## Using origin as a reference point
    /// ```
    /// # use nalgebra::Point2;
    /// let origin = Point2::<f32>::origin();
    /// let position = Point2::new(5.0, 3.0);
    ///
    /// // Calculate distance from origin
    /// let distance_from_origin = (position - origin).norm();
    /// assert!((distance_from_origin - 5.831).abs() < 0.01); // sqrt(25 + 9)
    /// ```
    ///
    /// ## Works with integer types too
    /// ```
    /// # use nalgebra::Point3;
    /// let origin = Point3::<i32>::origin();
    /// assert_eq!(origin, Point3::new(0, 0, 0));
    /// ```
    ///
    /// ## Equivalent to Default
    /// ```
    /// # use nalgebra::Point2;
    /// let origin1 = Point2::<f64>::origin();
    /// let origin2 = Point2::<f64>::default();
    /// assert_eq!(origin1, origin2);
    /// ```
    ///
    /// ## Initializing positions in a game or simulation
    /// ```
    /// # use nalgebra::Point2;
    /// // Starting position for a player character
    /// let player_spawn = Point2::<f32>::origin();
    ///
    /// // Create a point relative to origin
    /// let offset = Point2::new(10.0, 5.0);
    /// let new_position = player_spawn + offset.coords;
    /// assert_eq!(new_position, Point2::new(10.0, 5.0));
    /// ```
    ///
    /// # See Also
    ///
    /// * [`new`](crate::Point2::new) - Create a point with specific coordinates
    /// * [`from_slice`](Self::from_slice) - Create a point from a slice of values
    #[inline]
    pub fn origin() -> Self
    where
        T: Zero,
    {
        Self::from(OVector::from_element(T::zero()))
    }

    /// Creates a new point from a slice of coordinates.
    ///
    /// This method constructs a point by copying coordinate values from a slice.
    /// The slice must have at least as many elements as the point's dimension.
    /// If the slice is longer, only the first N elements are used.
    ///
    /// # Arguments
    ///
    /// * `components` - A slice containing the coordinate values
    ///
    /// # Panics
    ///
    /// Panics if the slice has fewer elements than the point's dimension.
    ///
    /// # Examples
    ///
    /// ## Creating points from slices
    /// ```
    /// # use nalgebra::{Point2, Point3};
    /// let data = [1.0, 2.0, 3.0];
    ///
    /// // Create a 2D point from the first 2 elements
    /// let pt = Point2::from_slice(&data[..2]);
    /// assert_eq!(pt, Point2::new(1.0, 2.0));
    ///
    /// // Create a 3D point from all 3 elements
    /// let pt = Point3::from_slice(&data);
    /// assert_eq!(pt, Point3::new(1.0, 2.0, 3.0));
    /// ```
    ///
    /// ## Loading point data from external sources
    /// ```
    /// # use nalgebra::Point3;
    /// // Simulating data from a file or network
    /// let serialized_data: Vec<f64> = vec![10.0, 20.0, 30.0];
    /// let point = Point3::from_slice(&serialized_data);
    /// assert_eq!(point, Point3::new(10.0, 20.0, 30.0));
    /// ```
    ///
    /// ## Parsing coordinate data
    /// ```
    /// # use nalgebra::Point2;
    /// let coords_str = "5.5,10.5";
    /// let coords: Vec<f64> = coords_str.split(',')
    ///     .map(|s| s.parse().unwrap())
    ///     .collect();
    ///
    /// let point = Point2::from_slice(&coords);
    /// assert_eq!(point, Point2::new(5.5, 10.5));
    /// ```
    ///
    /// ## Working with arrays of varying sizes
    /// ```
    /// # use nalgebra::Point2;
    /// // A longer array - only first 2 elements are used
    /// let data = [1.0, 2.0, 3.0, 4.0, 5.0];
    /// let point = Point2::from_slice(&data);
    /// assert_eq!(point, Point2::new(1.0, 2.0));
    /// ```
    ///
    /// ## Converting from buffer data
    /// ```
    /// # use nalgebra::Point3;
    /// // Vertex buffer data: [x, y, z, r, g, b]
    /// let vertex_data = [1.0, 2.0, 3.0, 1.0, 0.0, 0.0];
    ///
    /// // Extract position (first 3 elements)
    /// let position = Point3::from_slice(&vertex_data[0..3]);
    /// assert_eq!(position, Point3::new(1.0, 2.0, 3.0));
    /// ```
    ///
    /// # See Also
    ///
    /// * [`new`](crate::Point2::new) - Create a point with individual coordinates
    /// * [`origin`](Self::origin) - Create a point at the origin
    /// * [`from_homogeneous`](Self::from_homogeneous) - Create from homogeneous coordinates
    #[inline]
    pub fn from_slice(components: &[T]) -> Self {
        Self::from(OVector::from_row_slice(components))
    }

    /// Creates a point from homogeneous coordinates.
    ///
    /// This is the inverse operation of [`to_homogeneous`](crate::OPoint::to_homogeneous).
    /// It converts a vector in homogeneous coordinates (N+1 dimensions) back to a point
    /// in N-dimensional space by dividing all coordinates by the last component.
    ///
    /// Returns `None` if the last component (the homogeneous coordinate) is zero, as this
    /// represents a point at infinity which cannot be converted to a regular point.
    ///
    /// # Arguments
    ///
    /// * `v` - A vector in homogeneous coordinates (one dimension higher than the resulting point)
    ///
    /// # Returns
    ///
    /// - `Some(point)` if the homogeneous coordinate (last component) is non-zero
    /// - `None` if the homogeneous coordinate is zero (point at infinity)
    ///
    /// # How it works
    ///
    /// For a vector `(x, y, z, w)` in homogeneous coordinates:
    /// - If `w ≠ 0`: returns point `(x/w, y/w, z/w)`
    /// - If `w = 0`: returns `None` (represents a point at infinity)
    ///
    /// # Examples
    ///
    /// ## Basic conversion with w = 1
    /// ```
    /// # use nalgebra::{Point3, Vector4};
    /// let homogeneous = Vector4::new(1.0, 2.0, 3.0, 1.0);
    /// let point = Point3::from_homogeneous(homogeneous);
    /// assert_eq!(point, Some(Point3::new(1.0, 2.0, 3.0)));
    /// ```
    ///
    /// ## Automatic scaling when w ≠ 1
    /// ```
    /// # use nalgebra::{Point3, Vector4};
    /// // All coordinates are divided by the last component (2.0)
    /// let homogeneous = Vector4::new(2.0, 4.0, 6.0, 2.0);
    /// let point = Point3::from_homogeneous(homogeneous);
    /// assert_eq!(point, Some(Point3::new(1.0, 2.0, 3.0)));
    /// ```
    ///
    /// ## Handling perspective division
    /// ```
    /// # use nalgebra::{Point3, Vector4};
    /// // Common in 3D graphics after perspective projection
    /// let homogeneous = Vector4::new(10.0, 20.0, 30.0, 5.0);
    /// let point = Point3::from_homogeneous(homogeneous);
    /// assert_eq!(point, Some(Point3::new(2.0, 4.0, 6.0)));
    /// ```
    ///
    /// ## Returns None for points at infinity
    /// ```
    /// # use nalgebra::{Point3, Vector4};
    /// // When w = 0, the point is at infinity
    /// let homogeneous = Vector4::new(1.0, 2.0, 3.0, 0.0);
    /// let point = Point3::from_homogeneous(homogeneous);
    /// assert!(point.is_none());
    /// ```
    ///
    /// ## Works in any dimension
    /// ```
    /// # use nalgebra::{Point2, Vector3};
    /// let homogeneous = Vector3::new(4.0, 8.0, 2.0);
    /// let point = Point2::from_homogeneous(homogeneous);
    /// assert_eq!(point, Some(Point2::new(2.0, 4.0)));
    /// ```
    ///
    /// ## Practical use with transformation matrices
    /// ```
    /// # use nalgebra::{Point3, Vector4, Matrix4};
    /// // Apply a transformation matrix that produces homogeneous coords
    /// let point = Point3::new(1.0, 2.0, 3.0);
    /// let transform = Matrix4::new_scaling(2.0);
    ///
    /// let homogeneous = transform * point.to_homogeneous();
    /// let transformed = Point3::from_homogeneous(homogeneous).unwrap();
    /// assert_eq!(transformed, Point3::new(2.0, 4.0, 6.0));
    /// ```
    ///
    /// # See Also
    ///
    /// * [`to_homogeneous`](crate::OPoint::to_homogeneous) - Convert a point to homogeneous coordinates
    /// * [`coords`](crate::OPoint::coords) - Access the underlying coordinate vector
    #[inline]
    pub fn from_homogeneous(v: OVector<T, DimNameSum<D, U1>>) -> Option<Self>
    where
        T: Scalar + Zero + One + ClosedDivAssign,
        D: DimNameAdd<U1>,
        DefaultAllocator: Allocator<DimNameSum<D, U1>>,
    {
        if !v[D::DIM].is_zero() {
            let coords = v.generic_view((0, 0), (D::name(), Const::<1>)) / v[D::DIM].clone();
            Some(Self::from(coords))
        } else {
            None
        }
    }

    /// Converts the coordinates of this point to another numeric type.
    ///
    /// This method performs type conversion on all coordinates, changing them from
    /// type `T` to type `To`. This is useful when you need to convert between different
    /// numeric types like `f64` to `f32`, or `i32` to `f64`.
    ///
    /// # Type Parameters
    ///
    /// * `To` - The target type for the coordinates
    ///
    /// # Returns
    ///
    /// A new point with coordinates converted to the target type.
    ///
    /// # Examples
    ///
    /// ## Converting from f64 to f32
    /// ```
    /// # use nalgebra::Point2;
    /// let point_f64 = Point2::new(1.0f64, 2.5f64);
    /// let point_f32 = point_f64.cast::<f32>();
    /// assert_eq!(point_f32, Point2::new(1.0f32, 2.5f32));
    /// ```
    ///
    /// ## Converting from integers to floats
    /// ```
    /// # use nalgebra::Point3;
    /// let point_int = Point3::new(10i32, 20i32, 30i32);
    /// let point_float = point_int.cast::<f64>();
    /// assert_eq!(point_float, Point3::new(10.0, 20.0, 30.0));
    /// ```
    ///
    /// ## Converting from floats to integers (truncates)
    /// ```
    /// # use nalgebra::Point2;
    /// let point_float = Point2::new(3.7f64, 8.2f64);
    /// let point_int = point_float.cast::<i32>();
    /// assert_eq!(point_int, Point2::new(3, 8)); // Truncates decimal part
    /// ```
    ///
    /// ## Precision conversion for graphics
    /// ```
    /// # use nalgebra::Point3;
    /// // High precision calculations
    /// let precise_position = Point3::new(1.23456789f64, 2.34567890f64, 3.45678901f64);
    ///
    /// // Convert to GPU-friendly format (f32)
    /// let gpu_position = precise_position.cast::<f32>();
    /// assert_eq!(gpu_position.x, 1.2345679f32);
    /// ```
    ///
    /// ## Converting coordinate systems with different types
    /// ```
    /// # use nalgebra::Point2;
    /// // Screen coordinates (integers)
    /// let screen_pos = Point2::new(800, 600);
    ///
    /// // Convert to normalized coordinates (floats)
    /// let normalized = screen_pos.cast::<f64>();
    /// let normalized = normalized.map(|coord| coord / 1000.0);
    /// assert_eq!(normalized, Point2::new(0.8, 0.6));
    /// ```
    ///
    /// ## Working with mixed type systems
    /// ```
    /// # use nalgebra::Point3;
    /// // Index into a grid (integers)
    /// let grid_index = Point3::new(5, 10, 15);
    ///
    /// // Convert to world position (floats) with scaling
    /// let world_pos = grid_index.cast::<f32>().map(|coord| coord * 0.5);
    /// assert_eq!(world_pos, Point3::new(2.5, 5.0, 7.5));
    /// ```
    ///
    /// # See Also
    ///
    /// * [`map`](crate::OPoint::map) - Transform coordinates with a custom function
    /// * [`from_slice`](Self::from_slice) - Create a point from a slice
    pub fn cast<To: Scalar>(self) -> OPoint<To, D>
    where
        OPoint<To, D>: SupersetOf<Self>,
        DefaultAllocator: Allocator<D>,
    {
        crate::convert(self)
    }
}

/*
 *
 * Traits that build points.
 *
 */
impl<T: Scalar + Bounded, D: DimName> Bounded for OPoint<T, D>
where
    DefaultAllocator: Allocator<D>,
{
    #[inline]
    fn max_value() -> Self {
        Self::from(OVector::max_value())
    }

    #[inline]
    fn min_value() -> Self {
        Self::from(OVector::min_value())
    }
}

#[cfg(feature = "rand-no-std")]
impl<T: Scalar, D: DimName> Distribution<OPoint<T, D>> for StandardUniform
where
    StandardUniform: Distribution<T>,
    DefaultAllocator: Allocator<D>,
{
    /// Generate a `Point` where each coordinate is an independent variate from `[0, 1)`.
    #[inline]
    fn sample<'a, G: Rng + ?Sized>(&self, rng: &mut G) -> OPoint<T, D> {
        OPoint::from(rng.random::<OVector<T, D>>())
    }
}

#[cfg(feature = "arbitrary")]
impl<T: Scalar + Arbitrary + Send, D: DimName> Arbitrary for OPoint<T, D>
where
    <DefaultAllocator as Allocator<D>>::Buffer<T>: Send,
    DefaultAllocator: Allocator<D>,
{
    #[inline]
    fn arbitrary(g: &mut Gen) -> Self {
        Self::from(OVector::arbitrary(g))
    }
}

/*
 *
 * Small points construction from components.
 *
 */
// NOTE: the impl for Point1 is not with the others so that we
// can add a section with the impl block comment.
/// # Construction from individual components
impl<T: Scalar> Point1<T> {
    /// Initializes this point from its components.
    ///
    /// # Example
    ///
    /// ```
    /// # use nalgebra::Point1;
    /// let p = Point1::new(1.0);
    /// assert_eq!(p.x, 1.0);
    /// ```
    #[inline]
    pub const fn new(x: T) -> Self {
        Point {
            coords: Vector1::new(x),
        }
    }
}
macro_rules! componentwise_constructors_impl(
    ($($doc: expr_2021; $Point: ident, $Vector: ident, $($args: ident:$irow: expr_2021),*);* $(;)*) => {$(
        impl<T: Scalar> $Point<T> {
            #[doc = "Initializes this point from its components."]
            #[doc = "# Example\n```"]
            #[doc = $doc]
            #[doc = "```"]
            #[inline]
            pub const fn new($($args: T),*) -> Self {
                Point { coords: $Vector::new($($args),*) }
            }
        }
    )*}
);

componentwise_constructors_impl!(
    "# use nalgebra::Point2;\nlet p = Point2::new(1.0, 2.0);\nassert!(p.x == 1.0 && p.y == 2.0);";
    Point2, Vector2, x:0, y:1;
    "# use nalgebra::Point3;\nlet p = Point3::new(1.0, 2.0, 3.0);\nassert!(p.x == 1.0 && p.y == 2.0 && p.z == 3.0);";
    Point3, Vector3, x:0, y:1, z:2;
    "# use nalgebra::Point4;\nlet p = Point4::new(1.0, 2.0, 3.0, 4.0);\nassert!(p.x == 1.0 && p.y == 2.0 && p.z == 3.0 && p.w == 4.0);";
    Point4, Vector4, x:0, y:1, z:2, w:3;
    "# use nalgebra::Point5;\nlet p = Point5::new(1.0, 2.0, 3.0, 4.0, 5.0);\nassert!(p.x == 1.0 && p.y == 2.0 && p.z == 3.0 && p.w == 4.0 && p.a == 5.0);";
    Point5, Vector5, x:0, y:1, z:2, w:3, a:4;
    "# use nalgebra::Point6;\nlet p = Point6::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);\nassert!(p.x == 1.0 && p.y == 2.0 && p.z == 3.0 && p.w == 4.0 && p.a == 5.0 && p.b == 6.0);";
    Point6, Vector6, x:0, y:1, z:2, w:3, a:4, b:5;
);
