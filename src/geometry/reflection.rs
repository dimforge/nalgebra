use crate::base::constraint::{AreMultipliable, DimEq, SameNumberOfRows, ShapeConstraint};
use crate::base::{Const, Matrix, Unit, Vector};
use crate::dimension::{Dim, U1};
use crate::storage::{Storage, StorageMut};
use simba::scalar::ComplexField;

use crate::geometry::Point;

/// A reflection transformation with respect to a hyperplane.
///
/// A reflection (also known as a mirror transformation) flips points, vectors, or matrices
/// across a hyperplane. In 2D, this hyperplane is a line; in 3D, it's a plane; in higher
/// dimensions, it's a hyperplane. The reflection is defined by:
/// - An axis vector (perpendicular/normal to the hyperplane)
/// - A bias (the signed distance of the hyperplane from the origin along the axis)
///
/// # Reflection Mathematics
///
/// Given a point `p`, axis (unit normal) `n`, and bias `b`, the reflected point `p'` is:
/// ```text
/// p' = p - 2 * (n · p - b) * n
/// ```
///
/// where `n · p` is the dot product. This formula:
/// 1. Projects the point onto the axis: `n · p`
/// 2. Computes the signed distance from the hyperplane: `n · p - b`
/// 3. Reflects by moving twice this distance in the opposite direction
///
/// # Common Use Cases
///
/// - **Computer Graphics**: Mirror reflections, water reflections, symmetry operations
/// - **Physics Simulations**: Collision response with walls/surfaces
/// - **Signal Processing**: Boundary conditions in numerical methods
/// - **Linear Algebra**: Householder reflections for QR decomposition
/// - **Normal Mapping**: Reflecting light/view vectors across surface normals
///
/// # Type Parameters
///
/// * `T` - The scalar type (typically `f32` or `f64`)
/// * `D` - The dimension of the space
/// * `S` - The storage type for the axis vector
///
/// # Examples
///
/// See individual method documentation for detailed examples.
pub struct Reflection<T, D, S> {
    axis: Vector<T, D, S>,
    bias: T,
}

impl<T: ComplexField, S: Storage<T, Const<D>>, const D: usize> Reflection<T, Const<D>, S> {
    /// Creates a new reflection with respect to a hyperplane defined by a normal axis and a point.
    ///
    /// This constructs a reflection across a hyperplane (line in 2D, plane in 3D) that:
    /// - Is perpendicular (orthogonal) to the given `axis` (unit normal vector)
    /// - Passes through the given point `pt`
    ///
    /// This is often more intuitive than specifying a bias value directly, since you can
    /// visualize the hyperplane as "the surface passing through this point, perpendicular to
    /// this direction."
    ///
    /// # Mathematical Note
    ///
    /// The bias is computed as `b = axis · pt`, which represents the signed distance of the
    /// hyperplane from the origin along the axis direction. This ensures the hyperplane contains
    /// the point `pt`.
    ///
    /// # Parameters
    ///
    /// * `axis` - A unit vector perpendicular to the reflection hyperplane (the normal)
    /// * `pt` - A point that lies on the reflection hyperplane
    ///
    /// # Examples
    ///
    /// ## 2D reflection across a line through a point
    /// ```
    /// # use nalgebra::{Reflection, Vector2, Point2, Unit};
    /// // Create a reflection across a vertical line (perpendicular to X axis)
    /// // that passes through the point (3, 0)
    /// let axis = Unit::new_normalize(Vector2::new(1.0, 0.0)); // Normal pointing right
    /// let point = Point2::new(3.0, 0.0);
    /// let reflection = Reflection::new_containing_point(axis, &point);
    ///
    /// // The reflection axis is the X axis
    /// assert_eq!(reflection.axis()[0], 1.0);
    /// assert_eq!(reflection.axis()[1], 0.0);
    /// // The bias should be 3.0 (distance along X axis to the line)
    /// assert_eq!(reflection.bias(), 3.0);
    /// ```
    ///
    /// ## 3D reflection across a ground plane
    /// ```
    /// # use nalgebra::{Reflection, Vector3, Point3, Unit};
    /// // Create a reflection across a horizontal plane at height y = 2.0
    /// // (The plane perpendicular to Y axis passing through (0, 2, 0))
    /// let axis = Unit::new_normalize(Vector3::new(0.0, 1.0, 0.0)); // Normal pointing up
    /// let point = Point3::new(0.0, 2.0, 0.0); // Point on the plane
    /// let reflection = Reflection::new_containing_point(axis, &point);
    ///
    /// assert_eq!(reflection.axis()[0], 0.0);
    /// assert_eq!(reflection.axis()[1], 1.0);
    /// assert_eq!(reflection.axis()[2], 0.0);
    /// assert_eq!(reflection.bias(), 2.0);
    /// ```
    ///
    /// ## Game physics: Mirror reflection off a wall
    /// ```
    /// # use nalgebra::{Reflection, Vector2, Point2, Unit};
    /// # use std::f32;
    /// // A diagonal wall at 45 degrees passing through (5, 5)
    /// let wall_normal = Unit::new_normalize(Vector2::new(1.0, 1.0)); // 45° angle
    /// let wall_point = Point2::new(5.0, 5.0);
    /// let wall_reflection = Reflection::new_containing_point(wall_normal, &wall_point);
    ///
    /// // This creates a reflection across the diagonal line y = x passing through (5,5)
    /// // We can use this to bounce projectiles off the wall
    /// let bias = wall_reflection.bias();
    /// assert!((bias - 5.0 * f32::sqrt(2.0)).abs() < 1e-5);
    /// ```
    ///
    /// # See Also
    /// - [`new`](Self::new) - Create a reflection from an axis and bias directly
    /// - [`axis`](Self::axis) - Get the reflection axis (normal vector)
    /// - [`bias`](Self::bias) - Get the bias value
    pub fn new_containing_point(axis: Unit<Vector<T, Const<D>, S>>, pt: &Point<T, D>) -> Self {
        let bias = axis.dotc(&pt.coords);
        Self::new(axis, bias)
    }
}

impl<T: ComplexField, D: Dim, S: Storage<T, D>> Reflection<T, D, S> {
    /// Creates a new reflection with respect to a hyperplane defined by a normal axis and bias.
    ///
    /// This is the fundamental constructor for a reflection. It creates a reflection across
    /// a hyperplane (line in 2D, plane in 3D) that is perpendicular to `axis` and positioned
    /// at distance `bias` from the origin along the axis direction.
    ///
    /// # Understanding Bias
    ///
    /// The bias represents the signed distance of the hyperplane from the origin, measured
    /// along the axis direction. Think of it as "how far to slide the hyperplane along its normal."
    ///
    /// - **bias = 0**: The hyperplane passes through the origin
    /// - **bias > 0**: The hyperplane is shifted in the positive axis direction
    /// - **bias < 0**: The hyperplane is shifted in the negative axis direction
    ///
    /// For example, in 3D with axis = (0, 1, 0) [pointing up]:
    /// - bias = 0 → ground plane at y = 0
    /// - bias = 5 → plane at y = 5
    /// - bias = -3 → plane at y = -3
    ///
    /// # When to Use This vs. new_containing_point
    ///
    /// - Use `new()` when you know the mathematical bias value
    /// - Use [`new_containing_point()`](Self::new_containing_point) when you have a point on the hyperplane (more intuitive)
    ///
    /// # Parameters
    ///
    /// * `axis` - A unit vector perpendicular to the reflection hyperplane (the normal)
    /// * `bias` - The signed distance of the hyperplane from the origin along the axis
    ///
    /// # Examples
    ///
    /// ## 2D reflection across a vertical line
    /// ```
    /// # use nalgebra::{Reflection, Vector2, Unit};
    /// // Reflect across a vertical line at x = 0 (the Y axis)
    /// let axis = Unit::new_normalize(Vector2::new(1.0, 0.0)); // Normal pointing right
    /// let reflection = Reflection::new(axis, 0.0); // bias = 0, line through origin
    ///
    /// assert_eq!(reflection.axis()[0], 1.0);
    /// assert_eq!(reflection.bias(), 0.0);
    /// ```
    ///
    /// ## 3D reflection across ground plane
    /// ```
    /// # use nalgebra::{Reflection, Vector3, Unit};
    /// // Reflect across the XZ plane at y = 0 (ground plane)
    /// let axis = Unit::new_normalize(Vector3::new(0.0, 1.0, 0.0)); // Normal pointing up
    /// let reflection = Reflection::new(axis, 0.0);
    ///
    /// // This creates a reflection that flips points across y = 0
    /// assert_eq!(reflection.axis()[1], 1.0);
    /// assert_eq!(reflection.bias(), 0.0);
    /// ```
    ///
    /// ## 3D reflection across an elevated plane
    /// ```
    /// # use nalgebra::{Reflection, Vector3, Unit};
    /// // Reflect across a horizontal plane at y = 10
    /// let axis = Unit::new_normalize(Vector3::new(0.0, 1.0, 0.0)); // Normal pointing up
    /// let reflection = Reflection::new(axis, 10.0); // Plane shifted up by 10 units
    ///
    /// // This creates a reflection across the plane y = 10
    /// assert_eq!(reflection.bias(), 10.0);
    /// ```
    ///
    /// ## Game physics: Wall collision response
    /// ```
    /// # use nalgebra::{Reflection, Vector2, Unit};
    /// // A wall perpendicular to direction (0.6, 0.8) at distance 15 from origin
    /// let wall_normal = Unit::new_normalize(Vector2::new(0.6, 0.8));
    /// let wall_distance = 15.0_f32;
    /// let wall_reflection = Reflection::new(wall_normal, wall_distance);
    ///
    /// // This reflection can be used to bounce projectiles off the wall
    /// assert!((wall_reflection.bias() - 15.0_f32).abs() < 1e-6);
    /// ```
    ///
    /// ## Computer graphics: Mirror plane
    /// ```
    /// # use nalgebra::{Reflection, Vector3, Unit};
    /// // A mirror plane perpendicular to the camera's view direction
    /// let view_direction = Unit::new_normalize(Vector3::new(0.0, 0.0, -1.0));
    /// let mirror_distance = 5.0; // 5 units in front of camera
    /// let mirror = Reflection::new(view_direction, mirror_distance);
    ///
    /// // Use this to render mirror reflections in the scene
    /// assert_eq!(mirror.bias(), 5.0);
    /// ```
    ///
    /// # See Also
    /// - [`new_containing_point`](Self::new_containing_point) - Create from axis and a point on the hyperplane (often more intuitive)
    /// - [`axis`](Self::axis) - Get the reflection axis (normal vector)
    /// - [`bias`](Self::bias) - Get the bias value
    pub fn new(axis: Unit<Vector<T, D, S>>, bias: T) -> Self {
        Self {
            axis: axis.into_inner(),
            bias,
        }
    }

    /// Returns the reflection axis (the normal vector to the reflection hyperplane).
    ///
    /// The axis is a vector perpendicular (orthogonal) to the reflection hyperplane. It defines
    /// the direction in which points are "flipped" during the reflection. While this vector
    /// is typically a unit vector (length 1), this method returns it without guaranteeing
    /// normalization.
    ///
    /// # What is the Axis?
    ///
    /// Think of the axis as the "direction of the mirror's surface normal":
    /// - In 2D: If reflecting across a line, the axis points perpendicular to that line
    /// - In 3D: If reflecting across a plane, the axis points perpendicular to that plane
    /// - In general: The axis is the normal vector to the reflection hyperplane
    ///
    /// # Returns
    ///
    /// A reference to the axis vector (normal to the hyperplane)
    ///
    /// # Examples
    ///
    /// ## 2D: Inspecting reflection axis
    /// ```
    /// # use nalgebra::{Reflection, Vector2, Unit};
    /// // Create a reflection across a vertical line (Y axis)
    /// let axis = Unit::new_normalize(Vector2::new(1.0, 0.0));
    /// let reflection = Reflection::new(axis, 0.0);
    ///
    /// // Get the axis back
    /// let retrieved_axis = reflection.axis();
    /// assert_eq!(retrieved_axis[0], 1.0);
    /// assert_eq!(retrieved_axis[1], 0.0);
    /// ```
    ///
    /// ## 3D: Ground plane normal
    /// ```
    /// # use nalgebra::{Reflection, Vector3, Unit};
    /// // Reflection across ground plane (XZ plane)
    /// let up = Unit::new_normalize(Vector3::new(0.0, 1.0, 0.0));
    /// let ground_reflection = Reflection::new(up, 0.0);
    ///
    /// // The axis points upward (perpendicular to ground)
    /// let normal = ground_reflection.axis();
    /// assert_eq!(normal[0], 0.0);
    /// assert_eq!(normal[1], 1.0);
    /// assert_eq!(normal[2], 0.0);
    /// ```
    ///
    /// ## Game development: Wall normal for collision
    /// ```
    /// # use nalgebra::{Reflection, Vector2, Unit};
    /// // A wall with a 45-degree angle
    /// let wall_normal = Unit::new_normalize(Vector2::new(0.7071068_f32, 0.7071068_f32));
    /// let wall = Reflection::new(wall_normal, 5.0);
    ///
    /// // Access the normal for collision detection calculations
    /// let normal = wall.axis();
    /// let dot_product = normal[0] * 1.0_f32 + normal[1] * 0.0_f32; // Example calculation
    /// assert!((dot_product - 0.7071068_f32).abs() < 1e-5);
    /// ```
    ///
    /// ## Robotics: Surface normal for grasping
    /// ```
    /// # use nalgebra::{Reflection, Vector3, Unit};
    /// // Object surface described by its normal
    /// let surface_normal = Unit::new_normalize(Vector3::new(0.0, 0.0, 1.0));
    /// let surface = Reflection::new(surface_normal, 0.0);
    ///
    /// // Get normal for computing gripper approach direction
    /// let normal = surface.axis();
    /// // In robotics, the gripper often approaches along the surface normal
    /// assert_eq!(normal[2], 1.0);
    /// ```
    ///
    /// # See Also
    /// - [`bias`](Self::bias) - Get the bias (position of the hyperplane along the axis)
    /// - [`new`](Self::new) - Construct a reflection from axis and bias
    #[must_use]
    pub const fn axis(&self) -> &Vector<T, D, S> {
        &self.axis
    }

    /// Returns the reflection bias (the signed distance of the hyperplane from the origin).
    ///
    /// The bias represents where the reflection hyperplane is positioned along its normal axis.
    /// It's the signed distance from the origin to the hyperplane, measured along the axis direction.
    ///
    /// # Understanding Bias
    ///
    /// - **bias = 0**: The hyperplane passes through the origin
    /// - **bias > 0**: The hyperplane is shifted in the positive axis direction
    /// - **bias < 0**: The hyperplane is shifted in the negative axis direction
    ///
    /// For example, with a vertical axis (0, 1, 0) pointing up in 3D:
    /// - bias = 0 means the plane is at ground level (y = 0)
    /// - bias = 3 means the plane is 3 units above ground (y = 3)
    /// - bias = -2 means the plane is 2 units below ground (y = -2)
    ///
    /// # Mathematical Relation
    ///
    /// The bias `b` satisfies the equation: `axis · point = b` for any point on the hyperplane,
    /// where `·` denotes the dot product. This is the standard plane equation in normal form.
    ///
    /// # Returns
    ///
    /// The bias value (a scalar)
    ///
    /// # Examples
    ///
    /// ## 2D: Getting line position
    /// ```
    /// # use nalgebra::{Reflection, Vector2, Unit};
    /// // Vertical line at x = 5
    /// let axis = Unit::new_normalize(Vector2::new(1.0, 0.0));
    /// let reflection = Reflection::new(axis, 5.0);
    ///
    /// assert_eq!(reflection.bias(), 5.0);
    /// // This tells us the line is 5 units from origin along the X axis
    /// ```
    ///
    /// ## 3D: Checking plane height
    /// ```
    /// # use nalgebra::{Reflection, Vector3, Point3, Unit};
    /// // Horizontal plane at y = 10
    /// let axis = Unit::new_normalize(Vector3::new(0.0, 1.0, 0.0));
    /// let point_on_plane = Point3::new(0.0, 10.0, 0.0);
    /// let reflection = Reflection::new_containing_point(axis, &point_on_plane);
    ///
    /// // The bias tells us the plane's height
    /// assert_eq!(reflection.bias(), 10.0);
    /// ```
    ///
    /// ## Game physics: Distance to wall
    /// ```
    /// # use nalgebra::{Reflection, Vector2, Unit};
    /// // Wall at distance 20 from origin
    /// let wall_normal = Unit::new_normalize(Vector2::new(1.0, 0.0));
    /// let wall = Reflection::new(wall_normal, 20.0);
    ///
    /// // Use bias to check how far the wall is
    /// let distance_to_wall = wall.bias();
    /// assert_eq!(distance_to_wall, 20.0);
    ///
    /// // Can be used for proximity checks in game logic
    /// if distance_to_wall > 15.0 {
    ///     // Wall is far enough
    /// }
    /// ```
    ///
    /// ## Graphics: Mirror plane depth
    /// ```
    /// # use nalgebra::{Reflection, Vector3, Unit};
    /// // Mirror perpendicular to view direction, 5 units away
    /// let view_dir = Unit::new_normalize(Vector3::new(0.0, 0.0, -1.0));
    /// let mirror = Reflection::new(view_dir, 5.0);
    ///
    /// // Get the depth for rendering calculations
    /// let mirror_depth = mirror.bias();
    /// assert_eq!(mirror_depth, 5.0);
    /// // Use this to cull objects behind the mirror
    /// ```
    ///
    /// ## Robotics: Workspace boundary
    /// ```
    /// # use nalgebra::{Reflection, Vector3, Unit};
    /// // A plane defining the robot's workspace boundary
    /// let boundary_normal = Unit::new_normalize(Vector3::new(0.0, 0.0, 1.0));
    /// let boundary = Reflection::new(boundary_normal, 50.0);
    ///
    /// // Check if we're within the workspace limit
    /// let max_reach = boundary.bias();
    /// assert_eq!(max_reach, 50.0);
    /// ```
    ///
    /// # See Also
    /// - [`axis`](Self::axis) - Get the normal vector to the hyperplane
    /// - [`new`](Self::new) - Construct a reflection from axis and bias
    /// - [`new_containing_point`](Self::new_containing_point) - Construct from axis and a point (computes bias automatically)
    #[must_use]
    pub fn bias(&self) -> T {
        self.bias.clone()
    }

    // TODO: naming convention: reflect_to, reflect_assign ?
    /// Applies the reflection to the columns of a matrix in-place.
    ///
    /// This method reflects each column of the matrix `rhs` across the hyperplane defined by
    /// this reflection. Each column is treated as a separate vector to be reflected. This is
    /// particularly useful for:
    /// - Reflecting multiple points or vectors at once
    /// - Applying reflections in numerical linear algebra algorithms (e.g., QR decomposition)
    /// - Batch processing of geometric transformations
    ///
    /// The reflection is applied in-place, modifying the matrix directly for efficiency.
    ///
    /// # Mathematical Operation
    ///
    /// For each column vector `v`, the reflection computes:
    /// ```text
    /// v' = v - 2 * (axis · v - bias) * axis
    /// ```
    ///
    /// # Parameters
    ///
    /// * `rhs` - A mutable reference to the matrix whose columns will be reflected
    ///
    /// # Type Constraints
    ///
    /// The matrix must have the same number of rows as the dimension of this reflection's axis.
    ///
    /// # Examples
    ///
    /// ## 2D: Reflecting multiple points across a line
    /// ```
    /// # use nalgebra::{Reflection, Matrix2, Vector2, Unit};
    /// // Reflect across the Y axis (vertical line at x = 0)
    /// let axis = Unit::new_normalize(Vector2::new(1.0, 0.0));
    /// let reflection = Reflection::new(axis, 0.0);
    ///
    /// // Matrix with two column vectors (two 2D points)
    /// let mut points = Matrix2::new(
    ///     3.0, -2.0,  // First column: point (3, 5)
    ///     5.0,  7.0   // Second column: point (-2, 7)
    /// );
    ///
    /// // Reflect both points at once
    /// reflection.reflect(&mut points);
    ///
    /// // First point (3, 5) becomes (-3, 5)
    /// assert_eq!(points[(0, 0)], -3.0);
    /// assert_eq!(points[(1, 0)], 5.0);
    /// // Second point (-2, 7) becomes (2, 7)
    /// assert_eq!(points[(0, 1)], 2.0);
    /// assert_eq!(points[(1, 1)], 7.0);
    /// ```
    ///
    /// ## 3D: Batch reflection for graphics
    /// ```
    /// # use nalgebra::{Reflection, Matrix3, Vector3, Unit};
    /// // Ground plane reflection (y = 0)
    /// let axis = Unit::new_normalize(Vector3::new(0.0, 1.0, 0.0));
    /// let ground = Reflection::new(axis, 0.0);
    ///
    /// // Three 3D points stored as columns
    /// let mut vertices = Matrix3::new(
    ///     1.0, 2.0, 3.0,  // x coordinates
    ///     4.0, 5.0, 6.0,  // y coordinates (heights)
    ///     7.0, 8.0, 9.0   // z coordinates
    /// );
    ///
    /// // Mirror all vertices across ground plane
    /// ground.reflect(&mut vertices);
    ///
    /// // Y coordinates are negated (reflected)
    /// assert_eq!(vertices[(1, 0)], -4.0);
    /// assert_eq!(vertices[(1, 1)], -5.0);
    /// assert_eq!(vertices[(1, 2)], -6.0);
    /// // X and Z coordinates unchanged
    /// assert_eq!(vertices[(0, 0)], 1.0);
    /// assert_eq!(vertices[(2, 0)], 7.0);
    /// ```
    ///
    /// ## Physics: Bouncing particles off a wall
    /// ```
    /// # use nalgebra::{Reflection, Matrix2x3, Vector2, Unit};
    /// // Wall at x = 10
    /// let wall_normal = Unit::new_normalize(Vector2::new(1.0, 0.0));
    /// let wall = Reflection::new(wall_normal, 10.0);
    ///
    /// // Three particle positions
    /// let mut particles = Matrix2x3::new(
    ///     12.0, 15.0, 11.0,  // x positions (all past the wall)
    ///     3.0,  5.0,  2.0    // y positions
    /// );
    ///
    /// // Reflect particles that went past the wall
    /// wall.reflect(&mut particles);
    ///
    /// // Particles are mirrored back across x = 10
    /// assert_eq!(particles[(0, 0)], 8.0);   // 12 -> 8 (2 units past wall, reflected 2 units back)
    /// assert_eq!(particles[(0, 1)], 5.0);   // 15 -> 5 (5 units past, reflected 5 back)
    /// assert_eq!(particles[(0, 2)], 9.0);   // 11 -> 9 (1 unit past, reflected 1 back)
    /// ```
    ///
    /// ## Linear algebra: Householder reflection (QR decomposition)
    /// ```
    /// # use nalgebra::{Reflection, Matrix3, Vector3, Unit};
    /// // Householder reflection for QR decomposition
    /// let axis = Unit::new_normalize(Vector3::new(1.0, 1.0, 1.0));
    /// let householder = Reflection::new(axis, 0.0);
    ///
    /// let mut matrix = Matrix3::new(
    ///     1.0, 4.0, 7.0,
    ///     2.0, 5.0, 8.0,
    ///     3.0, 6.0, 9.0
    /// );
    ///
    /// // Apply Householder reflection to columns
    /// householder.reflect(&mut matrix);
    ///
    /// // Matrix is now transformed by the reflection
    /// // (Used in QR decomposition algorithms)
    /// ```
    ///
    /// # See Also
    /// - [`reflect_with_sign`](Self::reflect_with_sign) - Variant with a sign parameter for specialized algorithms
    /// - [`reflect_rows`](Self::reflect_rows) - Reflects rows instead of columns
    /// - [`new`](Self::new) - Create a reflection
    pub fn reflect<R2: Dim, C2: Dim, S2>(&self, rhs: &mut Matrix<T, R2, C2, S2>)
    where
        S2: StorageMut<T, R2, C2>,
        ShapeConstraint: SameNumberOfRows<R2, D>,
    {
        for i in 0..rhs.ncols() {
            // NOTE: we borrow the column twice here. First it is borrowed immutably for the
            // dot product, and then mutably. Somehow, this allows significantly
            // better optimizations of the dot product from the compiler.
            let m_two: T = crate::convert(-2.0f64);
            let factor = (self.axis.dotc(&rhs.column(i)) - self.bias.clone()) * m_two;
            rhs.column_mut(i).axpy(factor, &self.axis, T::one());
        }
    }

    // TODO: naming convention: reflect_to, reflect_assign ?
    /// Applies a signed reflection to the columns of a matrix in-place.
    ///
    /// This is a specialized variant of [`reflect`](Self::reflect) that includes an additional
    /// `sign` parameter. The sign acts as a scaling factor that can flip or modify the reflection.
    /// This is primarily used in advanced numerical linear algebra algorithms like QR decomposition
    /// and Householder transformations, where the sign of the reflection needs to be controlled.
    ///
    /// # What Does the Sign Parameter Do?
    ///
    /// The sign parameter modifies the reflection formula:
    /// - **sign = 1**: Standard reflection (same as [`reflect`](Self::reflect))
    /// - **sign = -1**: Inverted reflection (reflects to opposite side)
    /// - **Other values**: Scaled reflection (used in specialized algorithms)
    ///
    /// This is useful in numerical algorithms where you need to control the direction or
    /// strength of the reflection for numerical stability or algorithmic correctness.
    ///
    /// # Mathematical Operation
    ///
    /// For each column vector `v`, the signed reflection computes:
    /// ```text
    /// v' = sign * v - 2 * sign * (axis · v - bias) * axis
    /// ```
    ///
    /// # Parameters
    ///
    /// * `rhs` - A mutable reference to the matrix whose columns will be reflected
    /// * `sign` - A scalar that controls the direction/scaling of the reflection
    ///
    /// # When to Use This
    ///
    /// Most users should use [`reflect`](Self::reflect) instead. Use `reflect_with_sign` only when:
    /// - Implementing QR decomposition or similar algorithms
    /// - You need to control the orientation of the reflection
    /// - Working with complex numerical stability requirements
    ///
    /// # Examples
    ///
    /// ## Standard reflection with sign = 1
    /// ```
    /// # use nalgebra::{Reflection, Matrix2, Vector2, Unit};
    /// let axis = Unit::new_normalize(Vector2::new(1.0, 0.0));
    /// let reflection = Reflection::new(axis, 0.0);
    ///
    /// let mut points = Matrix2::new(
    ///     3.0, -2.0,
    ///     5.0,  7.0
    /// );
    ///
    /// // sign = 1.0 gives normal reflection
    /// reflection.reflect_with_sign(&mut points, 1.0);
    ///
    /// assert_eq!(points[(0, 0)], -3.0);
    /// assert_eq!(points[(0, 1)], 2.0);
    /// ```
    ///
    /// ## Inverted reflection with sign = -1
    /// ```
    /// # use nalgebra::{Reflection, Matrix2, Vector2, Unit};
    /// let axis = Unit::new_normalize(Vector2::new(1.0, 0.0));
    /// let reflection = Reflection::new(axis, 0.0);
    ///
    /// let mut points = Matrix2::new(
    ///     3.0, -2.0,
    ///     5.0,  7.0
    /// );
    ///
    /// // sign = -1.0 inverts the reflection
    /// reflection.reflect_with_sign(&mut points, -1.0);
    ///
    /// // Different result than standard reflection
    /// assert_eq!(points[(0, 0)], 3.0);
    /// assert_eq!(points[(0, 1)], -2.0);
    /// ```
    ///
    /// ## Householder reflection in QR decomposition
    /// ```
    /// # use nalgebra::{Reflection, Matrix3, Vector3, Unit};
    /// // Householder reflection with sign control for numerical stability
    /// let axis = Unit::new_normalize(Vector3::new(1.0, 1.0, 1.0));
    /// let householder = Reflection::new(axis, 0.0);
    ///
    /// let mut matrix = Matrix3::new(
    ///     1.0, 4.0, 7.0,
    ///     2.0, 5.0, 8.0,
    ///     3.0, 6.0, 9.0
    /// );
    ///
    /// // Use sign to control the reflection direction for algorithm stability
    /// let sign = 1.0;
    /// householder.reflect_with_sign(&mut matrix, sign);
    ///
    /// // Matrix is transformed with controlled reflection
    /// ```
    ///
    /// ## Custom scaling for animation
    /// ```
    /// # use nalgebra::{Reflection, Matrix2, Vector2, Unit};
    /// let axis = Unit::new_normalize(Vector2::new(0.0, 1.0));
    /// let reflection = Reflection::new(axis, 0.0);
    ///
    /// let mut vertices = Matrix2::new(
    ///     1.0, 2.0,
    ///     3.0, 4.0
    /// );
    ///
    /// // Use sign = 0.5 for a partial reflection effect
    /// reflection.reflect_with_sign(&mut vertices, 0.5);
    ///
    /// // Creates a dampened reflection effect
    /// ```
    ///
    /// # See Also
    /// - [`reflect`](Self::reflect) - Standard reflection without sign parameter (use this for most cases)
    /// - [`reflect_rows_with_sign`](Self::reflect_rows_with_sign) - Signed reflection for rows instead of columns
    /// - [`new`](Self::new) - Create a reflection
    pub fn reflect_with_sign<R2: Dim, C2: Dim, S2>(&self, rhs: &mut Matrix<T, R2, C2, S2>, sign: T)
    where
        S2: StorageMut<T, R2, C2>,
        ShapeConstraint: SameNumberOfRows<R2, D>,
    {
        for i in 0..rhs.ncols() {
            // NOTE: we borrow the column twice here. First it is borrowed immutably for the
            // dot product, and then mutably. Somehow, this allows significantly
            // better optimizations of the dot product from the compiler.
            let m_two = sign.clone().scale(crate::convert(-2.0f64));
            let factor = (self.axis.dotc(&rhs.column(i)) - self.bias.clone()) * m_two;
            rhs.column_mut(i).axpy(factor, &self.axis, sign.clone());
        }
    }

    /// Applies the reflection to the rows of a matrix in-place.
    ///
    /// This method reflects the rows of matrix `lhs` across the hyperplane defined by this
    /// reflection. Unlike [`reflect`](Self::reflect) which operates on columns, this method
    /// operates on rows. This is commonly used in matrix decompositions and when you need to
    /// transform matrices from the left side (left-multiplication).
    ///
    /// # Understanding Row vs Column Reflection
    ///
    /// - **Column reflection** ([`reflect`](Self::reflect)): Transforms vectors stored as columns (right-multiplication)
    /// - **Row reflection** (this method): Transforms vectors stored as rows (left-multiplication)
    ///
    /// In matrix algebra terms:
    /// - Column reflection: `M' = M * R` (reflection applied to columns)
    /// - Row reflection: `M' = R * M` (reflection applied to rows)
    ///
    /// # Work Vector
    ///
    /// This method requires a temporary "work" vector for intermediate calculations. This
    /// design allows for efficient memory reuse in algorithms that apply many reflections.
    /// The work vector must have the same number of rows as the matrix `lhs`.
    ///
    /// # Parameters
    ///
    /// * `lhs` - A mutable reference to the matrix whose rows will be reflected
    /// * `work` - A mutable work vector for temporary storage during computation
    ///
    /// # Type Constraints
    ///
    /// The matrix must have the same number of columns as the dimension of this reflection's axis.
    ///
    /// # Examples
    ///
    /// ## 2D: Reflecting matrix rows
    /// ```
    /// # use nalgebra::{Reflection, Matrix2, Vector2, Unit};
    /// // Reflect across Y axis (vertical line)
    /// let axis = Unit::new_normalize(Vector2::new(1.0, 0.0));
    /// let reflection = Reflection::new(axis, 0.0);
    ///
    /// let mut matrix = Matrix2::new(
    ///     3.0, 5.0,
    ///     -2.0, 7.0
    /// );
    ///
    /// let mut work = Vector2::zeros();
    ///
    /// // Reflect the rows
    /// reflection.reflect_rows(&mut matrix, &mut work);
    ///
    /// // First row (3, 5) becomes (-3, 5)
    /// assert_eq!(matrix[(0, 0)], -3.0);
    /// assert_eq!(matrix[(0, 1)], 5.0);
    /// // Second row (-2, 7) becomes (2, 7)
    /// assert_eq!(matrix[(1, 0)], 2.0);
    /// assert_eq!(matrix[(1, 1)], 7.0);
    /// ```
    ///
    /// ## 3D: Matrix transformation in QR decomposition
    /// ```
    /// # use nalgebra::{Reflection, Matrix3, Vector3, Unit};
    /// // Householder reflection for QR algorithm
    /// let axis = Unit::new_normalize(Vector3::new(1.0, 1.0, 1.0));
    /// let householder = Reflection::new(axis, 0.0);
    ///
    /// let mut matrix = Matrix3::new(
    ///     1.0, 2.0, 3.0,
    ///     4.0, 5.0, 6.0,
    ///     7.0, 8.0, 9.0
    /// );
    ///
    /// let mut work = Vector3::zeros();
    ///
    /// // Apply Householder reflection to rows (left multiplication)
    /// householder.reflect_rows(&mut matrix, &mut work);
    ///
    /// // Matrix rows are now reflected
    /// // This is a key step in QR decomposition algorithms
    /// ```
    ///
    /// ## Linear algebra: Batch transformation
    /// ```
    /// # use nalgebra::{Reflection, Matrix3x2, Vector3, Vector2, Unit};
    /// // Reflection in 2D space
    /// let axis = Unit::new_normalize(Vector2::new(0.7071068_f32, 0.7071068_f32));
    /// let reflection = Reflection::new(axis, 0.0);
    ///
    /// // Matrix representing multiple 2D data points (3 points, 2 coordinates each)
    /// let mut data = Matrix3x2::new(
    ///     1.0, 0.0,
    ///     2.0, 1.0,
    ///     3.0, 2.0
    /// );
    ///
    /// let mut work = Vector3::zeros();
    ///
    /// // Reflect all rows (all data points) at once
    /// reflection.reflect_rows(&mut data, &mut work);
    ///
    /// // All three points are now reflected across the diagonal line
    /// ```
    ///
    /// ## Numerical methods: Stability in decompositions
    /// ```
    /// # use nalgebra::{Reflection, Matrix2, Vector2, Unit};
    /// // Reflection used in numerical algorithms
    /// let axis = Unit::new_normalize(Vector2::new(1.0, 0.0));
    /// let reflection = Reflection::new(axis, 0.0);
    ///
    /// let mut transform_matrix = Matrix2::identity();
    /// let mut work = Vector2::zeros();
    ///
    /// // Build up a transformation by reflecting the identity matrix
    /// reflection.reflect_rows(&mut transform_matrix, &mut work);
    ///
    /// // The result is the reflection matrix itself
    /// assert_eq!(transform_matrix[(0, 0)], -1.0);
    /// assert_eq!(transform_matrix[(1, 1)], 1.0);
    /// ```
    ///
    /// # See Also
    /// - [`reflect`](Self::reflect) - Reflects columns instead of rows (more commonly used)
    /// - [`reflect_rows_with_sign`](Self::reflect_rows_with_sign) - Row reflection with sign parameter
    /// - [`new`](Self::new) - Create a reflection
    pub fn reflect_rows<R2: Dim, C2: Dim, S2, S3>(
        &self,
        lhs: &mut Matrix<T, R2, C2, S2>,
        work: &mut Vector<T, R2, S3>,
    ) where
        S2: StorageMut<T, R2, C2>,
        S3: StorageMut<T, R2>,
        ShapeConstraint: DimEq<C2, D> + AreMultipliable<R2, C2, D, U1>,
    {
        lhs.mul_to(&self.axis, work);

        if !self.bias.is_zero() {
            work.add_scalar_mut(-self.bias.clone());
        }

        let m_two: T = crate::convert(-2.0f64);
        lhs.gerc(m_two, work, &self.axis, T::one());
    }

    /// Applies a signed reflection to the rows of a matrix in-place.
    ///
    /// This is a specialized variant of [`reflect_rows`](Self::reflect_rows) that includes an
    /// additional `sign` parameter. This method combines row reflection with sign control,
    /// which is essential for advanced numerical algorithms like QR decomposition, Schur
    /// decomposition, and other matrix factorizations.
    ///
    /// # What Does the Sign Parameter Do?
    ///
    /// The sign parameter modifies the reflection applied to rows:
    /// - **sign = 1**: Standard row reflection (same as [`reflect_rows`](Self::reflect_rows))
    /// - **sign = -1**: Inverted row reflection
    /// - **Other values**: Scaled reflection for specialized algorithms
    ///
    /// This is particularly important in Householder transformations where the sign affects
    /// numerical stability and the structure of the resulting decomposition.
    ///
    /// # Work Vector
    ///
    /// Like [`reflect_rows`](Self::reflect_rows), this method requires a temporary work vector
    /// for efficient computation. The work vector is reused across multiple reflections in
    /// iterative algorithms.
    ///
    /// # Parameters
    ///
    /// * `lhs` - A mutable reference to the matrix whose rows will be reflected
    /// * `work` - A mutable work vector for temporary storage during computation
    /// * `sign` - A scalar that controls the direction/scaling of the reflection
    ///
    /// # When to Use This
    ///
    /// Most users should use [`reflect_rows`](Self::reflect_rows) instead. Use `reflect_rows_with_sign` only when:
    /// - Implementing QR decomposition or similar matrix factorizations
    /// - You need precise control over the reflection orientation for numerical stability
    /// - Working with advanced linear algebra algorithms that require sign control
    ///
    /// # Examples
    ///
    /// ## Standard row reflection with sign = 1
    /// ```
    /// # use nalgebra::{Reflection, Matrix2, Vector2, Unit};
    /// let axis = Unit::new_normalize(Vector2::new(1.0, 0.0));
    /// let reflection = Reflection::new(axis, 0.0);
    ///
    /// let mut matrix = Matrix2::new(
    ///     3.0, 5.0,
    ///     -2.0, 7.0
    /// );
    ///
    /// let mut work = Vector2::zeros();
    ///
    /// // sign = 1.0 gives normal row reflection
    /// reflection.reflect_rows_with_sign(&mut matrix, &mut work, 1.0);
    ///
    /// assert_eq!(matrix[(0, 0)], -3.0);
    /// assert_eq!(matrix[(0, 1)], 5.0);
    /// ```
    ///
    /// ## QR decomposition with controlled sign
    /// ```
    /// # use nalgebra::{Reflection, Matrix3, Vector3, Unit};
    /// // Householder reflection in QR algorithm
    /// let axis = Unit::new_normalize(Vector3::new(1.0, 1.0, 1.0));
    /// let householder = Reflection::new(axis, 0.0);
    ///
    /// let mut q_matrix = Matrix3::identity();
    /// let mut work = Vector3::zeros();
    ///
    /// // Apply reflection with sign control for numerical stability
    /// let sign = 1.0;
    /// householder.reflect_rows_with_sign(&mut q_matrix, &mut work, sign);
    ///
    /// // The Q matrix is now updated with the Householder transformation
    /// ```
    ///
    /// ## Inverted reflection with sign = -1
    /// ```
    /// # use nalgebra::{Reflection, Matrix2, Vector2, Unit};
    /// let axis = Unit::new_normalize(Vector2::new(1.0, 0.0));
    /// let reflection = Reflection::new(axis, 0.0);
    ///
    /// let mut matrix = Matrix2::new(
    ///     3.0, 5.0,
    ///     -2.0, 7.0
    /// );
    ///
    /// let mut work = Vector2::zeros();
    ///
    /// // sign = -1.0 creates an inverted reflection
    /// reflection.reflect_rows_with_sign(&mut matrix, &mut work, -1.0);
    ///
    /// // Different result from standard reflection
    /// // Note: the exact values depend on the implementation details
    /// ```
    ///
    /// ## Householder QR with sign for avoiding cancellation
    /// ```
    /// # use nalgebra::{Reflection, Matrix3, Vector3, Unit};
    /// // In QR decomposition, the sign is chosen to avoid catastrophic cancellation
    /// let axis = Unit::new_normalize(Vector3::new(0.8, 0.6, 0.0));
    /// let householder = Reflection::new(axis, 0.0);
    ///
    /// let mut r_matrix = Matrix3::new(
    ///     4.0, 1.0, -1.0,
    ///     0.0, 3.0,  2.0,
    ///     0.0, 0.0,  5.0
    /// );
    ///
    /// let mut work = Vector3::zeros();
    ///
    /// // The sign is typically chosen based on the first element's sign
    /// // to maximize numerical stability
    /// let sign = if r_matrix[(0, 0)] >= 0.0 { 1.0 } else { -1.0 };
    /// householder.reflect_rows_with_sign(&mut r_matrix, &mut work, sign);
    ///
    /// // Matrix is updated with numerically stable reflection
    /// ```
    ///
    /// ## Batch processing with controlled orientation
    /// ```
    /// # use nalgebra::{Reflection, Matrix2, Vector2, Unit};
    /// let axis = Unit::new_normalize(Vector2::new(0.0, 1.0));
    /// let reflection = Reflection::new(axis, 0.0);
    ///
    /// let mut data = Matrix2::new(
    ///     1.0, 2.0,
    ///     4.0, 5.0
    /// );
    ///
    /// let mut work = Vector2::zeros();
    ///
    /// // Use sign to control the transformation
    /// reflection.reflect_rows_with_sign(&mut data, &mut work, 1.0);
    ///
    /// // All rows are reflected with the specified sign
    /// ```
    ///
    /// # See Also
    /// - [`reflect_rows`](Self::reflect_rows) - Standard row reflection without sign (use this for most cases)
    /// - [`reflect_with_sign`](Self::reflect_with_sign) - Signed reflection for columns instead of rows
    /// - [`new`](Self::new) - Create a reflection
    pub fn reflect_rows_with_sign<R2: Dim, C2: Dim, S2, S3>(
        &self,
        lhs: &mut Matrix<T, R2, C2, S2>,
        work: &mut Vector<T, R2, S3>,
        sign: T,
    ) where
        S2: StorageMut<T, R2, C2>,
        S3: StorageMut<T, R2>,
        ShapeConstraint: DimEq<C2, D> + AreMultipliable<R2, C2, D, U1>,
    {
        lhs.mul_to(&self.axis, work);

        if !self.bias.is_zero() {
            work.add_scalar_mut(-self.bias.clone());
        }

        let m_two = sign.clone().scale(crate::convert(-2.0f64));
        lhs.gerc(m_two, work, &self.axis, sign);
    }
}
