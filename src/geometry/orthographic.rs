// Needed otherwise the rkyv macros generate code incompatible with rust-2024
#![cfg_attr(feature = "rkyv-serialize", allow(unsafe_op_in_unsafe_fn))]

#[cfg(feature = "arbitrary")]
use quickcheck::{Arbitrary, Gen};
#[cfg(feature = "rand-no-std")]
use rand::{
    Rng,
    distr::{Distribution, StandardUniform},
};
#[cfg(feature = "serde-serialize-no-std")]
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::fmt;

use simba::scalar::RealField;

use crate::base::dimension::U3;
use crate::base::storage::Storage;
use crate::base::{Matrix4, Vector, Vector3};

use crate::geometry::{Point3, Projective3};

#[cfg(feature = "rkyv-serialize")]
use rkyv::bytecheck;

/// A 3D orthographic projection stored as a homogeneous 4x4 matrix.
///
/// # What is Orthographic Projection?
///
/// Orthographic projection is a type of parallel projection where all projection lines are
/// perpendicular to the projection plane. Unlike perspective projection, objects do not appear
/// smaller as they get farther away - parallel lines remain parallel, and objects maintain
/// their size regardless of distance from the camera.
///
/// This makes orthographic projection ideal for:
/// - **2D games**: Platformers, top-down games, and side-scrollers
/// - **CAD applications**: Technical drawings where accurate measurements matter
/// - **Isometric views**: Games with an angled top-down perspective
/// - **UI rendering**: Overlays and HUDs that need consistent sizing
/// - **Blueprint/schematic views**: Engineering and architectural diagrams
///
/// # The View Cuboid
///
/// An orthographic projection defines a rectangular box (cuboid) in 3D space called the
/// "view cuboid" or "view volume". Everything inside this box is visible and gets mapped to
/// normalized device coordinates (NDC) in the range [-1, 1] for x, y, and z. Objects outside
/// this box are clipped (not rendered).
///
/// The view cuboid is defined by six planes:
/// - `left` and `right`: Define the x-axis bounds
/// - `bottom` and `top`: Define the y-axis bounds
/// - `znear` and `zfar`: Define the z-axis bounds (depth range)
///
/// # OpenGL Convention
///
/// This implementation follows the OpenGL convention where the camera looks down the negative
/// z-axis. This means `znear` and `zfar` are typically specified as positive values, but the
/// actual near plane is at `-znear` and the far plane is at `-zfar` in camera space.
///
/// # Example: 2D Game Setup
///
/// ```
/// # use nalgebra::{Orthographic3, Point3};
/// // Create an orthographic projection for a 2D game with a 800x600 viewport
/// // Origin at bottom-left, with depth range for layering sprites
/// let proj = Orthographic3::new(0.0, 800.0, 0.0, 600.0, -1.0, 1.0);
///
/// // A sprite at position (400, 300) in the center of the screen
/// let sprite_pos = Point3::new(400.0, 300.0, 0.0);
/// let ndc = proj.project_point(&sprite_pos);
/// // ndc will be at (0, 0, 0) - the center of the screen in NDC
/// ```
///
/// # Example: Isometric View
///
/// ```
/// # use nalgebra::{Orthographic3, Point3};
/// // Create an orthographic projection for an isometric game view
/// // Centered around origin with equal width/height for square pixels
/// let proj = Orthographic3::new(-10.0, 10.0, -10.0, 10.0, 0.1, 100.0);
///
/// // Objects within this 20x20x100 box will be visible
/// let visible = Point3::new(5.0, 5.0, -50.0);
/// let outside = Point3::new(15.0, 0.0, -50.0);  // Outside the view cuboid
/// ```
#[repr(C)]
#[cfg_attr(
    feature = "rkyv-serialize-no-std",
    derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize),
    archive(
        as = "Orthographic3<T::Archived>",
        bound(archive = "
        T: rkyv::Archive,
        Matrix4<T>: rkyv::Archive<Archived = Matrix4<T::Archived>>
    ")
    )
)]
#[cfg_attr(feature = "rkyv-serialize", derive(bytecheck::CheckBytes))]
#[derive(Copy, Clone)]
#[cfg_attr(feature = "defmt", derive(defmt::Format))]
pub struct Orthographic3<T> {
    matrix: Matrix4<T>,
}

impl<T: RealField> fmt::Debug for Orthographic3<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        self.matrix.fmt(f)
    }
}

impl<T: RealField> PartialEq for Orthographic3<T> {
    #[inline]
    fn eq(&self, right: &Self) -> bool {
        self.matrix == right.matrix
    }
}

#[cfg(feature = "bytemuck")]
unsafe impl<T> bytemuck::Zeroable for Orthographic3<T>
where
    T: RealField + bytemuck::Zeroable,
    Matrix4<T>: bytemuck::Zeroable,
{
}

#[cfg(feature = "bytemuck")]
unsafe impl<T> bytemuck::Pod for Orthographic3<T>
where
    T: RealField + bytemuck::Pod,
    Matrix4<T>: bytemuck::Pod,
{
}

#[cfg(feature = "serde-serialize-no-std")]
impl<T: RealField + Serialize> Serialize for Orthographic3<T> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.matrix.serialize(serializer)
    }
}

#[cfg(feature = "serde-serialize-no-std")]
impl<'a, T: RealField + Deserialize<'a>> Deserialize<'a> for Orthographic3<T> {
    fn deserialize<Des>(deserializer: Des) -> Result<Self, Des::Error>
    where
        Des: Deserializer<'a>,
    {
        let matrix = Matrix4::<T>::deserialize(deserializer)?;

        Ok(Self::from_matrix_unchecked(matrix))
    }
}

impl<T> Orthographic3<T> {
    /// Wraps the given matrix to interpret it as a 3D orthographic projection matrix.
    ///
    /// This is an "unchecked" constructor that assumes the provided matrix is already a valid
    /// orthographic projection matrix. No validation is performed - if the matrix doesn't
    /// represent a valid orthographic projection, the behavior of projection operations is
    /// undefined.
    ///
    /// # When to Use This
    ///
    /// Use this function when:
    /// - You've already computed an orthographic projection matrix manually
    /// - You're deserializing a matrix from a file or network
    /// - You're interfacing with external libraries that provide projection matrices
    /// - Performance is critical and you know the matrix is valid
    ///
    /// For most cases, prefer [`Orthographic3::new`] which constructs the matrix correctly.
    ///
    /// # Parameters
    ///
    /// * `matrix` - A 4x4 homogeneous matrix representing an orthographic projection
    ///
    /// # Example: Basic Usage
    ///
    /// ```
    /// # use nalgebra::{Orthographic3, Matrix4};
    /// // Manually construct an orthographic projection matrix
    /// // This matrix maps the view cuboid [1,10] x [2,20] x [0.1,1000] to NDC
    /// let mat = Matrix4::new(
    ///     2.0 / 9.0, 0.0,        0.0,         -11.0 / 9.0,
    ///     0.0,       2.0 / 18.0, 0.0,         -22.0 / 18.0,
    ///     0.0,       0.0,       -2.0 / 999.9, -1000.1 / 999.9,
    ///     0.0,       0.0,        0.0,         1.0
    /// );
    /// let proj = Orthographic3::from_matrix_unchecked(mat);
    /// assert_eq!(proj, Orthographic3::new(1.0, 10.0, 2.0, 20.0, 0.1, 1000.0));
    /// ```
    ///
    /// # Example: Loading from External Source
    ///
    /// ```
    /// # use nalgebra::{Orthographic3, Matrix4};
    /// // Simulating loading a projection matrix from a file or external library
    /// fn load_projection_from_file() -> Matrix4<f32> {
    ///     // In real code, this would read from a file
    ///     Matrix4::new(
    ///         0.1, 0.0, 0.0, 0.0,
    ///         0.0, 0.1, 0.0, 0.0,
    ///         0.0, 0.0, -0.02, -1.0,
    ///         0.0, 0.0, 0.0, 1.0
    ///     )
    /// }
    ///
    /// let matrix = load_projection_from_file();
    /// let proj = Orthographic3::from_matrix_unchecked(matrix);
    /// // Now you can use proj for transformations
    /// ```
    ///
    /// # See Also
    ///
    /// * [`Orthographic3::new`] - The safe way to create an orthographic projection
    /// * [`Orthographic3::as_matrix`] - Get a reference to the underlying matrix
    /// * [`Orthographic3::into_inner`] - Extract the underlying matrix
    #[inline]
    pub const fn from_matrix_unchecked(matrix: Matrix4<T>) -> Self {
        Self { matrix }
    }
}

impl<T: RealField> Orthographic3<T> {
    /// Creates a new orthographic projection matrix from the bounds of the view cuboid.
    ///
    /// This is the primary constructor for orthographic projections. It defines a rectangular
    /// box in 3D space (the view cuboid) that determines what is visible. Everything inside
    /// this box gets mapped to normalized device coordinates (NDC) in the range [-1, 1].
    ///
    /// # What This Does
    ///
    /// The projection transforms 3D points from camera/view space to NDC:
    /// - Points at the `left` edge map to NDC x = -1
    /// - Points at the `right` edge map to NDC x = +1
    /// - Points at the `bottom` edge map to NDC y = -1
    /// - Points at the `top` edge map to NDC y = +1
    /// - Points at the `znear` plane map to NDC z = -1
    /// - Points at the `zfar` plane map to NDC z = +1
    ///
    /// # OpenGL Convention
    ///
    /// This follows the OpenGL convention where the camera looks down the negative z-axis.
    /// The z-axis is flipped during projection, so objects farther away (more negative z)
    /// map to more positive NDC z values.
    ///
    /// # Parameters
    ///
    /// * `left` - The x-coordinate of the left vertical clipping plane
    /// * `right` - The x-coordinate of the right vertical clipping plane
    /// * `bottom` - The y-coordinate of the bottom horizontal clipping plane
    /// * `top` - The y-coordinate of the top horizontal clipping plane
    /// * `znear` - The distance to the near clipping plane (positive value)
    /// * `zfar` - The distance to the far clipping plane (positive value)
    ///
    /// # Panics
    ///
    /// Panics if `left == right`, `bottom == top`, or `znear == zfar`, as these would
    /// create a degenerate (zero-volume) view cuboid.
    ///
    /// # Example: 2D Platformer Game
    ///
    /// ```
    /// # use nalgebra::{Orthographic3, Point3};
    /// // Create a projection for a 2D platformer with a 16:9 viewport
    /// // Camera shows 32 units wide by 18 units tall, centered at origin
    /// let proj = Orthographic3::new(-16.0, 16.0, -9.0, 9.0, -1.0, 1.0);
    ///
    /// // Character at the center of the screen
    /// let player = Point3::new(0.0, 0.0, 0.0);
    /// let ndc = proj.project_point(&player);
    /// assert_eq!(ndc, Point3::new(0.0, 0.0, 0.0)); // Center in NDC
    /// ```
    ///
    /// # Example: UI Overlay (Pixel-Perfect)
    ///
    /// ```
    /// # use nalgebra::{Orthographic3, Point3};
    /// // Create a projection that maps directly to pixel coordinates
    /// // For a 1920x1080 screen with origin at bottom-left
    /// let proj = Orthographic3::new(0.0, 1920.0, 0.0, 1080.0, -1.0, 1.0);
    ///
    /// // A UI element at pixel (960, 540) - center of screen
    /// let button = Point3::new(960.0, 540.0, 0.0);
    /// let ndc = proj.project_point(&button);
    /// // ndc.x and ndc.y will be 0.0 (center in NDC space)
    /// ```
    ///
    /// # Example: Top-Down Strategy Game
    ///
    /// ```
    /// # use nalgebra::{Orthographic3, Point3};
    /// // Create a projection for a top-down RTS view
    /// // Show a 100x100 game unit area with depth for unit layering
    /// let proj = Orthographic3::new(-50.0, 50.0, -50.0, 50.0, 0.1, 100.0);
    ///
    /// // A unit at position (25, 25) in the game world
    /// let unit = Point3::new(25.0, 25.0, -10.0);
    /// let ndc = proj.project_point(&unit);
    /// // Unit appears at (0.5, 0.5, ...) in NDC - halfway to the right and up
    /// ```
    ///
    /// # Example: Cuboid Corners Map to NDC Cube
    ///
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{Orthographic3, Point3};
    /// let proj = Orthographic3::new(1.0, 10.0, 2.0, 20.0, 0.1, 1000.0);
    /// // Check this projection actually transforms the view cuboid into the double-unit cube.
    /// // See https://www.nalgebra.rs/docs/user_guide/projections#orthographic-projection for more details.
    /// let p1 = Point3::new(1.0, 2.0, -0.1);
    /// let p2 = Point3::new(1.0, 2.0, -1000.0);
    /// let p3 = Point3::new(1.0, 20.0, -0.1);
    /// let p4 = Point3::new(1.0, 20.0, -1000.0);
    /// let p5 = Point3::new(10.0, 2.0, -0.1);
    /// let p6 = Point3::new(10.0, 2.0, -1000.0);
    /// let p7 = Point3::new(10.0, 20.0, -0.1);
    /// let p8 = Point3::new(10.0, 20.0, -1000.0);
    ///
    /// assert_relative_eq!(proj.project_point(&p1), Point3::new(-1.0, -1.0, -1.0));
    /// assert_relative_eq!(proj.project_point(&p2), Point3::new(-1.0, -1.0,  1.0));
    /// assert_relative_eq!(proj.project_point(&p3), Point3::new(-1.0,  1.0, -1.0));
    /// assert_relative_eq!(proj.project_point(&p4), Point3::new(-1.0,  1.0,  1.0));
    /// assert_relative_eq!(proj.project_point(&p5), Point3::new( 1.0, -1.0, -1.0));
    /// assert_relative_eq!(proj.project_point(&p6), Point3::new( 1.0, -1.0,  1.0));
    /// assert_relative_eq!(proj.project_point(&p7), Point3::new( 1.0,  1.0, -1.0));
    /// assert_relative_eq!(proj.project_point(&p8), Point3::new( 1.0,  1.0,  1.0));
    ///
    /// // This also works with flipped axis. In other words, we allow that
    /// // `left > right`, `bottom > top`, and/or `znear > zfar`.
    /// let proj = Orthographic3::new(10.0, 1.0, 20.0, 2.0, 1000.0, 0.1);
    ///
    /// assert_relative_eq!(proj.project_point(&p1), Point3::new( 1.0,  1.0,  1.0));
    /// assert_relative_eq!(proj.project_point(&p2), Point3::new( 1.0,  1.0, -1.0));
    /// assert_relative_eq!(proj.project_point(&p3), Point3::new( 1.0, -1.0,  1.0));
    /// assert_relative_eq!(proj.project_point(&p4), Point3::new( 1.0, -1.0, -1.0));
    /// assert_relative_eq!(proj.project_point(&p5), Point3::new(-1.0,  1.0,  1.0));
    /// assert_relative_eq!(proj.project_point(&p6), Point3::new(-1.0,  1.0, -1.0));
    /// assert_relative_eq!(proj.project_point(&p7), Point3::new(-1.0, -1.0,  1.0));
    /// assert_relative_eq!(proj.project_point(&p8), Point3::new(-1.0, -1.0, -1.0));
    /// ```
    ///
    /// # See Also
    ///
    /// * [`Orthographic3::from_fov`] - Create from aspect ratio and field of view
    /// * [`Orthographic3::from_matrix_unchecked`] - Create from an existing matrix
    /// * [`Orthographic3::project_point`] - Transform points with this projection
    /// * [`Orthographic3::set_left_and_right`] - Modify the horizontal bounds
    /// * [`Orthographic3::set_bottom_and_top`] - Modify the vertical bounds
    /// * [`Orthographic3::set_znear_and_zfar`] - Modify the depth bounds
    #[inline]
    pub fn new(left: T, right: T, bottom: T, top: T, znear: T, zfar: T) -> Self {
        let matrix = Matrix4::<T>::identity();
        let mut res = Self::from_matrix_unchecked(matrix);

        res.set_left_and_right(left, right);
        res.set_bottom_and_top(bottom, top);
        res.set_znear_and_zfar(znear, zfar);

        res
    }

    /// Creates a new orthographic projection matrix from an aspect ratio and the vertical field of view.
    ///
    /// This is an alternative constructor useful when you want to match the visible area of a
    /// perspective projection but with orthographic rendering. It calculates the view cuboid
    /// dimensions based on the aspect ratio and vertical field of view angle.
    ///
    /// # Parameters
    ///
    /// * `aspect` - The aspect ratio (width / height) of the viewport. Must be non-zero.
    /// * `vfov` - The vertical field of view angle in radians
    /// * `znear` - The distance to the near clipping plane. Must not equal `zfar`.
    /// * `zfar` - The distance to the far clipping plane. Must not equal `znear`.
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - `znear == zfar` (would create zero depth range)
    /// - `aspect == 0` (would create zero width)
    ///
    /// # Example: Basic Usage
    ///
    /// ```
    /// # use nalgebra::Orthographic3;
    /// use std::f32::consts::PI;
    ///
    /// // Create projection with 16:9 aspect ratio and 60-degree vertical FOV
    /// let aspect = 16.0 / 9.0;
    /// let vfov = 60.0_f32.to_radians(); // Convert degrees to radians
    /// let proj = Orthographic3::from_fov(aspect, vfov, 0.1, 100.0);
    ///
    /// // The projection is now set up with bounds calculated from the FOV
    /// ```
    ///
    /// # Example: Matching a Perspective View
    ///
    /// ```
    /// # use nalgebra::{Orthographic3, Point3};
    /// // When switching between perspective and orthographic rendering,
    /// // this ensures a consistent visible area at the far plane
    /// let aspect = 1920.0 / 1080.0;  // Screen aspect ratio
    /// let vfov = 45.0_f32.to_radians();
    /// let ortho = Orthographic3::from_fov(aspect, vfov, 0.1, 1000.0);
    ///
    /// // Objects at the far plane will occupy the same screen space
    /// // as they would in a perspective projection
    /// ```
    ///
    /// # See Also
    ///
    /// * [`Orthographic3::new`] - Create with explicit view cuboid bounds
    /// * [`Orthographic3::set_znear_and_zfar`] - Adjust the depth range
    #[inline]
    pub fn from_fov(aspect: T, vfov: T, znear: T, zfar: T) -> Self {
        assert!(
            znear != zfar,
            "The far plane must not be equal to the near plane."
        );
        assert!(
            !relative_eq!(aspect, T::zero()),
            "The aspect ratio must not be zero."
        );

        let half: T = crate::convert(0.5);
        let width = zfar.clone() * (vfov * half.clone()).tan();
        let height = width.clone() / aspect;

        Self::new(
            -width.clone() * half.clone(),
            width * half.clone(),
            -height.clone() * half.clone(),
            height * half,
            znear,
            zfar,
        )
    }

    /// Computes the inverse of the orthographic projection matrix.
    ///
    /// The inverse transformation maps points from normalized device coordinates (NDC) back to
    /// camera/view space. This is useful for operations like:
    /// - Converting screen coordinates back to world space (unprojection)
    /// - Implementing picking/selection in 3D scenes
    /// - Camera movement calculations
    /// - Creating reverse transformations for effects
    ///
    /// The inverse of an orthographic projection is always well-defined since orthographic
    /// projections are affine transformations (no division by zero).
    ///
    /// # Returns
    ///
    /// A 4x4 matrix that is the inverse of this projection's transformation matrix.
    /// Multiplying this matrix by the original projection matrix yields the identity matrix.
    ///
    /// # Example: Verifying the Inverse
    ///
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{Orthographic3, Matrix4};
    /// let proj = Orthographic3::new(1.0, 10.0, 2.0, 20.0, 0.1, 1000.0);
    /// let inv = proj.inverse();
    ///
    /// // The inverse times the original equals identity
    /// assert_relative_eq!(inv * proj.as_matrix(), Matrix4::identity());
    /// assert_relative_eq!(proj.as_matrix() * inv, Matrix4::identity());
    ///
    /// // This also works with flipped axes
    /// let proj = Orthographic3::new(10.0, 1.0, 20.0, 2.0, 1000.0, 0.1);
    /// let inv = proj.inverse();
    /// assert_relative_eq!(inv * proj.as_matrix(), Matrix4::identity());
    /// assert_relative_eq!(proj.as_matrix() * inv, Matrix4::identity());
    /// ```
    ///
    /// # Example: Unprojection with Inverse
    ///
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{Orthographic3, Point3};
    /// let proj = Orthographic3::new(-10.0, 10.0, -10.0, 10.0, 0.1, 100.0);
    ///
    /// // A point in camera space
    /// let original = Point3::new(5.0, -3.0, -50.0);
    ///
    /// // Project it to NDC
    /// let ndc = proj.project_point(&original);
    ///
    /// // Use the inverse to transform back
    /// let inv = proj.inverse();
    /// let recovered = Point3::from(inv * ndc.to_homogeneous());
    ///
    /// assert_relative_eq!(recovered, original, epsilon = 1.0e-5);
    /// ```
    ///
    /// # Example: Mouse Picking in a 2D Game
    ///
    /// ```
    /// # use nalgebra::{Orthographic3, Point3, Matrix4};
    /// // Set up orthographic projection for a 2D game (800x600 screen)
    /// let proj = Orthographic3::new(0.0, 800.0, 0.0, 600.0, -1.0, 1.0);
    /// let inv = proj.inverse();
    ///
    /// // Mouse click at screen coordinates (400, 300) in NDC is (0, 0)
    /// let mouse_ndc = Point3::new(0.0, 0.0, 0.0);
    ///
    /// // Transform to world space using inverse
    /// let world_pos = Point3::from(inv * mouse_ndc.to_homogeneous());
    ///
    /// // world_pos is now (400, 300, 0) in game coordinates
    /// ```
    ///
    /// # Performance Note
    ///
    /// This method computes the inverse analytically, which is faster than using
    /// general matrix inversion. For repeated unprojections, consider using
    /// [`Orthographic3::unproject_point`] which is even more efficient.
    ///
    /// # See Also
    ///
    /// * [`Orthographic3::unproject_point`] - Faster method for unprojecting individual points
    /// * [`Orthographic3::as_matrix`] - Get the forward projection matrix
    /// * [`Orthographic3::project_point`] - Apply the forward transformation
    #[inline]
    #[must_use]
    pub fn inverse(&self) -> Matrix4<T> {
        let mut res = self.clone().to_homogeneous();

        let inv_m11 = T::one() / self.matrix[(0, 0)].clone();
        let inv_m22 = T::one() / self.matrix[(1, 1)].clone();
        let inv_m33 = T::one() / self.matrix[(2, 2)].clone();

        res[(0, 0)] = inv_m11.clone();
        res[(1, 1)] = inv_m22.clone();
        res[(2, 2)] = inv_m33.clone();

        res[(0, 3)] = -self.matrix[(0, 3)].clone() * inv_m11;
        res[(1, 3)] = -self.matrix[(1, 3)].clone() * inv_m22;
        res[(2, 3)] = -self.matrix[(2, 3)].clone() * inv_m33;

        res
    }

    /// Consumes this orthographic projection and returns the underlying homogeneous matrix.
    ///
    /// This method takes ownership of the projection and returns its 4x4 transformation matrix.
    /// The matrix can be used directly with graphics APIs (OpenGL, Vulkan, WebGPU, etc.) or
    /// multiplied with other transformation matrices.
    ///
    /// Use this when you need to extract the matrix for further processing or to pass it to
    /// rendering functions. If you need to keep using the projection, use [`as_matrix`]
    /// to get a reference instead.
    ///
    /// # Returns
    ///
    /// The 4x4 homogeneous transformation matrix representing this orthographic projection.
    ///
    /// # Example: Basic Usage
    ///
    /// ```
    /// # use nalgebra::{Orthographic3, Matrix4};
    /// let proj = Orthographic3::new(1.0, 10.0, 2.0, 20.0, 0.1, 1000.0);
    /// let expected = Matrix4::new(
    ///     2.0 / 9.0, 0.0,        0.0,         -11.0 / 9.0,
    ///     0.0,       2.0 / 18.0, 0.0,         -22.0 / 18.0,
    ///     0.0,       0.0,       -2.0 / 999.9, -1000.1 / 999.9,
    ///     0.0,       0.0,        0.0,         1.0
    /// );
    /// assert_eq!(proj.to_homogeneous(), expected);
    /// ```
    ///
    /// # Example: Sending to GPU
    ///
    /// ```
    /// # use nalgebra::Orthographic3;
    /// // Create projection for rendering
    /// let proj = Orthographic3::new(0.0, 800.0, 0.0, 600.0, -1.0, 1.0);
    ///
    /// // Extract matrix to send to shader uniform
    /// let matrix = proj.to_homogeneous();
    ///
    /// // In real code, you would upload this to your GPU:
    /// // gl.uniform_matrix4fv(location, false, matrix.as_slice());
    /// ```
    ///
    /// # Example: Combining Transformations
    ///
    /// ```
    /// # use nalgebra::{Orthographic3, Matrix4, Vector3};
    /// let proj = Orthographic3::new(-10.0, 10.0, -10.0, 10.0, 0.1, 100.0);
    ///
    /// // Create a view matrix (camera transformation)
    /// let view = Matrix4::new_translation(&Vector3::new(0.0, 0.0, -5.0));
    ///
    /// // Combine projection and view (in graphics, typically: proj * view * model)
    /// let view_proj = proj.to_homogeneous() * view;
    /// ```
    ///
    /// # See Also
    ///
    /// * [`Orthographic3::as_matrix`] - Get a reference without consuming the projection
    /// * [`Orthographic3::into_inner`] - Alias for this method
    /// * [`Orthographic3::from_matrix_unchecked`] - Construct from a matrix
    ///
    /// [`as_matrix`]: Orthographic3::as_matrix
    #[inline]
    #[must_use]
    pub fn to_homogeneous(self) -> Matrix4<T> {
        self.matrix
    }

    /// Returns a reference to the underlying homogeneous transformation matrix.
    ///
    /// This method provides read-only access to the projection matrix without consuming the
    /// `Orthographic3` struct. Use this when you need to inspect or use the matrix but want
    /// to keep the projection object for later use.
    ///
    /// This is particularly useful when you need to pass the matrix to multiple functions or
    /// when you want to query matrix properties without giving up ownership.
    ///
    /// # Returns
    ///
    /// A reference to the 4x4 homogeneous transformation matrix.
    ///
    /// # Example: Basic Usage
    ///
    /// ```
    /// # use nalgebra::{Orthographic3, Matrix4};
    /// let proj = Orthographic3::new(1.0, 10.0, 2.0, 20.0, 0.1, 1000.0);
    /// let expected = Matrix4::new(
    ///     2.0 / 9.0, 0.0,        0.0,         -11.0 / 9.0,
    ///     0.0,       2.0 / 18.0, 0.0,         -22.0 / 18.0,
    ///     0.0,       0.0,       -2.0 / 999.9, -1000.1 / 999.9,
    ///     0.0,       0.0,        0.0,         1.0
    /// );
    /// assert_eq!(*proj.as_matrix(), expected);
    /// ```
    ///
    /// # Example: Reusing the Projection
    ///
    /// ```
    /// # use nalgebra::{Orthographic3, Point3};
    /// let proj = Orthographic3::new(-10.0, 10.0, -10.0, 10.0, 0.1, 100.0);
    ///
    /// // Use the matrix reference multiple times
    /// let matrix_ref = proj.as_matrix();
    /// println!("Matrix determinant: {:?}", matrix_ref.determinant());
    /// println!("Matrix trace: {:?}", matrix_ref.trace());
    ///
    /// // proj is still available for use
    /// let point = Point3::new(5.0, 5.0, -50.0);
    /// let projected = proj.project_point(&point);
    /// ```
    ///
    /// # Example: Passing to Functions
    ///
    /// ```
    /// # use nalgebra::{Orthographic3, Matrix4};
    /// fn print_matrix_info(mat: &Matrix4<f32>) {
    ///     println!("Matrix: {}", mat);
    /// }
    ///
    /// let proj = Orthographic3::new(0.0, 800.0, 0.0, 600.0, -1.0, 1.0);
    ///
    /// // Pass matrix reference to function without consuming proj
    /// print_matrix_info(proj.as_matrix());
    ///
    /// // Can still use proj afterwards
    /// let bounds = (proj.left(), proj.right(), proj.bottom(), proj.top());
    /// ```
    ///
    /// # See Also
    ///
    /// * [`Orthographic3::to_homogeneous`] - Consumes and returns the matrix
    /// * [`Orthographic3::into_inner`] - Alias for `to_homogeneous`
    /// * [`Orthographic3::from_matrix_unchecked`] - Construct from a matrix
    #[inline]
    #[must_use]
    pub const fn as_matrix(&self) -> &Matrix4<T> {
        &self.matrix
    }

    /// Returns a reference to this orthographic projection as a `Projective3` transformation.
    ///
    /// This provides a view of the orthographic projection as a general projective transformation.
    /// `Projective3` is a more general type that can represent any 3D projective transformation,
    /// including perspective projections, affine transformations, and orthographic projections.
    ///
    /// This conversion is zero-cost (no data copying) and allows you to use the orthographic
    /// projection with generic code that works with any `Projective3` transformation.
    ///
    /// # Returns
    ///
    /// A reference to a `Projective3` that shares the same underlying matrix data.
    ///
    /// # Example: Basic Usage
    ///
    /// ```
    /// # use nalgebra::Orthographic3;
    /// let proj = Orthographic3::new(1.0, 10.0, 2.0, 20.0, 0.1, 1000.0);
    /// assert_eq!(proj.as_projective().to_homogeneous(), proj.to_homogeneous());
    /// ```
    ///
    /// # Example: Using with Generic Code
    ///
    /// ```
    /// # use nalgebra::{Orthographic3, Projective3, Point3};
    /// // A function that works with any projective transformation
    /// fn transform_point<T: nalgebra::RealField>(
    ///     proj: &Projective3<T>,
    ///     point: &Point3<T>
    /// ) -> Point3<T> {
    ///     proj.transform_point(point)
    /// }
    ///
    /// let ortho = Orthographic3::new(-10.0, 10.0, -10.0, 10.0, 0.1, 100.0);
    /// let point = Point3::new(5.0, 5.0, -50.0);
    ///
    /// // Use orthographic projection with generic projective function
    /// let transformed = transform_point(ortho.as_projective(), &point);
    /// ```
    ///
    /// # See Also
    ///
    /// * [`Orthographic3::to_projective`] - Consumes and converts to `Projective3`
    /// * [`Orthographic3::as_matrix`] - Get the underlying matrix
    #[inline]
    #[must_use]
    pub const fn as_projective(&self) -> &Projective3<T> {
        unsafe { &*(self as *const Orthographic3<T> as *const Projective3<T>) }
    }

    /// Consumes this orthographic projection and converts it into a `Projective3` transformation.
    ///
    /// This method takes ownership of the orthographic projection and returns an equivalent
    /// `Projective3`. Since both types share the same underlying matrix representation, this
    /// is a zero-cost conversion that simply reinterprets the type.
    ///
    /// `Projective3` is a more general type that can represent any 3D projective transformation.
    /// Use this conversion when you need the flexibility of working with a general projective
    /// transformation type.
    ///
    /// # Returns
    ///
    /// A `Projective3` transformation equivalent to this orthographic projection.
    ///
    /// # Example: Basic Conversion
    ///
    /// ```
    /// # use nalgebra::Orthographic3;
    /// let proj = Orthographic3::new(1.0, 10.0, 2.0, 20.0, 0.1, 1000.0);
    /// assert_eq!(proj.to_projective().to_homogeneous(), proj.to_homogeneous());
    /// ```
    ///
    /// # Example: Combining with Other Transformations
    ///
    /// ```
    /// # use nalgebra::{Orthographic3, Projective3, Matrix4, Vector3};
    /// let ortho = Orthographic3::new(-10.0, 10.0, -10.0, 10.0, 0.1, 100.0);
    ///
    /// // Convert to Projective3 for composition with other transformations
    /// let proj = ortho.to_projective();
    ///
    /// // Create another projective transformation (e.g., a shear)
    /// let shear_matrix = Matrix4::identity(); // Simplified - would be actual shear matrix
    /// let shear = Projective3::from_matrix_unchecked(shear_matrix);
    ///
    /// // Compose transformations
    /// let combined = proj * shear;
    /// ```
    ///
    /// # See Also
    ///
    /// * [`Orthographic3::as_projective`] - Get a reference without consuming
    /// * [`Orthographic3::to_homogeneous`] - Extract the matrix directly
    #[inline]
    #[must_use]
    pub fn to_projective(self) -> Projective3<T> {
        Projective3::from_matrix_unchecked(self.matrix)
    }

    /// Consumes the orthographic projection and returns the underlying homogeneous matrix.
    ///
    /// This is an alias for [`to_homogeneous`] and is the preferred method name going forward.
    /// It extracts the 4x4 transformation matrix from the projection, consuming the projection
    /// in the process.
    ///
    /// Use this when you need the matrix for direct manipulation, passing to graphics APIs,
    /// or combining with other matrix operations.
    ///
    /// # Returns
    ///
    /// The 4x4 homogeneous transformation matrix representing this orthographic projection.
    ///
    /// # Example: Basic Usage
    ///
    /// ```
    /// # use nalgebra::{Orthographic3, Matrix4};
    /// let proj = Orthographic3::new(1.0, 10.0, 2.0, 20.0, 0.1, 1000.0);
    /// let expected = Matrix4::new(
    ///     2.0 / 9.0, 0.0,        0.0,         -11.0 / 9.0,
    ///     0.0,       2.0 / 18.0, 0.0,         -22.0 / 18.0,
    ///     0.0,       0.0,       -2.0 / 999.9, -1000.1 / 999.9,
    ///     0.0,       0.0,        0.0,         1.0
    /// );
    /// assert_eq!(proj.into_inner(), expected);
    /// ```
    ///
    /// # Example: Extracting for Serialization
    ///
    /// ```
    /// # use nalgebra::Orthographic3;
    /// let proj = Orthographic3::new(0.0, 1920.0, 0.0, 1080.0, -1.0, 1.0);
    ///
    /// // Extract matrix for saving to file or sending over network
    /// let matrix = proj.into_inner();
    /// let matrix_data: Vec<f32> = matrix.iter().copied().collect();
    ///
    /// // Now matrix_data can be serialized
    /// ```
    ///
    /// # See Also
    ///
    /// * [`Orthographic3::to_homogeneous`] - Identical method (this is an alias)
    /// * [`Orthographic3::as_matrix`] - Get a reference without consuming
    /// * [`Orthographic3::from_matrix_unchecked`] - Reconstruct from a matrix
    ///
    /// [`to_homogeneous`]: Orthographic3::to_homogeneous
    #[inline]
    pub fn into_inner(self) -> Matrix4<T> {
        self.matrix
    }

    /// Retrieves the underlying homogeneous matrix.
    ///
    /// # Deprecated
    ///
    /// This method is deprecated. Use [`Orthographic3::into_inner`] instead, which has a
    /// clearer name and is consistent with standard Rust naming conventions.
    ///
    /// # See Also
    ///
    /// * [`Orthographic3::into_inner`] - The preferred replacement
    /// * [`Orthographic3::to_homogeneous`] - Alternative with clearer intent
    #[deprecated(note = "use `.into_inner()` instead")]
    #[inline]
    pub fn unwrap(self) -> Matrix4<T> {
        self.matrix
    }

    /// Returns the x-coordinate of the left clipping plane of the view cuboid.
    ///
    /// The left plane defines the minimum x-coordinate of the visible region. Points with
    /// x-coordinates less than this value will be clipped (not rendered). This value
    /// corresponds to NDC x = -1 after projection.
    ///
    /// # Returns
    ///
    /// The x-coordinate of the left edge of the view cuboid.
    ///
    /// # Example: Basic Usage
    ///
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::Orthographic3;
    /// let proj = Orthographic3::new(1.0, 10.0, 2.0, 20.0, 0.1, 1000.0);
    /// assert_relative_eq!(proj.left(), 1.0, epsilon = 1.0e-6);
    ///
    /// // Works with flipped axes too (left > right)
    /// let proj = Orthographic3::new(10.0, 1.0, 20.0, 2.0, 1000.0, 0.1);
    /// assert_relative_eq!(proj.left(), 10.0, epsilon = 1.0e-6);
    /// ```
    ///
    /// # Example: Querying View Bounds
    ///
    /// ```
    /// # use nalgebra::Orthographic3;
    /// let proj = Orthographic3::new(0.0, 800.0, 0.0, 600.0, -1.0, 1.0);
    ///
    /// // Get the horizontal extent of the view
    /// let width = proj.right() - proj.left();
    /// assert_eq!(width, 800.0);
    ///
    /// // Check if a point is within horizontal bounds
    /// let x = 400.0;
    /// let is_visible = x >= proj.left() && x <= proj.right();
    /// assert!(is_visible);
    /// ```
    ///
    /// # See Also
    ///
    /// * [`Orthographic3::right`] - Get the right clipping plane
    /// * [`Orthographic3::set_left`] - Modify the left clipping plane
    /// * [`Orthographic3::set_left_and_right`] - Set both horizontal bounds at once
    /// * [`Orthographic3::bottom`] - Get the bottom clipping plane
    /// * [`Orthographic3::top`] - Get the top clipping plane
    #[inline]
    #[must_use]
    pub fn left(&self) -> T {
        (-T::one() - self.matrix[(0, 3)].clone()) / self.matrix[(0, 0)].clone()
    }

    /// Returns the x-coordinate of the right clipping plane of the view cuboid.
    ///
    /// The right plane defines the maximum x-coordinate of the visible region. Points with
    /// x-coordinates greater than this value will be clipped (not rendered). This value
    /// corresponds to NDC x = +1 after projection.
    ///
    /// # Returns
    ///
    /// The x-coordinate of the right edge of the view cuboid.
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::Orthographic3;
    /// let proj = Orthographic3::new(1.0, 10.0, 2.0, 20.0, 0.1, 1000.0);
    /// assert_relative_eq!(proj.right(), 10.0, epsilon = 1.0e-6);
    ///
    /// let proj = Orthographic3::new(10.0, 1.0, 20.0, 2.0, 1000.0, 0.1);
    /// assert_relative_eq!(proj.right(), 1.0, epsilon = 1.0e-6);
    /// ```
    ///
    /// # See Also
    ///
    /// * [`Orthographic3::left`] - Get the left clipping plane
    /// * [`Orthographic3::set_right`] - Modify the right clipping plane
    /// * [`Orthographic3::set_left_and_right`] - Set both horizontal bounds at once
    #[inline]
    #[must_use]
    pub fn right(&self) -> T {
        (T::one() - self.matrix[(0, 3)].clone()) / self.matrix[(0, 0)].clone()
    }

    /// Returns the y-coordinate of the bottom clipping plane of the view cuboid.
    ///
    /// The bottom plane defines the minimum y-coordinate of the visible region. Points with
    /// y-coordinates less than this value will be clipped (not rendered). This value
    /// corresponds to NDC y = -1 after projection.
    ///
    /// # Returns
    ///
    /// The y-coordinate of the bottom edge of the view cuboid.
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::Orthographic3;
    /// let proj = Orthographic3::new(1.0, 10.0, 2.0, 20.0, 0.1, 1000.0);
    /// assert_relative_eq!(proj.bottom(), 2.0, epsilon = 1.0e-6);
    ///
    /// let proj = Orthographic3::new(10.0, 1.0, 20.0, 2.0, 1000.0, 0.1);
    /// assert_relative_eq!(proj.bottom(), 20.0, epsilon = 1.0e-6);
    /// ```
    ///
    /// # See Also
    ///
    /// * [`Orthographic3::top`] - Get the top clipping plane
    /// * [`Orthographic3::set_bottom`] - Modify the bottom clipping plane
    /// * [`Orthographic3::set_bottom_and_top`] - Set both vertical bounds at once
    #[inline]
    #[must_use]
    pub fn bottom(&self) -> T {
        (-T::one() - self.matrix[(1, 3)].clone()) / self.matrix[(1, 1)].clone()
    }

    /// Returns the y-coordinate of the top clipping plane of the view cuboid.
    ///
    /// The top plane defines the maximum y-coordinate of the visible region. Points with
    /// y-coordinates greater than this value will be clipped (not rendered). This value
    /// corresponds to NDC y = +1 after projection.
    ///
    /// # Returns
    ///
    /// The y-coordinate of the top edge of the view cuboid.
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::Orthographic3;
    /// let proj = Orthographic3::new(1.0, 10.0, 2.0, 20.0, 0.1, 1000.0);
    /// assert_relative_eq!(proj.top(), 20.0, epsilon = 1.0e-6);
    ///
    /// let proj = Orthographic3::new(10.0, 1.0, 20.0, 2.0, 1000.0, 0.1);
    /// assert_relative_eq!(proj.top(), 2.0, epsilon = 1.0e-6);
    /// ```
    ///
    /// # See Also
    ///
    /// * [`Orthographic3::bottom`] - Get the bottom clipping plane
    /// * [`Orthographic3::set_top`] - Modify the top clipping plane
    /// * [`Orthographic3::set_bottom_and_top`] - Set both vertical bounds at once
    #[inline]
    #[must_use]
    pub fn top(&self) -> T {
        (T::one() - self.matrix[(1, 3)].clone()) / self.matrix[(1, 1)].clone()
    }

    /// Returns the distance to the near clipping plane of the view cuboid.
    ///
    /// The near plane defines the closest z-distance that is visible. In OpenGL convention
    /// (used by this implementation), the camera looks down the negative z-axis, so the actual
    /// near plane is at `-znear`. Points closer to the camera than this will be clipped.
    /// This value corresponds to NDC z = -1 after projection.
    ///
    /// # Returns
    ///
    /// The distance to the near clipping plane.
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::Orthographic3;
    /// let proj = Orthographic3::new(1.0, 10.0, 2.0, 20.0, 0.1, 1000.0);
    /// assert_relative_eq!(proj.znear(), 0.1, epsilon = 1.0e-6);
    ///
    /// let proj = Orthographic3::new(10.0, 1.0, 20.0, 2.0, 1000.0, 0.1);
    /// assert_relative_eq!(proj.znear(), 1000.0, epsilon = 1.0e-6);
    /// ```
    ///
    /// # See Also
    ///
    /// * [`Orthographic3::zfar`] - Get the far clipping plane distance
    /// * [`Orthographic3::set_znear`] - Modify the near clipping plane
    /// * [`Orthographic3::set_znear_and_zfar`] - Set both depth bounds at once
    #[inline]
    #[must_use]
    pub fn znear(&self) -> T {
        (T::one() + self.matrix[(2, 3)].clone()) / self.matrix[(2, 2)].clone()
    }

    /// Returns the distance to the far clipping plane of the view cuboid.
    ///
    /// The far plane defines the farthest z-distance that is visible. In OpenGL convention
    /// (used by this implementation), the camera looks down the negative z-axis, so the actual
    /// far plane is at `-zfar`. Points farther from the camera than this will be clipped.
    /// This value corresponds to NDC z = +1 after projection.
    ///
    /// # Returns
    ///
    /// The distance to the far clipping plane.
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::Orthographic3;
    /// let proj = Orthographic3::new(1.0, 10.0, 2.0, 20.0, 0.1, 1000.0);
    /// assert_relative_eq!(proj.zfar(), 1000.0, epsilon = 1.0e-6);
    ///
    /// let proj = Orthographic3::new(10.0, 1.0, 20.0, 2.0, 1000.0, 0.1);
    /// assert_relative_eq!(proj.zfar(), 0.1, epsilon = 1.0e-6);
    /// ```
    ///
    /// # See Also
    ///
    /// * [`Orthographic3::znear`] - Get the near clipping plane distance
    /// * [`Orthographic3::set_zfar`] - Modify the far clipping plane
    /// * [`Orthographic3::set_znear_and_zfar`] - Set both depth bounds at once
    #[inline]
    #[must_use]
    pub fn zfar(&self) -> T {
        (-T::one() + self.matrix[(2, 3)].clone()) / self.matrix[(2, 2)].clone()
    }

    // TODO: when we get specialization, specialize the Mul impl instead.
    /// Projects a 3D point from camera/view space into normalized device coordinates (NDC).
    ///
    /// This is an optimized method that applies the orthographic projection transformation
    /// to a point. It's faster than multiplying by the full projection matrix because it
    /// takes advantage of the structure of orthographic projections (diagonal matrix with
    /// translation).
    ///
    /// The resulting point has coordinates in the range [-1, 1] for points within the view
    /// cuboid. Points outside this range are beyond the clipping planes but are not actually
    /// clipped by this function.
    ///
    /// # Parameters
    ///
    /// * `p` - A point in camera/view space to project
    ///
    /// # Returns
    ///
    /// The projected point in normalized device coordinates (NDC space).
    ///
    /// # Example: Basic Projection
    ///
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{Orthographic3, Point3};
    /// let proj = Orthographic3::new(1.0, 10.0, 2.0, 20.0, 0.1, 1000.0);
    ///
    /// // Project the eight corners of the view cuboid
    /// let p1 = Point3::new(1.0, 2.0, -0.1);
    /// let p2 = Point3::new(1.0, 2.0, -1000.0);
    /// let p3 = Point3::new(1.0, 20.0, -0.1);
    /// let p4 = Point3::new(1.0, 20.0, -1000.0);
    /// let p5 = Point3::new(10.0, 2.0, -0.1);
    /// let p6 = Point3::new(10.0, 2.0, -1000.0);
    /// let p7 = Point3::new(10.0, 20.0, -0.1);
    /// let p8 = Point3::new(10.0, 20.0, -1000.0);
    ///
    /// // All corners map to the NDC cube corners at (-1,-1,-1) to (1,1,1)
    /// assert_relative_eq!(proj.project_point(&p1), Point3::new(-1.0, -1.0, -1.0));
    /// assert_relative_eq!(proj.project_point(&p2), Point3::new(-1.0, -1.0,  1.0));
    /// assert_relative_eq!(proj.project_point(&p3), Point3::new(-1.0,  1.0, -1.0));
    /// assert_relative_eq!(proj.project_point(&p4), Point3::new(-1.0,  1.0,  1.0));
    /// assert_relative_eq!(proj.project_point(&p5), Point3::new( 1.0, -1.0, -1.0));
    /// assert_relative_eq!(proj.project_point(&p6), Point3::new( 1.0, -1.0,  1.0));
    /// assert_relative_eq!(proj.project_point(&p7), Point3::new( 1.0,  1.0, -1.0));
    /// assert_relative_eq!(proj.project_point(&p8), Point3::new( 1.0,  1.0,  1.0));
    /// ```
    ///
    /// # Example: 2D Game Object Positioning
    ///
    /// ```
    /// # use nalgebra::{Orthographic3, Point3};
    /// // Projection for a 2D game with 800x600 viewport
    /// let proj = Orthographic3::new(0.0, 800.0, 0.0, 600.0, -1.0, 1.0);
    ///
    /// // Project a game object at pixel (400, 300) - center of screen
    /// let sprite_pos = Point3::new(400.0, 300.0, 0.0);
    /// let ndc = proj.project_point(&sprite_pos);
    ///
    /// // In NDC, this is at (0, 0, 0) - the center
    /// assert_eq!(ndc, Point3::new(0.0, 0.0, 0.0));
    /// ```
    ///
    /// # Example: Checking Visibility
    ///
    /// ```
    /// # use nalgebra::{Orthographic3, Point3};
    /// let proj = Orthographic3::new(-10.0, 10.0, -10.0, 10.0, 0.1, 100.0);
    ///
    /// let visible_point = Point3::new(5.0, 5.0, -50.0);
    /// let outside_point = Point3::new(15.0, 0.0, -50.0);  // Outside right edge
    ///
    /// let ndc1 = proj.project_point(&visible_point);
    /// let ndc2 = proj.project_point(&outside_point);
    ///
    /// // Check if points are within the NDC cube [-1, 1]^3
    /// let is_visible = |p: &Point3<f32>| {
    ///     p.x >= -1.0 && p.x <= 1.0 && p.y >= -1.0 && p.y <= 1.0 && p.z >= -1.0 && p.z <= 1.0
    /// };
    ///
    /// assert!(is_visible(&ndc1));
    /// assert!(!is_visible(&ndc2));  // Outside the view
    /// ```
    ///
    /// # Performance Note
    ///
    /// This method is optimized for orthographic projections and is faster than multiplying
    /// the point by the full projection matrix. Use this instead of manual matrix multiplication
    /// when projecting many points.
    ///
    /// # See Also
    ///
    /// * [`Orthographic3::unproject_point`] - Reverse operation (NDC to camera space)
    /// * [`Orthographic3::project_vector`] - Project vectors (no translation)
    /// * [`Orthographic3::new`] - Create the projection defining the view cuboid
    #[inline]
    #[must_use]
    pub fn project_point(&self, p: &Point3<T>) -> Point3<T> {
        Point3::new(
            self.matrix[(0, 0)].clone() * p[0].clone() + self.matrix[(0, 3)].clone(),
            self.matrix[(1, 1)].clone() * p[1].clone() + self.matrix[(1, 3)].clone(),
            self.matrix[(2, 2)].clone() * p[2].clone() + self.matrix[(2, 3)].clone(),
        )
    }

    /// Transforms a point from normalized device coordinates (NDC) back to camera/view space.
    ///
    /// This is the inverse operation of [`project_point`]. It's optimized for orthographic
    /// projections and is faster than multiplying by the full inverse matrix. Use this to
    /// convert screen coordinates back to world space, implement mouse picking, or perform
    /// other unprojection operations.
    ///
    /// # Parameters
    ///
    /// * `p` - A point in normalized device coordinates (NDC) to unproject
    ///
    /// # Returns
    ///
    /// The unprojected point in camera/view space.
    ///
    /// # Example: Basic Unprojection
    ///
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{Orthographic3, Point3};
    /// let proj = Orthographic3::new(1.0, 10.0, 2.0, 20.0, 0.1, 1000.0);
    ///
    /// // Unproject the eight corners of the NDC cube back to view space
    /// let p1 = Point3::new(-1.0, -1.0, -1.0);
    /// let p2 = Point3::new(-1.0, -1.0,  1.0);
    /// let p3 = Point3::new(-1.0,  1.0, -1.0);
    /// let p4 = Point3::new(-1.0,  1.0,  1.0);
    /// let p5 = Point3::new( 1.0, -1.0, -1.0);
    /// let p6 = Point3::new( 1.0, -1.0,  1.0);
    /// let p7 = Point3::new( 1.0,  1.0, -1.0);
    /// let p8 = Point3::new( 1.0,  1.0,  1.0);
    ///
    /// // These map back to the view cuboid corners
    /// assert_relative_eq!(proj.unproject_point(&p1), Point3::new(1.0, 2.0, -0.1), epsilon = 1.0e-6);
    /// assert_relative_eq!(proj.unproject_point(&p2), Point3::new(1.0, 2.0, -1000.0), epsilon = 1.0e-6);
    /// assert_relative_eq!(proj.unproject_point(&p3), Point3::new(1.0, 20.0, -0.1), epsilon = 1.0e-6);
    /// assert_relative_eq!(proj.unproject_point(&p4), Point3::new(1.0, 20.0, -1000.0), epsilon = 1.0e-6);
    /// assert_relative_eq!(proj.unproject_point(&p5), Point3::new(10.0, 2.0, -0.1), epsilon = 1.0e-6);
    /// assert_relative_eq!(proj.unproject_point(&p6), Point3::new(10.0, 2.0, -1000.0), epsilon = 1.0e-6);
    /// assert_relative_eq!(proj.unproject_point(&p7), Point3::new(10.0, 20.0, -0.1), epsilon = 1.0e-6);
    /// assert_relative_eq!(proj.unproject_point(&p8), Point3::new(10.0, 20.0, -1000.0), epsilon = 1.0e-6);
    /// ```
    ///
    /// # Example: Mouse Picking in 2D
    ///
    /// ```
    /// # use nalgebra::{Orthographic3, Point3};
    /// // Set up orthographic projection for a 2D game (800x600 screen)
    /// let proj = Orthographic3::new(0.0, 800.0, 0.0, 600.0, -1.0, 1.0);
    ///
    /// // Mouse position in normalized device coordinates
    /// // (For example, converted from screen pixels)
    /// let mouse_ndc = Point3::new(0.5, -0.25, 0.0);
    ///
    /// // Unproject to get world coordinates
    /// let world_pos = proj.unproject_point(&mouse_ndc);
    ///
    /// // world_pos is now in game coordinate space
    /// // You can use this to determine what object the mouse is over
    /// ```
    ///
    /// # Example: Round-Trip Conversion
    ///
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{Orthographic3, Point3};
    /// let proj = Orthographic3::new(-10.0, 10.0, -10.0, 10.0, 0.1, 100.0);
    ///
    /// // Start with a point in view space
    /// let original = Point3::new(5.0, -3.0, -50.0);
    ///
    /// // Project to NDC and back
    /// let ndc = proj.project_point(&original);
    /// let recovered = proj.unproject_point(&ndc);
    ///
    /// // Should get back the original point
    /// assert_relative_eq!(recovered, original, epsilon = 1.0e-5);
    /// ```
    ///
    /// # Performance Note
    ///
    /// This method is optimized and is faster than computing the full projection matrix
    /// inverse and multiplying. Use this for unprojecting many points efficiently.
    ///
    /// # See Also
    ///
    /// * [`Orthographic3::project_point`] - The inverse operation (camera space to NDC)
    /// * [`Orthographic3::inverse`] - Get the full inverse matrix
    /// * [`Orthographic3::new`] - Create the projection defining the transformation
    ///
    /// [`project_point`]: Orthographic3::project_point
    #[inline]
    #[must_use]
    pub fn unproject_point(&self, p: &Point3<T>) -> Point3<T> {
        Point3::new(
            (p[0].clone() - self.matrix[(0, 3)].clone()) / self.matrix[(0, 0)].clone(),
            (p[1].clone() - self.matrix[(1, 3)].clone()) / self.matrix[(1, 1)].clone(),
            (p[2].clone() - self.matrix[(2, 3)].clone()) / self.matrix[(2, 2)].clone(),
        )
    }

    // TODO: when we get specialization, specialize the Mul impl instead.
    /// Projects a 3D vector using only the scaling part of the orthographic projection.
    ///
    /// Unlike [`project_point`], this method does not apply the translation component of the
    /// projection. This is appropriate for transforming directions, velocities, normals, or
    /// any vector quantity that represents a displacement rather than a position.
    ///
    /// This is an optimized method that is faster than full matrix multiplication.
    ///
    /// # Parameters
    ///
    /// * `p` - A vector in camera/view space to project
    ///
    /// # Returns
    ///
    /// The projected vector in NDC space (without translation).
    ///
    /// # Example: Basic Vector Projection
    ///
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{Orthographic3, Vector3};
    /// let proj = Orthographic3::new(1.0, 10.0, 2.0, 20.0, 0.1, 1000.0);
    ///
    /// // Project the basis vectors
    /// let v1 = Vector3::x();
    /// let v2 = Vector3::y();
    /// let v3 = Vector3::z();
    ///
    /// // The projection scales each component independently
    /// assert_relative_eq!(proj.project_vector(&v1), Vector3::x() * 2.0 / 9.0);
    /// assert_relative_eq!(proj.project_vector(&v2), Vector3::y() * 2.0 / 18.0);
    /// assert_relative_eq!(proj.project_vector(&v3), Vector3::z() * -2.0 / 999.9);
    /// ```
    ///
    /// # Example: Projecting a Direction Vector
    ///
    /// ```
    /// # use nalgebra::{Orthographic3, Vector3};
    /// let proj = Orthographic3::new(0.0, 800.0, 0.0, 600.0, -1.0, 1.0);
    ///
    /// // A direction vector (e.g., velocity of a moving object)
    /// let velocity = Vector3::new(100.0, 50.0, 0.0);
    ///
    /// // Project the velocity to NDC space
    /// let ndc_velocity = proj.project_vector(&velocity);
    ///
    /// // The projected velocity maintains directional information
    /// // but is scaled according to the projection
    /// ```
    ///
    /// # Example: Difference Between Vector and Point Projection
    ///
    /// ```
    /// # use nalgebra::{Orthographic3, Vector3, Point3};
    /// let proj = Orthographic3::new(-10.0, 10.0, -10.0, 10.0, 0.1, 100.0);
    ///
    /// // A displacement vector
    /// let displacement = Vector3::new(5.0, 5.0, 0.0);
    ///
    /// // Project as a vector (no translation)
    /// let proj_vector = proj.project_vector(&displacement);
    ///
    /// // If we project it as a point, translation is applied
    /// let as_point = Point3::new(5.0, 5.0, 0.0);
    /// let proj_point = proj.project_point(&as_point);
    ///
    /// // The results differ by the projection's translation component
    /// assert_ne!(proj_vector.x, proj_point.x);
    /// assert_ne!(proj_vector.y, proj_point.y);
    /// ```
    ///
    /// # See Also
    ///
    /// * [`Orthographic3::project_point`] - Project points (includes translation)
    /// * [`Orthographic3::unproject_point`] - Reverse operation for points
    ///
    /// [`project_point`]: Orthographic3::project_point
    #[inline]
    #[must_use]
    pub fn project_vector<SB>(&self, p: &Vector<T, U3, SB>) -> Vector3<T>
    where
        SB: Storage<T, U3>,
    {
        Vector3::new(
            self.matrix[(0, 0)].clone() * p[0].clone(),
            self.matrix[(1, 1)].clone() * p[1].clone(),
            self.matrix[(2, 2)].clone() * p[2].clone(),
        )
    }

    /// Modifies the x-coordinate of the left clipping plane of the view cuboid.
    ///
    /// This adjusts the minimum x-coordinate of the visible region while keeping the right
    /// plane fixed. The projection matrix is recalculated to reflect the new bounds.
    ///
    /// # Parameters
    ///
    /// * `left` - The new x-coordinate for the left clipping plane
    ///
    /// # Panics
    ///
    /// Panics if `left == right()`, as this would create a degenerate (zero-width) view cuboid.
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::Orthographic3;
    /// let mut proj = Orthographic3::new(1.0, 10.0, 2.0, 20.0, 0.1, 1000.0);
    /// proj.set_left(2.0);
    /// assert_relative_eq!(proj.left(), 2.0, epsilon = 1.0e-6);
    ///
    /// // It is OK to set a left offset greater than the current right offset.
    /// proj.set_left(20.0);
    /// assert_relative_eq!(proj.left(), 20.0, epsilon = 1.0e-6);
    /// ```
    ///
    /// # See Also
    ///
    /// * [`Orthographic3::set_right`] - Modify the right clipping plane
    /// * [`Orthographic3::set_left_and_right`] - Set both horizontal bounds at once
    /// * [`Orthographic3::left`] - Get the current left value
    #[inline]
    pub fn set_left(&mut self, left: T) {
        let right = self.right();
        self.set_left_and_right(left, right);
    }

    /// Sets the right offset of the view cuboid.
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::Orthographic3;
    /// let mut proj = Orthographic3::new(1.0, 10.0, 2.0, 20.0, 0.1, 1000.0);
    /// proj.set_right(15.0);
    /// assert_relative_eq!(proj.right(), 15.0, epsilon = 1.0e-6);
    ///
    /// // It is OK to set a right offset smaller than the current left offset.
    /// proj.set_right(-3.0);
    /// assert_relative_eq!(proj.right(), -3.0, epsilon = 1.0e-6);
    /// ```
    #[inline]
    pub fn set_right(&mut self, right: T) {
        let left = self.left();
        self.set_left_and_right(left, right);
    }

    /// Sets the bottom offset of the view cuboid.
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::Orthographic3;
    /// let mut proj = Orthographic3::new(1.0, 10.0, 2.0, 20.0, 0.1, 1000.0);
    /// proj.set_bottom(8.0);
    /// assert_relative_eq!(proj.bottom(), 8.0, epsilon = 1.0e-6);
    ///
    /// // It is OK to set a bottom offset greater than the current top offset.
    /// proj.set_bottom(50.0);
    /// assert_relative_eq!(proj.bottom(), 50.0, epsilon = 1.0e-6);
    /// ```
    #[inline]
    pub fn set_bottom(&mut self, bottom: T) {
        let top = self.top();
        self.set_bottom_and_top(bottom, top);
    }

    /// Sets the top offset of the view cuboid.
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::Orthographic3;
    /// let mut proj = Orthographic3::new(1.0, 10.0, 2.0, 20.0, 0.1, 1000.0);
    /// proj.set_top(15.0);
    /// assert_relative_eq!(proj.top(), 15.0, epsilon = 1.0e-6);
    ///
    /// // It is OK to set a top offset smaller than the current bottom offset.
    /// proj.set_top(-3.0);
    /// assert_relative_eq!(proj.top(), -3.0, epsilon = 1.0e-6);
    /// ```
    #[inline]
    pub fn set_top(&mut self, top: T) {
        let bottom = self.bottom();
        self.set_bottom_and_top(bottom, top);
    }

    /// Sets the near plane offset of the view cuboid.
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::Orthographic3;
    /// let mut proj = Orthographic3::new(1.0, 10.0, 2.0, 20.0, 0.1, 1000.0);
    /// proj.set_znear(8.0);
    /// assert_relative_eq!(proj.znear(), 8.0, epsilon = 1.0e-6);
    ///
    /// // It is OK to set a znear greater than the current zfar.
    /// proj.set_znear(5000.0);
    /// assert_relative_eq!(proj.znear(), 5000.0, epsilon = 1.0e-6);
    /// ```
    #[inline]
    pub fn set_znear(&mut self, znear: T) {
        let zfar = self.zfar();
        self.set_znear_and_zfar(znear, zfar);
    }

    /// Sets the far plane offset of the view cuboid.
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::Orthographic3;
    /// let mut proj = Orthographic3::new(1.0, 10.0, 2.0, 20.0, 0.1, 1000.0);
    /// proj.set_zfar(15.0);
    /// assert_relative_eq!(proj.zfar(), 15.0, epsilon = 1.0e-6);
    ///
    /// // It is OK to set a zfar smaller than the current znear.
    /// proj.set_zfar(-3.0);
    /// assert_relative_eq!(proj.zfar(), -3.0, epsilon = 1.0e-6);
    /// ```
    #[inline]
    pub fn set_zfar(&mut self, zfar: T) {
        let znear = self.znear();
        self.set_znear_and_zfar(znear, zfar);
    }

    /// Sets both the left and right clipping planes of the view cuboid simultaneously.
    ///
    /// This modifies the horizontal extent of the visible region. Both planes are set at once
    /// and the projection matrix is recalculated. This is more efficient than calling
    /// `set_left` and `set_right` separately.
    ///
    /// # Parameters
    ///
    /// * `left` - The x-coordinate of the left clipping plane
    /// * `right` - The x-coordinate of the right clipping plane
    ///
    /// # Panics
    ///
    /// Panics if `left == right`, as this would create a degenerate (zero-width) view cuboid.
    ///
    /// # Example: Basic Usage
    ///
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::Orthographic3;
    /// let mut proj = Orthographic3::new(1.0, 10.0, 2.0, 20.0, 0.1, 1000.0);
    /// proj.set_left_and_right(7.0, 70.0);
    /// assert_relative_eq!(proj.left(), 7.0, epsilon = 1.0e-6);
    /// assert_relative_eq!(proj.right(), 70.0, epsilon = 1.0e-6);
    ///
    /// // It is also OK to have `left > right` (flipped axis).
    /// proj.set_left_and_right(70.0, 7.0);
    /// assert_relative_eq!(proj.left(), 70.0, epsilon = 1.0e-6);
    /// assert_relative_eq!(proj.right(), 7.0, epsilon = 1.0e-6);
    /// ```
    ///
    /// # Example: Resizing Viewport for Window Resize
    ///
    /// ```
    /// # use nalgebra::Orthographic3;
    /// let mut proj = Orthographic3::new(0.0, 800.0, 0.0, 600.0, -1.0, 1.0);
    ///
    /// // Window resized to 1920x1080
    /// proj.set_left_and_right(0.0, 1920.0);
    /// proj.set_bottom_and_top(0.0, 1080.0);
    ///
    /// // Projection now matches new window dimensions
    /// ```
    ///
    /// # See Also
    ///
    /// * [`Orthographic3::set_left`] - Set only the left plane
    /// * [`Orthographic3::set_right`] - Set only the right plane
    /// * [`Orthographic3::set_bottom_and_top`] - Set vertical bounds
    /// * [`Orthographic3::set_znear_and_zfar`] - Set depth bounds
    #[inline]
    pub fn set_left_and_right(&mut self, left: T, right: T) {
        assert!(
            left != right,
            "The left corner must not be equal to the right corner."
        );
        self.matrix[(0, 0)] = crate::convert::<_, T>(2.0) / (right.clone() - left.clone());
        self.matrix[(0, 3)] = -(right.clone() + left.clone()) / (right - left);
    }

    /// Sets both the bottom and top clipping planes of the view cuboid simultaneously.
    ///
    /// This modifies the vertical extent of the visible region. Both planes are set at once
    /// and the projection matrix is recalculated. This is more efficient than calling
    /// `set_bottom` and `set_top` separately.
    ///
    /// # Parameters
    ///
    /// * `bottom` - The y-coordinate of the bottom clipping plane
    /// * `top` - The y-coordinate of the top clipping plane
    ///
    /// # Panics
    ///
    /// Panics if `bottom == top`, as this would create a degenerate (zero-height) view cuboid.
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::Orthographic3;
    /// let mut proj = Orthographic3::new(1.0, 10.0, 2.0, 20.0, 0.1, 1000.0);
    /// proj.set_bottom_and_top(7.0, 70.0);
    /// assert_relative_eq!(proj.bottom(), 7.0, epsilon = 1.0e-6);
    /// assert_relative_eq!(proj.top(), 70.0, epsilon = 1.0e-6);
    ///
    /// // It is also OK to have `bottom > top` (flipped axis).
    /// proj.set_bottom_and_top(70.0, 7.0);
    /// assert_relative_eq!(proj.bottom(), 70.0, epsilon = 1.0e-6);
    /// assert_relative_eq!(proj.top(), 7.0, epsilon = 1.0e-6);
    /// ```
    ///
    /// # See Also
    ///
    /// * [`Orthographic3::set_bottom`] - Set only the bottom plane
    /// * [`Orthographic3::set_top`] - Set only the top plane
    /// * [`Orthographic3::set_left_and_right`] - Set horizontal bounds
    /// * [`Orthographic3::set_znear_and_zfar`] - Set depth bounds
    #[inline]
    pub fn set_bottom_and_top(&mut self, bottom: T, top: T) {
        assert_ne!(
            bottom, top,
            "The top corner must not be equal to the bottom corner."
        );
        self.matrix[(1, 1)] = crate::convert::<_, T>(2.0) / (top.clone() - bottom.clone());
        self.matrix[(1, 3)] = -(top.clone() + bottom.clone()) / (top - bottom);
    }

    /// Sets both the near and far clipping planes of the view cuboid simultaneously.
    ///
    /// This modifies the depth range of the visible region. Both planes are set at once
    /// and the projection matrix is recalculated. This is more efficient than calling
    /// `set_znear` and `set_zfar` separately.
    ///
    /// The depth range determines which objects are visible based on their distance from
    /// the camera along the negative z-axis (OpenGL convention).
    ///
    /// # Parameters
    ///
    /// * `znear` - The distance to the near clipping plane
    /// * `zfar` - The distance to the far clipping plane
    ///
    /// # Panics
    ///
    /// Panics if `znear == zfar`, as this would create a degenerate (zero-depth) view cuboid.
    ///
    /// # Example: Basic Usage
    ///
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::Orthographic3;
    /// let mut proj = Orthographic3::new(1.0, 10.0, 2.0, 20.0, 0.1, 1000.0);
    /// proj.set_znear_and_zfar(50.0, 5000.0);
    /// assert_relative_eq!(proj.znear(), 50.0, epsilon = 1.0e-6);
    /// assert_relative_eq!(proj.zfar(), 5000.0, epsilon = 1.0e-6);
    ///
    /// // It is also OK to have `znear > zfar` (flipped axis).
    /// proj.set_znear_and_zfar(5000.0, 0.5);
    /// assert_relative_eq!(proj.znear(), 5000.0, epsilon = 1.0e-6);
    /// assert_relative_eq!(proj.zfar(), 0.5, epsilon = 1.0e-6);
    /// ```
    ///
    /// # Example: Adjusting Depth Range for 2D Layers
    ///
    /// ```
    /// # use nalgebra::Orthographic3;
    /// // 2D game with multiple sprite layers
    /// let mut proj = Orthographic3::new(0.0, 800.0, 0.0, 600.0, 0.0, 10.0);
    ///
    /// // Increase depth range to accommodate more layers
    /// proj.set_znear_and_zfar(0.0, 100.0);
    ///
    /// // Now can render sprites from z=0 to z=-100
    /// ```
    ///
    /// # See Also
    ///
    /// * [`Orthographic3::set_znear`] - Set only the near plane
    /// * [`Orthographic3::set_zfar`] - Set only the far plane
    /// * [`Orthographic3::set_left_and_right`] - Set horizontal bounds
    /// * [`Orthographic3::set_bottom_and_top`] - Set vertical bounds
    #[inline]
    pub fn set_znear_and_zfar(&mut self, znear: T, zfar: T) {
        assert!(
            zfar != znear,
            "The near-plane and far-plane must not be superimposed."
        );
        self.matrix[(2, 2)] = -crate::convert::<_, T>(2.0) / (zfar.clone() - znear.clone());
        self.matrix[(2, 3)] = -(zfar.clone() + znear.clone()) / (zfar - znear);
    }
}

#[cfg(feature = "rand-no-std")]
impl<T: RealField> Distribution<Orthographic3<T>> for StandardUniform
where
    StandardUniform: Distribution<T>,
{
    /// Generate an arbitrary random variate for testing purposes.
    fn sample<R: Rng + ?Sized>(&self, r: &mut R) -> Orthographic3<T> {
        use crate::base::helper;
        let left = r.random();
        let right = helper::reject_rand(r, |x: &T| *x > left);
        let bottom = r.random();
        let top = helper::reject_rand(r, |x: &T| *x > bottom);
        let znear = r.random();
        let zfar = helper::reject_rand(r, |x: &T| *x > znear);

        Orthographic3::new(left, right, bottom, top, znear, zfar)
    }
}

#[cfg(feature = "arbitrary")]
impl<T: RealField + Arbitrary> Arbitrary for Orthographic3<T>
where
    Matrix4<T>: Send,
{
    fn arbitrary(g: &mut Gen) -> Self {
        use crate::base::helper;
        let left = Arbitrary::arbitrary(g);
        let right = helper::reject(g, |x: &T| *x > left);
        let bottom = Arbitrary::arbitrary(g);
        let top = helper::reject(g, |x: &T| *x > bottom);
        let znear = Arbitrary::arbitrary(g);
        let zfar = helper::reject(g, |x: &T| *x > znear);

        Self::new(left, right, bottom, top, znear, zfar)
    }
}

impl<T: RealField> From<Orthographic3<T>> for Matrix4<T> {
    #[inline]
    fn from(orth: Orthographic3<T>) -> Self {
        orth.into_inner()
    }
}
