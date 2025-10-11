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

/// A 3D perspective projection stored as a homogeneous 4x4 matrix.
///
/// # What is Perspective Projection?
///
/// Perspective projection simulates how our eyes and cameras see the world: objects appear
/// smaller as they get farther away. This creates realistic depth perception where parallel
/// lines (like train tracks) appear to converge toward a vanishing point on the horizon.
///
/// A perspective projection is defined by:
/// - **Aspect ratio**: The width-to-height ratio of your viewport (e.g., 16:9 for widescreen)
/// - **Field of View (FOV)**: How wide the camera can "see" (typically 45-90 degrees)
/// - **Near plane**: The closest distance objects can be seen (e.g., 0.1 units)
/// - **Far plane**: The farthest distance objects can be seen (e.g., 1000.0 units)
///
/// # Common Use Cases
///
/// - **3D games**: Creating a first-person or third-person camera view
/// - **3D visualization**: Rendering architectural models, scientific data
/// - **Virtual reality**: Setting up eye projections for VR headsets
/// - **Computer graphics**: Any application requiring realistic 3D rendering
///
/// # Example
///
/// ```
/// use nalgebra::{Perspective3, Point3};
///
/// // Create a perspective projection for a typical 3D game camera
/// // - 16:9 aspect ratio (1.777...)
/// // - 60 degree field of view
/// // - See objects from 0.1 to 1000.0 units away
/// let aspect = 16.0 / 9.0;
/// let fov = std::f32::consts::PI / 3.0; // 60 degrees in radians
/// let perspective = Perspective3::new(aspect, fov, 0.1, 1000.0);
///
/// // Project a 3D point to screen space
/// let point_3d = Point3::new(1.0, 2.0, -5.0);
/// let point_2d = perspective.project_point(&point_3d);
/// println!("3D point {:?} projects to 2D: {:?}", point_3d, point_2d);
/// ```
#[repr(C)]
#[cfg_attr(
    feature = "rkyv-serialize-no-std",
    derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize),
    archive(
        as = "Perspective3<T::Archived>",
        bound(archive = "
        T: rkyv::Archive,
        Matrix4<T>: rkyv::Archive<Archived = Matrix4<T::Archived>>
    ")
    )
)]
#[cfg_attr(feature = "rkyv-serialize", derive(bytecheck::CheckBytes))]
#[derive(Copy, Clone)]
#[cfg_attr(feature = "defmt", derive(defmt::Format))]
pub struct Perspective3<T> {
    matrix: Matrix4<T>,
}

impl<T: RealField> fmt::Debug for Perspective3<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        self.matrix.fmt(f)
    }
}

impl<T: RealField> PartialEq for Perspective3<T> {
    #[inline]
    fn eq(&self, right: &Self) -> bool {
        self.matrix == right.matrix
    }
}

#[cfg(feature = "bytemuck")]
unsafe impl<T> bytemuck::Zeroable for Perspective3<T>
where
    T: RealField + bytemuck::Zeroable,
    Matrix4<T>: bytemuck::Zeroable,
{
}

#[cfg(feature = "bytemuck")]
unsafe impl<T> bytemuck::Pod for Perspective3<T>
where
    T: RealField + bytemuck::Pod,
    Matrix4<T>: bytemuck::Pod,
{
}

#[cfg(feature = "serde-serialize-no-std")]
impl<T: RealField + Serialize> Serialize for Perspective3<T> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.matrix.serialize(serializer)
    }
}

#[cfg(feature = "serde-serialize-no-std")]
impl<'a, T: RealField + Deserialize<'a>> Deserialize<'a> for Perspective3<T> {
    fn deserialize<Des>(deserializer: Des) -> Result<Self, Des::Error>
    where
        Des: Deserializer<'a>,
    {
        let matrix = Matrix4::<T>::deserialize(deserializer)?;

        Ok(Self::from_matrix_unchecked(matrix))
    }
}

impl<T> Perspective3<T> {
    /// Wraps the given matrix to interpret it as a 3D perspective projection matrix.
    ///
    /// This function creates a `Perspective3` from an existing 4x4 matrix without validating
    /// that the matrix actually represents a valid perspective projection. This is useful when
    /// you already have a projection matrix from another source (like a graphics library or
    /// file format) and want to wrap it in the `Perspective3` type.
    ///
    /// # Safety
    ///
    /// While this function is safe in terms of memory, it doesn't verify that the matrix is
    /// a valid perspective projection. Using an invalid matrix may produce incorrect results
    /// when projecting points or retrieving parameters like FOV.
    ///
    /// # Parameters
    ///
    /// - `matrix`: A 4x4 matrix that should represent a perspective projection
    ///
    /// # Example
    ///
    /// ```
    /// use nalgebra::{Perspective3, Matrix4};
    ///
    /// // Create a perspective matrix first
    /// let perspective = Perspective3::new(16.0 / 9.0, 1.0, 0.1, 100.0);
    /// let matrix = perspective.to_homogeneous();
    ///
    /// // Wrap it back into a Perspective3 (in practice, this matrix might
    /// // come from an external source like a file or graphics API)
    /// let perspective2 = Perspective3::from_matrix_unchecked(matrix);
    ///
    /// assert_eq!(perspective.as_matrix(), perspective2.as_matrix());
    /// ```
    ///
    /// # See Also
    ///
    /// - [`Perspective3::new`]: Create a perspective projection from parameters (safer alternative)
    /// - [`Perspective3::to_homogeneous`]: Extract the underlying matrix
    /// - [`Perspective3::as_matrix`]: Get a reference to the underlying matrix
    #[inline]
    pub const fn from_matrix_unchecked(matrix: Matrix4<T>) -> Self {
        Self { matrix }
    }
}

impl<T: RealField> Perspective3<T> {
    /// Creates a new perspective projection from the aspect ratio, field of view, and depth range.
    ///
    /// This is the primary way to create a perspective projection for 3D rendering. It simulates
    /// a camera or eye view where objects farther away appear smaller, creating realistic depth.
    ///
    /// # Parameters
    ///
    /// - `aspect`: The width-to-height ratio of your viewport
    ///   - Example: For 1920x1080 resolution, aspect = 1920.0 / 1080.0 = 1.777...
    ///   - Common values: 16:9 = 1.777, 4:3 = 1.333, 1:1 = 1.0
    /// - `fovy`: The vertical field of view angle in radians (how much you can see vertically)
    ///   - Smaller values = zoomed in (telephoto lens)
    ///   - Larger values = zoomed out (wide-angle lens)
    ///   - Typical range: 45-90 degrees (0.785-1.57 radians)
    ///   - Human eye ≈ 90-100 degrees, games often use 60-90 degrees
    /// - `znear`: The near clipping plane (closest visible distance)
    ///   - Objects closer than this won't be rendered
    ///   - Should be > 0 (typically 0.1 or 0.01)
    ///   - Too small can cause z-fighting (depth buffer precision issues)
    /// - `zfar`: The far clipping plane (farthest visible distance)
    ///   - Objects farther than this won't be rendered
    ///   - Should be > znear (typically 100-10000)
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - `znear` equals `zfar` (planes must be different)
    /// - `aspect` is zero (would result in division by zero)
    ///
    /// # Example: First-Person Game Camera
    ///
    /// ```
    /// use nalgebra::Perspective3;
    /// use std::f32::consts::PI;
    ///
    /// // Create a typical first-person shooter camera
    /// let aspect = 1920.0 / 1080.0;  // 16:9 widescreen
    /// let fov = PI / 3.0;             // 60 degrees
    /// let near = 0.1;                 // Objects closer than 10cm are clipped
    /// let far = 1000.0;               // See up to 1km away
    ///
    /// let projection = Perspective3::new(aspect, fov, near, far);
    ///
    /// // This projection can now be used to render a 3D scene
    /// println!("Created projection with FOV: {:.1} degrees", projection.fovy() * 180.0 / PI);
    /// ```
    ///
    /// # Example: Different Aspect Ratios
    ///
    /// ```
    /// use nalgebra::Perspective3;
    ///
    /// // Widescreen (16:9)
    /// let widescreen = Perspective3::new(16.0_f32 / 9.0, 1.0, 0.1, 100.0);
    ///
    /// // Square viewport (1:1)
    /// let square = Perspective3::new(1.0_f32, 1.0, 0.1, 100.0);
    ///
    /// // Portrait mode (9:16)
    /// let portrait = Perspective3::new(9.0_f32 / 16.0, 1.0, 0.1, 100.0);
    ///
    /// // Verify different aspects
    /// assert!((widescreen.aspect() - 16.0 / 9.0).abs() < 1e-5);
    /// assert!((square.aspect() - 1.0).abs() < 1e-5);
    /// ```
    ///
    /// # Example: Comparing FOV Values
    ///
    /// ```
    /// use nalgebra::Perspective3;
    /// use std::f32::consts::PI;
    ///
    /// let aspect = 16.0 / 9.0;
    ///
    /// // Narrow FOV (telephoto/zoomed in)
    /// let narrow = Perspective3::new(aspect, PI / 6.0, 0.1, 100.0);  // 30 degrees
    ///
    /// // Normal FOV (standard camera)
    /// let normal = Perspective3::new(aspect, PI / 3.0, 0.1, 100.0);  // 60 degrees
    ///
    /// // Wide FOV (wide-angle lens)
    /// let wide = Perspective3::new(aspect, PI / 2.0, 0.1, 100.0);    // 90 degrees
    ///
    /// assert!(narrow.fovy() < normal.fovy());
    /// assert!(normal.fovy() < wide.fovy());
    /// ```
    ///
    /// # See Also
    ///
    /// - [`Perspective3::from_matrix_unchecked`]: Create from an existing matrix
    /// - [`Perspective3::set_fovy`]: Change the field of view after creation
    /// - [`Perspective3::set_aspect`]: Change the aspect ratio after creation
    /// - [`Perspective3::set_znear`]: Change the near plane after creation
    /// - [`Perspective3::set_zfar`]: Change the far plane after creation
    pub fn new(aspect: T, fovy: T, znear: T, zfar: T) -> Self {
        assert!(
            relative_ne!(zfar, znear),
            "The near-plane and far-plane must not be superimposed."
        );
        assert!(
            !relative_eq!(aspect, T::zero()),
            "The aspect ratio must not be zero."
        );

        let matrix = Matrix4::identity();
        let mut res = Self::from_matrix_unchecked(matrix);

        res.set_fovy(fovy);
        res.set_aspect(aspect);
        res.set_znear_and_zfar(znear, zfar);

        res.matrix[(3, 3)] = T::zero();
        res.matrix[(3, 2)] = -T::one();

        res
    }

    /// Computes the inverse of the perspective projection matrix.
    ///
    /// The inverse matrix can transform points from normalized device coordinates (NDC)
    /// back to view/camera space. This is useful for operations like:
    /// - Ray casting from screen coordinates into 3D space
    /// - Converting depth buffer values back to world space distances
    /// - Picking: determining which 3D object was clicked on screen
    ///
    /// This operation is optimized for perspective matrices and is faster than computing
    /// a general 4x4 matrix inverse.
    ///
    /// # Returns
    ///
    /// A 4x4 matrix that is the inverse of this perspective projection.
    ///
    /// # Example: Basic Inverse
    ///
    /// ```
    /// use nalgebra::{Perspective3, Point3, Matrix4};
    ///
    /// let perspective = Perspective3::new(16.0_f32 / 9.0, 1.0, 0.1, 100.0);
    /// let inverse = perspective.inverse();
    ///
    /// // The inverse can transform projected points back to their original coordinates
    /// let original = Point3::new(1.0, 2.0, -5.0);
    /// let projected = perspective.project_point(&original);
    ///
    /// // Transform back using the inverse matrix (in homogeneous coordinates)
    /// let back = inverse * projected.to_homogeneous();
    /// // After perspective division, we should get close to the original
    /// let recovered = Point3::new(
    ///     back.x / back.w,
    ///     back.y / back.w,
    ///     back.z / back.w
    /// );
    ///
    /// assert!((recovered.x - original.x).abs() < 1e-5);
    /// assert!((recovered.y - original.y).abs() < 1e-5);
    /// assert!((recovered.z - original.z).abs() < 1e-5);
    /// ```
    ///
    /// # Example: Ray Casting from Screen Coordinates
    ///
    /// ```
    /// use nalgebra::{Perspective3, Point3, Vector4};
    ///
    /// let projection = Perspective3::new(16.0_f32 / 9.0, 1.0, 0.1, 100.0);
    /// let inv_projection = projection.inverse();
    ///
    /// // Convert a screen-space point (in NDC: -1 to 1 range) back to view space
    /// // For example, the center of the screen at the far plane
    /// let ndc = Vector4::new(0.0, 0.0, 1.0, 1.0);  // Center of screen, far plane
    /// let view_space = inv_projection * ndc;
    ///
    /// // After perspective division
    /// let view_point = Point3::new(
    ///     view_space.x / view_space.w,
    ///     view_space.y / view_space.w,
    ///     view_space.z / view_space.w
    /// );
    ///
    /// // The z-coordinate should be close to the far plane distance (but negative)
    /// assert!((view_point.z + 100.0).abs() < 1e-2);
    /// ```
    ///
    /// # See Also
    ///
    /// - [`Perspective3::unproject_point`]: Convenience method to unproject points
    /// - [`Perspective3::project_point`]: Project points in the forward direction
    /// - [`Perspective3::to_homogeneous`]: Get the forward projection matrix
    #[inline]
    #[must_use]
    pub fn inverse(&self) -> Matrix4<T> {
        let mut res = self.clone().to_homogeneous();

        res[(0, 0)] = T::one() / self.matrix[(0, 0)].clone();
        res[(1, 1)] = T::one() / self.matrix[(1, 1)].clone();
        res[(2, 2)] = T::zero();

        let m23 = self.matrix[(2, 3)].clone();
        let m32 = self.matrix[(3, 2)].clone();

        res[(2, 3)] = T::one() / m32.clone();
        res[(3, 2)] = T::one() / m23.clone();
        res[(3, 3)] = -self.matrix[(2, 2)].clone() / (m23 * m32);

        res
    }

    /// Converts this perspective projection into its underlying 4x4 homogeneous matrix.
    ///
    /// This consumes the `Perspective3` and returns the transformation matrix that can be
    /// used directly in rendering pipelines. The matrix transforms points from view/camera
    /// space to clip space (homogeneous coordinates), which are then converted to normalized
    /// device coordinates (NDC) by dividing by the w component.
    ///
    /// This is useful when you need to pass the projection matrix to a graphics API
    /// (like OpenGL, Vulkan, WebGPU) or combine it with other transformation matrices.
    ///
    /// # Returns
    ///
    /// A 4x4 matrix representing this perspective projection.
    ///
    /// # Example: Using with Graphics APIs
    ///
    /// ```
    /// use nalgebra::{Perspective3, Matrix4};
    ///
    /// let projection = Perspective3::new(16.0 / 9.0, 1.0, 0.1, 100.0);
    ///
    /// // Get the matrix to upload to a shader
    /// let proj_matrix: Matrix4<f32> = projection.to_homogeneous();
    ///
    /// // In a real application, you would upload this to your GPU:
    /// // gl.uniform_matrix4fv(proj_location, false, proj_matrix.as_slice());
    /// ```
    ///
    /// # Example: Combining Transformations
    ///
    /// ```
    /// use nalgebra::{Perspective3, Matrix4, Vector3};
    ///
    /// let projection = Perspective3::new(16.0 / 9.0, 1.0, 0.1, 100.0);
    /// let proj_matrix = projection.to_homogeneous();
    ///
    /// // Combine with a view matrix (camera transformation)
    /// let view_matrix = Matrix4::new_translation(&Vector3::new(0.0, 0.0, -5.0));
    /// let view_projection = proj_matrix * view_matrix;
    ///
    /// // view_projection can now transform from world space directly to clip space
    /// ```
    ///
    /// # Example: Owned vs Reference
    ///
    /// ```
    /// use nalgebra::Perspective3;
    ///
    /// let projection = Perspective3::new(16.0 / 9.0, 1.0, 0.1, 100.0);
    ///
    /// // to_homogeneous() consumes the perspective
    /// let matrix1 = projection.to_homogeneous();
    ///
    /// // If you need to keep the perspective, use as_matrix() instead
    /// let projection2 = Perspective3::new(16.0 / 9.0, 1.0, 0.1, 100.0);
    /// let matrix_ref = projection2.as_matrix();  // Just a reference
    /// let fov = projection2.fovy();  // Can still use projection2
    /// ```
    ///
    /// # See Also
    ///
    /// - [`Perspective3::as_matrix`]: Get a reference without consuming the perspective
    /// - [`Perspective3::into_inner`]: Same as `to_homogeneous` (more explicit name)
    /// - [`Perspective3::from_matrix_unchecked`]: Create a perspective from a matrix
    /// - [`Perspective3::inverse`]: Get the inverse transformation matrix
    #[inline]
    #[must_use]
    pub fn to_homogeneous(self) -> Matrix4<T> {
        self.matrix.clone_owned()
    }

    /// Returns a reference to the underlying 4x4 projection matrix without consuming the perspective.
    ///
    /// This is useful when you need to access the matrix multiple times or pass it to functions
    /// that only need a reference. Unlike [`to_homogeneous`](Self::to_homogeneous), this method
    /// doesn't consume the `Perspective3` object, so you can continue using it afterward.
    ///
    /// # Returns
    ///
    /// A reference to the 4x4 transformation matrix.
    ///
    /// # Example: Reading Matrix Elements
    ///
    /// ```
    /// use nalgebra::Perspective3;
    ///
    /// let projection = Perspective3::new(16.0 / 9.0, 1.0, 0.1, 100.0);
    ///
    /// // Access the matrix without consuming the projection
    /// let matrix = projection.as_matrix();
    /// println!("Matrix element (0,0): {}", matrix[(0, 0)]);
    ///
    /// // We can still use projection after this
    /// println!("FOV: {}", projection.fovy());
    /// println!("Aspect: {}", projection.aspect());
    /// ```
    ///
    /// # Example: Passing to Functions
    ///
    /// ```
    /// use nalgebra::{Perspective3, Matrix4};
    ///
    /// fn print_matrix_diagonal(m: &Matrix4<f32>) {
    ///     println!("Diagonal: {} {} {} {}",
    ///              m[(0,0)], m[(1,1)], m[(2,2)], m[(3,3)]);
    /// }
    ///
    /// let projection = Perspective3::new(16.0 / 9.0, 1.0, 0.1, 100.0);
    ///
    /// // Pass a reference to the matrix
    /// print_matrix_diagonal(projection.as_matrix());
    ///
    /// // projection is still valid here
    /// assert!(projection.znear() > 0.0);
    /// ```
    ///
    /// # Example: Multiple Matrix Operations
    ///
    /// ```
    /// use nalgebra::{Perspective3, Vector4};
    ///
    /// let projection = Perspective3::new(16.0 / 9.0, 1.0, 0.1, 100.0);
    /// let matrix = projection.as_matrix();
    ///
    /// // Use the matrix reference multiple times
    /// let point1 = Vector4::new(1.0, 0.0, -5.0, 1.0);
    /// let point2 = Vector4::new(0.0, 1.0, -5.0, 1.0);
    ///
    /// let transformed1 = matrix * point1;
    /// let transformed2 = matrix * point2;
    ///
    /// // Still have access to perspective parameters
    /// println!("Near plane: {}", projection.znear());
    /// ```
    ///
    /// # See Also
    ///
    /// - [`Perspective3::to_homogeneous`]: Get an owned matrix (consumes the perspective)
    /// - [`Perspective3::into_inner`]: Same as `to_homogeneous`
    /// - [`Perspective3::as_projective`]: Get a reference as `Projective3` type
    #[inline]
    #[must_use]
    pub const fn as_matrix(&self) -> &Matrix4<T> {
        &self.matrix
    }

    /// Returns a reference to this perspective projection viewed as a general `Projective3` transformation.
    ///
    /// `Projective3` is a more general type that can represent any projective transformation,
    /// while `Perspective3` is specifically a perspective projection. This method allows you to
    /// use the perspective in contexts that expect a general projective transformation.
    ///
    /// # Returns
    ///
    /// A reference to this transformation as a `Projective3`.
    ///
    /// # Example
    ///
    /// ```
    /// use nalgebra::{Perspective3, Point3};
    ///
    /// let perspective = Perspective3::new(16.0 / 9.0, 1.0, 0.1, 100.0);
    ///
    /// // View as a general projective transformation
    /// let projective = perspective.as_projective();
    ///
    /// // Can use Projective3 methods
    /// let point = Point3::new(1.0, 2.0, -5.0);
    /// let transformed = projective * point;
    /// ```
    ///
    /// # See Also
    ///
    /// - [`Perspective3::to_projective`]: Convert to an owned `Projective3`
    /// - [`Perspective3::as_matrix`]: Get the underlying matrix reference
    #[inline]
    #[must_use]
    pub const fn as_projective(&self) -> &Projective3<T> {
        unsafe { &*(self as *const Perspective3<T> as *const Projective3<T>) }
    }

    /// Converts this perspective projection into a general `Projective3` transformation.
    ///
    /// This consumes the `Perspective3` and returns a `Projective3`, which represents
    /// a more general projective transformation. This is useful when you need to work with
    /// the perspective in a context that requires the more general type.
    ///
    /// # Returns
    ///
    /// The same transformation as a `Projective3`.
    ///
    /// # Example
    ///
    /// ```
    /// use nalgebra::{Perspective3, Projective3, Point3};
    ///
    /// let perspective = Perspective3::new(16.0 / 9.0, 1.0, 0.1, 100.0);
    ///
    /// // Convert to a general projective transformation
    /// let projective: Projective3<f32> = perspective.to_projective();
    ///
    /// // Use it to transform points
    /// let point = Point3::new(1.0, 2.0, -5.0);
    /// let transformed = projective * point;
    /// ```
    ///
    /// # See Also
    ///
    /// - [`Perspective3::as_projective`]: Get a reference without consuming
    /// - [`Perspective3::to_homogeneous`]: Get the transformation matrix
    #[inline]
    #[must_use]
    pub fn to_projective(self) -> Projective3<T> {
        Projective3::from_matrix_unchecked(self.matrix)
    }

    /// Extracts the underlying 4x4 matrix, consuming the perspective projection.
    ///
    /// This method has the same effect as [`to_homogeneous`](Self::to_homogeneous) but with
    /// a more explicit name. Use this when you want to emphasize that you're extracting
    /// the inner representation.
    ///
    /// # Returns
    ///
    /// The 4x4 transformation matrix.
    ///
    /// # Example
    ///
    /// ```
    /// use nalgebra::{Perspective3, Matrix4};
    ///
    /// let perspective = Perspective3::new(16.0 / 9.0, 1.0, 0.1, 100.0);
    ///
    /// // Extract the matrix
    /// let matrix: Matrix4<f32> = perspective.into_inner();
    ///
    /// // Now you can use the matrix directly
    /// // (perspective is consumed and can't be used anymore)
    /// ```
    ///
    /// # See Also
    ///
    /// - [`Perspective3::to_homogeneous`]: Same functionality, different name
    /// - [`Perspective3::as_matrix`]: Get a reference without consuming
    #[inline]
    pub fn into_inner(self) -> Matrix4<T> {
        self.matrix
    }

    /// Retrieves the underlying homogeneous matrix.
    /// Deprecated: Use [`Perspective3::into_inner`] instead.
    #[deprecated(note = "use `.into_inner()` instead")]
    #[inline]
    pub fn unwrap(self) -> Matrix4<T> {
        self.matrix
    }

    /// Returns the aspect ratio (width / height) of the view frustum.
    ///
    /// The aspect ratio determines how the viewport is shaped. It should match your
    /// rendering window's aspect ratio to avoid distortion.
    ///
    /// # Returns
    ///
    /// The width-to-height ratio of the viewport.
    ///
    /// # Example
    ///
    /// ```
    /// use nalgebra::Perspective3;
    ///
    /// // Create a 16:9 widescreen projection
    /// let perspective = Perspective3::new(16.0_f32 / 9.0, 1.0, 0.1, 100.0);
    ///
    /// let aspect = perspective.aspect();
    /// assert!((aspect - 16.0 / 9.0).abs() < 1e-5);
    ///
    /// println!("Aspect ratio: {:.3}", aspect); // Should be ~1.778
    /// ```
    ///
    /// # Example: Matching Window Size
    ///
    /// ```
    /// use nalgebra::Perspective3;
    ///
    /// // If your window is 1920x1080 pixels:
    /// let window_width = 1920.0_f32;
    /// let window_height = 1080.0;
    /// let aspect = window_width / window_height;
    ///
    /// let projection = Perspective3::new(aspect, 1.0, 0.1, 100.0);
    ///
    /// // The projection's aspect matches the window
    /// assert!((projection.aspect() - aspect).abs() < 1e-5);
    /// ```
    ///
    /// # See Also
    ///
    /// - [`Perspective3::set_aspect`]: Change the aspect ratio
    /// - [`Perspective3::new`]: Create a perspective with an aspect ratio
    #[inline]
    #[must_use]
    pub fn aspect(&self) -> T {
        self.matrix[(1, 1)].clone() / self.matrix[(0, 0)].clone()
    }

    /// Returns the vertical field of view (FOV) angle in radians.
    ///
    /// The field of view determines how wide the camera can "see." A larger FOV creates
    /// a wide-angle view (more of the scene is visible), while a smaller FOV creates a
    /// telephoto/zoomed-in view.
    ///
    /// # Returns
    ///
    /// The vertical field of view in radians (not degrees).
    ///
    /// # Example: Basic Usage
    ///
    /// ```
    /// use nalgebra::Perspective3;
    /// use std::f32::consts::PI;
    ///
    /// let fov_degrees = 60.0;
    /// let fov_radians = fov_degrees * PI / 180.0;
    ///
    /// let perspective = Perspective3::new(16.0 / 9.0, fov_radians, 0.1, 100.0);
    ///
    /// let retrieved_fov = perspective.fovy();
    /// let retrieved_degrees = retrieved_fov * 180.0 / PI;
    ///
    /// assert!((retrieved_degrees - fov_degrees).abs() < 0.01);
    /// println!("FOV: {:.1} degrees", retrieved_degrees);
    /// ```
    ///
    /// # Example: Comparing Different FOVs
    ///
    /// ```
    /// use nalgebra::Perspective3;
    /// use std::f32::consts::PI;
    ///
    /// // Different camera perspectives
    /// let telephoto = Perspective3::new(1.0, PI / 6.0, 0.1, 100.0);    // 30 degrees - zoomed in
    /// let normal = Perspective3::new(1.0, PI / 3.0, 0.1, 100.0);       // 60 degrees - standard
    /// let wide_angle = Perspective3::new(1.0, PI * 2.0 / 3.0, 0.1, 100.0); // 120 degrees - wide
    ///
    /// assert!(telephoto.fovy() < normal.fovy());
    /// assert!(normal.fovy() < wide_angle.fovy());
    ///
    /// println!("Telephoto FOV: {:.1}°", telephoto.fovy() * 180.0 / PI);
    /// println!("Normal FOV: {:.1}°", normal.fovy() * 180.0 / PI);
    /// println!("Wide-angle FOV: {:.1}°", wide_angle.fovy() * 180.0 / PI);
    /// ```
    ///
    /// # See Also
    ///
    /// - [`Perspective3::set_fovy`]: Change the field of view
    /// - [`Perspective3::new`]: Create a perspective with a specific FOV
    #[inline]
    #[must_use]
    pub fn fovy(&self) -> T {
        (T::one() / self.matrix[(1, 1)].clone()).atan() * crate::convert(2.0)
    }

    /// Returns the distance to the near clipping plane.
    ///
    /// The near plane is the closest distance at which objects are rendered. Anything
    /// closer to the camera than this distance is clipped (not drawn). This value should
    /// always be positive.
    ///
    /// # Returns
    ///
    /// The distance from the camera to the near clipping plane.
    ///
    /// # Example: Basic Usage
    ///
    /// ```
    /// use nalgebra::Perspective3;
    ///
    /// let near = 0.1_f32;
    /// let far = 100.0;
    /// let perspective = Perspective3::new(16.0 / 9.0, 1.0, near, far);
    ///
    /// assert!((perspective.znear() - near).abs() < 1e-5);
    /// println!("Near plane: {}", perspective.znear());
    /// ```
    ///
    /// # Example: Understanding Near Plane Effects
    ///
    /// ```
    /// use nalgebra::Perspective3;
    ///
    /// // Very close near plane - good for small-scale scenes
    /// let close_view = Perspective3::new(1.0_f32, 1.0, 0.01, 100.0);
    /// assert!((close_view.znear() - 0.01).abs() < 1e-7);
    ///
    /// // Standard near plane - good for most 3D games
    /// let standard_view = Perspective3::new(1.0_f32, 1.0, 0.1, 1000.0);
    /// assert!((standard_view.znear() - 0.1).abs() < 1e-6);
    ///
    /// // Distant near plane - for large-scale scenes
    /// let far_view = Perspective3::new(1.0_f32, 1.0, 1.0, 10000.0);
    /// assert!((far_view.znear() - 1.0).abs() < 1e-5);
    /// ```
    ///
    /// # See Also
    ///
    /// - [`Perspective3::zfar`]: Get the far plane distance
    /// - [`Perspective3::set_znear`]: Change the near plane
    /// - [`Perspective3::set_znear_and_zfar`]: Change both planes simultaneously
    #[inline]
    #[must_use]
    pub fn znear(&self) -> T {
        let ratio =
            (-self.matrix[(2, 2)].clone() + T::one()) / (-self.matrix[(2, 2)].clone() - T::one());

        self.matrix[(2, 3)].clone() / (ratio * crate::convert(2.0))
            - self.matrix[(2, 3)].clone() / crate::convert(2.0)
    }

    /// Returns the distance to the far clipping plane.
    ///
    /// The far plane is the farthest distance at which objects are rendered. Anything
    /// farther from the camera than this distance is clipped (not drawn). This value
    /// should always be greater than the near plane distance.
    ///
    /// # Returns
    ///
    /// The distance from the camera to the far clipping plane.
    ///
    /// # Example: Basic Usage
    ///
    /// ```
    /// use nalgebra::Perspective3;
    ///
    /// let near = 0.1_f32;
    /// let far = 100.0;
    /// let perspective = Perspective3::new(16.0 / 9.0, 1.0, near, far);
    ///
    /// assert!((perspective.zfar() - far).abs() < 1e-4);
    /// println!("Far plane: {}", perspective.zfar());
    /// ```
    ///
    /// # Example: Different Scene Scales
    ///
    /// ```
    /// use nalgebra::Perspective3;
    ///
    /// // Indoor scene - limited view distance
    /// let indoor = Perspective3::new(1.0_f32, 1.0, 0.1, 50.0);
    /// assert!((indoor.zfar() - 50.0).abs() < 1e-3);
    ///
    /// // Outdoor scene - medium view distance
    /// let outdoor = Perspective3::new(1.0_f32, 1.0, 0.1, 1000.0);
    /// assert!((outdoor.zfar() - 1000.0).abs() < 1.0);
    ///
    /// // Space scene - very large view distance
    /// let space = Perspective3::new(1.0_f32, 1.0, 1.0, 100000.0);
    /// assert!((space.zfar() - 100000.0).abs() < 1000.0);
    /// ```
    ///
    /// # Example: Verifying Plane Order
    ///
    /// ```
    /// use nalgebra::Perspective3;
    ///
    /// let perspective = Perspective3::new(16.0_f32 / 9.0, 1.0, 0.1, 100.0);
    ///
    /// // Near plane should always be closer than far plane
    /// assert!(perspective.znear() < perspective.zfar());
    /// ```
    ///
    /// # See Also
    ///
    /// - [`Perspective3::znear`]: Get the near plane distance
    /// - [`Perspective3::set_zfar`]: Change the far plane
    /// - [`Perspective3::set_znear_and_zfar`]: Change both planes simultaneously
    #[inline]
    #[must_use]
    pub fn zfar(&self) -> T {
        let ratio =
            (-self.matrix[(2, 2)].clone() + T::one()) / (-self.matrix[(2, 2)].clone() - T::one());

        (self.matrix[(2, 3)].clone() - ratio * self.matrix[(2, 3)].clone()) / crate::convert(2.0)
    }

    // TODO: add a method to retrieve znear and zfar simultaneously?

    // TODO: when we get specialization, specialize the Mul impl instead.
    /// Projects a 3D point from view/camera space to normalized device coordinates (NDC).
    ///
    /// This transforms a point in 3D camera space to 2D screen coordinates. Points are
    /// transformed such that visible points end up in the [-1, 1] range on x and y axes.
    /// The z coordinate represents depth, also in the [-1, 1] range.
    ///
    /// This method is optimized for perspective projections and is faster than using
    /// matrix multiplication directly.
    ///
    /// # Parameters
    ///
    /// - `p`: A point in view/camera space (negative z values are in front of the camera)
    ///
    /// # Returns
    ///
    /// The projected point in normalized device coordinates (NDC).
    ///
    /// # Example: Basic Projection
    ///
    /// ```
    /// use nalgebra::{Perspective3, Point3};
    ///
    /// let projection = Perspective3::new(16.0_f32 / 9.0, 1.0, 0.1, 100.0);
    ///
    /// // A point 5 units in front of the camera
    /// let point_3d = Point3::new(1.0, 0.5, -5.0);
    /// let point_2d = projection.project_point(&point_3d);
    ///
    /// // The projected point is now in NDC space
    /// println!("3D: {:?}", point_3d);
    /// println!("2D: {:?}", point_2d);
    ///
    /// // Points within the frustum should be in [-1, 1] range
    /// assert!(point_2d.x.abs() <= 1.5); // allowing some margin
    /// assert!(point_2d.y.abs() <= 1.5);
    /// ```
    ///
    /// # Example: Multiple Points
    ///
    /// ```
    /// use nalgebra::{Perspective3, Point3};
    ///
    /// let projection = Perspective3::new(1.0, 1.0, 0.1, 100.0);
    ///
    /// // Project several points in the scene
    /// let points = vec![
    ///     Point3::new(0.0, 0.0, -1.0),   // Center, close
    ///     Point3::new(1.0, 0.0, -2.0),   // Right, medium distance
    ///     Point3::new(0.0, 1.0, -5.0),   // Top, farther
    /// ];
    ///
    /// for point in &points {
    ///     let projected = projection.project_point(point);
    ///     println!("{:?} -> {:?}", point, projected);
    /// }
    /// ```
    ///
    /// # Example: Converting to Screen Coordinates
    ///
    /// ```
    /// use nalgebra::{Perspective3, Point3};
    ///
    /// let projection = Perspective3::new(16.0 / 9.0, 1.0, 0.1, 100.0);
    /// let point = Point3::new(1.0, 2.0, -5.0);
    ///
    /// // Project to NDC
    /// let ndc = projection.project_point(&point);
    ///
    /// // Convert NDC to screen pixels (e.g., 1920x1080)
    /// let screen_width = 1920.0;
    /// let screen_height = 1080.0;
    /// let screen_x = (ndc.x + 1.0) * 0.5 * screen_width;
    /// let screen_y = (1.0 - ndc.y) * 0.5 * screen_height; // Flip y for screen coords
    ///
    /// println!("Screen position: ({:.0}, {:.0})", screen_x, screen_y);
    /// ```
    ///
    /// # See Also
    ///
    /// - [`Perspective3::unproject_point`]: Reverse operation (NDC to view space)
    /// - [`Perspective3::project_vector`]: Project a vector instead of a point
    /// - [`Perspective3::inverse`]: Get the inverse projection matrix
    #[inline]
    #[must_use]
    pub fn project_point(&self, p: &Point3<T>) -> Point3<T> {
        let inverse_denom = -T::one() / p[2].clone();
        Point3::new(
            self.matrix[(0, 0)].clone() * p[0].clone() * inverse_denom.clone(),
            self.matrix[(1, 1)].clone() * p[1].clone() * inverse_denom.clone(),
            (self.matrix[(2, 2)].clone() * p[2].clone() + self.matrix[(2, 3)].clone())
                * inverse_denom,
        )
    }

    /// Transforms a point from normalized device coordinates (NDC) back to view/camera space.
    ///
    /// This is the inverse operation of [`project_point`](Self::project_point). It takes a
    /// point in NDC space (where coordinates are typically in the [-1, 1] range) and transforms
    /// it back to 3D camera/view space. This is useful for ray casting, picking, and converting
    /// screen coordinates to 3D rays.
    ///
    /// This method is optimized for perspective projections and is faster than using the
    /// inverse matrix multiplication.
    ///
    /// # Parameters
    ///
    /// - `p`: A point in normalized device coordinates (NDC)
    ///
    /// # Returns
    ///
    /// The unprojected point in view/camera space.
    ///
    /// # Example: Basic Unprojection
    ///
    /// ```
    /// use nalgebra::{Perspective3, Point3};
    ///
    /// let projection = Perspective3::new(16.0_f32 / 9.0, 1.0, 0.1, 100.0);
    ///
    /// // Start with a 3D point
    /// let original = Point3::new(1.0, 2.0, -5.0);
    ///
    /// // Project it to NDC
    /// let projected = projection.project_point(&original);
    ///
    /// // Unproject back to 3D
    /// let unprojected = projection.unproject_point(&projected);
    ///
    /// // Should get back to (approximately) the original point
    /// assert!((unprojected.x - original.x).abs() < 1e-5);
    /// assert!((unprojected.y - original.y).abs() < 1e-5);
    /// assert!((unprojected.z - original.z).abs() < 1e-5);
    /// ```
    ///
    /// # Example: Ray Casting from Screen Coordinates
    ///
    /// ```
    /// use nalgebra::{Perspective3, Point3, Vector3};
    ///
    /// let projection = Perspective3::new(16.0_f32 / 9.0, 1.0, 0.1, 100.0);
    ///
    /// // Mouse clicked at screen position (960, 540) in a 1920x1080 window
    /// let screen_x = 960.0;
    /// let screen_y = 540.0;
    /// let screen_width = 1920.0;
    /// let screen_height = 1080.0;
    ///
    /// // Convert to NDC (normalized device coordinates)
    /// let ndc_x = (screen_x / screen_width) * 2.0 - 1.0;
    /// let ndc_y = 1.0 - (screen_y / screen_height) * 2.0; // Flip y
    ///
    /// // Unproject at near and far planes to get a ray
    /// let near_point_ndc = Point3::new(ndc_x, ndc_y, -1.0); // Near plane in NDC
    /// let far_point_ndc = Point3::new(ndc_x, ndc_y, 1.0);   // Far plane in NDC
    ///
    /// let near_point = projection.unproject_point(&near_point_ndc);
    /// let far_point = projection.unproject_point(&far_point_ndc);
    ///
    /// // Create a ray from near to far
    /// let ray_direction = (far_point - near_point).normalize();
    ///
    /// println!("Ray origin: {:?}", near_point);
    /// println!("Ray direction: {:?}", ray_direction);
    /// ```
    ///
    /// # Example: Picking (Determining What Was Clicked)
    ///
    /// ```
    /// use nalgebra::{Perspective3, Point3};
    ///
    /// let projection = Perspective3::new(16.0_f32 / 9.0, 1.0, 0.1, 100.0);
    ///
    /// // Center of screen at various depths
    /// let depths = vec![-1.0_f32, 0.0, 0.5, 1.0]; // NDC depth values
    ///
    /// for depth in depths {
    ///     let ndc_point = Point3::new(0.0, 0.0, depth);
    ///     let world_point = projection.unproject_point(&ndc_point);
    ///     println!("NDC depth {:.1} -> world z: {:.2}", depth, world_point.z);
    /// }
    /// ```
    ///
    /// # See Also
    ///
    /// - [`Perspective3::project_point`]: Forward operation (view space to NDC)
    /// - [`Perspective3::inverse`]: Get the inverse projection matrix
    #[inline]
    #[must_use]
    pub fn unproject_point(&self, p: &Point3<T>) -> Point3<T> {
        let inverse_denom =
            self.matrix[(2, 3)].clone() / (p[2].clone() + self.matrix[(2, 2)].clone());

        Point3::new(
            p[0].clone() * inverse_denom.clone() / self.matrix[(0, 0)].clone(),
            p[1].clone() * inverse_denom.clone() / self.matrix[(1, 1)].clone(),
            -inverse_denom,
        )
    }

    // TODO: when we get specialization, specialize the Mul impl instead.
    /// Projects a 3D direction vector from view/camera space using the perspective transformation.
    ///
    /// This is similar to [`project_point`](Self::project_point), but for direction vectors
    /// rather than positions. Vectors don't have a position, so they're affected differently
    /// by perspective projection. This is useful for transforming normals, light directions,
    /// or velocity vectors.
    ///
    /// This method is optimized for perspective projections and is faster than using
    /// matrix multiplication directly.
    ///
    /// # Parameters
    ///
    /// - `p`: A direction vector in view/camera space
    ///
    /// # Returns
    ///
    /// The projected vector in NDC space.
    ///
    /// # Example: Projecting Normals
    ///
    /// ```
    /// use nalgebra::{Perspective3, Vector3};
    ///
    /// let projection = Perspective3::new(16.0 / 9.0, 1.0, 0.1, 100.0);
    ///
    /// // A normal vector pointing right
    /// let normal = Vector3::new(1.0, 0.0, -1.0);
    /// let projected_normal = projection.project_vector(&normal);
    ///
    /// println!("Original normal: {:?}", normal);
    /// println!("Projected normal: {:?}", projected_normal);
    /// ```
    ///
    /// # Example: Projecting Multiple Vectors
    ///
    /// ```
    /// use nalgebra::{Perspective3, Vector3};
    ///
    /// let projection = Perspective3::new(1.0, 1.0, 0.1, 100.0);
    ///
    /// let vectors = vec![
    ///     Vector3::new(1.0, 0.0, -1.0),  // Right
    ///     Vector3::new(0.0, 1.0, -1.0),  // Up
    ///     Vector3::new(0.0, 0.0, -1.0),  // Forward
    /// ];
    ///
    /// for vec in &vectors {
    ///     let projected = projection.project_vector(vec);
    ///     println!("{:?} -> {:?}", vec, projected);
    /// }
    /// ```
    ///
    /// # See Also
    ///
    /// - [`Perspective3::project_point`]: Project a point (position) instead
    /// - [`Perspective3::unproject_point`]: Reverse operation for points
    #[inline]
    #[must_use]
    pub fn project_vector<SB>(&self, p: &Vector<T, U3, SB>) -> Vector3<T>
    where
        SB: Storage<T, U3>,
    {
        let inverse_denom = -T::one() / p[2].clone();
        Vector3::new(
            self.matrix[(0, 0)].clone() * p[0].clone() * inverse_denom.clone(),
            self.matrix[(1, 1)].clone() * p[1].clone() * inverse_denom,
            self.matrix[(2, 2)].clone(),
        )
    }

    /// Changes the aspect ratio (width / height) of this perspective projection.
    ///
    /// The aspect ratio should match your rendering window's dimensions to avoid distortion.
    /// Call this method when the window is resized to maintain correct proportions.
    ///
    /// # Parameters
    ///
    /// - `aspect`: The new width-to-height ratio (must be non-zero)
    ///
    /// # Panics
    ///
    /// Panics if the aspect ratio is zero.
    ///
    /// # Example: Handling Window Resize
    ///
    /// ```
    /// use nalgebra::Perspective3;
    ///
    /// let mut projection = Perspective3::new(16.0_f32 / 9.0, 1.0, 0.1, 100.0);
    ///
    /// // Window resized to 1280x720 (still 16:9)
    /// projection.set_aspect(1280.0 / 720.0);
    /// assert!((projection.aspect() - 16.0 / 9.0).abs() < 1e-5);
    ///
    /// // Window resized to square (1:1)
    /// projection.set_aspect(1.0);
    /// assert!((projection.aspect() - 1.0).abs() < 1e-5);
    /// ```
    ///
    /// # Example: Responsive Camera
    ///
    /// ```
    /// use nalgebra::Perspective3;
    ///
    /// let mut projection = Perspective3::new(1.0, 1.0, 0.1, 100.0);
    ///
    /// // Simulate different screen orientations
    /// let landscape = 1920.0 / 1080.0;  // 16:9 landscape
    /// let portrait = 1080.0 / 1920.0;   // 9:16 portrait
    ///
    /// projection.set_aspect(landscape);
    /// println!("Landscape aspect: {:.3}", projection.aspect());
    ///
    /// projection.set_aspect(portrait);
    /// println!("Portrait aspect: {:.3}", projection.aspect());
    /// ```
    ///
    /// # See Also
    ///
    /// - [`Perspective3::aspect`]: Get the current aspect ratio
    /// - [`Perspective3::set_fovy`]: Change the field of view
    /// - [`Perspective3::new`]: Create a projection with a specific aspect
    #[inline]
    pub fn set_aspect(&mut self, aspect: T) {
        assert!(
            !relative_eq!(aspect, T::zero()),
            "The aspect ratio must not be zero."
        );
        self.matrix[(0, 0)] = self.matrix[(1, 1)].clone() / aspect;
    }

    /// Changes the vertical field of view (FOV) of this perspective projection.
    ///
    /// This effectively "zooms" the camera in or out. A smaller FOV zooms in (telephoto),
    /// while a larger FOV zooms out (wide-angle). Use this for zoom effects, aiming down
    /// sights, or dynamic FOV changes during gameplay.
    ///
    /// # Parameters
    ///
    /// - `fovy`: The new vertical field of view in radians
    ///
    /// # Example: Zoom Effect
    ///
    /// ```
    /// use nalgebra::Perspective3;
    /// use std::f32::consts::PI;
    ///
    /// let mut projection = Perspective3::new(16.0 / 9.0, PI / 3.0, 0.1, 100.0);
    ///
    /// // Normal view (60 degrees)
    /// println!("Normal FOV: {:.1}°", projection.fovy() * 180.0 / PI);
    ///
    /// // Zoom in (narrower FOV, like aiming down sights)
    /// projection.set_fovy(PI / 6.0); // 30 degrees
    /// println!("Zoomed FOV: {:.1}°", projection.fovy() * 180.0 / PI);
    ///
    /// // Zoom out (wider FOV, like sprinting)
    /// projection.set_fovy(PI / 2.0); // 90 degrees
    /// println!("Wide FOV: {:.1}°", projection.fovy() * 180.0 / PI);
    /// ```
    ///
    /// # Example: Dynamic FOV (Speed Effect)
    ///
    /// ```
    /// use nalgebra::Perspective3;
    /// use std::f32::consts::PI;
    ///
    /// let mut projection = Perspective3::new(16.0 / 9.0, PI / 3.0, 0.1, 100.0);
    /// let base_fov = PI / 3.0; // 60 degrees
    ///
    /// // Simulate increasing speed (0.0 to 1.0)
    /// let speeds = [0.0, 0.5, 1.0];
    ///
    /// for speed in speeds {
    ///     // Increase FOV as speed increases (up to 90 degrees)
    ///     let fov = base_fov + speed * (PI / 6.0);
    ///     projection.set_fovy(fov);
    ///     println!("Speed {:.1}: FOV = {:.1}°", speed, projection.fovy() * 180.0 / PI);
    /// }
    /// ```
    ///
    /// # Example: Weapon Zoom
    ///
    /// ```
    /// use nalgebra::Perspective3;
    /// use std::f32::consts::PI;
    ///
    /// let mut camera = Perspective3::new(16.0 / 9.0, PI / 3.0, 0.1, 100.0);
    ///
    /// // Different zoom levels for a sniper scope
    /// let no_scope = PI / 3.0;      // 60 degrees
    /// let scope_2x = PI / 6.0;      // 30 degrees (2x zoom)
    /// let scope_4x = PI / 12.0;     // 15 degrees (4x zoom)
    ///
    /// // Toggle zoom
    /// camera.set_fovy(scope_4x);
    /// assert!((camera.fovy() - scope_4x).abs() < 0.01);
    /// ```
    ///
    /// # See Also
    ///
    /// - [`Perspective3::fovy`]: Get the current field of view
    /// - [`Perspective3::set_aspect`]: Change the aspect ratio
    /// - [`Perspective3::new`]: Create a projection with a specific FOV
    #[inline]
    pub fn set_fovy(&mut self, fovy: T) {
        let old_m22 = self.matrix[(1, 1)].clone();
        let new_m22 = T::one() / (fovy / crate::convert(2.0)).tan();
        self.matrix[(1, 1)] = new_m22.clone();
        self.matrix[(0, 0)] *= new_m22 / old_m22;
    }

    /// Changes the near clipping plane distance of this perspective projection.
    ///
    /// Objects closer than this distance won't be rendered. Adjusting the near plane can
    /// help with depth buffer precision issues (z-fighting) or allow closer object visibility.
    ///
    /// Note: Changing the near plane also requires retrieving and setting the far plane,
    /// which involves some computation. If you need to change both planes, use
    /// [`set_znear_and_zfar`](Self::set_znear_and_zfar) instead for better performance.
    ///
    /// # Parameters
    ///
    /// - `znear`: The new near plane distance (must be positive and less than far plane)
    ///
    /// # Example: Adjusting Near Plane
    ///
    /// ```
    /// use nalgebra::Perspective3;
    ///
    /// let mut projection = Perspective3::new(16.0_f32 / 9.0, 1.0, 0.1, 100.0);
    ///
    /// // Get closer to objects
    /// projection.set_znear(0.01);
    /// assert!((projection.znear() - 0.01).abs() < 1e-7);
    ///
    /// // Move near plane back
    /// projection.set_znear(1.0);
    /// assert!((projection.znear() - 1.0).abs() < 1e-5);
    /// ```
    ///
    /// # Example: Fixing Z-Fighting
    ///
    /// ```
    /// use nalgebra::Perspective3;
    ///
    /// // Very small near plane can cause z-fighting
    /// let mut projection = Perspective3::new(1.0_f32, 1.0, 0.001, 10000.0);
    ///
    /// // Increase near plane for better depth precision
    /// projection.set_znear(0.1);
    /// assert!((projection.znear() - 0.1).abs() < 1e-6);
    /// println!("Near plane adjusted to: {}", projection.znear());
    /// ```
    ///
    /// # See Also
    ///
    /// - [`Perspective3::znear`]: Get the current near plane
    /// - [`Perspective3::set_zfar`]: Change the far plane
    /// - [`Perspective3::set_znear_and_zfar`]: Change both planes efficiently
    #[inline]
    pub fn set_znear(&mut self, znear: T) {
        let zfar = self.zfar();
        self.set_znear_and_zfar(znear, zfar);
    }

    /// Changes the far clipping plane distance of this perspective projection.
    ///
    /// Objects farther than this distance won't be rendered. Adjusting the far plane lets
    /// you control how far the camera can see, which is useful for fog effects, performance
    /// optimization, or rendering large scenes.
    ///
    /// Note: Changing the far plane also requires retrieving and setting the near plane,
    /// which involves some computation. If you need to change both planes, use
    /// [`set_znear_and_zfar`](Self::set_znear_and_zfar) instead for better performance.
    ///
    /// # Parameters
    ///
    /// - `zfar`: The new far plane distance (must be greater than near plane)
    ///
    /// # Example: Adjusting View Distance
    ///
    /// ```
    /// use nalgebra::Perspective3;
    ///
    /// let mut projection = Perspective3::new(16.0_f32 / 9.0, 1.0, 0.1, 100.0);
    ///
    /// // Increase view distance for outdoor scenes
    /// projection.set_zfar(1000.0);
    /// assert!((projection.zfar() - 1000.0).abs() < 1.0);
    ///
    /// // Decrease for indoor scenes
    /// projection.set_zfar(50.0);
    /// assert!((projection.zfar() - 50.0).abs() < 1e-3);
    /// ```
    ///
    /// # Example: Dynamic Render Distance
    ///
    /// ```
    /// use nalgebra::Perspective3;
    ///
    /// let mut projection = Perspective3::new(1.0_f32, 1.0, 0.1, 100.0);
    ///
    /// // Adjust based on environment
    /// let indoor_distance = 50.0;
    /// let outdoor_distance = 500.0;
    /// let space_distance = 100000.0;
    ///
    /// // In a building
    /// projection.set_zfar(indoor_distance);
    /// assert!((projection.zfar() - indoor_distance).abs() < 1e-3);
    ///
    /// // Outside
    /// projection.set_zfar(outdoor_distance);
    /// assert!((projection.zfar() - outdoor_distance).abs() < 1.0);
    /// ```
    ///
    /// # See Also
    ///
    /// - [`Perspective3::zfar`]: Get the current far plane
    /// - [`Perspective3::set_znear`]: Change the near plane
    /// - [`Perspective3::set_znear_and_zfar`]: Change both planes efficiently
    #[inline]
    pub fn set_zfar(&mut self, zfar: T) {
        let znear = self.znear();
        self.set_znear_and_zfar(znear, zfar);
    }

    /// Changes both the near and far clipping plane distances simultaneously.
    ///
    /// This is more efficient than calling [`set_znear`](Self::set_znear) and
    /// [`set_zfar`](Self::set_zfar) separately, as it only updates the matrix once.
    /// Use this when you need to change both planes at the same time.
    ///
    /// # Parameters
    ///
    /// - `znear`: The new near plane distance (must be positive and less than `zfar`)
    /// - `zfar`: The new far plane distance (must be greater than `znear`)
    ///
    /// # Example: Switching Between Scene Types
    ///
    /// ```
    /// use nalgebra::Perspective3;
    ///
    /// let mut projection = Perspective3::new(16.0_f32 / 9.0, 1.0, 0.1, 100.0);
    ///
    /// // Configure for indoor scene
    /// projection.set_znear_and_zfar(0.1, 50.0);
    /// assert!((projection.znear() - 0.1).abs() < 1e-5);
    /// assert!((projection.zfar() - 50.0).abs() < 1e-3);
    ///
    /// // Switch to outdoor scene
    /// projection.set_znear_and_zfar(1.0, 1000.0);
    /// assert!((projection.znear() - 1.0).abs() < 1e-5);
    /// assert!((projection.zfar() - 1000.0).abs() < 1e-3);
    /// ```
    ///
    /// # Example: Adjusting Depth Range
    ///
    /// ```
    /// use nalgebra::Perspective3;
    ///
    /// let mut projection = Perspective3::new(1.0, 1.0, 0.1, 100.0);
    ///
    /// // Very close range (microscope view)
    /// projection.set_znear_and_zfar(0.001, 1.0);
    ///
    /// // Normal range (human scale)
    /// projection.set_znear_and_zfar(0.1, 100.0);
    ///
    /// // Very far range (planetary scale)
    /// projection.set_znear_and_zfar(100.0, 1000000.0);
    ///
    /// println!("Final range: {} to {}", projection.znear(), projection.zfar());
    /// ```
    ///
    /// # Example: Performance-Optimized Updates
    ///
    /// ```
    /// use nalgebra::Perspective3;
    ///
    /// let mut projection = Perspective3::new(16.0 / 9.0, 1.0, 0.1, 100.0);
    ///
    /// // Efficient: Update both at once
    /// projection.set_znear_and_zfar(0.5, 500.0);
    ///
    /// // Less efficient: Would call set_znear_and_zfar twice internally
    /// // projection.set_znear(0.5);
    /// // projection.set_zfar(500.0);
    /// ```
    ///
    /// # See Also
    ///
    /// - [`Perspective3::set_znear`]: Change only the near plane
    /// - [`Perspective3::set_zfar`]: Change only the far plane
    /// - [`Perspective3::znear`]: Get the current near plane
    /// - [`Perspective3::zfar`]: Get the current far plane
    #[inline]
    pub fn set_znear_and_zfar(&mut self, znear: T, zfar: T) {
        self.matrix[(2, 2)] = (zfar.clone() + znear.clone()) / (znear.clone() - zfar.clone());
        self.matrix[(2, 3)] = zfar.clone() * znear.clone() * crate::convert(2.0) / (znear - zfar);
    }
}

#[cfg(feature = "rand-no-std")]
impl<T: RealField> Distribution<Perspective3<T>> for StandardUniform
where
    StandardUniform: Distribution<T>,
{
    /// Generate an arbitrary random variate for testing purposes.
    fn sample<R: Rng + ?Sized>(&self, r: &mut R) -> Perspective3<T> {
        use crate::base::helper;
        let znear = r.random();
        let zfar = helper::reject_rand(r, |x: &T| !(x.clone() - znear.clone()).is_zero());
        let aspect = helper::reject_rand(r, |x: &T| !x.is_zero());

        Perspective3::new(aspect, r.random(), znear, zfar)
    }
}

#[cfg(feature = "arbitrary")]
impl<T: RealField + Arbitrary> Arbitrary for Perspective3<T> {
    fn arbitrary(g: &mut Gen) -> Self {
        use crate::base::helper;
        let znear: T = Arbitrary::arbitrary(g);
        let zfar = helper::reject(g, |x: &T| !(x.clone() - znear.clone()).is_zero());
        let aspect = helper::reject(g, |x: &T| !x.is_zero());

        Self::new(aspect, Arbitrary::arbitrary(g), znear, zfar)
    }
}

impl<T: RealField> From<Perspective3<T>> for Matrix4<T> {
    #[inline]
    fn from(pers: Perspective3<T>) -> Self {
        pers.into_inner()
    }
}
