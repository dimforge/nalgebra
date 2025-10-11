// Needed otherwise the rkyv macros generate code incompatible with rust-2024
#![cfg_attr(feature = "rkyv-serialize", allow(unsafe_op_in_unsafe_fn))]

use approx::{AbsDiffEq, RelativeEq, UlpsEq};
use num::{One, Zero};
use std::fmt;
use std::hash;

#[cfg(feature = "serde-serialize-no-std")]
use serde::{Deserialize, Deserializer, Serialize, Serializer};

#[cfg(feature = "serde-serialize-no-std")]
use crate::base::storage::Owned;

use simba::scalar::RealField;
use simba::simd::SimdRealField;

use crate::base::allocator::Allocator;
use crate::base::dimension::{DimNameAdd, DimNameSum, U1};
use crate::base::{Const, DefaultAllocator, OMatrix, SMatrix, SVector, Scalar, Unit};
use crate::geometry::Point;

#[cfg(feature = "rkyv-serialize")]
use rkyv::bytecheck;

/// A rotation matrix.
///
/// This is also known as an element of a Special Orthogonal (SO) group.
/// The `Rotation` type can either represent a 2D or 3D rotation, represented as a matrix.
/// For a rotation based on quaternions, see [`UnitQuaternion`](crate::UnitQuaternion) instead.
///
/// Note that instead of using the [`Rotation`] type in your code directly, you should use one
/// of its aliases: [`Rotation2`](crate::Rotation2), or [`Rotation3`](crate::Rotation3). Though
/// keep in mind that all the documentation of all the methods of these aliases will also appears on
/// this page.
///
/// # Construction
/// * [Identity <span style="float:right;">`identity`</span>](#identity)
/// * [From a 2D rotation angle <span style="float:right;">`new`…</span>](#construction-from-a-2d-rotation-angle)
/// * [From an existing 2D matrix or rotations <span style="float:right;">`from_matrix`, `rotation_between`, `powf`…</span>](#construction-from-an-existing-2d-matrix-or-rotations)
/// * [From a 3D axis and/or angles <span style="float:right;">`new`, `from_euler_angles`, `from_axis_angle`…</span>](#construction-from-a-3d-axis-andor-angles)
/// * [From a 3D eye position and target point <span style="float:right;">`look_at`, `look_at_lh`, `rotation_between`…</span>](#construction-from-a-3d-eye-position-and-target-point)
/// * [From an existing 3D matrix or rotations <span style="float:right;">`from_matrix`, `rotation_between`, `powf`…</span>](#construction-from-an-existing-3d-matrix-or-rotations)
///
/// # Transformation and composition
/// Note that transforming vectors and points can be done by multiplication, e.g., `rotation * point`.
/// Composing an rotation with another transformation can also be done by multiplication or division.
/// * [3D axis and angle extraction <span style="float:right;">`angle`, `euler_angles`, `scaled_axis`, `angle_to`…</span>](#3d-axis-and-angle-extraction)
/// * [2D angle extraction <span style="float:right;">`angle`, `angle_to`…</span>](#2d-angle-extraction)
/// * [Transformation of a vector or a point <span style="float:right;">`transform_vector`, `inverse_transform_point`…</span>](#transformation-of-a-vector-or-a-point)
/// * [Transposition and inversion <span style="float:right;">`transpose`, `inverse`…</span>](#transposition-and-inversion)
/// * [Interpolation <span style="float:right;">`slerp`…</span>](#interpolation)
///
/// # Conversion
/// * [Conversion to a matrix <span style="float:right;">`matrix`, `to_homogeneous`…</span>](#conversion-to-a-matrix)
///
#[repr(C)]
#[cfg_attr(
    feature = "rkyv-serialize-no-std",
    derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize),
    archive(
        as = "Rotation<T::Archived, D>",
        bound(archive = "
        T: rkyv::Archive,
        SMatrix<T, D, D>: rkyv::Archive<Archived = SMatrix<T::Archived, D, D>>
    ")
    )
)]
#[cfg_attr(feature = "rkyv-serialize", derive(bytecheck::CheckBytes))]
#[cfg_attr(feature = "defmt", derive(defmt::Format))]
#[derive(Copy, Clone)]
pub struct Rotation<T, const D: usize> {
    matrix: SMatrix<T, D, D>,
}

impl<T: fmt::Debug, const D: usize> fmt::Debug for Rotation<T, D> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        self.matrix.fmt(formatter)
    }
}

impl<T: Scalar + hash::Hash, const D: usize> hash::Hash for Rotation<T, D>
where
    <DefaultAllocator as Allocator<Const<D>, Const<D>>>::Buffer<T>: hash::Hash,
{
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        self.matrix.hash(state)
    }
}

#[cfg(feature = "bytemuck")]
unsafe impl<T, const D: usize> bytemuck::Zeroable for Rotation<T, D>
where
    T: Scalar + bytemuck::Zeroable,
    SMatrix<T, D, D>: bytemuck::Zeroable,
{
}

#[cfg(feature = "bytemuck")]
unsafe impl<T, const D: usize> bytemuck::Pod for Rotation<T, D>
where
    T: Scalar + bytemuck::Pod,
    SMatrix<T, D, D>: bytemuck::Pod,
{
}

#[cfg(feature = "serde-serialize-no-std")]
impl<T: Scalar, const D: usize> Serialize for Rotation<T, D>
where
    Owned<T, Const<D>, Const<D>>: Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.matrix.serialize(serializer)
    }
}

#[cfg(feature = "serde-serialize-no-std")]
impl<'a, T: Scalar, const D: usize> Deserialize<'a> for Rotation<T, D>
where
    Owned<T, Const<D>, Const<D>>: Deserialize<'a>,
{
    fn deserialize<Des>(deserializer: Des) -> Result<Self, Des::Error>
    where
        Des: Deserializer<'a>,
    {
        let matrix = SMatrix::<T, D, D>::deserialize(deserializer)?;

        Ok(Self::from_matrix_unchecked(matrix))
    }
}

impl<T, const D: usize> Rotation<T, D> {
    /// Creates a new rotation from the given square matrix.
    ///
    /// The matrix orthonormality is not checked.
    ///
    /// # Example
    /// ```
    /// # use nalgebra::{Rotation2, Rotation3, Matrix2, Matrix3};
    /// # use std::f32;
    /// let mat = Matrix3::new(0.8660254, -0.5,      0.0,
    ///                        0.5,       0.8660254, 0.0,
    ///                        0.0,       0.0,       1.0);
    /// let rot = Rotation3::from_matrix_unchecked(mat);
    ///
    /// assert_eq!(*rot.matrix(), mat);
    ///
    ///
    /// let mat = Matrix2::new(0.8660254, -0.5,
    ///                        0.5,       0.8660254);
    /// let rot = Rotation2::from_matrix_unchecked(mat);
    ///
    /// assert_eq!(*rot.matrix(), mat);
    /// ```
    #[inline]
    pub const fn from_matrix_unchecked(matrix: SMatrix<T, D, D>) -> Self {
        Self { matrix }
    }
}

/// # Conversion to a matrix
impl<T: Scalar, const D: usize> Rotation<T, D> {
    /// Returns a reference to the underlying matrix representation of this rotation.
    ///
    /// A rotation matrix is a special orthogonal matrix that represents a rotation transformation.
    /// In 2D, this is a 2×2 matrix, and in 3D, it's a 3×3 matrix. These matrices have the property
    /// that their transpose equals their inverse, and their determinant is 1.
    ///
    /// This method provides direct access to the internal matrix representation without copying,
    /// which is useful for:
    /// - Inspecting individual matrix elements
    /// - Passing to APIs that expect matrix types
    /// - Low-level optimizations where you need the raw matrix data
    ///
    /// # Examples
    ///
    /// ## Basic 3D rotation inspection
    /// ```
    /// # use nalgebra::{Rotation3, Vector3, Matrix3};
    /// # use std::f32;
    /// // Create a 30-degree rotation around the Z axis
    /// let rot = Rotation3::from_axis_angle(&Vector3::z_axis(), f32::consts::FRAC_PI_6);
    ///
    /// // Access the underlying matrix
    /// let matrix = rot.matrix();
    ///
    /// // The matrix should be approximately:
    /// // [cos(30°), -sin(30°), 0]
    /// // [sin(30°),  cos(30°), 0]
    /// // [0,         0,        1]
    /// let expected = Matrix3::new(0.8660254, -0.5,      0.0,
    ///                             0.5,       0.8660254, 0.0,
    ///                             0.0,       0.0,       1.0);
    /// assert_eq!(*matrix, expected);
    /// ```
    ///
    /// ## 2D rotation for game development
    /// ```
    /// # use nalgebra::{Rotation2, Matrix2};
    /// # use std::f32;
    /// // Rotate 30 degrees counterclockwise (common in 2D games)
    /// let rot = Rotation2::new(f32::consts::FRAC_PI_6);
    ///
    /// // Get the rotation matrix for custom rendering code
    /// let matrix = rot.matrix();
    ///
    /// let expected = Matrix2::new(0.8660254, -0.5,
    ///                             0.5,       0.8660254);
    /// assert_eq!(*matrix, expected);
    /// ```
    ///
    /// ## Using matrix elements in robotics
    /// ```
    /// # use nalgebra::{Rotation3, Vector3};
    /// # use std::f32;
    /// let rot = Rotation3::from_axis_angle(&Vector3::x_axis(), f32::consts::FRAC_PI_4);
    /// let matrix = rot.matrix();
    ///
    /// // Access individual elements for sensor fusion algorithms
    /// let element_00 = matrix[(0, 0)];
    /// let element_12 = matrix[(1, 2)];
    ///
    /// // Elements can be used in custom computations
    /// assert!((element_00 - 1.0).abs() < 1e-6); // First element should be 1.0
    /// ```
    ///
    /// # See Also
    /// - [`into_inner`](Self::into_inner) - Consumes the rotation and returns the owned matrix
    /// - [`to_homogeneous`](Self::to_homogeneous) - Converts to a homogeneous transformation matrix
    #[inline]
    #[must_use]
    pub const fn matrix(&self) -> &SMatrix<T, D, D> {
        &self.matrix
    }

    /// A mutable reference to the underlying matrix representation of this rotation.
    ///
    /// # Safety
    ///
    /// Invariants of the rotation matrix should not be violated.
    #[inline]
    #[deprecated(note = "Use `.matrix_mut_unchecked()` instead.")]
    pub const unsafe fn matrix_mut(&mut self) -> &mut SMatrix<T, D, D> {
        &mut self.matrix
    }

    /// A mutable reference to the underlying matrix representation of this rotation.
    ///
    /// This is suffixed by "_unchecked" because this allows the user to replace the
    /// matrix by another one that is non-inversible or non-orthonormal. If one of
    /// those properties is broken, subsequent method calls may return bogus results.
    #[inline]
    pub const fn matrix_mut_unchecked(&mut self) -> &mut SMatrix<T, D, D> {
        &mut self.matrix
    }

    /// Consumes this rotation and returns the underlying matrix.
    ///
    /// This method takes ownership of the rotation and returns the internal matrix representation.
    /// Unlike [`matrix()`](Self::matrix) which returns a reference, this method moves the matrix
    /// out of the rotation, which can be more efficient when you no longer need the rotation wrapper.
    ///
    /// Use this when:
    /// - You need an owned matrix rather than a reference
    /// - You're done with the rotation and want to extract its matrix
    /// - You need to pass the matrix to a function that takes ownership
    ///
    /// # Examples
    ///
    /// ## Converting 3D rotation to matrix for rendering
    /// ```
    /// # use nalgebra::{Rotation3, Vector3, Matrix3};
    /// # use std::f32;
    /// // Create a rotation for a camera orientation
    /// let rot = Rotation3::from_axis_angle(&Vector3::z_axis(), f32::consts::FRAC_PI_6);
    ///
    /// // Extract the matrix to send to a GPU shader
    /// let mat = rot.into_inner();
    ///
    /// let expected = Matrix3::new(0.8660254, -0.5,      0.0,
    ///                             0.5,       0.8660254, 0.0,
    ///                             0.0,       0.0,       1.0);
    /// assert_eq!(mat, expected);
    ///
    /// // Note: rot is no longer accessible here (it was moved)
    /// ```
    ///
    /// ## 2D game sprite rotation
    /// ```
    /// # use nalgebra::{Rotation2, Matrix2};
    /// # use std::f32;
    /// // Calculate rotation for a spinning game object
    /// let angle = f32::consts::FRAC_PI_6; // 30 degrees
    /// let rot = Rotation2::new(angle);
    ///
    /// // Extract matrix for the graphics engine
    /// let transform_matrix = rot.into_inner();
    ///
    /// let expected = Matrix2::new(0.8660254, -0.5,
    ///                             0.5,       0.8660254);
    /// assert_eq!(transform_matrix, expected);
    /// ```
    ///
    /// ## Robotics: Exporting orientation data
    /// ```
    /// # use nalgebra::{Rotation3, Vector3};
    /// # use std::f32;
    /// // Robot arm joint rotation
    /// let joint_rotation = Rotation3::from_axis_angle(
    ///     &Vector3::y_axis(),
    ///     f32::consts::FRAC_PI_4
    /// );
    ///
    /// // Convert to matrix for data logging or network transmission
    /// let rotation_matrix = joint_rotation.into_inner();
    ///
    /// // Now rotation_matrix can be serialized or processed further
    /// assert_eq!(rotation_matrix.nrows(), 3);
    /// assert_eq!(rotation_matrix.ncols(), 3);
    /// ```
    ///
    /// # See Also
    /// - [`matrix`](Self::matrix) - Returns a reference to the matrix without consuming the rotation
    /// - [`to_homogeneous`](Self::to_homogeneous) - Converts to a homogeneous transformation matrix
    #[inline]
    pub fn into_inner(self) -> SMatrix<T, D, D> {
        self.matrix
    }

    /// Unwraps the underlying matrix.
    /// Deprecated: Use [`Rotation::into_inner`] instead.
    #[deprecated(note = "use `.into_inner()` instead")]
    #[inline]
    pub fn unwrap(self) -> SMatrix<T, D, D> {
        self.matrix
    }

    /// Converts this rotation into its equivalent homogeneous transformation matrix.
    ///
    /// A homogeneous transformation matrix is one dimension larger than the rotation matrix,
    /// allowing rotations and translations to be combined in a single matrix. This is
    /// fundamental in computer graphics and robotics for representing complex transformations.
    ///
    /// For a 2D rotation, this produces a 3×3 matrix. For a 3D rotation, this produces a 4×4 matrix.
    /// The extra row and column are set to create an identity transformation in the homogeneous
    /// coordinate (the last row is [0, ..., 0, 1] and the last column is [0, ..., 0, 1]).
    ///
    /// This is equivalent to calling `self.into()` which uses the `Into` trait.
    ///
    /// # Examples
    ///
    /// ## 3D graphics transformation pipeline
    /// ```
    /// # use nalgebra::{Rotation3, Vector3, Matrix4, Point3};
    /// # use std::f32;
    /// // Create a rotation for a 3D model
    /// let rot = Rotation3::from_axis_angle(&Vector3::z_axis(), f32::consts::FRAC_PI_6);
    ///
    /// // Convert to homogeneous matrix for GPU
    /// let homogeneous = rot.to_homogeneous();
    ///
    /// // This is now a 4×4 matrix suitable for graphics pipelines
    /// let expected = Matrix4::new(
    ///     0.8660254, -0.5,      0.0, 0.0,
    ///     0.5,       0.8660254, 0.0, 0.0,
    ///     0.0,       0.0,       1.0, 0.0,
    ///     0.0,       0.0,       0.0, 1.0
    /// );
    /// assert_eq!(homogeneous, expected);
    /// ```
    ///
    /// ## 2D game transformations
    /// ```
    /// # use nalgebra::{Rotation2, Matrix3, Point2};
    /// # use std::f32;
    /// // Rotate a 2D sprite
    /// let rot = Rotation2::new(f32::consts::FRAC_PI_6); // 30 degrees
    ///
    /// // Convert to homogeneous form for combining with translation
    /// let homogeneous = rot.to_homogeneous();
    ///
    /// // This creates a 3×3 matrix
    /// let expected = Matrix3::new(
    ///     0.8660254, -0.5,      0.0,
    ///     0.5,       0.8660254, 0.0,
    ///     0.0,       0.0,       1.0
    /// );
    /// assert_eq!(homogeneous, expected);
    /// ```
    ///
    /// ## Robotics: Combining rotation with translation
    /// ```
    /// # use nalgebra::{Rotation3, Vector3, Matrix4, Translation3, Isometry3};
    /// # use std::f32;
    /// // Robot end-effector orientation
    /// let rotation = Rotation3::from_axis_angle(&Vector3::y_axis(), f32::consts::FRAC_PI_4);
    ///
    /// // Convert to homogeneous matrix
    /// let rot_homogeneous = rotation.to_homogeneous();
    ///
    /// // Can now be combined with translation matrices for full pose representation
    /// // The homogeneous form allows matrix multiplication to combine transformations
    /// assert_eq!(rot_homogeneous.nrows(), 4);
    /// assert_eq!(rot_homogeneous.ncols(), 4);
    ///
    /// // Last row is always [0, 0, 0, 1] for a pure rotation
    /// assert_eq!(rot_homogeneous[(3, 0)], 0.0);
    /// assert_eq!(rot_homogeneous[(3, 1)], 0.0);
    /// assert_eq!(rot_homogeneous[(3, 2)], 0.0);
    /// assert_eq!(rot_homogeneous[(3, 3)], 1.0);
    /// ```
    ///
    /// # See Also
    /// - [`matrix`](Self::matrix) - Get the underlying rotation matrix (non-homogeneous)
    /// - [`into_inner`](Self::into_inner) - Extract the rotation matrix
    #[inline]
    #[must_use]
    pub fn to_homogeneous(&self) -> OMatrix<T, DimNameSum<Const<D>, U1>, DimNameSum<Const<D>, U1>>
    where
        T: Zero + One,
        Const<D>: DimNameAdd<U1>,
        DefaultAllocator: Allocator<DimNameSum<Const<D>, U1>, DimNameSum<Const<D>, U1>>,
    {
        // We could use `SMatrix::to_homogeneous()` here, but that would imply
        // adding the additional traits `DimAdd` and `IsNotStaticOne`. Maybe
        // these things will get nicer once specialization lands in Rust.
        let mut res = OMatrix::<T, DimNameSum<Const<D>, U1>, DimNameSum<Const<D>, U1>>::identity();
        res.fixed_view_mut::<D, D>(0, 0).copy_from(&self.matrix);

        res
    }
}

/// # Transposition and inversion
impl<T: Scalar, const D: usize> Rotation<T, D> {
    /// Returns the transpose of this rotation matrix.
    ///
    /// For rotation matrices, the transpose is equal to the inverse. This is a special property
    /// of orthogonal matrices: R^T = R^-1. This makes computing the inverse of a rotation very
    /// efficient compared to general matrix inversion.
    ///
    /// Transposing a rotation matrix gives you the opposite rotation. If a rotation `R` rotates
    /// from frame A to frame B, then `R.transpose()` rotates from frame B back to frame A.
    ///
    /// This method returns a new rotation; use [`transpose_mut`](Self::transpose_mut) to modify
    /// the rotation in place.
    ///
    /// # Mathematical Note
    ///
    /// For a rotation matrix R:
    /// - R * R^T = I (identity matrix)
    /// - R^T * R = I (identity matrix)
    /// - This is equivalent to calling `.inverse()`
    ///
    /// # Examples
    ///
    /// ## 3D rotation reversal
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{Rotation3, Vector3, Point3};
    /// // Rotate 90 degrees around Y axis
    /// let rot = Rotation3::new(Vector3::new(1.0, 2.0, 3.0));
    ///
    /// // Get the reverse rotation
    /// let rot_reversed = rot.transpose();
    ///
    /// // Applying both rotations returns to identity
    /// assert_relative_eq!(rot * rot_reversed, Rotation3::identity(), epsilon = 1.0e-6);
    /// assert_relative_eq!(rot_reversed * rot, Rotation3::identity(), epsilon = 1.0e-6);
    /// ```
    ///
    /// ## 2D game object rotation undo
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{Rotation2, Point2};
    /// // Rotate sprite 68.75 degrees
    /// let rot = Rotation2::new(1.2);
    ///
    /// // Transpose gives the opposite rotation
    /// let rot_back = rot.transpose();
    ///
    /// // Together they cancel out
    /// assert_relative_eq!(rot * rot_back, Rotation2::identity(), epsilon = 1.0e-6);
    /// assert_relative_eq!(rot_back * rot, Rotation2::identity(), epsilon = 1.0e-6);
    /// ```
    ///
    /// ## Robotics: Coordinate frame transformation
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{Rotation3, Vector3, Point3};
    /// # use std::f32;
    /// // Rotation from robot base to camera frame
    /// let base_to_camera = Rotation3::from_axis_angle(
    ///     &Vector3::z_axis(),
    ///     f32::consts::FRAC_PI_2
    /// );
    ///
    /// // Transpose gives camera to base transformation
    /// let camera_to_base = base_to_camera.transpose();
    ///
    /// // Transform a point from camera frame back to base frame
    /// let point_in_camera = Point3::new(1.0, 0.0, 0.0);
    /// let point_in_base = camera_to_base * point_in_camera;
    ///
    /// // Verify round-trip transformation
    /// let round_trip = base_to_camera * point_in_base;
    /// assert_relative_eq!(round_trip, point_in_camera, epsilon = 1.0e-6);
    /// ```
    ///
    /// # See Also
    /// - [`inverse`](Self::inverse) - Identical to transpose for rotation matrices
    /// - [`transpose_mut`](Self::transpose_mut) - In-place version of this operation
    /// - [`inverse_mut`](Self::inverse_mut) - In-place version of inverse (same as transpose_mut)
    #[inline]
    #[must_use = "Did you mean to use transpose_mut()?"]
    pub fn transpose(&self) -> Self {
        Self::from_matrix_unchecked(self.matrix.transpose())
    }

    /// Returns the inverse of this rotation.
    ///
    /// For rotation matrices, the inverse is equal to the transpose. This is a fundamental property
    /// of orthogonal matrices and makes inverting rotations very efficient (just swap rows and columns).
    ///
    /// The inverse rotation "undoes" the original rotation. If you apply a rotation and then its inverse,
    /// you get back to where you started. This is crucial for:
    /// - Converting between coordinate frames
    /// - Undoing transformations
    /// - Solving for unknown rotations in equations
    ///
    /// This method returns a new rotation; use [`inverse_mut`](Self::inverse_mut) to modify
    /// the rotation in place.
    ///
    /// # Mathematical Properties
    ///
    /// For any rotation R:
    /// - R * R^-1 = I (identity)
    /// - R^-1 * R = I (identity)
    /// - R^-1 = R^T (inverse equals transpose)
    /// - (R^-1)^-1 = R (inverse of inverse is original)
    ///
    /// # Examples
    ///
    /// ## 3D camera orientation
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{Rotation3, Vector3, Point3};
    /// # use std::f32;
    /// // Camera rotation
    /// let camera_rot = Rotation3::new(Vector3::new(1.0, 2.0, 3.0));
    ///
    /// // To transform world coordinates to camera space, use inverse
    /// let world_to_camera = camera_rot.inverse();
    ///
    /// // Verify they cancel out
    /// assert_relative_eq!(camera_rot * world_to_camera, Rotation3::identity(), epsilon = 1.0e-6);
    /// assert_relative_eq!(world_to_camera * camera_rot, Rotation3::identity(), epsilon = 1.0e-6);
    /// ```
    ///
    /// ## 2D game object rotation
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{Rotation2, Point2};
    /// // Rotate character 68.75 degrees
    /// let character_rotation = Rotation2::new(1.2);
    ///
    /// // Need to undo rotation? Use inverse
    /// let undo_rotation = character_rotation.inverse();
    ///
    /// // Applying rotation then inverse returns to original orientation
    /// assert_relative_eq!(character_rotation * undo_rotation, Rotation2::identity(), epsilon = 1.0e-6);
    /// assert_relative_eq!(undo_rotation * character_rotation, Rotation2::identity(), epsilon = 1.0e-6);
    /// ```
    ///
    /// ## Robotics: Forward and inverse kinematics
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{Rotation3, Vector3, Point3};
    /// # use std::f32;
    /// // Joint rotation in forward kinematics
    /// let joint_rotation = Rotation3::from_axis_angle(
    ///     &Vector3::y_axis(),
    ///     f32::consts::FRAC_PI_4
    /// );
    ///
    /// // For inverse kinematics, we need the opposite transformation
    /// let inverse_joint_rotation = joint_rotation.inverse();
    ///
    /// // Transform a point and back again
    /// let point = Point3::new(1.0, 2.0, 3.0);
    /// let transformed = joint_rotation * point;
    /// let original = inverse_joint_rotation * transformed;
    ///
    /// assert_relative_eq!(original, point, epsilon = 1.0e-6);
    /// ```
    ///
    /// # See Also
    /// - [`transpose`](Self::transpose) - Identical to inverse for rotation matrices
    /// - [`inverse_mut`](Self::inverse_mut) - In-place version of this operation
    /// - [`transpose_mut`](Self::transpose_mut) - In-place version of transpose (same as inverse_mut)
    #[inline]
    #[must_use = "Did you mean to use inverse_mut()?"]
    pub fn inverse(&self) -> Self {
        self.transpose()
    }

    /// Transposes this rotation in-place.
    ///
    /// This modifies the rotation to its transpose without allocating a new rotation.
    /// For rotation matrices, transposing gives the inverse rotation, so this effectively
    /// reverses the rotation in place.
    ///
    /// This is more efficient than creating a new transposed rotation when you don't need
    /// to keep the original. Use [`transpose`](Self::transpose) if you need both the original
    /// and transposed rotations.
    ///
    /// # Examples
    ///
    /// ## 3D rotation reversal
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{Rotation3, Vector3};
    /// let rot = Rotation3::new(Vector3::new(1.0, 2.0, 3.0));
    /// let mut reversed_rot = Rotation3::new(Vector3::new(1.0, 2.0, 3.0));
    /// reversed_rot.transpose_mut();
    ///
    /// // Now reversed_rot is the inverse of rot
    /// assert_relative_eq!(rot * reversed_rot, Rotation3::identity(), epsilon = 1.0e-6);
    /// assert_relative_eq!(reversed_rot * rot, Rotation3::identity(), epsilon = 1.0e-6);
    /// ```
    ///
    /// ## 2D efficient rotation update
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::Rotation2;
    /// let rot = Rotation2::new(1.2);
    /// let mut working_rot = Rotation2::new(1.2);
    ///
    /// // Modify in-place to save allocation
    /// working_rot.transpose_mut();
    ///
    /// assert_relative_eq!(rot * working_rot, Rotation2::identity(), epsilon = 1.0e-6);
    /// assert_relative_eq!(working_rot * rot, Rotation2::identity(), epsilon = 1.0e-6);
    /// ```
    ///
    /// # See Also
    /// - [`transpose`](Self::transpose) - Returns a new transposed rotation
    /// - [`inverse_mut`](Self::inverse_mut) - Identical operation (inverse in-place)
    #[inline]
    pub fn transpose_mut(&mut self) {
        self.matrix.transpose_mut()
    }

    /// Inverts this rotation in-place.
    ///
    /// This modifies the rotation to its inverse without allocating a new rotation.
    /// For rotation matrices, the inverse equals the transpose, so this is computationally
    /// very efficient (just swaps rows and columns).
    ///
    /// This is more efficient than creating a new inverse rotation when you don't need
    /// to keep the original. Use [`inverse`](Self::inverse) if you need both rotations.
    ///
    /// # Examples
    ///
    /// ## 3D transformation chain
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{Rotation3, Vector3, Point3};
    /// let rot = Rotation3::new(Vector3::new(1.0, 2.0, 3.0));
    /// let mut inverse_rot = Rotation3::new(Vector3::new(1.0, 2.0, 3.0));
    ///
    /// // Invert in-place to use in transformation pipeline
    /// inverse_rot.inverse_mut();
    ///
    /// assert_relative_eq!(rot * inverse_rot, Rotation3::identity(), epsilon = 1.0e-6);
    /// assert_relative_eq!(inverse_rot * rot, Rotation3::identity(), epsilon = 1.0e-6);
    /// ```
    ///
    /// ## 2D game state update
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::Rotation2;
    /// let rot = Rotation2::new(1.2);
    /// let mut undo_rotation = Rotation2::new(1.2);
    ///
    /// // Efficiently create the inverse rotation
    /// undo_rotation.inverse_mut();
    ///
    /// assert_relative_eq!(rot * undo_rotation, Rotation2::identity(), epsilon = 1.0e-6);
    /// assert_relative_eq!(undo_rotation * rot, Rotation2::identity(), epsilon = 1.0e-6);
    /// ```
    ///
    /// # See Also
    /// - [`inverse`](Self::inverse) - Returns a new inverse rotation
    /// - [`transpose_mut`](Self::transpose_mut) - Identical operation (transpose in-place)
    #[inline]
    pub fn inverse_mut(&mut self) {
        self.transpose_mut()
    }
}

/// # Transformation of a vector or a point
impl<T: SimdRealField, const D: usize> Rotation<T, D>
where
    T::Element: SimdRealField,
{
    /// Rotates the given point by this rotation.
    ///
    /// This applies the rotation transformation to a point in space. Points represent positions,
    /// and rotating a point means changing its position around the origin (or rotation center).
    ///
    /// This method is equivalent to the multiplication `self * pt`, but can be more readable
    /// in some contexts. The transformation is performed by matrix-vector multiplication.
    ///
    /// # Parameters
    ///
    /// * `pt` - The point to rotate
    ///
    /// # Returns
    ///
    /// A new point representing the rotated position
    ///
    /// # Examples
    ///
    /// ## 3D point rotation for game physics
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use std::f32;
    /// # use nalgebra::{Point3, Rotation3, Vector3};
    /// // Rotate 90 degrees around Y axis
    /// let rot = Rotation3::new(Vector3::y() * f32::consts::FRAC_PI_2);
    ///
    /// // Rotate a point in 3D space
    /// let point = Point3::new(1.0, 2.0, 3.0);
    /// let rotated = rot.transform_point(&point);
    ///
    /// // X becomes Z, Z becomes -X, Y stays the same
    /// assert_relative_eq!(rotated, Point3::new(3.0, 2.0, -1.0), epsilon = 1.0e-6);
    /// ```
    ///
    /// ## 2D sprite position rotation
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use std::f32;
    /// # use nalgebra::{Point2, Rotation2};
    /// // Rotate 90 degrees counterclockwise
    /// let rot = Rotation2::new(f32::consts::FRAC_PI_2);
    ///
    /// // Rotate a 2D point (e.g., sprite position)
    /// let point = Point2::new(3.0, 0.0);
    /// let rotated = rot.transform_point(&point);
    ///
    /// // Point rotates from +X axis to +Y axis
    /// assert_relative_eq!(rotated, Point2::new(0.0, 3.0), epsilon = 1.0e-6);
    /// ```
    ///
    /// ## Robotics: Sensor data transformation
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use std::f32;
    /// # use nalgebra::{Point3, Rotation3, Vector3};
    /// // Robot sensor mounted at 45 degrees
    /// let sensor_rotation = Rotation3::from_axis_angle(
    ///     &Vector3::z_axis(),
    ///     f32::consts::FRAC_PI_4
    /// );
    ///
    /// // Detected object position in sensor frame
    /// let object_in_sensor_frame = Point3::new(1.0, 0.0, 0.0);
    ///
    /// // Transform to robot base frame
    /// let object_in_base_frame = sensor_rotation.transform_point(&object_in_sensor_frame);
    ///
    /// // The point is now rotated 45 degrees
    /// assert_relative_eq!(
    ///     object_in_base_frame,
    ///     Point3::new(0.7071068, 0.7071068, 0.0),
    ///     epsilon = 1.0e-6
    /// );
    /// ```
    ///
    /// # See Also
    /// - [`transform_vector`](Self::transform_vector) - Rotate a vector (direction) instead of a point
    /// - [`inverse_transform_point`](Self::inverse_transform_point) - Apply the inverse rotation to a point
    #[inline]
    #[must_use]
    pub fn transform_point(&self, pt: &Point<T, D>) -> Point<T, D> {
        self * pt
    }

    /// Rotates the given vector by this rotation.
    ///
    /// This applies the rotation transformation to a vector (direction). Unlike points,
    /// vectors represent directions and magnitudes, not positions. Rotating a vector
    /// changes its direction while preserving its length.
    ///
    /// This method is equivalent to the multiplication `self * v`, but can be more readable
    /// in some contexts. The transformation is performed by matrix-vector multiplication.
    ///
    /// # Important Distinction
    ///
    /// - Use `transform_vector` for directions, velocities, forces, normals
    /// - Use `transform_point` for positions, locations
    /// - Both use the same underlying matrix multiplication for rotations
    ///
    /// # Parameters
    ///
    /// * `v` - The vector to rotate
    ///
    /// # Returns
    ///
    /// A new vector with the same magnitude but rotated direction
    ///
    /// # Examples
    ///
    /// ## 3D velocity vector rotation
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use std::f32;
    /// # use nalgebra::{Rotation3, Vector3};
    /// // Rotate 90 degrees around Y axis
    /// let rot = Rotation3::new(Vector3::y() * f32::consts::FRAC_PI_2);
    ///
    /// // Rotate a velocity vector
    /// let velocity = Vector3::new(1.0, 2.0, 3.0);
    /// let rotated_velocity = rot.transform_vector(&velocity);
    ///
    /// // Direction changed, magnitude preserved
    /// assert_relative_eq!(rotated_velocity, Vector3::new(3.0, 2.0, -1.0), epsilon = 1.0e-6);
    /// assert_relative_eq!(rotated_velocity.magnitude(), velocity.magnitude(), epsilon = 1.0e-6);
    /// ```
    ///
    /// ## 2D force vector in game physics
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use std::f32;
    /// # use nalgebra::{Rotation2, Vector2};
    /// // Rotate 45 degrees
    /// let rot = Rotation2::new(f32::consts::FRAC_PI_4);
    ///
    /// // Force pointing right
    /// let force = Vector2::new(10.0, 0.0);
    /// let rotated_force = rot.transform_vector(&force);
    ///
    /// // Force now points at 45 degrees, same magnitude
    /// assert_relative_eq!(rotated_force, Vector2::new(7.071068, 7.071068), epsilon = 1.0e-5);
    /// assert_relative_eq!(rotated_force.magnitude(), 10.0, epsilon = 1.0e-6);
    /// ```
    ///
    /// ## Robotics: Surface normal transformation
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use std::f32;
    /// # use nalgebra::{Rotation3, Vector3};
    /// // Robot gripper rotation
    /// let gripper_rotation = Rotation3::from_axis_angle(
    ///     &Vector3::x_axis(),
    ///     f32::consts::FRAC_PI_2
    /// );
    ///
    /// // Surface normal in gripper frame (pointing up)
    /// let normal_in_gripper = Vector3::new(0.0, 1.0, 0.0);
    ///
    /// // Transform normal to base frame
    /// let normal_in_base = gripper_rotation.transform_vector(&normal_in_gripper);
    ///
    /// // Normal is now pointing forward in base frame
    /// assert_relative_eq!(normal_in_base, Vector3::new(0.0, 0.0, 1.0), epsilon = 1.0e-6);
    /// ```
    ///
    /// # See Also
    /// - [`transform_point`](Self::transform_point) - Rotate a point (position) instead of a vector
    /// - [`inverse_transform_vector`](Self::inverse_transform_vector) - Apply the inverse rotation to a vector
    #[inline]
    #[must_use]
    pub fn transform_vector(&self, v: &SVector<T, D>) -> SVector<T, D> {
        self * v
    }

    /// Rotates the given point by the inverse of this rotation.
    ///
    /// This applies the reverse rotation to a point, which is more efficient than
    /// explicitly computing the inverse rotation and then transforming the point.
    /// It uses the fact that for rotation matrices, R^T * v = R^-1 * v.
    ///
    /// This is commonly used when:
    /// - Converting from world space to local space
    /// - Undoing a previous rotation
    /// - Working with inverse kinematics
    ///
    /// # Performance Note
    ///
    /// This is more efficient than calling `rot.inverse().transform_point(pt)` because
    /// it avoids creating a new rotation object and uses a single transpose-multiply operation.
    ///
    /// # Parameters
    ///
    /// * `pt` - The point to rotate by the inverse rotation
    ///
    /// # Returns
    ///
    /// A new point rotated in the opposite direction
    ///
    /// # Examples
    ///
    /// ## 3D world-to-local transformation
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use std::f32;
    /// # use nalgebra::{Point3, Rotation3, Vector3};
    /// // Object rotation in world space
    /// let rot = Rotation3::new(Vector3::y() * f32::consts::FRAC_PI_2);
    ///
    /// // Convert world point to object's local space
    /// let world_point = Point3::new(1.0, 2.0, 3.0);
    /// let local_point = rot.inverse_transform_point(&world_point);
    ///
    /// assert_relative_eq!(local_point, Point3::new(-3.0, 2.0, 1.0), epsilon = 1.0e-6);
    /// ```
    ///
    /// ## 2D camera space conversion
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use std::f32;
    /// # use nalgebra::{Point2, Rotation2};
    /// // Camera rotation
    /// let camera_rot = Rotation2::new(f32::consts::FRAC_PI_4);
    ///
    /// // Convert screen point to camera-local coordinates
    /// let screen_point = Point2::new(5.0, 5.0);
    /// let camera_local = camera_rot.inverse_transform_point(&screen_point);
    ///
    /// // Verify round-trip
    /// let back_to_screen = camera_rot.transform_point(&camera_local);
    /// assert_relative_eq!(back_to_screen, screen_point, epsilon = 1.0e-6);
    /// ```
    ///
    /// ## Robotics: Sensor to world coordinates
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use std::f32;
    /// # use nalgebra::{Point3, Rotation3, Vector3};
    /// // Sensor mounted with rotation
    /// let sensor_rotation = Rotation3::from_axis_angle(
    ///     &Vector3::z_axis(),
    ///     f32::consts::FRAC_PI_2
    /// );
    ///
    /// // Point detected in world frame
    /// let world_point = Point3::new(0.0, 1.0, 0.0);
    ///
    /// // Convert to sensor's local frame for processing
    /// let sensor_point = sensor_rotation.inverse_transform_point(&world_point);
    ///
    /// // In sensor frame, the point appears rotated -90 degrees around Z
    /// assert_relative_eq!(sensor_point, Point3::new(1.0, 0.0, 0.0), epsilon = 1.0e-6);
    /// ```
    ///
    /// # See Also
    /// - [`transform_point`](Self::transform_point) - Apply the forward rotation
    /// - [`inverse_transform_vector`](Self::inverse_transform_vector) - Apply inverse rotation to a vector
    /// - [`inverse`](Self::inverse) - Get the inverse rotation itself
    #[inline]
    #[must_use]
    pub fn inverse_transform_point(&self, pt: &Point<T, D>) -> Point<T, D> {
        Point::from(self.inverse_transform_vector(&pt.coords))
    }

    /// Rotates the given vector by the inverse of this rotation.
    ///
    /// This applies the reverse rotation to a vector, which is more efficient than
    /// explicitly computing the inverse rotation and then transforming the vector.
    /// It uses transpose-multiply (R^T * v) which is equivalent to inverse-multiply (R^-1 * v)
    /// for rotation matrices.
    ///
    /// This is commonly used when:
    /// - Converting directions from world space to local space
    /// - Reversing a previous rotation on a direction vector
    /// - Working with coordinate frame transformations
    ///
    /// # Performance Note
    ///
    /// This is more efficient than calling `rot.inverse().transform_vector(v)` because
    /// it uses a specialized transpose-multiply operation.
    ///
    /// # Parameters
    ///
    /// * `v` - The vector to rotate by the inverse rotation
    ///
    /// # Returns
    ///
    /// A new vector rotated in the opposite direction (same magnitude)
    ///
    /// # Examples
    ///
    /// ## 3D velocity transformation
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use std::f32;
    /// # use nalgebra::{Rotation3, Vector3};
    /// // Object rotation
    /// let rot = Rotation3::new(Vector3::y() * f32::consts::FRAC_PI_2);
    ///
    /// // Convert world velocity to object's local frame
    /// let world_velocity = Vector3::new(1.0, 2.0, 3.0);
    /// let local_velocity = rot.inverse_transform_vector(&world_velocity);
    ///
    /// assert_relative_eq!(local_velocity, Vector3::new(-3.0, 2.0, 1.0), epsilon = 1.0e-6);
    /// // Magnitude is preserved
    /// assert_relative_eq!(local_velocity.magnitude(), world_velocity.magnitude(), epsilon = 1.0e-6);
    /// ```
    ///
    /// ## 2D force vector conversion
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use std::f32;
    /// # use nalgebra::{Rotation2, Vector2};
    /// // Character rotation in game
    /// let rotation = Rotation2::new(f32::consts::FRAC_PI_4);
    ///
    /// // Force applied in world frame
    /// let world_force = Vector2::new(10.0, 0.0);
    ///
    /// // Convert to character's local frame for physics simulation
    /// let local_force = rotation.inverse_transform_vector(&world_force);
    ///
    /// // Verify round-trip
    /// let back_to_world = rotation.transform_vector(&local_force);
    /// assert_relative_eq!(back_to_world, world_force, epsilon = 1.0e-6);
    /// ```
    ///
    /// ## Robotics: Coordinate frame conversion
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use std::f32;
    /// # use nalgebra::{Rotation3, Vector3};
    /// // End-effector rotation
    /// let effector_rot = Rotation3::from_axis_angle(
    ///     &Vector3::x_axis(),
    ///     f32::consts::FRAC_PI_2
    /// );
    ///
    /// // Gravity vector in world frame
    /// let gravity_world = Vector3::new(0.0, 0.0, -9.81);
    ///
    /// // Convert to end-effector's frame for control calculations
    /// let gravity_local = effector_rot.inverse_transform_vector(&gravity_world);
    ///
    /// assert_relative_eq!(gravity_local, Vector3::new(0.0, 9.81, 0.0), epsilon = 1.0e-5);
    /// ```
    ///
    /// # See Also
    /// - [`transform_vector`](Self::transform_vector) - Apply the forward rotation
    /// - [`inverse_transform_point`](Self::inverse_transform_point) - Apply inverse rotation to a point
    /// - [`inverse_transform_unit_vector`](Self::inverse_transform_unit_vector) - Specialized version for unit vectors
    #[inline]
    #[must_use]
    pub fn inverse_transform_vector(&self, v: &SVector<T, D>) -> SVector<T, D> {
        self.matrix().tr_mul(v)
    }

    /// Rotates the given unit vector by the inverse of this rotation.
    ///
    /// This is a specialized version of [`inverse_transform_vector`](Self::inverse_transform_vector)
    /// for unit vectors (vectors with magnitude 1). Since rotation preserves vector length,
    /// the result is guaranteed to also be a unit vector, so we can skip normalization.
    ///
    /// Unit vectors are commonly used to represent:
    /// - Direction axes (X, Y, Z axes)
    /// - Surface normals
    /// - Orientation directions
    ///
    /// # Performance Note
    ///
    /// This avoids the normalization step that would be needed if you transformed a regular
    /// vector and then normalized it, making it more efficient for unit vectors.
    ///
    /// # Parameters
    ///
    /// * `v` - The unit vector to rotate by the inverse rotation
    ///
    /// # Returns
    ///
    /// A new unit vector rotated in the opposite direction
    ///
    /// # Examples
    ///
    /// ## 3D axis transformation
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use std::f32;
    /// # use nalgebra::{Rotation3, Vector3};
    /// // Rotate 90 degrees around Z
    /// let rot = Rotation3::new(Vector3::z() * f32::consts::FRAC_PI_2);
    ///
    /// // Transform the X axis by the inverse rotation
    /// let transformed_x = rot.inverse_transform_unit_vector(&Vector3::x_axis());
    ///
    /// // X axis becomes -Y axis
    /// assert_relative_eq!(transformed_x, -Vector3::y_axis(), epsilon = 1.0e-6);
    /// ```
    ///
    /// ## 2D direction vector
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use std::f32;
    /// # use nalgebra::{Rotation2, Vector2, Unit};
    /// // 45 degree rotation
    /// let rot = Rotation2::new(f32::consts::FRAC_PI_4);
    ///
    /// // Unit vector pointing right
    /// let right = Unit::new_normalize(Vector2::new(1.0, 0.0));
    ///
    /// // Rotate by inverse (effectively -45 degrees)
    /// let rotated = rot.inverse_transform_unit_vector(&right);
    ///
    /// // Result is still a unit vector
    /// assert_relative_eq!(rotated.magnitude(), 1.0, epsilon = 1.0e-6);
    /// ```
    ///
    /// ## Robotics: Normal vector transformation
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use std::f32;
    /// # use nalgebra::{Rotation3, Vector3};
    /// // Camera rotation
    /// let camera_rot = Rotation3::from_axis_angle(
    ///     &Vector3::y_axis(),
    ///     f32::consts::FRAC_PI_2
    /// );
    ///
    /// // Surface normal in world frame
    /// let world_normal = Vector3::z_axis();
    ///
    /// // Transform to camera frame for lighting calculations
    /// let camera_normal = camera_rot.inverse_transform_unit_vector(&world_normal);
    ///
    /// // Normal is now pointing in camera's local coordinates
    /// assert_relative_eq!(camera_normal, -Vector3::x_axis(), epsilon = 1.0e-6);
    /// ```
    ///
    /// # See Also
    /// - [`inverse_transform_vector`](Self::inverse_transform_vector) - General version for any vector
    /// - [`transform_vector`](Self::transform_vector) - Apply the forward rotation
    #[inline]
    #[must_use]
    pub fn inverse_transform_unit_vector(&self, v: &Unit<SVector<T, D>>) -> Unit<SVector<T, D>> {
        Unit::new_unchecked(self.inverse_transform_vector(&**v))
    }
}

impl<T: Scalar + Eq, const D: usize> Eq for Rotation<T, D> {}

impl<T: Scalar + PartialEq, const D: usize> PartialEq for Rotation<T, D> {
    #[inline]
    fn eq(&self, right: &Self) -> bool {
        self.matrix == right.matrix
    }
}

impl<T, const D: usize> AbsDiffEq for Rotation<T, D>
where
    T: Scalar + AbsDiffEq,
    T::Epsilon: Clone,
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

impl<T, const D: usize> RelativeEq for Rotation<T, D>
where
    T: Scalar + RelativeEq,
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
        self.matrix
            .relative_eq(&other.matrix, epsilon, max_relative)
    }
}

impl<T, const D: usize> UlpsEq for Rotation<T, D>
where
    T: Scalar + UlpsEq,
    T::Epsilon: Clone,
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

/*
 *
 * Display
 *
 */
impl<T, const D: usize> fmt::Display for Rotation<T, D>
where
    T: RealField + fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let precision = f.precision().unwrap_or(3);

        writeln!(f, "Rotation matrix {{")?;
        write!(f, "{:.*}", precision, self.matrix)?;
        writeln!(f, "}}")
    }
}

//          //         /*
//          //          *
//          //          * Absolute
//          //          *
//          //          */
//          //         impl<T: Absolute> Absolute for $t<T> {
//          //             type AbsoluteValue = $submatrix<T::AbsoluteValue>;
//          //
//          //             #[inline]
//          //             fn abs(m: &$t<T>) -> $submatrix<T::AbsoluteValue> {
//          //                 Absolute::abs(&m.submatrix)
//          //             }
//          //         }
