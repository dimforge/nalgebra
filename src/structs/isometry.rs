use std::fmt;
use std::ops::{Add, Sub, Mul, Neg, MulAssign};

use rand::{Rand, Rng};
use num::One;
use structs::matrix::{Matrix3, Matrix4};
use traits::structure::{Cast, Dimension, Column, BaseFloat, BaseNum};
use traits::operations::{Inverse, ApproxEq};
use traits::geometry::{RotationMatrix, Rotation, Rotate, AbsoluteRotate, Transform, Transformation,
                       Translate, Translation, ToHomogeneous};
use structs::vector::{Vector1, Vector2, Vector3};
use structs::point::{Point2, Point3};
use structs::rotation::{Rotation2, Rotation3};

#[cfg(feature="arbitrary")]
use quickcheck::{Arbitrary, Gen};


/// Two dimensional **direct** isometry.
///
/// This is the composition of a rotation followed by a translation. Vectors `Vector2` are not
/// affected by the translational component of this transformation while points `Point2` are.
/// Isometrymetries conserve angles and distances, hence do not allow shearing nor scaling.
#[repr(C)]
#[derive(Eq, PartialEq, RustcEncodable, RustcDecodable, Clone, Debug, Copy)]
pub struct Isometry2<N> {
    /// The rotation applicable by this isometry.
    pub rotation:    Rotation2<N>,
    /// The translation applicable by this isometry.
    pub translation: Vector2<N>
}

/// Three dimensional **direct** isometry.
///
/// This is the composition of a rotation followed by a translation. Vectors `Vector3` are not
/// affected by the translational component of this transformation while points `Point3` are.
/// Isometrymetries conserve angles and distances, hence do not allow shearing nor scaling.
#[repr(C)]
#[derive(Eq, PartialEq, RustcEncodable, RustcDecodable, Clone, Debug, Copy)]
pub struct Isometry3<N> {
    /// The rotation applicable by this isometry.
    pub rotation:    Rotation3<N>,
    /// The translation applicable by this isometry.
    pub translation: Vector3<N>
}

impl<N: Clone + BaseFloat> Isometry3<N> {
    /// Creates an isometry that corresponds to the local frame of an observer standing at the
    /// point `eye` and looking toward `target`.
    ///
    /// It maps the view direction `target - eye` to the positive `z` axis and the origin to the
    /// `eye`.
    ///
    /// # Arguments
    ///   * eye - The observer position.
    ///   * target - The target position.
    ///   * up - Vertical direction. The only requirement of this parameter is to not be collinear
    ///   to `eye - at`. Non-collinearity is not checked.
    #[inline]
    pub fn new_observer_frame(eye: &Point3<N>, target: &Point3<N>, up: &Vector3<N>) -> Isometry3<N> {
        let new_rotation_matrix = Rotation3::new_observer_frame(&(*target - *eye), up);
        Isometry3::new_with_rotation_matrix(eye.as_vector().clone(), new_rotation_matrix)
    }

    /// Builds a right-handed look-at view matrix.
    ///
    /// This conforms to the common notion of right handed look-at matrix from the computer
    /// graphics community.
    ///
    /// # Arguments
    ///   * eye - The eye position.
    ///   * target - The target position.
    ///   * up - A vector approximately aligned with required the vertical axis. The only
    ///   requirement of this parameter is to not be collinear to `target - eye`.
    #[inline]
    pub fn look_at_rh(eye: &Point3<N>, target: &Point3<N>, up: &Vector3<N>) -> Isometry3<N> {
        let rotation   = Rotation3::look_at_rh(&(*target - *eye), up);
        let trans = rotation * (-*eye);

        Isometry3::new_with_rotation_matrix(trans.to_vector(), rotation)
    }

    /// Builds a left-handed look-at view matrix.
    ///
    /// This conforms to the common notion of left handed look-at matrix from the computer
    /// graphics community.
    ///
    /// # Arguments
    ///   * eye - The eye position.
    ///   * target - The target position.
    ///   * up - A vector approximately aligned with required the vertical axis. The only
    ///   requirement of this parameter is to not be collinear to `target - eye`.
    #[inline]
    pub fn look_at_lh(eye: &Point3<N>, target: &Point3<N>, up: &Vector3<N>) -> Isometry3<N> {
        let rotation   = Rotation3::look_at_lh(&(*target - *eye), up);
        let trans = rotation * (-*eye);

        Isometry3::new_with_rotation_matrix(trans.to_vector(), rotation)
    }
}

isometry_impl!(Isometry2, Rotation2, Vector2, Vector1);
rotation_matrix_impl!(Isometry2, Rotation2, Vector2, Vector1);
rotation_impl!(Isometry2, Rotation2, Vector1);
dim_impl!(Isometry2, 2);
one_impl!(Isometry2);
absolute_rotate_impl!(Isometry2, Vector2);
rand_impl!(Isometry2);
approx_eq_impl!(Isometry2);
to_homogeneous_impl!(Isometry2, Matrix3);
inverse_impl!(Isometry2);
transform_impl!(Isometry2, Point2);
transformation_impl!(Isometry2);
rotate_impl!(Isometry2, Vector2);
translation_impl!(Isometry2, Vector2);
translate_impl!(Isometry2, Point2);
isometry_mul_isometry_impl!(Isometry2);
isometry_mul_rotation_impl!(Isometry2, Rotation2);
isometry_mul_point_impl!(Isometry2, Point2);
isometry_mul_vec_impl!(Isometry2, Vector2);
arbitrary_isometry_impl!(Isometry2);
isometry_display_impl!(Isometry2);

isometry_impl!(Isometry3, Rotation3, Vector3, Vector3);
rotation_matrix_impl!(Isometry3, Rotation3, Vector3, Vector3);
rotation_impl!(Isometry3, Rotation3, Vector3);
dim_impl!(Isometry3, 3);
one_impl!(Isometry3);
absolute_rotate_impl!(Isometry3, Vector3);
rand_impl!(Isometry3);
approx_eq_impl!(Isometry3);
to_homogeneous_impl!(Isometry3, Matrix4);
inverse_impl!(Isometry3);
transform_impl!(Isometry3, Point3);
transformation_impl!(Isometry3);
rotate_impl!(Isometry3, Vector3);
translation_impl!(Isometry3, Vector3);
translate_impl!(Isometry3, Point3);
isometry_mul_isometry_impl!(Isometry3);
isometry_mul_rotation_impl!(Isometry3, Rotation3);
isometry_mul_point_impl!(Isometry3, Point3);
isometry_mul_vec_impl!(Isometry3, Vector3);
arbitrary_isometry_impl!(Isometry3);
isometry_display_impl!(Isometry3);
