//! Isometric transformations.

#![allow(missing_doc)]

use std::num::{Zero, One};
use rand::{Rand, Rng};
use structs::mat::{Mat3, Mat4, Mat5};
use traits::structure::{Cast, Dim, Col};
use traits::operations::{Inv, ApproxEq};
use traits::geometry::{RotationMatrix, Rotation, Rotate, AbsoluteRotate, Transform, Transformation,
                       Translate, Translation, ToHomogeneous};

use structs::vec::{Vec1, Vec2, Vec3, Vec4, Vec2MulRhs, Vec3MulRhs, Vec4MulRhs};
use structs::rot::{Rot2, Rot3, Rot4};


/// Two dimensional isometry.
///
/// This is the composition of a rotation followed by a translation.
/// Isometries conserve angles and distances, hence do not allow shearing nor scaling.
#[deriving(TotalEq, Eq, Encodable, Decodable, Clone, Show)]
pub struct Iso2<N> {
    /// The rotation applicable by this isometry.
    pub rotation:    Rot2<N>,
    /// The translation applicable by this isometry.
    pub translation: Vec2<N>
}

/// Three dimensional isometry.
///
/// This is the composition of a rotation followed by a translation.
/// Isometries conserve angles and distances, hence do not allow shearing nor scaling.
#[deriving(TotalEq, Eq, Encodable, Decodable, Clone, Show)]
pub struct Iso3<N> {
    /// The rotation applicable by this isometry.
    pub rotation:    Rot3<N>,
    /// The translation applicable by this isometry.
    pub translation: Vec3<N>
}

/// Four dimensional isometry.
///
/// Isometries conserve angles and distances, hence do not allow shearing nor scaling.
#[deriving(TotalEq, Eq, Encodable, Decodable, Clone, Show)]
pub struct Iso4<N> {
    /// The rotation applicable by this isometry.
    pub rotation:    Rot4<N>,
    /// The translation applicable by this isometry.
    pub translation: Vec4<N>
}

impl<N: Clone + Float> Iso3<N> {
    /// Reorient and translate this transformation such that its local `x` axis points to a given
    /// direction.  Note that the usually known `look_at` function does the same thing but with the
    /// `z` axis. See `look_at_z` for that.
    ///
    /// # Arguments
    ///   * eye - The new translation of the transformation.
    ///   * at - The point to look at. `at - eye` is the direction the matrix `x` axis will be
    ///   aligned with.
    ///   * up - Vector pointing up. The only requirement of this parameter is to not be colinear
    ///   with `at`. Non-colinearity is not checked.
    pub fn look_at(&mut self, eye: &Vec3<N>, at: &Vec3<N>, up: &Vec3<N>) {
        self.rotation.look_at(&(*at - *eye), up);
        self.translation = eye.clone();
    }

    /// Reorient and translate this transformation such that its local `z` axis points to a given
    /// direction.
    ///
    /// # Arguments
    ///   * eye - The new translation of the transformation.
    ///   * at - The point to look at. `at - eye` is the direction the matrix `x` axis will be
    ///   aligned with
    ///   * up - Vector pointing `up`. The only requirement of this parameter is to not be colinear
    ///   with `at`. Non-colinearity is not checked.
    pub fn look_at_z(&mut self, eye: &Vec3<N>, at: &Vec3<N>, up: &Vec3<N>) {
        self.rotation.look_at_z(&(*at - *eye), up);
        self.translation = eye.clone();
    }
}

impl<N> Iso4<N> {
    // XXX remove that when iso_impl works for Iso4
    /// Creates a new isometry from a rotation matrix and a vector.
    #[inline]
    pub fn new_with_rotmat(translation: Vec4<N>, rotation: Rot4<N>) -> Iso4<N> {
        Iso4 {
            rotation:    rotation,
            translation: translation
        }
    }
}

iso_impl!(Iso2, Rot2, Vec2, Vec1)
double_dispatch_binop_decl_trait!(Iso2, Iso2MulRhs)
mul_redispatch_impl!(Iso2, Iso2MulRhs)
rotation_matrix_impl!(Iso2, Rot2, Vec2, Vec1)
rotation_impl!(Iso2, Rot2, Vec1)
dim_impl!(Iso2, 2)
one_impl!(Iso2)
absolute_rotate_impl!(Iso2, Vec2)
rand_impl!(Iso2)
approx_eq_impl!(Iso2)
to_homogeneous_impl!(Iso2, Mat3)
inv_impl!(Iso2)
transform_impl!(Iso2, Vec2)
transformation_impl!(Iso2)
rotate_impl!(Iso2, Vec2)
translation_impl!(Iso2, Vec2)
translate_impl!(Iso2, Vec2)
iso_mul_iso_impl!(Iso2, Iso2MulRhs)
iso_mul_vec_impl!(Iso2, Vec2, Iso2MulRhs)
vec_mul_iso_impl!(Iso2, Vec2, Vec2MulRhs)

iso_impl!(Iso3, Rot3, Vec3, Vec3)
double_dispatch_binop_decl_trait!(Iso3, Iso3MulRhs)
mul_redispatch_impl!(Iso3, Iso3MulRhs)
rotation_matrix_impl!(Iso3, Rot3, Vec3, Vec3)
rotation_impl!(Iso3, Rot3, Vec3)
dim_impl!(Iso3, 3)
one_impl!(Iso3)
absolute_rotate_impl!(Iso3, Vec3)
rand_impl!(Iso3)
approx_eq_impl!(Iso3)
to_homogeneous_impl!(Iso3, Mat4)
inv_impl!(Iso3)
transform_impl!(Iso3, Vec3)
transformation_impl!(Iso3)
rotate_impl!(Iso3, Vec3)
translation_impl!(Iso3, Vec3)
translate_impl!(Iso3, Vec3)
iso_mul_iso_impl!(Iso3, Iso3MulRhs)
iso_mul_vec_impl!(Iso3, Vec3, Iso3MulRhs)
vec_mul_iso_impl!(Iso3, Vec3, Vec3MulRhs)

// iso_impl!(Iso4, Rot4, Vec4, Vec4)
double_dispatch_binop_decl_trait!(Iso4, Iso4MulRhs)
mul_redispatch_impl!(Iso4, Iso4MulRhs)
// rotation_matrix_impl!(Iso4, Rot4, Vec4, Vec4)
// rotation_impl!(Iso4, Rot4, Vec4)
dim_impl!(Iso4, 4)
one_impl!(Iso4)
absolute_rotate_impl!(Iso4, Vec4)
// rand_impl!(Iso4)
approx_eq_impl!(Iso4)
to_homogeneous_impl!(Iso4, Mat5)
inv_impl!(Iso4)
transform_impl!(Iso4, Vec4)
transformation_impl!(Iso4)
rotate_impl!(Iso4, Vec4)
translation_impl!(Iso4, Vec4)
translate_impl!(Iso4, Vec4)
iso_mul_iso_impl!(Iso4, Iso4MulRhs)
iso_mul_vec_impl!(Iso4, Vec4, Iso4MulRhs)
vec_mul_iso_impl!(Iso4, Vec4, Vec4MulRhs)
