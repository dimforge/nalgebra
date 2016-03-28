use std::fmt;
use std::ops::{Add, Sub, Mul, Neg};

use rand::{Rand, Rng};
use num::One;
use structs::mat::{Mat3, Mat4};
use traits::structure::{Cast, Dim, Col, BaseFloat, BaseNum};
use traits::operations::{Inv, ApproxEq};
use traits::geometry::{RotationMatrix, Rotation, Rotate, AbsoluteRotate, Transform, Transformation,
                       Translate, Translation, ToHomogeneous};
use structs::vec::{Vec1, Vec2, Vec3};
use structs::pnt::{Pnt2, Pnt3};
use structs::rot::{Rot2, Rot3};

#[cfg(feature="arbitrary")]
use quickcheck::{Arbitrary, Gen};


/// Two dimensional isometry.
///
/// This is the composition of a rotation followed by a translation. Vectors `Vec2` are not
/// affected by the translational component of this transformation while points `Pnt2` are.
/// Isometries conserve angles and distances, hence do not allow shearing nor scaling.
#[repr(C)]
#[derive(Eq, PartialEq, RustcEncodable, RustcDecodable, Clone, Debug, Copy)]
pub struct Iso2<N> {
    /// The rotation applicable by this isometry.
    pub rotation:    Rot2<N>,
    /// The translation applicable by this isometry.
    pub translation: Vec2<N>
}

/// Three dimensional isometry.
///
/// This is the composition of a rotation followed by a translation. Vectors `Vec3` are not
/// affected by the translational component of this transformation while points `Pnt3` are.
/// Isometries conserve angles and distances, hence do not allow shearing nor scaling.
#[repr(C)]
#[derive(Eq, PartialEq, RustcEncodable, RustcDecodable, Clone, Debug, Copy)]
pub struct Iso3<N> {
    /// The rotation applicable by this isometry.
    pub rotation:    Rot3<N>,
    /// The translation applicable by this isometry.
    pub translation: Vec3<N>
}

impl<N: Clone + BaseFloat> Iso3<N> {
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
    pub fn new_observer_frame(eye: &Pnt3<N>, target: &Pnt3<N>, up: &Vec3<N>) -> Iso3<N> {
        let new_rotmat = Rot3::new_observer_frame(&(*target - *eye), up);
        Iso3::new_with_rotmat(eye.as_vec().clone(), new_rotmat)
    }

    /// Builds a look-at view matrix.
    ///
    /// This conforms to the common notion of "look-at" matrix from the computer graphics
    /// community. Its maps the view direction `target - eye` to the **negative** `z` axis and the
    /// `eye` to the origin.
    ///
    /// # Arguments
    ///   * eye - The eye position.
    ///   * target - The target position.
    ///   * up - The vertical view direction. It must not be to collinear to `eye - target`.
    #[inline]
    pub fn new_look_at(eye: &Pnt3<N>, target: &Pnt3<N>, up: &Vec3<N>) -> Iso3<N> {
        let new_rotmat = Rot3::new_look_at(&(*target - *eye), up);
        Iso3::new_with_rotmat(new_rotmat * (-*eye.as_vec()), new_rotmat)
    }
}

iso_impl!(Iso2, Rot2, Vec2, Vec1);
rotation_matrix_impl!(Iso2, Rot2, Vec2, Vec1);
rotation_impl!(Iso2, Rot2, Vec1);
dim_impl!(Iso2, 2);
one_impl!(Iso2);
absolute_rotate_impl!(Iso2, Vec2);
rand_impl!(Iso2);
approx_eq_impl!(Iso2);
to_homogeneous_impl!(Iso2, Mat3);
inv_impl!(Iso2);
transform_impl!(Iso2, Pnt2);
transformation_impl!(Iso2);
rotate_impl!(Iso2, Vec2);
translation_impl!(Iso2, Vec2);
translate_impl!(Iso2, Pnt2);
iso_mul_iso_impl!(Iso2);
iso_mul_pnt_impl!(Iso2, Pnt2);
pnt_mul_iso_impl!(Iso2, Pnt2);
iso_mul_vec_impl!(Iso2, Vec2);
vec_mul_iso_impl!(Iso2, Vec2);
arbitrary_iso_impl!(Iso2);
iso_display_impl!(Iso2);

iso_impl!(Iso3, Rot3, Vec3, Vec3);
rotation_matrix_impl!(Iso3, Rot3, Vec3, Vec3);
rotation_impl!(Iso3, Rot3, Vec3);
dim_impl!(Iso3, 3);
one_impl!(Iso3);
absolute_rotate_impl!(Iso3, Vec3);
rand_impl!(Iso3);
approx_eq_impl!(Iso3);
to_homogeneous_impl!(Iso3, Mat4);
inv_impl!(Iso3);
transform_impl!(Iso3, Pnt3);
transformation_impl!(Iso3);
rotate_impl!(Iso3, Vec3);
translation_impl!(Iso3, Vec3);
translate_impl!(Iso3, Pnt3);
iso_mul_iso_impl!(Iso3);
iso_mul_pnt_impl!(Iso3, Pnt3);
pnt_mul_iso_impl!(Iso3, Pnt3);
iso_mul_vec_impl!(Iso3, Vec3);
vec_mul_iso_impl!(Iso3, Vec3);
arbitrary_iso_impl!(Iso3);
iso_display_impl!(Iso3);
