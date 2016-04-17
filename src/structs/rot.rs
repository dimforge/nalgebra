//! Rotations matrices.

use std::fmt;
use std::ops::{Mul, Neg, MulAssign, Index};
use rand::{Rand, Rng};
use num::{Zero, One};
use traits::geometry::{Rotate, Rotation, AbsoluteRotate, RotationMatrix, RotationTo, Transform,
                       ToHomogeneous, Norm, Cross};
use traits::structure::{Cast, Dim, Row, Col, BaseFloat, BaseNum, Eye, Diag};
use traits::operations::{Absolute, Inv, Transpose, ApproxEq};
use structs::vec::{Vec1, Vec2, Vec3};
use structs::pnt::{Pnt2, Pnt3};
use structs::mat::{Mat2, Mat3, Mat4};
#[cfg(feature="arbitrary")]
use quickcheck::{Arbitrary, Gen};


/// Two dimensional rotation matrix.
#[repr(C)]
#[derive(Eq, PartialEq, RustcEncodable, RustcDecodable, Clone, Debug, Hash, Copy)]
pub struct Rot2<N> {
    submat: Mat2<N>
}

impl<N: Clone + BaseFloat + Neg<Output = N>> Rot2<N> {
    /// Builds a 2 dimensional rotation matrix from an angle in radian.
    pub fn new(angle: Vec1<N>) -> Rot2<N> {
        let (sia, coa) = angle.x.sin_cos();

        Rot2 {
            submat: Mat2::new(coa.clone(), -sia, sia, coa)
        }
    }
}

impl<N: BaseFloat + Clone> Rotation<Vec1<N>> for Rot2<N> {
    #[inline]
    fn rotation(&self) -> Vec1<N> {
        Vec1::new((-self.submat.m12).atan2(self.submat.m11.clone()))
    }

    #[inline]
    fn inv_rotation(&self) -> Vec1<N> {
        -self.rotation()
    }

    #[inline]
    fn append_rotation_mut(&mut self, rot: &Vec1<N>) {
        *self = Rotation::append_rotation(self, rot)
    }

    #[inline]
    fn append_rotation(&self, rot: &Vec1<N>) -> Rot2<N> {
        Rot2::new(rot.clone()) * *self
    }

    #[inline]
    fn prepend_rotation_mut(&mut self, rot: &Vec1<N>) {
        *self = Rotation::prepend_rotation(self, rot)
    }

    #[inline]
    fn prepend_rotation(&self, rot: &Vec1<N>) -> Rot2<N> {
        *self * Rot2::new(rot.clone())
    }

    #[inline]
    fn set_rotation(&mut self, rot: Vec1<N>) {
        *self = Rot2::new(rot)
    }
}

impl<N: BaseFloat> RotationTo for Rot2<N> {
    type AngleType = N;
    type DeltaRotationType = Rot2<N>;

    #[inline]
    fn angle_to(&self, other: &Self) -> N {
        self.rotation_to(other).rotation().norm()
    }

    #[inline]
    fn rotation_to(&self, other: &Self) -> Rot2<N> {
        *other * ::inv(self).unwrap()
    }
}

impl<N: Rand + BaseFloat> Rand for Rot2<N> {
    #[inline]
    fn rand<R: Rng>(rng: &mut R) -> Rot2<N> {
        Rot2::new(rng.gen())
    }
}

impl<N: BaseFloat> AbsoluteRotate<Vec2<N>> for Rot2<N> {
    #[inline]
    fn absolute_rotate(&self, v: &Vec2<N>) -> Vec2<N> {
        // the matrix is skew-symetric, so we dont need to compute the absolute value of every
        // component.
        let m11 = ::abs(&self.submat.m11);
        let m12 = ::abs(&self.submat.m12);
        let m22 = ::abs(&self.submat.m22);

        Vec2::new(m11 * v.x + m12 * v.y, m12 * v.x + m22 * v.y)
    }
}

#[cfg(feature="arbitrary")]
impl<N: Arbitrary + Clone + BaseFloat + Neg<Output = N>> Arbitrary for Rot2<N> {
    fn arbitrary<G: Gen>(g: &mut G) -> Rot2<N> {
        Rot2::new(Arbitrary::arbitrary(g))
    }
}


/*
 * 3d rotation
 */
/// Three dimensional rotation matrix.
#[repr(C)]
#[derive(Eq, PartialEq, RustcEncodable, RustcDecodable, Clone, Debug, Hash, Copy)]
pub struct Rot3<N> {
    submat: Mat3<N>
}


impl<N: Clone + BaseFloat> Rot3<N> {
    /// Builds a 3 dimensional rotation matrix from an axis and an angle.
    ///
    /// # Arguments
    ///   * `axisangle` - A vector representing the rotation. Its magnitude is the amount of rotation
    ///   in radian. Its direction is the axis of rotation.
    pub fn new(axisangle: Vec3<N>) -> Rot3<N> {
        if ::is_zero(&Norm::sqnorm(&axisangle)) {
            ::one()
        }
        else {
            let mut axis   = axisangle;
            let angle      = axis.normalize_mut();
            let _1: N      = ::one();
            let ux         = axis.x.clone();
            let uy         = axis.y.clone();
            let uz         = axis.z.clone();
            let sqx        = ux * ux;
            let sqy        = uy * uy;
            let sqz        = uz * uz;
            let (sin, cos) = angle.sin_cos();
            let one_m_cos  = _1 - cos;

            Rot3 {
                submat: Mat3::new(
                            (sqx + (_1 - sqx) * cos),
                            (ux * uy * one_m_cos - uz * sin),
                            (ux * uz * one_m_cos + uy * sin),

                            (ux * uy * one_m_cos + uz * sin),
                            (sqy + (_1 - sqy) * cos),
                            (uy * uz * one_m_cos - ux * sin),

                            (ux * uz * one_m_cos - uy * sin),
                            (uy * uz * one_m_cos + ux * sin),
                            (sqz + (_1 - sqz) * cos))
            }
        }
    }

    /// Builds a rotation matrix from an orthogonal matrix.
    ///
    /// This is unsafe because the orthogonality of `mat` is not checked.
    pub unsafe fn new_with_mat(mat: Mat3<N>) -> Rot3<N> {
        Rot3 {
            submat: mat
        }
    }

    /// Creates a new rotation from Euler angles.
    ///
    /// The primitive rotations are applied in order: 1 roll − 2 pitch − 3 yaw.
    pub fn new_with_euler_angles(roll: N, pitch: N, yaw: N) -> Rot3<N> {
        let (sr, cr) = roll.sin_cos();
        let (sp, cp) = pitch.sin_cos();
        let (sy, cy) = yaw.sin_cos();

        unsafe {
            Rot3::new_with_mat(
                Mat3::new(
                    cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr,
                    sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr,
                    -sp,     cp * sr,                cp * cr
                )
            )
        }
    }
}

impl<N: Clone + BaseFloat> Rot3<N> {
    /// Creates a rotation that corresponds to the local frame of an observer standing at the
    /// origin and looking toward `dir`.
    ///
    /// It maps the view direction `dir` to the positive `z` axis.
    ///
    /// # Arguments
    ///   * dir - The look direction, that is, direction the matrix `z` axis will be aligned with.
    ///   * up - The vertical direction. The only requirement of this parameter is to not be
    ///   collinear
    ///   to `dir`. Non-collinearity is not checked.
    #[inline]
    pub fn new_observer_frame(dir: &Vec3<N>, up: &Vec3<N>) -> Rot3<N> {
        let zaxis = Norm::normalize(dir);
        let xaxis = Norm::normalize(&Cross::cross(up, &zaxis));
        let yaxis = Norm::normalize(&Cross::cross(&zaxis, &xaxis));

        unsafe {
            Rot3::new_with_mat(Mat3::new(
                xaxis.x.clone(), yaxis.x.clone(), zaxis.x.clone(),
                xaxis.y.clone(), yaxis.y.clone(), zaxis.y.clone(),
                xaxis.z        , yaxis.z        , zaxis.z))
        }
    }


    /// Builds a right-handed look-at view matrix without translation.
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
    pub fn look_at_rh(dir: &Vec3<N>, up: &Vec3<N>) -> Rot3<N> {
        Rot3::new_observer_frame(&(-*dir), up).inv().unwrap()
    }

    /// Builds a left-handed look-at view matrix without translation.
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
    pub fn look_at_lh(dir: &Vec3<N>, up: &Vec3<N>) -> Rot3<N> {
        Rot3::new_observer_frame(&(*dir), up).inv().unwrap()
    }
}

impl<N: Clone + BaseFloat + Cast<f64>>
Rotation<Vec3<N>> for Rot3<N> {
    #[inline]
    fn rotation(&self) -> Vec3<N> {
        let angle = ((self.submat.m11 + self.submat.m22 + self.submat.m33 - ::one()) / Cast::from(2.0)).acos();

        if angle != angle {
            // FIXME: handle that correctly
            ::zero()
        }
        else if ::is_zero(&angle) {
            ::zero()
        }
        else {
            let m32_m23 = self.submat.m32 - self.submat.m23;
            let m13_m31 = self.submat.m13 - self.submat.m31;
            let m21_m12 = self.submat.m21 - self.submat.m12;

            let denom = (m32_m23 * m32_m23 + m13_m31 * m13_m31 + m21_m12 * m21_m12).sqrt();

            if ::is_zero(&denom) {
                // XXX: handle that properly
                // panic!("Internal error: singularity.")
                ::zero()
            }
            else {
                let a_d = angle / denom;

                Vec3::new(m32_m23 * a_d, m13_m31 * a_d, m21_m12 * a_d)
            }
        }
    }

    #[inline]
    fn inv_rotation(&self) -> Vec3<N> {
        -self.rotation()
    }


    #[inline]
    fn append_rotation_mut(&mut self, rot: &Vec3<N>) {
        *self = Rotation::append_rotation(self, rot)
    }

    #[inline]
    fn append_rotation(&self, axisangle: &Vec3<N>) -> Rot3<N> {
        Rot3::new(axisangle.clone()) * *self
    }

    #[inline]
    fn prepend_rotation_mut(&mut self, rot: &Vec3<N>) {
        *self = Rotation::prepend_rotation(self, rot)
    }

    #[inline]
    fn prepend_rotation(&self, axisangle: &Vec3<N>) -> Rot3<N> {
        *self * Rot3::new(axisangle.clone())
    }

    #[inline]
    fn set_rotation(&mut self, axisangle: Vec3<N>) {
        *self = Rot3::new(axisangle)
    }
}

impl<N: BaseFloat> RotationTo for Rot3<N> {
    type AngleType = N;
    type DeltaRotationType = Rot3<N>;

    #[inline]
    fn angle_to(&self, other: &Self) -> N {
        // FIXME: refactor to avoid the normalization of the rotation axisangle vector.
        self.rotation_to(other).rotation().norm()
    }

    #[inline]
    fn rotation_to(&self, other: &Self) -> Rot3<N> {
        *other * ::inv(self).unwrap()
    }
}

impl<N: Clone + Rand + BaseFloat> Rand for Rot3<N> {
    #[inline]
    fn rand<R: Rng>(rng: &mut R) -> Rot3<N> {
        Rot3::new(rng.gen())
    }
}

impl<N: BaseFloat> AbsoluteRotate<Vec3<N>> for Rot3<N> {
    #[inline]
    fn absolute_rotate(&self, v: &Vec3<N>) -> Vec3<N> {
        Vec3::new(
            ::abs(&self.submat.m11) * v.x + ::abs(&self.submat.m12) * v.y + ::abs(&self.submat.m13) * v.z,
            ::abs(&self.submat.m21) * v.x + ::abs(&self.submat.m22) * v.y + ::abs(&self.submat.m23) * v.z,
            ::abs(&self.submat.m31) * v.x + ::abs(&self.submat.m32) * v.y + ::abs(&self.submat.m33) * v.z)
    }
}

#[cfg(feature="arbitrary")]
impl<N: Arbitrary + Clone + BaseFloat> Arbitrary for Rot3<N> {
    fn arbitrary<G: Gen>(g: &mut G) -> Rot3<N> {
        Rot3::new(Arbitrary::arbitrary(g))
    }
}


/*
 * Common implementations.
 */

submat_impl!(Rot2, Mat2);
rotate_impl!(Rot2, Vec2, Pnt2);
transform_impl!(Rot2, Vec2, Pnt2);
dim_impl!(Rot2, 2);
rot_mul_rot_impl!(Rot2);
rot_mul_vec_impl!(Rot2, Vec2);
vec_mul_rot_impl!(Rot2, Vec2);
rot_mul_pnt_impl!(Rot2, Pnt2);
pnt_mul_rot_impl!(Rot2, Pnt2);
one_impl!(Rot2);
eye_impl!(Rot2);
rotation_matrix_impl!(Rot2, Vec2, Vec1);
col_impl!(Rot2, Vec2);
row_impl!(Rot2, Vec2);
index_impl!(Rot2);
absolute_impl!(Rot2, Mat2);
to_homogeneous_impl!(Rot2, Mat3);
inv_impl!(Rot2);
transpose_impl!(Rot2);
approx_eq_impl!(Rot2);
diag_impl!(Rot2, Vec2);
rot_display_impl!(Rot2);

submat_impl!(Rot3, Mat3);
rotate_impl!(Rot3, Vec3, Pnt3);
transform_impl!(Rot3, Vec3, Pnt3);
dim_impl!(Rot3, 3);
rot_mul_rot_impl!(Rot3);
rot_mul_vec_impl!(Rot3, Vec3);
vec_mul_rot_impl!(Rot3, Vec3);
rot_mul_pnt_impl!(Rot3, Pnt3);
pnt_mul_rot_impl!(Rot3, Pnt3);
one_impl!(Rot3);
eye_impl!(Rot3);
rotation_matrix_impl!(Rot3, Vec3, Vec3);
col_impl!(Rot3, Vec3);
row_impl!(Rot3, Vec3);
index_impl!(Rot3);
absolute_impl!(Rot3, Mat3);
to_homogeneous_impl!(Rot3, Mat4);
inv_impl!(Rot3);
transpose_impl!(Rot3);
approx_eq_impl!(Rot3);
diag_impl!(Rot3, Vec3);
rot_display_impl!(Rot3);
