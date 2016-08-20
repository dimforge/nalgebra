//! Rotations matrices.

use std::fmt;
use std::ops::{Mul, MulAssign, Index};
use rand::{Rand, Rng};
use num::{Zero, One};
use traits::geometry::{Rotate, Rotation, AbsoluteRotate, RotationMatrix, RotationTo, Transform,
                       ToHomogeneous, Norm, Cross};
use traits::structure::{Cast, Dimension, Row, Column, BaseFloat, BaseNum, Eye, Diagonal};
use traits::operations::{Absolute, Inverse, Transpose, ApproxEq};
use structs::vector::{Vector1, Vector2, Vector3};
use structs::point::{Point2, Point3};
use structs::matrix::{Matrix2, Matrix3, Matrix4};

#[cfg(feature="arbitrary")]
use quickcheck::{Arbitrary, Gen};

#[cfg(feature="abstract_algebra")]
use_special_orthogonal_group_modules!();

/// Two dimensional rotation matrix.
#[repr(C)]
#[derive(Eq, PartialEq, RustcEncodable, RustcDecodable, Clone, Debug, Hash, Copy)]
pub struct Rotation2<N> {
    submatrix: Matrix2<N>
}

impl<N: BaseFloat> Rotation2<N> {
    /// Builds a 2 dimensional rotation matrix from an angle in radian.
    pub fn new(angle: Vector1<N>) -> Rotation2<N> {
        let (sia, coa) = angle.x.sin_cos();

        Rotation2 {
            submatrix: Matrix2::new(coa, -sia, sia, coa)
        }
    }
}

impl<N: BaseFloat> Rotation<Vector1<N>> for Rotation2<N> {
    #[inline]
    fn rotation(&self) -> Vector1<N> {
        Vector1::new((-self.submatrix.m12).atan2(self.submatrix.m11))
    }

    #[inline]
    fn inverse_rotation(&self) -> Vector1<N> {
        -self.rotation()
    }

    #[inline]
    fn append_rotation_mut(&mut self, rotation: &Vector1<N>) {
        *self = Rotation::append_rotation(self, rotation)
    }

    #[inline]
    fn append_rotation(&self, rotation: &Vector1<N>) -> Rotation2<N> {
        Rotation2::new(*rotation) * *self
    }

    #[inline]
    fn prepend_rotation_mut(&mut self, rotation: &Vector1<N>) {
        *self = Rotation::prepend_rotation(self, rotation)
    }

    #[inline]
    fn prepend_rotation(&self, rotation: &Vector1<N>) -> Rotation2<N> {
        *self * Rotation2::new(*rotation)
    }

    #[inline]
    fn set_rotation(&mut self, rotation: Vector1<N>) {
        *self = Rotation2::new(rotation)
    }
}

impl<N: BaseFloat> RotationTo for Rotation2<N> {
    type AngleType = N;
    type DeltaRotationType = Rotation2<N>;

    #[inline]
    fn angle_to(&self, other: &Self) -> N {
        self.rotation_to(other).rotation().norm()
    }

    #[inline]
    fn rotation_to(&self, other: &Self) -> Rotation2<N> {
        *other * ::inverse(self).unwrap()
    }
}

impl<N: Rand + BaseFloat> Rand for Rotation2<N> {
    #[inline]
    fn rand<R: Rng>(rng: &mut R) -> Rotation2<N> {
        Rotation2::new(rng.gen())
    }
}

impl<N: BaseFloat> AbsoluteRotate<Vector2<N>> for Rotation2<N> {
    #[inline]
    fn absolute_rotate(&self, v: &Vector2<N>) -> Vector2<N> {
        // the matrix is skew-symetric, so we dont need to compute the absolute value of every
        // component.
        let m11 = ::abs(&self.submatrix.m11);
        let m12 = ::abs(&self.submatrix.m12);
        let m22 = ::abs(&self.submatrix.m22);

        Vector2::new(m11 * v.x + m12 * v.y, m12 * v.x + m22 * v.y)
    }
}


/*
 * 3d rotation
 */
/// Three dimensional rotation matrix.
#[repr(C)]
#[derive(Eq, PartialEq, RustcEncodable, RustcDecodable, Clone, Debug, Hash, Copy)]
pub struct Rotation3<N> {
    submatrix: Matrix3<N>
}


impl<N: BaseFloat> Rotation3<N> {
    /// Builds a 3 dimensional rotation matrix from an axis and an angle.
    ///
    /// # Arguments
    ///   * `axisangle` - A vector representing the rotation. Its magnitude is the amount of rotation
    ///   in radian. Its direction is the axis of rotation.
    pub fn new(axisangle: Vector3<N>) -> Rotation3<N> {
        if ::is_zero(&Norm::norm_squared(&axisangle)) {
            ::one()
        }
        else {
            let mut axis   = axisangle;
            let angle      = axis.normalize_mut();
            let ux         = axis.x;
            let uy         = axis.y;
            let uz         = axis.z;
            let sqx        = ux * ux;
            let sqy        = uy * uy;
            let sqz        = uz * uz;
            let (sin, cos) = angle.sin_cos();
            let one_m_cos  = ::one::<N>() - cos;

            Rotation3 {
                submatrix: Matrix3::new(
                            (sqx + (::one::<N>() - sqx) * cos),
                            (ux * uy * one_m_cos - uz * sin),
                            (ux * uz * one_m_cos + uy * sin),

                            (ux * uy * one_m_cos + uz * sin),
                            (sqy + (::one::<N>() - sqy) * cos),
                            (uy * uz * one_m_cos - ux * sin),

                            (ux * uz * one_m_cos - uy * sin),
                            (uy * uz * one_m_cos + ux * sin),
                            (sqz + (::one::<N>() - sqz) * cos))
            }
        }
    }

    /// Builds a rotation matrix from an orthogonal matrix.
    pub fn from_matrix_unchecked(matrix: Matrix3<N>) -> Rotation3<N> {
        Rotation3 {
            submatrix: matrix
        }
    }

    /// Creates a new rotation from Euler angles.
    ///
    /// The primitive rotations are applied in order: 1 roll − 2 pitch − 3 yaw.
    pub fn from_euler_angles(roll: N, pitch: N, yaw: N) -> Rotation3<N> {
        let (sr, cr) = roll.sin_cos();
        let (sp, cp) = pitch.sin_cos();
        let (sy, cy) = yaw.sin_cos();

        Rotation3::from_matrix_unchecked(
            Matrix3::new(
                cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr,
                sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr,
                -sp,     cp * sr,                cp * cr
                )
            )
    }
}

impl<N: BaseFloat> Rotation3<N> {
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
    pub fn new_observer_frame(dir: &Vector3<N>, up: &Vector3<N>) -> Rotation3<N> {
        let zaxis = Norm::normalize(dir);
        let xaxis = Norm::normalize(&Cross::cross(up, &zaxis));
        let yaxis = Norm::normalize(&Cross::cross(&zaxis, &xaxis));

        Rotation3::from_matrix_unchecked(Matrix3::new(
                xaxis.x, yaxis.x, zaxis.x,
                xaxis.y, yaxis.y, zaxis.y,
                xaxis.z, yaxis.z, zaxis.z))
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
    pub fn look_at_rh(dir: &Vector3<N>, up: &Vector3<N>) -> Rotation3<N> {
        Rotation3::new_observer_frame(&(-*dir), up).inverse().unwrap()
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
    pub fn look_at_lh(dir: &Vector3<N>, up: &Vector3<N>) -> Rotation3<N> {
        Rotation3::new_observer_frame(&(*dir), up).inverse().unwrap()
    }
}

impl<N: BaseFloat> Rotation<Vector3<N>> for Rotation3<N> {
    #[inline]
    fn rotation(&self) -> Vector3<N> {
        let angle = ((self.submatrix.m11 + self.submatrix.m22 + self.submatrix.m33 - ::one()) / Cast::from(2.0)).acos();

        if !angle.is_finite()  || ::is_zero(&angle) {
            // FIXME: handle the non-finite case robustly.
            ::zero()
        }
        else {
            let m32_m23 = self.submatrix.m32 - self.submatrix.m23;
            let m13_m31 = self.submatrix.m13 - self.submatrix.m31;
            let m21_m12 = self.submatrix.m21 - self.submatrix.m12;

            let denom = (m32_m23 * m32_m23 + m13_m31 * m13_m31 + m21_m12 * m21_m12).sqrt();

            if ::is_zero(&denom) {
                // XXX: handle that properly
                // panic!("Internal error: singularity.")
                ::zero()
            }
            else {
                let a_d = angle / denom;

                Vector3::new(m32_m23 * a_d, m13_m31 * a_d, m21_m12 * a_d)
            }
        }
    }

    #[inline]
    fn inverse_rotation(&self) -> Vector3<N> {
        -self.rotation()
    }


    #[inline]
    fn append_rotation_mut(&mut self, rotation: &Vector3<N>) {
        *self = Rotation::append_rotation(self, rotation)
    }

    #[inline]
    fn append_rotation(&self, axisangle: &Vector3<N>) -> Rotation3<N> {
        Rotation3::new(*axisangle) * *self
    }

    #[inline]
    fn prepend_rotation_mut(&mut self, rotation: &Vector3<N>) {
        *self = Rotation::prepend_rotation(self, rotation)
    }

    #[inline]
    fn prepend_rotation(&self, axisangle: &Vector3<N>) -> Rotation3<N> {
        *self * Rotation3::new(*axisangle)
    }

    #[inline]
    fn set_rotation(&mut self, axisangle: Vector3<N>) {
        *self = Rotation3::new(axisangle)
    }
}

impl<N: BaseFloat> RotationTo for Rotation3<N> {
    type AngleType = N;
    type DeltaRotationType = Rotation3<N>;

    #[inline]
    fn angle_to(&self, other: &Self) -> N {
        // FIXME: refactor to avoid the normalization of the rotation axisangle vector.
        self.rotation_to(other).rotation().norm()
    }

    #[inline]
    fn rotation_to(&self, other: &Self) -> Rotation3<N> {
        *other * ::inverse(self).unwrap()
    }
}

impl<N: Rand + BaseFloat> Rand for Rotation3<N> {
    #[inline]
    fn rand<R: Rng>(rng: &mut R) -> Rotation3<N> {
        Rotation3::new(rng.gen())
    }
}

impl<N: BaseFloat> AbsoluteRotate<Vector3<N>> for Rotation3<N> {
    #[inline]
    fn absolute_rotate(&self, v: &Vector3<N>) -> Vector3<N> {
        Vector3::new(
            ::abs(&self.submatrix.m11) * v.x + ::abs(&self.submatrix.m12) * v.y + ::abs(&self.submatrix.m13) * v.z,
            ::abs(&self.submatrix.m21) * v.x + ::abs(&self.submatrix.m22) * v.y + ::abs(&self.submatrix.m23) * v.z,
            ::abs(&self.submatrix.m31) * v.x + ::abs(&self.submatrix.m32) * v.y + ::abs(&self.submatrix.m33) * v.z)
    }
}


/*
 * Common implementations.
 */

rotation_impl!(Rotation2, Matrix2, Vector2, Vector1, Point2, Matrix3);
dim_impl!(Rotation2, 2);

rotation_impl!(Rotation3, Matrix3, Vector3, Vector3, Point3, Matrix4);
dim_impl!(Rotation3, 3);
