//! Rotations matrices.

#![allow(missing_doc)]

use std::num::{Zero, One};
use rand::{Rand, Rng};
use traits::geometry::{Rotate, Rotation, AbsoluteRotate, RotationMatrix, Transform, ToHomogeneous,
                       Norm, Cross};
use traits::structure::{Cast, Dim, Row, Col};
use traits::operations::{Absolute, Inv, Transpose, ApproxEq};
use structs::vec::{Vec1, Vec2, Vec3, Vec4, Vec2MulRhs, Vec3MulRhs, Vec4MulRhs};
use structs::mat::{Mat2, Mat3, Mat4, Mat5};

mod metal;
mod rot_macros;

/// Two dimensional rotation matrix.
#[deriving(TotalEq, Eq, Encodable, Decodable, Clone, Show, Hash)]
pub struct Rot2<N> {
    submat: Mat2<N>
}

impl<N: Clone + FloatMath + Neg<N>> Rot2<N> {
    /// Builds a 2 dimensional rotation matrix from an angle in radian.
    pub fn new(angle: Vec1<N>) -> Rot2<N> {
        let (sia, coa) = angle.x.sin_cos();

        Rot2 {
            submat: Mat2::new(coa.clone(), -sia, sia.clone(), coa)
        }
    }
}

impl<N: FloatMath + Clone>
Rotation<Vec1<N>> for Rot2<N> {
    #[inline]
    fn rotation(&self) -> Vec1<N> {
        Vec1::new((-self.submat.m12).atan2(self.submat.m11.clone()))
    }

    #[inline]
    fn inv_rotation(&self) -> Vec1<N> {
        -self.rotation()
    }

    #[inline]
    fn append_rotation(&mut self, rot: &Vec1<N>) {
        *self = Rotation::append_rotation_cpy(self, rot)
    }

    #[inline]
    fn append_rotation_cpy(t: &Rot2<N>, rot: &Vec1<N>) -> Rot2<N> {
        Rot2::new(rot.clone()) * *t
    }

    #[inline]
    fn prepend_rotation(&mut self, rot: &Vec1<N>) {
        *self = Rotation::prepend_rotation_cpy(self, rot)
    }

    #[inline]
    fn prepend_rotation_cpy(t: &Rot2<N>, rot: &Vec1<N>) -> Rot2<N> {
        *t * Rot2::new(rot.clone())
    }

    #[inline]
    fn set_rotation(&mut self, rot: Vec1<N>) {
        *self = Rot2::new(rot)
    }
}

impl<N: Clone + Rand + FloatMath + Neg<N>> Rand for Rot2<N> {
    #[inline]
    fn rand<R: Rng>(rng: &mut R) -> Rot2<N> {
        Rot2::new(rng.gen())
    }
}

impl<N: Signed> AbsoluteRotate<Vec2<N>> for Rot2<N> {
    #[inline]
    fn absolute_rotate(&self, v: &Vec2<N>) -> Vec2<N> {
        // the matrix is skew-symetric, so we dont need to compute the absolute value of every
        // component.
        let m11 = self.submat.m11.abs();
        let m12 = self.submat.m12.abs();
        let m22 = self.submat.m22.abs();

        Vec2::new(m11 * v.x + m12 * v.y, m12 * v.x + m22 * v.y)
    }
}

/*
 * 3d rotation
 */
/// Three dimensional rotation matrix.
#[deriving(TotalEq, Eq, Encodable, Decodable, Clone, Show, Hash)]
pub struct Rot3<N> {
    submat: Mat3<N>
}


impl<N: Clone + FloatMath> Rot3<N> {
    /// Builds a 3 dimensional rotation matrix from an axis and an angle.
    ///
    /// # Arguments
    ///   * `axisangle` - A vector representing the rotation. Its magnitude is the amount of rotation
    ///   in radian. Its direction is the axis of rotation.
    pub fn new(axisangle: Vec3<N>) -> Rot3<N> {
        if Norm::sqnorm(&axisangle).is_zero() {
            One::one()
        }
        else {
            let mut axis   = axisangle;
            let angle      = axis.normalize();
            let _1: N      = One::one();
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
}

impl<N: Clone + Float> Rot3<N> {
    /// Reorient this matrix such that its local `x` axis points to a given point. Note that the
    /// usually known `look_at` function does the same thing but with the `z` axis. See `look_at_z`
    /// for that.
    ///
    /// # Arguments
    ///   * at - The point to look at. It is also the direction the matrix `x` axis will be aligned
    ///   with
    ///   * up - Vector pointing `up`. The only requirement of this parameter is to not be colinear
    ///   with `at`. Non-colinearity is not checked.
    pub fn look_at(&mut self, at: &Vec3<N>, up: &Vec3<N>) {
        let xaxis = Norm::normalize_cpy(at);
        let zaxis = Norm::normalize_cpy(&Cross::cross(up, &xaxis));
        let yaxis = Cross::cross(&zaxis, &xaxis);

        self.submat = Mat3::new(
            xaxis.x.clone(), yaxis.x.clone(), zaxis.x.clone(),
            xaxis.y.clone(), yaxis.y.clone(), zaxis.y.clone(),
            xaxis.z        , yaxis.z        , zaxis.z)
    }

    /// Reorient this matrix such that its local `z` axis points to a given point. 
    ///
    /// # Arguments
    ///   * at - The point to look at. It is also the direction the matrix `y` axis will be aligned
    ///   with
    ///   * up - Vector pointing `up`. The only requirement of this parameter is to not be colinear
    ///   with `at`. Non-colinearity is not checked.
    pub fn look_at_z(&mut self, at: &Vec3<N>, up: &Vec3<N>) {
        let zaxis = Norm::normalize_cpy(at);
        let xaxis = Norm::normalize_cpy(&Cross::cross(up, &zaxis));
        let yaxis = Cross::cross(&zaxis, &xaxis);

        self.submat = Mat3::new(
            xaxis.x.clone(), yaxis.x.clone(), zaxis.x.clone(),
            xaxis.y.clone(), yaxis.y.clone(), zaxis.y.clone(),
            xaxis.z        , yaxis.z        , zaxis.z)
    }
}

impl<N: Clone + FloatMath + Cast<f32>>
Rotation<Vec3<N>> for Rot3<N> {
    #[inline]
    fn rotation(&self) -> Vec3<N> {
        let angle = ((self.submat.m11 + self.submat.m22 + self.submat.m33 - One::one()) / Cast::from(2.0)).acos();

        if angle != angle {
            // FIXME: handle that correctly
            Zero::zero()
        }
        else if angle.is_zero() {
            Zero::zero()
        }
        else {
            let m32_m23 = self.submat.m32 - self.submat.m23;
            let m13_m31 = self.submat.m13 - self.submat.m31;
            let m21_m12 = self.submat.m21 - self.submat.m12;

            let denom = (m32_m23 * m32_m23 + m13_m31 * m13_m31 + m21_m12 * m21_m12).sqrt();

            if denom.is_zero() {
                // XXX: handle that properly
                // fail!("Internal error: singularity.")
                Zero::zero()
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
    fn append_rotation(&mut self, rot: &Vec3<N>) {
        *self = Rotation::append_rotation_cpy(self, rot)
    }

    #[inline]
    fn append_rotation_cpy(t: &Rot3<N>, axisangle: &Vec3<N>) -> Rot3<N> {
        Rot3::new(axisangle.clone()) * *t
    }

    #[inline]
    fn prepend_rotation(&mut self, rot: &Vec3<N>) {
        *self = Rotation::prepend_rotation_cpy(self, rot)
    }

    #[inline]
    fn prepend_rotation_cpy(t: &Rot3<N>, axisangle: &Vec3<N>) -> Rot3<N> {
        *t * Rot3::new(axisangle.clone())
    }

    #[inline]
    fn set_rotation(&mut self, axisangle: Vec3<N>) {
        *self = Rot3::new(axisangle)
    }
}

impl<N: Clone + Rand + FloatMath>
Rand for Rot3<N> {
    #[inline]
    fn rand<R: Rng>(rng: &mut R) -> Rot3<N> {
        Rot3::new(rng.gen())
    }
}

impl<N: Signed> AbsoluteRotate<Vec3<N>> for Rot3<N> {
    #[inline]
    fn absolute_rotate(&self, v: &Vec3<N>) -> Vec3<N> {
        Vec3::new(
            self.submat.m11.abs() * v.x + self.submat.m12.abs() * v.y + self.submat.m13.abs() * v.z,
            self.submat.m21.abs() * v.x + self.submat.m22.abs() * v.y + self.submat.m23.abs() * v.z,
            self.submat.m31.abs() * v.x + self.submat.m32.abs() * v.y + self.submat.m33.abs() * v.z)
    }
}

/// Four dimensional rotation matrix.
#[deriving(TotalEq, Eq, Encodable, Decodable, Clone, Show, Hash)]
pub struct Rot4<N> {
    submat: Mat4<N>
}

// impl<N> Rot4<N> {
//     pub fn new(left_iso: Quat<N>, right_iso: Quat<N>) -> Rot4<N> {
//         assert!(left_iso.is_unit());
//         assert!(right_iso.is_unright);
// 
//         let mat_left_iso = Mat4::new(
//             left_iso.x, -left_iso.y, -left_iso.z, -left_iso.w,
//             left_iso.y,  left_iso.x, -left_iso.w,  left_iso.z,
//             left_iso.z,  left_iso.w,  left_iso.x, -left_iso.y,
//             left_iso.w, -left_iso.z,  left_iso.y,  left_iso.x);
//         let mat_right_iso = Mat4::new(
//             right_iso.x, -right_iso.y, -right_iso.z, -right_iso.w,
//             right_iso.y,  right_iso.x,  right_iso.w, -right_iso.z,
//             right_iso.z, -right_iso.w,  right_iso.x,  right_iso.y,
//             right_iso.w,  right_iso.z, -right_iso.y,  right_iso.x);
// 
//         Rot4 {
//             submat: mat_left_iso * mat_right_iso
//         }
//     }
// }

impl<N: Signed> AbsoluteRotate<Vec4<N>> for Rot4<N> {
    #[inline]
    fn absolute_rotate(&self, v: &Vec4<N>) -> Vec4<N> {
        Vec4::new(
            self.submat.m11.abs() * v.x + self.submat.m12.abs() * v.y +
            self.submat.m13.abs() * v.z + self.submat.m14.abs() * v.w,

            self.submat.m21.abs() * v.x + self.submat.m22.abs() * v.y +
            self.submat.m23.abs() * v.z + self.submat.m24.abs() * v.w,

            self.submat.m31.abs() * v.x + self.submat.m32.abs() * v.y +
            self.submat.m33.abs() * v.z + self.submat.m34.abs() * v.w,

            self.submat.m41.abs() * v.x + self.submat.m42.abs() * v.y +
            self.submat.m43.abs() * v.z + self.submat.m44.abs() * v.w)
    }
}

impl<N: Float + Clone>
Rotation<Vec4<N>> for Rot4<N> {
    #[inline]
    fn rotation(&self) -> Vec4<N> {
        fail!("Not yet implemented")
    }

    #[inline]
    fn inv_rotation(&self) -> Vec4<N> {
        fail!("Not yet implemented")
    }

    #[inline]
    fn append_rotation(&mut self, _: &Vec4<N>) {
        fail!("Not yet implemented")
    }

    #[inline]
    fn append_rotation_cpy(_: &Rot4<N>, _: &Vec4<N>) -> Rot4<N> {
        fail!("Not yet implemented")
    }

    #[inline]
    fn prepend_rotation(&mut self, _: &Vec4<N>) {
        fail!("Not yet implemented")
    }

    #[inline]
    fn prepend_rotation_cpy(_: &Rot4<N>, _: &Vec4<N>) -> Rot4<N> {
        fail!("Not yet implemented")
    }

    #[inline]
    fn set_rotation(&mut self, _: Vec4<N>) {
        fail!("Not yet implemented")
    }
}


/*
 * Common implementations.
 */

double_dispatch_binop_decl_trait!(Rot2, Rot2MulRhs)
mul_redispatch_impl!(Rot2, Rot2MulRhs)
submat_impl!(Rot2, Mat2)
rotate_impl!(Rot2, Vec2)
transform_impl!(Rot2, Vec2)
dim_impl!(Rot2, 2)
rot_mul_rot_impl!(Rot2, Rot2MulRhs)
rot_mul_vec_impl!(Rot2, Vec2, Rot2MulRhs)
vec_mul_rot_impl!(Rot2, Vec2, Vec2MulRhs)
one_impl!(Rot2)
rotation_matrix_impl!(Rot2, Vec2, Vec1)
col_impl!(Rot2, Vec2)
row_impl!(Rot2, Vec2)
absolute_impl!(Rot2, Mat2)
to_homogeneous_impl!(Rot2, Mat3)
inv_impl!(Rot2)
transpose_impl!(Rot2)
approx_eq_impl!(Rot2)

double_dispatch_binop_decl_trait!(Rot3, Rot3MulRhs)
mul_redispatch_impl!(Rot3, Rot3MulRhs)
submat_impl!(Rot3, Mat3)
rotate_impl!(Rot3, Vec3)
transform_impl!(Rot3, Vec3)
dim_impl!(Rot3, 3)
rot_mul_rot_impl!(Rot3, Rot3MulRhs)
rot_mul_vec_impl!(Rot3, Vec3, Rot3MulRhs)
vec_mul_rot_impl!(Rot3, Vec3, Vec3MulRhs)
one_impl!(Rot3)
rotation_matrix_impl!(Rot3, Vec3, Vec3)
col_impl!(Rot3, Vec3)
row_impl!(Rot3, Vec3)
absolute_impl!(Rot3, Mat3)
to_homogeneous_impl!(Rot3, Mat4)
inv_impl!(Rot3)
transpose_impl!(Rot3)
approx_eq_impl!(Rot3)

double_dispatch_binop_decl_trait!(Rot4, Rot4MulRhs)
mul_redispatch_impl!(Rot4, Rot4MulRhs)
submat_impl!(Rot4, Mat4)
rotate_impl!(Rot4, Vec4)
transform_impl!(Rot4, Vec4)
dim_impl!(Rot4, 4)
rot_mul_rot_impl!(Rot4, Rot4MulRhs)
rot_mul_vec_impl!(Rot4, Vec4, Rot4MulRhs)
vec_mul_rot_impl!(Rot4, Vec4, Vec4MulRhs)
one_impl!(Rot4)
rotation_matrix_impl!(Rot4, Vec4, Vec4)
col_impl!(Rot4, Vec4)
row_impl!(Rot4, Vec4)
absolute_impl!(Rot4, Mat4)
to_homogeneous_impl!(Rot4, Mat5)
inv_impl!(Rot4)
transpose_impl!(Rot4)
approx_eq_impl!(Rot4)
