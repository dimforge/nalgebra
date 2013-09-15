use std::num::{One, Zero};
use std::rand::{Rand, Rng, RngUtil};
use std::cmp::ApproxEq;
use traits::cross::Cross;
use traits::dim::Dim;
use traits::inv::Inv;
use traits::row::Row;
use traits::col::Col;
use traits::transpose::Transpose;
use traits::absolute::Absolute;
use traits::rotation::{Rotation, Rotate, RotationMatrix};
use traits::transformation::{Transform}; // FIXME: implement Transformation and Transformable
use traits::homogeneous::ToHomogeneous;
use traits::indexable::Indexable;
use traits::norm::Norm;
use traits::comp::absolute_rotate::AbsoluteRotate;
use vec::{Vec1, Vec2, Vec3, Vec2MulRhs, Vec3MulRhs};
use mat::{Mat2, Mat3};

#[path = "../metal.rs"]
mod metal;

/// Matrix wrapper representing rotation matrix. It is built uppon another matrix and ensures (at
/// the type-level) that it will always represent a rotation. Rotation matrices have some
/// properties useful for performances, like the fact that the inversion is simply a transposition.
#[deriving(Eq, ToStr, Clone)]
pub struct Rotmat<M> {
    priv submat: M
}

/// Trait of object `o` which can be multiplied by a `Rotmat` `r`: `r * o`.
pub trait RotmatMulRhs<M, Res> {
    /// Multiplies a rotation matrix by `Self`.
    fn binop(left: &Rotmat<M>, right: &Self) -> Res;
}

impl<M, Rhs: RotmatMulRhs<M, Res>, Res> Mul<Rhs, Res> for Rotmat<M> {
    #[inline(always)]
    fn mul(&self, other: &Rhs) -> Res {
        RotmatMulRhs::binop(self, other)
    }
}

impl<M: Clone> Rotmat<M> {
    /// Gets a copy of the internal representation of the rotation.
    pub fn submat(&self) -> M {
        self.submat.clone()
    }
}

impl<N: Clone + Trigonometric + Neg<N>> Rotmat<Mat2<N>> {
    /// Builds a 2 dimensional rotation matrix from an angle in radian.
    pub fn from_angle(angle: N) -> Rotmat<Mat2<N>> {
        let (sia, coa) = angle.sin_cos();

        Rotmat {
            submat: Mat2::new(coa.clone(), -sia, sia.clone(), coa)
        }
    }
}

impl<N: Clone + Trigonometric + Num + Algebraic> Rotmat<Mat3<N>> {
    /// Builds a 3 dimensional rotation matrix from an axis and an angle.
    ///
    /// # Arguments
    ///   * `axisangle` - A vector representing the rotation. Its magnitude is the amount of rotation
    ///   in radian. Its direction is the axis of rotation.
    pub fn from_axis_angle(axisangle: Vec3<N>) -> Rotmat<Mat3<N>> {
        if axisangle.sqnorm().is_zero() {
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

            Rotmat {
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

impl<N: Clone + Num + Algebraic> Rotmat<Mat3<N>> {
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
        let xaxis = at.normalized();
        let zaxis = up.cross(&xaxis).normalized();
        let yaxis = zaxis.cross(&xaxis);

        self.submat = Mat3::new(xaxis.x.clone(), yaxis.x.clone(), zaxis.x.clone(),
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
        let zaxis = at.normalized();
        let xaxis = up.cross(&zaxis).normalized();
        let yaxis = zaxis.cross(&xaxis);

        self.submat = Mat3::new(xaxis.x.clone(), yaxis.x.clone(), zaxis.x.clone(),
        xaxis.y.clone(), yaxis.y.clone(), zaxis.y.clone(),
        xaxis.z        , yaxis.z        , zaxis.z)
    }
}

impl<N: Trigonometric + Num + Clone>
RotationMatrix<Vec2<N>, Vec1<N>, Rotmat<Mat2<N>>> for Rotmat<Mat2<N>> {
    #[inline]
    fn to_rot_mat(&self) -> Rotmat<Mat2<N>> {
        self.clone()
    }
}

impl<N: Trigonometric + Num + Clone>
Rotation<Vec1<N>> for Rotmat<Mat2<N>> {
    #[inline]
    fn rotation(&self) -> Vec1<N> {
        Vec1::new((-self.submat.at((0, 1))).atan2(&self.submat.at((0, 0))))
    }

    #[inline]
    fn inv_rotation(&self) -> Vec1<N> {
        -self.rotation()
    }

    #[inline]
    fn rotate_by(&mut self, rot: &Vec1<N>) {
        *self = self.rotated(rot)
    }

    #[inline]
    fn rotated(&self, rot: &Vec1<N>) -> Rotmat<Mat2<N>> {
        Rotmat::from_angle(rot.x.clone()) * *self
    }

    #[inline]
    fn set_rotation(&mut self, rot: Vec1<N>) {
        *self = Rotmat::from_angle(rot.x)
    }
}

impl<N: NumCast + Algebraic + Trigonometric + Num + Clone>
RotationMatrix<Vec3<N>, Vec3<N>, Rotmat<Mat3<N>>> for Rotmat<Mat3<N>> {
    #[inline]
    fn to_rot_mat(&self) -> Rotmat<Mat3<N>> {
        self.clone()
    }
}

impl<N: Clone + Trigonometric + Num + Algebraic + NumCast>
Rotation<Vec3<N>> for Rotmat<Mat3<N>> {
    #[inline]
    fn rotation(&self) -> Vec3<N> {
        let angle = ((self.submat.m11 + self.submat.m22 + self.submat.m33 - One::one()) / NumCast::from(2.0)).acos();

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
    fn rotate_by(&mut self, rot: &Vec3<N>) {
        *self = self.rotated(rot)
    }

    #[inline]
    fn rotated(&self, axisangle: &Vec3<N>) -> Rotmat<Mat3<N>> {
        Rotmat::from_axis_angle(axisangle.clone()) * *self
    }

    #[inline]
    fn set_rotation(&mut self, axisangle: Vec3<N>) {
        *self = Rotmat::from_axis_angle(axisangle)
    }
}

impl<N: Clone + Rand + Trigonometric + Neg<N>> Rand for Rotmat<Mat2<N>> {
    #[inline]
    fn rand<R: Rng>(rng: &mut R) -> Rotmat<Mat2<N>> {
        Rotmat::from_angle(rng.gen())
    }
}

impl<M, V: RotmatMulRhs<M, V> + Mul<Rotmat<M>, V>> Rotate<V> for Rotmat<M> {
    #[inline]
    fn rotate(&self, v: &V) -> V {
        self * *v
    }

    #[inline]
    fn inv_rotate(&self, v: &V) -> V {
        v * *self
    }
}

impl<M, V: RotmatMulRhs<M, V> + Mul<Rotmat<M>, V>> Transform<V> for Rotmat<M> {
    #[inline]
    fn transform(&self, v: &V) -> V {
        self.rotate(v)
    }

    #[inline]
    fn inv_transform(&self, v: &V) -> V {
        self.inv_rotate(v)
    }
}

impl<N: Clone + Rand + Trigonometric + Num + Algebraic>
Rand for Rotmat<Mat3<N>> {
    #[inline]
    fn rand<R: Rng>(rng: &mut R) -> Rotmat<Mat3<N>> {
        Rotmat::from_axis_angle(rng.gen())
    }
}

impl<M: Dim> Dim for Rotmat<M> {
    #[inline]
    fn dim(_: Option<Rotmat<M>>) -> uint {
        Dim::dim(None::<M>)
    }
}

impl<M: One + Zero> One for Rotmat<M> {
    #[inline]
    fn one() -> Rotmat<M> {
        Rotmat { submat: One::one() }
    }
}

impl<M: Mul<M, M>> RotmatMulRhs<M, Rotmat<M>> for Rotmat<M> {
    #[inline]
    fn binop(left: &Rotmat<M>, right: &Rotmat<M>) -> Rotmat<M> {
        Rotmat { submat: left.submat * right.submat }
    }
}

/*
 * Right/Left multiplication implementation for Vec3 and Vec2.
 */
impl<M: Mul<Vec3<N>, Vec3<N>>, N> RotmatMulRhs<M, Vec3<N>> for Vec3<N> {
    #[inline]
    fn binop(left: &Rotmat<M>, right: &Vec3<N>) -> Vec3<N> {
        left.submat * *right 
    }
}

impl<M: Mul<Vec2<N>, Vec2<N>>, N> RotmatMulRhs<M, Vec2<N>> for Vec2<N> {
    #[inline]
    fn binop(left: &Rotmat<M>, right: &Vec2<N>) -> Vec2<N> {
        left.submat * *right 
    }
}

impl<N, M: Vec3MulRhs<N, Vec3<N>>> Vec3MulRhs<N, Vec3<N>> for Rotmat<M> {
    #[inline]
    fn binop(left: &Vec3<N>, right: &Rotmat<M>) -> Vec3<N> {
        *left * right.submat
    }
}

impl<N, M: Vec2MulRhs<N, Vec2<N>>> Vec2MulRhs<N, Vec2<N>> for Rotmat<M> {
    #[inline]
    fn binop(left: &Vec2<N>, right: &Rotmat<M>) -> Vec2<N> {
        *left * right.submat
    }
}

impl<M: Transpose> Inv for Rotmat<M> {
    #[inline]
    fn inplace_inverse(&mut self) -> bool {
        self.transpose();

        true
    }

    #[inline]
    fn inverse(&self) -> Option<Rotmat<M>> {
        Some(self.transposed())
    }
}

impl<M: Transpose>
Transpose for Rotmat<M> {
    #[inline]
    fn transposed(&self) -> Rotmat<M> {
        Rotmat { submat: self.submat.transposed() }
    }

    #[inline]
    fn transpose(&mut self) {
        self.submat.transpose()
    }
}

impl<M: Row<R>, R> Row<R> for Rotmat<M> {
    #[inline]
    fn num_rows(&self) -> uint {
        self.submat.num_rows()
    }
    #[inline]
    fn row(&self, i: uint) -> R {
        self.submat.row(i)
    }

    #[inline]
    fn set_row(&mut self, i: uint, row: R) {
        self.submat.set_row(i, row);
    }
}

impl<M: Col<C>, C> Col<C> for Rotmat<M> {
    #[inline]
    fn num_cols(&self) -> uint {
        self.submat.num_cols()
    }

    #[inline]
    fn col(&self, i: uint) -> C {
        self.submat.col(i)
    }

    #[inline]
    fn set_col(&mut self, i: uint, col: C) {
        self.submat.set_col(i, col);
    }
}

// we loose the info that we are a rotation matrix
impl<M: ToHomogeneous<M2>, M2> ToHomogeneous<M2> for Rotmat<M> {
    #[inline]
    fn to_homogeneous(&self) -> M2 {
        self.submat.to_homogeneous()
    }
}

impl<N: ApproxEq<N>, M: ApproxEq<N>> ApproxEq<N> for Rotmat<M> {
    #[inline]
    fn approx_epsilon() -> N {
        // ApproxEq::<N>::approx_epsilon()
        fail!("approx_epsilon is broken since rust revision 8693943676487c01fa09f5f3daf0df6a1f71e24d.")
    }

    #[inline]
    fn approx_eq(&self, other: &Rotmat<M>) -> bool {
        self.submat.approx_eq(&other.submat)
    }

    #[inline]
    fn approx_eq_eps(&self, other: &Rotmat<M>, epsilon: &N) -> bool {
        self.submat.approx_eq_eps(&other.submat, epsilon)
    }
}

impl<M: Absolute<M2>, M2> Absolute<M2> for Rotmat<M> {
    #[inline]
    fn absolute(&self) -> M2 {
        self.submat.absolute()
    }
}

impl<N: Signed> AbsoluteRotate<Vec3<N>> for Rotmat<Mat3<N>> {
    #[inline]
    fn absolute_rotate(&self, v: &Vec3<N>) -> Vec3<N> {
        Vec3::new(
            self.submat.m11.abs() * v.x + self.submat.m12.abs() * v.y + self.submat.m13.abs() * v.z,
            self.submat.m21.abs() * v.x + self.submat.m22.abs() * v.y + self.submat.m23.abs() * v.z,
            self.submat.m31.abs() * v.x + self.submat.m32.abs() * v.y + self.submat.m33.abs() * v.z)
    }
}

impl<N: Signed> AbsoluteRotate<Vec2<N>> for Rotmat<Mat2<N>> {
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
