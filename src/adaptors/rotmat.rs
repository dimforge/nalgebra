use std::num::{One, Zero};
use std::rand::{Rand, Rng, RngUtil};
use std::cmp::ApproxEq;
use traits::rlmul::{RMul, LMul};
use traits::cross::Cross;
use traits::dim::Dim;
use traits::inv::Inv;
use traits::transpose::Transpose;
use traits::rotation::{Rotation, Rotate};
use traits::transformation::{Transform}; // FIXME: implement Transformation and Transformable
use traits::homogeneous::ToHomogeneous;
use traits::indexable::Indexable;
use traits::vector::AlgebraicVec;
use vec::Vec1;
use mat::{Mat2, Mat3};
use vec::Vec3;

/// Matrix wrapper representing rotation matrix. It is built uppon another matrix and ensures (at
/// the type-level) that it will always represent a rotation. Rotation matrices have some
/// properties useful for performances, like the fact that the inversion is simply a transposition.
#[deriving(Eq, ToStr, Clone)]
pub struct Rotmat<M> {
    priv submat: M
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
            let _1         = One::one::<N>();
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
}

impl<N: Clone + Trigonometric + Num + Algebraic>
Rotation<Vec3<N>> for Rotmat<Mat3<N>> {
    #[inline]
    fn rotation(&self) -> Vec3<N> {
        fail!("Not yet implemented.")
    }

    #[inline]
    fn inv_rotation(&self) -> Vec3<N> {
        fail!("Not yet implemented.")
    }


    #[inline]
    fn rotate_by(&mut self, rot: &Vec3<N>) {
        *self = self.rotated(rot)
    }

    #[inline]
    fn rotated(&self, axisangle: &Vec3<N>) -> Rotmat<Mat3<N>> {
        Rotmat::from_axis_angle(axisangle.clone()) * *self
    }
}

impl<N: Clone + Rand + Trigonometric + Neg<N>> Rand for Rotmat<Mat2<N>> {
    #[inline]
    fn rand<R: Rng>(rng: &mut R) -> Rotmat<Mat2<N>> {
        Rotmat::from_angle(rng.gen())
    }
}

impl<M: RMul<V> + LMul<V>, V> Rotate<V> for Rotmat<M> {
    #[inline]
    fn rotate(&self, v: &V) -> V {
        self.rmul(v)
    }

    #[inline]
    fn inv_rotate(&self, v: &V) -> V {
        self.lmul(v)
    }
}

impl<M: RMul<V> + LMul<V>, V> Transform<V> for Rotmat<M> {
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
    fn dim() -> uint {
        Dim::dim::<M>()
    }
}

impl<M: One + Zero> One for Rotmat<M> {
    #[inline]
    fn one() -> Rotmat<M> {
        Rotmat { submat: One::one() }
    }
}

impl<M: Mul<M, M>> Mul<Rotmat<M>, Rotmat<M>> for Rotmat<M> {
    #[inline]
    fn mul(&self, other: &Rotmat<M>) -> Rotmat<M> {
        Rotmat { submat: self.submat.mul(&other.submat) }
    }
}

impl<V, M: RMul<V>> RMul<V> for Rotmat<M> {
    #[inline]
    fn rmul(&self, other: &V) -> V {
        self.submat.rmul(other)
    }
}

impl<V, M: LMul<V>> LMul<V> for Rotmat<M> {
    #[inline]
    fn lmul(&self, other: &V) -> V {
        self.submat.lmul(other)
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

// we loose the info that we are a rotation matrix
impl<M: ToHomogeneous<M2>, M2> ToHomogeneous<M2> for Rotmat<M> {
    fn to_homogeneous(&self) -> M2 {
        self.submat.to_homogeneous()
    }
}

impl<N: ApproxEq<N>, M: ApproxEq<N>> ApproxEq<N> for Rotmat<M> {
    #[inline]
    fn approx_epsilon() -> N {
        ApproxEq::approx_epsilon::<N, N>()
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
