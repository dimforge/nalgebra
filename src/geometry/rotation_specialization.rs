#[cfg(feature = "arbitrary")]
use crate::base::storage::Owned;
#[cfg(feature = "arbitrary")]
use quickcheck::{Arbitrary, Gen};

use alga::general::RealField;
use num::Zero;
use rand::distributions::{Distribution, OpenClosed01, Standard};
use rand::Rng;
use std::ops::Neg;

use crate::base::dimension::{U1, U2, U3};
use crate::base::storage::Storage;
use crate::base::{Matrix2, Matrix3, MatrixN, Unit, Vector, Vector1, Vector3, VectorN};

use crate::geometry::{Rotation2, Rotation3, UnitComplex, UnitQuaternion};

/*
 *
 * 2D Rotation matrix.
 *
 */
impl<N: RealField> Rotation2<N> {
    /// Builds a 2 dimensional rotation matrix from an angle in radian.
    ///
    /// # Example
    ///
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use std::f32;
    /// # use nalgebra::{Rotation2, Point2};
    /// let rot = Rotation2::new(f32::consts::FRAC_PI_2);
    ///
    /// assert_relative_eq!(rot * Point2::new(3.0, 4.0), Point2::new(-4.0, 3.0));
    /// ```
    pub fn new(angle: N) -> Self {
        let (sia, coa) = angle.sin_cos();
        Self::from_matrix_unchecked(Matrix2::new(coa, -sia, sia, coa))
    }

    /// Builds a 2 dimensional rotation matrix from an angle in radian wrapped in a 1-dimensional vector.
    ///
    ///
    /// This is generally used in the context of generic programming. Using
    /// the `::new(angle)` method instead is more common.
    #[inline]
    pub fn from_scaled_axis<SB: Storage<N, U1>>(axisangle: Vector<N, U1, SB>) -> Self {
        Self::new(axisangle[0])
    }

    /// Builds a rotation matrix by extracting the rotation part of the given transformation `m`.
    ///
    /// This is an iterative method. See `.from_matrix_eps` to provide mover
    /// convergence parameters and starting solution.
    /// This implements "A Robust Method to Extract the Rotational Part of Deformations" by Müller et al.
    pub fn from_matrix(m: &Matrix2<N>) -> Self {
        Self::from_matrix_eps(m, N::default_epsilon(), 0, Self::identity())
    }

    /// Builds a rotation matrix by extracting the rotation part of the given transformation `m`.
    ///
    /// This implements "A Robust Method to Extract the Rotational Part of Deformations" by Müller et al.
    ///
    /// # Parameters
    ///
    /// * `m`: the matrix from which the rotational part is to be extracted.
    /// * `eps`: the angular errors tolerated between the current rotation and the optimal one.
    /// * `max_iter`: the maximum number of iterations. Loops indefinitely until convergence if set to `0`.
    /// * `guess`: an estimate of the solution. Convergence will be significantly faster if an initial solution close
    ///           to the actual solution is provided. Can be set to `Rotation2::identity()` if no other
    ///           guesses come to mind.
    pub fn from_matrix_eps(m: &Matrix2<N>, eps: N, mut max_iter: usize, guess: Self) -> Self {
        if max_iter == 0 {
            max_iter = usize::max_value();
        }

        let mut rot = guess.into_inner();

        for _ in 0..max_iter {
            let axis = rot.column(0).perp(&m.column(0)) +
                rot.column(1).perp(&m.column(1));
            let denom = rot.column(0).dot(&m.column(0)) +
                rot.column(1).dot(&m.column(1));

            let angle = axis / (denom.abs() + N::default_epsilon());
            if angle.abs() > eps {
                rot = Self::new(angle) * rot;
            } else {
                break;
            }
        }

        Self::from_matrix_unchecked(rot)
    }

    /// The rotation matrix required to align `a` and `b` but with its angle.
    ///
    /// This is the rotation `R` such that `(R * a).angle(b) == 0 && (R * a).dot(b).is_positive()`.
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{Vector2, Rotation2};
    /// let a = Vector2::new(1.0, 2.0);
    /// let b = Vector2::new(2.0, 1.0);
    /// let rot = Rotation2::rotation_between(&a, &b);
    /// assert_relative_eq!(rot * a, b);
    /// assert_relative_eq!(rot.inverse() * b, a);
    /// ```
    #[inline]
    pub fn rotation_between<SB, SC>(a: &Vector<N, U2, SB>, b: &Vector<N, U2, SC>) -> Self
    where
        SB: Storage<N, U2>,
        SC: Storage<N, U2>,
    {
        crate::convert(UnitComplex::rotation_between(a, b).to_rotation_matrix())
    }

    /// The smallest rotation needed to make `a` and `b` collinear and point toward the same
    /// direction, raised to the power `s`.
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{Vector2, Rotation2};
    /// let a = Vector2::new(1.0, 2.0);
    /// let b = Vector2::new(2.0, 1.0);
    /// let rot2 = Rotation2::scaled_rotation_between(&a, &b, 0.2);
    /// let rot5 = Rotation2::scaled_rotation_between(&a, &b, 0.5);
    /// assert_relative_eq!(rot2 * rot2 * rot2 * rot2 * rot2 * a, b, epsilon = 1.0e-6);
    /// assert_relative_eq!(rot5 * rot5 * a, b, epsilon = 1.0e-6);
    /// ```
    #[inline]
    pub fn scaled_rotation_between<SB, SC>(
        a: &Vector<N, U2, SB>,
        b: &Vector<N, U2, SC>,
        s: N,
    ) -> Self
    where
        SB: Storage<N, U2>,
        SC: Storage<N, U2>,
    {
        crate::convert(UnitComplex::scaled_rotation_between(a, b, s).to_rotation_matrix())
    }

    /// The rotation angle.
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::Rotation2;
    /// let rot = Rotation2::new(1.78);
    /// assert_relative_eq!(rot.angle(), 1.78);
    /// ```
    #[inline]
    pub fn angle(&self) -> N {
        self.matrix()[(1, 0)].atan2(self.matrix()[(0, 0)])
    }

    /// The rotation angle needed to make `self` and `other` coincide.
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::Rotation2;
    /// let rot1 = Rotation2::new(0.1);
    /// let rot2 = Rotation2::new(1.7);
    /// assert_relative_eq!(rot1.angle_to(&rot2), 1.6);
    /// ```
    #[inline]
    pub fn angle_to(&self, other: &Self) -> N {
        self.rotation_to(other).angle()
    }

    /// The rotation matrix needed to make `self` and `other` coincide.
    ///
    /// The result is such that: `self.rotation_to(other) * self == other`.
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::Rotation2;
    /// let rot1 = Rotation2::new(0.1);
    /// let rot2 = Rotation2::new(1.7);
    /// let rot_to = rot1.rotation_to(&rot2);
    ///
    /// assert_relative_eq!(rot_to * rot1, rot2);
    /// assert_relative_eq!(rot_to.inverse() * rot2, rot1);
    /// ```
    #[inline]
    pub fn rotation_to(&self, other: &Self) -> Self {
        other * self.inverse()
    }


    /// Ensure this rotation is an orthonormal rotation matrix. This is useful when repeated
    /// computations might cause the matrix from progressively not being orthonormal anymore.
    #[inline]
    pub fn renormalize(&mut self) {
        let mut c = UnitComplex::from(*self);
        let _ = c.renormalize();

        *self = Self::from_matrix_eps(self.matrix(), N::default_epsilon(), 0, c.into())
    }


    /// Raise the quaternion to a given floating power, i.e., returns the rotation with the angle
    /// of `self` multiplied by `n`.
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::Rotation2;
    /// let rot = Rotation2::new(0.78);
    /// let pow = rot.powf(2.0);
    /// assert_relative_eq!(pow.angle(), 2.0 * 0.78);
    /// ```
    #[inline]
    pub fn powf(&self, n: N) -> Self {
        Self::new(self.angle() * n)
    }

    /// The rotation angle returned as a 1-dimensional vector.
    ///
    /// This is generally used in the context of generic programming. Using
    /// the `.angle()` method instead is more common.
    #[inline]
    pub fn scaled_axis(&self) -> VectorN<N, U1> {
        Vector1::new(self.angle())
    }
}

impl<N: RealField> Distribution<Rotation2<N>> for Standard
where OpenClosed01: Distribution<N>
{
    /// Generate a uniformly distributed random rotation.
    #[inline]
    fn sample<'a, R: Rng + ?Sized>(&self, rng: &'a mut R) -> Rotation2<N> {
        Rotation2::new(rng.sample(OpenClosed01) * N::two_pi())
    }
}

#[cfg(feature = "arbitrary")]
impl<N: RealField + Arbitrary> Arbitrary for Rotation2<N>
where Owned<N, U2, U2>: Send
{
    #[inline]
    fn arbitrary<G: Gen>(g: &mut G) -> Self {
        Self::new(N::arbitrary(g))
    }
}

/*
 *
 * 3D Rotation matrix.
 *
 */
impl<N: RealField> Rotation3<N> {
    /// Builds a 3 dimensional rotation matrix from an axis and an angle.
    ///
    /// # Arguments
    ///   * `axisangle` - A vector representing the rotation. Its magnitude is the amount of rotation
    ///   in radian. Its direction is the axis of rotation.
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use std::f32;
    /// # use nalgebra::{Rotation3, Point3, Vector3};
    /// let axisangle = Vector3::y() * f32::consts::FRAC_PI_2;
    /// // Point and vector being transformed in the tests.
    /// let pt = Point3::new(4.0, 5.0, 6.0);
    /// let vec = Vector3::new(4.0, 5.0, 6.0);
    /// let rot = Rotation3::new(axisangle);
    ///
    /// assert_relative_eq!(rot * pt, Point3::new(6.0, 5.0, -4.0), epsilon = 1.0e-6);
    /// assert_relative_eq!(rot * vec, Vector3::new(6.0, 5.0, -4.0), epsilon = 1.0e-6);
    ///
    /// // A zero vector yields an identity.
    /// assert_eq!(Rotation3::new(Vector3::<f32>::zeros()), Rotation3::identity());
    /// ```
    pub fn new<SB: Storage<N, U3>>(axisangle: Vector<N, U3, SB>) -> Self {
        let axisangle = axisangle.into_owned();
        let (axis, angle) = Unit::new_and_get(axisangle);
        Self::from_axis_angle(&axis, angle)
    }

    /// Builds a rotation matrix by extracting the rotation part of the given transformation `m`.
    ///
    /// This is an iterative method. See `.from_matrix_eps` to provide mover
    /// convergence parameters and starting solution.
    /// This implements "A Robust Method to Extract the Rotational Part of Deformations" by Müller et al.
    pub fn from_matrix(m: &Matrix3<N>) -> Self {
        Self::from_matrix_eps(m, N::default_epsilon(), 0, Self::identity())
    }

    /// Builds a rotation matrix by extracting the rotation part of the given transformation `m`.
    ///
    /// This implements "A Robust Method to Extract the Rotational Part of Deformations" by Müller et al.
    ///
    /// # Parameters
    ///
    /// * `m`: the matrix from which the rotational part is to be extracted.
    /// * `eps`: the angular errors tolerated between the current rotation and the optimal one.
    /// * `max_iter`: the maximum number of iterations. Loops indefinitely until convergence if set to `0`.
    /// * `guess`: a guess of the solution. Convergence will be significantly faster if an initial solution close
    ///           to the actual solution is provided. Can be set to `Rotation3::identity()` if no other
    ///           guesses come to mind.
    pub fn from_matrix_eps(m: &Matrix3<N>, eps: N, mut max_iter: usize, guess: Self) -> Self {
        if max_iter == 0 {
            max_iter = usize::max_value();
        }

        let mut rot = guess.into_inner();

        for _ in 0..max_iter {
            let axis = rot.column(0).cross(&m.column(0)) +
                rot.column(1).cross(&m.column(1)) +
                rot.column(2).cross(&m.column(2));
            let denom = rot.column(0).dot(&m.column(0)) +
                rot.column(1).dot(&m.column(1)) +
                rot.column(2).dot(&m.column(2));

            let axisangle = axis / (denom.abs() + N::default_epsilon());

            if let Some((axis, angle)) = Unit::try_new_and_get(axisangle, eps) {
                rot = Rotation3::from_axis_angle(&axis, angle) * rot;
            } else {
                break;
            }
        }

        Self::from_matrix_unchecked(rot)
    }

    /// Builds a 3D rotation matrix from an axis scaled by the rotation angle.
    ///
    /// This is the same as `Self::new(axisangle)`.
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use std::f32;
    /// # use nalgebra::{Rotation3, Point3, Vector3};
    /// let axisangle = Vector3::y() * f32::consts::FRAC_PI_2;
    /// // Point and vector being transformed in the tests.
    /// let pt = Point3::new(4.0, 5.0, 6.0);
    /// let vec = Vector3::new(4.0, 5.0, 6.0);
    /// let rot = Rotation3::new(axisangle);
    ///
    /// assert_relative_eq!(rot * pt, Point3::new(6.0, 5.0, -4.0), epsilon = 1.0e-6);
    /// assert_relative_eq!(rot * vec, Vector3::new(6.0, 5.0, -4.0), epsilon = 1.0e-6);
    ///
    /// // A zero vector yields an identity.
    /// assert_eq!(Rotation3::from_scaled_axis(Vector3::<f32>::zeros()), Rotation3::identity());
    /// ```
    pub fn from_scaled_axis<SB: Storage<N, U3>>(axisangle: Vector<N, U3, SB>) -> Self {
        Self::new(axisangle)
    }

    /// Builds a 3D rotation matrix from an axis and a rotation angle.
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use std::f32;
    /// # use nalgebra::{Rotation3, Point3, Vector3};
    /// let axis = Vector3::y_axis();
    /// let angle = f32::consts::FRAC_PI_2;
    /// // Point and vector being transformed in the tests.
    /// let pt = Point3::new(4.0, 5.0, 6.0);
    /// let vec = Vector3::new(4.0, 5.0, 6.0);
    /// let rot = Rotation3::from_axis_angle(&axis, angle);
    ///
    /// assert_eq!(rot.axis().unwrap(), axis);
    /// assert_eq!(rot.angle(), angle);
    /// assert_relative_eq!(rot * pt, Point3::new(6.0, 5.0, -4.0), epsilon = 1.0e-6);
    /// assert_relative_eq!(rot * vec, Vector3::new(6.0, 5.0, -4.0), epsilon = 1.0e-6);
    ///
    /// // A zero vector yields an identity.
    /// assert_eq!(Rotation3::from_scaled_axis(Vector3::<f32>::zeros()), Rotation3::identity());
    /// ```
    pub fn from_axis_angle<SB>(axis: &Unit<Vector<N, U3, SB>>, angle: N) -> Self
    where SB: Storage<N, U3> {
        if angle.is_zero() {
            Self::identity()
        } else {
            let ux = axis.as_ref()[0];
            let uy = axis.as_ref()[1];
            let uz = axis.as_ref()[2];
            let sqx = ux * ux;
            let sqy = uy * uy;
            let sqz = uz * uz;
            let (sin, cos) = angle.sin_cos();
            let one_m_cos = N::one() - cos;

            Self::from_matrix_unchecked(MatrixN::<N, U3>::new(
                sqx + (N::one() - sqx) * cos,
                ux * uy * one_m_cos - uz * sin,
                ux * uz * one_m_cos + uy * sin,
                ux * uy * one_m_cos + uz * sin,
                sqy + (N::one() - sqy) * cos,
                uy * uz * one_m_cos - ux * sin,
                ux * uz * one_m_cos - uy * sin,
                uy * uz * one_m_cos + ux * sin,
                sqz + (N::one() - sqz) * cos,
            ))
        }
    }

    /// Creates a new rotation from Euler angles.
    ///
    /// The primitive rotations are applied in order: 1 roll − 2 pitch − 3 yaw.
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::Rotation3;
    /// let rot = Rotation3::from_euler_angles(0.1, 0.2, 0.3);
    /// let euler = rot.euler_angles();
    /// assert_relative_eq!(euler.0, 0.1, epsilon = 1.0e-6);
    /// assert_relative_eq!(euler.1, 0.2, epsilon = 1.0e-6);
    /// assert_relative_eq!(euler.2, 0.3, epsilon = 1.0e-6);
    /// ```
    pub fn from_euler_angles(roll: N, pitch: N, yaw: N) -> Self {
        let (sr, cr) = roll.sin_cos();
        let (sp, cp) = pitch.sin_cos();
        let (sy, cy) = yaw.sin_cos();

        Self::from_matrix_unchecked(MatrixN::<N, U3>::new(
            cy * cp,
            cy * sp * sr - sy * cr,
            cy * sp * cr + sy * sr,
            sy * cp,
            sy * sp * sr + cy * cr,
            sy * sp * cr - cy * sr,
            -sp,
            cp * sr,
            cp * cr,
        ))
    }

    /// Creates Euler angles from a rotation.
    ///
    /// The angles are produced in the form (roll, pitch, yaw).
    #[deprecated(note = "This is renamed to use `.euler_angles()`.")]
    pub fn to_euler_angles(&self) -> (N, N, N) {
        self.euler_angles()
    }

    /// Euler angles corresponding to this rotation from a rotation.
    ///
    /// The angles are produced in the form (roll, pitch, yaw).
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::Rotation3;
    /// let rot = Rotation3::from_euler_angles(0.1, 0.2, 0.3);
    /// let euler = rot.euler_angles();
    /// assert_relative_eq!(euler.0, 0.1, epsilon = 1.0e-6);
    /// assert_relative_eq!(euler.1, 0.2, epsilon = 1.0e-6);
    /// assert_relative_eq!(euler.2, 0.3, epsilon = 1.0e-6);
    /// ```
    pub fn euler_angles(&self) -> (N, N, N) {
        // Implementation informed by "Computing Euler angles from a rotation matrix", by Gregory G. Slabaugh
        //  https://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.371.6578
        if self[(2, 0)].abs() < N::one() {
            let yaw = -self[(2, 0)].asin();
            let roll = (self[(2, 1)] / yaw.cos()).atan2(self[(2, 2)] / yaw.cos());
            let pitch = (self[(1, 0)] / yaw.cos()).atan2(self[(0, 0)] / yaw.cos());
            (roll, yaw, pitch)
        } else if self[(2, 0)] <= -N::one() {
            (self[(0, 1)].atan2(self[(0, 2)]), N::frac_pi_2(), N::zero())
        } else {
            (
                -self[(0, 1)].atan2(-self[(0, 2)]),
                -N::frac_pi_2(),
                N::zero(),
            )
        }
    }

    /// Ensure this rotation is an orthonormal rotation matrix. This is useful when repeated
    /// computations might cause the matrix from progressively not being orthonormal anymore.
    #[inline]
    pub fn renormalize(&mut self) {
        let mut c = UnitQuaternion::from(*self);
        let _ = c.renormalize();

        *self = Self::from_matrix_eps(self.matrix(), N::default_epsilon(), 0, c.into())
    }

    /// Creates a rotation that corresponds to the local frame of an observer standing at the
    /// origin and looking toward `dir`.
    ///
    /// It maps the `z` axis to the direction `dir`.
    ///
    /// # Arguments
    ///   * dir - The look direction, that is, direction the matrix `z` axis will be aligned with.
    ///   * up - The vertical direction. The only requirement of this parameter is to not be
    ///   collinear to `dir`. Non-collinearity is not checked.
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use std::f32;
    /// # use nalgebra::{Rotation3, Vector3};
    /// let dir = Vector3::new(1.0, 2.0, 3.0);
    /// let up = Vector3::y();
    ///
    /// let rot = Rotation3::face_towards(&dir, &up);
    /// assert_relative_eq!(rot * Vector3::z(), dir.normalize());
    /// ```
    #[inline]
    pub fn face_towards<SB, SC>(dir: &Vector<N, U3, SB>, up: &Vector<N, U3, SC>) -> Self
    where
        SB: Storage<N, U3>,
        SC: Storage<N, U3>,
    {
        let zaxis = dir.normalize();
        let xaxis = up.cross(&zaxis).normalize();
        let yaxis = zaxis.cross(&xaxis).normalize();

        Self::from_matrix_unchecked(MatrixN::<N, U3>::new(
            xaxis.x, yaxis.x, zaxis.x, xaxis.y, yaxis.y, zaxis.y, xaxis.z, yaxis.z, zaxis.z,
        ))
    }

    /// Deprecated: Use [Rotation3::face_towards] instead.
    #[deprecated(note="renamed to `face_towards`")]
    pub fn new_observer_frames<SB, SC>(dir: &Vector<N, U3, SB>, up: &Vector<N, U3, SC>) -> Self
    where
        SB: Storage<N, U3>,
        SC: Storage<N, U3>,
    {
        Self::face_towards(dir, up)
    }

    /// Builds a right-handed look-at view matrix without translation.
    ///
    /// It maps the view direction `dir` to the **negative** `z` axis.
    /// This conforms to the common notion of right handed look-at matrix from the computer
    /// graphics community.
    ///
    /// # Arguments
    ///   * dir - The direction toward which the camera looks.
    ///   * up - A vector approximately aligned with required the vertical axis. The only
    ///   requirement of this parameter is to not be collinear to `dir`.
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use std::f32;
    /// # use nalgebra::{Rotation3, Vector3};
    /// let dir = Vector3::new(1.0, 2.0, 3.0);
    /// let up = Vector3::y();
    ///
    /// let rot = Rotation3::look_at_rh(&dir, &up);
    /// assert_relative_eq!(rot * dir.normalize(), -Vector3::z());
    /// ```
    #[inline]
    pub fn look_at_rh<SB, SC>(dir: &Vector<N, U3, SB>, up: &Vector<N, U3, SC>) -> Self
    where
        SB: Storage<N, U3>,
        SC: Storage<N, U3>,
    {
        Self::face_towards(&dir.neg(), up).inverse()
    }

    /// Builds a left-handed look-at view matrix without translation.
    ///
    /// It maps the view direction `dir` to the **positive** `z` axis.
    /// This conforms to the common notion of left handed look-at matrix from the computer
    /// graphics community.
    ///
    /// # Arguments
    ///   * dir - The direction toward which the camera looks.
    ///   * up - A vector approximately aligned with required the vertical axis. The only
    ///   requirement of this parameter is to not be collinear to `dir`.
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use std::f32;
    /// # use nalgebra::{Rotation3, Vector3};
    /// let dir = Vector3::new(1.0, 2.0, 3.0);
    /// let up = Vector3::y();
    ///
    /// let rot = Rotation3::look_at_lh(&dir, &up);
    /// assert_relative_eq!(rot * dir.normalize(), Vector3::z());
    /// ```
    #[inline]
    pub fn look_at_lh<SB, SC>(dir: &Vector<N, U3, SB>, up: &Vector<N, U3, SC>) -> Self
    where
        SB: Storage<N, U3>,
        SC: Storage<N, U3>,
    {
        Self::face_towards(dir, up).inverse()
    }

    /// The rotation matrix required to align `a` and `b` but with its angle.
    ///
    /// This is the rotation `R` such that `(R * a).angle(b) == 0 && (R * a).dot(b).is_positive()`.
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{Vector3, Rotation3};
    /// let a = Vector3::new(1.0, 2.0, 3.0);
    /// let b = Vector3::new(3.0, 1.0, 2.0);
    /// let rot = Rotation3::rotation_between(&a, &b).unwrap();
    /// assert_relative_eq!(rot * a, b, epsilon = 1.0e-6);
    /// assert_relative_eq!(rot.inverse() * b, a, epsilon = 1.0e-6);
    /// ```
    #[inline]
    pub fn rotation_between<SB, SC>(a: &Vector<N, U3, SB>, b: &Vector<N, U3, SC>) -> Option<Self>
    where
        SB: Storage<N, U3>,
        SC: Storage<N, U3>,
    {
        Self::scaled_rotation_between(a, b, N::one())
    }

    /// The smallest rotation needed to make `a` and `b` collinear and point toward the same
    /// direction, raised to the power `s`.
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{Vector3, Rotation3};
    /// let a = Vector3::new(1.0, 2.0, 3.0);
    /// let b = Vector3::new(3.0, 1.0, 2.0);
    /// let rot2 = Rotation3::scaled_rotation_between(&a, &b, 0.2).unwrap();
    /// let rot5 = Rotation3::scaled_rotation_between(&a, &b, 0.5).unwrap();
    /// assert_relative_eq!(rot2 * rot2 * rot2 * rot2 * rot2 * a, b, epsilon = 1.0e-6);
    /// assert_relative_eq!(rot5 * rot5 * a, b, epsilon = 1.0e-6);
    /// ```
    #[inline]
    pub fn scaled_rotation_between<SB, SC>(
        a: &Vector<N, U3, SB>,
        b: &Vector<N, U3, SC>,
        n: N,
    ) -> Option<Self>
    where
        SB: Storage<N, U3>,
        SC: Storage<N, U3>,
    {
        // FIXME: code duplication with Rotation.
        if let (Some(na), Some(nb)) = (a.try_normalize(N::zero()), b.try_normalize(N::zero())) {
            let c = na.cross(&nb);

            if let Some(axis) = Unit::try_new(c, N::default_epsilon()) {
                return Some(Self::from_axis_angle(&axis, na.dot(&nb).acos() * n));
            }

            // Zero or PI.
            if na.dot(&nb) < N::zero() {
                // PI
                //
                // The rotation axis is undefined but the angle not zero. This is not a
                // simple rotation.
                return None;
            }
        }

        Some(Self::identity())
    }

    /// The rotation angle in [0; pi].
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{Unit, Rotation3, Vector3};
    /// let axis = Unit::new_normalize(Vector3::new(1.0, 2.0, 3.0));
    /// let rot = Rotation3::from_axis_angle(&axis, 1.78);
    /// assert_relative_eq!(rot.angle(), 1.78);
    /// ```
    #[inline]
    pub fn angle(&self) -> N {
        ((self.matrix()[(0, 0)] + self.matrix()[(1, 1)] + self.matrix()[(2, 2)] - N::one())
            / crate::convert(2.0))
        .acos()
    }

    /// The rotation axis. Returns `None` if the rotation angle is zero or PI.
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{Rotation3, Vector3, Unit};
    /// let axis = Unit::new_normalize(Vector3::new(1.0, 2.0, 3.0));
    /// let angle = 1.2;
    /// let rot = Rotation3::from_axis_angle(&axis, angle);
    /// assert_relative_eq!(rot.axis().unwrap(), axis);
    ///
    /// // Case with a zero angle.
    /// let rot = Rotation3::from_axis_angle(&axis, 0.0);
    /// assert!(rot.axis().is_none());
    /// ```
    #[inline]
    pub fn axis(&self) -> Option<Unit<Vector3<N>>> {
        let axis = VectorN::<N, U3>::new(
            self.matrix()[(2, 1)] - self.matrix()[(1, 2)],
            self.matrix()[(0, 2)] - self.matrix()[(2, 0)],
            self.matrix()[(1, 0)] - self.matrix()[(0, 1)],
        );

        Unit::try_new(axis, N::default_epsilon())
    }

    /// The rotation axis multiplied by the rotation angle.
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{Rotation3, Vector3, Unit};
    /// let axisangle = Vector3::new(0.1, 0.2, 0.3);
    /// let rot = Rotation3::new(axisangle);
    /// assert_relative_eq!(rot.scaled_axis(), axisangle, epsilon = 1.0e-6);
    /// ```
    #[inline]
    pub fn scaled_axis(&self) -> Vector3<N> {
        if let Some(axis) = self.axis() {
            axis.into_inner() * self.angle()
        } else {
            Vector::zero()
        }
    }

    /// The rotation axis and angle in ]0, pi] of this unit quaternion.
    ///
    /// Returns `None` if the angle is zero.
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{Rotation3, Vector3, Unit};
    /// let axis = Unit::new_normalize(Vector3::new(1.0, 2.0, 3.0));
    /// let angle = 1.2;
    /// let rot = Rotation3::from_axis_angle(&axis, angle);
    /// let axis_angle = rot.axis_angle().unwrap();
    /// assert_relative_eq!(axis_angle.0, axis);
    /// assert_relative_eq!(axis_angle.1, angle);
    ///
    /// // Case with a zero angle.
    /// let rot = Rotation3::from_axis_angle(&axis, 0.0);
    /// assert!(rot.axis_angle().is_none());
    /// ```
    #[inline]
    pub fn axis_angle(&self) -> Option<(Unit<Vector3<N>>, N)> {
        if let Some(axis) = self.axis() {
            Some((axis, self.angle()))
        } else {
            None
        }
    }

    /// The rotation angle needed to make `self` and `other` coincide.
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{Rotation3, Vector3};
    /// let rot1 = Rotation3::from_axis_angle(&Vector3::y_axis(), 1.0);
    /// let rot2 = Rotation3::from_axis_angle(&Vector3::x_axis(), 0.1);
    /// assert_relative_eq!(rot1.angle_to(&rot2), 1.0045657, epsilon = 1.0e-6);
    /// ```
    #[inline]
    pub fn angle_to(&self, other: &Self) -> N {
        self.rotation_to(other).angle()
    }

    /// The rotation matrix needed to make `self` and `other` coincide.
    ///
    /// The result is such that: `self.rotation_to(other) * self == other`.
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{Rotation3, Vector3};
    /// let rot1 = Rotation3::from_axis_angle(&Vector3::y_axis(), 1.0);
    /// let rot2 = Rotation3::from_axis_angle(&Vector3::x_axis(), 0.1);
    /// let rot_to = rot1.rotation_to(&rot2);
    /// assert_relative_eq!(rot_to * rot1, rot2, epsilon = 1.0e-6);
    /// ```
    #[inline]
    pub fn rotation_to(&self, other: &Self) -> Self {
        other * self.inverse()
    }

    /// Raise the quaternion to a given floating power, i.e., returns the rotation with the same
    /// axis as `self` and an angle equal to `self.angle()` multiplied by `n`.
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{Rotation3, Vector3, Unit};
    /// let axis = Unit::new_normalize(Vector3::new(1.0, 2.0, 3.0));
    /// let angle = 1.2;
    /// let rot = Rotation3::from_axis_angle(&axis, angle);
    /// let pow = rot.powf(2.0);
    /// assert_relative_eq!(pow.axis().unwrap(), axis, epsilon = 1.0e-6);
    /// assert_eq!(pow.angle(), 2.4);
    /// ```
    #[inline]
    pub fn powf(&self, n: N) -> Self {
        if let Some(axis) = self.axis() {
            Self::from_axis_angle(&axis, self.angle() * n)
        } else if self.matrix()[(0, 0)] < N::zero() {
            let minus_id = MatrixN::<N, U3>::from_diagonal_element(-N::one());
            Self::from_matrix_unchecked(minus_id)
        } else {
            Self::identity()
        }
    }
}

impl<N: RealField> Distribution<Rotation3<N>> for Standard
where OpenClosed01: Distribution<N>
{
    /// Generate a uniformly distributed random rotation.
    #[inline]
    fn sample<'a, R: Rng + ?Sized>(&self, rng: &mut R) -> Rotation3<N> {
        // James Arvo.
        // Fast random rotation matrices.
        // In D. Kirk, editor, Graphics Gems III, pages 117-120. Academic, New York, 1992.

        // Compute a random rotation around Z
        let theta = N::two_pi() * rng.sample(OpenClosed01);
        let (ts, tc) = theta.sin_cos();
        let a = MatrixN::<N, U3>::new(
            tc,
            ts,
            N::zero(),
            -ts,
            tc,
            N::zero(),
            N::zero(),
            N::zero(),
            N::one(),
        );

        // Compute a random rotation *of* Z
        let phi = N::two_pi() * rng.sample(OpenClosed01);
        let z = rng.sample(OpenClosed01);
        let (ps, pc) = phi.sin_cos();
        let sqrt_z = z.sqrt();
        let v = Vector3::new(pc * sqrt_z, ps * sqrt_z, (N::one() - z).sqrt());
        let mut b = v * v.transpose();
        b += b;
        b -= MatrixN::<N, U3>::identity();

        Rotation3::from_matrix_unchecked(b * a)
    }
}

#[cfg(feature = "arbitrary")]
impl<N: RealField + Arbitrary> Arbitrary for Rotation3<N>
where
    Owned<N, U3, U3>: Send,
    Owned<N, U3>: Send,
{
    #[inline]
    fn arbitrary<G: Gen>(g: &mut G) -> Self {
        Self::new(VectorN::arbitrary(g))
    }
}
