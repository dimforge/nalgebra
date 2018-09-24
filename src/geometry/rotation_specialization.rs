#[cfg(feature = "arbitrary")]
use base::storage::Owned;
#[cfg(feature = "arbitrary")]
use quickcheck::{Arbitrary, Gen};

use alga::general::Real;
use num::Zero;
use rand::distributions::{Distribution, Standard, OpenClosed01};
use rand::Rng;
use std::ops::Neg;

use base::dimension::{U1, U2, U3};
use base::storage::Storage;
use base::{MatrixN, Unit, Vector, Vector1, Vector3, VectorN};

use geometry::{Rotation2, Rotation3, UnitComplex};

/*
 *
 * 2D Rotation matrix.
 *
 */
impl<N: Real> Rotation2<N> {
    /// Builds a 2 dimensional rotation matrix from an angle in radian.
    pub fn new(angle: N) -> Self {
        let (sia, coa) = angle.sin_cos();
        Self::from_matrix_unchecked(MatrixN::<N, U2>::new(coa, -sia, sia, coa))
    }

    /// Builds a 2 dimensional rotation matrix from an angle in radian wrapped in a 1-dimensional vector.
    ///
    /// Equivalent to `Self::new(axisangle[0])`.
    #[inline]
    pub fn from_scaled_axis<SB: Storage<N, U1>>(axisangle: Vector<N, U1, SB>) -> Self {
        Self::new(axisangle[0])
    }

    /// The rotation matrix required to align `a` and `b` but with its angle.
    ///
    /// This is the rotation `R` such that `(R * a).angle(b) == 0 && (R * a).dot(b).is_positive()`.
    #[inline]
    pub fn rotation_between<SB, SC>(a: &Vector<N, U2, SB>, b: &Vector<N, U2, SC>) -> Self
    where
        SB: Storage<N, U2>,
        SC: Storage<N, U2>,
    {
        ::convert(UnitComplex::rotation_between(a, b).to_rotation_matrix())
    }

    /// The smallest rotation needed to make `a` and `b` collinear and point toward the same
    /// direction, raised to the power `s`.
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
        ::convert(UnitComplex::scaled_rotation_between(a, b, s).to_rotation_matrix())
    }
}

impl<N: Real> Rotation2<N> {
    /// The rotation angle.
    #[inline]
    pub fn angle(&self) -> N {
        self.matrix()[(1, 0)].atan2(self.matrix()[(0, 0)])
    }

    /// The rotation angle needed to make `self` and `other` coincide.
    #[inline]
    pub fn angle_to(&self, other: &Rotation2<N>) -> N {
        self.rotation_to(other).angle()
    }

    /// The rotation matrix needed to make `self` and `other` coincide.
    ///
    /// The result is such that: `self.rotation_to(other) * self == other`.
    #[inline]
    pub fn rotation_to(&self, other: &Rotation2<N>) -> Rotation2<N> {
        other * self.inverse()
    }

    /// Raise the quaternion to a given floating power, i.e., returns the rotation with the angle
    /// of `self` multiplied by `n`.
    #[inline]
    pub fn powf(&self, n: N) -> Rotation2<N> {
        Self::new(self.angle() * n)
    }

    /// The rotation angle returned as a 1-dimensional vector.
    #[inline]
    pub fn scaled_axis(&self) -> VectorN<N, U1> {
        Vector1::new(self.angle())
    }
}

impl<N: Real> Distribution<Rotation2<N>> for Standard
where
    OpenClosed01: Distribution<N>,
{
    /// Generate a uniformly distributed random rotation.
    #[inline]
    fn sample<'a, R: Rng + ?Sized>(&self, rng: &'a mut R) -> Rotation2<N> {
        Rotation2::new(rng.sample(OpenClosed01) * N::two_pi())
    }
}

#[cfg(feature = "arbitrary")]
impl<N: Real + Arbitrary> Arbitrary for Rotation2<N>
where
    Owned<N, U2, U2>: Send,
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
impl<N: Real> Rotation3<N> {
    /// Builds a 3 dimensional rotation matrix from an axis and an angle.
    ///
    /// # Arguments
    ///   * `axisangle` - A vector representing the rotation. Its magnitude is the amount of rotation
    ///   in radian. Its direction is the axis of rotation.
    pub fn new<SB: Storage<N, U3>>(axisangle: Vector<N, U3, SB>) -> Self {
        let axisangle = axisangle.into_owned();
        let (axis, angle) = Unit::new_and_get(axisangle);
        Self::from_axis_angle(&axis, angle)
    }

    /// Builds a 3D rotation matrix from an axis scaled by the rotation angle.
    pub fn from_scaled_axis<SB: Storage<N, U3>>(axisangle: Vector<N, U3, SB>) -> Self {
        Self::new(axisangle)
    }

    /// Builds a 3D rotation matrix from an axis and a rotation angle.
    pub fn from_axis_angle<SB>(axis: &Unit<Vector<N, U3, SB>>, angle: N) -> Self
    where
        SB: Storage<N, U3>,
    {
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
    /// The angles are produced in the form (roll, yaw, pitch).
    pub fn to_euler_angles(&self) -> (N, N, N) {
        // Implementation informed by "Computing Euler angles from a rotation matrix", by Gregory G. Slabaugh
        //  http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.371.6578
        if self[(2, 0)].abs() != N::one() {
            let yaw = -self[(2, 0)].asin();
            let roll = (self[(2, 1)] / yaw.cos()).atan2(self[(2, 2)] / yaw.cos());
            let pitch = (self[(1, 0)] / yaw.cos()).atan2(self[(0, 0)] / yaw.cos());
            (roll, yaw, pitch)
        } else if self[(2, 0)] == -N::one() {
            (self[(0, 1)].atan2(self[(0, 2)]), N::frac_pi_2(), N::zero())
        } else {
            (
                -self[(0, 1)].atan2(-self[(0, 2)]),
                -N::frac_pi_2(),
                N::zero(),
            )
        }
    }

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
    pub fn new_observer_frame<SB, SC>(dir: &Vector<N, U3, SB>, up: &Vector<N, U3, SC>) -> Self
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
    pub fn look_at_rh<SB, SC>(dir: &Vector<N, U3, SB>, up: &Vector<N, U3, SC>) -> Self
    where
        SB: Storage<N, U3>,
        SC: Storage<N, U3>,
    {
        Self::new_observer_frame(&dir.neg(), up).inverse()
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
    pub fn look_at_lh<SB, SC>(dir: &Vector<N, U3, SB>, up: &Vector<N, U3, SC>) -> Self
    where
        SB: Storage<N, U3>,
        SC: Storage<N, U3>,
    {
        Self::new_observer_frame(dir, up).inverse()
    }

    /// The rotation matrix required to align `a` and `b` but with its angle.
    ///
    /// This is the rotation `R` such that `(R * a).angle(b) == 0 && (R * a).dot(b).is_positive()`.
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

    /// The rotation angle.
    #[inline]
    pub fn angle(&self) -> N {
        ((self.matrix()[(0, 0)] + self.matrix()[(1, 1)] + self.matrix()[(2, 2)] - N::one())
            / ::convert(2.0))
            .acos()
    }

    /// The rotation axis. Returns `None` if the rotation angle is zero or PI.
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
    #[inline]
    pub fn scaled_axis(&self) -> Vector3<N> {
        if let Some(axis) = self.axis() {
            axis.unwrap() * self.angle()
        } else {
            Vector::zero()
        }
    }

    /// The rotation angle needed to make `self` and `other` coincide.
    #[inline]
    pub fn angle_to(&self, other: &Rotation3<N>) -> N {
        self.rotation_to(other).angle()
    }

    /// The rotation matrix needed to make `self` and `other` coincide.
    ///
    /// The result is such that: `self.rotation_to(other) * self == other`.
    #[inline]
    pub fn rotation_to(&self, other: &Rotation3<N>) -> Rotation3<N> {
        other * self.inverse()
    }

    /// Raise the quaternion to a given floating power, i.e., returns the rotation with the same
    /// axis as `self` and an angle equal to `self.angle()` multiplied by `n`.
    #[inline]
    pub fn powf(&self, n: N) -> Rotation3<N> {
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

impl<N: Real> Distribution<Rotation3<N>> for Standard
where
    OpenClosed01: Distribution<N>,
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
            tc, ts, N::zero(),
            -ts, tc, N::zero(),
            N::zero(), N::zero(), N::one()
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
impl<N: Real + Arbitrary> Arbitrary for Rotation3<N>
where
    Owned<N, U3, U3>: Send,
    Owned<N, U3>: Send,
{
    #[inline]
    fn arbitrary<G: Gen>(g: &mut G) -> Self {
        Self::new(VectorN::arbitrary(g))
    }
}
