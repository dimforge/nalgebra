use approx::{AbsDiffEq, RelativeEq, UlpsEq};
use num_complex::Complex;
use std::fmt;

use alga::general::Real;
use base::{Matrix2, Matrix3, Unit, Vector1};
use geometry::Rotation2;

/// A complex number with a norm equal to 1.
pub type UnitComplex<N> = Unit<Complex<N>>;

impl<N: Real> UnitComplex<N> {
    /// The rotation angle in `]-pi; pi]` of this unit complex number.
    #[inline]
    pub fn angle(&self) -> N {
        self.im.atan2(self.re)
    }

    /// The sine of the rotation angle.
    #[inline]
    pub fn sin_angle(&self) -> N {
        self.im
    }

    /// The cosine of the rotation angle.
    #[inline]
    pub fn cos_angle(&self) -> N {
        self.re
    }

    /// The rotation angle returned as a 1-dimensional vector.
    #[inline]
    pub fn scaled_axis(&self) -> Vector1<N> {
        Vector1::new(self.angle())
    }

    /// The rotation axis and angle in ]0, pi] of this complex number.
    ///
    /// Returns `None` if the angle is zero.
    #[inline]
    pub fn axis_angle(&self) -> Option<(Unit<Vector1<N>>, N)> {
        let ang = self.angle();

        if ang.is_zero() {
            None
        } else if ang.is_sign_negative() {
            Some((Unit::new_unchecked(Vector1::x()), -ang))
        } else {
            Some((Unit::new_unchecked(-Vector1::<N>::x()), ang))
        }
    }

    /// The underlying complex number.
    ///
    /// Same as `self.as_ref()`.
    #[inline]
    pub fn complex(&self) -> &Complex<N> {
        self.as_ref()
    }

    /// Compute the conjugate of this unit complex number.
    #[inline]
    pub fn conjugate(&self) -> Self {
        UnitComplex::new_unchecked(self.conj())
    }

    /// Inverts this complex number if it is not zero.
    #[inline]
    pub fn inverse(&self) -> Self {
        self.conjugate()
    }

    /// The rotation angle needed to make `self` and `other` coincide.
    #[inline]
    pub fn angle_to(&self, other: &Self) -> N {
        let delta = self.rotation_to(other);
        delta.angle()
    }

    /// The unit complex number needed to make `self` and `other` coincide.
    ///
    /// The result is such that: `self.rotation_to(other) * self == other`.
    #[inline]
    pub fn rotation_to(&self, other: &Self) -> Self {
        other / self
    }

    /// Compute in-place the conjugate of this unit complex number.
    #[inline]
    pub fn conjugate_mut(&mut self) {
        let me = self.as_mut_unchecked();
        me.im = -me.im;
    }

    /// Inverts in-place this unit complex number.
    #[inline]
    pub fn inverse_mut(&mut self) {
        self.conjugate_mut()
    }

    /// Raise this unit complex number to a given floating power.
    ///
    /// This returns the unit complex number that identifies a rotation angle equal to
    /// `self.angle() × n`.
    #[inline]
    pub fn powf(&self, n: N) -> Self {
        Self::from_angle(self.angle() * n)
    }

    /// Builds the rotation matrix corresponding to this unit complex number.
    #[inline]
    pub fn to_rotation_matrix(&self) -> Rotation2<N> {
        let r = self.re;
        let i = self.im;

        Rotation2::from_matrix_unchecked(Matrix2::new(r, -i, i, r))
    }

    /// Converts this unit complex number into its equivalent homogeneous transformation matrix.
    #[inline]
    pub fn to_homogeneous(&self) -> Matrix3<N> {
        self.to_rotation_matrix().to_homogeneous()
    }
}

impl<N: Real + fmt::Display> fmt::Display for UnitComplex<N> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "UnitComplex angle: {}", self.angle())
    }
}

impl<N: Real> AbsDiffEq for UnitComplex<N> {
    type Epsilon = N;

    #[inline]
    fn default_epsilon() -> Self::Epsilon {
        N::default_epsilon()
    }

    #[inline]
    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        self.re.abs_diff_eq(&other.re, epsilon) && self.im.abs_diff_eq(&other.im, epsilon)
    }
}

impl<N: Real> RelativeEq for UnitComplex<N> {
    #[inline]
    fn default_max_relative() -> Self::Epsilon {
        N::default_max_relative()
    }

    #[inline]
    fn relative_eq(
        &self,
        other: &Self,
        epsilon: Self::Epsilon,
        max_relative: Self::Epsilon,
    ) -> bool {
        self.re.relative_eq(&other.re, epsilon, max_relative)
            && self.im.relative_eq(&other.im, epsilon, max_relative)
    }
}

impl<N: Real> UlpsEq for UnitComplex<N> {
    #[inline]
    fn default_max_ulps() -> u32 {
        N::default_max_ulps()
    }

    #[inline]
    fn ulps_eq(&self, other: &Self, epsilon: Self::Epsilon, max_ulps: u32) -> bool {
        self.re.ulps_eq(&other.re, epsilon, max_ulps)
            && self.im.ulps_eq(&other.im, epsilon, max_ulps)
    }
}
