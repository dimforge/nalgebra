use std::fmt;
use approx::ApproxEq;
use num_complex::Complex;

use alga::general::Real;
use core::{Unit, Vector1, Matrix2, Matrix3};
use geometry::Rotation2;

/// A complex number with a norm equal to 1.
///
/// <center>
/// <big><b>
/// Due to a [bug](https://github.com/rust-lang/rust/issues/32077) in rustdoc, the documentation
/// below has been written manually lists only method signatures.<br>
/// Trait implementations are not listed either.
/// </b></big>
/// </center>
///
/// Please refer directly to the documentation written above each function definition on the source
/// code for more details.
///
///
/// <h2 id="methods">Methods</h2>
///
/// <h4 class="method"><span class="invisible">
/// <code>fn <a class="fnname">angle</a>(&self) -> N</code>
/// </h4>
///
/// <h4 class="method"><span class="invisible">
/// <code>fn <a class="fnname">angle_to</a>(&self, other: &Self) -> N</code>
/// </h4>
///
/// <h4 class="method"><span class="invisible">
/// <code>fn <a class="fnname">complex</a>(&self) -> &Complex</code>
/// </h4>
///
/// <h4 class="method"><span class="invisible">
/// <code>fn <a class="fnname">conjugate</a>(&self) -> Self</code>
/// </h4>
///
/// <h4 class="method"><span class="invisible">
/// <code>fn <a class="fnname">conjugate_mut</a>(&mut self)</code>
/// </h4>
///
/// <h4 class="method"><span class="invisible">
/// <code>fn <a class="fnname">from_angle</a>(angle: N) -> Self</code>
/// </h4>
///
/// <h4 class="method"><span class="invisible">
/// <code>fn <a class="fnname">from_complex</a>(q: Complex) -> Self</code>
/// </h4>
///
/// <h4 class="method"><span class="invisible">
/// <code>fn <a class="fnname">from_rotation_matrix</a>(rotmat: &Rotation) -> Self</code>
/// </h4>
///
/// <h4 class="method"><span class="invisible">
/// <code>fn <a class="fnname">from_scaled_axis</a>(axisangle: Vector) -> Self</code>
/// </h4>
///
/// <h4 class="method"><span class="invisible">
/// <code>fn <a class="fnname">identity</a>() -> Self</code>
/// </h4>
///
/// <h4 class="method"><span class="invisible">
/// <code>fn <a class="fnname">inverse</a>(&self) -> Self</code>
/// </h4>
///
/// <h4 class="method"><span class="invisible">
/// <code>fn <a class="fnname">inverse_mut</a>(&mut self)</code>
/// </h4>
///
/// <h4 class="method"><span class="invisible">
/// <code>fn <a class="fnname">new</a>(angle: N) -> Self</code>
/// </h4>
///
/// <h4 class="method"><span class="invisible">
/// <code>fn <a class="fnname">powf</a>(&self, n: N) -> Self</code>
/// </h4>
///
/// <h4 class="method"><span class="invisible">
/// <code>fn <a class="fnname">rotation_between</a>(a: &Vector<N, U2, SB>, b: &Vector) -> Self</code>
/// </h4>
///
/// <h4 class="method"><span class="invisible">
/// <code>fn <a class="fnname">rotation_to</a>(&self, other: &Self) -> Self</code>
/// </h4>
///
/// <h4 class="method"><span class="invisible">
/// <code>fn <a class="fnname">scaled_axis</a>(&self) -> Vector1</code>
/// </h4>
///
/// <h4 class="method"><span class="invisible">
/// <code>fn <a class="fnname">scaled_rotation_between</a>(a: &Vector<N, U2, SB>, b: &Vector, s: N) -> Self</code>
/// </h4>
///
/// <h4 class="method"><span class="invisible">
/// <code>fn <a class="fnname">to_homogeneous</a>(&self) -> Matrix3</code>
/// </h4>
///
/// <h4 class="method"><span class="invisible">
/// <code>fn <a class="fnname">to_rotation_matrix</a>(&self) -> Rotation2</code>
/// </h4>
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

        Rotation2::from_matrix_unchecked(Matrix2::new(r, -i,
                                                      i,  r))
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

impl<N: Real> ApproxEq for UnitComplex<N> {
    type Epsilon = N;

    #[inline]
    fn default_epsilon() -> Self::Epsilon {
        N::default_epsilon()
    }

    #[inline]
    fn default_max_relative() -> Self::Epsilon {
        N::default_max_relative()
    }

    #[inline]
    fn default_max_ulps() -> u32 {
        N::default_max_ulps()
    }

    #[inline]
    fn relative_eq(&self, other: &Self, epsilon: Self::Epsilon, max_relative: Self::Epsilon) -> bool {
        self.re.relative_eq(&other.re, epsilon, max_relative) &&
        self.im.relative_eq(&other.im, epsilon, max_relative)
    }

    #[inline]
    fn ulps_eq(&self, other: &Self, epsilon: Self::Epsilon, max_ulps: u32) -> bool {
        self.re.ulps_eq(&other.re, epsilon, max_ulps) &&
        self.im.ulps_eq(&other.im, epsilon, max_ulps)
    }
}
