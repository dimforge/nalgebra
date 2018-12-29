#[cfg(feature = "arbitrary")]
use quickcheck::{Arbitrary, Gen};

use num::One;
use num_complex::Complex;
use rand::distributions::{Distribution, OpenClosed01, Standard};
use rand::Rng;

use alga::general::Real;
use base::dimension::{U1, U2};
use base::storage::Storage;
use base::{Unit, Vector};
use geometry::{Rotation2, UnitComplex};

impl<N: Real> UnitComplex<N> {
    /// The unit complex number multiplicative identity.
    ///
    /// # Example
    /// ```
    /// # use nalgebra::UnitComplex;
    /// let rot1 = UnitComplex::identity();
    /// let rot2 = UnitComplex::new(1.7);
    ///
    /// assert_eq!(rot1 * rot2, rot2);
    /// assert_eq!(rot2 * rot1, rot2);
    /// ```
    #[inline]
    pub fn identity() -> Self {
        Self::new_unchecked(Complex::new(N::one(), N::zero()))
    }

    /// Builds the unit complex number corresponding to the rotation with the given angle.
    ///
    /// # Example
    ///
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # extern crate nalgebra;
    /// # use std::f32;
    /// # use nalgebra::{UnitComplex, Point2};
    /// let rot = UnitComplex::new(f32::consts::FRAC_PI_2);
    ///
    /// assert_relative_eq!(rot * Point2::new(3.0, 4.0), Point2::new(-4.0, 3.0));
    /// ```
    #[inline]
    pub fn new(angle: N) -> Self {
        let (sin, cos) = angle.sin_cos();
        Self::from_cos_sin_unchecked(cos, sin)
    }

    /// Builds the unit complex number corresponding to the rotation with the angle.
    ///
    /// Same as `Self::new(angle)`.
    ///
    /// # Example
    ///
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # extern crate nalgebra;
    /// # use std::f32;
    /// # use nalgebra::{UnitComplex, Point2};
    /// let rot = UnitComplex::from_angle(f32::consts::FRAC_PI_2);
    ///
    /// assert_relative_eq!(rot * Point2::new(3.0, 4.0), Point2::new(-4.0, 3.0));
    /// ```
    // FIXME: deprecate this.
    #[inline]
    pub fn from_angle(angle: N) -> Self {
        Self::new(angle)
    }

    /// Builds the unit complex number from the sinus and cosinus of the rotation angle.
    ///
    /// The input values are not checked to actually be cosines and sine of the same value.
    /// Is is generally preferable to use the `::new(angle)` constructor instead.
    ///
    /// # Example
    ///
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # extern crate nalgebra;
    /// # use std::f32;
    /// # use nalgebra::{UnitComplex, Vector2, Point2};
    /// let angle = f32::consts::FRAC_PI_2;
    /// let rot = UnitComplex::from_cos_sin_unchecked(angle.cos(), angle.sin());
    ///
    /// assert_relative_eq!(rot * Point2::new(3.0, 4.0), Point2::new(-4.0, 3.0));
    /// ```
    #[inline]
    pub fn from_cos_sin_unchecked(cos: N, sin: N) -> Self {
        UnitComplex::new_unchecked(Complex::new(cos, sin))
    }

    /// Builds a unit complex rotation from an angle in radian wrapped in a 1-dimensional vector.
    ///
    /// This is generally used in the context of generic programming. Using
    /// the `::new(angle)` method instead is more common.
    #[inline]
    pub fn from_scaled_axis<SB: Storage<N, U1>>(axisangle: Vector<N, U1, SB>) -> Self {
        Self::from_angle(axisangle[0])
    }

    /// Creates a new unit complex number from a complex number.
    ///
    /// The input complex number will be normalized.
    #[inline]
    pub fn from_complex(q: Complex<N>) -> Self {
        Self::from_complex_and_get(q).0
    }

    /// Creates a new unit complex number from a complex number.
    ///
    /// The input complex number will be normalized. Returns the norm of the complex number as well.
    #[inline]
    pub fn from_complex_and_get(q: Complex<N>) -> (Self, N) {
        let norm = (q.im * q.im + q.re * q.re).sqrt();
        (Self::new_unchecked(q / norm), norm)
    }

    /// Builds the unit complex number from the corresponding 2D rotation matrix.
    ///
    /// # Example
    /// ```
    /// # use nalgebra::{Rotation2, UnitComplex};
    /// let rot = Rotation2::new(1.7);
    /// let complex = UnitComplex::from_rotation_matrix(&rot);
    /// assert_eq!(complex, UnitComplex::new(1.7));
    /// ```
    // FIXME: add UnitComplex::from(...) instead?
    #[inline]
    pub fn from_rotation_matrix(rotmat: &Rotation2<N>) -> Self {
        Self::new_unchecked(Complex::new(rotmat[(0, 0)], rotmat[(1, 0)]))
    }

    /// The unit complex needed to make `a` and `b` be collinear and point toward the same
    /// direction.
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # extern crate nalgebra;
    /// # use nalgebra::{Vector2, UnitComplex};
    /// let a = Vector2::new(1.0, 2.0);
    /// let b = Vector2::new(2.0, 1.0);
    /// let rot = UnitComplex::rotation_between(&a, &b);
    /// assert_relative_eq!(rot * a, b);
    /// assert_relative_eq!(rot.inverse() * b, a);
    /// ```
    #[inline]
    pub fn rotation_between<SB, SC>(a: &Vector<N, U2, SB>, b: &Vector<N, U2, SC>) -> Self
    where
        SB: Storage<N, U2>,
        SC: Storage<N, U2>,
    {
        Self::scaled_rotation_between(a, b, N::one())
    }

    /// The smallest rotation needed to make `a` and `b` collinear and point toward the same
    /// direction, raised to the power `s`.
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # extern crate nalgebra;
    /// # use nalgebra::{Vector2, UnitComplex};
    /// let a = Vector2::new(1.0, 2.0);
    /// let b = Vector2::new(2.0, 1.0);
    /// let rot2 = UnitComplex::scaled_rotation_between(&a, &b, 0.2);
    /// let rot5 = UnitComplex::scaled_rotation_between(&a, &b, 0.5);
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
        // FIXME: code duplication with Rotation.
        if let (Some(na), Some(nb)) = (
            Unit::try_new(a.clone_owned(), N::zero()),
            Unit::try_new(b.clone_owned(), N::zero()),
        ) {
            Self::scaled_rotation_between_axis(&na, &nb, s)
        } else {
            Self::identity()
        }
    }

    /// The unit complex needed to make `a` and `b` be collinear and point toward the same
    /// direction.
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # extern crate nalgebra;
    /// # use nalgebra::{Unit, Vector2, UnitComplex};
    /// let a = Unit::new_normalize(Vector2::new(1.0, 2.0));
    /// let b = Unit::new_normalize(Vector2::new(2.0, 1.0));
    /// let rot = UnitComplex::rotation_between_axis(&a, &b);
    /// assert_relative_eq!(rot * a, b);
    /// assert_relative_eq!(rot.inverse() * b, a);
    /// ```
    #[inline]
    pub fn rotation_between_axis<SB, SC>(
        a: &Unit<Vector<N, U2, SB>>,
        b: &Unit<Vector<N, U2, SC>>,
    ) -> Self
    where
        SB: Storage<N, U2>,
        SC: Storage<N, U2>,
    {
        Self::scaled_rotation_between_axis(a, b, N::one())
    }

    /// The smallest rotation needed to make `a` and `b` collinear and point toward the same
    /// direction, raised to the power `s`.
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # extern crate nalgebra;
    /// # use nalgebra::{Unit, Vector2, UnitComplex};
    /// let a = Unit::new_normalize(Vector2::new(1.0, 2.0));
    /// let b = Unit::new_normalize(Vector2::new(2.0, 1.0));
    /// let rot2 = UnitComplex::scaled_rotation_between_axis(&a, &b, 0.2);
    /// let rot5 = UnitComplex::scaled_rotation_between_axis(&a, &b, 0.5);
    /// assert_relative_eq!(rot2 * rot2 * rot2 * rot2 * rot2 * a, b, epsilon = 1.0e-6);
    /// assert_relative_eq!(rot5 * rot5 * a, b, epsilon = 1.0e-6);
    /// ```
    #[inline]
    pub fn scaled_rotation_between_axis<SB, SC>(
        na: &Unit<Vector<N, U2, SB>>,
        nb: &Unit<Vector<N, U2, SC>>,
        s: N,
    ) -> Self
    where
        SB: Storage<N, U2>,
        SC: Storage<N, U2>,
    {
        let sang = na.perp(&nb);
        let cang = na.dot(&nb);

        Self::from_angle(sang.atan2(cang) * s)
    }
}

impl<N: Real> One for UnitComplex<N> {
    #[inline]
    fn one() -> Self {
        Self::identity()
    }
}

impl<N: Real> Distribution<UnitComplex<N>> for Standard
where OpenClosed01: Distribution<N>
{
    /// Generate a uniformly distributed random `UnitComplex`.
    #[inline]
    fn sample<'a, R: Rng + ?Sized>(&self, rng: &mut R) -> UnitComplex<N> {
        UnitComplex::from_angle(rng.sample(OpenClosed01) * N::two_pi())
    }
}

#[cfg(feature = "arbitrary")]
impl<N: Real + Arbitrary> Arbitrary for UnitComplex<N> {
    #[inline]
    fn arbitrary<G: Gen>(g: &mut G) -> Self {
        UnitComplex::from_angle(N::arbitrary(g))
    }
}
