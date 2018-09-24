#[cfg(feature = "arbitrary")]
use quickcheck::{Arbitrary, Gen};

use num::One;
use num_complex::Complex;
use rand::distributions::{Distribution, Standard, OpenClosed01};
use rand::Rng;

use alga::general::Real;
use base::allocator::Allocator;
use base::dimension::{U1, U2};
use base::storage::Storage;
use base::{DefaultAllocator, Unit, Vector};
use geometry::{Rotation, UnitComplex};

impl<N: Real> UnitComplex<N> {
    /// The unit complex number multiplicative identity.
    #[inline]
    pub fn identity() -> Self {
        Self::new_unchecked(Complex::new(N::one(), N::zero()))
    }

    /// Builds the unit complex number corresponding to the rotation with the angle.
    #[inline]
    pub fn new(angle: N) -> Self {
        let (sin, cos) = angle.sin_cos();
        Self::from_cos_sin_unchecked(cos, sin)
    }

    /// Builds the unit complex number corresponding to the rotation with the angle.
    ///
    /// Same as `Self::new(angle)`.
    #[inline]
    pub fn from_angle(angle: N) -> Self {
        Self::new(angle)
    }

    /// Builds the unit complex number from the sinus and cosinus of the rotation angle.
    ///
    /// The input values are not checked.
    #[inline]
    pub fn from_cos_sin_unchecked(cos: N, sin: N) -> Self {
        UnitComplex::new_unchecked(Complex::new(cos, sin))
    }

    /// Builds a unit complex rotation from an angle in radian wrapped in a 1-dimensional vector.
    ///
    /// Equivalent to `Self::new(axisangle[0])`.
    #[inline]
    pub fn from_scaled_axis<SB: Storage<N, U1, U1>>(axisangle: Vector<N, U1, SB>) -> Self {
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
    /// The input complex number will be normalized. Returns the complex number norm as well.
    #[inline]
    pub fn from_complex_and_get(q: Complex<N>) -> (Self, N) {
        let norm = (q.im * q.im + q.re * q.re).sqrt();
        (Self::new_unchecked(q / norm), norm)
    }

    /// Builds the unit complex number from the corresponding 2D rotation matrix.
    #[inline]
    pub fn from_rotation_matrix(rotmat: &Rotation<N, U2>) -> Self
    where
        DefaultAllocator: Allocator<N, U2, U2>,
    {
        Self::new_unchecked(Complex::new(rotmat[(0, 0)], rotmat[(1, 0)]))
    }

    /// The unit complex needed to make `a` and `b` be collinear and point toward the same
    /// direction.
    #[inline]
    pub fn rotation_between<SB, SC>(a: &Vector<N, U2, SB>, b: &Vector<N, U2, SC>) -> Self
    where
        SB: Storage<N, U2, U1>,
        SC: Storage<N, U2, U1>,
    {
        Self::scaled_rotation_between(a, b, N::one())
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
        SB: Storage<N, U2, U1>,
        SC: Storage<N, U2, U1>,
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
where
    OpenClosed01: Distribution<N>,
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
