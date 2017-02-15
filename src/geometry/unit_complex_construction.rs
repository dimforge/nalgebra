#[cfg(feature = "arbitrary")]
use quickcheck::{Arbitrary, Gen};

use num::One;
use num_complex::Complex;
use rand::{Rand, Rng};

use alga::general::Real;
use core::ColumnVector;
use core::dimension::{U1, U2};
use core::storage::Storage;
use geometry::{UnitComplex, RotationBase};


impl<N: Real> UnitComplex<N> {
    /// The unit complex number multiplicative identity.
    #[inline]
    pub fn identity() -> Self {
        Self::new_unchecked(Complex::new(N::one(), N::zero()))
    }

    /// Builds the unit complex number corresponding to the rotation with the angle.
    #[inline]
    pub fn new(angle: N) -> Self {
        let (s, c) = angle.sin_cos();
        UnitComplex::new_unchecked(Complex::new(c, s))
    }

    /// Builds the unit complex number corresponding to the rotation with the angle.
    ///
    /// Same as `Self::new(angle)`.
    #[inline]
    pub fn from_angle(angle: N) -> Self {
        Self::new(angle)
    }

    /// Builds a unit complex rotation from an angle in radian wrapped in a 1-dimensional vector.
    ///
    /// Equivalent to `Self::new(axisangle[0])`.
    #[inline]
    pub fn from_scaled_axis<SB: Storage<N, U1, U1>>(axisangle: ColumnVector<N, U1, SB>) -> Self {
        Self::from_angle(axisangle[0])
    }

    /// Creates a new unit complex number from a complex number.
    ///
    /// The input complex number will be normalized.
    #[inline]
    pub fn from_complex(q: Complex<N>) -> Self {
        Self::new_unchecked(q / (q.im * q.im + q.re * q.re).sqrt())
    }

    /// Builds the unit complex number from the corresponding 2D rotation matrix.
    #[inline]
    pub fn from_rotation_matrix<S: Storage<N, U2, U2>>(rotmat: &RotationBase<N, U2, S>) -> Self {
        Self::new_unchecked(Complex::new(rotmat[(0, 0)], rotmat[(1, 0)]))
    }

    /// The unit complex needed to make `a` and `b` be collinear and point toward the same
    /// direction.
    #[inline]
    pub fn rotation_between<SB, SC>(a: &ColumnVector<N, U2, SB>, b: &ColumnVector<N, U2, SC>) -> Self
        where SB: Storage<N, U2, U1>,
              SC: Storage<N, U2, U1> {
        Self::scaled_rotation_between(a, b, N::one())
    }

    /// The smallest rotation needed to make `a` and `b` collinear and point toward the same
    /// direction, raised to the power `s`.
    #[inline]
    pub fn scaled_rotation_between<SB, SC>(a: &ColumnVector<N, U2, SB>, b: &ColumnVector<N, U2, SC>, s: N) -> Self
        where SB: Storage<N, U2, U1>,
              SC: Storage<N, U2, U1> {
        if let (Some(na), Some(nb)) = (a.try_normalize(N::zero()), b.try_normalize(N::zero())) {
            let sang = na.perp(&nb);
            let cang = na.dot(&nb);

            Self::from_angle(sang.atan2(cang) * s)
        }
        else {
            Self::identity()
        }
    }
}

impl<N: Real> One for UnitComplex<N> {
    #[inline]
    fn one() -> Self {
        Self::identity()
    }
}

impl<N: Real + Rand> Rand for UnitComplex<N> {
    #[inline]
    fn rand<R: Rng>(rng: &mut R) -> Self {
        UnitComplex::from_angle(N::rand(rng))
    }
}

#[cfg(feature="arbitrary")]
impl<N: Real + Arbitrary> Arbitrary for UnitComplex<N> {
    #[inline]
    fn arbitrary<G: Gen>(g: &mut G) -> Self {
        UnitComplex::from_angle(N::arbitrary(g))

    }
}
