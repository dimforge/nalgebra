// This module implement several methods to fill some
// missing features of num-complex when it comes to randomness.

use quickcheck::{Arbitrary, Gen};
use rand::distributions::{Standard, Distribution};
use rand::Rng;
use num_complex::Complex;
use na::RealField;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct RandComplex<N>(pub Complex<N>);

impl<N: Arbitrary + RealField> Arbitrary for RandComplex<N> {
    #[inline]
    fn arbitrary<G: Gen>(rng: &mut G) -> Self {
        let im = Arbitrary::arbitrary(rng);
        let re = Arbitrary::arbitrary(rng);
        RandComplex(Complex::new(re, im))
    }
}

impl<N: RealField> Distribution<RandComplex<N>> for Standard
    where
        Standard: Distribution<N>,
{
    #[inline]
    fn sample<'a, G: Rng + ?Sized>(&self, rng: &'a mut G) -> RandComplex<N> {
        let re = rng.gen();
        let im = rng.gen();
        RandComplex(Complex::new(re, im))
    }
}

// This is a wrapper similar to RandComplex, but for non-complex.
// This exists only to make generic tests easier to write.
// Generates variates in the range [0, 1). Do we want this? E.g. we could use standard normal samples instead.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct RandScalar<N>(pub N);

impl<N: Arbitrary> Arbitrary for RandScalar<N> {
    #[inline]
    fn arbitrary<G: Gen>(rng: &mut G) -> Self {
        RandScalar(Arbitrary::arbitrary(rng))
    }
}

impl<N: RealField> Distribution<RandScalar<N>> for Standard
    where
        Standard: Distribution<N>,
{
    #[inline]
    fn sample<'a, G: Rng + ?Sized>(&self, rng: &'a mut G) -> RandScalar<N> {
        RandScalar(self.sample(rng))
    }
}