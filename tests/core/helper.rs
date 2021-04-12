// This module implement several methods to fill some
// missing features of num-complex when it comes to randomness.

use na::RealField;
use num_complex::Complex;
use quickcheck::{Arbitrary, Gen};
use rand::distributions::{Distribution, Standard};
use rand::Rng;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct RandComplex<T>(pub Complex<T>);

impl<T: Arbitrary + RealField> Arbitrary for RandComplex<T> {
    #[inline]
    fn arbitrary(rng: &mut Gen) -> Self {
        let im = Arbitrary::arbitrary(rng);
        let re = Arbitrary::arbitrary(rng);
        RandComplex(Complex::new(re, im))
    }
}

impl<T: RealField> Distribution<RandComplex<T>> for Standard
where
    Standard: Distribution<T>,
{
    #[inline]
    fn sample<'a, G: Rng + ?Sized>(&self, rng: &'a mut G) -> RandComplex<T> {
        let re = rng.gen();
        let im = rng.gen();
        RandComplex(Complex::new(re, im))
    }
}

// This is a wrapper similar to RandComplex, but for non-complex.
// This exists only to make generic tests easier to write.
//
// Generates variates in the range [0, 1).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct RandScalar<T>(pub T);

impl<T: Arbitrary> Arbitrary for RandScalar<T> {
    #[inline]
    fn arbitrary(rng: &mut Gen) -> Self {
        RandScalar(Arbitrary::arbitrary(rng))
    }
}

impl<T: RealField> Distribution<RandScalar<T>> for Standard
where
    Standard: Distribution<T>,
{
    #[inline]
    fn sample<'a, G: Rng + ?Sized>(&self, rng: &'a mut G) -> RandScalar<T> {
        RandScalar(self.sample(rng))
    }
}
