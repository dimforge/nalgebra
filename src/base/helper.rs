#[cfg(feature = "arbitrary")]
use quickcheck::{Arbitrary, Gen};

#[cfg(feature = "rand-no-std")]
use rand::{
    distributions::{Distribution, Standard},
    Rng,
};

/// Simple helper function for rejection sampling
#[cfg(feature = "arbitrary")]
#[doc(hidden)]
#[inline]
pub fn reject<F: FnMut(&T) -> bool, T: Arbitrary>(g: &mut Gen, f: F) -> T {
    use std::iter;
    iter::repeat(())
        .map(|_| Arbitrary::arbitrary(g))
        .find(f)
        .unwrap()
}

#[doc(hidden)]
#[inline]
#[cfg(feature = "rand-no-std")]
pub fn reject_rand<G: Rng + ?Sized, F: FnMut(&T) -> bool, T>(g: &mut G, f: F) -> T
where
    Standard: Distribution<T>,
{
    use std::iter;
    iter::repeat(()).map(|_| g.gen()).find(f).unwrap()
}
