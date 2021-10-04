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

/// Check that the first and last offsets conform to the specification of a CSR matrix
#[inline]
#[must_use]
pub fn first_and_last_offsets_are_ok(
    major_offsets: &Vec<usize>,
    minor_indices: &Vec<usize>,
) -> bool {
    let first_offset_ok = *major_offsets.first().unwrap() == 0;
    let last_offset_ok = *major_offsets.last().unwrap() == minor_indices.len();
    return first_offset_ok && last_offset_ok;
}
