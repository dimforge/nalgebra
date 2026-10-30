#[cfg(feature = "arbitrary")]
use quickcheck::{Arbitrary, Gen};

#[cfg(feature = "rand-no-std")]
use rand::{
    Rng, RngExt,
    distr::{Distribution, StandardUniform},
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
    StandardUniform: Distribution<T>,
{
    use std::iter;
    iter::repeat(()).map(|_| g.random()).find(f).unwrap()
}

/// Computes the number of elements of a matrix with shape `(nrows, ncols)`.
///
/// Panics if `nrows * ncols` overflows `usize`. All dimension validations
/// must go through this so that wrapped products cannot make an undersized
/// buffer look valid (issue #1620).
#[inline]
pub(crate) fn matrix_len(nrows: usize, ncols: usize) -> usize {
    nrows.checked_mul(ncols).expect(
        "Matrix dimensions are too large: nrows * ncols overflows usize.",
    )
}
