#[cfg(feature = "arbitrary")]
use quickcheck::{Arbitrary, Gen};

use num::{Zero, One};
use rand::{Rng, Rand};

use alga::general::ClosedAdd;

use core::{ColumnVector, Scalar};
use core::dimension::{DimName, U1, U2, U3, U4, U5, U6};
use core::storage::OwnedStorage;
use core::allocator::OwnedAllocator;

use geometry::TranslationBase;

impl<N, D: DimName, S> TranslationBase<N, D, S>
    where N: Scalar + Zero,
          S: OwnedStorage<N, D, U1>,
          S::Alloc: OwnedAllocator<N, D, U1, S> {
    /// Creates a new square identity rotation of the given `dimension`.
    #[inline]
    pub fn identity() -> TranslationBase<N, D, S> {
        Self::from_vector(ColumnVector::<N, D, S>::from_element(N::zero()))
    }
}

impl<N, D: DimName, S> One for TranslationBase<N, D, S>
    where N: Scalar + Zero + ClosedAdd,
          S: OwnedStorage<N, D, U1>,
          S::Alloc: OwnedAllocator<N, D, U1, S> {
    #[inline]
    fn one() -> Self {
        Self::identity()
    }
}

impl<N, D: DimName, S> Rand for TranslationBase<N, D, S>
    where N: Scalar + Rand,
          S: OwnedStorage<N, D, U1>,
          S::Alloc: OwnedAllocator<N, D, U1, S> {
    #[inline]
    fn rand<G: Rng>(rng: &mut G) -> Self {
        Self::from_vector(rng.gen())
    }
}


#[cfg(feature = "arbitrary")]
impl<N, D: DimName, S> Arbitrary for TranslationBase<N, D, S>
    where N: Scalar + Arbitrary + Send,
          S: OwnedStorage<N, D, U1> + Send,
          S::Alloc: OwnedAllocator<N, D, U1, S> {
    #[inline]
    fn arbitrary<G: Gen>(rng: &mut G) -> Self {
        Self::from_vector(Arbitrary::arbitrary(rng))
    }
}

/*
 *
 * Small translation construction from components.
 *
 */
macro_rules! componentwise_constructors_impl(
    ($($D: ty, $($args: ident:$irow: expr),*);* $(;)*) => {$(
        impl<N, S> TranslationBase<N, $D, S>
            where N: Scalar,
                  S: OwnedStorage<N, $D, U1>,
                  S::Alloc: OwnedAllocator<N, $D, U1, S> {
            /// Initializes this matrix from its components.
            #[inline]
            pub fn new($($args: N),*) -> Self {
                Self::from_vector(ColumnVector::<N, $D, S>::new($($args),*))
            }
        }
    )*}
);

componentwise_constructors_impl!(
    U1, x:0;
    U2, x:0, y:1;
    U3, x:0, y:1, z:2;
    U4, x:0, y:1, z:2, w:3;
    U5, x:0, y:1, z:2, w:3, a:4;
    U6, x:0, y:1, z:2, w:3, a:4, b:5;
);
