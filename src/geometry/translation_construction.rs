#[cfg(feature = "arbitrary")]
use quickcheck::{Arbitrary, Gen};
#[cfg(feature = "arbitrary")]
use core::storage::Owned;

use num::{Zero, One};
use rand::{Rng, Rand};

use alga::general::ClosedAdd;

use core::{DefaultAllocator, Scalar, VectorN};
use core::dimension::{DimName, U1, U2, U3, U4, U5, U6};
use core::allocator::Allocator;

use geometry::Translation;

impl<N: Scalar + Zero, D: DimName> Translation<N, D>
    where DefaultAllocator: Allocator<N, D> {

    /// Creates a new square identity rotation of the given `dimension`.
    #[inline]
    pub fn identity() -> Translation<N, D> {
        Self::from_vector(VectorN::<N, D>::from_element(N::zero()))
    }
}

impl<N: Scalar + Zero + ClosedAdd, D: DimName> One for Translation<N, D>
    where DefaultAllocator: Allocator<N, D> {
    #[inline]
    fn one() -> Self {
        Self::identity()
    }
}

impl<N: Scalar + Rand, D: DimName> Rand for Translation<N, D>
    where DefaultAllocator: Allocator<N, D> {
    #[inline]
    fn rand<G: Rng>(rng: &mut G) -> Self {
        Self::from_vector(rng.gen())
    }
}


#[cfg(feature = "arbitrary")]
impl<N: Scalar + Arbitrary, D: DimName> Arbitrary for Translation<N, D>
    where DefaultAllocator: Allocator<N, D>,
          Owned<N, D>: Send {
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
        impl<N: Scalar> Translation<N, $D>
            where DefaultAllocator: Allocator<N, $D> {
            /// Initializes this matrix from its components.
            #[inline]
            pub fn new($($args: N),*) -> Self {
                Self::from_vector(VectorN::<N, $D>::new($($args),*))
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
