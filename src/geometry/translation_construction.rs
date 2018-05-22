#[cfg(feature = "arbitrary")]
use base::storage::Owned;
#[cfg(feature = "arbitrary")]
use quickcheck::{Arbitrary, Gen};

use num::{One, Zero};
use rand::distributions::{Distribution, Standard};
use rand::Rng;

use alga::general::ClosedAdd;

use base::allocator::Allocator;
use base::dimension::{DimName, U1, U2, U3, U4, U5, U6};
use base::{DefaultAllocator, Scalar, VectorN};

use geometry::Translation;

impl<N: Scalar + Zero, D: DimName> Translation<N, D>
where
    DefaultAllocator: Allocator<N, D>,
{
    /// Creates a new square identity rotation of the given `dimension`.
    #[inline]
    pub fn identity() -> Translation<N, D> {
        Self::from_vector(VectorN::<N, D>::from_element(N::zero()))
    }
}

impl<N: Scalar + Zero + ClosedAdd, D: DimName> One for Translation<N, D>
where
    DefaultAllocator: Allocator<N, D>,
{
    #[inline]
    fn one() -> Self {
        Self::identity()
    }
}

impl<N: Scalar, D: DimName> Distribution<Translation<N, D>> for Standard
where
    DefaultAllocator: Allocator<N, D>,
    Standard: Distribution<N>,
{
    #[inline]
    fn sample<'a, G: Rng + ?Sized>(&self, rng: &'a mut G) -> Translation<N, D> {
        Translation::from_vector(rng.gen::<VectorN<N, D>>())
    }
}

#[cfg(feature = "arbitrary")]
impl<N: Scalar + Arbitrary, D: DimName> Arbitrary for Translation<N, D>
where
    DefaultAllocator: Allocator<N, D>,
    Owned<N, D>: Send,
{
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
