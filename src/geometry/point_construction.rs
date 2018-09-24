#[cfg(feature = "arbitrary")]
use quickcheck::{Arbitrary, Gen};

use num::{Bounded, One, Zero};
use rand::distributions::{Distribution, Standard};
use rand::Rng;

use alga::general::ClosedDiv;
use base::allocator::Allocator;
use base::dimension::{DimName, DimNameAdd, DimNameSum, U1, U2, U3, U4, U5, U6};
use base::{DefaultAllocator, Scalar, VectorN};

use geometry::Point;

impl<N: Scalar, D: DimName> Point<N, D>
where
    DefaultAllocator: Allocator<N, D>,
{
    /// Creates a new point with uninitialized coordinates.
    #[inline]
    pub unsafe fn new_uninitialized() -> Self {
        Self::from_coordinates(VectorN::new_uninitialized())
    }

    /// Creates a new point with all coordinates equal to zero.
    #[inline]
    pub fn origin() -> Self
    where
        N: Zero,
    {
        Self::from_coordinates(VectorN::from_element(N::zero()))
    }

    /// Creates a new point from its homogeneous vector representation.
    ///
    /// In practice, this builds a D-dimensional points with the same first D component as `v`
    /// divided by the last component of `v`. Returns `None` if this divisor is zero.
    #[inline]
    pub fn from_homogeneous(v: VectorN<N, DimNameSum<D, U1>>) -> Option<Self>
    where
        N: Scalar + Zero + One + ClosedDiv,
        D: DimNameAdd<U1>,
        DefaultAllocator: Allocator<N, DimNameSum<D, U1>>,
    {
        if !v[D::dim()].is_zero() {
            let coords = v.fixed_slice::<D, U1>(0, 0) / v[D::dim()];
            Some(Self::from_coordinates(coords))
        } else {
            None
        }
    }
}

/*
 *
 * Traits that build points.
 *
 */
impl<N: Scalar + Bounded, D: DimName> Bounded for Point<N, D>
where
    DefaultAllocator: Allocator<N, D>,
{
    #[inline]
    fn max_value() -> Self {
        Self::from_coordinates(VectorN::max_value())
    }

    #[inline]
    fn min_value() -> Self {
        Self::from_coordinates(VectorN::min_value())
    }
}

impl<N: Scalar, D: DimName> Distribution<Point<N, D>> for Standard
where
    DefaultAllocator: Allocator<N, D>,
    Standard: Distribution<N>,
{
    #[inline]
    fn sample<'a, G: Rng + ?Sized>(&self, rng: &mut G) -> Point<N, D> {
        Point::from_coordinates(rng.gen::<VectorN<N, D>>())
    }
}

#[cfg(feature = "arbitrary")]
impl<N: Scalar + Arbitrary + Send, D: DimName> Arbitrary for Point<N, D>
where
    DefaultAllocator: Allocator<N, D>,
    <DefaultAllocator as Allocator<N, D>>::Buffer: Send,
{
    #[inline]
    fn arbitrary<G: Gen>(g: &mut G) -> Self {
        Point::from_coordinates(VectorN::arbitrary(g))
    }
}

/*
 *
 * Small points construction from components.
 *
 */
macro_rules! componentwise_constructors_impl(
    ($($D: ty, $($args: ident:$irow: expr),*);* $(;)*) => {$(
        impl<N: Scalar> Point<N, $D>
            where DefaultAllocator: Allocator<N, $D> {
            /// Initializes this matrix from its components.
            #[inline]
            pub fn new($($args: N),*) -> Point<N, $D> {
                unsafe {
                    let mut res = Self::new_uninitialized();
                    $( *res.get_unchecked_mut($irow) = $args; )*

                    res
                }
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

macro_rules! from_array_impl(
    ($($D: ty, $len: expr);*) => {$(
      impl <N: Scalar> From<[N; $len]> for Point<N, $D> {
          fn from (coords: [N; $len]) -> Self {
              Point::from_coordinates(coords.into())
          }
      }
    )*}
);

from_array_impl!(U1, 1; U2, 2; U3, 3; U4, 4; U5, 5; U6, 6);
