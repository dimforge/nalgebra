#[cfg(feature = "arbitrary")]
use quickcheck::{Arbitrary, Gen};

use rand::{Rand, Rng};
use num::{Zero, One, Bounded};

use alga::general::ClosedDiv;
use core::{Scalar, ColumnVector};
use core::storage::{Storage, OwnedStorage};
use core::allocator::{Allocator, OwnedAllocator};
use core::dimension::{DimName, DimNameAdd, DimNameSum, U1, U2, U3, U4, U5, U6};

use geometry::PointBase;

impl<N, D: DimName, S> PointBase<N, D, S>
    where N: Scalar,
          S: OwnedStorage<N, D, U1>,
          S::Alloc: OwnedAllocator<N, D, U1, S> {
    /// Creates a new point with uninitialized coordinates.
    #[inline]
    pub unsafe fn new_uninitialized() -> Self {
        Self::from_coordinates(ColumnVector::<_, D, _>::new_uninitialized())
    }

    /// Creates a new point with all coordinates equal to zero.
    #[inline]
    pub fn origin() -> Self
        where N: Zero {
        Self::from_coordinates(ColumnVector::<_, D, _>::from_element(N::zero()))
    }

    /// Creates a new point from its homogeneous vector representation.
    ///
    /// In practice, this builds a D-dimensional points with the same first D component as `v`
    /// divided by the last component of `v`. Returns `None` if this divisor is zero.
    #[inline]
    pub fn from_homogeneous<SB>(v: ColumnVector<N, DimNameSum<D, U1>, SB>) -> Option<Self>
        where N:  Scalar + Zero + One + ClosedDiv,
              D:  DimNameAdd<U1>,
              SB: Storage<N, DimNameSum<D, U1>, U1, Alloc = S::Alloc>,
              S::Alloc: Allocator<N, DimNameSum<D, U1>, U1> {

        if !v[D::dim()].is_zero() {
            let coords =  v.fixed_slice::<D, U1>(0, 0) / v[D::dim()];
            Some(Self::from_coordinates(coords))
        }
        else {
            None
        }
    }
}


/*
 *
 * Traits that buid points.
 *
 */
impl<N, D: DimName, S> Bounded for PointBase<N, D, S>
    where N: Scalar + Bounded,
          S: OwnedStorage<N, D, U1>,
          S::Alloc: OwnedAllocator<N, D, U1, S> {
    #[inline]
    fn max_value() -> Self {
        Self::from_coordinates(ColumnVector::max_value())
    }

    #[inline]
    fn min_value() -> Self {
        Self::from_coordinates(ColumnVector::min_value())
    }
}

impl<N, D: DimName, S> Rand for PointBase<N, D, S>
    where N: Scalar + Rand,
          S: OwnedStorage<N, D, U1>,
          S::Alloc: OwnedAllocator<N, D, U1, S> {
    #[inline]
    fn rand<G: Rng>(rng: &mut G) -> Self {
        PointBase::from_coordinates(rng.gen())
    }
}

#[cfg(feature="arbitrary")]
impl<N, D: DimName, S> Arbitrary for PointBase<N, D, S>
    where N: Scalar + Arbitrary + Send,
          S: OwnedStorage<N, D, U1> + Send,
          S::Alloc: OwnedAllocator<N, D, U1, S> {
    #[inline]
    fn arbitrary<G: Gen>(g: &mut G) -> Self {
        PointBase::from_coordinates(ColumnVector::arbitrary(g))
    }
}

/*
 *
 * Small points construction from components.
 *
 */
macro_rules! componentwise_constructors_impl(
    ($($D: ty, $($args: ident:$irow: expr),*);* $(;)*) => {$(
        impl<N, S> PointBase<N, $D, S>
            where N: Scalar,
                  S: OwnedStorage<N, $D, U1>,
                  S::Alloc: OwnedAllocator<N, $D, U1, S> {
            /// Initializes this matrix from its components.
            #[inline]
            pub fn new($($args: N),*) -> PointBase<N, $D, S> {
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
