use alga::general::{Field, Real, MeetSemilattice, JoinSemilattice, Lattice};
use alga::linear::{AffineSpace, EuclideanSpace};

use core::{ColumnVector, Scalar};
use core::dimension::{DimName, U1};
use core::storage::OwnedStorage;
use core::allocator::OwnedAllocator;

use geometry::PointBase;


impl<N, D: DimName, S> AffineSpace for PointBase<N, D, S>
    where N: Scalar + Field,
          S: OwnedStorage<N, D, U1>,
          S::Alloc: OwnedAllocator<N, D, U1, S> {
    type Translation = ColumnVector<N, D, S>;
}

impl<N, D: DimName, S> EuclideanSpace for PointBase<N, D, S>
    where N: Real,
          S: OwnedStorage<N, D, U1>,
          S::Alloc: OwnedAllocator<N, D, U1, S> {
    type Coordinates = ColumnVector<N, D, S>;
    type Real        = N;

    #[inline]
    fn origin() -> Self {
        Self::origin()
    }

    #[inline]
    fn coordinates(&self) -> Self::Coordinates {
        self.coords.clone()
    }

    #[inline]
    fn from_coordinates(coords: Self::Coordinates) -> Self {
        Self::from_coordinates(coords)
    }

    #[inline]
    fn scale_by(&self, n: N) -> Self {
        self * n
    }
}

/*
 *
 * Ordering
 *
 */
impl<N, D: DimName, S> MeetSemilattice for PointBase<N, D, S>
    where N: Scalar + MeetSemilattice,
          S: OwnedStorage<N, D, U1>,
          S::Alloc: OwnedAllocator<N, D, U1, S> {
    #[inline]
    fn meet(&self, other: &Self) -> Self {
        PointBase::from_coordinates(self.coords.meet(&other.coords))
    }
}

impl<N, D: DimName, S> JoinSemilattice for PointBase<N, D, S>
    where N: Scalar + JoinSemilattice,
          S: OwnedStorage<N, D, U1>,
          S::Alloc: OwnedAllocator<N, D, U1, S> {
    #[inline]
    fn join(&self, other: &Self) -> Self {
        PointBase::from_coordinates(self.coords.join(&other.coords))
    }
}


impl<N, D: DimName, S> Lattice for PointBase<N, D, S>
    where N: Scalar + Lattice,
          S: OwnedStorage<N, D, U1>,
          S::Alloc: OwnedAllocator<N, D, U1, S> {
    #[inline]
    fn meet_join(&self, other: &Self) -> (Self, Self) {
        let (meet, join) = self.coords.meet_join(&other.coords);

        (PointBase::from_coordinates(meet), PointBase::from_coordinates(join))
    }
}
