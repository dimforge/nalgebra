use alga::general::{Field, JoinSemilattice, Lattice, MeetSemilattice, RealField};
use alga::linear::{AffineSpace, EuclideanSpace};

use crate::base::{CVectorN, Scalar};

use crate::geometry::Point;

impl<N: Scalar + Field, const D: usize> AffineSpace for Point<N, D>
where
    N: Scalar + Field,
{
    type Translation = CVectorN<N, D>;
}

impl<N: RealField + simba::scalar::RealField, const D: usize> EuclideanSpace for Point<N, D> {
    type Coordinates = CVectorN<N, D>;
    type RealField = N;

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
        Self::from(coords)
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
impl<N, const D: usize> MeetSemilattice for Point<N, D>
where
    N: Scalar + MeetSemilattice,
{
    #[inline]
    fn meet(&self, other: &Self) -> Self {
        Self::from(self.coords.meet(&other.coords))
    }
}

impl<N, const D: usize> JoinSemilattice for Point<N, D>
where
    N: Scalar + JoinSemilattice,
{
    #[inline]
    fn join(&self, other: &Self) -> Self {
        Self::from(self.coords.join(&other.coords))
    }
}

impl<N, const D: usize> Lattice for Point<N, D>
where
    N: Scalar + Lattice,
{
    #[inline]
    fn meet_join(&self, other: &Self) -> (Self, Self) {
        let (meet, join) = self.coords.meet_join(&other.coords);

        (Self::from(meet), Self::from(join))
    }
}
