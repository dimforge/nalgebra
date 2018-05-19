use num::One;

use alga::general::Real;

use base::{DefaultAllocator, MatrixN};
use base::dimension::{DimNameAdd, DimNameSum, U1};
use base::allocator::Allocator;

use geometry::{TCategory, Transform};

impl<N: Real, D: DimNameAdd<U1>, C: TCategory> Transform<N, D, C>
where
    DefaultAllocator: Allocator<N, DimNameSum<D, U1>, DimNameSum<D, U1>>,
{
    /// Creates a new identity transform.
    #[inline]
    pub fn identity() -> Self {
        Self::from_matrix_unchecked(MatrixN::<_, DimNameSum<D, U1>>::identity())
    }
}

impl<N: Real, D: DimNameAdd<U1>, C: TCategory> One for Transform<N, D, C>
where
    DefaultAllocator: Allocator<N, DimNameSum<D, U1>, DimNameSum<D, U1>>,
{
    /// Creates a new identity transform.
    #[inline]
    fn one() -> Self {
        Self::identity()
    }
}
