use num::{One, Zero};

use alga::general::{ClosedAdd, ClosedMul};

use base::{DefaultAllocator, MatrixN, Scalar};
use base::dimension::DimName;
use base::allocator::Allocator;

use geometry::Rotation;

impl<N, D: DimName> Rotation<N, D>
where
    N: Scalar + Zero + One,
    DefaultAllocator: Allocator<N, D, D>,
{
    /// Creates a new square identity rotation of the given `dimension`.
    #[inline]
    pub fn identity() -> Rotation<N, D> {
        Self::from_matrix_unchecked(MatrixN::<N, D>::identity())
    }
}

impl<N, D: DimName> One for Rotation<N, D>
where
    N: Scalar + Zero + One + ClosedAdd + ClosedMul,
    DefaultAllocator: Allocator<N, D, D>,
{
    #[inline]
    fn one() -> Self {
        Self::identity()
    }
}
