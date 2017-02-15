use num::{Zero, One};

use alga::general::{ClosedAdd, ClosedMul};

use core::{SquareMatrix, Scalar};
use core::dimension::DimName;
use core::storage::OwnedStorage;
use core::allocator::OwnedAllocator;

use geometry::RotationBase;

impl<N, D: DimName, S> RotationBase<N, D, S>
    where N: Scalar + Zero + One,
          S: OwnedStorage<N, D, D>,
          S::Alloc: OwnedAllocator<N, D, D, S> {
    /// Creates a new square identity rotation of the given `dimension`.
    #[inline]
    pub fn identity() -> RotationBase<N, D, S> {
        Self::from_matrix_unchecked(SquareMatrix::<N, D, S>::identity())
    }
}

impl<N, D: DimName, S> One for RotationBase<N, D, S>
    where N: Scalar + Zero + One + ClosedAdd + ClosedMul,
          S: OwnedStorage<N, D, D>,
          S::Alloc: OwnedAllocator<N, D, D, S> {
    #[inline]
    fn one() -> Self {
        Self::identity()
    }
}
