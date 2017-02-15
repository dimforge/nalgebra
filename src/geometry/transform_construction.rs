use num::{Zero, One};

use alga::general::Field;

use core::{Scalar, OwnedSquareMatrix};
use core::dimension::{DimNameAdd, DimNameSum, U1};
use core::storage::OwnedStorage;
use core::allocator::OwnedAllocator;

use geometry::{TransformBase, TCategory};


impl<N, D, S, C: TCategory> TransformBase<N, D, S, C>
    where N: Scalar + Zero + One,
          D: DimNameAdd<U1>,
          S: OwnedStorage<N, DimNameSum<D, U1>, DimNameSum<D, U1>>,
          S::Alloc: OwnedAllocator<N, DimNameSum<D, U1>, DimNameSum<D, U1>, S> {
    /// Creates a new identity transform.
    #[inline]
    pub fn identity() -> Self {
        Self::from_matrix_unchecked(OwnedSquareMatrix::<N, _, S::Alloc>::identity())
    }
}

impl<N, D, S, C: TCategory> One for TransformBase<N, D, S, C>
    where N: Scalar + Field,
          D: DimNameAdd<U1>,
          S: OwnedStorage<N, DimNameSum<D, U1>, DimNameSum<D, U1>>,
          S::Alloc: OwnedAllocator<N, DimNameSum<D, U1>, DimNameSum<D, U1>, S> {
    /// Creates a new identity transform.
    #[inline]
    fn one() -> Self {
        Self::identity()
    }
}

