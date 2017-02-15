use alga::general::{SubsetOf, SupersetOf};

use core::{Scalar, Matrix};
use core::dimension::Dim;
use core::constraint::{ShapeConstraint, SameNumberOfRows, SameNumberOfColumns};
use core::storage::OwnedStorage;
use core::allocator::{OwnedAllocator, SameShapeAllocator};


// FIXME: too bad this won't work allo slice conversions.
impl<N1, N2, R1, C1, R2, C2, SA, SB> SubsetOf<Matrix<N2, R2, C2, SB>> for Matrix<N1, R1, C1, SA>
    where R1: Dim, C1: Dim, R2: Dim, C2: Dim,
          N1: Scalar,
          N2: Scalar + SupersetOf<N1>,
          SA: OwnedStorage<N1, R1, C1>,
          SB: OwnedStorage<N2, R2, C2>,
          SB::Alloc: OwnedAllocator<N2, R2, C2, SB>,
          SA::Alloc: OwnedAllocator<N1, R1, C1, SA> +
                     SameShapeAllocator<N1, R1, C1, R2, C2, SA>,
          ShapeConstraint: SameNumberOfRows<R1, R2> + SameNumberOfColumns<C1, C2> {
    #[inline]
    fn to_superset(&self) -> Matrix<N2, R2, C2, SB> {
        let (nrows, ncols) = self.shape();
        let nrows2 = R2::from_usize(nrows);
        let ncols2 = C2::from_usize(ncols);

        let mut res = unsafe { Matrix::<N2, R2, C2, SB>::new_uninitialized_generic(nrows2, ncols2) };
        for i in 0 .. nrows {
            for j in 0 .. ncols {
                unsafe {
                    *res.get_unchecked_mut(i, j) = N2::from_subset(self.get_unchecked(i, j))
                }
            }
        }

        res
    }

    #[inline]
    fn is_in_subset(m: &Matrix<N2, R2, C2, SB>) -> bool {
        m.iter().all(|e| e.is_in_subset())
    }

    #[inline]
    unsafe fn from_superset_unchecked(m: &Matrix<N2, R2, C2, SB>) -> Self {
        let (nrows2, ncols2) = m.shape();
        let nrows = R1::from_usize(nrows2);
        let ncols = C1::from_usize(ncols2);

        let mut res = Self::new_uninitialized_generic(nrows, ncols);
        for i in 0 .. nrows2 {
            for j in 0 .. ncols2 {
                *res.get_unchecked_mut(i, j) = m.get_unchecked(i, j).to_subset_unchecked()
            }
        }

        res
    }
}
