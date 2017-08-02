use num::One;
use alga::general::ClosedNeg;

use core::{Scalar, Matrix, VectorN, DefaultAllocator};
use dimension::{Dim, U1};
use storage::StorageMut;
use allocator::Allocator;


/// A sequence of permutations.
pub struct PermutationSequence<D: Dim>
    where DefaultAllocator: Allocator<(usize, usize), D>{
    len:  usize,
    ipiv: VectorN<(usize, usize), D>
}

impl<D: Dim> PermutationSequence<D>
    where DefaultAllocator: Allocator<(usize, usize), D> {

    // XXX: Add non-generic constructors.
    /// Creates a new sequence of D identity permutations.
    #[inline]
    pub fn identity_generic(dim: D) -> Self {
        unsafe {
            PermutationSequence {
                len:  0,
                ipiv: VectorN::new_uninitialized_generic(dim, U1)
            }
        }
    }

    /// Adds the interchange of the row `i` with the row `i2` to this sequence of permutations.
    #[inline]
    pub fn append_permutation(&mut self, i: usize, i2: usize) {
        if i != i2 {
            assert!(self.len < self.ipiv.len(), "Maximum number of permutations exceeded.");
            self.ipiv[self.len] = (i, i2);
            self.len += 1;
        }
    }

    /// Permutes the rows of `rhs`, applying a sequence of permutations from the 0-th row to the
    /// last.
    #[inline]
    pub fn permute_rows<N: Scalar, R2: Dim, C2: Dim, S2>(&self, rhs: &mut Matrix<N, R2, C2, S2>)
        where S2: StorageMut<N, R2, C2> {

        for i in self.ipiv.rows_range(.. self.len).iter() {
            rhs.swap_rows(i.0, i.1)
        }
    }

    /// Permutes the rows of `rhs` using the inverse permutation matrix of this LUP decomposition.
    #[inline]
    pub fn inv_permute_rows<N: Scalar, R2: Dim, C2: Dim, S2>(&self, rhs: &mut Matrix<N, R2, C2, S2>)
        where S2: StorageMut<N, R2, C2> {

        for i in 0 .. self.len {
            let (i1, i2) = self.ipiv[self.len - i - 1];
            rhs.swap_rows(i1, i2)
        }
    }

    /// Permutes the columns of `rhs`, applying a sequence of permutations from the 0-th row to the
    /// last.
    #[inline]
    pub fn permute_columns<N: Scalar, R2: Dim, C2: Dim, S2>(&self, rhs: &mut Matrix<N, R2, C2, S2>)
        where S2: StorageMut<N, R2, C2> {

        for i in self.ipiv.rows_range(.. self.len).iter() {
            rhs.swap_columns(i.0, i.1)
        }
    }

    /// Permutes the columns of `rhs` using the inverse permutation matrix of this LUP decomposition.
    #[inline]
    pub fn inv_permute_columns<N: Scalar, R2: Dim, C2: Dim, S2>(&self, rhs: &mut Matrix<N, R2, C2, S2>)
        where S2: StorageMut<N, R2, C2> {

        for i in 0 .. self.len {
            let (i1, i2) = self.ipiv[self.len - i - 1];
            rhs.swap_columns(i1, i2)
        }
    }

    /// The number of permutations applied by this matrix.
    pub fn len(&self) -> usize {
        self.len
    }

    /// The determinant of the matrix corresponding to this permutation.
    #[inline]
    pub fn determinant<N: One + ClosedNeg>(&self) -> N {
        if self.len % 2 == 0 {
            N::one()
        }
        else {
            -N::one()
        }
    }
}
