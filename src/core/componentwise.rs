// Non-convensional componentwise operators.

use num::Signed;

use alga::general::{ClosedMul, ClosedDiv};

use core::{DefaultAllocator, Scalar, Matrix, MatrixMN, MatrixSum};
use core::dimension::Dim;
use core::storage::{Storage, StorageMut};
use core::allocator::{Allocator, SameShapeAllocator};
use core::constraint::{ShapeConstraint, SameNumberOfRows, SameNumberOfColumns};


/// The type of the result of a matrix componentwise operation.
pub type MatrixComponentOp<N, R1, C1, R2, C2> = MatrixSum<N, R1, C1, R2, C2>;

impl<N: Scalar, R: Dim, C: Dim, S: Storage<N, R, C>> Matrix<N, R, C, S> {
    /// Computes the componentwise absolute value.
    #[inline]
    pub fn abs(&self) -> MatrixMN<N, R, C>
        where N: Signed,
              DefaultAllocator: Allocator<N, R, C> {
        let mut res = self.clone_owned();

        for e in res.iter_mut() {
            *e = e.abs();
        }

        res
    }

    // FIXME: add other operators like component_ln, component_pow, etc. ?
}

macro_rules! component_binop_impl(
    ($($binop: ident, $binop_mut: ident, $Trait: ident . $binop_assign: ident, $desc:expr, $desc_mut:expr);* $(;)*) => {$(
        impl<N: Scalar, R1: Dim, C1: Dim, SA: Storage<N, R1, C1>> Matrix<N, R1, C1, SA> {
            #[doc = $desc]
            #[inline]
            pub fn $binop<R2, C2, SB>(&self, rhs: &Matrix<N, R2, C2, SB>) -> MatrixComponentOp<N, R1, C1, R2, C2>
                where N: $Trait,
                      R2: Dim, C2: Dim,
                      SB: Storage<N, R2, C2>,
                      DefaultAllocator: SameShapeAllocator<N, R1, C1, R2, C2>,
                      ShapeConstraint:  SameNumberOfRows<R1, R2> + SameNumberOfColumns<C1, C2> {
        
                assert_eq!(self.shape(), rhs.shape(), "Componentwise mul/div: mismatched matrix dimensions.");
                let mut res = self.clone_owned_sum();

                for j in 0 .. res.ncols() {
                    for i in 0 .. res.nrows() {
                        unsafe {
                            res.get_unchecked_mut(i, j).$binop_assign(*rhs.get_unchecked(i, j));
                        }
                    }
                }

                res
            }
        }

        impl<N: Scalar, R1: Dim, C1: Dim, SA: StorageMut<N, R1, C1>> Matrix<N, R1, C1, SA> {
            #[doc = $desc_mut]
            #[inline]
            pub fn $binop_mut<R2, C2, SB>(&mut self, rhs: &Matrix<N, R2, C2, SB>)
                where N: $Trait,
                      R2: Dim,
                      C2: Dim,
                      SB: Storage<N, R2, C2>,
                      ShapeConstraint: SameNumberOfRows<R1, R2> + SameNumberOfColumns<C1, C2> {

                assert_eq!(self.shape(), rhs.shape(), "Componentwise mul/div: mismatched matrix dimensions.");

                for j in 0 .. self.ncols() {
                    for i in 0 .. self.nrows() {
                        unsafe {
                            self.get_unchecked_mut(i, j).$binop_assign(*rhs.get_unchecked(i, j));
                        }
                    }
                }
            }
        }
    )*}
);

component_binop_impl!(
    component_mul, component_mul_mut, ClosedMul.mul_assign,
    "Componentwise matrix multiplication.", "Mutable, componentwise matrix multiplication.";
    component_div, component_div_mut, ClosedDiv.div_assign,
    "Componentwise matrix division.", "Mutable, componentwise matrix division.";
    // FIXME: add other operators like bitshift, etc. ?
);
