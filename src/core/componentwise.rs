// Non-convensional componentwise operators.

use num::Signed;

use alga::general::{ClosedMul, ClosedDiv};

use core::{Scalar, Matrix, OwnedMatrix, MatrixSum};
use core::dimension::Dim;
use core::storage::{Storage, StorageMut};
use core::allocator::SameShapeAllocator;
use core::constraint::{ShapeConstraint, SameNumberOfRows, SameNumberOfColumns};


/// The type of the result of a matrix componentwise operation.
pub type MatrixComponentOp<N, R1, C1, R2, C2, SA> = MatrixSum<N, R1, C1, R2, C2, SA>;

impl<N: Scalar, R: Dim, C: Dim, S: Storage<N, R, C>> Matrix<N, R, C, S> {
    /// Computes the componentwise absolute value.
    #[inline]
    pub fn abs(&self) -> OwnedMatrix<N, R, C, S::Alloc>
        where N: Signed {
        let mut res = self.clone_owned();

        for e in res.iter_mut() {
            *e = e.abs();
        }

        res
    }

    // FIXME: add other operators like component_ln, component_pow, etc. ?
}

macro_rules! component_binop_impl(
    ($($binop: ident, $binop_mut: ident, $Trait: ident . $binop_assign: ident);* $(;)*) => {$(
        impl<N: Scalar, R: Dim, C: Dim, S: Storage<N, R, C>> Matrix<N, R, C, S> {
            /// Componentwise matrix multiplication.
            #[inline]
            pub fn $binop<R2, C2, SB>(&self, rhs: &Matrix<N, R2, C2, SB>) -> MatrixComponentOp<N, R, C, R2, C2, S>
                where N: $Trait,
                      R2: Dim, C2: Dim,
                      SB: Storage<N, R2, C2>,
                      S::Alloc: SameShapeAllocator<N, R, C, R2, C2, S>,
                      ShapeConstraint: SameNumberOfRows<R, R2> + SameNumberOfColumns<C, C2> {
        
                let mut res = self.clone_owned_sum();
        
                for (res, rhs) in res.iter_mut().zip(rhs.iter()) {
                    res.$binop_assign(*rhs);
                }
        
                res
            }
        }
        
        impl<N: Scalar, R: Dim, C: Dim, S: StorageMut<N, R, C>> Matrix<N, R, C, S> {
            /// Componentwise matrix multiplication.
            #[inline]
            pub fn $binop_mut<R2, C2, SB>(&mut self, rhs: &Matrix<N, R2, C2, SB>)
                where N: $Trait,
                      R2: Dim,
                      C2: Dim,
                      SB: Storage<N, R2, C2>,
                      ShapeConstraint: SameNumberOfRows<R, R2> + SameNumberOfColumns<C, C2> {
                for (me, rhs) in self.iter_mut().zip(rhs.iter()) {
                    me.$binop_assign(*rhs);
                }
            }
        }
    )*}
);

component_binop_impl!(
    component_mul, component_mul_mut, ClosedMul.mul_assign;
    component_div, component_div_mut, ClosedDiv.div_assign;
    // FIXME: add other operators like bitshift, etc. ?
);
