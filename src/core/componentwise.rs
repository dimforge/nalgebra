// Non-convensional componentwise operators.

use std::ops::{Add, Mul};
use num::{Zero, Signed};

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
    ($($binop: ident, $binop_mut: ident, $binop_assign: ident, $cbpy: ident, $Trait: ident . $op: ident . $op_assign: ident, $desc:expr, $desc_mut:expr);* $(;)*) => {$(
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
                            res.get_unchecked_mut(i, j).$op_assign(*rhs.get_unchecked(i, j));
                        }
                    }
                }

                res
            }
        }

        impl<N: Scalar, R1: Dim, C1: Dim, SA: StorageMut<N, R1, C1>> Matrix<N, R1, C1, SA> {
            // componentwise binop plus Y.
            #[inline]
            pub fn $cbpy<R2, C2, SB, R3, C3, SC>(&mut self, alpha: N, a: &Matrix<N, R2, C2, SB>, b: &Matrix<N, R3, C3, SC>, beta: N)
                where N: $Trait + Zero + Mul<N, Output = N> + Add<N, Output = N>,
                      R2: Dim, C2: Dim,
                      R3: Dim, C3: Dim,
                      SB: Storage<N, R2, C2>,
                      SC: Storage<N, R3, C3>,
                      ShapeConstraint: SameNumberOfRows<R1, R2> + SameNumberOfColumns<C1, C2> +
                                       SameNumberOfRows<R1, R3> + SameNumberOfColumns<C1, C3> {
                assert_eq!(self.shape(), a.shape(), "Componentwise mul/div: mismatched matrix dimensions.");
                assert_eq!(self.shape(), b.shape(), "Componentwise mul/div: mismatched matrix dimensions.");

                if beta.is_zero() {
                    for j in 0 .. self.ncols() {
                        for i in 0 .. self.nrows() {
                            unsafe {
                                let res = alpha * a.get_unchecked(i, j).$op(*b.get_unchecked(i, j));
                                *self.get_unchecked_mut(i, j) = res;
                            }
                        }
                    }
                }
                else {
                    for j in 0 .. self.ncols() {
                        for i in 0 .. self.nrows() {
                            unsafe {
                                let res = alpha * a.get_unchecked(i, j).$op(*b.get_unchecked(i, j));
                                *self.get_unchecked_mut(i, j) = beta * *self.get_unchecked(i, j) + res;
                            }
                        }
                    }
                }
            }

            #[doc = $desc_mut]
            #[inline]
            pub fn $binop_assign<R2, C2, SB>(&mut self, rhs: &Matrix<N, R2, C2, SB>)
                where N: $Trait,
                      R2: Dim,
                      C2: Dim,
                      SB: Storage<N, R2, C2>,
                      ShapeConstraint: SameNumberOfRows<R1, R2> + SameNumberOfColumns<C1, C2> {

                assert_eq!(self.shape(), rhs.shape(), "Componentwise mul/div: mismatched matrix dimensions.");

                for j in 0 .. self.ncols() {
                    for i in 0 .. self.nrows() {
                        unsafe {
                            self.get_unchecked_mut(i, j).$op_assign(*rhs.get_unchecked(i, j));
                        }
                    }
                }
            }

            #[doc = $desc_mut]
            #[inline]
            #[deprecated(note = "This is renamed using the `_assign` sufix instead of the `_mut` suffix.")]
            pub fn $binop_mut<R2, C2, SB>(&mut self, rhs: &Matrix<N, R2, C2, SB>)
                where N: $Trait,
                      R2: Dim,
                      C2: Dim,
                      SB: Storage<N, R2, C2>,
                      ShapeConstraint: SameNumberOfRows<R1, R2> + SameNumberOfColumns<C1, C2> {
                self.$binop_assign(rhs)
            }
        }
    )*}
);

component_binop_impl!(
    component_mul, component_mul_mut, component_mul_assign, cmpy, ClosedMul.mul.mul_assign,
    "Componentwise matrix multiplication.", "Mutable, componentwise matrix multiplication.";
    component_div, component_div_mut, component_div_assign, cdpy, ClosedDiv.div.div_assign,
    "Componentwise matrix division.", "Mutable, componentwise matrix division.";
    // FIXME: add other operators like bitshift, etc. ?
);
