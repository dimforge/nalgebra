//! This module provides the matrix exponential (pow) function to square matrices.

use std::ops::DivAssign;

use crate::{allocator::Allocator, DefaultAllocator, DimMin, MatrixN};
use num::PrimInt;
use simba::scalar::ComplexField;

impl<N: ComplexField, D> MatrixN<N, D>
where
    D: DimMin<D, Output = D>,
    DefaultAllocator: Allocator<N, D, D>,
    DefaultAllocator: Allocator<N, D>,
{
    /// Computes the square of this matrix and writes it into a given buffer.
    fn square_buf(&mut self, buf: &mut Self) {
        // We unroll the first iteration to avoid new_uninitialized.
        let mut aux_col = self.column(0).clone_owned();
        aux_col = &*self * aux_col;
        buf.column_mut(0).copy_from(&aux_col);

        // We multiply the matrix by its i-th column,
        for i in 1..self.ncols() {
            aux_col.copy_from(&self.column(i));
            aux_col = &*self * aux_col;
            self.column_mut(i).copy_from(&aux_col);
        }
    }

    /// Attempts to raise this matrix to an integral power `e` in-place. If this
    /// matrix is non-invertible and `e` is negative, it leaves this matrix
    /// untouched and returns `false`. Otherwise, it returns `true` and
    /// overwrites this matrix with the result.
    pub fn pow_mut<T: PrimInt + DivAssign>(&mut self, mut e: T) -> bool {
        let zero = T::zero();

        // A matrix raised to the zeroth power is just the identity.
        if e == zero {
            self.fill_with_identity();
            return true;
        }

        // If e is negative, we compute the inverse matrix, then raise it to the
        // power of -e.
        if e < zero {
            if !self.try_inverse_mut() {
                return false;
            }
        }

        let one = T::one();
        let two = T::from(2u8).unwrap();

        // We use the buffer to hold the result of multiplier ^ 2, thus avoiding
        // extra allocations.
        let mut multiplier = self.clone();
        let mut buf = self.clone();

        // Exponentiation by squares.
        loop {
            if e % two == one {
                *self *= &multiplier;
            }

            e /= two;
            multiplier.square_buf(&mut buf);
            multiplier.copy_from(&buf);

            if e == zero {
                return true;
            }
        }
    }

    /// Attempts to raise this matrix to an integral power `e`. If this matrix
    /// is non-invertible and `e` is negative, it returns `None`. Otherwise, it
    /// returns the result as a new matrix. Uses exponentiation by squares.
    pub fn pow<T: PrimInt + DivAssign>(&self, e: T) -> Option<Self> {
        let mut clone = self.clone();

        if clone.pow_mut(e) {
            Some(clone)
        } else {
            None
        }
    }
}
