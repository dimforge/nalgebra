//! This module provides the matrix exponential (pow) function to square matrices.

use std::ops::DivAssign;

use crate::{allocator::Allocator, DefaultAllocator, DimMin, MatrixN};
use num::PrimInt;
use simba::scalar::ComplexField;

impl<N: ComplexField, D> MatrixN<N, D>
where
    D: DimMin<D, Output = D>,
    DefaultAllocator: Allocator<N, D, D>,
{
    /// Attempts to raise this matrix to an integer power in-place. Returns
    /// `false` and leaves `self` untouched if the power is negative and the
    /// matrix is non-invertible.
    pub fn pow_mut<T: PrimInt + DivAssign>(&mut self, mut e: T) -> bool {
        let zero = T::zero();

        if e == zero {
            self.fill_with_identity();
            return true;
        }

        if e < zero {
            if !self.try_inverse_mut() {
                return false;
            }
        }

        let one = T::one();
        let two = T::from(2u8).unwrap();
        let mut multiplier = self.clone();

        while e != zero {
            if e % two == one {
                *self *= &multiplier;
            }

            e /= two;
            multiplier *= multiplier.clone();
        }

        true
    }

    /// Raise this matrix to an integer power. Returns `None` only if the power
    /// is negative and the matrix is non-invertible.
    pub fn pow<T: PrimInt + DivAssign>(&self, e: T) -> Option<Self> {
        let mut clone = self.clone();

        if clone.pow_mut(e) {
            Some(clone)
        } else {
            None
        }
    }
}
