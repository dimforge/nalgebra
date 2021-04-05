//! This module provides the matrix exponential (pow) function to square matrices.

use std::ops::DivAssign;

use crate::{allocator::Allocator, DefaultAllocator, DimMin, DimMinimum, MatrixN};
use num::PrimInt;
use simba::scalar::ComplexField;

impl<N: ComplexField, D> MatrixN<N, D>
where
    D: DimMin<D, Output = D>,
    DefaultAllocator: Allocator<N, D, D>
        + Allocator<(usize, usize), DimMinimum<D, D>>
        + Allocator<N, D>
        + Allocator<N::RealField, D>
        + Allocator<N::RealField, D, D>,
{
    /// Raises a matrix to an integer power using exponentiation by squares.
    /// Returns `None` only when the matrix is non-invertible and raised to a
    /// negative power.
    pub fn pow<T: PrimInt + DivAssign>(&self, mut e: T) -> Option<Self> {
        let zero = T::zero();

        if e == zero {
            let mut i = self.clone();
            i.fill_with_identity();
            return Some(i);
        }

        let mut acc;
        if e < zero {
            acc = self.clone().try_inverse()?;
        } else {
            acc = self.clone();
        }

        let one = T::one();
        let two = T::from(2u8).unwrap();
        let mut multiplier = acc.clone();

        while e != zero {
            if e % two == one {
                acc *= &multiplier;
            }

            e /= two;
            multiplier *= multiplier.clone();
        }

        Some(acc)
    }
}
