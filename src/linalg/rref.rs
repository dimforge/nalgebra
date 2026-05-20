//! Functions for computing the rref of a matrix.

use core::ops::{Mul, Sub, SubAssign};
use simba::scalar::{ClosedMul, RealField};
use std::ops::{DivAssign, MulAssign};

use crate::allocator::Allocator;
use crate::base::dimension::Dim;
use crate::base::{Const, DefaultAllocator, OMatrix, OVector, Scalar};

// Implementation of:
// https://rosettacode.org/wiki/Reduced_row_echelon_form
/// Compute the reduced row echelon form of a matrix.
pub fn rref<T: RealField, D: Dim>(matrix: &OMatrix<T, D, D>) -> OMatrix<T, D, D>
where
    DefaultAllocator: Allocator<T, D, D> + Allocator<T, D>,
    DefaultAllocator: Allocator<T, crate::base::dimension::Const<1>, D> + Allocator<T, D>,
    T: Scalar + ClosedMul,
{
    let mut matrix = matrix.clone();
    let mut lead = 0;
    let row_count = matrix.nrows();
    let column_count = matrix.ncols();

    for r in 0..row_count {
        if column_count <= lead {
            break;
        }

        let mut i = r;

        // Should this have an epsilon comparison instead?
        while matrix[(i, lead)] == crate::convert(0.0) {
            i += 1;
            if row_count == i {
                i = r;
                lead += 1;
                if column_count == lead {
                    break;
                }
            }
        }

        matrix.swap_rows(i, r);

        if matrix[(r, lead)] > T::default_epsilon() {
            let t = matrix[(r, lead)].clone();
            matrix.row_mut(r).div_assign(t);
        }

        for i in 0..row_count {
            if i != r {
                let lv = -matrix[(i, lead)].clone();
                let r_row = matrix.row(r).clone() * lv;

                matrix.row_mut(i).sub_assign(r_row);
            }
        }

        lead += 1;
    }

    matrix
}
