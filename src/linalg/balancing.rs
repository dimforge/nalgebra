//! Functions for balancing a matrix.

use simba::scalar::RealField;
use std::ops::{DivAssign, MulAssign};

use crate::allocator::Allocator;
use crate::base::dimension::Dim;
use crate::base::storage::Storage;
use crate::base::{Const, DefaultAllocator, OMatrix, OVector};

/// Applies in-place a modified Parlett and Reinsch matrix balancing with 2-norm to the matrix and returns
/// the corresponding diagonal transformation.
///
/// See https://arxiv.org/pdf/1401.5766.pdf
pub fn balance_parlett_reinsch<T: RealField, D: Dim>(matrix: &mut OMatrix<T, D, D>) -> OVector<T, D>
where
    DefaultAllocator: Allocator<T, D, D> + Allocator<T, D>,
{
    assert!(matrix.is_square(), "Unable to balance a non-square matrix.");

    let dim = matrix.data.shape().0;
    let radix: T = crate::convert(2.0f64);
    let mut d = OVector::from_element_generic(dim, Const::<1>, T::one());

    let mut converged = false;

    while !converged {
        converged = true;

        for i in 0..dim.value() {
            let mut n_col = matrix.column(i).norm_squared();
            let mut n_row = matrix.row(i).norm_squared();
            let mut f = T::one();

            let s = n_col + n_row;
            n_col = n_col.sqrt();
            n_row = n_row.sqrt();

            if n_col.is_zero() || n_row.is_zero() {
                continue;
            }

            while n_col < n_row / radix {
                n_col *= radix;
                n_row /= radix;
                f *= radix;
            }

            while n_col >= n_row * radix {
                n_col /= radix;
                n_row *= radix;
                f /= radix;
            }

            let eps: T = crate::convert(0.95);
            #[allow(clippy::suspicious_operation_groupings)]
            if n_col * n_col + n_row * n_row < eps * s {
                converged = false;
                d[i] *= f;
                matrix.column_mut(i).mul_assign(f);
                matrix.row_mut(i).div_assign(f);
            }
        }
    }

    d
}

/// Computes in-place `D * m * D.inverse()`, where `D` is the matrix with diagonal `d`.
pub fn unbalance<T: RealField, D: Dim>(m: &mut OMatrix<T, D, D>, d: &OVector<T, D>)
where
    DefaultAllocator: Allocator<T, D, D> + Allocator<T, D>,
{
    assert!(m.is_square(), "Unable to unbalance a non-square matrix.");
    assert_eq!(m.nrows(), d.len(), "Unbalancing: mismatched dimensions.");

    for j in 0..d.len() {
        let mut col = m.column_mut(j);
        let denom = T::one() / d[j];

        for i in 0..d.len() {
            col[i] *= d[i] * denom;
        }
    }
}
