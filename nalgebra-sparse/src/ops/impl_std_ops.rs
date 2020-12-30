use crate::csr::CsrMatrix;
use crate::csc::CscMatrix;

use std::ops::{Add, Mul};
use crate::ops::serial::{spadd_csr_prealloc, spadd_csc_prealloc, spadd_pattern,
                         spmm_pattern, spmm_csr_prealloc, spmm_csc_prealloc};
use nalgebra::{ClosedAdd, ClosedMul, Scalar};
use num_traits::{Zero, One};
use std::sync::Arc;
use crate::ops::{Op};

/// Helper macro for implementing binary operators for different matrix types
/// See below for usage.
macro_rules! impl_bin_op {
    ($trait:ident, $method:ident,
        <$($life:lifetime),*>($a:ident : $a_type:ty, $b:ident : $b_type:ty) -> $ret:ty $body:block)
        =>
    {
        impl<$($life,)* T> $trait<$b_type> for $a_type
        where
            T: Scalar + ClosedAdd + ClosedMul + Zero + One
        {
            type Output = $ret;
            fn $method(self, rhs: $b_type) -> Self::Output {
                let $a = self;
                let $b = rhs;
                $body
            }
        }
    }
}

macro_rules! impl_add {
    ($($args:tt)*) => {
        impl_bin_op!(Add, add, $($args)*);
    }
}

/// Implements a + b for all combinations of reference and owned matrices, for
/// CsrMatrix or CscMatrix.
macro_rules! impl_spadd {
    ($matrix_type:ident, $spadd_fn:ident) => {
        impl_add!(<'a>(a: &'a $matrix_type<T>, b: &'a $matrix_type<T>) -> $matrix_type<T> {
            // If both matrices have the same pattern, then we can immediately re-use it
            let pattern = if Arc::ptr_eq(a.pattern(), b.pattern()) {
                Arc::clone(a.pattern())
            } else {
                Arc::new(spadd_pattern(a.pattern(), b.pattern()))
            };
            let values = vec![T::zero(); pattern.nnz()];
            // We are giving data that is valid by definition, so it is safe to unwrap below
            let mut result = $matrix_type::try_from_pattern_and_values(pattern, values)
                .unwrap();
            $spadd_fn(T::zero(), &mut result, T::one(), Op::NoOp(&a)).unwrap();
            $spadd_fn(T::one(), &mut result, T::one(), Op::NoOp(&b)).unwrap();
            result
        });

        impl_add!(<'a>(a: $matrix_type<T>, b: &'a $matrix_type<T>) -> $matrix_type<T> {
            let mut a = a;
            if Arc::ptr_eq(a.pattern(), b.pattern()) {
                $spadd_fn(T::one(), &mut a, T::one(), Op::NoOp(b)).unwrap();
                a
            } else {
                &a + b
            }
        });

        impl_add!(<'a>(a: &'a $matrix_type<T>, b: $matrix_type<T>) -> $matrix_type<T> {
            b + a
        });
        impl_add!(<>(a: $matrix_type<T>, b: $matrix_type<T>) -> $matrix_type<T> {
            a + &b
        });
    }
}

impl_spadd!(CsrMatrix, spadd_csr_prealloc);
impl_spadd!(CscMatrix, spadd_csc_prealloc);

macro_rules! impl_mul {
    ($($args:tt)*) => {
        impl_bin_op!(Mul, mul, $($args)*);
    }
}

/// Implements a + b for all combinations of reference and owned matrices, for
/// CsrMatrix or CscMatrix.
macro_rules! impl_spmm {
    ($matrix_type:ident, $pattern_fn:expr, $spmm_fn:expr) => {
        impl_mul!(<'a>(a: &'a $matrix_type<T>, b: &'a $matrix_type<T>) -> $matrix_type<T> {
            let pattern = $pattern_fn(a.pattern(), b.pattern());
            let values = vec![T::zero(); pattern.nnz()];
            let mut result = $matrix_type::try_from_pattern_and_values(Arc::new(pattern), values)
                .unwrap();
            $spmm_fn(T::zero(),
                     &mut result,
                     T::one(),
                     Op::NoOp(a),
                     Op::NoOp(b))
                .expect("Internal error: spmm failed (please debug).");
            result
        });
        impl_mul!(<'a>(a: &'a $matrix_type<T>, b: $matrix_type<T>) -> $matrix_type<T> { a * &b});
        impl_mul!(<'a>(a: $matrix_type<T>, b: &'a $matrix_type<T>) -> $matrix_type<T> { &a * b});
        impl_mul!(<>(a: $matrix_type<T>, b: $matrix_type<T>) -> $matrix_type<T> { &a * &b});
    }
}

impl_spmm!(CsrMatrix, spmm_pattern, spmm_csr_prealloc);
// Need to switch order of operations for CSC pattern
impl_spmm!(CscMatrix, |a, b| spmm_pattern(b, a), spmm_csc_prealloc);