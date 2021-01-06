use crate::csr::CsrMatrix;
use crate::csc::CscMatrix;

use std::ops::{Add, Div, DivAssign, Mul, MulAssign, Sub, Neg};
use crate::ops::serial::{spadd_csr_prealloc, spadd_csc_prealloc, spadd_pattern,
                         spmm_pattern, spmm_csr_prealloc, spmm_csc_prealloc};
use nalgebra::{ClosedAdd, ClosedMul, ClosedSub, ClosedDiv, Scalar};
use num_traits::{Zero, One};
use std::sync::Arc;
use crate::ops::{Op};

/// Helper macro for implementing binary operators for different matrix types
/// See below for usage.
macro_rules! impl_bin_op {
    ($trait:ident, $method:ident,
        <$($life:lifetime),* $(,)? $($scalar_type:ident $(: $bounds:path)?)?>($a:ident : $a_type:ty, $b:ident : $b_type:ty) -> $ret:ty $body:block)
        =>
    {
        impl<$($life,)* $($scalar_type)?> $trait<$b_type> for $a_type
        where
            // Note: The Neg bound is currently required because we delegate e.g.
            // Sub to SpAdd with negative coefficients. This is not well-defined for
            // unsigned data types.
            $($scalar_type: $($bounds + )? Scalar + ClosedAdd + ClosedSub + ClosedMul + Zero + One + Neg<Output=T>)?
        {
            type Output = $ret;
            fn $method(self, $b: $b_type) -> Self::Output {
                let $a = self;
                $body
            }
        }
    };
}

/// Implements a +/- b for all combinations of reference and owned matrices, for
/// CsrMatrix or CscMatrix.
macro_rules! impl_sp_plus_minus {
    // We first match on some special-case syntax, and forward to the actual implementation
    ($matrix_type:ident, $spadd_fn:ident, +) => {
        impl_sp_plus_minus!(Add, add, $matrix_type, $spadd_fn, +, T::one());
    };
    ($matrix_type:ident, $spadd_fn:ident, -) => {
        impl_sp_plus_minus!(Sub, sub, $matrix_type, $spadd_fn, -, -T::one());
    };
    ($trait:ident, $method:ident, $matrix_type:ident, $spadd_fn:ident, $sign:tt, $factor:expr) => {
        impl_bin_op!($trait, $method,
            <'a, T>(a: &'a $matrix_type<T>, b: &'a $matrix_type<T>) -> $matrix_type<T> {
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
            $spadd_fn(T::one(), &mut result, $factor * T::one(), Op::NoOp(&b)).unwrap();
            result
        });

        impl_bin_op!($trait, $method,
            <'a, T>(a: $matrix_type<T>, b: &'a $matrix_type<T>) -> $matrix_type<T> {
            let mut a = a;
            if Arc::ptr_eq(a.pattern(), b.pattern()) {
                $spadd_fn(T::one(), &mut a, $factor * T::one(), Op::NoOp(b)).unwrap();
                a
            } else {
                &a $sign b
            }
        });

        impl_bin_op!($trait, $method,
            <'a, T>(a: &'a $matrix_type<T>, b: $matrix_type<T>) -> $matrix_type<T> {
            let mut b = b;
            if Arc::ptr_eq(a.pattern(), b.pattern()) {
                $spadd_fn($factor * T::one(), &mut b, T::one(), Op::NoOp(a)).unwrap();
                b
            } else {
                a $sign &b
            }
        });
        impl_bin_op!($trait, $method, <T>(a: $matrix_type<T>, b: $matrix_type<T>) -> $matrix_type<T> {
            a $sign &b
        });
    }
}

impl_sp_plus_minus!(CsrMatrix, spadd_csr_prealloc, +);
impl_sp_plus_minus!(CsrMatrix, spadd_csr_prealloc, -);
impl_sp_plus_minus!(CscMatrix, spadd_csc_prealloc, +);
impl_sp_plus_minus!(CscMatrix, spadd_csc_prealloc, -);

macro_rules! impl_mul {
    ($($args:tt)*) => {
        impl_bin_op!(Mul, mul, $($args)*);
    }
}

/// Implements a + b for all combinations of reference and owned matrices, for
/// CsrMatrix or CscMatrix.
macro_rules! impl_spmm {
    ($matrix_type:ident, $pattern_fn:expr, $spmm_fn:expr) => {
        impl_mul!(<'a, T>(a: &'a $matrix_type<T>, b: &'a $matrix_type<T>) -> $matrix_type<T> {
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
        impl_mul!(<'a, T>(a: &'a $matrix_type<T>, b: $matrix_type<T>) -> $matrix_type<T> { a * &b});
        impl_mul!(<'a, T>(a: $matrix_type<T>, b: &'a $matrix_type<T>) -> $matrix_type<T> { &a * b});
        impl_mul!(<T>(a: $matrix_type<T>, b: $matrix_type<T>) -> $matrix_type<T> { &a * &b});
    }
}

impl_spmm!(CsrMatrix, spmm_pattern, spmm_csr_prealloc);
// Need to switch order of operations for CSC pattern
impl_spmm!(CscMatrix, |a, b| spmm_pattern(b, a), spmm_csc_prealloc);

/// Implements Scalar * Matrix operations for *concrete* scalar types. The reason this is necessary
/// is that we are not able to implement Mul<Matrix<T>> for all T generically due to orphan rules.
macro_rules! impl_concrete_scalar_matrix_mul {
    ($matrix_type:ident, $($scalar_type:ty),*) => {
        // For each concrete scalar type, forward the implementation of scalar * matrix
        // to matrix * scalar, which we have already implemented through generics
        $(
            impl_mul!(<>(a: $scalar_type, b: $matrix_type<$scalar_type>)
                -> $matrix_type<$scalar_type> { b * a });
            impl_mul!(<'a>(a: $scalar_type, b: &'a $matrix_type<$scalar_type>)
                -> $matrix_type<$scalar_type> { b * a });
            impl_mul!(<'a>(a: &'a $scalar_type, b: $matrix_type<$scalar_type>)
                -> $matrix_type<$scalar_type> { b * (*a) });
            impl_mul!(<'a>(a: &'a $scalar_type, b: &'a $matrix_type<$scalar_type>)
                -> $matrix_type<$scalar_type> { b * *a });
        )*
    }
}

/// Implements multiplication between matrix and scalar for various matrix types
macro_rules! impl_scalar_mul {
    ($matrix_type: ident) => {
        impl_mul!(<'a, T>(a: &'a $matrix_type<T>, b: &'a T) -> $matrix_type<T> {
            let values: Vec<_> = a.values()
                .iter()
                .map(|v_i| v_i.inlined_clone() * b.inlined_clone())
                .collect();
            $matrix_type::try_from_pattern_and_values(Arc::clone(a.pattern()), values).unwrap()
        });
        impl_mul!(<'a, T>(a: &'a $matrix_type<T>, b: T) -> $matrix_type<T> {
            a * &b
        });
        impl_mul!(<'a, T>(a: $matrix_type<T>, b: &'a T) -> $matrix_type<T> {
            let mut a = a;
            for value in a.values_mut() {
                *value = b.inlined_clone() * value.inlined_clone();
            }
            a
        });
        impl_mul!(<T>(a: $matrix_type<T>, b: T) -> $matrix_type<T> {
            a * &b
        });
        impl_concrete_scalar_matrix_mul!(
            $matrix_type,
            i8, i16, i32, i64, isize, f32, f64);

        impl<T> MulAssign<T> for $matrix_type<T>
        where
            T: Scalar + ClosedAdd + ClosedMul + Zero + One
        {
            fn mul_assign(&mut self, scalar: T) {
                for val in self.values_mut() {
                    *val *= scalar.inlined_clone();
                }
            }
        }

        impl<'a, T> MulAssign<&'a T> for $matrix_type<T>
        where
            T: Scalar + ClosedAdd + ClosedMul + Zero + One
        {
            fn mul_assign(&mut self, scalar: &'a T) {
                for val in self.values_mut() {
                    *val *= scalar.inlined_clone();
                }
            }
        }
    }
}

impl_scalar_mul!(CsrMatrix);
impl_scalar_mul!(CscMatrix);

macro_rules! impl_neg {
    ($matrix_type:ident) => {
        impl<T> Neg for $matrix_type<T>
        where
            T: Scalar + Neg<Output=T>
        {
            type Output = $matrix_type<T>;

            fn neg(mut self) -> Self::Output {
                for v_i in self.values_mut() {
                    *v_i = -v_i.inlined_clone();
                }
                self
            }
        }

        impl<'a, T> Neg for &'a $matrix_type<T>
        where
            T: Scalar + Neg<Output=T>
        {
            type Output = $matrix_type<T>;

            fn neg(self) -> Self::Output {
                // TODO: This is inefficient. Ideally we'd have a method that would let us
                // obtain both the sparsity pattern and values from the matrix,
                // and then modify the values before creating a new matrix from the pattern
                // and negated values.
                - self.clone()
            }
        }
    }
}

impl_neg!(CsrMatrix);
impl_neg!(CscMatrix);

macro_rules! impl_div {
    ($matrix_type:ident) => {
        impl_bin_op!(Div, div, <T: ClosedDiv>(matrix: $matrix_type<T>, scalar: T) -> $matrix_type<T> {
            let mut matrix = matrix;
            matrix /= scalar;
            matrix
        });
        impl_bin_op!(Div, div, <'a, T: ClosedDiv>(matrix: $matrix_type<T>, scalar: &T) -> $matrix_type<T> {
            matrix / scalar.inlined_clone()
        });
        impl_bin_op!(Div, div, <'a, T: ClosedDiv>(matrix: &'a $matrix_type<T>, scalar: T) -> $matrix_type<T> {
            let new_values = matrix.values()
                .iter()
                .map(|v_i| v_i.inlined_clone() / scalar.inlined_clone())
                .collect();
            $matrix_type::try_from_pattern_and_values(Arc::clone(matrix.pattern()), new_values)
                .unwrap()
        });
        impl_bin_op!(Div, div, <'a, T: ClosedDiv>(matrix: &'a $matrix_type<T>, scalar: &'a T) -> $matrix_type<T> {
            matrix / scalar.inlined_clone()
        });

        impl<T> DivAssign<T> for $matrix_type<T>
            where T : Scalar + ClosedAdd + ClosedMul + ClosedDiv + Zero + One
        {
            fn div_assign(&mut self, scalar: T) {
                self.values_mut().iter_mut().for_each(|v_i| *v_i /= scalar.inlined_clone());
            }
        }

        impl<'a, T> DivAssign<&'a T> for $matrix_type<T>
            where T : Scalar + ClosedAdd + ClosedMul + ClosedDiv + Zero + One
        {
            fn div_assign(&mut self, scalar: &'a T) {
                *self /= scalar.inlined_clone();
            }
        }
    }
}

impl_div!(CsrMatrix);
impl_div!(CscMatrix);