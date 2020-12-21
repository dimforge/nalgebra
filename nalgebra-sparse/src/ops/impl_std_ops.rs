use crate::csr::CsrMatrix;

use std::ops::{Add, Mul};
use crate::ops::serial::{spadd_csr_prealloc, spadd_pattern, spmm_pattern, spmm_csr_prealloc};
use nalgebra::{ClosedAdd, ClosedMul, Scalar};
use num_traits::{Zero, One};
use std::sync::Arc;
use crate::ops::{Op};

impl<'a, T> Add<&'a CsrMatrix<T>> for &'a CsrMatrix<T>
where
    // TODO: Consider introducing wrapper trait for these things? It's technically a "Ring",
    // I guess...
    T: Scalar + ClosedAdd + ClosedMul + Zero + One
{
    type Output = CsrMatrix<T>;

    fn add(self, rhs: &'a CsrMatrix<T>) -> Self::Output {
        let pattern = spadd_pattern(self.pattern(), rhs.pattern());
        let values = vec![T::zero(); pattern.nnz()];
        // We are giving data that is valid by definition, so it is safe to unwrap below
        let mut result = CsrMatrix::try_from_pattern_and_values(Arc::new(pattern), values)
            .unwrap();
        spadd_csr_prealloc(T::zero(), &mut result, T::one(), Op::NoOp(&self)).unwrap();
        spadd_csr_prealloc(T::one(), &mut result, T::one(), Op::NoOp(&rhs)).unwrap();
        result
    }
}

impl<'a, T> Add<&'a CsrMatrix<T>> for CsrMatrix<T>
where
    T: Scalar + ClosedAdd + ClosedMul + Zero + One
{
    type Output = CsrMatrix<T>;

    fn add(mut self, rhs: &'a CsrMatrix<T>) -> Self::Output {
        if Arc::ptr_eq(self.pattern(), rhs.pattern()) {
            spadd_csr_prealloc(T::one(), &mut self, T::one(), Op::NoOp(rhs)).unwrap();
            self
        } else {
            &self + rhs
        }
    }
}

impl<'a, T> Add<CsrMatrix<T>> for &'a CsrMatrix<T>
    where
        T: Scalar + ClosedAdd + ClosedMul + Zero + One
{
    type Output = CsrMatrix<T>;

    fn add(self, rhs: CsrMatrix<T>) -> Self::Output {
        rhs + self
    }
}

impl<T> Add<CsrMatrix<T>> for CsrMatrix<T>
where
    T: Scalar + ClosedAdd + ClosedMul + Zero + One
{
    type Output = Self;

    fn add(self, rhs: CsrMatrix<T>) -> Self::Output {
        self + &rhs
    }
}

/// Helper macro for implementing matrix multiplication for different matrix types
/// See below for usage.
macro_rules! impl_matrix_mul {
    (<$($life:lifetime),*>($a_name:ident : $a:ty, $b_name:ident : $b:ty) -> $ret:ty $body:block)
        =>
    {
        impl<$($life,)* T> Mul<$b> for $a
        where
            T: Scalar + ClosedAdd + ClosedMul + Zero + One
        {
            type Output = $ret;
            fn mul(self, rhs: $b) -> Self::Output {
                let $a_name = self;
                let $b_name = rhs;
                $body
            }
        }
    }
}

impl_matrix_mul!(<'a>(a: &'a CsrMatrix<T>, b: &'a CsrMatrix<T>) -> CsrMatrix<T> {
    let pattern = spmm_pattern(a.pattern(), b.pattern());
    let values = vec![T::zero(); pattern.nnz()];
    let mut result = CsrMatrix::try_from_pattern_and_values(Arc::new(pattern), values)
        .unwrap();
    spmm_csr_prealloc(T::zero(),
             &mut result,
             T::one(),
             Op::NoOp(a),
             Op::NoOp(b))
        .expect("Internal error: spmm failed (please debug).");
    result
});
impl_matrix_mul!(<'a>(a: &'a CsrMatrix<T>, b: CsrMatrix<T>) -> CsrMatrix<T> { a * &b});
impl_matrix_mul!(<'a>(a: CsrMatrix<T>, b: &'a CsrMatrix<T>) -> CsrMatrix<T> { &a * b});
impl_matrix_mul!(<>(a: CsrMatrix<T>, b: CsrMatrix<T>) -> CsrMatrix<T> { &a * &b});