use crate::csr::CsrMatrix;

use std::ops::Add;
use crate::ops::serial::{spadd_csr, spadd_build_pattern};
use nalgebra::{ClosedAdd, ClosedMul, Scalar};
use num_traits::{Zero, One};
use std::sync::Arc;
use crate::ops::Transpose;
use crate::pattern::SparsityPattern;

impl<'a, T> Add<&'a CsrMatrix<T>> for &'a CsrMatrix<T>
where
    // TODO: Consider introducing wrapper trait for these things? It's technically a "Ring",
    // I guess...
    T: Scalar + ClosedAdd + ClosedMul + Zero + One
{
    type Output = CsrMatrix<T>;

    fn add(self, rhs: &'a CsrMatrix<T>) -> Self::Output {
        let mut pattern = SparsityPattern::new(self.nrows(), self.ncols());
        spadd_build_pattern(&mut pattern, self.pattern(), rhs.pattern());
        let values = vec![T::zero(); pattern.nnz()];
        // We are giving data that is valid by definition, so it is safe to unwrap below
        let mut result = CsrMatrix::try_from_pattern_and_values(Arc::new(pattern), values)
            .unwrap();
        spadd_csr(&mut result, T::zero(), T::one(), Transpose(false), &self).unwrap();
        spadd_csr(&mut result, T::one(), T::one(), Transpose(false), &rhs).unwrap();
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
            spadd_csr(&mut self, T::one(), T::one(), Transpose(false), &rhs).unwrap();
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