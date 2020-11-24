use crate::coo::CooMatrix;
use crate::convert::serial::*;
use nalgebra::{Matrix, Scalar, Dim, ClosedAdd, DMatrix};
use nalgebra::storage::{Storage};
use num_traits::Zero;
use crate::csr::CsrMatrix;
use crate::csc::CscMatrix;

impl<'a, T, R, C, S> From<&'a Matrix<T, R, C, S>> for CooMatrix<T>
where
    T: Scalar + Zero,
    R: Dim,
    C: Dim,
    S: Storage<T, R, C>
{
    fn from(matrix: &'a Matrix<T, R, C, S>) -> Self {
        convert_dense_coo(matrix)
    }
}

impl<'a, T> From<&'a CooMatrix<T>> for DMatrix<T>
where
    T: Scalar + Zero + ClosedAdd,
{
    fn from(coo: &'a CooMatrix<T>) -> Self {
        convert_coo_dense(coo)
    }
}

impl<'a, T> From<&'a CooMatrix<T>> for CsrMatrix<T>
where
    T: Scalar + Zero + ClosedAdd
{
    fn from(matrix: &'a CooMatrix<T>) -> Self {
        convert_coo_csr(matrix)
    }
}

impl<'a, T> From<&'a CsrMatrix<T>> for CooMatrix<T>
where
    T: Scalar + Zero + ClosedAdd
{
    fn from(matrix: &'a CsrMatrix<T>) -> Self {
        convert_csr_coo(matrix)
    }
}

impl<'a, T, R, C, S> From<&'a Matrix<T, R, C, S>> for CsrMatrix<T>
where
    T: Scalar + Zero,
    R: Dim,
    C: Dim,
    S: Storage<T, R, C>
{
    fn from(matrix: &'a Matrix<T, R, C, S>) -> Self {
        convert_dense_csr(matrix)
    }
}

impl<'a, T> From<&'a CsrMatrix<T>> for DMatrix<T>
where
    T: Scalar + Zero + ClosedAdd
{
    fn from(matrix: &'a CsrMatrix<T>) -> Self {
        convert_csr_dense(matrix)
    }
}

impl<'a, T> From<&'a CooMatrix<T>> for CscMatrix<T>
where
    T: Scalar + Zero + ClosedAdd
{
    fn from(matrix: &'a CooMatrix<T>) -> Self {
        convert_coo_csc(matrix)
    }
}

impl<'a, T> From<&'a CscMatrix<T>> for CooMatrix<T>
where
    T: Scalar + Zero
{
    fn from(matrix: &'a CscMatrix<T>) -> Self {
        convert_csc_coo(matrix)
    }
}

impl<'a, T, R, C, S> From<&'a Matrix<T, R, C, S>> for CscMatrix<T>
    where
        T: Scalar + Zero,
        R: Dim,
        C: Dim,
        S: Storage<T, R, C>
{
    fn from(matrix: &'a Matrix<T, R, C, S>) -> Self {
        convert_dense_csc(matrix)
    }
}

impl<'a, T> From<&'a CscMatrix<T>> for DMatrix<T>
    where
        T: Scalar + Zero + ClosedAdd
{
    fn from(matrix: &'a CscMatrix<T>) -> Self {
        convert_csc_dense(matrix)
    }
}

impl<'a, T> From<&'a CscMatrix<T>> for CsrMatrix<T>
    where
        T: Scalar + Zero
{
    fn from(matrix: &'a CscMatrix<T>) -> Self {
        convert_csc_csr(matrix)
    }
}

impl<'a, T> From<&'a CsrMatrix<T>> for CscMatrix<T>
where
    T: Scalar + Zero
{
    fn from(matrix: &'a CsrMatrix<T>) -> Self {
        convert_csr_csc(matrix)
    }
}